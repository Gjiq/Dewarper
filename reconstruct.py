"""
Art-plate layout reconstruction pipeline.
1. Classify each page: DARK_BG | FULL_BLEED | TEXT_LEFT | TEXT | SKIP
2. Detect art panel bounding boxes per page type
3. Clean-crop each panel (straight edges, 75th-percentile inward cut)
4. Paste each panel at its original page position on a fresh canvas

TEXT pages (full index / contents / colophon / essay pages) bypass panel
detection entirely: they go through the text-page treatment in dewarp_text.py
(curl dewarp + flat-field whitening, original pixels relocated, no OCR).
"""
import cv2
import numpy as np
import os
from scipy.ndimage import uniform_filter1d, gaussian_filter
from scipy.signal import find_peaks

# Text-page treatment (curl dewarp + flat-field whiten). Ships alongside this
# script; imported by same-folder name so the package stays self-contained.
from dewarp_text import process_text_page, flatfield_whiten
import json
import deskew_crop
from deskew_crop import compose_fullbleed, build_paper_profile, art_pictures, compose_multi
from deskew_crop import _reliable_angle, _bg_texture
import page_dewarp
import magenta_crop
import dark_art
import text_blocks
import re

# MANUAL border override. If a page is submitted with art borders hand-drawn as a
# continuous MAGENTA outline (one specific colour), those loops are taken as the
# art rectangles, overriding all auto-detection. Lets a bad auto-crop be fixed by
# re-submitting the page with the correct borders drawn on. See magenta_crop.py.
MAGENTA_OVERRIDE = True

# CENTER_SINGLE: on single-picture FULL_BLEED pages, place the cleaned art at the
# PAGE CENTRE (equal margins H + V) instead of its scanned position -- corrects the
# binding-gutter shift and the empty band art-annual plates leave below the picture.
# Page furniture (header / caption / page number) stays put; only the picture moves.
# Single-picture pages only: multi-picture pages keep each plate in its own place.
CENTER_SINGLE = False

# CENTER_MULTI: on multi-picture pages, after every plate + caption is placed,
# rigidly centre the WHOLE content block (all non-paper: pictures, captions,
# headers, plate/page numbers, footer) within the page margins, so the layout sits
# with even margins instead of being pushed to one side by the scan/binding gutter.
# The layout moves as one unit (relative positions unchanged). Auto multi pages
# only; the manual magenta/clip path is left as authored.
CENTER_MULTI = False


# DESKEW_SHORT_TEXT (v25): on FULL_BLEED pages, also deskew SHORT text -- 1-2 line
# captions, headers and single credit lines -- not just the 3+ line paragraphs that
# straighten_paragraphs already levels. Reuses the SAME tested baseline-deskew +
# wipe/restamp path (text_blocks.straighten_paragraphs) with min_lines=1,max_lines=2
# and the same saturation guard, so it cannot stamp a plate. NEW this version and
# NOT yet verified on a real light-bg page in-session -- set False to restore exact
# v24 behaviour (short furniture kept at its scan skew in the whitened page).
DESKEW_SHORT_TEXT = True
SAFE_TEXT = True   # v30: do NOT wipe/ink-restamp page text (caused blobs/speckle/
#                   art-text superimposition). Keep original scanned text on the
#                   whitened page; only the ART is straightened. Toggle off to revert.

# v43: STRAIGHTEN THE PAGE TEXT to match the deskewed plates, via the guarded
# text_blocks.deskew_text_blocks (NOT the old SAFE_TEXT wipe/ink path). It only
# rotates a caption/credit/header block when its own tilt exceeds a threshold, uses
# the detected ART rectangles as no-go zones, and rejects any straightened footprint
# that would land on art or on another text block -- so deskewed text never steps
# over other text or art, and glyphs are composited as original pixels (no ink blobs).
DESKEW_PAGE_TEXT = True


# PAGE-LEVEL curl pre-pass: dewarp the whole original scan BEFORE detecting or
# cropping pictures, using multiple picture borders (residual-gated) to pin one
# low-order field. Removes the smooth binding-curl bow (CURVATURE only) so the
# per-picture deskew (which handles per-picture SLOPE) then works on a flat page.
# No-ops unless >=2 clean borders agree. Single-picture pages fall through to
# compose_fullbleed.
PAGE_DEWARP = True

# Per-panel deslope for TEXT_LEFT / DARK_BG pages. These go through clean_crop,
# which only makes a straight axis-aligned cut and never rotates -- so any panel
# tilt rode straight through. With this on, each panel's reliable angle is removed
# (same _reliable_angle the FULL_BLEED deskew uses) before the straight cut, so
# the clean curl/slope split holds for every page type.
DESLOPE_TEXT = True

# RESUME: skip any page whose output already exists in OUTPUT_DIR. Lets a large
# staged batch be processed across several runs WITHOUT clearing INPUT_DIR -- so
# the whole batch stays staged and the paper profile is rebuilt over all of it
# every run (stable threshold), while finished pages are not re-rendered. Set
# False to force a full re-render.
RESUME = True

# EDGE_SLIVER_FIX: after an art page is composed, remove off-white page that
# leaked into the art block along its straight deskew edges -- the thin triangular
# slivers/seams from a small residual deskew angle. Strict near-white + border-
# connected, so pale ART (misty skies, light backgrounds) is preserved. Applied
# to every *_reconstructed.jpg (FULL_BLEED / multi / TEXT_LEFT / DARK_BG); TEXT
# pages are untouched. See deskew_crop.remove_edge_slivers.
EDGE_SLIVER_FIX = False   # OFF: a properly dewarped plate fills its rectangle, so
                          # there is no sliver to paint over. A leftover sliver means
                          # the dewarp under-reached and must be fixed there, NOT
                          # hidden with paper texture. (remove_edge_slivers kept in
                          # deskew_crop.py for reference but no longer wired in.)

INPUT_DIR  = os.environ.get('DEWARP_INPUT',  '/home/claude/work/input')
OUTPUT_DIR = os.environ.get('DEWARP_OUTPUT', '/home/claude/work/output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Page classifier ───────────────────────────────────────────────────────────

def _count_text_lines(gray):
    """Flat-field, then count horizontal text rows across the full page width."""
    bgs  = cv2.GaussianBlur(gray, (0, 0), 120)
    flat = np.clip(gray.astype(np.float32) / (bgs.astype(np.float32) + 1e-3) * 255, 0, 255)
    dark = 255 - flat
    prof = gaussian_filter(dark.mean(axis=1).astype(np.float32), 3)
    thr  = prof.mean() + 0.4 * prof.std()
    pk, _ = find_peaks(prof, distance=24, prominence=8, height=thr)
    return len(pk)


def _picture_fraction(img_bgr):
    """Fraction of the page densely covered by PICTURE (a colour plate / photo), as
    opposed to paper or text. Non-paper ink is measured, then LOCAL coverage: a picture
    is densely covered (~0.7-1.0); a text paragraph is sparse (~0.15-0.35, strokes with
    paper between); blank/marbled leaves read as either near-0 or, for heavy mottling,
    high coverage (correctly NOT text+bg). Used by the >74% text+bg TEXT gate."""
    H, W = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    coarse = cv2.GaussianBlur(gray, (0, 0), 9)
    nonpaper = ((S > 45) | (gray < coarse.astype(np.float32) * 0.80)).astype(np.float32)
    k = max(31, (min(H, W) // 20) | 1)
    density = cv2.boxFilter(nonpaper, -1, (k, k))
    return float((density > 0.55).mean())


def text_structure(gray, dark_text=True):
    """COLOUR-BLIND text detector (v20). Returns (area_fraction, n_line_blobs).

    Text is a dense stack of LINE-SHAPED ink runs: many wide-and-short connected
    components at a regular vertical pitch. We binarise (adaptively), close
    horizontally so glyphs in a line merge into one run, then keep components that
    look like a text line (wide, short, high aspect, well filled). The page-area
    fraction those cover, and their count, separate a text page from an art plate
    by 20-40x with no overlap -- and unlike the old gate it does NOT look at
    colour or ink-percentage, so a text page that ALSO carries a colour artwork,
    or sits on warm/tinted paper, still reads as text.

      dark_text=True  : dark ink on light paper (THRESH_BINARY_INV)
      dark_text=False : light/white text on a dark ground (THRESH_BINARY) --
                        the polarity needed for white-on-black pages.

    Reference (Spectrum 20): text pages 53-72% / 150-180 lines; art plates
    1.6-2.7% / 9-15 lines.
    """
    H, W = gray.shape
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    mode = cv2.THRESH_BINARY_INV if dark_text else cv2.THRESH_BINARY
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, mode, 31, 15)
    k = max(15, W // 60)
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    n, _, st, _ = cv2.connectedComponentsWithStats(closed, 8)
    tpx = 0; lines = 0
    for i in range(1, n):
        x, y, w, h, a = st[i, :5]
        if 3 < h < H * 0.05 and w > W * 0.06 and w / max(h, 1) > 4 and a > w * 0.15:
            tpx += w * h; lines += 1
    return tpx / float(H * W), lines


def page_skew_dark(img_bgr, rng=4.0, step=0.2):
    """Robust whole-page rotation (deg) for an art-on-black page, from its CONTENT
    via a projection profile. On a black ground there is no art-vs-paper border to
    fit a line to (which is why the old per-box deslope read ~0 and the art was
    left skewed), so we score by how sharply the page's horizontal structure --
    white-text baselines (weighted heavily) plus general edges -- separates into
    rows under each trial rotation. The angle with the crispest row profile is the
    skew; rotating the whole page by it straightens the ART and every text line
    together. Returns the rotation to APPLY (negative of the measured tilt sense
    handled by the search)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    loc = cv2.GaussianBlur(gray, (0, 0), 25).astype(np.int16)
    txt = ((gray.astype(np.int16) - loc) > 18).astype(np.float32)   # white text
    edg = cv2.Canny(gray, 40, 120).astype(np.float32) / 255.0
    feat = txt * 3.0 + edg
    # downscale for speed
    s = 700.0 / max(H, W)
    if s < 1.0:
        feat = cv2.resize(feat, (int(W * s), int(H * s)))
    fH, fW = feat.shape
    best_a, best = 0.0, -1.0
    for a in np.arange(-rng, rng + 1e-6, step):
        M = cv2.getRotationMatrix2D((fW / 2, fH / 2), a, 1.0)
        r = cv2.warpAffine(feat, M, (fW, fH), flags=cv2.INTER_NEAREST)
        score = float(np.var(r.sum(axis=1)))
        if score > best:
            best_a, best = float(a), score
    return best_a


def dark_ground_stats(img_bgr, ring_frac=0.045):
    """GLOSS-TOLERANT art-on-black detector (v20). A black-ground page leaves a
    dark, fairly consistent MARGIN ring at the page edges; scanner GLOSS /
    reflection only ever brightens a MINORITY of that ring, so a robust median +
    dark-fraction over the ring ride straight over the gloss (where a single
    corner sample is fooled -- Spectrum 20 p036 corner=82, p305 corner=100, both
    actually black-ground). Returns (ring_median, dark_fraction, gloss_fraction).
      ring_median  : median luminance of the outer ring (low => dark ground)
      dark_fraction: share of ring darker than 70 (high => consistent dark margin)
      gloss_fraction: share of ring brighter than 120 (the specular minority)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    b = max(8, int(min(H, W) * ring_frac))
    ring = np.concatenate([gray[:b, :].ravel(), gray[-b:, :].ravel(),
                           gray[:, :b].ravel(), gray[:, -b:].ravel()])
    return (float(np.median(ring)),
            float((ring < 70).mean()),
            float((ring > 120).mean()))


def white_caption_components(gray, lo=0.78, hi=0.99, vthr=180):
    """Count small BRIGHT (white) components floating in the lower band -- the
    artist caption / credit line that sits in the dark ground on an art-on-black
    page. A secondary confirmation for dark-ground classification, and the set we
    must PRESERVE (the v19 dark compose erased it -> lost 'Charles Vess')."""
    H, W = gray.shape
    band = gray[int(H * lo):int(H * hi), :]
    if band.size == 0:
        return 0
    bw = (band > vthr).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(bw, 8)
    return sum(1 for i in range(1, n)
               if 4 < st[i, 2] < W * 0.5 and 2 < st[i, 3] < band.shape[0] * 0.5
               and st[i, 4] > 10)


def white_caption_mask(gray, ring_med, vthr=None):
    """Full-page mask of bright caption/header text floating on the dark ground:
    pixels much brighter than the ground that form small text-sized blobs. Used
    to (a) keep the caption out of the art box and (b) PRESERVE it on the canvas
    so the dark compose no longer erases the artist credit (v19 lost it)."""
    H, W = gray.shape
    thr = vthr if vthr is not None else max(150, ring_med + 90)
    bw = (gray > thr).astype(np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(bw, 8)
    keep = np.zeros((H, W), np.uint8)
    for i in range(1, n):
        x, y, w, h, a = st[i, :5]
        # text-sized: not a huge lit art region, not a single speck
        if a < 0.02 * W * H and h < H * 0.06 and w < W * 0.85 and a > 8:
            keep[lab == i] = 255
    return keep


def _merge_overlapping(boxes, iou_grow=0.10):
    """Union boxes that overlap or nearly touch, so one artwork split into pieces
    by the mask returns as a single rectangle."""
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        out = []
        while boxes:
            x, y, w, h = boxes.pop()
            x1, y1 = x + w, y + h
            merged = True
            while merged:
                merged = False
                rest = []
                for (a, b, c, d) in boxes:
                    a1, b1 = a + c, b + d
                    gx = (x1 - x) * iou_grow; gy = (y1 - y) * iou_grow
                    if a <= x1 + gx and a1 >= x - gx and b <= y1 + gy and b1 >= y - gy:
                        x, y = min(x, a), min(y, b)
                        x1, y1 = max(x1, a1), max(y1, b1)
                        merged = True; changed = True
                    else:
                        rest.append((a, b, c, d))
                boxes = rest
            out.append((x, y, x1 - x, y1 - y))
        boxes = out
    return boxes


def _offwhite_canvas(H, W, tone=244, grain=3):
    """Synthesised clean off-white page with faint grain (NOT a dead flat fill),
    for the off-white delivery of an art-on-black page -- the dark ground has no
    real paper to sample, so we make neutral off-white paper to lift the plate
    onto."""
    base = np.full((H, W, 3), tone, np.uint8)
    n = np.random.default_rng(0).normal(0, grain, (H, W, 1)).astype(np.float32)
    return np.clip(base.astype(np.float32) + n, 0, 255).astype(np.uint8)


def classify_page(img_bgr):
    """
    Returns one of: 'DARK_BG' | 'FULL_BLEED' | 'TEXT_LEFT' | 'TEXT' | 'SKIP'
    """
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat  = hsv[:,:,1]

    # ── Art on a BLACK ground (gloss-tolerant, v20) ───────────────────────────
    # Replaces the old single-point  corner<80  test, which a glossy reflection
    # in a corner defeats (p036 corner=82, p305 corner=100 were both black-ground
    # and slipped through to the art/skip paths). Decide on the EDGE-RING median +
    # dark-fraction instead: a black-ground page has a dark, mostly-consistent
    # margin all round, and gloss only brightens a minority of that ring, so the
    # robust stats stay dark. A white caption floating in the lower band is a
    # secondary confirmation.
    ring_med, dark_frac, gloss_frac = dark_ground_stats(img_bgr)
    if ring_med < 60 and dark_frac > 0.55:
        return 'DARK_BG', ring_med

    bg = float(np.median([gray[10,10], gray[10,W-10], gray[H-10,10], gray[H-10,W-10]]))
    if bg < 80:
        return 'DARK_BG', bg

    WHITE = bg - 20
    content_pct = (gray < WHITE).mean() * 100

    # ── Full text page (index / contents / colophon / essay), v20 ─────────────
    # STRUCTURE, not colour: a page that is largely covered by line-shaped text
    # runs is a text page, even if it carries a colour artwork inset or sits on
    # tinted paper. The old gate keyed on  art_pct<8 and mean_sat<30 , which every
    # essay page with an embedded plate (14-18: mean_sat 30-38, art_pct 8-16)
    # failed despite having 36-63 detected text lines -- so they fell to FULL_BLEED
    # or were dropped as SKIP on a knife-edge ink %. text_structure() separates
    # text (53-72% area) from art plates (<3%) by 20x with no colour dependence.
    text_frac, text_lines = text_structure(gray, dark_text=True)
    if text_frac > 0.30 and text_lines >= 20:
        return 'TEXT', bg

    # ── >74% TEXT+PAPER = a TEXT page (v40) ───────────────────────────────────
    # A page that is more than 74% text + background (i.e. LESS than 26% picture) and
    # carries real text is a text page -- essays, indices, contents, credit/divider
    # pages, editorial intros -- even when the structural gate above (which keys on
    # text AREA) narrowly misses them. picture_fraction measures how much of the page
    # is a dense colour plate / photo; 1 - that is the text+paper share. The text_lines
    # guard keeps blank / decorative / marbled leaves (few real lines, or high picture
    # coverage from mottling) on the SKIP path, and no art plate qualifies (plates run
    # 37-55% picture -> ~45-63% text+paper, well under the bar).
    if text_lines >= 20 and (1.0 - _picture_fraction(img_bgr)) > 0.74:
        return 'TEXT', bg

    if content_pct < 22:
        return 'SKIP', bg

    col_sat = uniform_filter1d(sat.mean(axis=0).astype(float), size=50)
    leftmost_sat  = col_sat[:int(W*0.03)].mean()
    rightmost_sat = col_sat[int(W*0.97):].mean()

    if leftmost_sat < 35:
        strip_end = 0
        for x in range(W // 2):
            if col_sat[x] > 50:
                strip_end = x
                break
        if strip_end > 50:
            # Count real text rows in the left zone (flat-fielded, so paper
            # vignette/noise can't masquerade as text). A genuine text column
            # has many rows; a short artist caption (5-7 lines) or blank margin
            # does not — those are single-plate pages and fall through to FULL_BLEED.
            zone = gray[:, :min(strip_end, 200)]
            if _count_text_lines(zone) > 30:
                return 'TEXT_LEFT', bg

    # Bilateral binder margins with no text column = centered art panel
    if leftmost_sat < 35 and rightmost_sat < 35:
        return 'FULL_BLEED', bg

    return 'FULL_BLEED', bg


# Pages that auto-classification gets wrong — override here
PAGE_OVERRIDES = {
    # Spectrum 24 p009/p106/p202: classify_page sent three real full-bleed PAINTINGS
    # to SKIP via the "mottled / decorative leaf" branch (colour cov 23-70%, meanL
    # 131-157, dark ground). p010 (colour 7%, ink 16%, meanL 167) also dropped: a
    # muted low-sat plate -- forced FULL_BLEED and confirmed by content-retention that
    # it reconstructs a real plate (not a spurious box). Verified by MEASUREMENT this
    # session (inline image viewer down, as in the Spectrum 9 p045/p131 session);
    # QC crops kept for a later eyeball. Same surgical fix as Spectrum 16/13/12 -- do
    # NOT loosen classify_page.
    'Spectrum 24_Page_009.jpg': 'FULL_BLEED',
    'Spectrum 24_Page_010.jpg': 'FULL_BLEED',
    'Spectrum 24_Page_106.jpg': 'FULL_BLEED',
    'Spectrum 24_Page_202.jpg': 'FULL_BLEED',
    # Spectrum 22 p002/p011/p083/p101/p130/p155/p179/p249: classify_page sent eight
    # real full-bleed PAINTINGS to SKIP via the "mottled / decorative leaf" branch
    # (heavy/darker coverage on a shaded ground; colour cov ~3-52%, meanL ~145-168).
    # Verified on-screen this session -- all eight are single full-bleed art plates;
    # p249 (3% colour) and p155 (11%) are MUTED low-saturation paintings, not text
    # pages. Forced FULL_BLEED reconstructions were eyeballed clean (no spurious box).
    # Same surgical approach as Spectrum 16/13/12/10/9 -- do NOT loosen classify_page.
    # NB filenames are the lowercase 'spectrum 22_Page_NNN.jpg' exactly as delivered.
    'spectrum 22_Page_002.jpg': 'FULL_BLEED',
    'spectrum 22_Page_011.jpg': 'FULL_BLEED',
    'spectrum 22_Page_083.jpg': 'FULL_BLEED',
    'spectrum 22_Page_101.jpg': 'FULL_BLEED',
    'spectrum 22_Page_130.jpg': 'FULL_BLEED',
    'spectrum 22_Page_155.jpg': 'FULL_BLEED',
    'spectrum 22_Page_179.jpg': 'FULL_BLEED',
    'spectrum 22_Page_249.jpg': 'FULL_BLEED',
    # Spectrum 16 p006/p108/p180/p188/p207/p235: classify_page sent six real
    # full-bleed PAINTINGS to SKIP via the "mottled / decorative leaf" branch
    # (heavy saturated coverage on a shaded/dark ground). Verified on-screen this
    # session -- all six are single full-bleed art plates (colour cov 22-63%,
    # meanL 135-168). Forced FULL_BLEED gives clean reconstructions. Same surgical
    # approach as Spectrum 13 p080/086/106/177 and Spectrum 12 p019/p077 -- do NOT
    # loosen classify_page. p002 is a genuine blank white leaf (meanL 255, 0% colour)
    # and is correctly left as SKIP.
    'Spectrum 16_Page_006.jpg': 'FULL_BLEED',
    'Spectrum 16_Page_108.jpg': 'FULL_BLEED',
    'Spectrum 16_Page_180.jpg': 'FULL_BLEED',
    'Spectrum 16_Page_188.jpg': 'FULL_BLEED',
    'Spectrum 16_Page_207.jpg': 'FULL_BLEED',
    'Spectrum 16_Page_235.jpg': 'FULL_BLEED',
    # Spectrum 13 p080/p086/p106/p177: classify_page sent four real full-bleed
    # PAINTINGS to SKIP -- each read as a "mottled / decorative leaf" (the SKIP branch)
    # because of heavy saturated coverage on a dark ground (colour cov 28-55%). Verified
    # on-screen this session: all four are single full-bleed art plates. Forced
    # FULL_BLEED gives a clean reconstruction (same surgical fix as Spectrum 12 p019/p077,
    # Spectrum 10 p042/p121, Spectrum 9 p045 -- do NOT loosen classify_page). p002 is a
    # genuine blank leaf and is correctly left as SKIP.
    'Spectrum 13_Page_080.jpg': 'FULL_BLEED',
    'Spectrum 13_Page_086.jpg': 'FULL_BLEED',
    'Spectrum 13_Page_106.jpg': 'FULL_BLEED',
    'Spectrum 13_Page_177.jpg': 'FULL_BLEED',
    # Spectrum 12 p003/p019/p077: classify_page sent three real content pages to SKIP.
    # Verified on-screen + by measurement this session (same surgical approach as the
    # Spectrum 9/10 entries below -- do NOT loosen classify_page):
    #   p003 -> TEXT: light TONED title/section page (meanL 198, ~2.3%% dark ink). SKIP
    #           dropped it; TEXT whitens the ground and keeps the title text.
    #   p019 -> FULL_BLEED: full-bleed saturated PAINTING (96.7%% colour coverage) read
    #           as a mottled/decorative leaf and dropped (cf. Spectrum 9 p045).
    #   p077 -> FULL_BLEED: full-bleed atmospheric PAINTING (29.6%% colour coverage)
    #           dropped the same way (cf. Spectrum 10 p042).
    'Spectrum 12_Page_003.jpg': 'TEXT',
    'Spectrum 12_Page_019.jpg': 'FULL_BLEED',
    'Spectrum 12_Page_077.jpg': 'FULL_BLEED',
    # Spectrum 10 p042/p120/p121: classify_page sent all three to SKIP. Verified
    # (measurement + on-screen) they are real content, so pin them (same surgical
    # approach as the Spectrum 9 p045/p131 entries -- do NOT loosen classify_page):
    #   p042 -> FULL_BLEED: vivid multi-plate showcase (3 real plates, 37%% colour)
    #           read as a mottled/decorative leaf and dropped.
    #   p121 -> FULL_BLEED: single full-bleed painting (31%% colour) dropped likewise.
    #   p120 -> TEXT: low-sat TONED text page (meanL 182). SKIP dropped it; FULL_BLEED
    #           would find a spurious box. TEXT whitens (182->227) and keeps the text.
    'Spectrum 10_Page_042.jpg': 'FULL_BLEED',
    'Spectrum 10_Page_121.jpg': 'FULL_BLEED',
    'Spectrum 10_Page_120.jpg': 'TEXT',
    # Spectrum 9 p045/p131: classify_page sent both to SKIP (p045 reads as a heavily
    # "mottled" high-coverage leaf -- but it is a vivid full-bleed painting; p131 is a
    # low-sat page). Forced FULL_BLEED to attempt a reconstruction; verify the crops.
    'Spectrum 9_Page_045.jpg': 'FULL_BLEED',
    'Spectrum 9_Page_131.jpg': 'TEXT',
    # Force a page type for specific filenames when the classifier guesses wrong.
    # Example: 'some_page.jpg': 'SKIP'  (skip)  or  'TEXT' / 'FULL_BLEED' / 'DARK_BG'.
    # Leave empty to rely entirely on classify_page().
    'Spectrum 20_Page_006.jpg': 'SKIP',   # copyright/indicia page (ISBN, sponsor
                                          # logos) with a decorative dragon behind
                                          # the text -- not a credited plate; the
                                          # art path tore it.
    # Spectrum 4 artist-index pages: multi-column small-text directory like p156/157
    # (which classify_page sends to TEXT correctly). p158/p159 slip to FULL_BLEED;
    # under v37 the art path + rewritten _clean_background ERASES most of the index
    # text (only the column inside the detected rect survives) -- worse than the v34
    # grey-ground result. Force TEXT so they get the clean whitened-columns path the
    # sibling index pages get. (classify_page under-scores text_structure on these two
    # scans; a future gate tune could catch them, but do not loosen it enough to pull
    # real plates into TEXT.)
    'Spectrum 4_Page_158_Image_0001.jpg': 'TEXT',
    'Spectrum 4_Page_159_Image_0001.jpg': 'TEXT',
    # Spectrum 2 back-matter artist index (two credit columns, ~8 and ~7 lines of
    # small low-saturation text). classify_page sent p154 to SKIP -- but it holds real
    # index text (59%% non-paper, sat 12), so force TEXT for the clean whitened-columns
    # path like its sibling index pages.
    'Spectrum 2_Page_154.jpg': 'TEXT',
    # (Spectrum 2 p013 essay and p153 index no longer need overrides -- the v40 >74%
    # text+paper TEXT gate classifies both as TEXT automatically.)
}


# ── Panel detection ───────────────────────────────────────────────────────────

def detect_panels(img_bgr, page_type, bg_level, exclude_mask=None):
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    WHITE = bg_level - 20

    if page_type == 'FULL_BLEED':
        # One art rectangle on shaded paper (optionally with a caption block,
        # award header, or page number off to the side). Largest solid 'content'
        # component = the art. Primary mask is tone-based (flat-fielded dark OR
        # saturated); if that fragments on a low-contrast painting (small/thin
        # result) retry with a texture-inclusive mask. Sparse caption text and
        # page numbers stay as small separate components and lose to the art.
        sat   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
        illum = cv2.GaussianBlur(gray, (0, 0), 120)
        flat  = np.clip(gray.astype(np.float32) / (illum.astype(np.float32) + 1e-3) * 255, 0, 255)

        def _pick(content):
            content = cv2.morphologyEx(content, cv2.MORPH_OPEN,  np.ones((9, 9),   np.uint8))
            content = cv2.morphologyEx(content, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
            ncc, _, st, _ = cv2.connectedComponentsWithStats(content, 8)
            if ncc <= 1:
                return None
            j = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
            return [int(st[j, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                            cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT)]

        box = _pick((((flat < 198) | (sat > 50)).astype(np.uint8)) * 255)
        if box is None or box[2]*box[3] < 0.25*W*H or box[2] < 0.3*W or box[3] < 0.3*H:
            edges = cv2.dilate(cv2.Canny(gray, 40, 120), np.ones((5, 5), np.uint8))
            dens  = cv2.blur(edges.astype(np.float32), (31, 31))
            alt = _pick((((dens > 25) | (sat > 40) | (flat < 175)).astype(np.uint8)) * 255)
            if alt is not None and (box is None or alt[2]*alt[3] > box[2]*box[3]):
                box = alt
        if box is None:
            return [(0, 0, W, H)]
        x, y, w, h = box
        p = 6
        x = max(0, x - p); y = max(0, y - p)
        w = min(W - x, w + 2*p); h = min(H - y, h + 2*p)
        return [(x, y, w, h)]

    if page_type == 'DARK_BG':
        # Art-on-black (v20). The old (gray<=50 floodfill-from-corner) approach is
        # defeated two ways on these pages: GLOSS makes parts of the "black" bright
        # (so it is not flooded as background and leaks into the art mask), and the
        # art's own DARK passages sit at the same luminance as the ground (so a pure
        # brightness cut cannot separate them). Instead build a positive ART mask
        # from three cues that the smooth dark ground lacks, then take the central
        # mass and exclude the consistent dark edge-margin and the white caption.
        ring_med = bg_level                      # dark ground level (ring median)
        sat   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
        # structure: art has edges/detail; the ground (even glossy) is smooth
        edges = cv2.dilate(cv2.Canny(gray, 30, 100), np.ones((3, 3), np.uint8))
        dens  = cv2.blur(edges.astype(np.float32), (41, 41))
        bright = gray.astype(np.float32) > (ring_med + 38)   # lit foreground
        # art = clearly brighter than ground, OR coloured, OR textured/detailed
        art = ((bright) | (sat > 55) | (dens > 18)).astype(np.uint8) * 255
        # drop the white CAPTION band so the credit line doesn't extend the box
        cap = white_caption_mask(gray, ring_med)
        art[cap > 0] = 0
        # drop any paragraph-block region (handled separately) so a page-filling
        # text blurb (e.g. a back-cover) doesn't pull the art box over the whole page
        if exclude_mask is not None:
            art[exclude_mask > 0] = 0
        # suppress the thin dark edge-margin ring (binding streaks / corner gloss)
        mb = max(6, int(min(H, W) * 0.012))
        art[:mb, :] = 0; art[-mb:, :] = 0; art[:, :mb] = 0; art[:, -mb:] = 0
        art = cv2.morphologyEx(art, cv2.MORPH_OPEN,  np.ones((7, 7),   np.uint8))
        art = cv2.morphologyEx(art, cv2.MORPH_CLOSE, np.ones((45, 45), np.uint8))
        ncc, _, st, _ = cv2.connectedComponentsWithStats(art, 8)
        boxes = []
        for i in range(1, ncc):
            x, y, w, h, a = st[i, :5]
            if w * h > 0.05 * W * H and 0.1 < w / max(h, 1) < 10 and a > 0.02 * W * H:
                boxes.append((x, y, w, h))
        # merge boxes of one split artwork (heavy overlap / adjacency) into one
        boxes = _merge_overlapping(boxes)
        return sorted(boxes, key=lambda b: (b[1], b[0])) or [(0, 0, W, H)]


    # TEXT_LEFT: find art panels using Canny, exclude text column
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:,:,1]

    # Locate text column right edge
    col_sat = uniform_filter1d(sat.mean(axis=0).astype(float), size=50)
    text_col_right = 0
    leftmost_sat = col_sat[:int(W*0.03)].mean()
    if leftmost_sat < 35:
        for x in range(W // 2):
            if col_sat[x] > 50:
                text_col_right = x
                break

    edges  = cv2.Canny(gray, 30, 100)
    kernel = np.ones((30,30), np.uint8)
    closed = cv2.morphologyEx(cv2.dilate(edges, kernel), cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < 0.015*W*H or w < 80 or h < 80 or h >= H:
            continue
        # Exclude if mostly in text column zone
        if x + w//2 < text_col_right + 50:
            continue
        # Reject very low saturation blobs (binder margins, white strips)
        patch = img_bgr[y:y+h, x:x+w]
        phsv  = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        mean_sat = phsv[:,:,1].mean()
        if mean_sat < 35 and w/h < 0.4:
            continue
        if mean_sat < 12:  # pure margin/white strip, regardless of shape
            continue

        # If blob spans nearly full width AND full height, it's a merged blob.
        # Split it by finding horizontal white gaps (rows > 80% white) in the middle third.
        if w > W * 0.7 and h > H * 0.5:
            sub_gray = gray[y:y+h, max(0, text_col_right):]
            row_means = sub_gray.mean(axis=1)
            gap_threshold = max(180, WHITE - 30)
            gap_rows = np.where(row_means > gap_threshold)[0]
            # Find a gap in the middle third of this box
            mid_lo = h // 4; mid_hi = 3 * h // 4
            mid_gaps = gap_rows[(gap_rows > mid_lo) & (gap_rows < mid_hi)]
            if len(mid_gaps) >= 5:
                split_y = y + int(np.median(mid_gaps[:10]))
                # Top sub-panel
                if split_y - y > 100:
                    boxes.append((text_col_right, y, W - text_col_right, split_y - y))
                # Bottom sub-panel
                if (y + h) - split_y > 100:
                    boxes.append((text_col_right, split_y, W - text_col_right, (y + h) - split_y))
                continue

        boxes.append((x,y,w,h))

    if not boxes:
        # Fallback: content to the right of text column
        cx = max(text_col_right, int(W*0.05))
        content = (gray[:, cx:] < WHITE).astype(np.uint8)
        rows = np.where(content.any(axis=1))[0]
        cols = np.where(content.any(axis=0))[0]
        if len(rows) and len(cols):
            boxes = [(cx + int(cols[0]), int(rows[0]),
                      int(cols[-1]-cols[0]), int(rows[-1]-rows[0]))]

    return sorted(boxes, key=lambda b:(b[1]//500, b[0]))


# ── Clean straight crop ───────────────────────────────────────────────────────

def clean_crop(img_bgr, bx, by, bw, bh, bg_level, search=100, smooth=60):
    """
    Scan inward from each bbox edge to find the printed border.
    90th-percentile of each edge profile = deepest inward position
    = guaranteed straight rectangular cut.
    Returns (cropped_image, (abs_x0, abs_y0, abs_x1, abs_y1))
    """
    H, W = img_bgr.shape[:2]
    WHITE = bg_level - 20
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(float)
    sy    = np.abs(np.diff(gray, axis=0))
    sx    = np.abs(np.diff(gray, axis=1))

    at_top   = by <= 5
    at_bot   = by + bh >= H - 5
    at_left  = bx <= 5
    at_right = bx + bw >= W - 5

    def first_above(arr):
        h = np.where(arr > 8)[0]; return int(h[0]) if len(h) else 0
    def last_above(arr):
        h = np.where(arr > 8)[0]; return int(h[-1]) if len(h) else len(arr)-1
    def first_nw(arr):
        h = np.where(arr < WHITE)[0]; return int(h[0]) if len(h) else 0
    def last_nw(arr):
        h = np.where(arr < WHITE)[0]; return int(h[-1]) if len(h) else len(arr)-1

    top_arr, bot_arr, left_arr, right_arr = [], [], [], []
    for cx in range(bx, bx+bw):
        y0,y1 = max(0,by-search), min(H-1,by+search)
        top_arr.append(y0 + (first_nw(gray[y0:y1,cx]) if at_top  else first_above(sy[y0:y1,cx])))
        y0b,y1b = max(0,by+bh-search), min(H-1,by+bh+search)
        bot_arr.append(y0b + (last_nw(gray[y0b:y1b,cx]) if at_bot else last_above(sy[y0b:y1b,cx])))
    for ry in range(by, by+bh):
        x0,x1 = max(0,bx-search), min(W-1,bx+search)
        left_arr.append(x0 + (first_nw(gray[ry,x0:x1]) if at_left  else first_above(sx[ry,x0:x1])))
        x0r,x1r = max(0,bx+bw-search), min(W-1,bx+bw+search)
        right_arr.append(x0r + (last_nw(gray[ry,x0r:x1r]) if at_right else last_above(sx[ry,x0r:x1r])))

    top   = uniform_filter1d(np.array(top_arr,   float), size=smooth)
    bot   = uniform_filter1d(np.array(bot_arr,   float), size=smooth)
    left  = uniform_filter1d(np.array(left_arr,  float), size=smooth)
    right = uniform_filter1d(np.array(right_arr, float), size=smooth)

    y0c = int(np.percentile(top,   75))
    y1c = int(np.percentile(bot,   25))
    x0c = int(np.percentile(left,  75))
    x1c = int(np.percentile(right, 25))

    y0c = max(0,y0c); y1c = min(H,y1c)
    x0c = max(0,x0c); x1c = min(W,x1c)

    if y1c <= y0c or x1c <= x0c:
        return img_bgr[by:by+bh, bx:bx+bw].copy(), (bx,by,bx+bw,by+bh)

    return img_bgr[y0c:y1c, x0c:x1c].copy(), (x0c, y0c, x1c, y1c)


def deslope_crop(img_bgr, bx, by, bw, bh, bg_level, min_angle=0.08):
    """clean_crop, but first remove the panel's tilt. _reliable_angle measures the
    panel's rotation off its strong edges (and self-rejects via its residual gate,
    returning ~0 when no trustworthy angle exists). If the tilt is meaningful, a
    padded region around the box is rotated about the box centre, then clean_crop
    makes its straight cut on the now-upright panel. Pasting at the returned page
    coords lands it back in place (rotation about the centre preserves position).
    Falls back to plain clean_crop when the panel is already square."""
    theta = _reliable_angle(img_bgr, (bx, by, bw, bh), bg_level)
    if abs(theta) < min_angle:
        clean, pos = clean_crop(img_bgr, bx, by, bw, bh, bg_level)
        return clean, pos, 0.0
    H, W = img_bgr.shape[:2]
    pad = int(0.06 * max(bw, bh))
    sx0, sy0 = max(0, bx - pad), max(0, by - pad)
    sx1, sy1 = min(W, bx + bw + pad), min(H, by + bh + pad)
    sub = img_bgr[sy0:sy1, sx0:sx1]
    sH, sW = sub.shape[:2]
    cx, cy = (bx + bw / 2.0 - sx0), (by + bh / 2.0 - sy0)
    M = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)
    rot = cv2.warpAffine(sub, M, (sW, sH), flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_REPLICATE)
    clean, (rx0, ry0, rx1, ry1) = clean_crop(rot, bx - sx0, by - sy0, bw, bh, bg_level)
    return clean, (sx0 + rx0, sy0 + ry0, sx0 + rx1, sy0 + ry1), theta


# ── Main ──────────────────────────────────────────────────────────────────────

files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith('.jpg'))
if os.environ.get('DEWARP_IMPORT') == '1':
    # Imported as a function library (ablation harness) -- do not run the batch
    # driver (profile build + per-page loop). Callers set PAPER_PROFILE themselves.
    files = []

# ── Magenta markup <-> clean original pairing ────────────────────────────────
# A magenta page is a MARKUP copy (re-compressed, ink over the art): it supplies
# only the box geometry, and the deliverable is reconstructed from the pixel-
# aligned CLEAN ORIGINAL. The magenta workflow therefore only makes sense for a
# DUPLICATE PAIR: the same page sent twice, one copy WITH magenta and one WITHOUT.
#
# LOGICAL GATE (v17): magenta is only ever evaluated for pages that have a
# DUPLICATE-STEM partner. We group staged files by NORMALISED stem (download-dedup
# suffixes like __2_, (1) and _orig/_clean/_mag tags stripped) and ONLY decode +
# magenta-check the groups with >=2 files. A page with a UNIQUE stem (no duplicate)
# can never be a valid markup, so it is NEVER checked for magenta colour at all --
# it goes straight to normal processing. This both saves work on ordinary batches
# (no per-page magenta scan) and removes the false-positive risk where a clean page
# that merely contains red/pink/magenta paint was mistaken for a markup.
# Within a duplicate group: a file with >=1 real magenta LOOP is a markup; a file
# with no loop is a clean candidate; markup+clean of matching size are paired.
#   - _orig_for[markup]    -> the clean original to reconstruct from (the pair)
#   - _consumed_originals  -> files used as an original (skip as standalone)
#   - _unpaired_markups    -> markup in a duplicate group with NO clean partner
#                             (reconstruction refused; ask for the clean original)

def _normstem(fname):
    # Strip download-dedup + markup tags AND OS duplicate-copy suffixes so a markup
    # pairs with its clean original. v28 adds the Windows/browser " - Copy" /
    # "_-_Copy" / "- Copy (2)" and macOS " copy" / " copy 2" forms (looped, so a
    # stacked tag like "_orig - Copy" is fully reduced). Anchored to the end and
    # case-insensitive; ordinary page names ("..._Image_0001") are left untouched.
    s = fname[:-4] if fname.lower().endswith('.jpg') else fname
    pat = re.compile(
        r'(__\d+_?|_\(\d+\)|_orig(?:inal)?|_clean|_mag(?:enta)?'
        r'|[ _]-[ _]?[Cc]opy(?:[ _]?\(\d+\))?'      # "name - Copy", "name_-_Copy (2)"
        r'|[ _][Cc]opy(?:[ _]?\d+)?)$',              # macOS "name copy", "name copy 2"
        flags=re.IGNORECASE)
    while True:
        ns = pat.sub('', s)
        if ns == s:
            return ns
        s = ns

_by_norm = {}
for _f in files:
    _by_norm.setdefault(_normstem(_f), []).append(_f)

_orig_for = {}
_consumed_originals = set()
_unpaired_markups = set()
# Only duplicate-stem groups are decoded/checked for magenta (the logical gate).
for _norm, _group in _by_norm.items():
    if len(_group) < 2 or not MAGENTA_OVERRIDE:
        continue                                   # unique stem -> never magenta
    _dec = {g: cv2.imread(os.path.join(INPUT_DIR, g)) for g in _group}
    # A file is a MARKUP only if it has real closed magenta LOOPS (>=1 box); a
    # clean original can carry incidental magenta-ish art pixels (reddish/purple
    # paint) that trip the loose has_magenta pixel test but form no loop.
    # v54: a text-block GUIDE is an open box (no closed loop) yet still a markup, so
    # also accept an assembled guide with >=3 drawn sides (incidental paint forms
    # no such thin-stroke rectangle, so this stays false-positive-safe).
    import magenta_text as _mtext
    def _is_markup(im):
        if im is None:
            return False
        if len(magenta_crop.magenta_boxes(im)) >= 1:
            return True
        # a text WARP guide is open (no loop): a stack of >=2 horizontal guides, or a
        # >=3-sided box. Incidental paint forms no such thin-stroke set -> safe.
        for r in _mtext.magenta_regions(im):
            if len(r['hlines']) >= 2 or (len(r['hlines']) + len(r['vlines'])) >= 3:
                return True
        return False
    _mags   = [g for g, im in _dec.items() if _is_markup(im)]
    _cleans = [g for g, im in _dec.items() if im is not None and g not in _mags]
    for _mg in _mags:
        _paired = False
        for _cl in _cleans:
            if _cl not in _consumed_originals and _dec[_cl].shape[:2] == _dec[_mg].shape[:2]:
                _orig_for[_mg] = _cl
                _consumed_originals.add(_cl)
                _paired = True
                break
        if not _paired:
            _unpaired_markups.add(_mg)


# Off-white paper profile. REBUILT EVERY RUN over the files currently staged in
# INPUT_DIR. The profile (s_max/dv) is an ENVELOPE over the staged batch, so it is
# stable across that batch (the files you dropped in -- e.g. ~80 pages) and can
# never silently inherit a previous run's numbers. No cached profile is read back
# to skip this; paper_profile.json is (over)written purely as a provenance record
# (page count + build time).
#   The envelope is most stable over MANY files: building over a large staged batch
# gives one consistent threshold for all of them, whereas a tiny slice can drift
# (a few saturated plates move s_max/dv). So stage the whole batch together rather
# than a handful of files per run -- see _MIN_PROFILE_PAGES below.
_pp = os.path.join(os.path.dirname(__file__), 'paper_profile.json')
_MIN_PROFILE_PAGES = 20
_CACHE_PROFILE = os.environ.get('DEWARP_CACHED_PROFILE') == '1'
if files and _CACHE_PROFILE and os.path.exists(_pp):
    # CHUNKED-RUN FAST PATH (output-neutral): the staged batch is fixed across
    # chunks, so the envelope is identical every run. Load the already-built
    # profile instead of re-decoding all staged files each chunk. Only enabled
    # explicitly via DEWARP_CACHED_PROFILE=1 for a multi-chunk batch.
    import json as _json2
    with open(_pp) as _f:
        deskew_crop.PAPER_PROFILE = _json2.load(_f)
    _p = deskew_crop.PAPER_PROFILE
    print(f"paper profile (CACHED, staged set fixed): "
          f"s_max={_p['s_max']} dv={_p['dv']}")
elif files:
    deskew_crop.PAPER_PROFILE = deskew_crop.write_paper_profile(
        [os.path.join(INPUT_DIR, f) for f in files], _pp)
    _p = deskew_crop.PAPER_PROFILE
    print(f"paper profile (built this run over {_p['n_pages']} staged files): "
          f"s_max={_p['s_max']} dv={_p['dv']}")
    if len(files) < _MIN_PROFILE_PAGES:
        print(f"  NOTE: only {len(files)} files staged -- the envelope is most "
              f"stable over >= {_MIN_PROFILE_PAGES} files. Stage the whole batch "
              "together so every file shares one threshold.")
else:
    print("paper profile: no files staged in INPUT_DIR; nothing to build.")

print(f'Processing {len(files)} pages...\n')

for fname in files:
    stem = fname.replace('.jpg', '')

    # This staged file is the clean original feeding a magenta markup -> it is
    # reconstructed via that markup's box geometry, not as a standalone page.
    if fname in _consumed_originals:
        print(f'{fname}: clean original for a magenta markup — consumed, not standalone')
        continue

    out_stem = _normstem(fname)   # magenta output is named after the normalised page

    # Resume: if this page already has an output (reconstructed under its own or its
    # normalised stem, or text), skip it before the costly decode.
    if RESUME and (
        os.path.exists(os.path.join(OUTPUT_DIR, f'{out_stem}_reconstructed.jpg')) or
        os.path.exists(os.path.join(OUTPUT_DIR, f'{stem}_reconstructed.jpg')) or
        os.path.exists(os.path.join(OUTPUT_DIR, f'{stem}_text.jpg'))):
        print(f'{fname}: already done — skipped')
        continue

    path = os.path.join(INPUT_DIR, fname)
    img  = cv2.imread(path)
    if img is None: continue


    H, W = img.shape[:2]

    # Magenta handling is driven ENTIRELY by the duplicate-pair pre-pass above; a
    # page is touched here only if it was paired (or flagged unpaired) there. A
    # unique-stem page reaches this point WITHOUT ever being magenta-checked and
    # falls straight through to normal processing.
    if fname in _orig_for:
        # Paired markup: BOXES from the markup, PIXELS from the clean original
        # (full deskew/dewarp on clean pixels), never the markup copy.
        orig_name = _orig_for[fname]
        orig_img = cv2.imread(os.path.join(INPUT_DIR, orig_name))
        # v26: respect ground polarity. magenta_dewarp ALWAYS whitens (paper=True),
        # which is wrong for art-on-black. A DARK_BG original keeps its black ground:
        # its magenta boxes drive the dark keep-black chain (dark_art) instead.
        o_type, _obg = classify_page(orig_img)
        if orig_name in PAGE_OVERRIDES: o_type = PAGE_OVERRIDES[orig_name]
        if fname in PAGE_OVERRIDES:     o_type = PAGE_OVERRIDES[fname]
        if o_type == 'DARK_BG':
            mboxes = magenta_crop.magenta_boxes(img)
            out = dark_art.reconstruct_dark_art_with_boxes(orig_img, mboxes)
            # v55: magenta-guided TEXT-WARP correction also on dark (art-on-black)
            # pages -- e.g. a white-on-black colophon flattened to its warp guides.
            # Source the composed dark page so the block keeps the BLACK ground.
            import magenta_text as _mtext
            _ntext = 0
            for _r in _mtext.magenta_regions(img):
                if not _mtext.region_is_text(orig_img, _r['bbox']):
                    continue
                _patch, _xy = _mtext.correct_text_block(out, _r, flatfield_whiten=None)
                if _patch is not None:
                    _px, _py = _xy
                    out[_py:_py + _patch.shape[0], _px:_px + _patch.shape[1]] = _patch
                    _ntext += 1
            cv2.imwrite(os.path.join(OUTPUT_DIR, f'{out_stem}_reconstructed.jpg'),
                        out, [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f'{fname}: MAGENTA boxes ({len(mboxes)}) + original {orig_name} '
                  f'-> DARK keep-black reconstruct'
                  + (f' + {_ntext} text block(s) corrected' if _ntext else ''))
            continue
        out, boxes, thetas = magenta_crop.magenta_dewarp(img, orig_img, flatfield_whiten)
        if out is None:
            # TEXT-ONLY markup (no art boxes): no plates to compose, so build the base
            # by flat-field whitening the clean original -- paper lifted to page white,
            # text/colour (e.g. a spine) kept IN PLACE so the guides still register.
            out = flatfield_whiten(orig_img)
            boxes, thetas = [], []
        # v54: magenta-guided TEXT-BLOCK correction. A guide over a TEXT column
        # (open or closed box; the TOP stroke is a skew/warp reference, not a crop
        # edge) is flattened to its guides and whitened, then pasted over the art
        # result. Art-only markups have no text region, so this is a no-op for them.
        import magenta_text as _mtext
        _tregions = [r for r in _mtext.magenta_regions(img)
                     if _mtext.region_is_text(orig_img, r['bbox'])]
        _ntext = 0
        for _r in _tregions:
            # Flatten the ALREADY-COMPOSED, page-whitened region so the text block
            # keeps the page's own background tone (no separate whitening pass).
            _patch, _xy = _mtext.correct_text_block(out, _r, flatfield_whiten=None)
            if _patch is not None:
                _px, _py = _xy
                out[_py:_py + _patch.shape[0], _px:_px + _patch.shape[1]] = _patch
                _ntext += 1
        if EDGE_SLIVER_FIX:
            out = deskew_crop.remove_edge_slivers(out)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f'{out_stem}_reconstructed.jpg'),
                    out, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f'{fname}: MAGENTA boxes ({len(boxes)}) + original {orig_name} -> '
              'dewarped ' + ', '.join(f'{t:+.2f}' for t in thetas) + ' deg'
              + (f' + {_ntext} text block(s) corrected' if _ntext else ''))
        continue
    if fname in _unpaired_markups:
        # Markup in a duplicate group but no clean original of matching size: do NOT
        # reconstruct the markup; save its boxes and ask for the pixel-aligned original.
        boxes = magenta_crop.magenta_boxes(img)
        sc = os.path.join(OUTPUT_DIR, f'{out_stem}.magboxes.json')
        with open(sc, 'w') as fh:
            json.dump({'boxes': boxes, 'shape': list(img.shape[:2]), 'markup': fname}, fh)
        print(f'{fname}: MAGENTA markup, {len(boxes)} box(es) — no matching clean '
              f'original. NOT reconstructed; stage the pixel-aligned original (same '
              f'scan, no magenta) and re-run; boxes saved -> {os.path.basename(sc)}')
        continue

    page_type, bg = classify_page(img)
    if fname in PAGE_OVERRIDES:
        page_type = PAGE_OVERRIDES[fname]
    print(f'{fname}: {page_type}', end='')

    if page_type == 'SKIP':
        print(' — skipped')
        continue

    # Page-level curl pre-pass (multi-picture pages only; auto-no-op otherwise).
    if PAGE_DEWARP and page_type in ('FULL_BLEED', 'TEXT_LEFT', 'DARK_BG'):
        img, _pd = page_dewarp.dewarp_page(img, bg)
        if _pd.get('applied'):
            print(f" [page-decurl {_pd['edges_ok']}edges {_pd['field_range']}]", end='')

    if page_type == 'TEXT':
        # No panels: curl dewarp + flat-field whiten, original pixels relocated.
        out = process_text_page(img)
        stem = fname.replace('.jpg', '')
        outpath = os.path.join(OUTPUT_DIR, f'{stem}_text.jpg')
        cv2.imwrite(outpath, out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(' — dewarped + whitened')
        continue

    if page_type == 'FULL_BLEED':
        # Single art rectangle. Background = the real page paper, flat-field
        # whitened (vignette removed, paper TEXTURE preserved) so the art's own
        # deckle/margin blends in instead of clashing with a flat fill. All page
        # text (page number, plate number, artist caption, signature) is retained
        # automatically because it lives in the whitened page. The pristine,
        # straight-cropped art is restored on top so it isn't tone-shifted.
        # Deskew (dewarp the single art rectangle) THEN deliberately crop a hair
        # INTO the art: deepest inward straight cut + margin guarantees a
        # perfect axis-aligned rectangle with no paper sliver / skew wedge /
        # tilted keyline. Whitened page underneath keeps all page text; the
        # original skewed art rim is erased to paper so no ghost shows.
        # How many distinct artworks on the page? Index / showcase pages carry
        # 2+ separate pictures plus a caption column; ordinary plates carry one.
        pics = art_pictures(img, bg)
        stem = fname.replace('.jpg', '')
        if len(pics) >= 2:
            print(f' -> {len(pics)} pictures (multi)')
            out, thetas = compose_multi(img, bg, flatfield_whiten, pics,
                                         center_content=CENTER_MULTI)
            print('    deskew ' + ', '.join(f'{t:+.2f}' for t in thetas) + ' deg')
        else:
            print(' -> 1 panel')
            out, theta = compose_fullbleed(img, pics[0], bg, flatfield_whiten,
                                            center=CENTER_SINGLE)
            print(f'    deskew {theta:+.2f} deg')
        if EDGE_SLIVER_FIX:
            out = deskew_crop.remove_edge_slivers(out)
        # v43: straighten the page TEXT (caption/credit columns, headers) to match the
        # deskewed plates. The detected picture rectangles are passed as no-go zones
        # (dilated a touch to cover the deskew shift), and the routine itself rejects
        # any straightened block that would land on art or on another text block.
        if DESKEW_PAGE_TEXT:
            _H, _W = out.shape[:2]
            _dil = int(0.015 * min(_H, _W))
            art_ex = [(max(0, x0-_dil), max(0, y0-_dil),
                       (x1-x0)+2*_dil, (y1-y0)+2*_dil) for (x0, y0, x1, y1) in pics]
            out, ta = text_blocks.deskew_text_blocks(out, art_ex, dark_text=True)
            if ta:
                print('    text-deskew ' + ', '.join(f'{t:+.2f}' for t in ta) + ' deg')
        # paragraphs (3+ lines) beside the plate: deskew each as its own block on
        # the finished page (detected on `out` so centred layouts stay aligned),
        # placed without overlapping art or being cropped -- black text on pale.
        if not SAFE_TEXT:
            out, pa = text_blocks.straighten_paragraphs(
                out, out, [], detect_dark_text=True, stamp='ink', wipe=True)
            if pa:
                print('    para-deskew ' + ', '.join(f'{t:+.2f}' for t in pa) + ' deg')
        # v25: short text (1-2 line captions / headers / credit lines) -- same
        # tested deskew+restamp path, capped to short blocks so it complements the
        # 3+ line pass above. Toggle DESKEW_SHORT_TEXT.
        if DESKEW_SHORT_TEXT and not SAFE_TEXT:
            out, sa = text_blocks.straighten_paragraphs(
                out, out, [], detect_dark_text=True, stamp='ink', wipe=True,
                min_lines=1, max_lines=2)
            if sa:
                print('    short-text-deskew ' + ', '.join(f'{t:+.2f}' for t in sa) + ' deg')
        cv2.imwrite(os.path.join(OUTPUT_DIR, f'{stem}_reconstructed.jpg'),
                    out, [cv2.IMWRITE_JPEG_QUALITY, 92])
        continue

    dark_bg   = page_type == 'DARK_BG'

    # Canvas/background. TEXT_LEFT pages are on scanned paper: use the real paper
    # TEXTURE by basing the canvas on the flat-field-whitened page (lighting
    # normalised, paper grain + caption text + page/plate numbers kept) rather
    # than a flat average colour. Cleaned panels paste on top; anywhere a crop
    # bites in, the original art shows through the base instead of a blank fill.
    H, W = img.shape[:2]

    if dark_bg:
        # ART-ON-BLACK = a LIGHT-BG PAGE INVERTED. Reuse the light pipeline's CORE
        # dewarp (deskew_deep_crop = deskew + deep-crop + perspective rectify) on the
        # inverted page, invert the cleaned rectangle back, and place it on the KEPT
        # black page (header + caption stay; only the art footprint is repainted with
        # grain sampled from the page's own dark ground). No paper centering/erase.
        stem = fname.replace('.jpg', '')
        inv = 255 - img
        tmp_inv = os.path.join('/tmp', f'_inv_{os.getpid()}.jpg')
        cv2.imwrite(tmp_inv, inv)
        _prof = deskew_crop.PAPER_PROFILE
        deskew_crop.PAPER_PROFILE = deskew_crop.build_paper_profile([tmp_inv])
        bg_inv = int(np.percentile(cv2.cvtColor(inv, cv2.COLOR_BGR2GRAY), 85))
        crop_inv, (pcx, pcy), theta, comp = deskew_crop.deskew_deep_crop(inv, bg_inv)
        deskew_crop.PAPER_PROFILE = _prof
        crop = 255 - crop_inv                         # cleaned art, back to dark
        base = img.copy()                             # keep the black page
        dark = _bg_texture(img, paper=False)
        fp = cv2.dilate(comp.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
        base[fp] = dark[fp]                           # repaint old art footprint with sampled grain
        ch, cw = crop.shape[:2]
        px = int(round(pcx - cw/2)); py = int(round(pcy - ch/2))
        px = max(0, min(px, W - cw)); py = max(0, min(py, H - ch))
        base[py:py+ch, px:px+cw] = crop
        print(f' -> light dewarp via inversion, deskew {theta:+.2f} deg')
        cv2.imwrite(os.path.join(OUTPUT_DIR, f'{stem}_reconstructed.jpg'),
                    base, [cv2.IMWRITE_JPEG_QUALITY, 92])
        continue

    canvas = flatfield_whiten(img)

    boxes = detect_panels(img, page_type, bg)
    print(f' -> {len(boxes)} panels', end='')

    desloped = []
    for (bx, by, bw, bh) in boxes:
        if DESLOPE_TEXT:
            clean, (cx0, cy0, cx1, cy1), theta = deslope_crop(img, bx, by, bw, bh, bg)
        else:
            clean, (cx0, cy0, cx1, cy1) = clean_crop(img, bx, by, bw, bh, bg)
            theta = 0.0
        desloped.append(theta)
        cH, cW = clean.shape[:2]
        dy0=max(0,cy0); dy1=min(H,cy0+cH)
        dx0=max(0,cx0); dx1=min(W,cx0+cW)
        sy0=dy0-cy0;    sy1=sy0+(dy1-dy0)
        sx0=dx0-cx0;    sx1=sx0+(dx1-dx0)
        if dy1>dy0 and dx1>dx0:
            canvas[dy0:dy1, dx0:dx1] = clean[sy0:sy1, sx0:sx1]
    nz = [t for t in desloped if abs(t) > 0.001]
    if nz:
        print('  deslope ' + ', '.join(f'{t:+.2f}' for t in nz) + ' deg')
    else:
        print('')
    outpath = os.path.join(OUTPUT_DIR, f'{stem}_reconstructed.jpg')
    if EDGE_SLIVER_FIX:
        canvas = deskew_crop.remove_edge_slivers(canvas)
    cv2.imwrite(outpath, canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])

print('\nDone.')

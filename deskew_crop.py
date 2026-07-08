"""
Deskew + deliberate-inward crop for scanned art plates.

Order of operations (dewarp-then-bite):
  1. DEWARP: rotate the page so the art is axis-aligned. The rotation angle is
     estimated from the art's border edges, but ONLY from edges that fit a
     straight line reliably (low residual). On dark-on-dark plates a noisy edge
     (e.g. a near-black sky meeting a dark scan margin) is discarded instead of
     dragging the angle off -- which previously over-rotated dark-on-dark plates.
  2. DELIBERATE INWARD CUT: take the straightened art rectangle and inset every
     side by a margin, so the axis-aligned crop is guaranteed fully inside the
     art -- no paper sliver, no skew wedge, no tilted keyline.

Extent + footprint come from a robust "non-paper largest component" (NOT from
detect_panels, whose bbox can blow out to the full page on dark plates). The
same component, dilated, is the erase footprint, so no washed art ghost can
survive around the crop, and the distant page/plate numbers (separate small
components) are never touched.

All heavy steps run at reduced resolution; the art output is full-res.
"""
import cv2
import numpy as np


# ----------------------------------------------------------------------------
# Paper-colour profile (off-white range), shared across the staged batch
# ----------------------------------------------------------------------------
PAPER_PROFILE = None   # set by the pipeline; dict {'s_max':int,'dv':int}


def _margin_pixels(hsv, H, W):
    o = int(0.05 * min(H, W)); i = int(0.015 * min(H, W))
    ring = np.ones((H, W), bool); ring[o:-o, o:-o] = False
    ring[:i, :] = False; ring[-i:, :] = False; ring[:, :i] = False; ring[:, -i:] = False
    px = hsv[ring]
    vth = np.percentile(px[:, 2], 60); sth = np.percentile(px[:, 1], 70)
    return px[(px[:, 2] >= vth) & (px[:, 1] <= sth)]   # bright low-sat cluster = paper


def build_paper_profile(paths, scale=0.34):
    """Analyse all pages together -> global off-white chroma/value bounds wide
    enough to cover every paper stock in the batch (grey, beige, blue, yellow ...).

    v36 -- ONLY PAGES WITH A REAL PAPER MARGIN DEFINE THE ENVELOPE. The earlier
    builder sampled the border ring of EVERY page as "paper". On a FULL-BLEED
    painting the border ring IS art, so those pages injected art saturation/darkness
    into the envelope and blew it up (one very dark scan hit s_max 211 / dv 70), after which the
    art/paper test flagged only near-neon or very dark pixels as art and every muted /
    desaturated / dark painting was mis-read as paper. Now each page is sampled only
    from its bright, low-saturation PAPER-LIKE ring pixels, and a page whose ring is
    mostly art (few paper-like pixels) is SKIPPED entirely, so it cannot pollute the
    bounds. The envelope still covers tinted stock (beige/yellow/blue are bright and
    only mildly saturated, so they pass the paper-like gate and contribute their tint)."""
    s98 = []; vdrop = []; n_paper = 0
    for p in paths:
        im = cv2.imread(p)
        if im is None:
            continue
        H, W = im.shape[:2]
        sm = cv2.resize(im, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV)
        Hs, Ws = sm.shape[:2]
        o = int(0.05 * min(Hs, Ws)); i = int(0.015 * min(Hs, Ws))
        ring = np.ones((Hs, Ws), bool); ring[o:-o, o:-o] = False
        ring[:i, :] = False; ring[-i:, :] = False; ring[:, :i] = False; ring[:, -i:] = False
        px = hsv[ring]
        if len(px) < 50:
            continue
        paper = px[(px[:, 2] > 190) & (px[:, 1] < 75)]      # bright + low-sat = real paper
        if len(paper) < 60 or len(paper) < 0.20 * len(px):  # ring is mostly art -> skip page
            continue
        n_paper += 1
        s98.append(np.percentile(paper[:, 1], 98))          # this page's paper chroma
        wp = np.median(paper[:, 2])
        vdrop.append(wp - np.percentile(paper[:, 2], 3))    # this page's paper value spread
    if not s98:                                             # batch has no clean-margin page
        return {'s_max': 60, 'dv': 45, 'n_paper': 0}
    # v62.3 ROBUST ENVELOPE. The old p95(+30 flat) let a big, varied batch drift the
    # saturation ceiling loose (100 pages -> s_max 101 vs 83 on a clean set), because a
    # handful of faintly-tinted/contaminated margins push the 95th percentile up and the
    # flat slack stacks on top -- loosening the art/paper test for EVERY page and causing
    # over-segmentation / mis-crops. Use an OUTLIER-RESISTANT centre (upper-quartile) plus
    # proportional slack, and CAP the result so a large batch can never sit looser than a
    # clean small set would. Batch size no longer degrades the thresholds.
    s_hi = float(np.percentile(s98, 75))                    # robust upper-quartile, not p95
    s_max = int(round(min(s_hi + 22, 88)))                  # capped: big batch can't drift loose
    d_hi = float(np.percentile(vdrop, 75))
    dv    = int(round(min(d_hi + 22, 78)))
    return {'s_max': s_max, 'dv': dv, 'n_paper': n_paper}


# ----------------------------------------------------------------------------
# Stamped profile write
# ----------------------------------------------------------------------------
# The paper profile is rebuilt every run over the staged batch; it is written to
# disk only as a provenance record (page count + build time), never read back to
# identify or match a particular set of files.
import json as _json
import time as _time


def write_paper_profile(paths, out_path):
    """Build the off-white paper profile over `paths` (the staged batch) and write
    it to out_path with the page count and build time. Returns the written dict
    (carries s_max/dv plus the stamp keys; the extra keys are harmless to the crop
    code)."""
    prof = dict(build_paper_profile(paths))
    prof['n_pages'] = len(paths)
    prof['built_at'] = _time.strftime('%Y-%m-%d %H:%M:%S')
    with open(out_path, 'w') as f:
        _json.dump(prof, f, indent=2)
    return prof


def _page_whitepoint(sm_hsv, H, W):
    pap = _margin_pixels(sm_hsv, H, W)
    if len(pap) >= 50:
        m = float(np.median(pap[:, 2]))
        if m > 185:                                  # margin is genuine bright paper
            return m
    # full-bleed page (no real paper margin): use the brightest low-sat pixels
    # anywhere as the white reference, so the art/paper test isn't anchored to a
    # dark art margin (which made dark/muted full-bleed art read as paper).
    flat = sm_hsv.reshape(-1, 3)
    lowsat = flat[flat[:, 1] < 60]
    if len(lowsat) > 200:
        return float(np.percentile(lowsat[:, 2], 96))
    return float(np.percentile(flat[:, 2], 96))



def _sample_edges(img, bx, by, bw, bh, bg, search=110):
    H, W = img.shape[:2]
    WHITE = bg - 20
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sy = np.abs(np.diff(gray, axis=0))
    sx = np.abs(np.diff(gray, axis=1))
    at_top = by <= 5; at_bot = by + bh >= H - 5
    at_left = bx <= 5; at_right = bx + bw >= W - 5

    def first_true(mask, axis):
        return np.where(mask.any(axis=axis), mask.argmax(axis=axis), 0)

    def last_true(mask, axis):
        any_ = mask.any(axis=axis); n = mask.shape[axis]
        idx = n - 1 - np.flip(mask, axis=axis).argmax(axis=axis)
        return np.where(any_, idx, n - 1)

    cols = slice(bx, bx + bw); rows = slice(by, by + bh)
    y0, y1 = max(0, by - search), min(H - 1, by + search)
    sub = (gray[y0:y1, cols] < WHITE) if at_top else (sy[y0:y1, cols] > 8)
    top = y0 + first_true(sub, 0).astype(np.float32)
    y0b, y1b = max(0, by + bh - search), min(H - 1, by + bh + search)
    sub = (gray[y0b:y1b, cols] < WHITE) if at_bot else (sy[y0b:y1b, cols] > 8)
    bot = y0b + last_true(sub, 0).astype(np.float32)
    x0, x1 = max(0, bx - search), min(W - 1, bx + search)
    sub = (gray[rows, x0:x1] < WHITE) if at_left else (sx[rows, x0:x1] > 8)
    left = x0 + first_true(sub, 1).astype(np.float32)
    x0r, x1r = max(0, bx + bw - search), min(W - 1, bx + bw + search)
    sub = (gray[rows, x0r:x1r] < WHITE) if at_right else (sx[rows, x0r:x1r] > 8)
    right = x0r + last_true(sub, 1).astype(np.float32)
    return top, bot, left, right


def _reliable_angle(img, box, bg, max_resid=15.0):
    """Median tilt (deg) from only the border edges that fit a line well.
    Returns 0.0 if no edge is reliable (treat as already straight)."""
    bx, by, bw, bh = box
    votes = []
    for arr in _sample_edges(img, bx, by, bw, bh, bg):
        x = np.arange(len(arr), dtype=np.float32)
        a, b = np.polyfit(x, arr, 1)
        resid = np.median(np.abs(arr - (a * x + b)))
        if resid < max_resid:
            votes.append(np.degrees(np.arctan(a)))
    return float(np.median(votes)) if votes else 0.0


def _text_line_mask(sm):
    """Binary mask (sm-resolution) of PAGE TEXT: caption / credit / body-copy lines.
    Text is a stack of WIDE, SHORT, HIGH-ASPECT dark ink runs at a regular pitch
    (the same signature text_structure keys on). This mask is SUBTRACTED from the
    art mask so caption text is treated as BACKGROUND, never art: it can no longer
    bridge separate plates into one page-wide blob, can no longer drag an art box
    sideways into the text column, and is never carried inside a plate's crop and
    re-stamped offset from the restored page text (the 'art text' smash, v42).

    Deliberately conservative: only genuine multi-glyph LINE runs qualify (h < 5%%
    of the page, w > 6%%, aspect > 4, well filled). Isolated marks, a plate's dark
    line-work, or a signature do NOT form regular wide-short runs and are kept."""
    sh, sw = sm.shape[:2]
    g = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(cv2.GaussianBlur(g, (3, 3), 0), 255,
                               cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    k = max(15, sw // 60)
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(closed, 8)
    tm = np.zeros((sh, sw), np.uint8)
    for i in range(1, n):
        x, y, w, h, a = st[i, :5]
        if 3 < h < sh * 0.05 and w > sw * 0.06 and w / max(h, 1) > 4 and a > w * 0.15:
            tm[lab == i] = 1
    return cv2.dilate(tm, np.ones((9, 9), np.uint8))


def _art_component(orig, bg, scale=0.34):
    """Full-res binary mask of the art region (largest non-paper component)."""
    H, W = orig.shape[:2]
    sw, sh = int(W * scale), int(H * scale)
    sm = cv2.resize(orig, (sw, sh), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]; val = hsv[:, :, 2]
    prof = PAPER_PROFILE
    wp = _page_whitepoint(hsv, sh, sw) if prof else None
    if prof and wp is not None:
        # ART = not the off-white paper: darker than this page's paper
        # white point (by the global value-drop) OR more saturated than paper.
        m = (((val.astype(int) < (wp - prof['dv'])) | (sat > prof['s_max']))).astype(np.uint8)
    else:
        m = (((g.astype(int) < (bg - 14)) | (sat > 34))).astype(np.uint8)
    # v70: also treat TEXTURED regions as art. LIGHT-toned art (a pale drawing / faint wash)
    # can sit at or above the paper white-point with low saturation, so the value/sat test
    # above misses it -- yet, unlike smooth paper, it carries fine detail. Without this the
    # art bounding box stops short of the light art (typically along the page BOTTOM) and that
    # art is later whitened / wiped to paper (seen on heavily-inked art pages). Add
    # medium-scale edge-dense regions; smooth paper reads ~0 and page text is removed next.
    _edges = cv2.Canny(g, 40, 120)
    _dens = cv2.blur((_edges > 0).astype(np.float32), (23, 23))
    m = (m | (_dens > 0.055)).astype(np.uint8)
    m[_text_line_mask(sm) > 0] = 0        # v42: text is background, never art
    b = int(0.012 * sw)
    m[:b, :] = 0; m[-b:, :] = 0; m[:, :b] = 0; m[:, -b:] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    # v42: CLOSE 25 -> 13. 25 was large enough to bridge any residual caption ink the
    # text mask missed onto an adjacent plate, so the plate's art component (and thus
    # its crop) stretched sideways into the text column and pasted a grey strip over it.
    # 13 still solidifies a plate interior but no longer laps onto neighbouring text.
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        areas = st[1:, cv2.CC_STAT_AREA]
        main = 1 + int(np.argmax(areas)); amax = float(areas.max())
        keep = np.zeros(n, bool); keep[main] = True
        # v44: REUNITE parts of ONE plate. A light ceiling / sky / transition band can
        # drop the art mask below threshold, splitting a strip (usually the TOP) into its
        # own component. Taking only the largest component dropped that strip -> it was
        # never cropped or erased and survived as a separate, slightly-wider rectangle on
        # the page (seen across a run of text-heavy pages).
        # A component is merged in when it OVERLAPS the current art region on one axis and
        # is only a small gap away on the other AND the connecting strip actually holds
        # art (mean>0.18) -- so a real plate is reunited but a paper-separated neighbour
        # (empty strip) is left alone. Iterated so a chain of bands reconnects.
        changed = True
        while changed:
            changed = False
            idx = [i for i in range(1, n) if keep[i]]
            rx0 = min(st[i, 0] for i in idx); ry0 = min(st[i, 1] for i in idx)
            rx1 = max(st[i, 0] + st[i, 2] for i in idx); ry1 = max(st[i, 1] + st[i, 3] for i in idx)
            for i in range(1, n):
                if keep[i] or st[i, 4] < 0.03 * amax:
                    continue
                x, y, w, h = st[i, 0], st[i, 1], st[i, 2], st[i, 3]
                ox = min(x + w, rx1) - max(x, rx0)
                oy = min(y + h, ry1) - max(y, ry0)
                merged = False
                if ox > 0.25 * min(w, rx1 - rx0):                 # stacked: x-overlap, y-gap
                    if y >= ry1:      strip = m[ry1:y, max(rx0, x):min(rx1, x + w)]; gap = y - ry1
                    elif y + h <= ry0: strip = m[y + h:ry0, max(rx0, x):min(rx1, x + w)]; gap = ry0 - (y + h)
                    else:             strip = None; gap = 0
                    if (strip is not None and strip.size and gap < 0.15 * (ry1 - ry0)
                            and strip.mean() > 0.35 and float(strip.mean(axis=1).min()) > 0.12):
                        merged = True
                if not merged and oy > 0.25 * min(h, ry1 - ry0):  # side-by-side: y-overlap, x-gap
                    if x >= rx1:      strip = m[max(ry0, y):min(ry1, y + h), rx1:x]; gap = x - rx1
                    elif x + w <= rx0: strip = m[max(ry0, y):min(ry1, y + h), x + w:rx0]; gap = rx0 - (x + w)
                    else:             strip = None; gap = 0
                    if (strip is not None and strip.size and gap < 0.15 * (rx1 - rx0)
                            and strip.mean() > 0.35 and float(strip.mean(axis=0).min()) > 0.12):
                        merged = True
                if merged:
                    keep[i] = True; changed = True
        m = np.isin(lab, np.where(keep)[0]).astype(np.uint8)
    return cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)


def _paper_tone(whp):
    H, W = whp.shape[:2]
    b = max(20, int(W * 0.04))
    mk = np.zeros((H, W), bool)
    mk[:b, :] = mk[-b:, :] = mk[:, :b] = mk[:, -b:] = True
    return np.median(whp[mk].reshape(-1, 3), axis=0).astype(np.uint8)


def _bg_texture(img, block=240, paper=True):
    """A full-page background made by mirror-tiling a GENEROUS sampled patch of the
    cleanest background, so erased/empty regions get real paper (or dark) GRAIN
    instead of a flat average colour. The source patch is the most uniform bright
    (paper=True) or dark (paper=False) block on the page, found fast with box
    filters; it is reflected into a seamless 2x2 tile and repeated to page size.

    v35: for paper=True the score now PENALISES saturated regions, so the sampled
    patch is genuinely low-saturation paper -- never a faintly-coloured area whose
    mirror-tiling produced the yellow chevron patches seen around full-bleed art."""
    H, W = img.shape[:2]
    b = int(max(64, min(block, H // 3, W // 3)))
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    m  = cv2.boxFilter(g, -1, (b, b))
    m2 = cv2.boxFilter(g * g, -1, (b, b))
    std = np.sqrt(np.maximum(m2 - m * m, 0.0))
    score = (m if paper else (255.0 - m)) - 3.0 * std     # bright/dark AND uniform
    if paper:
        S = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32)
        score -= 2.5 * cv2.boxFilter(S, -1, (b, b))       # and LOW-SATURATION (true paper)
    h2 = b // 2
    score[:h2, :] = -1e9; score[-h2:, :] = -1e9
    score[:, :h2] = -1e9; score[:, -h2:] = -1e9
    cy, cx = np.unravel_index(int(np.argmax(score)), score.shape)
    y0 = min(max(0, cy - h2), H - b); x0 = min(max(0, cx - h2), W - b)
    src = img[y0:y0 + b, x0:x0 + b]
    top  = np.hstack([src, src[:, ::-1]])
    bot  = np.hstack([src[::-1], src[::-1, ::-1]])
    tile = np.vstack([top, bot])                                # 2b x 2b, seamless
    ny = H // tile.shape[0] + 2; nx = W // tile.shape[1] + 2
    big = np.tile(tile, (ny, nx, 1))[:H, :W]
    return big


def _paper_white(orig):
    """Robust paper-white colour: the bright, low-saturation pixels of the page
    border ring (so the rotation-gap corners of a full-bleed page are filled with
    real margin paper, ignoring any art that bleeds to the edge)."""
    H, W = orig.shape[:2]
    b = max(12, int(0.035 * min(H, W)))
    ring = np.vstack([orig[:b].reshape(-1, 3), orig[-b:].reshape(-1, 3),
                      orig[:, :b].reshape(-1, 3), orig[:, -b:].reshape(-1, 3)])
    hsv = cv2.cvtColor(ring.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    pap = ring[(hsv[:, 1] < 40) & (hsv[:, 2] > 200)]
    if len(pap) > 100:
        return np.median(pap, axis=0).astype(np.uint8)
    return np.array([243, 243, 243], np.uint8)


def _text_furniture_keep(whp_text, keep_pic):
    """Mask (filled bounding boxes) of genuine PAGE TEXT FURNITURE -- credit columns,
    section headers, page/plate numbers, the 'editorial/comics' sub-title -- detected
    on the whitened original OUTSIDE the placed art. Passed to _clean_background as
    extra `keep`, so the background wipe NEVER erases text (which the lossy text-restore
    then can't fully rebuild -- the source of the eroded/ghosted credit column and the
    faded/hollow bold header). Bounding boxes are FILLED, so even an outline/embossed
    header whose interior is paper-bright is protected whole, not reduced to strokes.
    Only text-shaped clusters are kept; large blocky colour masses (art the box missed)
    are left for _clean_background's own art-keep logic, so artwork is untouched."""
    H, W = whp_text.shape[:2]
    L = cv2.cvtColor(whp_text, cv2.COLOR_BGR2GRAY).astype(np.int16)
    S = cv2.cvtColor(whp_text, cv2.COLOR_BGR2HSV)[:, :, 1]
    paperlvl = cv2.blur(L.astype(np.uint8), (61, 61)).astype(np.int16)
    ink = (L < paperlvl - 20) & (S < 90)
    ink &= ~keep_pic
    ink_u = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    # merge glyphs -> word/line/block clusters (horizontal merge dominant, like text)
    merged = cv2.dilate(ink_u, cv2.getStructuringElement(cv2.MORPH_RECT, (27, 13)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(merged, 8)
    keeptxt = np.zeros((H, W), bool)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        dens = ink_u[y:y+h, x:x+w].sum() / max(1, w*h)
        blocky_art = (min(w, h) > 0.12 * min(H, W)) and (a > 0.02 * H * W)
        thin_fringe = (min(w, h) < 6)                       # a lone hairline = frame/fringe, not text
        if a >= 12 and h <= 1100 and w <= 0.62 * W and dens > 0.03 \
                and not blocky_art and not thin_fringe:
            keeptxt[y:y+h, x:x+w] = True                    # FILL the bbox (protect glyph interiors)
    return keeptxt


def _clean_background(whp, paper, keep):
    """Set the page background (outside the kept picture rect(s)) to clean paper, so no
    stray scan shadow / fringe / speckle survives in the margins -- but NEVER erase a
    SUBSTANTIAL non-paper mass (colour OR dark): that is real art (e.g. part of a painting
    the detected box under-covered), and wiping it would punch a hole through the artwork.
    Only paper, thin fringe and small specks are wiped."""
    H, W = whp.shape[:2]
    Lw = cv2.cvtColor(whp, cv2.COLOR_BGR2GRAY)
    Sw = cv2.cvtColor(whp, cv2.COLOR_BGR2HSV)[:, :, 1]
    paperish = (Lw > 205) & (Sw < 45)
    bg = ~keep
    nonpap = ((~paperish) & bg).astype(np.uint8)
    nonpap = cv2.morphologyEx(nonpap, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(nonpap, 8)
    keepart = np.zeros((H, W), bool)
    minside = 0.03 * min(H, W)
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]; w = st[i, cv2.CC_STAT_WIDTH]; h = st[i, cv2.CC_STAT_HEIGHT]
        # real art = substantial AND blocky (a chunk of a painting). A THIN elongated mass
        # (min side small) is a border/gutter frame or a fringe line -> wiped, not kept.
        if a > 0.0012 * H * W and min(w, h) > minside:
            keepart |= lab == i
    wipe = bg & ~keepart
    out = whp.copy()
    out[wipe] = paper[wipe]
    return out


def _art_component_dark(orig, T=30):
    """Mask of the art on a BLACK-ground page: pixels whose colour differs from the
    page-margin colour (sampled from the edge ring) by more than T in Lab. This is
    the dark-polarity analogue of _art_component (which keys on dark-on-light)."""
    H, W = orig.shape[:2]
    lab = cv2.cvtColor(orig, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = max(8, int(min(H, W) * 0.03))
    ring = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                           lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    pg = np.median(ring, axis=0)
    d = (np.linalg.norm(lab - pg, axis=2) > T).astype(np.uint8)
    k = max(9, int(min(H, W) * 0.012))
    d = cv2.morphologyEx(d, cv2.MORPH_OPEN,  np.ones((k, k), np.uint8))
    d = cv2.morphologyEx(d, cv2.MORPH_CLOSE, np.ones((k*5, k*5), np.uint8))
    n, lab2, st, _ = cv2.connectedComponentsWithStats(d, 8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return (lab2 == i).astype(np.uint8)


def _dark_edge_angle(comp):
    """Skew (deg) of a dark-ground art rectangle, fit ROBUSTLY from its 4 border
    edges (first/last art pixel per row/col), each line fit with outlier rejection
    so the irregular interior silhouette doesn't pollute it. Median of the edges
    that fit cleanly; 0.0 if none do (genuinely no straight border to trust)."""
    H, W = comp.shape
    votes = []
    def rfit(idx, pos, n, resid_max):
        if len(idx) < n * 0.4:
            return None
        a, b = np.polyfit(idx, pos, 1)
        for _ in range(3):
            r = pos - (a*idx + b); keep = np.abs(r) < 2*np.std(r) + 2
            if keep.sum() < len(idx) * 0.5:
                break
            a, b = np.polyfit(idx[keep], pos[keep], 1)
        resid = np.median(np.abs(pos - (a*idx + b)))
        return (a, resid) if resid < resid_max else None
    cols = np.where(comp.any(0))[0]; rows = np.where(comp.any(1))[0]
    if len(cols):
        c = cols.astype(float)
        f = rfit(c, comp[:, cols].argmax(0).astype(float), W, H*0.02)
        if f: votes.append(np.degrees(np.arctan(f[0])))
        f = rfit(c, (H-1-comp[::-1, cols].argmax(0)).astype(float), W, H*0.02)
        if f: votes.append(np.degrees(np.arctan(f[0])))
    if len(rows):
        rr = rows.astype(float)
        f = rfit(rr, comp[rows, :].argmax(1).astype(float), H, W*0.02)
        if f: votes.append(-np.degrees(np.arctan(f[0])))
        f = rfit(rr, (W-1-comp[rows, ::-1].argmax(1)).astype(float), H, W*0.02)
        if f: votes.append(-np.degrees(np.arctan(f[0])))
    return float(np.median(votes)) if votes else 0.0


def compose_dark(orig, T=30, margin_frac=0.02):
    """Dewarp the art on a BLACK-ground page the SAME way the light-bg path does --
    fit the art rectangle's border, deskew off it, deep-crop to a crisp axis-aligned
    rectangle -- but KEEP THE BLACK GROUND. Returns (out, theta, n_pieces).

    The original page is kept as the base (its dark ground, header and caption stay
    exactly as scanned); only the art's own footprint is repainted -- first to grain
    SAMPLED FROM THE PAGE'S OWN DARK GROUND, then the deskewed + deep-cropped art is
    placed back on top. No white, no ground replacement, gaps filled with real dark
    grain."""
    H, W = orig.shape[:2]
    comp = _art_component_dark(orig, T)
    if comp is None:
        return orig.copy(), 0.0, 0
    ys, xs = np.where(comp > 0)
    cx, cy = (xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0
    theta = _dark_edge_angle(comp)
    crop, (pcx, pcy) = _rotate_and_crop(orig, comp, theta, margin_frac, cx, cy)
    dark = _bg_texture(orig, paper=False)
    base = orig.copy()
    fp = cv2.dilate(comp, np.ones((25, 25), np.uint8)).astype(bool)
    base[fp] = dark[fp]
    ch, cw = crop.shape[:2]
    px = int(round(pcx - cw/2)); py = int(round(pcy - ch/2))
    px = max(0, min(px, W - cw)); py = max(0, min(py, H - ch))
    base[py:py+ch, px:px+cw] = crop
    return base, theta, 1


def _rotate_and_crop(orig, comp, theta, margin_frac, cx, cy):
    """Rotate orig+comp by theta about (cx,cy), return (crop, (pcx,pcy))."""
    H, W = orig.shape[:2]
    M = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)
    rot = cv2.warpAffine(orig, M, (W, H), flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_REPLICATE)
    rotc = cv2.warpAffine(comp, M, (W, H), flags=cv2.INTER_NEAREST)
    ys, xs = np.where(rotc > 0)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    mx = int((x1 - x0) * margin_frac); my = int((y1 - y0) * margin_frac)
    x0 += mx; y0 += my; x1 -= mx; y1 -= my
    x0 = max(0, x0); y0 = max(0, y0); x1 = min(W, x1); y1 = min(H, y1)
    crop = rot[y0:y1, x0:x1].copy()
    return crop, ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _fit_edge(idx, pos):
    """Robust line fit pos ~ a*idx + b. Returns (slope_a, residual_std, n_kept)."""
    idx = np.asarray(idx, float); pos = np.asarray(pos, float)
    if len(idx) < 40:
        return None
    a, b = np.polyfit(idx, pos, 1)
    keep = np.ones(len(idx), bool)
    for _ in range(4):
        r = pos - (a * idx + b)
        keep = np.abs(r) < 2.0 * np.std(r) + 3
        if keep.sum() < 40:
            break
        a, b = np.polyfit(idx[keep], pos[keep], 1)
    return a, float(np.std((pos - (a * idx + b))[keep])), int(keep.sum())


def _fit_gap(idx, pos):
    """Robust line fit pos=a*idx+b. Returns (a, resid_std, n_inliers) or None."""
    idx = np.asarray(idx, float); pos = np.asarray(pos, float)
    if len(idx) < 30:
        return None
    a, b = np.polyfit(idx, pos, 1)
    for _ in range(3):
        r = pos - (a * idx + b); k = np.abs(r) < 2 * np.std(r) + 2
        if k.sum() < 25:
            break
        a, b = np.polyfit(idx[k], pos[k], 1)
    return a, float(np.std((pos - (a * idx + b))[k])), int(k.sum())


def residual_tilt(crop, resid_thr=6.0, min_angle=0.3, max_angle=5.0,
                  gap_thr=6, min_gap_frac=0.15):
    """Measure the residual rotation of an already-deskewed art crop FROM THE
    TRIANGLE (sliver) it leaves.

    Pass 1 deskew can leave a small tilt where the art's true (printed) edge was
    too soft for the first angle estimate. That tilt shows up as a triangular wedge
    of page along one straight crop edge. The wedge IS the evidence of the warp, so
    we measure the tilt from it: along each edge, the art boundary is CLAMPED to the
    crop on the tight side (reads flat) and pulls away from it over the GAP (the
    sliver) on the tilted side. Fitting a line to ONLY the gap rows recovers the
    true edge angle -- which a whole-edge fit washes out, because the clamped flat
    rows dominate (the bug that made v6 miss 027/031/035).

    Returns the degrees to ADD to the pass-1 angle so the tilted edge becomes axis-
    aligned and the art fills the rectangle (no triangle). 0.0 if no clean tilted
    gap is found (then the edge is genuinely straight -- a soft band, handled by
    remove_edge_slivers, not a warp)."""
    H, W = crop.shape[:2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    val = hsv[:, :, 2].astype(int); sat = hsv[:, :, 1].astype(int)
    ring = np.concatenate([hsv[:8, :, 2].ravel(), hsv[:, :8, 2].ravel(),
                           hsv[:, -8:, 2].ravel()])
    wp = int(np.percentile(ring, 80))
    # SOLID art only (saturated OR clearly dark) -- excludes page and grey band.
    solid = (((val < wp - 95) | (sat > 66))).astype(np.uint8)
    solid = cv2.morphologyEx(solid, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    nn, lab, st, _ = cv2.connectedComponentsWithStats(solid, 8)
    if nn <= 1:
        return 0.0
    m = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])); art = (lab == m)
    x0, y0, w, h, _ = st[m]
    cand = []
    for side in ('L', 'R'):                         # boundary x per row
        ys = []; xs = []
        for y in range(y0, y0 + h):
            c = np.where(art[y])[0]
            if len(c):
                ys.append(y); xs.append(c.min() if side == 'L' else c.max())
        ys = np.array(ys); xs = np.array(xs)
        if len(xs) < H * 0.3:
            continue
        clamp = np.percentile(xs, 5 if side == 'L' else 95)
        gap = (xs > clamp + gap_thr) if side == 'L' else (xs < clamp - gap_thr)
        if gap.sum() < H * min_gap_frac:
            continue
        f = _fit_gap(ys[gap], xs[gap])
        if f and f[1] < resid_thr:
            cand.append(-np.degrees(np.arctan(f[0])))
    for side in ('T', 'B'):                          # boundary y per col
        xs = []; ys = []
        for x in range(x0, x0 + w):
            c = np.where(art[:, x])[0]
            if len(c):
                xs.append(x); ys.append(c.min() if side == 'T' else c.max())
        xs = np.array(xs); ys = np.array(ys)
        if len(ys) < W * 0.3:
            continue
        clamp = np.percentile(ys, 5 if side == 'T' else 95)
        gap = (ys > clamp + gap_thr) if side == 'T' else (ys < clamp - gap_thr)
        if gap.sum() < W * min_gap_frac:
            continue
        f = _fit_gap(xs[gap], ys[gap])
        if f and f[1] < resid_thr:
            cand.append(np.degrees(np.arctan(f[0])))
    cand = [c for c in cand if abs(c) > min_angle]
    if not cand:
        return 0.0
    return float(np.clip(max(cand, key=abs), -max_angle, max_angle))


def _art_quad(crop):
    """Find the art's actual 4 corners (TL,TR,BR,BL) and how well it fills that
    quad. Used to detect off-white CORNER TRIANGLES (a corner pulled in from the
    bounding rectangle = residual keystone/perspective warp, not a rotation)."""
    H, W = crop.shape[:2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    val = hsv[:, :, 2].astype(int); sat = hsv[:, :, 1].astype(int)
    ring = np.concatenate([val[:8].ravel(), val[-8:].ravel(),
                           val[:, :8].ravel(), val[:, -8:].ravel()])
    wp = int(np.percentile(ring, 80))
    solid = (((val < wp - 60) | (sat > 50))).astype(np.uint8)
    solid = cv2.morphologyEx(solid, cv2.MORPH_OPEN,  np.ones((7, 7), np.uint8))
    solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8))
    nn, lab, st, _ = cv2.connectedComponentsWithStats(solid, 8)
    if nn <= 1:
        return None
    m = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    cnt, _ = cv2.findContours((lab == m).astype(np.uint8),
                              cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnt, key=cv2.contourArea)
    hull = cv2.convexHull(c).reshape(-1, 2).astype(np.float32)
    s = hull.sum(1); d = hull[:, 0] - hull[:, 1]
    q = np.array([hull[np.argmin(s)], hull[np.argmax(d)],
                  hull[np.argmax(s)], hull[np.argmin(d)]], np.float32)  # TL TR BR BL
    fill = cv2.contourArea(c) / (cv2.contourArea(q) + 1e-9)
    return q, fill


def close_triangles(crop, lo=0.012, hi=0.060, min_fill=0.86, min_span=0.88):
    """FINAL dewarp pass. After deskew + rotation re-dewarp, a plate can still show
    off-white CORNER TRIANGLES because the art is a keystoned quadrilateral (one
    edge tilted relative to its opposite -- e.g. a top edge that slopes while the
    bottom is level), which no rotation can fix. This finds the art's 4 corners and
    perspective-warps that quad to its bounding RECTANGLE, so the triangles fill
    with art. Handles top, side, or corner warps in one step.

    Gated to stay safe: only warps when one plate DOMINATES the crop (span>=0.88,
    v46: RAISED from 0.55 -- at 0.55 a mis-detected sub-quad could warp away up to
    45%% of the plate, cropping the top off a painting and leaving that strip stranded
    on the page, which is the 'separated top rectangle' users hit on p016-019 etc.),
    the art fills its quad (fill>=0.86, so an L-shaped/garbage component is skipped),
    and the corner pull is a real-but-plausible 1.2%%-6%% of the plate."""
    r = _art_quad(crop)
    if r is None:
        return crop
    q, fill = r
    H, W = crop.shape[:2]
    x0, y0 = q[:, 0].min(), q[:, 1].min(); x1, y1 = q[:, 0].max(), q[:, 1].max()
    if min((x1 - x0) / W, (y1 - y0) / H) < min_span or fill < min_fill:
        return crop
    rect = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
    plate = max(x1 - x0, y1 - y0)
    dev = float(np.hypot(*(q - rect).T).max()) / max(plate, 1)
    if dev < lo or dev > hi:
        return crop
    Wd = int(round(max(np.hypot(*(q[1] - q[0])), np.hypot(*(q[2] - q[3])))))
    Hd = int(round(max(np.hypot(*(q[3] - q[0])), np.hypot(*(q[2] - q[1])))))
    # v46: never let the warp SHRINK the plate materially -- a real keystone fix keeps
    # essentially the whole plate (it only widens a short edge). If the target rectangle
    # is much smaller than the crop, the quad was mis-detected: leave the crop untouched.
    if Wd < 0.90 * W or Hd < 0.90 * H:
        return crop
    M = cv2.getPerspectiveTransform(
        q, np.array([[0, 0], [Wd - 1, 0], [Wd - 1, Hd - 1], [0, Hd - 1]], np.float32))
    return cv2.warpPerspective(crop, M, (Wd, Hd), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


# When True, deskew_deep_crop runs a residual-tilt correction (second pass) so a
# small under-rotation left by the first pass -- the cause of the page-vs-art
# 'triangle' wedge -- is removed. See residual_tilt().
REDEWARP = True


# Inward crop inset ('inline crop'). Default 0.015 = crop a hair INTO the art for a
# crisp edge; reconstruct sets this per page (0 = keep the full detected art extent).
CROP_MARGIN_FRAC = 0.015


def deskew_deep_crop(orig, bg, margin_frac=None):
    """Return (crop_bgr, (pcx,pcy), theta_deg, comp_mask). Deskewed + inset.

    Two-pass: pass 1 deskews by the reliable border angle; pass 2 measures any
    residual tilt left on the cropped art (from its strong-colour edge) and, if
    real, re-deskews the original by (theta1 + residual) -- removing the triangular
    page-vs-art wedge that a small under-rotation leaves along a straight edge."""
    if margin_frac is None:
        margin_frac = CROP_MARGIN_FRAC
    H, W = orig.shape[:2]
    comp = _art_component(orig, bg)
    ys, xs = np.where(comp > 0)
    box = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min()))
    cx, cy = box[0] + box[2] / 2.0, box[1] + box[3] / 2.0

    theta = _reliable_angle(orig, box, bg)
    crop, center = _rotate_and_crop(orig, comp, theta, margin_frac, cx, cy)

    if REDEWARP:
        # The triangle reveals a residual rotation. Measuring it from the gap can
        # overestimate (a slightly bowed edge reads steeper over the gap), so we
        # converge with a DAMPED fixed-point iteration toward zero residual rather
        # than trusting one measurement, and accept only if it actually converged
        # (rejecting a painting's angled content that never flattens).
        if abs(residual_tilt(crop)) > 0.4:
            total = 0.0; cc, cen = crop, center; converged = False
            for _ in range(5):
                res = residual_tilt(cc)
                if abs(res) < 0.35:
                    converged = True; break
                total = float(np.clip(total + 0.5 * res, -5.0, 5.0))
                cc, cen = _rotate_and_crop(orig, comp, theta + total, margin_frac, cx, cy)
            if converged and abs(total) > 0.25:
                crop, center, theta = cc, cen, theta + total

    # FINAL measure: look for off-white CORNER TRIANGLES (a keystone the rotation
    # can't fix) and perspective-warp the art quad to a rectangle to close them.
    crop = close_triangles(crop)
    crop = _trim_dirty_edges(crop)

    return crop, center, theta, comp


def _trim_dirty_edges(crop, cap_frac=0.035):
    """Trim a DIRTY outer edge off a cropped plate so it reads CLEAN: a thin dark
    scan-gutter / torn-page / printed-keyline strip, or a paper overshoot, sitting at
    the very edge. An edge line is removed when it is a near-solid DARK strip (a torn
    page edge: its 75th-pct brightness is very low) while the art just inside is not, OR
    much darker than just-inside, OR near-paper overshoot. Capped, and a genuinely dark
    ARTWORK edge (dark AND similar to the inside) is never trimmed."""
    h, w = crop.shape[:2]
    if h < 40 or w < 40:
        return crop
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    capv = max(1, int(cap_frac * h)); caph = max(1, int(cap_frac * w))

    def dirty(line, inside):
        m = float(np.median(line)); p75 = float(np.percentile(line, 75))
        mi = float(np.median(inside)); pi75 = float(np.percentile(inside, 75))
        torn = p75 < 55 and pi75 > p75 + 25       # near-solid dark edge, lighter just inside
        darkline = m < mi - 40                      # much darker than inside (thin gutter line)
        overshoot = m > 236                         # paper sliver
        return torn or darkline or overshoot

    off = max(2, capv // 2)
    t = 0
    while t < capv and dirty(g[t, :], g[min(h - 1, t + off), :]):
        t += 1
    b = 0
    while b < capv and dirty(g[h - 1 - b, :], g[max(0, h - 1 - b - off), :]):
        b += 1
    offc = max(2, caph // 2)
    l = 0
    while l < caph and dirty(g[:, l], g[:, min(w - 1, l + offc)]):
        l += 1
    r = 0
    while r < caph and dirty(g[:, w - 1 - r], g[:, max(0, w - 1 - r - offc)]):
        r += 1
    if t + b >= h or l + r >= w:
        return crop
    return crop[t:h - b, l:w - r]


def _nonpaper_mask(orig, scale=0.34):
    """Full-res mask of non-paper (art) pixels, using the paper profile."""
    H, W = orig.shape[:2]
    sm = cv2.resize(orig, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV)
    prof = PAPER_PROFILE
    if prof is not None:
        wp = _page_whitepoint(hsv, *sm.shape[:2])
        if wp is not None:
            m = ((hsv[:, :, 2].astype(int) < (wp - prof['dv'])) | (hsv[:, :, 1] > prof['s_max']))
            return cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    g = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY)
    m = ((g.astype(int) < (np.median(g) - 14)) | (hsv[:, :, 1] > 34))
    return cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0


def _restore_page_text(out, whp_text, art_rects):
    """Repaint the faithful scanned PAGE TEXT (dark ink on bright paper) back onto the
    composed page, wherever the composed page is currently PAPER-BRIGHT (i.e. no real
    art occupies that spot). Text on paper is background and must never be lost to an
    over-detected art region -- this restores the credit column, header and page
    numbers even when art_pictures over-extends across them, while never painting over
    actual placed art (those spots are not paper-bright). Text-sized dark components
    only; glyphs copied verbatim from the whitened original -- nothing re-rendered,
    thickened or invented. (art_rects kept for signature compatibility; unused.)
    """
    H, W = out.shape[:2]
    L = cv2.cvtColor(whp_text, cv2.COLOR_BGR2GRAY).astype(np.int16)
    paperlvl = cv2.blur(L.astype(np.uint8), (61, 61)).astype(np.int16)
    darker = (L < paperlvl - 35)                          # locally dark = text/furniture
    sat = cv2.cvtColor(whp_text, cv2.COLOR_BGR2HSV)[:, :, 1]
    lowsat = sat < 60                                     # black ink, NOT coloured art
    out_paper = cv2.blur(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), (9, 9)) > 168
    m = darker & lowsat & out_paper                       # dark text now sitting on erased paper
    # Restore TEXT-SIZED components ONLY (glyphs, words, page/plate numbers). A large or
    # long connected dark mass is NOT text -- it is a torn-page edge / art fringe that the
    # erase exposed; restoring it would re-stamp a ragged dark border (the residue). Filter
    # by component size so only real text comes back.
    mu = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(mu, 8)
    keep = np.zeros((H, W), bool)
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]; w = st[i, cv2.CC_STAT_WIDTH]; h = st[i, cv2.CC_STAT_HEIGHT]
        if 4 <= a <= 9000 and h <= 75 and w <= 1200:      # glyph / word / number, not an edge mass
            keep[lab == i] = True
    out[keep] = whp_text[keep]
    return out


def _reclaim_credit_blocks(out, whp_text):
    """Reclaim a CREDIT TEXT BLOCK as clean background. On institutional/index pages
    art_pictures merges the credit column into one big art panel, so that column is
    pasted back as a DESKEWED art crop -- an off-white strip (never whitened) with the
    scanned text baked in, which then doubles against the restored text and shows a grey
    seam. A credit block is a dense, tall cluster of small GRAYSCALE text (text-density
    ~0.14, vs ~0.02 for stray captions over art); art -- even pale art -- never forms one
    (it lives in a colourful neighbourhood / has no dense small-text rows). Where such a
    block exists, overwrite its box with the WHITENED ORIGINAL (clean paper + the single
    faithful text), removing the false-art strip AND the doubling, then lightly unsharp
    the reclaimed text so the fuzzy edges read cleaner. No re-render, no OCR; pages with
    no credit block (ordinary plates) are left untouched.
    """
    H, W = out.shape[:2]
    L = cv2.cvtColor(whp_text, cv2.COLOR_BGR2GRAY)
    S = cv2.cvtColor(whp_text, cv2.COLOR_BGR2HSV)[:, :, 1]
    pl = cv2.blur(L, (61, 61))
    ink = ((L.astype(np.int16) < pl.astype(np.int16) - 30) & (pl > 198) & (S < 55)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    # v44: vectorised (was a per-component Python loop with a full-image slice each ->
    # 12-30s/page, an effective hang on faint pages). Identical: small = glyph-sized ink.
    a = st[:, cv2.CC_STAT_AREA]; w = st[:, cv2.CC_STAT_WIDTH]; h = st[:, cv2.CC_STAT_HEIGHT]
    ok = (a >= 5) & (a <= 8000) & (h <= 70) & (w <= 900); ok[0] = False
    small = ok[lab].astype(np.uint8)
    cluster = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_RECT, (71, 51)))
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(cluster, 8)
    blocks = []
    for i in range(1, n2):
        x, y, w, h, a = st2[i]
        dens = small[y:y+h, x:x+w].sum() / max(1, a)
        if a > 0.015 * H * W and dens > 0.06 and h > 0.12 * H:     # dense tall small-text = credits
            blocks.append((x, y, x + w, y + h))
    if not blocks:
        return out
    paper = cv2.dilate(L, np.ones((35, 35), np.uint8)); Sb = cv2.blur(S, (41, 41))
    region = np.zeros((H, W), bool)
    for (x0, y0, x1, y1) in blocks:
        mx = int(0.018 * W); my = int(0.010 * H)
        region[max(0, y0-my):min(H, y1+my), max(0, x0-mx):min(W, x1+mx)] = True
    m = region & (paper > 212) & (Sb < 45)
    out[m] = whp_text[m]
    sharp = cv2.addWeighted(out, 1.5, cv2.GaussianBlur(out, (0, 0), 1.4), -0.5, 0)
    out[m] = sharp[m]
    return out


def _banner_ink(patch):
    """Banner ink: dark OR green (box outline, underline, dark colour block, letters).
    The dark colour block (e.g. an "EDITORIAL" plate, dark with orange letters) reads
    dark, so the whole block is captured and its letter holes are filled later."""
    L = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(patch.astype(int))
    return ((L < 150) | ((g > r + 12) & (g > b + 12)))


def _fill_holes(mask):
    """Fill interior holes of a binary mask (so the letters/white inside a banner
    box or colour block become part of the solid banner footprint). cv2-only."""
    m = mask.astype(np.uint8)
    h, w = m.shape
    ff = m.copy()
    z = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, z, (0, 0), 1)                     # flood the outside background
    holes = ff == 0                                     # unreached background == interior holes
    return (m.astype(bool) | holes)


def _banner_blob(patch, seed_xy):
    """Solid footprint of the ONE banner mass containing seed_xy. A generous close
    bridges letters to the box/underline into a single component (so a banner is never
    split into "underline straight / letters tilted"); credit lines below are a SEPARATE
    component (vertical gap > the close height) and are excluded; holes are filled so the
    colour block / box interior are covered. Returns a bool mask the size of patch."""
    ink = _banner_ink(patch).astype(np.uint8)
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((15, 51), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    sx, sy = int(seed_xy[0]), int(seed_xy[1])
    sx = max(0, min(sx, patch.shape[1] - 1)); sy = max(0, min(sy, patch.shape[0] - 1))
    lid = lab[sy, sx]
    if lid == 0:                                        # seed landed off ink -> nearest ink pixel
        ys, xs = np.where(ink > 0)
        if len(xs) == 0:
            return np.zeros(ink.shape, bool)
        j = int(((xs - sx) ** 2 + (ys - sy) ** 2).argmin()); lid = lab[ys[j], xs[j]]
    return _fill_holes(lab == lid)


def _place_logo_straight(out, whp_text):
    """Section-header / award LOGO banner in the top band: a wide BOXED or underlined
    banner ("SPECTRUM 4 BOOKS", "INSTITUTIONAL", "GOLD AWARD EDITORIAL", ...). It is
    (near-)straight in the source but the page compose tilts it; reclaim the WHOLE banner
    from the whitened original, level it, and drop it back straight, erasing the tilted
    version. Returns out unchanged on pages with no such banner.

    v34 -- WIDE BANNER NO LONGER CUT IN HALF. The old detector blanked everything past
    0.55*W ("logos are top-left, keep art out"), which sliced a wide award banner in two:
    the left half was reclaimed level while the right half kept the tilted composite (the
    "GOLD AWARD | EDITORIAL" seam). Art is now kept out STRUCTURALLY instead: a painting
    reaching into the top band is TALL and fails the short-bar height gate, and the banner
    footprint is grown only along its OWN contiguous row-band, stopping at the wide paper
    gap before any side art -- so the full banner is reclaimed and a real right-side
    artwork is still never touched."""
    H, W = out.shape[:2]
    band = int(0.16 * H)                                # a little taller: holds a tilted wide banner
    sub = whp_text[:band]
    L = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    bb, gg, rr = cv2.split(sub.astype(int))
    mask = (((L < 140) | ((gg > rr + 12) & (gg > bb + 12)))).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 41), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    # Banner seed: the strongest WIDE THIN BAR in the band. "Thin" is measured by the
    # MEDIAN per-column ink span, NOT the bounding-box height -- a wide banner tilted by
    # a few degrees has a tall bbox (box_height + width*tan(angle)) yet a thin per-column
    # span (~the box height), so this is tilt-invariant and admits wide award banners
    # that the old bbox-height gate wrongly rejected. A painting reaching into the band
    # fills it -> large per-column span -> rejected.
    best = None
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if not (w > 0.18 * W and a > 0.0022 * W * H):
            continue
        comp = lab == i
        cols = np.where(comp.any(axis=0))[0]
        spans = []
        for c in cols[::3]:
            rws = np.where(comp[:, c])[0]
            if len(rws):
                spans.append(rws[-1] - rws[0] + 1)
        if not spans:
            continue
        mspan = float(np.median(spans))
        if 0.010 * H < mspan < 0.06 * H and w > 6 * mspan:
            if best is None or a > best[-1]:
                best = (x, y, w, h, a)
    if best is None:
        return out
    x, y, w, h, a = best
    y0, y1 = y, y + h
    # TRUE horizontal extent: from a column known to be inside the banner (just inside
    # its left edge), walk its own row-band left and right along columns carrying banner
    # ink, stopping after a sustained run of paper columns. This recovers the full width
    # of a wide banner yet never jumps the wide paper gap into a side painting (so the
    # seed merging with distant art, if it ever happened, can't widen the result either).
    bandL = cv2.cvtColor(whp_text[max(0, y0 - 2):min(H, y1 + 2)], cv2.COLOR_BGR2GRAY)
    is_col = (bandL < 150).mean(axis=0) > 0.06         # banner cols carry the underline+box; paper ~0
    gap_cap = int(0.02 * W)
    cseed = min(W - 1, x + max(3, min(w // 4, int(0.05 * W))))
    left = cseed; g = 0; c = cseed
    while c > 0:
        if is_col[c]:
            left = c; g = 0
        else:
            g += 1
            if g > gap_cap:
                break
        c -= 1
    right = cseed; g = 0; c = cseed
    while c < W:
        if is_col[c]:
            right = c; g = 0
        else:
            g += 1
            if g > gap_cap:
                break
        c += 1
    if right - left < 0.15 * W:                         # not a real banner span
        return out
    # Banner top-edge tilt, fit robustly across the FULL width.
    cs, tops = [], []
    for c in range(left, right, 3):
        col = np.where(L[max(0, y0 - 10):y1 + 10, c] < 120)[0]
        if len(col):
            cs.append(c); tops.append(col.min())
    if len(cs) < 20:
        return out
    cs = np.array(cs); tops = np.array(tops); m, b0 = np.polyfit(cs, tops, 1)
    for _ in range(3):
        r = tops - (m * cs + b0); k = np.abs(r) < 2 * np.std(r) + 2
        m, b0 = np.polyfit(cs[k], tops[k], 1)
    src_ang = float(np.degrees(np.arctan(m)))
    # ANTI-ART GATES (v37): a printed banner has a CLEAN STRAIGHT top edge and a UNIFORM
    # thin height across its width. Art (a dark painting mass / a demon / sketchy strokes
    # reaching into the top band) has a ragged top edge and wildly varying height, so it
    # used to be mis-read as a tilted banner and ERASED -- punching a white rectangle into
    # the art (p043/p044). Reject anything whose top edge is not a clean line, or whose
    # bar height is not consistent.
    resid = float(np.std((tops - (m * cs + b0))[k]))
    if resid > 0.006 * H:                               # ragged top edge -> not a banner
        return out
    ext = []
    for c in range(left, right, 5):
        rws = np.where(L[max(0, y0 - 4):y1 + 4, c] < 140)[0]
        if len(rws):
            ext.append(rws[-1] - rws[0] + 1)
    if len(ext) < 10:
        return out
    ext = np.array(ext, float)
    if np.median(ext) <= 0 or (np.std(ext) / np.median(ext)) > 0.7:   # irregular height -> art
        return out
    # Reclaim region: the detected bbox already spans the banner's tilt swing, so only a
    # small uniform margin is added. Levelling REDUCES the vertical extent, so the levelled
    # banner can never clip. The banner footprint (a connected blob) is what gets erased /
    # repainted, so neighbouring credit text below stays untouched even if the box is tall.
    mx = int(0.012 * W); my = int(0.012 * H)
    rx0, ry0 = max(0, left - mx), max(0, y0 - my)
    rx1, ry1 = min(W, right + mx), min(H, y1 + my)
    P = whp_text[ry0:ry1, rx0:rx1].copy()
    seed = (cseed - rx0, (y0 + y1) // 2 - ry0)
    oldmask = _banner_blob(P, seed)                     # tilted banner footprint (== out's)
    if oldmask.sum() < 50:
        return out
    if abs(src_ang) > 0.4:                              # level a genuinely tilted banner
        ch, cw = P.shape[:2]
        Mr = cv2.getRotationMatrix2D((cw / 2, ch / 2), src_ang, 1.0)
        Pr = cv2.warpAffine(P, Mr, (cw, ch), flags=cv2.INTER_CUBIC, borderValue=(235, 235, 235))
        newmask = cv2.warpAffine(oldmask.astype(np.uint8), Mr, (cw, ch),
                                 flags=cv2.INTER_NEAREST) > 0
    else:
        Pr = P; newmask = oldmask
    # Erase the tilted banner to local paper, then drop the levelled banner back. Only
    # banner-footprint pixels are touched, so neighbouring paper/text is left intact.
    reg = out[ry0:ry1, rx0:rx1]
    Lr = cv2.cvtColor(reg, cv2.COLOR_BGR2GRAY)
    pap = reg[Lr > 205]
    paper_col = (np.median(pap, axis=0) if len(pap) > 50 else np.array([242, 242, 242])).astype(np.uint8)
    # erase a slightly DILATED old footprint so the soft anti-aliased fringe of the tilted
    # banner edge is wiped too (otherwise a faint ghost line is left above the level banner)
    erase = cv2.dilate(oldmask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    reg[erase] = paper_col
    reg[newmask] = Pr[newmask]
    out[ry0:ry1, rx0:rx1] = reg
    return out


def _credit_text_zone(out):
    """Mask of the dense small-text CREDIT blocks (so the darkest-point cut is confined
    to credit columns and NEVER touches art -- pale art included, since art forms no
    dense small-text cluster). Same structural detector as _reclaim_credit_blocks."""
    H, W = out.shape[:2]
    L = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY); S = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)[:, :, 1]
    pl = cv2.blur(L, (61, 61))
    ink = ((L.astype(np.int16) < pl.astype(np.int16) - 30) & (pl > 198) & (S < 55)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    small = np.zeros((H, W), np.uint8)
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]; w = st[i, cv2.CC_STAT_WIDTH]; h = st[i, cv2.CC_STAT_HEIGHT]
        if 5 <= a <= 8000 and h <= 70 and w <= 900:
            small[lab == i] = 1
    cl = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_RECT, (71, 51)))
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(cl, 8)
    zone = np.zeros((H, W), bool)
    for i in range(1, n2):
        x, y, w, h, a = st2[i]
        dens = small[y:y+h, x:x+w].sum() / max(1, a)
        if a > 0.015 * H * W and dens > 0.06 and h > 0.12 * H:
            zone[max(0, y-30):min(H, y+h+30), max(0, x-30):min(W, x+w+30)] = True
    return zone


def _despeckle_paper(out):
    """Two-stage text cleaning, both confined so they can NEVER damage art:

    (1) ISOLATED-SMUDGE REMOVAL: remove lone faint grey marks on paper that don't touch
        a glyph and aren't part of a (faint) text row. Marks adjacent to a glyph
        (periods, i/j dots, commas, quotes) and marks that line up into a text row are
        protected; only acts where the local background is paper, never on art.
    (2) DARKEST-POINT CUT (60%, credit text zones only): inside the dense small-text
        credit blocks, find the darkest ink and set every pixel that is >=40% lighter
        (i.e. keep the darkest 60% of the ink->paper range) to clean paper. This wipes
        the residual grey background/haze and crisps the text without thinning it. It is
        restricted to credit blocks via _credit_text_zone, so pale art SURROUNDING the
        text is never cut (art forms no dense small-text cluster).
    """
    H, W = out.shape[:2]
    L = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    pl = cv2.blur(L, (61, 61))
    paper_zone = pl > 200
    # (1) isolated-smudge removal -------------------------------------------------
    ink = ((L.astype(np.int16) < pl.astype(np.int16) - 22) & paper_zone).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    # v44: vectorised (was a Python loop over ~70-90k components -> 20-30s/page and an
    # effective hang on faint pages like p149). Identical result: glyph = pixels whose
    # component area >= 25.
    area = st[:, cv2.CC_STAT_AREA]
    glyph = ((area[lab] >= 25) & (lab > 0)).astype(np.uint8)
    near = cv2.dilate(glyph, np.ones((21, 21), np.uint8))         # ~10px protection
    faint = ((L.astype(np.int16) < pl.astype(np.int16) - 12) & paper_zone).astype(np.uint8)
    horline = cv2.dilate(faint, cv2.getStructuringElement(cv2.MORPH_RECT, (41, 1)))
    nh, lh, sh, _ = cv2.connectedComponentsWithStats(horline, 8)
    width = sh[:, cv2.CC_STAT_WIDTH]
    is_line = ((width[lh] > 70) & (lh > 0)).astype(np.uint8)      # a row of text, not a smudge
    n2, lab2, st2, c2 = cv2.connectedComponentsWithStats(faint, 8)
    med = cv2.medianBlur(out, 7)
    # replace lone faint marks (area<=70) whose CENTROID is in-bounds and neither near a
    # glyph nor on a text line -- same per-centroid test as before, vectorised per label.
    area2 = st2[:, cv2.CC_STAT_AREA]
    cxr = c2[:, 0]; cyr = c2[:, 1]
    inb = (cxr >= 0) & (cxr < W) & (cyr >= 0) & (cyr < H)
    cxs = np.clip(cxr.astype(int), 0, W - 1); cys = np.clip(cyr.astype(int), 0, H - 1)
    excl = (near[cys, cxs] > 0) | (is_line[cys, cxs] > 0)
    keep_lab = (area2 <= 70) & inb & ~excl
    keep_lab[0] = False
    replace = keep_lab[lab2]
    out[replace] = med[replace]
    # (2) darkest-point cut, credit text zones only -------------------------------
    zone = _credit_text_zone(out)
    region = zone & (cv2.blur(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), (61, 61)) > 195)
    if region.sum() > 100:
        Lf = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)
        vals = Lf[region]
        D = float(np.percentile(vals, 2)); Wp = float(np.percentile(vals, 92))
        thr = D + 0.60 * (Wp - D)                                 # keep darkest 60%
        out[region & (Lf > thr)] = [245, 245, 245]
    return out


def _wipe_soft_bands(whp, keep, tmask, fill_img, pwl):
    """v47: remove the faint SCAN-SHADOW / under-whitened GREY BAND that can survive in a
    page MARGIN outside the placed art -- the "separate, slightly-wider strip at the top"
    users flagged as 'choppy' on text-heavy pages
    118/124-127. It is NOT split art (the plate IS detected and placed correctly -- verified
    p018: art top y=401, the strip is a scan-shadow band at y~90-240 in the top MARGIN) and
    NOT a component in the art mask -- it is a soft grey band the flatfield leaves a few
    levels below paper. v45 wiped it with a per-PIXEL near-paper test (Lw>pwl-28 & Sat<30),
    but the band is partly soft ART FRINGE (measured p018: ~48%% of the band is darker than
    that L cut, ~21%% is faintly tinted), so about half of it survived -> the visible ghost
    rectangle. v47 wipes it by CONNECTED COMPONENT instead: form the soft-grey mark mask (a
    little below paper, but NOT solid ink and NOT saturated), then wipe a whole component
    when it is genuinely SOFT, LOW-SAT and non-ink. Protections, so nothing real is erased:
      * `keep` (the placed art crop / pictures) is excluded -- artwork is never touched;
      * `tmask` (page text lines) is excluded -- captions / credits never touched;
      * SOLID INK is outside the mark mask (L <= pwl-90), so page/plate NUMBERS, signatures
        and caption glyphs can never sit inside a wiped component;
      * SATURATED colour is excluded (S >= 45 forming the mask, mean S >= 40 per component),
        so a colour-art remnant a box under-covered is left intact rather than wiped;
      * a wiped component's MEDIAN must be soft grey (pwl-70 < med < pwl-3): a darker mass
        (real dark art / ink) has a lower median and is skipped.
    Fill is the caller's `fill_img` (flat clean paper in compose_fullbleed, the paper
    texture in compose_multi), so the wiped band matches the surrounding margin."""
    H, W = whp.shape[:2]
    out = whp.copy()
    L = cv2.cvtColor(whp, cv2.COLOR_BGR2GRAY).astype(int)
    S = cv2.cvtColor(whp, cv2.COLOR_BGR2HSV)[:, :, 1].astype(int)
    lowsat = S < 40
    mark = ((L < pwl - 3) & (L > pwl - 90) & lowsat
            & (~keep) & (tmask == 0)).astype(np.uint8)
    if int(mark.sum()) == 0:
        return out
    # a SMALL close only bridges the band's own internal specks; it must NOT reach across a
    # plate edge, so re-intersect with the low-sat / keep / text guards afterwards -- the
    # close can otherwise ENGULF adjacent tinted plate pixels (measured on p042: the 9x9
    # close pulled sat~56 ceiling pixels into the grey band and wiped them). After this,
    # a wiped pixel is GUARANTEED low-saturation (S<40) and outside art/text.
    mark = cv2.morphologyEx(mark, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mark[~lowsat] = 0; mark[keep] = 0; mark[tmask > 0] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(mark, 8)
    if n <= 1:
        return out
    # VECTORISE the cheap gates (v44b lesson: never loop median/percentile over ~10^4-10^5
    # grain components -- it stalls, e.g. p039). Mean saturation per label via bincount, and
    # skip anything below a real-band size; only the few LARGE low-sat components then pay
    # for the median / p90 test.
    areas = st[:, cv2.CC_STAT_AREA]
    flat = lab.ravel()
    meanS = np.bincount(flat, weights=S.ravel(), minlength=n) / np.maximum(np.bincount(flat, minlength=n), 1)
    # v70: TEXTURE guard. A real scan-shadow band is SMOOTH (almost no internal edges); a
    # patch of LIGHT-TONED art (a pale drawing / faint wash with detail) is also low-sat and
    # soft-grey and can fall OUTSIDE the detected art crop, so without this it matched the
    # band profile and got wiped -- erasing art, worst at the page bottom where the plate
    # detector under-reaches (seen on heavily-inked art pages). Measure Canny-edge
    # density inside each candidate component's bounding box: a flat band reads ~0, light art
    # reads well above. Components with real internal structure are left intact.
    edges = cv2.Canny(cv2.cvtColor(whp, cv2.COLOR_BGR2GRAY), 40, 120)
    min_band = max(400, int(0.00015 * H * W))         # grain specks (< this) are despeckle's job
    for i in range(1, n):
        if areas[i] < min_band or meanS[i] >= 30:
            continue
        bx, by, bw, bh = st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP], st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        if float((edges[by:by + bh, bx:bx + bw] > 0).mean()) > 0.018:
            continue                                  # textured -> art, never wipe
        sel = lab == i
        Lmed = float(np.median(L[sel])); Sp90 = float(np.percentile(S[sel], 90))
        # wipe only a genuinely SOFT, LOW-SAT, non-ink band. Sp90 rejects a component that
        # still carries a saturated TAIL (grey shadow touching a tinted plate edge) even if
        # its mean is low; the median floor rejects a darker (real dark-art) mass.
        if Sp90 < 45 and (pwl - 60) < Lmed < (pwl - 3):
            out[sel] = fill_img[sel]
    return out


def compose_fullbleed(orig, box, bg, flatfield_whiten, center=False):
    """Whitened upright page (page/plate numbers retained) with the deskewed,
    deliberately-inset art rectangle pasted on top and the full original art
    footprint erased to paper so no washed ghost survives.

    `box` is accepted for signature compatibility but ignored -- the art region
    is detected robustly inside.

    center=True (single-picture pages): the cleaned art is placed at the PAGE
    CENTRE (equal left/right and top/bottom margins) instead of its original
    scanned position, correcting the binding-gutter shift and the empty band that
    art-annual plates leave below the picture. Page furniture (award header,
    caption block, page number) stays where it is in the whitened canvas -- only
    the picture moves -- and the original footprint is still erased to paper, so no
    ghost is left at the old position."""
    H, W = orig.shape[:2]
    crop, (pcx, pcy), theta, comp = deskew_deep_crop(orig, bg)
    whp = flatfield_whiten(orig)
    whp_text = whp.copy()                                # faithful page text, kept for restore
    # CLEAN uniform paper fill (sampled page white). Erased regions read as clean paper,
    # never a grey/grain patch -- the source of the residue the user flagged (v37).
    mh = max(8, int(0.05 * H)); mw = max(8, int(0.04 * W))
    ring = np.concatenate([whp[:mh].reshape(-1, 3), whp[-mh:].reshape(-1, 3),
                           whp[:, :mw].reshape(-1, 3), whp[:, -mw:].reshape(-1, 3)])
    pw = np.percentile(ring, 80, axis=0).astype(np.uint8)
    paper = np.empty_like(whp); paper[:] = pw
    # Erase the original art footprint AND its soft fringe to clean paper (generous dilate),
    # so no dark rim / speckle survives outside where the clean crop will land.
    d = max(15, int(0.012 * min(H, W)))
    fp = cv2.dilate(comp, np.ones((d, d), np.uint8)).astype(bool)
    whp[fp] = paper[fp]
    ch, cw = crop.shape[:2]
    if center:
        px = (W - cw) // 2; py = (H - ch) // 2
    else:
        px = int(round(pcx - cw / 2)); py = int(round(pcy - ch / 2))
    px = max(0, min(px, W - cw)); py = max(0, min(py, H - ch))
    # Paste the clean deskewed crop.
    whp[py:py + ch, px:px + cw] = crop
    # CLEAN BACKGROUND: everything outside the pasted crop becomes clean paper (no stray
    # original-art line / scan-gutter shadow / fringe anywhere in the margins); the page
    # number / caption are restored on top next.
    keep_pic = np.zeros((H, W), bool)
    keep_pic[py:py + ch, px:px + cw] = True
    # v39: do NOT globally wipe the background to flat paper. That wipe erased the
    # credit column / header (page text lives OUTSIDE the art box), and the text-restore
    # could only partially stamp it back -> eroded/ghosted text (p020/p032). The whitened
    # page is already clean paper + faithful text everywhere outside the art; the art rim
    # is removed by the footprint erase above. Keep the page; only reclaim a merged credit
    # strip (when art_pictures swallowed the column) and place logo / despeckle.
    whp = _reclaim_credit_blocks(whp, whp_text)
    whp = _place_logo_straight(whp, whp_text)
    whp = _despeckle_paper(whp)
    # v45: level the faint under-whitened PAPER rectangle that can survive above/around the
    # art (the "separate, slightly-wider strip at the top" flagged on single-art pages e.g.
    # p016/p082 -- paper the flatfield left a few levels grey next to the sharp art edge,
    # made visible by the cleaner erased halo). OUTSIDE the placed crop, low-saturation
    # NEAR-PAPER pixels are set to the clean paper tone. Colour art (sat), ink/dark art
    # (value) and the page TEXT (mask) are far outside this band and untouched.
    pwl = int(np.mean(pw))
    Lw = cv2.cvtColor(whp, cv2.COLOR_BGR2GRAY)
    Sat = cv2.cvtColor(whp, cv2.COLOR_BGR2HSV)[:, :, 1]
    _sm = cv2.resize(whp_text, (int(W * 0.34), int(H * 0.34)), interpolation=cv2.INTER_AREA)
    tmask = cv2.resize(_text_line_mask(_sm), (W, H), interpolation=cv2.INTER_NEAREST)
    tmask = cv2.dilate(tmask, np.ones((9, 9), np.uint8))
    # v47: connected-component soft-band wipe (supersedes the v45 per-pixel faint_bg, which
    # left ~half of the shadow band because it was partly darker/tinted than a paper cut).
    whp = _wipe_soft_bands(whp, keep_pic, tmask, paper, pwl)
    return whp, theta


def _compose_fullbleed_OLD(orig, box, bg, flatfield_whiten, center=False):
    H, W = orig.shape[:2]
    crop, (pcx, pcy), theta, comp = deskew_deep_crop(orig, bg)
    whp = flatfield_whiten(orig)
    whp_text = whp.copy()                                # faithful page text, kept for restore
    if center:
        mh = max(8, int(0.05 * H)); mw = max(8, int(0.04 * W))
        ring = np.concatenate([whp[:mh].reshape(-1, 3), whp[-mh:].reshape(-1, 3),
                               whp[:, :mw].reshape(-1, 3), whp[:, -mw:].reshape(-1, 3)])
        pw = np.percentile(ring, 80, axis=0).astype(np.uint8)   # paper white, ignores margin text
        paper = np.empty_like(whp); paper[:] = pw
    else:
        paper = _bg_texture(whp, paper=True)         # textured paper, not a flat tone
    fp = cv2.dilate(comp, np.ones((15, 15), np.uint8)).astype(bool)
    whp[fp] = paper[fp]
    ch, cw = crop.shape[:2]
    if center:
        px = (W - cw) // 2; py = (H - ch) // 2
    else:
        px = int(round(pcx - cw / 2)); py = int(round(pcy - ch / 2))
    px = max(0, min(px, W - cw)); py = max(0, min(py, H - ch))

    pad = int(0.035 * max(cw, ch))
    fy0, fy1 = max(0, py - pad), min(H, py + ch + pad)
    fx0, fx1 = max(0, px - pad), min(W, px + cw + pad)
    frame = np.zeros((H, W), bool)
    frame[fy0:fy1, fx0:fx1] = True
    frame[py:py + ch, px:px + cw] = False

    nonpaper = _nonpaper_mask(orig).astype(np.uint8)
    nlab, lab, st, _ = cv2.connectedComponentsWithStats(nonpaper, 8)
    keep = np.zeros((H, W), bool)
    if nlab > 1:
        main = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))   # the art mass
        for i in range(1, nlab):
            if i == main:
                continue
            x, y, w_, h_, area = st[i]
            if area < 0.006 * cw * ch and w_ < 0.16 * cw and h_ < 0.05 * ch:
                keep |= lab == i
    Lw = cv2.cvtColor(whp, cv2.COLOR_BGR2GRAY); Sw = cv2.cvtColor(whp, cv2.COLOR_BGR2HSV)[:, :, 1]
    clean_paper = (Lw > 212) & (Sw < 40)
    wipe = frame & ~keep & ~clean_paper
    whp[wipe] = paper[wipe]

    whp[py:py + ch, px:px + cw] = crop
    whp = _restore_page_text(whp, whp_text, [(px, py, px + cw, py + ch)])
    whp = _reclaim_credit_blocks(whp, whp_text)
    whp = _place_logo_straight(whp, whp_text)
    whp = _despeckle_paper(whp)
    return whp, theta


def remove_edge_slivers(page, wp_margin=22, s_lo=55, soft_dv=78, soft_sat=48,
                        solid_sat=66, solid_dv=95, cap_frac=0.025):
    """Remove the off-white PAGE that leaks into an art block along its straight
    deskew edges -- both the bright near-white slivers AND the dull GREY TRANSITION
    band (val well below the page white point, e.g. ~190) that a strict near-white
    test misses.

    Two tiers, both refilled with clean paper texture:
      1. the clean margin = border-connected strict near-white;
      2. the margin is then grown inward into adjacent SOFT pixels (mildly bright,
         low-saturation -- the grey transition slivers), but only within a shallow
         CAP (cap_frac of the short side) and never across SOLID ART (saturated OR
         clearly dark). The cap + the solid-art barrier mean a low-saturation light
         ART region (a cream cloth, a tan plate, a pale sky) is preserved: its body
         is either beyond the cap from the margin or fenced off by solid art, while
         the thin grey edge sliver beside it is absorbed. Interior pale art is never
         border-connected, so it is untouched."""
    import numpy as _np
    H, W = page.shape[:2]
    hsv = cv2.cvtColor(page, cv2.COLOR_BGR2HSV)
    val = hsv[:, :, 2].astype(int); sat = hsv[:, :, 1].astype(int)
    ring = _np.concatenate([val[:12].ravel(), val[-12:].ravel(),
                            val[:, :12].ravel(), val[:, -12:].ravel()])
    wp = int(_np.percentile(ring, 80))

    def _border_cc(mask):
        n, lab = cv2.connectedComponents(mask.astype(_np.uint8), 8)
        bl = set(lab[0, :]).union(lab[-1, :]).union(lab[:, 0]).union(lab[:, -1])
        bl.discard(0)
        if not bl:
            return _np.zeros((H, W), bool)
        return _np.isin(lab, list(bl))

    strict = ((val > wp - wp_margin) & (sat < s_lo)).astype(_np.uint8)
    strict = cv2.morphologyEx(strict, cv2.MORPH_OPEN, _np.ones((3, 3), _np.uint8))
    margin = _border_cc(strict)
    if not margin.any():
        return page
    solid = ((sat > solid_sat) | (val < wp - solid_dv))
    soft = ((val > wp - soft_dv) & (sat < soft_sat) & (~solid))
    cap = max(8, int(cap_frac * min(H, W)))
    near = cv2.dilate(margin.astype(_np.uint8),
                      _np.ones((2 * cap + 1, 2 * cap + 1), _np.uint8)) > 0
    cand = (margin | (near & soft)).astype(_np.uint8)
    fill = _border_cc(cand)
    tex = _bg_texture(page, paper=True)
    out = page.copy(); out[fill] = tex[fill]
    return out


def _merge_boxes(boxes, gut):
    """Union-merge [x0,y0,x1,y1] boxes that overlap or sit within `gut` px."""
    boxes = [list(b) for b in boxes]; changed = True
    while changed:
        changed = False; out = []
        while boxes:
            a = boxes.pop(); merged = True
            while merged:
                merged = False; rest = []
                for b in boxes:
                    if (a[0]-gut < b[2] and b[0]-gut < a[2] and
                        a[1]-gut < b[3] and b[1]-gut < a[3]):
                        a = [min(a[0],b[0]), min(a[1],b[1]),
                             max(a[2],b[2]), max(a[3],b[3])]
                        merged = True; changed = True
                    else:
                        rest.append(b)
                boxes = rest
            out.append(a)
        boxes = out
    return boxes


def _should_merge(a, b, m, art_thresh=0.22):
    """Two near boxes belong to ONE artwork only if the strip BETWEEN them holds
    art. If the gap is page-coloured (art coverage < thresh) they are SEPARATE
    plates and must stay split -- the cause of the merged-plate 'triangle' (a
    second painting that never gets isolated or deskewed)."""
    ox = min(a[2], b[2]) - max(a[0], b[0])
    oy = min(a[3], b[3]) - max(a[1], b[1])
    if ox > 0 and oy > 0:
        # Bounding boxes OVERLAP. For a single split artwork the union adds almost no
        # new area; for two DIAGONALLY-placed SEPARATE plates the union ENGULFS a big
        # new paper/text corner -- and because text is subtracted from the art mask,
        # that corner reads ~0. Merging there stamped a grey text strip with doubled
        # glyphs into the empty corner (v42: a two-column index page, right-column
        # plates vs bottom-left pencil plate overlapping diagonally). Reject the merge
        # when it would swallow a meaningful NEW region that holds no art.
        ux0, uy0 = min(a[0], b[0]), min(a[1], b[1])
        ux1, uy1 = max(a[2], b[2]), max(a[3], b[3])
        new = np.ones((uy1 - uy0, ux1 - ux0), bool)
        new[a[1]-uy0:a[3]-uy0, a[0]-ux0:a[2]-ux0] = False   # not-in-a
        new[b[1]-uy0:b[3]-uy0, b[0]-ux0:b[2]-ux0] = False   # not-in-b
        if new.sum() > 0.05 * new.size:                     # merge engulfs real new area
            sub = m[uy0:uy1, ux0:ux1]
            if float(sub[new].mean()) < 0.06:               # ...and it's empty -> separate plates
                return False
        return True                                  # tightly-fitting / split artwork
    if ox > 0 and oy <= 0:                            # vertical gap, overlap in x
        ylo, yhi = min(a[3], b[3]), max(a[1], b[1])
        xlo, xhi = max(a[0], b[0]), min(a[2], b[2])
        strip = m[ylo:yhi, xlo:xhi]
        return strip.size > 0 and strip.mean() > art_thresh
    if oy > 0 and ox <= 0:                            # horizontal gap, overlap in y
        xlo, xhi = min(a[2], b[2]), max(a[0], b[0])
        ylo, yhi = max(a[1], b[1]), min(a[3], b[3])
        strip = m[ylo:yhi, xlo:xhi]
        return strip.size > 0 and strip.mean() > art_thresh
    return False                                     # diagonal-only -> separate


def _merge_boxes_gapaware(boxes, gut, m):
    """Union-merge [x0,y0,x1,y1] boxes that overlap or sit within `gut` px AND have
    art in the gap between them (see _should_merge). Page-gap-separated plates stay
    split so each is deskewed independently."""
    boxes = [list(b) for b in boxes]; changed = True
    while changed:
        changed = False; i = 0
        while i < len(boxes):
            j = i + 1
            while j < len(boxes):
                a, b = boxes[i], boxes[j]
                near = (a[0] - gut < b[2] and b[0] - gut < a[2] and
                        a[1] - gut < b[3] and b[1] - gut < a[3])
                if near and _should_merge(a, b, m):
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]),
                                max(a[2], b[2]), max(a[3], b[3])]
                    boxes.pop(j); changed = True
                else:
                    j += 1
            i += 1
    return boxes


def art_pictures(orig, bg, scale=0.34, min_area=0.035, min_dim=0.11,
                 min_strong=0.45):
    """Detect distinct art PICTURES on a page. Returns full-res
    [x0,y0,x1,y1] boxes (always >=1). A picture is a sizeable component whose
    pixels are MOSTLY strong art (clearly past the paper envelope:
    sat > s_max+30 OR value < whitepoint-dv-30). That strong-coverage test is
    what separates a real picture (>=~0.7, even a light-background one) from a
    soft paper vignette / shadow blob or a thin page-edge sliver (<=~0.15) that
    merely grazes the paper threshold. Components of one split painting are then
    bbox-merged so a single artwork is never returned as several pictures."""
    H, W = orig.shape[:2]; sw, sh = int(W*scale), int(H*scale)
    sm = cv2.resize(orig, (sw, sh), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(int); val = hsv[:, :, 2].astype(int)
    prof = PAPER_PROFILE
    wp = _page_whitepoint(hsv, sh, sw) if prof else None
    if prof and wp is not None:
        m = (((val < (wp-prof['dv'])) | (sat > prof['s_max']))).astype(np.uint8)
        strong = ((val < (wp-prof['dv']-30)) | (sat > prof['s_max']+30))
    else:
        g = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY).astype(int)
        m = (((g < (bg-14)) | (sat > 34))).astype(np.uint8)
        strong = ((g < (bg-44)) | (sat > 64))
    m[_text_line_mask(sm) > 0] = 0        # v42: text is background, never art
    b = int(0.012*sw); m[:b,:]=0; m[-b:,:]=0; m[:,:b]=0; m[:,-b:]=0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  np.ones((5,5),   np.uint8))
    # v42: CLOSE was 25 -- big enough to bridge the paper GUTTERS between abutting
    # plates (and the text->plate gap) so a 2x2 showcase grid fused into ONE
    # page-wide component and went down the single-plate path (deskewed whole, then
    # text re-stamped on top = the caption ghost). 11 solidifies a plate's interior
    # without leaping a real gutter; _merge_boxes_gapaware below re-joins the pieces
    # of a SINGLE painting (art in the gap) while leaving gutter-separated plates split.
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((11,11), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    pageA = sw*sh; pics = []; allc = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 0.01*pageA:
            continue
        scov = strong[lab == i].mean()
        allc.append((a/pageA, [x, y, x+w, y+h]))
        if (a/pageA >= min_area and w/sw >= min_dim and h/sh >= min_dim
                and scov >= min_strong):
            pics.append([x, y, x+w, y+h])
    if not pics:
        if not allc:
            return [[0, 0, W, H]]
        pics = [max(allc, key=lambda t: t[0])[1]]
    # Merge the pieces of a SINGLE split painting (unchanged: gutter-separated plates
    # stay separate via the art-in-gap test in _should_merge).
    pics = _merge_boxes_gapaware(pics, int(0.03 * sw), m)
    # v44: a light ceiling / sky / transition band can split the TOP (or a side) of a
    # plate into a SUB-GATE component (too short/weak to be its own picture) that used to
    # be dropped -> never cropped or erased -> a separate, slightly-wider leftover strip
    # (seen on some pages). Absorb such a strip into its plate, but SAFELY: an extra is
    # merged ONLY if it connects (art in the gap) to EXACTLY ONE picture. If it would
    # touch two pictures it is left out, so a stray blob can never BRIDGE two separate
    # arts into one. Isolated sub-gate blobs (vignettes/noise) are simply dropped.
    gut = int(0.03 * sw)
    for (af, box) in allc:
        if not (0.012 <= af < min_area):
            continue
        hits = []
        for k, p in enumerate(pics):
            near = (box[0]-gut < p[2] and p[0]-gut < box[2] and
                    box[1]-gut < p[3] and p[1]-gut < box[3])
            if near and _should_merge(box, p, m):
                hits.append(k)
        if len(hits) == 1:                     # belongs to one plate -> absorb
            p = pics[hits[0]]
            pics[hits[0]] = [min(p[0], box[0]), min(p[1], box[1]),
                             max(p[2], box[2]), max(p[3], box[3])]
        # len(hits) == 0 -> isolated blob, drop;  len(hits) >= 2 -> would bridge, skip
    fx, fy = W/float(sw), H/float(sh)
    full = [[int(x0*fx), int(y0*fy), int(x1*fx), int(y1*fy)]
            for x0, y0, x1, y1 in pics]
    full = _split_stacked_plates(orig, full)
    full = _resolve_box_overlaps(orig, full)
    return full


def _split_stacked_plates(orig, boxes, min_h_frac=0.80, min_gutter_frac=0.011):
    """Split a near-full-height box that actually holds TWO vertically-stacked plates
    separated by a PAPER gutter. The permissive batch mask (large dv/s_max) can bridge
    the paper gap between two plates in one column into a single tall component. Left
    merged, that box takes the TOP plate's full width and runs it all the way down the
    page, so it is composited on top of the neighbour below it (the 'grossly enlarged
    art dropped on the neighbour' on stacked-plate pages). Find the bright paper
    band spanning the box interior and cut the box into its two real plates."""
    H, W = orig.shape[:2]
    g = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    white = float(np.percentile(np.concatenate(
        [g[:20].ravel(), g[-20:].ravel(), g[:, :20].ravel(), g[:, -20:].ravel()]), 85))
    out = []
    for (x0, y0, x1, y1) in boxes:
        bh, bw = y1 - y0, x1 - x0
        if bh < min_h_frac * H or bw < 0.12 * W:
            out.append([x0, y0, x1, y1]); continue
        ix0 = x0 + int(0.12 * bw); ix1 = x1 - int(0.12 * bw)
        rowb = g[y0:y1, ix0:ix1].mean(axis=1)
        n = len(rowb); lo, hi = int(0.20 * n), int(0.80 * n)
        paper = np.where(rowb[lo:hi] > (white - 22))[0]
        if paper.size == 0:
            out.append([x0, y0, x1, y1]); continue
        runs = np.split(paper, np.where(np.diff(paper) > 4)[0] + 1)
        band = max(runs, key=len)
        if len(band) < max(30, int(min_gutter_frac * H)):
            out.append([x0, y0, x1, y1]); continue
        cut0 = y0 + lo + int(band[0]); cut1 = y0 + lo + int(band[-1])
        out.append([x0, y0, x1, cut0])
        out.append([x0, cut1, x1, y1])
    return out


def _resolve_box_overlaps(orig, boxes, iters=4):
    """Clip overlapping detected boxes so no plate is composited on top of another.
    For each overlapping pair, cut both boxes back to the PAPER gutter that runs
    through the overlap: a vertical gutter for a side-by-side overlap (the taller
    overlap), a horizontal gutter for a stacked one."""
    H, W = orig.shape[:2]
    g = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    b = [list(x) for x in boxes]
    for _ in range(iters):
        moved = False
        for i in range(len(b)):
            for j in range(i + 1, len(b)):
                a, c = b[i], b[j]
                ox = min(a[2], c[2]) - max(a[0], c[0])
                oy = min(a[3], c[3]) - max(a[1], c[1])
                if ox <= 0 or oy <= 0:
                    continue
                gx0, gx1 = max(a[0], c[0]), min(a[2], c[2])
                gy0, gy1 = max(a[1], c[1]), min(a[3], c[3])
                if gx1 - gx0 < 2 or gy1 - gy0 < 2:
                    continue
                if oy >= ox:                                   # side by side -> vertical gutter
                    col = g[gy0:gy1, gx0:gx1].mean(axis=0)
                    cut = gx0 + int(np.argmax(col))
                    left, right = (a, c) if (a[0] + a[2]) < (c[0] + c[2]) else (c, a)
                    left[2] = min(left[2], cut); right[0] = max(right[0], cut)
                else:                                          # stacked -> horizontal gutter
                    row = g[gy0:gy1, gx0:gx1].mean(axis=1)
                    cut = gy0 + int(np.argmax(row))
                    top, bot = (a, c) if (a[1] + a[3]) < (c[1] + c[3]) else (c, a)
                    top[3] = min(top[3], cut); bot[1] = max(bot[1], cut)
                moved = True
        if not moved:
            break
    return [x for x in b if x[2] - x[0] > 0.05 * W and x[3] - x[1] > 0.05 * H]


def _center_content(page, pic_rects):
    """Rigidly centre the whole content block (pictures + caption columns + headers
    + plate/page numbers + footer) within the page margins, so the composed layout
    sits with even margins instead of being pushed to one side by the scan/binding
    gutter. The block moves as ONE unit -- relative positions unchanged.

    The content bbox is the union of (a) the KNOWN placed-picture rectangles -- exact,
    so a pale sky or light passage inside a picture is never dropped -- and (b) the
    caption/footer text, detected as non-paper OUTSIDE the pictures and OUTSIDE a thin
    page-edge band (so binding-edge streaks and corner flecks don't drag the box to
    the border). The block is cut and re-pasted centred on fresh paper, leaving no
    ghost at the old footprint and clean paper in the freed margins."""
    H, W = page.shape[:2]
    if not pic_rects:
        return page
    px0 = min(r[0] for r in pic_rects); py0 = min(r[1] for r in pic_rects)
    px1 = max(r[2] for r in pic_rects); py1 = max(r[3] for r in pic_rects)
    # caption / footer / header text: non-paper away from the pictures and edges
    g = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(page, cv2.COLOR_BGR2HSV)
    wp = np.percentile(np.concatenate([g[:20].ravel(), g[-20:].ravel(),
                                       g[:, :20].ravel(), g[:, -20:].ravel()]), 80)
    txt = ((hsv[:, :, 1] > 50) | (g.astype(int) < wp - 55)).astype(np.uint8)
    b = int(0.018 * min(H, W))                       # ignore binding-edge band
    txt[:b] = 0; txt[-b:] = 0; txt[:, :b] = 0; txt[:, -b:] = 0
    for (rx0, ry0, rx1, ry1) in pic_rects:           # ignore the pictures themselves
        txt[ry0:ry1, rx0:rx1] = 0
    txt = cv2.morphologyEx(txt, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    col = txt.sum(0); row = txt.sum(1)
    cols = np.where(col > 0.004 * H)[0]; rows = np.where(row > 0.004 * W)[0]
    cx0, cx1, cy0, cy1 = px0, px1, py0, py1
    if cols.size:
        cx0 = min(cx0, int(cols.min())); cx1 = max(cx1, int(cols.max()))
    if rows.size:
        cy0 = min(cy0, int(rows.min())); cy1 = max(cy1, int(rows.max()))
    cx0, cy0 = max(0, cx0), max(0, cy0); cx1, cy1 = min(W, cx1), min(H, cy1)
    content = page[cy0:cy1, cx0:cx1]
    ch, cw = content.shape[:2]
    canvas = _bg_texture(page, paper=True)
    nx0 = max(0, (W - cw) // 2); ny0 = max(0, (H - ch) // 2)
    canvas[ny0:ny0 + ch, nx0:nx0 + cw] = content
    return canvas


def compose_multi(orig, bg, flatfield_whiten, boxes, clip_boxes=False, center_content=False):
    """Multi-picture compositor: whiten the page once (keeping caption columns,
    headers and plate/page numbers), then deskew + deliberately-inset-crop EACH
    detected picture independently and paste it back over its own footprint.

    Each picture is processed by running the single-plate deskew_deep_crop on a
    padded crop around its box, so the per-picture art component, angle and
    inward bite are all local to that artwork. Original skewed rims are erased to
    paper via each picture's dilated footprint; text living between/around the
    pictures is untouched (its footprint is never erased).

    clip_boxes=True (MANUAL/MAGENTA boxes): mask everything OUTSIDE each box (within
    its padded sub-crop) to page background BEFORE deskew_deep_crop, so the art-edge
    detection / deskew / keystone-close see ONLY this picture's art. Without this,
    the 8% outward pad can reach into an adjacent picture, and the corner-fit warp
    then pulls in / stretches the neighbour (seen on one two-plate page: the top-right
    plate's pad overlapped the left plate -> the whole plate was keystone-stretched).
    For auto-detected boxes (art_pictures) leave it False -- those boxes are derived
    from the art itself and the pad is wanted to capture art just outside the bbox."""
    H, W = orig.shape[:2]
    whp = flatfield_whiten(orig)
    whp_text = whp.copy()                                # faithful page text, kept for restore
    # CLEAN uniform paper fill (sampled page white) -- erased regions read as clean paper,
    # never a grey/grain patch (v37 clean-crop).
    mh = max(8, int(0.05 * H)); mw = max(8, int(0.04 * W))
    _ring = np.concatenate([whp[:mh].reshape(-1, 3), whp[-mh:].reshape(-1, 3),
                            whp[:, :mw].reshape(-1, 3), whp[:, -mw:].reshape(-1, 3)])
    pw = np.percentile(_ring, 80, axis=0).astype(np.uint8)
    paper = np.empty_like(whp); paper[:] = pw
    orig_sat = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV)[:, :, 1] if clip_boxes else None
    full_fp = np.zeros((H, W), bool)
    placed = []; thetas = []
    for (x0, y0, x1, y1) in boxes:
        # v42: pad PER AXIS. The old 0.08*max(w,h) used the LONGER side for BOTH axes,
        # so a TALL narrow box (e.g. two plates stacked in one column) got a huge
        # HORIZONTAL pad that reached sideways into the caption column; the crop then
        # overwrote the right half of every text line and pasted its grey left margin
        # over them (a two-column index page). Per-axis pad keeps the reach proportional to each
        # side while still capturing the soft art fringe just outside the box.
        padx = int(0.08 * (x1 - x0)); pady = int(0.08 * (y1 - y0))
        cx0, cy0 = max(0, x0-padx), max(0, y0-pady)
        cx1, cy1 = min(W, x1+padx), min(H, y1+pady)
        sub = orig[cy0:cy1, cx0:cx1]
        if clip_boxes:
            # Confine art detection to THIS box: paint the pad ring (everything in
            # the sub outside the box) to flat page bg so neighbouring pictures that
            # fall inside the pad are seen as background, not art. The box's own art
            # edge (over real page bg) is still found normally and deskewed/cropped.
            sub = sub.copy()
            bx0, by0 = x0 - cx0, y0 - cy0
            bx1, by1 = x1 - cx0, y1 - cy0
            ring = np.ones(sub.shape[:2], bool)
            ring[by0:by1, bx0:bx1] = False
            sub[ring] = bg
        crop, (pcx, pcy), theta, comp = deskew_deep_crop(sub, bg)
        thetas.append(theta)
        if clip_boxes:
            # MANUAL/MAGENTA box is authoritative. Erase the WHOLE box, PLUS any
            # COLOURED-ART bleed in a band just outside it (the plate edge that
            # spilled past a magenta line drawn a hair inside the art), so no thin
            # original-art sliver is stranded in the margin. The bleed test is
            # saturation-gated, so low-sat CAPTION TEXT and paper in the band are
            # preserved; and the erase never enters another picture's box.
            short = min(x1 - x0, y1 - y0)
            mrg  = max(8, int(0.012 * short))        # box gets a small full-erase margin
            band = max(mrg, int(0.06 * short))       # how far out to chase coloured bleed
            region = np.zeros((H, W), bool)
            region[max(0, y0-mrg):min(H, y1+mrg), max(0, x0-mrg):min(W, x1+mrg)] = True
            ring = np.zeros((H, W), bool)
            ring[max(0, y0-band):min(H, y1+band), max(0, x0-band):min(W, x1+band)] = True
            ring[max(0, y0-mrg):min(H, y1+mrg), max(0, x0-mrg):min(W, x1+mrg)] = False
            region |= (ring & (orig_sat > 50))       # coloured art only; text(low-sat) safe
            for (ox0, oy0, ox1, oy1) in boxes:
                if (ox0, oy0, ox1, oy1) != (x0, y0, x1, y1):
                    region[oy0:oy1, ox0:ox1] = False
            full_fp |= region
        else:
            d = max(15, int(0.012 * min(H, W)))          # generous: cover the soft art fringe
            fp = cv2.dilate(comp, np.ones((d, d), np.uint8)).astype(bool)
            full_fp[cy0:cy1, cx0:cx1] |= fp
        placed.append((crop, cx0 + pcx, cy0 + pcy))
    whp[full_fp] = paper[full_fp]
    pic_rects = []
    for (crop, gcx, gcy) in placed:
        ch, cw = crop.shape[:2]
        px = int(round(gcx - cw/2)); py = int(round(gcy - ch/2))
        px = max(0, min(px, W - cw)); py = max(0, min(py, H - ch))
        whp[py:py+ch, px:px+cw] = crop
        pic_rects.append((px, py, px + cw, py + ch))
    # GUARANTEED-CLEAN BORDER around every placed picture: a thin band just outside each
    # crop is set to clean paper unconditionally (but never over another picture), so no
    # original-art speckle/fringe survives touching a plate edge -- the residue ring.
    for (px, py, qx, qy) in pic_rects:
        cw, ch = qx - px, qy - py
        bw = max(6, int(0.015 * max(cw, ch)))
        band = np.zeros((H, W), bool)
        band[max(0, py-bw):min(H, qy+bw), max(0, px-bw):min(W, qx+bw)] = True
        band[py:qy, px:qx] = False
        for (ox, oy, oqx, oqy) in pic_rects:            # don't wipe into another picture
            band[oy:oqy, ox:oqx] = False
        whp[band] = paper[band]
    # CLEAN BACKGROUND: everything that is NOT a placed picture becomes clean paper, so
    # no stray original-art line, scan-gutter shadow or fringe survives anywhere in the
    # margins. The credit column / page numbers are then restored on top from whp_text.
    keep_pics = np.zeros((H, W), bool)
    for (px, py, qx, qy) in pic_rects:
        keep_pics[py:qy, px:qx] = True
    # v39: no global background wipe (see compose_fullbleed). Per-picture footprint erase
    # + the guaranteed-clean border band above already remove every plate's fringe; the
    # rest of the page is the whitened original, so caption columns / headers / numbers
    # stay faithful instead of being wiped and imperfectly restored.
    if center_content:                               # centre the whole placed layout
        whp = _center_content(whp, pic_rects)
    whp = _reclaim_credit_blocks(whp, whp_text)
    whp = _place_logo_straight(whp, whp_text)
    whp = _despeckle_paper(whp)
    # v45: level faint under-whitened PAPER outside the placed pictures (same fix as
    # compose_fullbleed) -- low-saturation near-paper pixels outside the pictures go to the
    # clean paper tone; colour art, ink/dark art and page TEXT are untouched.
    pwl = int(np.mean(paper[0, 0]))
    Lw = cv2.cvtColor(whp, cv2.COLOR_BGR2GRAY)
    Sat = cv2.cvtColor(whp, cv2.COLOR_BGR2HSV)[:, :, 1]
    _sm = cv2.resize(whp_text, (int(W * 0.34), int(H * 0.34)), interpolation=cv2.INTER_AREA)
    tmask = cv2.resize(_text_line_mask(_sm), (W, H), interpolation=cv2.INTER_NEAREST)
    tmask = cv2.dilate(tmask, np.ones((9, 9), np.uint8))
    # v47: connected-component soft-band wipe (supersedes the v45 per-pixel faint_bg). Fills
    # with the paper texture so a wiped margin band matches the surrounding page.
    whp = _wipe_soft_bands(whp, keep_pics, tmask, paper, pwl)
    return whp, thetas

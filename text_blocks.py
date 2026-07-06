"""
text_blocks.py  — paragraph text blocks as first-class, independently-dewarped
regions (Dewarp Pipeline v21).

RULE (per spec): any paragraph of 3+ lines is treated as its own block. It is
detected, DEWARPED on its own (deskewed off its text baselines), and placed back
so it neither overlaps art / another block nor gets cropped. The same rule holds
in BOTH polarities:
    dark_text=False : white text on a dark ground (art-on-black captions, the
                      back-cover blurb, etc.)
    dark_text=True  : black text on pale paper (caption paragraphs beside a plate)

Only the text STROKES are composited (via the foreground mask), never the block's
local background, so nothing extra is stamped onto the canvas. Short captions
(< 3 lines) are left to the caller's bitmap-preserve path; this module owns the
paragraphs.
"""
import cv2
import numpy as np


def fg_mask(gray, dark_text):
    """Polarity-aware local-contrast text foreground. Subtracting a blurred local
    background makes this work white-on-dark as well as black-on-light (only the
    sign flips), where a fixed adaptive-threshold polarity fails on a dark ground."""
    loc = cv2.GaussianBlur(gray, (0, 0), 25).astype(np.int16)
    d = gray.astype(np.int16) - loc
    if dark_text:
        d = -d
    return (d > 18).astype(np.uint8) * 255


def _line_boxes(gray, dark_text):
    """Per-text-line boxes: foreground -> horizontal RLSA close -> line-shaped CCs."""
    H, W = gray.shape
    m = fg_mask(gray, dark_text)
    k = max(15, W // 60)
    cl = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    cl = cv2.morphologyEx(cl, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(cl, 8)
    return [(st[i, 0], st[i, 1], st[i, 2], st[i, 3]) for i in range(1, n)
            if 3 < st[i, 3] < H * 0.05 and st[i, 2] > W * 0.06
            and st[i, 2] / max(st[i, 3], 1) > 4 and st[i, 4] > st[i, 2] * 0.12]


def _regular(member_lines, page_h):
    """True only if a cluster of line boxes really looks like prose: enough lines,
    consistent line HEIGHTS and a regular vertical PITCH. Smoke / hatching / art
    texture can form a few horizontal runs but they are irregular, so this rejects
    them (the saturation guard misses low-sat art; this is the structural guard)."""
    if len(member_lines) < 3:
        return False
    hs = np.array([h for (_x, _y, _w, h) in member_lines], float)
    ys = np.sort(np.array([_y + h / 2.0 for (_x, _y, _w, h) in member_lines]))
    pitch = np.diff(ys)
    if len(pitch) < 2:
        return False
    h_cv = hs.std() / max(hs.mean(), 1e-3)
    p_cv = pitch.std() / max(pitch.mean(), 1e-3)
    med_h = float(np.median(hs))
    # The decisive cue is HEIGHT CONSISTENCY: prose lines share a glyph height
    # (h_cv ~0.1), whereas smoke/hatching runs vary wildly (h_cv >0.5). Pitch is a
    # weak cue (real paragraphs have gaps between them), so it is only a loose
    # backstop; span is not used (a real 3-line caption is as short as a texture
    # fleck). med_h keeps the glyph height plausible.
    return (h_cv < 0.45 and p_cv < 1.0
            and 0.006 * page_h < med_h < 0.05 * page_h)


def _regular_short(member_lines, page_h):
    """Relaxed regularity for SHORT caption/header blocks (1-2 lines), where a
    pitch test is impossible. Keeps only plausible glyph heights and (for 2 lines)
    consistent line heights, so art-texture flecks are still rejected. Used only
    when the short-text deskew pass is enabled (v25)."""
    if not member_lines:
        return False
    hs = np.array([h for (_x, _y, _w, h) in member_lines], float)
    med_h = float(np.median(hs))
    if not (0.006 * page_h < med_h < 0.05 * page_h):
        return False
    if len(member_lines) >= 2:
        if hs.std() / max(hs.mean(), 1e-3) > 0.40:
            return False
    return True


def _group(lines, min_lines=3, page_h=None):
    """Cluster line boxes that share a column (x-overlap) and are vertically
    adjacent into paragraph blocks; keep blocks of >= min_lines lines that pass the
    text-regularity test."""
    if not lines:
        return []
    if page_h is None:
        page_h = max(y + h for (_x, y, _w, h) in lines)
    lines = sorted(lines, key=lambda b: b[1])
    used = [False] * len(lines)
    blocks = []
    for i, (x, y, w, h) in enumerate(lines):
        if used[i]:
            continue
        cur = [i]; used[i] = True
        cx0, cx1, cy1, lh = x, x + w, y + h, h
        changed = True
        while changed:
            changed = False
            for j, (x2, y2, w2, h2) in enumerate(lines):
                if used[j]:
                    continue
                ox = max(0, min(cx1, x2 + w2) - max(cx0, x2))
                xov = ox / max(1, min(cx1 - cx0, w2))
                if xov > 0.55 and -lh * 0.6 < (y2 - cy1) < lh * 1.8:
                    cur.append(j); used[j] = True
                    cx0 = min(cx0, x2); cx1 = max(cx1, x2 + w2)
                    cy1 = max(cy1, y2 + h2); lh = h2; changed = True
        members = [lines[k] for k in cur]
        ok = _regular(members, page_h) if len(cur) >= 3 else _regular_short(members, page_h)
        if len(cur) >= min_lines and ok:
            xs = [m[0] for m in members]; ys = [m[1] for m in members]
            xe = [m[0] + m[2] for m in members]; ye = [m[1] + m[3] for m in members]
            blocks.append((min(xs), min(ys), max(xe) - min(xs),
                           max(ye) - min(ys), len(cur)))
    return blocks


def _deskew_angle(gray, dark_text, rng=6.0, step=0.3):
    """Projection-profile deskew: the rotation whose row-sum profile is sharpest
    (text lines best separated). Robust to ragged / justified margins, unlike a
    left-edge slope fit."""
    m = fg_mask(gray, dark_text).astype(np.float32)
    H, W = m.shape
    best_ang, best = 0.0, -1.0
    for ang in np.arange(-rng, rng + 1e-6, step):
        M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
        r = cv2.warpAffine(m, M, (W, H), flags=cv2.INTER_NEAREST)
        score = float(np.var(r.sum(axis=1)))
        if score > best:
            best_ang, best = float(ang), score
    return best_ang


def _overlaps(a, boxes, frac=0.15):
    """True if box a overlaps any box in `boxes` by > frac of a's area."""
    ax, ay, aw, ah = a
    aA = max(1, aw * ah)
    for (x, y, w, h) in boxes:
        ox = max(0, min(ax + aw, x + w) - max(ax, x))
        oy = max(0, min(ay + ah, y + h) - max(ay, y))
        if ox * oy > frac * aA:
            return True
    return False


def find_blocks(img_bgr, dark_text, min_lines=3, exclude_boxes=None, max_lines=None):
    """Paragraph blocks (>= min_lines) on the page, dropping any that sit on art.
    max_lines (v25) caps block size so a short-text pass can take only the 1-2 line
    captions/headers the 3+ line paragraph pass leaves behind."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blocks = _group(_line_boxes(gray, dark_text), min_lines, page_h=gray.shape[0])
    exclude_boxes = exclude_boxes or []
    blocks = [b for b in blocks if not _overlaps(b[:4], exclude_boxes)]
    if max_lines is not None:
        blocks = [b for b in blocks if b[4] <= max_lines]
    return blocks


def dewarp_block(img_bgr, bbox, dark_text, pad_frac=0.08):
    """Deskew one paragraph block independently and return what to stamp:
        (px, py, text_vals, text_mask)
    where text_mask marks the deskewed text strokes and text_vals are their
    (rotated) source pixels, positioned to paste at (px, py) on the page. Only the
    strokes are returned -- never the block's background -- so the caller can place
    it on any canvas without stamping a rectangle. A pad around the crop guarantees
    the rotation never clips a glyph (no cropping)."""
    H, W = img_bgr.shape[:2]
    x, y, w, h = bbox[:4]
    px0 = max(0, int(x - w * pad_frac)); py0 = max(0, int(y - h * pad_frac))
    px1 = min(W, int(x + w * (1 + pad_frac))); py1 = min(H, int(y + h * (1 + pad_frac)))
    crop = img_bgr[py0:py1, px0:px1]
    cg = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    ang = _deskew_angle(cg, dark_text)
    cH, cW = crop.shape[:2]
    M = cv2.getRotationMatrix2D((cW / 2, cH / 2), ang, 1.0)
    rot = cv2.warpAffine(crop, M, (cW, cH), flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_REPLICATE)
    rmask = fg_mask(cv2.cvtColor(rot, cv2.COLOR_BGR2GRAY), dark_text)
    rmask = cv2.dilate(rmask, np.ones((2, 2), np.uint8))
    return px0, py0, rot, rmask, ang


def straighten_paragraphs(canvas, orig_img, art_boxes, detect_dark_text,
                          stamp, wipe=False, ink=(38, 38, 38), sat_max=70,
                          min_lines=3, max_lines=None):
    """Replace each paragraph block with an independently-deskewed version, placed
    without overlapping art or being cropped. Returns (canvas, angles). (Legacy path,
    kept for the SAFE_TEXT-off flow; the guarded page-text pass is deskew_text_blocks.)
    """
    blocks = find_blocks(orig_img, detect_dark_text, min_lines, art_boxes, max_lines=max_lines)
    angles = []
    H, W = canvas.shape[:2]
    hsv = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)
    for b in blocks:
        bx, by, bw, bh = b[:4]
        if float(hsv[by:by+bh, bx:bx+bw, 1].mean()) > sat_max:
            continue
        px, py, rot, rmask, ang = dewarp_block(orig_img, b, detect_dark_text)
        angles.append(ang)
        rh, rw = rmask.shape[:2]
        py = max(0, min(py, H - rh)); px = max(0, min(px, W - rw))
        if wipe:
            paper = np.median(canvas[max(0, by-6):by+bh+6,
                                     max(0, bx-6):bx+bw+6].reshape(-1, 3), axis=0)
            canvas[by:by+bh, bx:bx+bw] = paper.astype(np.uint8)
        dst = canvas[py:py+rh, px:px+rw]
        if dst.shape[:2] != rmask.shape[:2]:
            rmask = rmask[:dst.shape[0], :dst.shape[1]]
            rot = rot[:dst.shape[0], :dst.shape[1]]
        sel = rmask > 0
        dst[sel] = (np.array(ink, np.uint8) if stamp == 'ink' else rot[sel])
        canvas[py:py+rh, px:px+rw] = dst
    return canvas, angles


def deskew_text_blocks(canvas, art_boxes, dark_text=True, min_lines=2,
                       min_angle=0.25, max_angle=2.0, sat_max=70, pad_frac=0.05):
    """Straighten each PAGE-TEXT block (caption / credit column, header line) so the
    text matches the deskewed plates -- WITHOUT ever stepping onto art or another
    text block (v43). Returns (canvas, angles).

    Guarantees, in order:
      * A block that overlaps an ART box is never detected (find_blocks excludes it)
        and a colourful region is dropped (sat_max), so a plate is never re-inked.
      * A block is only ROTATED when its own tilt exceeds min_angle -- otherwise it
        is left exactly as scanned (no needless re-stamp = no speckle risk).
      * The straightened FOOTPRINT is rejected (block left as-is) if it would land on
        an art box OR on any block already straightened this pass, so deskewed text
        can never overlap other text or art.
      * Only then is the block's own footprint wiped to local paper and the deskewed
        strokes composited using the ORIGINAL grey pixel values (never a flat ink
        fill), so glyph edges stay clean.
    """
    H, W = canvas.shape[:2]
    hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)
    blocks = find_blocks(canvas, dark_text, min_lines=min_lines, exclude_boxes=art_boxes)
    blocks = sorted(blocks, key=lambda b: (b[1], b[0]))     # deterministic top->bottom
    placed = list(art_boxes)                                # occupied zones (art first)
    reserved = []                                           # every block's own spot
    angles = []
    for b in blocks:
        bx, by, bw, bh = b[:4]
        if float(hsv[by:by+bh, bx:bx+bw, 1].mean()) > sat_max:
            continue                                        # colourful -> not page text
        px, py, rot, rmask, ang = dewarp_block(canvas, b, dark_text, pad_frac=pad_frac)
        rh, rw = rmask.shape[:2]
        py = max(0, min(py, H - rh)); px = max(0, min(px, W - rw))
        foot = (px, py, rw, rh)
        reserved.append((bx, by, bw, bh))                   # its original area is spoken for
        if abs(ang) < min_angle or abs(ang) > max_angle:
            # below min_angle: already level -> leave the pixels as scanned.
            # above max_angle: page tilt on these scans is ~1 deg; a 3-6 deg reading is
            # an UNRELIABLE projection-profile estimate (short/justified block, edge
            # curl), so do NOT trust it -- leaving the text untouched is always safe.
            continue
        # GUARD: never straighten onto art or onto another block's footprint.
        others = placed + [r for r in reserved if r != (bx, by, bw, bh)]
        if _overlaps(foot, others, frac=0.02):
            continue
        # wipe own footprint to local paper, then stamp deskewed strokes (orig pixels)
        ring = canvas[max(0, py-8):min(H, py+rh+8),
                      max(0, px-8):min(W, px+rw+8)].reshape(-1, 3)
        paper = np.median(ring, axis=0).astype(np.uint8)
        canvas[py:py+rh, px:px+rw] = paper
        dst = canvas[py:py+rh, px:px+rw]
        sel = rmask[:dst.shape[0], :dst.shape[1]] > 0
        r2 = rot[:dst.shape[0], :dst.shape[1]]
        dst[sel] = r2[sel]
        canvas[py:py+rh, px:px+rw] = dst
        placed.append(foot); angles.append(ang)
    return canvas, angles


def dewarp_text_areas(canvas, art_boxes, dark_text=True, min_lines=1, sat_max=70,
                      min_disp=3.0, pad_frac=0.05):
    """Detect every text area (>= min_lines) and CURL-dewarp each independently.

    Columns are separated automatically (find_blocks clusters lines by x-overlap, so
    side-by-side columns never merge), and each column / heading / caption gets its
    OWN baseline field -- so a column that curls differently from its neighbour is
    corrected on its own. min_lines=1 also catches bold 1-2 line titles. A block is
    left exactly as-is if it sits on art, is colourful (a plate), or its measured
    warp is below min_disp px (clean text is never needlessly re-sampled). Returns
    (canvas, n_dewarped)."""
    import dewarp_text as _dt
    H, W = canvas.shape[:2]
    hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)
    blocks = find_blocks(canvas, dark_text, min_lines=min_lines, exclude_boxes=art_boxes)
    n = 0
    for b in sorted(blocks, key=lambda b: (b[1], b[0])):
        x, y, w, h = b[:4]
        if float(hsv[y:y+h, x:x+w, 1].mean()) > sat_max:
            continue
        pad = int(pad_frac * max(w, h))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        crop = canvas[y0:y1, x0:x1]
        dw, mag, mode = _dt.dewarp_text_area(crop)
        if mode != 'none' and mag >= min_disp:
            canvas[y0:y1, x0:x1] = dw
            n += 1
    return canvas, n

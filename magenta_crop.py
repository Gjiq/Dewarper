"""
magenta_crop -- MANUAL border override (boxes come from the markup; PIXELS come
from the clean original).

If a page is sent with art borders hand-drawn as a continuous MAGENTA outline, the
outline is taken as ground truth for WHERE the art rectangles are: each closed
magenta loop is one art box. Overrides all auto-detection of borders.

>>> MAGENTA SUPPLIES BOX GEOMETRY ONLY -- NEVER RECONSTRUCT FROM THE MAGENTA IMAGE <<<
  The magenta-marked file is a MARKUP copy: it is usually re-saved/re-compressed
  (often a fraction of the original's size and quality) and it has ink drawn over
  the art. The deliverable must be reconstructed from the CLEAN ORIGINAL scan, with
  the full deskew/dewarp applied, using the magenta loops only as the manual art
  boxes. So the correct workflow needs TWO images that are PIXEL-ALIGNED (the
  magenta drawn on a copy of the exact original scan):
      magenta copy  -> magenta_boxes()  -> the manual [x0,y0,x1,y1] art boxes
      clean original + those boxes -> magenta_dewarp() -> full deskew/dewarp/compose
  If you are handed a magenta page and do NOT have its clean original, STOP and ask
  for the original before reconstructing -- do not dewarp/ship the magenta version.

magenta_compose() (the OLD literal crop-and-paste straight off the magenta image,
no deskew) is kept only as a quick PREVIEW of the box placement. It is NOT a
deliverable: it carries the markup copy's lower quality and skips the dewarp. Use
magenta_dewarp() with the original for anything shipped.

The magenta is one specific colour (HSV hue ~163-165, high sat) and the lines must
be continuous/closed. Works for single-image, multi-image and text-column pages.
"""
import cv2
import numpy as np
import deskew_crop as dk

# The one magenta (sampled: H 163-165, S ~200-234, V ~182-209) with margin.
MAGENTA_LO = np.array([156, 130, 110], np.uint8)
MAGENTA_HI = np.array([172, 255, 255], np.uint8)


def magenta_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, MAGENTA_LO, MAGENTA_HI)


def has_magenta(img, min_px=2000):
    return int((magenta_mask(img) > 0).sum()) >= min_px


def magenta_boxes(img, min_area_frac=0.03, min_dim=0.15, min_rect=0.80):
    """Closed magenta loops -> inner-edge [x0,y0,x1,y1] art boxes (page order).

    Robust to small breaks in the line: gaps are bridged DIRECTIONALLY (long
    horizontal + vertical closes) so a rectangle outline re-closes without merging
    parallel edges of adjacent boxes. Each closed loop's hole is one art border."""
    H, W = img.shape[:2]
    m = magenta_mask(img)
    if (m > 0).sum() < 2000:
        return []
    kd = max(41, int(0.018 * max(H, W))) | 1
    mh = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((1, kd), np.uint8))
    mv = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((kd, 1), np.uint8))
    mcl = cv2.morphologyEx(cv2.bitwise_or(mh, mv), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cnts, hier = cv2.findContours(mcl, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    A = H * W
    boxes = []
    if hier is not None:
        for i in range(len(cnts)):
            if hier[0][i][3] == -1:                       # only holes (enclosed interiors)
                continue
            x, y, w, h = cv2.boundingRect(cnts[i])
            if not (w * h > min_area_frac * A and w > min_dim * W and h > min_dim * H):
                continue
            # A hand-drawn art box is LARGE and a rectangular OUTLINE: its enclosed
            # hole FILLS its bounding rect (~0.95+). Incidental magenta-hued ART that
            # happens to enclose a hole is irregular (~0.6) -> reject, so a clean page
            # with red/pink paint (e.g. a red tablecloth) is not read as a markup.
            if cv2.contourArea(cnts[i]) < min_rect * w * h:
                continue
            boxes.append((int(x), int(y), int(x + w), int(y + h)))
    boxes.sort(key=lambda b: (round(b[1] / (0.12 * H)), b[0]))   # top-to-bottom, then left-to-right
    return boxes


def _corner_bg(img):
    """Page background level = median of the four corner grays. Same anchor the
    classifier uses for `bg`, recomputed here so the manual path is self-contained."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = g.shape[:2]
    return float(np.median([g[10, 10], g[10, W-10], g[H-10, 10], g[H-10, W-10]]))


def magenta_dewarp(magenta_img, orig_img, flatfield_whiten, bg=None):
    """CORRECT magenta workflow: boxes from the markup, PIXELS from the original.

    Take the box geometry from the magenta-marked page, then reconstruct from the
    CLEAN ORIGINAL: each magenta loop becomes a manual art box and the ORIGINAL
    pixels in that box are deskewed + inward-cropped (the same full dewarp the
    FULL_BLEED multi path runs) and composited onto the whitened original page.
    The magenta copy is touched ONLY to locate the boxes; not one of its pixels
    reaches the output.

    magenta_img and orig_img MUST be pixel-aligned -- draw the magenta on a COPY of
    the exact original scan so the loops land on the right pixels. Mismatched sizes
    raise (a resized/re-photographed markup cannot be trusted to register).

    Returns (reconstructed_page, boxes, thetas), or (None, [], []) if the markup
    carries no usable magenta outline. `flatfield_whiten` is passed in (from
    dewarp_text) to keep this module import-light; `bg` defaults to the original's
    corner background.
    """
    import deskew_crop as dk
    if magenta_img.shape[:2] != orig_img.shape[:2]:
        raise ValueError(
            "magenta %s and original %s are not the same size -- draw the magenta "
            "on a COPY of the exact original scan so the two are pixel-aligned."
            % (magenta_img.shape[:2], orig_img.shape[:2]))
    boxes = magenta_boxes(magenta_img)
    if not boxes:
        return None, [], []
    if bg is None:
        bg = _corner_bg(orig_img)
    out, thetas = dk.compose_multi(orig_img, bg, flatfield_whiten, boxes, clip_boxes=True)
    return out, boxes, thetas


def magenta_compose(img):
    """PREVIEW ONLY -- literal crop-and-paste straight off the magenta image, no
    deskew/dewarp. Carries the markup copy's (usually lower) quality, so it is NOT
    a deliverable; use magenta_dewarp() with the clean original for anything
    shipped. Returns (preview_page, boxes) using the magenta loops as art borders,
    or (None, []) if no usable magenta outline is present."""
    boxes = magenta_boxes(img)
    if not boxes:
        return None, []
    tex = dk._bg_texture(img, paper=True)
    out = tex.copy()
    box_mask = np.zeros(img.shape[:2], bool)
    for (x0, y0, x1, y1) in boxes:
        box_mask[y0:y1, x0:x1] = True
    keep = ~box_mask                                    # text / page numbers / paper outside loops
    out[keep] = img[keep]
    for (x0, y0, x1, y1) in boxes:                      # art to the inner edge of each loop
        out[y0:y1, x0:x1] = img[y0:y1, x0:x1]
    # scrub the magenta line + its soft halo (now only outside the boxes) to texture
    resid = cv2.dilate(magenta_mask(out), np.ones((9, 9), np.uint8))
    out[resid > 0] = tex[resid > 0]
    return out, boxes


if __name__ == '__main__':
    import sys, os
    args = sys.argv[1:]
    orig_path = None
    if '--original' in args:
        i = args.index('--original')
        orig_path = args[i + 1]
        del args[i:i + 2]

    if orig_path:
        # CORRECT path: boxes from the magenta markup, pixels from the clean original.
        from dewarp_text import flatfield_whiten
        orig = cv2.imread(orig_path)
        for p in args:                              # remaining args = magenta markup(s)
            mag = cv2.imread(p)
            out, boxes, thetas = magenta_dewarp(mag, orig, flatfield_whiten)
            if out is None:
                print(os.path.basename(p), 'NO MAGENTA'); continue
            stem, ext = os.path.splitext(orig_path)
            cv2.imwrite(f'{stem}_reconstructed{ext}', out, [cv2.IMWRITE_JPEG_QUALITY, 92])
            print('%s boxes from %s + original %s -> %d box(es) dewarped %s deg' % (
                len(boxes), os.path.basename(p), os.path.basename(orig_path),
                len(boxes), ', '.join('%+.2f' % t for t in thetas)))
    else:
        # No original supplied -> PREVIEW ONLY + a loud reminder. Never ship this.
        for p in args:
            im = cv2.imread(p)
            out, boxes = magenta_compose(im)
            if out is None:
                print(os.path.basename(p), 'NO MAGENTA'); continue
            stem, ext = os.path.splitext(p)
            cv2.imwrite(f'{stem}_magcrop{ext}', out, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(os.path.basename(p), len(boxes), 'boxes', boxes,
                  '\n  PREVIEW ONLY (literal crop off the markup, no dewarp). For a '
                  'deliverable, re-run with:\n  python3 magenta_crop.py --original '
                  '<CLEAN_ORIGINAL>.jpg ' + os.path.basename(p))


# ── Full-page art FRAME (magenta as art/background separator) ──────────────────
# For a full-bleed painting whose art edge blends into a dark page margin (so the
# art/background boundary can't be found automatically), the markup draws a single
# magenta frame that hugs the page edges. That frame IS the art boundary: trust it
# as the art quadrilateral and perspective-rectify it to an upright rectangle,
# dropping everything outside (margins, header, page number).

def _order_quad(pts):
    """Order 4 points as TL, TR, BR, BL."""
    pts = np.asarray(pts, np.float32)
    s = pts.sum(1); d = np.diff(pts, 1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()],
                     pts[s.argmax()], pts[d.argmax()]], np.float32)


def magenta_frame_corners(markup):
    """The 4 corners of the art region enclosed by a magenta frame (the INNER edge
    of the magenta ring = the art side of the boundary). Returns TL,TR,BR,BL or None."""
    mask = magenta_mask(markup)
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return None
    holes = [c for c, h in zip(cnts, hier[0]) if h[3] != -1]
    if not holes:
        return None
    c = max(holes, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    ap = cv2.approxPolyDP(c, 0.02 * peri, True)
    pts = ap.reshape(-1, 2) if len(ap) == 4 else c.reshape(-1, 2)
    return _order_quad(pts)


def is_full_page_frame(markup, margin_frac=0.06, area_frac=0.55):
    """True when the magenta is a single loop hugging the page edges (a full-page
    art frame), as opposed to an inset art-plate box. Used to route to the
    perspective frame-dewarp instead of the normal box/dark paths."""
    q = magenta_frame_corners(markup)
    if q is None:
        return False
    H, W = markup.shape[:2]
    mL = q[:, 0].min() / W; mR = (W - q[:, 0].max()) / W
    mT = q[:, 1].min() / H; mB = (H - q[:, 1].max()) / H
    area = cv2.contourArea(q.astype(np.int32)) / float(W * H)
    return area > area_frac and max(mL, mR, mT, mB) < margin_frac


def magenta_frame_dewarp(clean, markup):
    """Perspective-rectify the art enclosed by a magenta frame to an upright
    rectangle. `clean` supplies the pixels, `markup` the frame geometry. Returns the
    isolated, dewarped art (background outside the frame removed) or None."""
    q = magenta_frame_corners(markup)
    if q is None:
        return None
    tl, tr, br, bl = q
    W = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    H = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
    M = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(clean, M, (W, H),
                               flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def magenta_frame_blackcover(clean, markup, seam=5, safety=4):
    """Full-page FRAME reconstruction (reseat-and-cover).

    Perspective-rectify the framed art to a straight NATIVE-size rectangle (rectify
    UNCHANGED), then cover the residual warped edge NOT with a black band but by
    STRETCHING the straightened art outward over it: the art is rectified into a rect
    inset by each side's own warp slant (+safety) inside a larger canvas, so
    warpPerspective fills those bands with the perspective-straightened pixels from just
    outside the frame -- the new straight art becomes the top layer covering the warped
    border underneath, no black. Band width per side scales with the warp. Returns the
    full page, or None if no frame is found."""
    q = magenta_frame_corners(markup)
    if q is None:
        return None
    (TLx, TLy), (TRx, TRy), (BRx, BRy), (BLx, BLy) = q
    W = int(round(max(np.hypot(TRx - TLx, TRy - TLy), np.hypot(BRx - BLx, BRy - BLy))))
    H = int(round(max(np.hypot(BLx - TLx, BLy - TLy), np.hypot(BRx - TRx, BRy - TRy))))
    Hc, Wc = clean.shape[:2]
    ax0, ay0 = int(q[:, 0].min()), int(q[:, 1].min())
    # per-side stretch = that quad edge's slant (how far it warps) + safety
    pt = int(round(abs(TLy - TRy))) + safety
    pb = int(round(abs(BLy - BRy))) + safety
    pl = int(round(abs(TLx - BLx))) + safety
    pr = int(round(abs(TRx - BRx))) + safety
    oW = W + pl + pr; oH = H + pt + pb
    # map the frame quad -> a rect INSET by the per-side stretch inside the oW x oH canvas;
    # warpPerspective straightens the content just outside the frame into the bands,
    # covering the warped edge with real (not black) pixels.
    dst = np.array([[pl, pt], [pl + W, pt], [pl + W, pt + H], [pl, pt + H]], np.float32)
    patch = cv2.warpPerspective(clean, cv2.getPerspectiveTransform(q, dst), (oW, oH),
                                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    out = clean.copy()
    px0 = ax0 - pl; py0 = ay0 - pt                        # where the patch top-left lands
    sx0 = max(0, -px0); sy0 = max(0, -py0)
    dx0 = max(0, px0); dy0 = max(0, py0)
    dx1 = min(Wc, px0 + oW); dy1 = min(Hc, py0 + oH)
    out[dy0:dy1, dx0:dx1] = patch[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
    out = _protect_text_in_stretch(out, clean, (dy0, dy1, dx0, dx1), (ay0, ay0 + H, ax0, ax0 + W))
    return out


# ── WHITE/BLACK printed keyline frame (auto analogue of the magenta frame) ─────
def _runs(strip, thr, maxrun):
    b = (strip > thr).astype(np.uint8)
    if not b.any():
        return []
    idx = np.flatnonzero(np.diff(np.concatenate(([0], b, [0]))))
    s, e = idx[0::2], idx[1::2]
    return [(a + z) / 2.0 for a, z in zip(s, e) if z - a <= maxrun]


def _ransac_line(us, vs, span, tol=10, iters=400, min_span=0.55):
    us = np.asarray(us, float); vs = np.asarray(vs, float); n = len(us)
    if n < 20:
        return None
    rng = np.random.default_rng(0); best = None; bestc = 0
    for _ in range(iters):
        i, j = rng.integers(0, n, 2)
        if us[i] == us[j]:
            continue
        a = (vs[j] - vs[i]) / (us[j] - us[i]); b = vs[i] - a * us[i]
        inl = np.abs(vs - (a * us + b)) < tol
        if int(inl.sum()) > bestc and (us[inl].max() - us[inl].min()) > min_span * span:
            bestc = int(inl.sum()); best = inl
    if best is None:
        return None
    return np.polyfit(us[best], vs[best], 1)


def keyline_frame_corners(img, ring_frac=0.18, maxrun=40):
    """Find the artwork's own printed rectangular keyline (WHITE or BLACK, any
    thickness, possibly warped) and return its 4 corners TL,TR,BR,BL -- the auto
    analogue of a magenta frame. Searches only the margin ring, collects every thin
    bright/dark run, and RANSAC-fits the ONE full-length straight line per edge (so
    header text / art gloss are ignored). Validated by the caller. Returns quad|None."""
    H, W = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int16)
    rx = int(ring_frac * W); ry = int(ring_frac * H)
    ground = int(np.percentile(g, 30))
    for thr, cmp in ((max(150, ground + 90), 'gt'),               # white keyline
                     (min(90, int(np.percentile(g, 70)) - 90), 'lt')):  # black keyline (light pages)
        gg = g if cmp == 'gt' else (255 - g)
        tt = thr if cmp == 'gt' else (255 - thr)
        Ly = []; Lx = []; Ry = []; Rx = []; Tx = []; Ty = []; Bx = []; By = []
        for y in range(H):
            for x in _runs(gg[y, :rx], tt, maxrun): Ly.append(y); Lx.append(x)
            for x in _runs(gg[y, W-rx:], tt, maxrun): Ry.append(y); Rx.append(W-rx+x)
        for x in range(W):
            for yv in _runs(gg[:ry, x], tt, maxrun): Tx.append(x); Ty.append(yv)
            for yv in _runs(gg[H-ry:, x], tt, maxrun): Bx.append(x); By.append(H-ry+yv)
        cL = _ransac_line(Ly, Lx, H); cR = _ransac_line(Ry, Rx, H)
        cT = _ransac_line(Tx, Ty, W); cB = _ransac_line(Bx, By, W)
        if any(c is None for c in (cL, cR, cT, cB)):
            continue

        def inter(a1, a2):
            a, b = a1; c, d = a2; y = (c*b + d) / (1 - c*a); x = a*y + b
            return np.array([x, y], np.float32)
        q = np.array([inter(cL, cT), inter(cR, cT), inter(cR, cB), inter(cL, cB)], np.float32)
        # VALIDATE: every fitted edge must actually sit on the keyline tone.
        def eb(kind, c):
            if kind in 'LR':
                v = [g[yy, min(W-1, max(0, int(np.polyval(c, yy))))]
                     for yy in np.linspace(0.12*H, 0.88*H, 60).astype(int)]
            else:
                v = [g[min(H-1, max(0, int(np.polyval(c, xx)))), xx]
                     for xx in np.linspace(0.12*W, 0.88*W, 60).astype(int)]
            return float(np.median(v))
        vals = [eb('L', cL), eb('R', cR), eb('T', cT), eb('B', cB)]
        ok = all(v > 150 for v in vals) if cmp == 'gt' else all(v < 105 for v in vals)
        if ok and (q[:, 0].max()-q[:, 0].min()) > 0.5*W and (q[:, 1].max()-q[:, 1].min()) > 0.5*H:
            return q
    return None


def keyline_frame_dewarp(img):
    """Rectify the page to its detected keyline quad (perspective). Returns dewarped
    page, or None if no validated frame is found."""
    q = keyline_frame_corners(img)
    if q is None:
        return None
    H, W = img.shape[:2]
    x0, y0 = float(q[:, 0].min()), float(q[:, 1].min())
    x1, y1 = float(q[:, 0].max()), float(q[:, 1].max())
    dst = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(q.astype(np.float32), dst),
                               (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


# ── auto WHITE keyline frame deskew (4-step: ring→lines→corners→perspective) ──
def keyline_frame_deskew(img, ring_frac=0.18):
    """Detect the artwork's printed white keyline in the outer margin ring, fit the four
    edges as straight lines (robust, so header/gloss are rejected), intersect to corners,
    and perspective-warp that skewed quad to an axis-aligned rectangle. Returns the
    deskewed page, or None if a validated frame isn't found (all four edges must sit on
    the bright keyline). No bow-flatten -- perspective only."""
    H, W = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ground = int(np.percentile(g, 30)); thr = max(150, ground + 90)
    rx = int(ring_frac * W); ry = int(ring_frac * H)
    def pLR(s):
        p = []
        for y in range(int(0.05*H), int(0.95*H), 3):
            b = np.where(g[y, :rx] > thr)[0] if s == 'L' else np.where(g[y, W-rx:] > thr)[0]
            if len(b): p.append((y, b[0] if s == 'L' else W-rx+b[-1]))
        return np.array(p, float)
    def pTB(s):
        p = []
        for x in range(int(0.05*W), int(0.95*W), 3):
            b = np.where(g[:ry, x] > thr)[0] if s == 'T' else np.where(g[H-ry:, x] > thr)[0]
            if len(b): p.append((x, b[0] if s == 'T' else H-ry+b[-1]))
        return np.array(p, float)
    def rob(pts):
        if len(pts) < 30: return None
        u, v = pts[:, 0], pts[:, 1]; c = np.polyfit(u, v, 1)
        for _ in range(6):
            r = v - np.polyval(c, u); k = np.abs(r) < 2*np.std(r) + 2
            if k.sum() < len(u)*0.4: break
            c = np.polyfit(u[k], v[k], 1)
        return c
    cL, cR, cT, cB = rob(pLR('L')), rob(pLR('R')), rob(pTB('T')), rob(pTB('B'))
    if any(c is None for c in (cL, cR, cT, cB)):
        return None
    # validate: every fitted edge must sit on the bright keyline along its length
    def eb(kind, c):
        if kind in 'LR':
            v = [g[y, min(W-1, max(0, int(np.polyval(c, y))))] for y in np.linspace(0.12*H, 0.88*H, 60).astype(int)]
        else:
            v = [g[min(H-1, max(0, int(np.polyval(c, x)))), x] for x in np.linspace(0.12*W, 0.88*W, 60).astype(int)]
        return float(np.median(v))
    if not (eb('L', cL) > 150 and eb('R', cR) > 150 and eb('T', cT) > 150 and eb('B', cB) > 150):
        return None
    def inter(cxy, cyx, ys):
        y = ys
        for _ in range(6): x = np.polyval(cxy, y); y = np.polyval(cyx, x)
        return [x, y]
    TL = inter(cL, cT, np.polyval(cT, 0.08*W)); TR = inter(cR, cT, np.polyval(cT, 0.92*W))
    BR = inter(cR, cB, np.polyval(cB, 0.92*W)); BL = inter(cL, cB, np.polyval(cB, 0.08*W))
    src = np.array([TL, TR, BR, BL], np.float32)
    X0 = min(TL[0], BL[0]); X1 = max(TR[0], BR[0]); Y0 = min(TL[1], TR[1]); Y1 = max(BL[1], BR[1])
    if (X1-X0) < 0.5*W or (Y1-Y0) < 0.5*H:
        return None
    dst = np.array([[X0, Y0], [X1, Y0], [X1, Y1], [X0, Y1]], np.float32)
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (W, H),
                               flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


# ── Auto WHITE/BLACK keyline-frame DESKEW (the 4-step method) ──────────────────
def _kl_thin_runs(strip, thr, maxrun=40):
    b = (strip > thr).astype(np.uint8)
    if not b.any():
        return []
    idx = np.flatnonzero(np.diff(np.concatenate(([0], b, [0])))); s, e = idx[0::2], idx[1::2]
    return [(a + z) / 2.0 for a, z in zip(s, e) if z - a <= maxrun]


def _kl_pts_LR(g, side, thr, rx, H, W):
    # ALL thin bright runs per row -> the keyline is the one continuous across the edge;
    # header/banner/gloss give scattered runs that RANSAC drops.
    p = []
    for y in range(int(0.05*H), int(0.95*H), 3):
        strip = g[y, :rx] if side == 'L' else g[y, W-rx:]
        for xr in _kl_thin_runs(strip, thr):
            p.append((y, xr if side == 'L' else W-rx+xr))
    return np.array(p, float) if p else np.zeros((0, 2))


def _kl_pts_TB(g, side, thr, ry, H, W):
    p = []
    for x in range(int(0.05*W), int(0.95*W), 3):
        strip = g[:ry, x] if side == 'T' else g[H-ry:, x]
        for yr in _kl_thin_runs(strip, thr):
            p.append((x, yr if side == 'T' else H-ry+yr))
    return np.array(p, float) if p else np.zeros((0, 2))


def _kl_robust_line(pts, deg=1, span=None, tol=8, iters=500, min_span=0.55, max_slope=0.15):
    """RANSAC the single FULL-LENGTH straight keyline out of all runs (so scattered
    header/banner/gloss points can't pull the fit), rejecting candidates steeper than
    max_slope (a real frame edge is near-axis-aligned; steep diagonals through clutter
    are spurious), then refine on inliers at `deg` (deg-3 for the bowed top/bottom).
    Returns None if no near-straight line spans the edge."""
    if len(pts) < 20:
        return None
    u, v = pts[:, 0], pts[:, 1]
    if span is None:
        span = u.max() - u.min()
    rng = np.random.default_rng(0); best = None; bc = 0
    for _ in range(iters):
        i, j = rng.integers(0, len(u), 2)
        if u[i] == u[j]:
            continue
        a = (v[j] - v[i]) / (u[j] - u[i]); b = v[i] - a*u[i]
        if abs(a) > max_slope:                      # not a near-axis-aligned frame edge
            continue
        inl = np.abs(v - (a*u + b)) < tol
        if int(inl.sum()) > bc and (u[inl].max() - u[inl].min()) > min_span*span:
            bc = int(inl.sum()); best = inl
    if best is None:
        return None
    return np.polyfit(u[best], v[best], deg)


def _protect_text_in_stretch(out, clean, outer, core, margin=10):
    """In the STRETCHED RING (outer rect minus the core dewarped art), restore the
    ORIGINAL clean pixels wherever there is TEXT (a caption/header), plus a small margin
    -- so the stretch never distorts text. Only the ring is touched; the core art is left
    as dewarped. outer/core are (y0, y1, x0, x1)."""
    Y0, Y1, X0, X1 = outer; cy0, cy1, cx0, cx1 = core
    g = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    ink = (g.astype(np.int16) < (cv2.blur(g, (41, 41)).astype(np.int16) - 16)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    tm = np.zeros_like(ink)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if 3 < h < 70 and w < 400 and 5 < a < 4000:      # small stroke/char = text-like
            tm[lab == i] = 1
    if tm.sum() < 50:
        return out
    tm = cv2.dilate(tm, np.ones((margin*2+1, margin*2+1), np.uint8))
    ring = np.zeros(g.shape, bool); ring[Y0:Y1, X0:X1] = True
    ring[max(0, cy0):cy1, max(0, cx0):cx1] = False        # never touch the core art
    m = (tm > 0) & ring
    out[m] = clean[m]
    return out


def keyline_deskew(img, ring_frac=0.18):
    """Deskew AND de-bow a page to its own printed rectangular keyline (WHITE or BLACK,
    warped ok). 4 steps: (1) keyline bright/dark spike in the outer margin ring only;
    (2) robust fit -- left/right as straight lines, top/bottom as deg-3 CURVES; (3) build
    a separable transfinite map (horizontal position interpolated between the left/right
    edge curves, vertical between the top/bottom edge curves); (4) remap to a straight
    rectangle -- this fixes skew AND edge bow. Accepted ONLY if all four fitted edges sit
    on the keyline tone. Returns the dewarped page, or None (caller does normal work)."""
    H, W = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rx = int(ring_frac*W); ry = int(ring_frac*H)
    ground = int(np.percentile(g, 30))
    for mode, gg, thr in (('white', g, max(150, ground+90)),
                          ('black', 255-g, 255-min(90, int(np.percentile(g, 70))-90))):
        L = _kl_pts_LR(gg, 'L', thr, rx, H, W); R = _kl_pts_LR(gg, 'R', thr, rx, H, W)
        T = _kl_pts_TB(gg, 'T', thr, ry, H, W); B = _kl_pts_TB(gg, 'B', thr, ry, H, W)
        cL = _kl_robust_line(L, 1); cR = _kl_robust_line(R, 1)
        cT = _kl_robust_line(T, 3); cB = _kl_robust_line(B, 3)     # top/bottom as curves
        if any(c is None for c in (cL, cR, cT, cB)):
            continue

        def edge_spike(kind, c):
            """(on-line median, brighter-neighbour median). A real keyline is a bright
            line standing out from a DARKER surround on BOTH sides -- not just 'bright'
            (which any paper margin is). Sampling +/-30px perpendicular gives the surround."""
            def samp(doff):
                if kind in 'LR':
                    return float(np.median([g[yy, min(W-1, max(0, int(np.polyval(c, yy)) + doff))]
                                            for yy in np.linspace(0.12*H, 0.88*H, 60).astype(int)]))
                return float(np.median([g[min(H-1, max(0, int(np.polyval(c, xx)) + doff)), xx]
                                        for xx in np.linspace(0.12*W, 0.88*W, 60).astype(int)]))
            on = samp(0); a = samp(30); b = samp(-30)
            return on, max(a, b), min(a, b)
        spikes = {}
        for k, c in (('L', cL), ('R', cR), ('T', cT), ('B', cB)):
            on, hi, lo = edge_spike(k, c)
            spikes[k] = (on > 150 and on - hi > 40) if mode == 'white' else (on < 95 and lo - on > 40)
        npass = sum(spikes.values())
        if npass == 4:
            pass                                         # all four keylines confirmed
        elif npass == 3 and mode == 'white':
            # 3 strong keylines + 1 weak/missing -> INFER the missing edge from the
            # rectangle (its position = the extent of the two strong perpendicular
            # keylines). Lets a frame whose top rule is faint/broken (e.g. under a header)
            # still dewarp off its three solid edges instead of being rejected.
            bad = [k for k, v in spikes.items() if not v][0]

            def _inl(pts, c, axis):
                if len(pts) < 20:
                    return None
                m = np.abs(pts[:, 1] - np.polyval(c, pts[:, 0])) < 8
                return pts[m, axis] if m.sum() >= 10 else None
            if bad in ('T', 'B'):
                eL = _inl(L, cL, 0); eR = _inl(R, cR, 0)
                if eL is None or eR is None:
                    continue
                if bad == 'T':
                    cT = np.array([0.0, min(np.percentile(eL, 4), np.percentile(eR, 4))])
                else:
                    cB = np.array([0.0, max(np.percentile(eL, 96), np.percentile(eR, 96))])
            else:
                eT = _inl(T, cT, 1); eB = _inl(B, cB, 1)
                if eT is None or eB is None:
                    continue
                if bad == 'L':
                    cL = np.array([0.0, min(np.percentile(eT, 4), np.percentile(eB, 4))])
                else:
                    cR = np.array([0.0, max(np.percentile(eT, 96), np.percentile(eB, 96))])
        else:
            continue
        x0 = int(np.polyval(cL, H/2)); x1 = int(np.polyval(cR, H/2))
        y0 = int(np.mean(np.polyval(cT, np.linspace(0.1*W, 0.9*W, 50))))
        y1 = int(np.mean(np.polyval(cB, np.linspace(0.1*W, 0.9*W, 50))))
        w = x1 - x0; h = y1 - y0
        if w < 0.5*W or h < 0.5*H:
            continue
        # STRETCH the straightened art 5% outward past the keyline so its edge covers the
        # old WARPED border underneath (the new art becomes the top layer). Source samples
        # just past the keyline (curves extrapolate); U/V run slightly beyond [0,1].
        pad = 0.05
        X0 = max(0, int(x0 - pad*w)); X1 = min(W, int(x1 + pad*w))
        Y0 = max(0, int(y0 - pad*h)); Y1 = min(H, int(y1 + pad*h))
        xs = np.arange(X0, X1); ys = np.arange(Y0, Y1)
        U = (xs - x0) / float(w); V = (ys - y0) / float(h); UU, VV = np.meshgrid(U, V)
        Xt = x0 + UU*w; Yt = np.polyval(cT, Xt); Yb = np.polyval(cB, Xt)
        Yl = y0 + VV*h; Xl = np.polyval(cL, Yl); Xr = np.polyval(cR, Yl)
        Sx = ((1-UU)*Xl + UU*Xr).astype(np.float32)
        Sy = ((1-VV)*Yt + VV*Yb).astype(np.float32)
        out = img.copy()
        out[Y0:Y1, X0:X1] = cv2.remap(img, Sx, Sy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        out = _protect_text_in_stretch(out, img, (Y0, Y1, X0, X1), (y0, y1, x0, x1))
        return out
    return None

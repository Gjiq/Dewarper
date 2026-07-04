"""
dark_art.py  -- dark-background art-annual plate reconstruction (DARK_BG branch).

Chain (light path is untouched; this is dark-branch only):
  1. detect art box by LIGHTNESS vs the sampled dark-bg edges  (NO saturation)
  2. vertical-rectify the box top/bottom edges  -> box edges axis-aligned
  3. crop to a perfect rectangle 5px inside the plate edge
  4. reconstruct the page background: remove art+text, inpaint the holes
  5. lay the reconstructed bg down first, then composite:
       - the dewarped art box, centered HORIZONTALLY (original vertical kept)
       - the page text, deskewed to its own baseline
Detection keys on lightness only, so warm / cool / dark art all behave the same.
"""
import cv2, numpy as np


# ---------- page background reference (sample the dark-bg edges) ----------
def page_lab(img, frac=0.012):
    H, W = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    e = np.concatenate([lab[:int(H*frac)].reshape(-1, 3),
                        lab[:, :int(W*frac)].reshape(-1, 3),
                        lab[:, -int(W*frac):].reshape(-1, 3)])
    return np.median(e, 0)            # pL, pA, pB


# ---------- art-box mask by COLOR and LIGHTNESS vs the sampled page ----------
# (no saturation: lightness = L-pL; color = per-channel a/b difference from the page)
def art_body(img, dL=22, dC=14, roi=None):
    H, W = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    pL, pA, pB = page_lab(img)
    content = ((L > pL + dL) | (np.abs(A - pA) > dC) | (np.abs(B - pB) > dC)).astype(np.uint8)
    content = cv2.morphologyEx(content, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    if roi is not None:                                   # constrain to a magenta box
        rx0, ry0, rx1, ry1 = roi
        keep = np.zeros_like(content); keep[ry0:ry1, rx0:rx1] = content[ry0:ry1, rx0:rx1]
        content = keep
    # the art box is the large SOLID block; find it by row/col content projection
    rc = content.mean(1); cc = content.mean(0)
    rows = np.where(rc > 0.45 * (content.max() or 1))[0]
    colsi = np.where(cc > 0.45 * (content.max() or 1))[0]
    if len(rows) < 20 or len(colsi) < 20:
        return None
    ry = max(np.split(rows, np.where(np.diff(rows) > 60)[0] + 1), key=len)
    cxg = max(np.split(colsi, np.where(np.diff(colsi) > 60)[0] + 1), key=len)
    y0, y1, x0, x1 = ry.min(), ry.max(), cxg.min(), cxg.max()
    body = np.zeros((H, W), np.uint8); body[y0:y1+1, x0:x1+1] = 1
    return body


def _rfit(t, p, deg):
    t = np.asarray(t, float); p = np.asarray(p, float)
    c = np.polyfit(t, p, deg)
    for _ in range(6):
        r = p - np.polyval(c, t); k = np.abs(r) < 1.3*np.std(r) + 3
        if k.sum() < len(t)*0.4:
            break
        c = np.polyfit(t[k], p[k], deg)
    return c


# ---------- vertical rectification of the box ----------
def rectify(img, body):
    H, W = img.shape[:2]
    x, y, w, h = cv2.boundingRect(body)
    cols = np.arange(x, x+w)
    tY = np.array([np.argmax(body[:, c]) for c in cols], float)
    bY = np.array([H-1-np.argmax(body[::-1, c]) for c in cols], float)
    ct = _rfit(cols, tY, 3); cb = _rfit(cols, bY, 3)
    topC = np.polyval(ct, cols); botC = np.polyval(cb, cols)
    T = topC.min(); B = botC.max()
    mapx = np.tile(np.arange(W, dtype=np.float32), (H, 1))
    mapy = np.tile(np.arange(H, dtype=np.float32)[:, None], (1, W))
    yy = np.arange(int(T), int(B)+1, dtype=np.float32)
    for j, c in enumerate(cols):
        t, bm = topC[j], max(botC[j], topC[j]+10)
        mapy[int(T):int(B)+1, c] = t + (yy - T)*(bm - t)/(B - T)
    out = cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return out, (x, int(T), x+w, int(B))


# ---------- text: letter blobs outside the art box ----------
def _letter_mask(reg):
    B, G, R = reg[:, :, 0].astype(int), reg[:, :, 1].astype(int), reg[:, :, 2].astype(int)
    V = reg.max(2)
    m = ((V > 150) | ((R > 120) & (G < 100) & (B < 100))).astype(np.uint8)   # white OR red text
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lb, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        a, ww, hh = st[i, cv2.CC_STAT_AREA], st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        if 25 < a < 9000 and hh < 90 and ww < 200:
            keep[lb == i] = 1
    return keep


def _slope(m):
    ys, xs = np.where(m)
    if len(xs) < 60:
        return 0.0
    a, b = np.polyfit(xs.astype(float), ys.astype(float), 1)
    for _ in range(4):
        r = ys - (a*xs + b); k = np.abs(r) < 1.3*np.std(r) + 2
        a, b = np.polyfit(xs[k].astype(float), ys[k].astype(float), 1)
    return a


def text_bands(img, rects):
    """Return list of (y0,y1) horizontal bands of text outside the art box(es).
    `rects` is one (x0,y0,x1,y1) tuple or a list of them."""
    H, W = img.shape[:2]
    if rects and isinstance(rects[0], (int, np.integer)):
        rects = [rects]
    lm = _letter_mask(img)
    for (x0, y0, x1, y1) in rects:                       # ignore text inside any plate
        lm[max(0, y0-8):min(H, y1+8), max(0, x0-8):min(W, x1+8)] = 0
    rowhit = (lm.sum(1) > 0).astype(np.uint8)
    bands = []
    ys = np.where(rowhit)[0]
    if len(ys):
        for g in np.split(ys, np.where(np.diff(ys) > 40)[0] + 1):
            if len(g) > 4:
                bands.append((max(0, g.min()-12), min(H, g.max()+12)))
    return bands


def _drop_text(out, orig, y0, y1):
    bd = orig[y0:y1].copy(); bh, bw = bd.shape[:2]
    lmask = _letter_mask(bd)
    ys_, xs_ = np.where(lmask)
    if len(xs_) < 20:
        return
    cx, cy = float(xs_.mean()), float(ys_.mean())   # rotate about the TEXT itself,
    al = np.degrees(np.arctan(_slope(lmask)))         # not the page-wide band centre:
    best = None                                       # a right-side caption swung
    for th in (al, -al):                              # ~20px about bw/2 and clipped its
        M = cv2.getRotationMatrix2D((cx, cy), th, 1.0)  # first line out of the band.
        rot = cv2.warpAffine(bd, M, (bw, bh), flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0))
        a2 = abs(np.degrees(np.arctan(_slope(_letter_mask(rot)))))
        if best is None or a2 < best[0]:
            best = (a2, rot)
    rot = best[1]
    tm = cv2.dilate(_letter_mask(rot).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    sub = out[y0:y1]; sub[tm] = rot[tm]; out[y0:y1] = sub


# ---------- full reconstruction ----------
def reconstruct_dark_art(img):
    H, W = img.shape[:2]
    body = art_body(img)
    if body is None:
        return img.copy()
    dw, (ax0, ay0, ax1, ay1) = rectify(img, body)
    bands = text_bands(img, (ax0, ay0, ax1, ay1))

    # removal mask = art footprint + text, then inpaint the page background
    mask = np.zeros((H, W), np.uint8)
    mask[max(0, ay0-6):min(H, ay1+6), max(0, ax0-6):min(W, ax1+6)] = 255
    for (by0, by1) in bands:
        lm = _letter_mask(img[by0:by1]); mask[by0:by1][lm > 0] = 255
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8))
    bg = cv2.inpaint(img, mask, 7, cv2.INPAINT_TELEA)

    # lay bg first, then the dewarped art box centered horizontally (vertical kept)
    out = bg.copy()
    ix0, iy0, ix1, iy1 = ax0+5, ay0+5, ax1-5, ay1-5
    art = dw[iy0:iy1, ix0:ix1]; ah, aw = art.shape[:2]
    cx0 = (W - aw)//2
    out[iy0:iy0+ah, cx0:cx0+aw] = art

    # drop the text back in, deskewed
    for (by0, by1) in bands:
        _drop_text(out, img, by0, by1)
    return out


# ---------- magenta-box-driven dark reconstruction (manual override) ----------
def _box_edge_curves(img, box, dThr=28):
    """Tilted top/bottom edge curves of the art inside a magenta box, read from the
    bright KEYLINE across the FULL box width (works even where the art interior is
    near-black, since the white border is the topmost/bottommost bright pixel)."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = box
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    thr = page_lab(img)[0] + dThr
    cols = np.arange(x0, x1)
    tY = np.full(len(cols), np.nan); bY = np.full(len(cols), np.nan)
    for j, c in enumerate(cols):
        br = np.where(L[y0:y1, c] > thr)[0]
        if len(br):
            tY[j] = y0 + br[0]; bY[j] = y0 + br[-1]

    def fitc(vals):
        ok = ~np.isnan(vals)
        if ok.sum() < 8:
            return np.full(len(cols), np.nanmedian(vals))
        t = cols[ok].astype(float); p = vals[ok]
        c = np.polyfit(t, p, 3)
        for _ in range(5):
            r = p - np.polyval(c, t); k = np.abs(r) < 1.3*np.std(r) + 3
            if k.sum() < len(t)*0.4:
                break
            c = np.polyfit(t[k], p[k], 3)
        return np.polyval(c, cols.astype(float))
    return cols, fitc(tY), fitc(bY)


def _uniform_dark_page(img, boxes):
    """STEP 3 surface: one uniform dark texture for the whole page.
    Sample the real page ground (darkest margin pixels, away from art+text),
    take its mean colour + grain sigma, and synthesize a flat field of that
    colour with matching low-amplitude grain. No inpaint smears, no leftover art."""
    H, W = img.shape[:2]
    inbox = np.zeros((H, W), bool)
    for (x0, y0, x1, y1) in boxes:
        inbox[max(0, y0-20):min(H, y1+20), max(0, x0-20):min(W, x1+20)] = True
    L = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ground = (~inbox) & (L < 60)                       # dark page ground only
    samp = img[ground]
    if len(samp) < 5000:
        ground = (~inbox) & (L < np.percentile(L[~inbox], 25))
        samp = img[ground]
    mean = samp.reshape(-1, 3).mean(0)
    sig = float(np.clip(samp.reshape(-1, 3).std(0).mean(), 2.0, 7.0))
    bg = np.empty((H, W, 3), np.float32)
    bg[:] = mean
    rng = np.random.default_rng(H * 100000 + W)        # seeded -> reproducible page
    bg += rng.normal(0, sig, (H, W, 1))                # subtle uniform grain
    return np.clip(bg, 0, 255).astype(np.uint8), mean


def _baseline_angle(mask):
    """Skew angle of a text line from its BASELINE (per-column lowest letter pixel),
    robustly fit. The all-pixel slope is inflated by italics and uneven letter
    heights (it read ~7 deg on a ~2 deg line and tilted it); the baseline does not.
    Returns degrees, clamped to +/-8."""
    H, W = mask.shape
    cols, base = [], []
    for c in range(W):
        ys = np.where(mask[:, c])[0]
        if len(ys):
            cols.append(c); base.append(ys.max())
    if len(cols) < 30:
        return 0.0
    cols = np.array(cols, float); base = np.array(base, float)
    a, b = np.polyfit(cols, base, 1)
    for _ in range(4):
        r = base - (a * cols + b); k = np.abs(r) < 1.3 * np.std(r) + 2
        if k.sum() < len(cols) * 0.4:
            break
        a, b = np.polyfit(cols[k], base[k], 1)
    return float(np.clip(np.degrees(np.arctan(a)), -8, 8))


def _page_text_mask(img, boxes, g):
    """Mask of PAGE TEXT glyphs only. Text is bright-on-dark AND has high local edge
    density (strokes); a gloss blob or page sheen is bright but SMOOTH (no internal
    edges) so it is excluded no matter its size or fill. Everything inside/at the art
    boxes is excluded too. This keeps thin captions, the bold title and the underline
    rule, and drops gloss/sheen/edge-bleed -- anything that is not art-in-a-guide and
    not text becomes background.
    """
    H, W = img.shape[:2]
    L = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    edge = (cv2.magnitude(gx, gy) > 40).astype(np.float32)
    eden = cv2.boxFilter(edge, -1, (15, 15))         # local stroke-edge density
    bright = L > (g + 38)
    keep = (bright & (eden > 0.10)).astype(np.uint8)
    for (x0, y0, x1, y1) in boxes:                    # drop ONLY the exact art footprint
        keep[max(0, y0):min(H, y1), max(0, x0):min(W, x1)] = 0   # (captions just below a
                                                                 #  box stay -- guides give
                                                                 #  geometry, never gate text)
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # drop residual specks
    n, lab, st, _ = cv2.connectedComponentsWithStats(keep, 8)
    out = np.zeros((H, W), np.uint8)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= 8:
            out[lab == i] = 1
    return out


def _straighten_and_drop_text(bg, img, boxes, ground_mean):
    """Straighten each text BLOCK by its own baseline and drop it on the uniform bg.

    Text is segmented into blocks that are connected horizontally within a line but
    NOT across wide gaps, so two captions sitting side-by-side at the same height are
    straightened INDEPENDENTLY (a shared row-band fit was averaging them into a tilt).
    Each block is gated to its glyph mask (gloss/sheen/art excluded), rotated on the
    full canvas about its own centroid (no clip), blended by brightness (thin, whole
    letters; no blobs; one copy -> no ghost). Real scanned glyphs, only straightened.
    """
    H, W = img.shape[:2]
    g = float(np.mean(ground_mean))
    lo, hi = g + 16.0, g + 95.0
    L = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    tmask = _page_text_mask(img, boxes, g)
    # bridge letters within a line, but keep a big horizontal gap as a block boundary
    conn = cv2.dilate(tmask, cv2.getStructuringElement(cv2.MORPH_RECT, (35, 3)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(conn, 8)
    out = bg.copy()
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < 120:
            continue
        comp = (lab == i)
        glyph = (tmask > 0) & comp
        ys_, xs_ = np.where(glyph)
        if len(xs_) < 25:
            continue
        ang = _baseline_angle(glyph)
        cx, cy = float(xs_.mean()), float(ys_.mean())
        region = cv2.dilate(glyph.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2)
        region = cv2.GaussianBlur(region.astype(np.float32), (0, 0), 1.4)
        alpha = np.clip((L - lo) / (hi - lo), 0, 1) * np.clip(region, 0, 1)
        M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
        ar = cv2.warpAffine(alpha, M, (W, H), flags=cv2.INTER_LINEAR)
        cr = cv2.warpAffine(img,   M, (W, H), flags=cv2.INTER_CUBIC)
        y0, y1 = int(ys_.min()) - 30, int(ys_.max()) + 30
        ya, yb = max(0, y0), min(H, y1)
        a = np.clip(ar[ya:yb], 0, 1)[..., None]
        dst = out[ya:yb].astype(np.float32)
        out[ya:yb] = np.clip(dst * (1 - a) + cr[ya:yb].astype(np.float32) * a, 0, 255).astype(np.uint8)
    return out


def reconstruct_dark_art_with_boxes(img, boxes):
    """Dark (KEEP-BLACK) magenta-box reconstruction, in strict order:
      1. take ALL page text out
      2. straighten each art plate by its magenta guide (rectify to axis-aligned)
      3. lay down ONE uniform dark page (flat ground colour + matching grain)
      4. drop the straightened plates onto it
      5. straighten each text line by the baseline it traces and drop it back,
         thin (no dilation), one copy (no ghost), at its own row (no overlap)
    Falls back to auto if no boxes are given.
    """
    H, W = img.shape[:2]
    if not boxes:
        return reconstruct_dark_art(img)

    # 1 + 2: rectify each plate from its magenta box (text removed implicitly by
    #         rebuilding the page from a clean dark field below).
    dewarped = []
    for box in boxes:
        x0, y0, x1, y1 = box
        cols, topC, botC = _box_edge_curves(img, box)
        T = float(np.min(topC)); Bm = float(np.max(botC))
        mapx = np.tile(np.arange(W, dtype=np.float32), (H, 1))
        mapy = np.tile(np.arange(H, dtype=np.float32)[:, None], (1, W))
        yy = np.arange(int(T), int(Bm) + 1, dtype=np.float32)
        for j, c in enumerate(cols):
            t, bm = topC[j], max(botC[j], topC[j] + 10)
            mapy[int(T):int(Bm) + 1, c] = t + (yy - T) * (bm - t) / (Bm - T)
        dw = cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        dewarped.append((dw, (x0, int(T), x1, int(Bm))))

    # 3: uniform dark page
    out, ground_mean = _uniform_dark_page(img, boxes)

    # 4: drop the straightened plates
    single = len(dewarped) == 1
    for dw, (ax0, ay0, ax1, ay1) in dewarped:
        ix0, iy0, ix1, iy1 = ax0 + 5, ay0 + 5, ax1 - 5, ay1 - 5
        art = dw[iy0:iy1, ix0:ix1]; ah, aw = art.shape[:2]
        if single:
            cx0 = (W - aw) // 2
            out[iy0:iy0 + ah, cx0:cx0 + aw] = art
        else:
            out[iy0:iy0 + ah, ix0:ix0 + aw] = art

    # 5: straighten the text and drop it in, no ghost, no overlap
    out = _straighten_and_drop_text(out, img, boxes, ground_mean)
    return out

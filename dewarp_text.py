"""
textpage / dewarp_text — text-page treatment for scanned pages.

Two operations, both keep/relocate original pixels (no OCR):

  dewarp(img)            curl correction. Detects text rows in vertical bands,
                         tracks each row's y(x) curve, builds a smooth vertical
                         displacement field that flattens every baseline, remaps.
                         Multi-column safe (bands stay inside a column; field
                         interpolates across gutters). Geometry only.

  flatfield_whiten(img)  illumination correction. Estimates the paper surface
                         from PAPER ONLY (text + photos/art masked + inpainted),
                         divides it out per channel so paper -> uniform near-white
                         and the vignette/colour-cast is removed. Large dark
                         regions (photos, art) get a mild uniform gain, not blown
                         out.

  process_text_page(img) dewarp THEN whiten — the full text-page treatment used
                         by the TEXT branch of the reconstruction pipeline.
"""
import cv2, numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

# -- dewarp ---------------------------------------------------------------------

def _flatfield_gray(gray, sigma=120):
    bg = cv2.GaussianBlur(gray, (0, 0), sigma)
    flat = (gray.astype(np.float32) / (bg.astype(np.float32) + 1e-3)) * 255.0
    return np.clip(flat, 0, 255).astype(np.uint8)

def _dewarp_art_mask(img_bgr):
    """Mask of PICTURE regions (colour plates / photos) on a text page, so the curl
    field is fit to REAL text baselines only and pictures are not warped into waves.
    Pictures are DENSELY covered by non-paper ink (coverage ~0.7-1.0); text is SPARSE
    (~0.15-0.35 -- strokes with paper between), so a local non-paper COVERAGE map cleanly
    separates them without swallowing paragraphs. Large dense blobs -> filled bbox."""
    H, W = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    coarse = cv2.GaussianBlur(gray, (0, 0), 9)
    nonpaper = ((S > 45) | (gray < coarse.astype(np.float32) * 0.80)).astype(np.float32)
    k = max(31, (min(H, W) // 20) | 1)
    density = cv2.boxFilter(nonpaper, -1, (k, k))          # local non-paper coverage
    picture = (density > 0.55).astype(np.uint8)            # dense = plate/photo, not text
    picture = cv2.morphologyEx(picture, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(picture, 8)
    art = np.zeros((H, W), np.uint8)
    minside = 0.06 * min(H, W)
    for i in range(1, n):
        x = st[i, cv2.CC_STAT_LEFT]; y = st[i, cv2.CC_STAT_TOP]
        w = st[i, cv2.CC_STAT_WIDTH]; h = st[i, cv2.CC_STAT_HEIGHT]; a = st[i, cv2.CC_STAT_AREA]
        if a > 0.010 * H * W and min(w, h) > minside:       # a real picture
            art[y:y+h, x:x+w] = 1                           # FILL the plate's bounding box
    return cv2.dilate(art, np.ones((13, 13), np.uint8)).astype(bool)


def estimate_field(img_bgr, N=56, min_track_bands=4, line_min_dist=24,
                   prominence=8, smooth_sigma=(60, 40)):
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    dark = 255 - _flatfield_gray(gray)
    dark[_dewarp_art_mask(img_bgr)] = 0            # fit baselines to TEXT only, not plate texture
    bw = W / N
    xs, band_peaks = [], []
    for i in range(N):
        x0, x1 = int(i * bw), int((i + 1) * bw)
        xs.append((x0 + x1) / 2.0)
        prof = gaussian_filter(dark[:, x0:x1].mean(axis=1).astype(np.float32), 3)
        thr = prof.mean() + 0.4 * prof.std()
        pk, _ = find_peaks(prof, distance=line_min_dist, prominence=prominence, height=thr)
        band_peaks.append(pk.astype(np.float32))

    tol = line_min_dist * 0.7
    tracks, active = [], []
    for i, pk in enumerate(band_peaks):
        used = [False] * len(pk)
        new_active = []
        for tr, last_y in active:
            if len(pk) == 0:
                continue
            d = np.abs(pk - last_y); j = int(np.argmin(d))
            if not used[j] and d[j] <= tol:
                tr.append((i, float(pk[j]))); used[j] = True
                new_active.append((tr, float(pk[j])))
        for j, p in enumerate(pk):
            if not used[j]:
                tr = [(i, float(p))]; tracks.append(tr); new_active.append((tr, float(p)))
        active = new_active

    pts, dys = [], []
    for tr in tracks:
        if len(tr) < min_track_bands:
            continue
        ys = np.array([y for _, y in tr]); target = np.median(ys)
        for (bi, y) in tr:
            pts.append((xs[bi], y)); dys.append(target - y)
    if len(pts) < 10:
        return np.zeros((H, W), np.float32), 0
    # ZERO-ANCHORS: pin the displacement to 0 in PICTURE interiors and along the page
    # border, so the curl field (fit to text baselines) does NOT extrapolate across an
    # embedded plate and warp it into waves. Text keeps its correction; art stays rigid.
    art = _dewarp_art_mask(img_bgr)
    art_core = cv2.erode(art.astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool)
    step = max(40, min(H, W) // 24)
    for yy in range(0, H, step):
        for xx in range(0, W, step):
            if art_core[yy, xx]:
                pts.append((xx, yy)); dys.append(0.0)
    for xx in range(0, W, step):                      # top/bottom border -> 0
        pts.append((xx, 0)); dys.append(0.0); pts.append((xx, H - 1)); dys.append(0.0)
    for yy in range(0, H, step):                      # left/right border -> 0
        pts.append((0, yy)); dys.append(0.0); pts.append((W - 1, yy)); dys.append(0.0)
    pts = np.array(pts); dys = np.array(dys)
    gx, gy = np.meshgrid(np.arange(W), np.arange(H))
    D = griddata(pts, dys, (gx, gy), method='linear')
    Dn = griddata(pts, dys, (gx, gy), method='nearest')
    D[np.isnan(D)] = Dn[np.isnan(D)]
    return gaussian_filter(D.astype(np.float32), smooth_sigma), len(pts)

def dewarp(img_bgr, **kw):
    H, W = img_bgr.shape[:2]
    D, _ = estimate_field(img_bgr, **kw)
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    return cv2.remap(img_bgr, gx, (gy - D).astype(np.float32),
                     interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# -- flat-field whitening -------------------------------------------------------

def _picture_region_mask(gray, sat):
    """Whole PICTURE regions (colour plate / photo), by non-paper COVERAGE density and
    filled bounding boxes -- including UNIFORMLY DARK interiors (a dark painting/photo),
    which a local-contrast test misses. Used so flatfield_whiten inpaints the whole plate
    to a uniform paper level and gives it a mild uniform gain, instead of dividing a tiny
    dark background out and blowing the plate into red/yellow noise."""
    H, W = gray.shape
    coarse = cv2.GaussianBlur(gray, (0, 0), max(3, min(H, W) // 120))
    nonpaper = ((sat > 45) | (gray < coarse * 0.80)).astype(np.float32)
    k = max(9, (min(H, W) // 20) | 1)
    dens = cv2.boxFilter(nonpaper, -1, (k, k))
    pic = cv2.morphologyEx((dens > 0.50).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(pic, 8)
    out = np.zeros((H, W), np.uint8)
    minside = 0.05 * min(H, W)
    for i in range(1, n):
        x, y, w, h, a = st[i, :5]
        if a > 0.006 * H * W and min(w, h) > minside:
            out[y:y+h, x:x+w] = 1
    return out


def _content_mask(gray, sat, scale_big=0.06):
    H, W = gray.shape
    big = max(15, int(min(H, W) * scale_big) | 1)
    coarse = cv2.GaussianBlur(gray, (big, big), 0)
    dark = gray < (coarse.astype(np.float32) * 0.86)   # text / photo edges
    coloured = sat > 45                                # saturated art
    m = (dark | coloured).astype(np.uint8) * 255
    m = np.maximum(m, _picture_region_mask(gray, sat) * 255)  # whole plates, incl. dark interiors
    return cv2.dilate(m, np.ones((7, 7), np.uint8), 1)

def flatfield_whiten(img_bgr, target=246, work_w=900):
    H, W = img_bgr.shape[:2]
    small = cv2.resize(img_bgr, (work_w, int(H * work_w / W)), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sat = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 1]
    mask = _content_mask(g, sat)
    blur = max(15, int(small.shape[1] * 0.12) | 1)
    B = np.zeros_like(small, np.float32)
    for c in range(3):
        filled = cv2.inpaint(small[:, :, c], mask, 9, cv2.INPAINT_TELEA)
        B[:, :, c] = cv2.GaussianBlur(filled.astype(np.float32), (blur, blur), 0)
    B = np.clip(cv2.resize(B, (W, H), interpolation=cv2.INTER_LINEAR), 8, None)
    gain = np.clip(target / B, None, 2.2)              # cap gain -> no dark-region blowout
    out = img_bgr.astype(np.float32) * gain
    return np.clip(out, 0, 255).astype(np.uint8)

# -- full treatment -------------------------------------------------------------

def process_text_page(img_bgr, whiten=True):
    out = dewarp(img_bgr)
    if whiten:
        out = flatfield_whiten(out)
    return out

if __name__ == '__main__':
    import sys, os
    # Usage: python3 dewarp_text.py <input.jpg> [<input.jpg> ...]
    # Writes <name>_text.jpg next to each input.
    for path in sys.argv[1:]:
        im = cv2.imread(path)
        stem = os.path.splitext(path)[0]
        cv2.imwrite(f'{stem}_text.jpg', process_text_page(im),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(path, 'done')


def _track_baselines(crop):
    """Track text baselines across N vertical bands. Returns (xs, tracks, N)."""
    H, W = crop.shape[:2]
    N = int(min(44, max(12, W // 28)))
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    dark = 255 - _flatfield_gray(gray)
    bw = W / N
    xs, band_peaks = [], []
    for i in range(N):
        x0, x1 = int(i * bw), int((i + 1) * bw)
        xs.append((x0 + x1) / 2.0)
        prof = gaussian_filter(dark[:, x0:x1].mean(axis=1).astype(np.float32), 3)
        pk, _ = find_peaks(prof, distance=18, prominence=5,
                           height=prof.mean() + 0.3 * prof.std())
        band_peaks.append(pk.astype(np.float32))
    tol = 18 * 0.8
    tracks, active = [], []
    for i, pk in enumerate(band_peaks):
        used = [False] * len(pk); new_active = []
        for tr, last in active:
            if len(pk) == 0:
                continue
            d = np.abs(pk - last); j = int(np.argmin(d))
            if not used[j] and d[j] <= tol:
                tr.append((i, float(pk[j]))); used[j] = True
                new_active.append((tr, float(pk[j])))
        for j, p in enumerate(pk):
            if not used[j]:
                tr = [(i, float(p))]; tracks.append(tr); new_active.append((tr, float(p)))
        active = new_active
    return xs, tracks, N


def _baseline_tilt_deg(tracks, xs):
    """Median tilt (deg) of the tracked baselines -- for the single-angle deskew."""
    slopes = []
    for tr in tracks:
        if len(tr) < 3:
            continue
        X = np.array([xs[bi] for bi, _ in tr], np.float32)
        Y = np.array([y for _, y in tr], np.float32)
        if X.max() - X.min() < 1:
            continue
        slopes.append(np.polyfit(X, Y, 1)[0])
    if not slopes:
        return 0.0
    return float(np.degrees(np.arctan(np.median(slopes))))


def _flatness(tracks, N):
    """Mean vertical spread of tracked baselines about their own median (px). 0 = every
    line dead horizontal. Captures BOTH tilt and curl (a tilted line has high spread
    about its median too), so smaller = flatter. Inclusive threshold so a 1-2 line
    heading's tilt is still measurable (lets the deskew fallback be self-verified)."""
    need = max(3, N // 3)
    devs = [float(np.std([y for _, y in tr])) for tr in tracks if len(tr) >= need]
    return float(np.mean(devs)) if devs else 0.0


def dewarp_text_area(crop, smooth=(60, 30), min_field_lines=4, span_frac=0.55,
                     max_disp_lh=1.0, min_deskew=0.3, max_deskew=3.0):
    """Straighten one text area with a deliberately conservative policy:

      * A curl FIELD is built ONLY when >= min_field_lines baselines are each tracked
        across >= span_frac of the block width (reliable, full-span lines). The field
        flattens each baseline to its median y, is CLAMPED to +/- max_disp_lh
        line-heights, heavily smoothed, and ANCHORED to zero along the crop's top and
        bottom edges so it can never ripple art sitting just above/below the text.
      * Otherwise (few lines, short/split tracks, captions, headings) it falls back to
        a SINGLE-ANGLE deskew: the whole block is rotated by one angle, so every line
        stays straight and parallel -- a line can never be bent into a V by how the
        block was split. Applied only for a real tilt (min_deskew..max_deskew deg);
        below/above that the block is left exactly as scanned.

    Returns (result, max_disp_px, mode) where mode is 'field' | 'deskew' | 'none'.
    """
    H, W = crop.shape[:2]
    xs, tracks, N = _track_baselines(crop)

    # reliable = full-span baselines only; a split half-line (< span_frac) is excluded
    # so it can never pull one end of a line to a different level (no V / kink).
    need = max(4, int(span_frac * N))
    reliable = [tr for tr in tracks if len(tr) >= need]

    if len(reliable) >= 2:
        mids = sorted(float(np.median([y for _, y in tr])) for tr in reliable)
        gaps = np.diff(mids)
        lh = float(np.median(gaps)) if len(gaps) else 40.0
    else:
        lh = 40.0
    clamp = max(6.0, max_disp_lh * lh)

    if len(reliable) >= min_field_lines:
        pts, dys = [], []
        for tr in reliable:
            ys = np.array([y for _, y in tr]); target = float(np.median(ys))
            for (bi, y) in tr:
                pts.append((xs[bi], y))
                dys.append(float(np.clip(target - y, -clamp, clamp)))
        step = max(20, W // 24)
        for xx in range(0, W, step):                    # pin top & bottom edges to 0
            pts.append((xx, 0)); dys.append(0.0)
            pts.append((xx, H - 1)); dys.append(0.0)
        pts = np.array(pts); dys = np.array(dys)
        gx, gy = np.meshgrid(np.arange(W), np.arange(H))
        D = griddata(pts, dys, (gx, gy), method='linear')
        Dn = griddata(pts, dys, (gx, gy), method='nearest')
        D[np.isnan(D)] = Dn[np.isnan(D)]
        D = np.clip(D, -clamp, clamp)
        D = gaussian_filter(D.astype(np.float32), smooth)
        cand = cv2.remap(crop, gx.astype(np.float32), (gy - D).astype(np.float32),
                         interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        mag, mode = float(np.abs(D).max()), 'field'
    else:
        ang = _baseline_tilt_deg(tracks, xs)           # single-angle deskew (no V)
        if min_deskew <= abs(ang) <= max_deskew:
            M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), -ang, 1.0)
            cand = cv2.warpAffine(crop, M, (W, H), flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
            mag, mode = float(abs(np.sin(np.radians(ang))) * W / 2.0), 'deskew'
        else:
            cand, mag, mode = crop, 0.0, 'none'

    if mode == 'none':
        return crop, 0.0, 'none'

    # SELF-VERIFY: keep the correction ONLY if it measurably flattened the baselines.
    # A wrong sign, a mis-track, or a sparse block can never make text worse -- it is
    # left exactly as scanned instead.
    before = _flatness(tracks, N)
    _, tracks2, N2 = _track_baselines(cand)
    after = _flatness(tracks2, N2)
    if before > 1.0 and after < 0.9 * before:
        return cand, mag, mode
    return crop, 0.0, 'none'

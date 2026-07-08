"""
textpage / dewarp_text — text-page treatment for scanned pages.

v64 — text-dewarp refinement (per-3-line local treatment):
  * _track_baselines is now PITCH-AWARE (one peak per line); the old fixed
    distance=18 locked onto sub-line features on high-res scans, halving the apparent
    pitch and wrecking every downstream fit.
  * dewarp_text_area now removes the LOW-FREQUENCY curl in groups of `band` (=3)
    consecutive lines: each band pools its lines and fits ONE shared trend, so per-line
    peak jitter cancels and only real local tilt/bow is corrected. Bands are feathered
    (no seam); each line keeps its own median (spacing preserved). Accepted only if the
    curl metric drops without adding jitter -> never warps text to tracking noise.
  * Sparse but clearly-skewed blocks (e.g. the copyright/credits panel) get a
    SIGN-ROBUST single-angle deskew (tries both rotation signs, keeps the flatter),
    fixing tilts the old deskew rotated the wrong way and then self-rejected.

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


def _estimate_pitch(dark, default=40.0):
    """Line pitch (px) from the whole-crop darkness profile -- the median gap between
    row peaks. Used to tune peak spacing so we lock ONE peak per text line instead of
    sub-line features (serif / x-height structure), which on high-res scans otherwise
    doubles the track count and halves the apparent pitch, wrecking the curl fit."""
    prof = gaussian_filter(dark.mean(axis=1).astype(np.float32), 3)
    pk, _ = find_peaks(prof, distance=20, prominence=5,
                       height=prof.mean() + 0.25 * prof.std())
    if len(pk) >= 3:
        return float(np.clip(np.median(np.diff(pk)), 14.0, 200.0))
    return default


def _track_baselines(crop):
    """Track text baselines across N vertical bands, ONE peak per line. Returns
    (xs, tracks, N). Peak spacing and match tolerance are tied to the estimated line
    pitch so serifed / small-pitch scans still yield one baseline per line."""
    H, W = crop.shape[:2]
    N = int(min(44, max(12, W // 28)))
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    dark = 255 - _flatfield_gray(gray)
    pitch = _estimate_pitch(dark)
    dist = max(14, int(pitch * 0.6))
    tol = max(12.0, pitch * 0.45)
    bw = W / N
    xs, band_peaks = [], []
    for i in range(N):
        x0, x1 = int(i * bw), int((i + 1) * bw)
        xs.append((x0 + x1) / 2.0)
        prof = gaussian_filter(dark[:, x0:x1].mean(axis=1).astype(np.float32), 3)
        pk, _ = find_peaks(prof, distance=dist, prominence=5,
                           height=prof.mean() + 0.3 * prof.std())
        band_peaks.append(pk.astype(np.float32))
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


def _dewarp_text_area_legacy(crop, smooth=(60, 30), min_field_lines=4, span_frac=0.55,
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


# -- banded (every-N-lines) local dewarp -----------------------------------------
# Rationale: one smoothed field over a whole column, pinned to 0 at the crop's top
# and bottom, under-corrects -- residual baseline bow ~10 px survives, worst at the
# top and bottom lines (exactly where page/gutter curl is strongest). Instead we
# treat the column in small groups of `band` consecutive baselines: each group is
# flattened to ITS OWN local baselines (so tilt AND curl are followed piecewise, not
# averaged away), groups are blended with a vertical feather so there is no seam, and
# the whole thing is kept only if it measurably flattens the block (never worse).

def _baseline_curves(xs, tracks, min_span_bands=5, poly_deg=2, min_sep=0.0):
    """One smooth curve y(x) per REAL text line (flat-extrapolated outside its tracked
    x-span), with a robust residual-rejection pass so a stray peak can't tilt a line.
    Fragment tracks are rejected (span < min_span_bands) and near-duplicate baselines
    within `min_sep` px (two fragments of one line) are merged to the longer-tracked
    one, so grouping every N lines really means N *lines*. Returns list top->bottom."""
    cand = []
    for tr in tracks:
        if len(tr) < min_span_bands:
            continue
        X = np.array([xs[b] for b, _ in tr], float)
        Y = np.array([y for _, y in tr], float)
        o = np.argsort(X); X, Y = X[o], Y[o]
        deg = poly_deg if (X.max() - X.min() > 1 and len(X) >= poly_deg + 2) else 1
        p = np.polyfit(X, Y, deg)
        r = Y - np.polyval(p, X); s = float(np.std(r)) + 1e-6
        keep = np.abs(r) < 2.5 * s
        if keep.sum() >= deg + 2:
            p = np.polyfit(X[keep], Y[keep], deg)
        cand.append({'xr': (float(X.min()), float(X.max())), 'poly': p,
                     'target': float(np.median(Y)), 'ymid': float(np.median(Y)),
                     'span': len(tr)})
    cand.sort(key=lambda c: c['ymid'])
    if min_sep > 0:                       # merge fragments of the same physical line
        merged = []
        for c in cand:
            if merged and c['ymid'] - merged[-1]['ymid'] < min_sep:
                if c['span'] > merged[-1]['span']:
                    merged[-1] = c       # keep the longer-tracked fragment
            else:
                merged.append(c)
        cand = merged
    return cand


def _curve_y(c, xq):
    x0, x1 = c['xr']
    return np.polyval(c['poly'], np.clip(xq, x0, x1))   # flat outside tracked span


def _group_indices(m, band):
    """Consecutive groups of `band`; a trailing remainder smaller than `band` is
    absorbed into the previous group (never an orphan sliver)."""
    groups, i = [], 0
    while i < m:
        if 0 < m - (i + band) < band:      # remainder too small -> take the rest now
            groups.append(list(range(i, m))); break
        groups.append(list(range(i, min(i + band, m)))); i += band
    return groups


def _band_weight(rows, top, bot, feather):
    """Trapezoid: 1 inside [top,bot], linear ramp to 0 across `feather` on each side."""
    w = np.ones_like(rows, np.float32)
    up = rows < top;  w[up] = np.clip(1.0 - (top - rows[up]) / feather, 0, 1)
    dn = rows > bot;  w[dn] = np.clip(1.0 - (rows[dn] - bot) / feather, 0, 1)
    return w


def _line_shape_and_stats(xs, tracks, N, W, min_span_frac=0.4):
    """Clean per-line baselines (one per line), each as (X, Y, median_y, tilt_deg,
    jitter_px). tilt = deg-1 slope; jitter = max deviation from that line's own straight
    fit (the high-freq peak-finder noise we must NOT chase). Sorted top->bottom."""
    need = max(6, N // 2)
    out = []
    for tr in tracks:
        if len(tr) < need:
            continue
        X = np.array([xs[i] for i, _ in tr], float)
        Y = np.array([y for _, y in tr], float)
        if X.max() - X.min() < min_span_frac * W:
            continue
        o = np.argsort(X); X, Y = X[o], Y[o]
        p = np.polyfit(X, Y, 1)
        tilt = float(np.degrees(np.arctan(p[0])))
        jit = float(np.max(np.abs(Y - np.polyval(p, X))))
        out.append([X, Y, float(np.median(Y)), tilt, jit])
    out.sort(key=lambda r: r[2])
    return out


def _curl_metric(lines):
    """Low-frequency curl of a column: spread of per-line tilt after 3-line smoothing
    (removes jitter, keeps the top/bottom bow). Returns (curl_std_deg, mean_jitter_px)."""
    if len(lines) < 3:
        return 0.0, 0.0
    tilts = np.array([r[3] for r in lines])
    k = 3
    sm = np.convolve(tilts, np.ones(k) / k, mode='valid')
    return float(np.std(sm)), float(np.mean([r[4] for r in lines]))


def dewarp_text_area_banded(crop, band=3, max_disp_lh=1.2, max_band_deg=2.5,
                            min_baselines=4):
    """Remove the LOW-FREQUENCY curl of a text column using a SLIDING window of `band`
    (=3) consecutive lines: each line's correction is the single shared trend g(x) fit
    to itself + its neighbours (pooled, de-medianed) -- so per-line peak-finder jitter
    cancels (3 lines agree on the real local tilt/bow) while EVERY line still gets its
    own control profile, keeping the field's vertical resolution fine. The field is NOT
    pinned to zero at the crop's top/bottom (that was what left residual bow on the first
    and last lines) -- it flat-extrapolates, so the extreme lines are fully corrected.
    Line spacing is preserved (each line keeps its own median). Kept only if the curl
    metric drops WITHOUT adding jitter (never warps text to noise). Falls back to the
    legacy field/deskew for sparse blocks. Returns (result, max_disp_px, mode)."""
    H, W = crop.shape[:2]
    xs, tracks, N = _track_baselines(crop)
    lines = _line_shape_and_stats(xs, tracks, N, W)
    m = len(lines)
    if m < max(3, min_baselines):
        return _dewarp_text_area_legacy(crop)

    mids = np.array([r[2] for r in lines], float)
    # Refuse a block that is NOT a clean single column: if the typical tracked line
    # spans well under the block width, the block is multi-column (find_blocks can merge
    # side-by-side columns) or ragged. Warping across a gutter pools baselines from
    # different columns into one 'band' and corrupts them -- leave it for the per-column
    # blocks that find_blocks also emits. (Genuine tilt on such a block is still caught
    # by the sign-robust deskew in the entry wrapper.)
    widths = np.array([r[0].max() - r[0].min() for r in lines], float)
    if float(np.median(widths)) < 0.62 * W:
        return crop, 0.0, 'none'
    lh = max(8.0, float(np.median(np.diff(mids))))
    clamp = min(max(6.0, max_disp_lh * lh), 24.0)   # absolute cap: real text curl/tilt is small
    half = band // 2
    xq = np.linspace(0, W - 1, min(W, 140))

    def shared_trend(i):
        """Shared -g(x) for the window centred on line i (jitter-cancelled)."""
        lo, hi = max(0, i - half), min(m, i + half + 1)
        idxs = list(range(lo, hi))
        # only pool neighbours that are actually adjacent (no paragraph gap jump)
        idxs = [j for j in idxs if abs(mids[j] - mids[i]) <= 1.8 * lh * band]
        Xp = np.concatenate([lines[j][0] for j in idxs])
        Yp = np.concatenate([lines[j][1] - lines[j][2] for j in idxs])
        if len(Xp) < 4 or Xp.max() - Xp.min() < 1:
            return np.zeros_like(xq)
        p1 = np.polyfit(Xp, Yp, 1); r1 = np.std(Yp - np.polyval(p1, Xp))
        p = p1
        if len(Xp) >= 6:
            p2 = np.polyfit(Xp, Yp, 2); r2 = np.std(Yp - np.polyval(p2, Xp))
            if r2 < 0.8 * r1:
                p = p2
        g = np.polyval(p, xq)
        ang = np.degrees(np.arctan((g[-1] - g[0]) / max(1.0, xq[-1] - xq[0])))
        if abs(ang) > max_band_deg:
            g = g * (max_band_deg / abs(ang))
        return np.clip(-g, -clamp, clamp)

    prof = np.vstack([shared_trend(i) for i in range(m)])       # (m, len(xq))
    ext = np.concatenate(([-1e9], mids, [1e9]))                 # flat-extrapolate edges
    rows = np.arange(H)
    coarse = np.empty((H, len(xq)), np.float32)
    for j in range(len(xq)):
        col = np.concatenate(([prof[0, j]], prof[:, j], [prof[-1, j]]))
        coarse[:, j] = np.interp(rows, ext, col)
    D = coarse[:, np.linspace(0, len(xq) - 1, W).astype(int)]
    D = gaussian_filter(D, (max(3.0, lh * 0.15), max(6.0, W / 50.0)))
    D = np.clip(D, -clamp, clamp).astype(np.float32)

    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    cand = cv2.remap(crop, gx, (gy - D).astype(np.float32),
                     interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    c0, j0 = _curl_metric(lines)
    t0 = abs(_baseline_tilt_deg(tracks, xs))
    xs2, tr2, N2 = _track_baselines(cand)
    c1, j1 = _curl_metric(_line_shape_and_stats(xs2, tr2, N2, W))
    t1 = abs(_baseline_tilt_deg(tr2, xs2))
    # keep only if curl dropped, jitter did not rise, AND the block's own tilt did not
    # grow (a short block at a page extreme can be handed a spurious shared trend that
    # curls flatter but tilts more -- reject that outright: never worse).
    if (c0 > 0.12 and c1 < 0.85 * c0 and j1 <= 1.06 * max(j0, 1e-6)
            and t1 <= t0 + 0.15):
        return cand, float(np.abs(D).max()), 'banded'
    return crop, 0.0, 'none'



def _consistent_tilt(crop):
    """Median tilt (deg) of reliable, wide-spanning baselines, plus whether they AGREE.
    Returns (median_tilt, n_lines, consistent). A real page skew makes most baselines
    tilt the same way by a similar amount; a spurious estimate (short fragment of a
    merged multi-column region, padding bleed) has mixed signs / big spread -> not
    consistent -> not deskewed. This is what separates a genuinely-skewed copyright
    panel from an unreliable 6-line fragment."""
    xs, tracks, N = _track_baselines(crop)
    H, W = crop.shape[:2]
    tl = []
    for tr in tracks:
        if len(tr) < max(3, N // 4):
            continue
        X = np.array([xs[i] for i, _ in tr], float)
        Y = np.array([y for _, y in tr], float)
        if X.max() - X.min() < 0.35 * W:
            continue
        tl.append(np.degrees(np.arctan(np.polyfit(X, Y, 1)[0])))
    if len(tl) < 3:
        return 0.0, len(tl), False
    tl = np.array(tl); med = float(np.median(tl))
    same = float(np.mean(np.sign(tl) == np.sign(med)))
    mad = float(np.median(np.abs(tl - med)))
    ok = same >= 0.70 and mad <= max(0.35, 0.6 * abs(med))
    return med, len(tl), ok


def _robust_deskew(crop, min_deg=0.3, max_deg=4.0):
    """Single-angle deskew for a sparse but CONSISTENTLY-skewed block (e.g. a
    copyright/credits panel). The consistency test up front (baselines agree on sign and
    magnitude) is the guarantee: rotating by their median tilt then flattens them
    (per-line residual = tilt - median, whose median is 0). No post-rotation re-track --
    that measurement is unreliable on short blocks and was the source of wrong-sign
    over-rotations. If the block's baselines disagree (a merged multi-column region or
    padding bleed), it is left untouched. Returns (result, disp_px, applied)."""
    H, W = crop.shape[:2]
    med, n, ok = _consistent_tilt(crop)
    if not ok or abs(med) < min_deg or abs(med) > max_deg:
        return crop, 0.0, False
    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), med, 1.0)
    out = cv2.warpAffine(crop, M, (W, H), flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_REPLICATE)
    return out, float(abs(np.sin(np.radians(med))) * W / 2.0), True


def _single_column_ok(crop):
    """False only if the block is clearly MULTI-COLUMN: enough lines to judge, each
    spanning well under the block width (side-by-side columns that find_blocks merged).
    Warping or deskewing such a block as one corrupts it (baselines from different
    columns pooled together, a single rotation applied across a gutter). Few-line blocks
    always pass -- they are judged by the deskew's own consistency test."""
    xs, tracks, N = _track_baselines(crop)
    W = crop.shape[1]
    ws = [max(xs[i] for i, _ in tr) - min(xs[i] for i, _ in tr)
          for tr in tracks if len(tr) >= max(3, N // 4)]
    if len(ws) < 6:
        return True
    return float(np.median(ws)) / W >= 0.60


def dewarp_text_area(crop, band=3, **kw):
    """Entry point used by text_blocks.dewarp_text_areas. Multi-column blocks are left
    untouched (the per-column blocks find_blocks also emits are handled on their own);
    otherwise the banded low-freq curl corrector runs, and if it declines (sparse block)
    a consistency-gated deskew catches a clear tilt. `band` = lines/band (default 3)."""
    if not _single_column_ok(crop):
        return crop, 0.0, 'none'
    res, mag, mode = dewarp_text_area_banded(crop, band=band)
    if mode != 'none':
        return res, mag, mode
    d, disp, ok = _robust_deskew(crop)
    if ok:
        return d, disp, 'deskew'
    return crop, 0.0, 'none'

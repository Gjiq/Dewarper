"""
page_dewarp — PAGE-LEVEL curl correction, run on the original scan BEFORE any
picture is detected or cropped. Removes the smooth binding-curl bow so that the
existing per-picture deskew (which handles per-picture SLOPE) works on a flat page.

Robustness mirrors how _reliable_angle handles slope: never trace an edge column
by column. Instead fit each picture border to a low-order curve with a RESIDUAL
GATE -- an edge that is not a clean line (a treeline, a sky fading to paper) fails
the gate and is discarded. The surviving clean edges from several pictures, at
different heights, over-constrain ONE low-order page field [1,x,y,x^2,y^2,xy].
Too smooth to bow a straight edge; too over-constrained for one bad edge to move.
If fewer than `min_edges` clean borders survive, or the field is tiny, it no-ops.
"""
import cv2
import numpy as np
import deskew_crop as dk


def _edge_samples(gray, gy, x0, x1, ey, band, resid_thr=3.5, min_found_frac=0.5):
    """One picture border -> clean low-order samples, or None if it fails the gate."""
    H, W = gray.shape
    x1 = min(x1, W - 1)
    lo = max(0, ey - band); hi = min(H, ey + band + 1)
    sub = gy[lo:hi, x0:x1 + 1]
    ridx = np.argmax(sub, axis=0)
    strength = sub[ridx, np.arange(sub.shape[1])]
    ys = (lo + ridx).astype(np.float64)
    xs = np.arange(sub.shape[1]).astype(np.float64) + x0
    if xs.size < 30:
        return None
    # a column has a real border where its gradient is decisive vs this edge's peak
    found = strength >= 0.25 * strength.max()
    if found.mean() < min_found_frac:
        return None                                   # border too faint / patchy
    xk, yk = xs[found], ys[found]
    xc = (xk - xs.mean()) / (np.ptp(xs) + 1e-6)
    # robust quadratic fit + residual gate (slope-style "do I trust this edge")
    w = np.ones_like(xk)
    coef = None
    for _ in range(4):
        V = np.vander(xc, 3)
        coef, *_ = np.linalg.lstsq(V * w[:, None], yk * w, rcond=None)
        r = yk - V @ coef
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
        w = (np.abs(r / (4.685 * s)) < 1).astype(float)
    inl = w > 0
    resid = np.median(np.abs((yk - V @ coef)[inl]))
    if resid > resid_thr or inl.sum() < 0.5 * xk.size:
        return None                                   # not a clean straight border
    xcf = (xs - xs.mean()) / (np.ptp(xs) + 1e-6)
    fit = np.vander(xcf, 3) @ coef
    # Remove only CURVATURE (the bow). Each edge's slope is left in place for the
    # per-picture deskew (which now runs on SIDE_TEXT panels too). Target = the
    # straight line through the fitted curve; dy bends the curve onto that line.
    lin = np.polyval(np.polyfit(xcf, fit, 1), xcf)
    dy = lin - fit
    step = max(1, xs.size // 80)
    return xs[::step], fit[::step], dy[::step]


def estimate_field(orig, bg, min_edges=2):
    H, W = orig.shape[:2]
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    gy = np.abs(cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=5))
    band = max(16, int(0.014 * H))

    boxes = dk.art_pictures(orig, bg)
    X, Y, DY = [], [], []
    n_ok = 0; n_try = 0
    for (x0, y0, x1, y1) in boxes:
        if (x1 - x0) < 0.12 * W:
            continue
        for ey in (y0, y1):
            n_try += 1
            r = _edge_samples(gray, gy, x0, x1, ey, band)
            if r is None:
                continue
            xs, fit, dy = r
            X.append(xs); Y.append(fit); DY.append(dy); n_ok += 1
    if n_ok < min_edges:
        return None, {"edges_ok": n_ok, "edges_tried": n_try, "reason": "too few clean edges"}

    X = np.concatenate(X); Y = np.concatenate(Y); DY = np.concatenate(DY)
    xn = X / W; yn = Y / H
    # field order adapts to how many independent edges (y-levels) constrain it:
    # >=3 edges -> quadratic in y; exactly 2 -> linear in y (drop the y^2 term).
    cols = [np.ones_like(xn), xn, yn, xn * xn, xn * yn]
    if n_ok >= 3:
        cols.insert(4, yn * yn)
    Phi = np.stack(cols, 1)
    beta, *_ = np.linalg.lstsq(Phi, DY, rcond=None)

    gx, gyg = np.meshgrid(np.arange(W), np.arange(H))
    xg = gx / W; yg = gyg / H
    terms = [np.ones_like(xg), xg, yg, xg * xg, xg * yg]
    if n_ok >= 3:
        terms.insert(4, yg * yg)
    D = sum(b * t for b, t in zip(beta, terms)).astype(np.float32)
    return D, {"edges_ok": n_ok, "edges_tried": n_try,
               "field_range": [round(float(D.min()), 1), round(float(D.max()), 1)]}


def dewarp_page(orig, bg, min_field=4.0, max_field_frac=0.05):
    H, W = orig.shape[:2]
    D, info = estimate_field(orig, bg)
    if D is None:
        info["applied"] = False
        return orig, info
    fr = max(abs(float(D.min())), abs(float(D.max())))
    if fr < min_field:
        info["applied"] = False; info["reason"] = "flat"
        return orig, info
    # UPPER SANITY CAP: a real binding curl displaces a page by at most a few % of
    # its height (observed here <=31 px). A much larger field means the low-order
    # fit went degenerate (ill-conditioned edge set -> wild coefficients), and
    # applying it TEARS/duplicates the page (seen on p082: field range ~1765 px
    # duplicated the left painting). Reject it and no-op rather than warp.
    if fr > max_field_frac * H:
        info["applied"] = False
        info["reason"] = f"degenerate field ({fr:.0f}px > {int(max_field_frac*100)}% H) — skipped"
        return orig, info
    gx, gyg = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    out = cv2.remap(orig, gx, (gyg - D).astype(np.float32),
                    interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    info["applied"] = True
    return out, info

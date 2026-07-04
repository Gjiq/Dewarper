"""
magenta_text.py  — magenta-GUIDE handling for TEXT blocks, and open-box grouping
for art plates (Dewarp Pipeline v54).

Two jobs, both driven by the same magenta markup the artist draws on a
pixel-aligned copy of the clean scan:

  1. GROUP magenta strokes into REGIONS even when the box is OPEN or its corners
     don't meet. `magenta_regions()` returns, per region, whichever of the four
     sides were drawn (top / bottom / left / right) as polylines, plus the region
     bbox inferred from the sides that ARE present. This fixes the recurring
     "detached edge / open corner" case (a right edge drawn as its own stroke,
     or a box that just doesn't close) that `magenta_crop.magenta_boxes` — which
     needs a closed loop — silently drops.

  2. CORRECT a TEXT block to its guides. The artist marks a text column with a
     magenta guide that need NOT be a full box:
        - left-justified text  -> often NO right guide
        - right-justified text -> often NO left guide
        - any/all four sides may be present
     The TOP guide is NOT a crop boundary: it traces how the top row of text is
     skewed / warped, and the block is flattened so that guide becomes straight
     and level (using the BOTTOM guide too when present to carry the correction
     through the block). `correct_text_block()` does that, then whitens.

Routing text-vs-art is by CONTENT: classify the CLEAN-ORIGINAL pixels inside a
region (see `region_is_text`). A guide over prose -> text correction; a guide
over a plate -> the normal art dewarp. The magenta pixels only ever supply
geometry; not one of them reaches the output.
"""
import cv2
import numpy as np
import magenta_crop as mc


# ── region grouping ───────────────────────────────────────────────────────────

def _edge_segments(mask):
    """Split the magenta strokes into long HORIZONTAL and VERTICAL edge segments
    (a box's four sides), regardless of whether the box is one connected stroke
    or several. Directional opening isolates each orientation; CCs then give the
    individual segments with their extents."""
    H, W = mask.shape
    kh = max(41, int(0.05 * W)) | 1
    kv = max(41, int(0.05 * H)) | 1
    hor = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((1, kh), np.uint8))
    ver = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((kv, 1), np.uint8))
    segs = []
    for om, orient in ((hor, 'H'), (ver, 'V')):
        n, lab, st, cen = cv2.connectedComponentsWithStats(om, 8)
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] < 300:
                continue
            x, y = st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP]
            w, h = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
            segs.append({'o': orient, 'x0': int(x), 'y0': int(y),
                         'x1': int(x + w), 'y1': int(y + h),
                         'cx': float(cen[i][0]), 'cy': float(cen[i][1])})
    return segs


def _assemble(segs, s):
    """Union H and V edges that are two SIDES of the same box. An H (row hy, span
    [hx0,hx1]) and a V (col vx, span [vy0,vy1]) belong together if the H's x-span
    REACHES the V's column and the V's y-span REACHES the H's row (they'd meet at a
    corner even if the artist's corner is offset/open). Only H–V links, so the two
    parallel edges in a gutter between adjacent plates never merge — neighbours
    stay separate while a box's own (possibly open) sides group."""
    parent = list(range(len(segs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def compat(h, v):
        return (h['x0'] - s <= v['cx'] <= h['x1'] + s and
                v['y0'] - s <= h['cy'] <= v['y1'] + s)

    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            a, b = segs[i], segs[j]
            if a['o'] == b['o']:
                continue
            h, v = (a, b) if a['o'] == 'H' else (b, a)
            if compat(h, v):
                parent[find(i)] = find(j)
    groups = {}
    for i in range(len(segs)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _centerline(mask, seg):
    """Trace an edge segment's centreline as a polyline, so a curved/skewed guide
    keeps its shape. H -> one (x, mean-y) per column; V -> one (mean-x, y) per row."""
    x0, y0, x1, y1 = seg['x0'], seg['y0'], seg['x1'], seg['y1']
    sub = mask[y0:y1, x0:x1] > 0
    pts = []
    if seg['o'] == 'H':
        for c in range(sub.shape[1]):
            rows = np.where(sub[:, c])[0]
            if len(rows):
                pts.append((x0 + c, y0 + rows.mean()))
    else:
        for r in range(sub.shape[0]):
            cols = np.where(sub[r, :])[0]
            if len(cols):
                pts.append((x0 + cols.mean(), y0 + r))
    return np.array(pts, np.float64) if pts else None


def _group_lines(gs, mask):
    """All horizontal guide polylines (sorted top->bottom) and vertical polylines
    of a group. A text guide keeps EVERY horizontal (each is a warp reference at its
    depth), not just top/bottom."""
    hs = sorted((g for g in gs if g['o'] == 'H'), key=lambda g: g['cy'])
    vs = sorted((g for g in gs if g['o'] == 'V'), key=lambda g: g['cx'])
    hlines = [_centerline(mask, s) for s in hs]
    vlines = [_centerline(mask, s) for s in vs]
    hlines = [p for p in hlines if p is not None and len(p) >= 5]
    vlines = [p for p in vlines if p is not None and len(p) >= 5]
    return hlines, vlines


def _stack_horizontals(hsegs, min_overlap=0.35):
    """Union orphan horizontal segments (no vertical to bind them into a box) that
    share an x-column into a WARP-GUIDE STACK: several underlines drawn at
    increasing depth to trace how the text warp changes down the page. Only
    x-overlapping horizontals join, so two separate columns stay separate. Returns
    a list of segment-lists."""
    parent = list(range(len(hsegs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(hsegs)):
        for j in range(i + 1, len(hsegs)):
            a, b = hsegs[i], hsegs[j]
            ov = min(a['x1'], b['x1']) - max(a['x0'], b['x0'])
            if ov > min_overlap * min(a['x1'] - a['x0'], b['x1'] - b['x0']):
                parent[find(i)] = find(j)
    groups = {}
    for i in range(len(hsegs)):
        groups.setdefault(find(i), []).append(hsegs[i])
    return list(groups.values())


def magenta_regions(img, tol=None):
    """All magenta-marked regions, open or closed. Each is a dict:
        'bbox'   : (x0, y0, x1, y1)
        'hlines' : horizontal warp-guide polylines, sorted top->bottom (0..N)
        'vlines' : vertical bound polylines (left/right), if any
    Boxes (art plates, boxed text) come from span-reach assembly; a bare STACK of
    horizontal warp guides with no box (drawn underlines at increasing depth) is
    assembled by x-overlap. Adjacent plates stay separate; open boxes still group."""
    H, W = img.shape[:2]
    m = (mc.magenta_mask(img) > 0).astype(np.uint8) * 255
    if (m > 0).sum() < 2000:
        return []
    if tol is None:
        tol = max(20, int(0.010 * max(H, W)))         # span-reach slack
    segs = _edge_segments(m)
    box_groups = []
    orphan_h = []
    for grp in _assemble(segs, tol):
        gs = [segs[i] for i in grp]
        if any(s['o'] == 'V' for s in gs):
            box_groups.append(gs)                     # a real box (has a vertical side)
        else:
            orphan_h.extend(s for s in gs if s['o'] == 'H')
    final = list(box_groups) + _stack_horizontals(orphan_h)
    regions = []
    for gs in final:
        x0 = min(s['x0'] for s in gs); y0 = min(s['y0'] for s in gs)
        x1 = max(s['x1'] for s in gs); y1 = max(s['y1'] for s in gs)
        hlines, vlines = _group_lines(gs, m)
        if (x1 - x0) < 0.10 * W:
            continue
        if (y1 - y0) < 0.10 * H and len(hlines) < 2:   # a lone underline is not a block
            continue
        regions.append({'bbox': (x0, y0, x1, y1), 'hlines': hlines, 'vlines': vlines})
    regions.sort(key=lambda r: (round(r['bbox'][1] / (0.12 * H)), r['bbox'][0]))
    return regions


# ── text vs art routing ───────────────────────────────────────────────────────

def region_is_text(orig_img, bbox):
    """Content test on the CLEAN ORIGINAL inside bbox: prose reads as many thin,
    regular rows on a low-colour ground. Tried in BOTH polarities so it fires for
    dark ink on pale paper AND white text on a black ground (colophon / copyright
    pages). Reuses the text-line detector so the cue matches the rest of the
    pipeline."""
    import text_blocks as tb
    x0, y0, x1, y1 = bbox
    crop = orig_img[max(0, y0):y1, max(0, x0):x1]
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    if (hsv[:, :, 1] > 50).mean() > 0.20:             # a painting, not a text column
        return False
    page_h = orig_img.shape[0]
    for dark in (True, False):
        lines = tb._line_boxes(g, dark_text=dark)
        if tb._regular(lines, page_h) or len(lines) >= 4:
            return True
    return False


# ── text-block correction ─────────────────────────────────────────────────────

def _fit_offsets(poly, x0, x1):
    """Sample a guide polyline to one y per integer column in [x0,x1); gaps are
    filled by linear interpolation, ends held. Returns y(x) array."""
    xs = poly[:, 0]
    ys = poly[:, 1]
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    grid = np.arange(x0, x1)
    return np.interp(grid, xs, ys, left=ys[0], right=ys[-1])


def correct_text_block(src_img, region, flatfield_whiten=None, bg=246):
    """Flatten a text block to its magenta warp guides. The guides are NOT a
    boundary — each horizontal guide is a WARP REFERENCE at its own depth, tracing
    how the text is skewed/curled at that row. With several stacked guides the warp
    can be corrected even when it WORSENS down the page (a page_dewarp-style
    column remap that is piecewise-linear between guides, not a single global tilt).

      * >= 2 guides -> per column, the guides' y-values are the control points; each
        maps to its own flat level and rows in between/beyond are interpolated and
        linearly extrapolated, so every guide becomes straight and level.
      * 1 guide     -> only a slope is known, so the whole block is deskewed by that
        guide's tilt (carried parallel).

    src_img is the image the patch is sampled from — pass the ALREADY-COMPOSED page
    (leave flatfield_whiten=None) so the block keeps the page's own background tone
    in EITHER polarity (bright paper or black ground); exposed edges replicate the
    page border. Returns (patch_bgr, (px0, py0)) or (None, None)."""
    hlines = region.get('hlines', [])
    if not hlines:
        return None, None
    H, W = src_img.shape[:2]
    x0, y0, x1, y1 = region['bbox']
    # extend a margin past the outermost guides so the text they sit under/over is
    # carried too (guides are usually drawn as under/overlines, not block edges).
    marg = max(20, int(0.02 * H))
    gy_top = min(float(np.median(p[:, 1])) for p in hlines)
    gy_bot = max(float(np.median(p[:, 1])) for p in hlines)
    y0 = int(max(0, min(y0, gy_top - marg)))
    y1 = int(min(H, max(y1, gy_bot + marg)))
    x0 = max(0, x0); x1 = min(W, x1)
    crop = src_img[y0:y1, x0:x1].copy()
    ch, cw = crop.shape[:2]

    # sample every guide to one src-row per column, and its flat target level
    G = np.stack([_fit_offsets(p, x0, x1) - y0 for p in hlines], 0)   # (n, cw)
    levels = np.array([float(np.median(g)) for g in G])               # (n,) dest rows
    order = np.argsort(levels)
    G = G[order]; levels = levels[order]
    n = len(levels)

    map_x = np.tile(np.arange(cw, dtype=np.float32), (ch, 1))
    map_y = np.empty((ch, cw), np.float32)
    dest = np.arange(ch, dtype=np.float32)
    if n == 1:
        # single guide -> per-column vertical shift so the guide row flattens (deskew,
        # carried parallel through the whole block)
        g0 = G[0]
        for c in range(cw):
            map_y[:, c] = dest + (g0[c] - levels[0])
    else:
        # piecewise-linear dest->src per column through all guide control points,
        # with linear extrapolation past the first/last guide
        for c in range(cw):
            src = np.interp(dest, levels, G[:, c])
            lo = dest < levels[0]
            hi = dest > levels[-1]
            if lo.any():
                s0 = (G[1, c] - G[0, c]) / max(levels[1] - levels[0], 1.0)
                src[lo] = G[0, c] + (dest[lo] - levels[0]) * s0
            if hi.any():
                s1 = (G[-1, c] - G[-2, c]) / max(levels[-1] - levels[-2], 1.0)
                src[hi] = G[-1, c] + (dest[hi] - levels[-1]) * s1
            map_y[:, c] = src
    out = cv2.remap(crop, map_x, map_y, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE)   # keep the page's own bg tone
    if flatfield_whiten is not None:
        out = flatfield_whiten(out)
    return out, (x0, y0)


if __name__ == '__main__':
    import sys
    clean = cv2.imread(sys.argv[1])
    mark = cv2.imread(sys.argv[2])
    regs = magenta_regions(mark)
    print(f'{len(regs)} region(s):')
    for r in regs:
        kind = 'TEXT' if region_is_text(clean, r['bbox']) else 'ART'
        print(f"  {r['bbox']}  hlines={len(r['hlines'])} vlines={len(r['vlines'])} -> {kind}")

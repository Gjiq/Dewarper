"""
Spectrum 6 layout reconstruction pipeline.
0. Straighten text (deskew + dewarp) on text-heavy pages
1. Classify each page: DARK_BG | FULL_BLEED | TEXT_LEFT | SKIP
2. Detect art panel bounding boxes per page type
3. Clean-crop each panel (straight edges, 90th-percentile inward cut)
4. Paste each panel at its original page position on a fresh canvas
"""
import cv2
import numpy as np
import os
import sys
from scipy.ndimage import uniform_filter1d


# ── STEP 0: text straightening (deskew + per-line dewarp) ─────────────────
def _text_line_comps(gray):
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 31, 15)
    li = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_RECT, (85, 1)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(li, 8)
    H, W = gray.shape
    keep = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w < 0.35 * W or h < 18 or h > 0.18 * H:
            continue
        keep.append(i)
    return lab, keep

def _skew_angle(gray):
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 31, 15)
    h, w = th.shape
    small = cv2.resize(th, (max(1, w // 3), max(1, h // 3)))
    best_a, best = 0.0, -1.0
    for a in np.arange(-5, 5.01, 0.5):
        M = cv2.getRotationMatrix2D((small.shape[1] / 2, small.shape[0] / 2), a, 1)
        r = cv2.warpAffine(small, M, (small.shape[1], small.shape[0]), flags=cv2.INTER_NEAREST)
        p = r.sum(1).astype(float)
        s = ((p - p.mean()) ** 2).mean()
        if s > best:
            best, best_a = s, a
    return best_a

def straighten_text(block):
    """Deskew then dewarp text. Self-gating: pages without enough full-width
    text lines (i.e. art/painting pages) are returned untouched."""
    H, W = block.shape[:2]
    gray = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
    lab, keep = _text_line_comps(gray)
    if len(keep) < 4:
        return block                      # not a text page -> leave alone

    a = _skew_angle(gray)
    if abs(a) > 0.2:
        M = cv2.getRotationMatrix2D((W / 2, H / 2), a, 1)
        block = cv2.warpAffine(block, M, (W, H), flags=cv2.INTER_CUBIC,
                               borderValue=(255, 255, 255))
        gray = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
        lab, keep = _text_line_comps(gray)
        if len(keep) < 2:
            return block

    xs = np.arange(W); yc_full = []; d_full = []
    for i in keep:
        comp = (lab == i)
        yc = np.full(W, np.nan)
        for cx in np.where(comp.any(axis=0))[0]:
            yc[cx] = np.where(comp[:, cx])[0].mean()
        v = ~np.isnan(yc)
        yc = np.interp(xs, xs[v], yc[v])
        yc = cv2.GaussianBlur(yc.reshape(1, -1), (0, 0), 30).ravel()
        yc_full.append(yc); d_full.append(yc.mean() - yc)
    if len(yc_full) < 2:
        return block
    yc_full = np.array(yc_full); d_full = np.array(d_full)
    grid = np.arange(H); D = np.zeros((H, W), np.float32)
    for cx in range(W):
        yv = yc_full[:, cx]; o = np.argsort(yv)
        D[:, cx] = np.interp(grid, yv[o], d_full[o, cx])
    D = np.clip(D, -60, 60)               # cap so non-text areas aren't yanked
    map_x = np.tile(xs, (H, 1)).astype(np.float32)
    map_y = np.repeat(grid, W).reshape(H, W).astype(np.float32) - D
    return cv2.remap(block, map_x, map_y, cv2.INTER_CUBIC, borderValue=(255, 255, 255))


if len(sys.argv) < 2:
    print("Usage: drag a folder of JPGs onto RUN.bat")
    input("Press Enter to exit...")
    sys.exit(1)

INPUT_DIR  = sys.argv[1].strip('"').strip("'")
OUTPUT_DIR = os.path.join(INPUT_DIR, 'reconstructed')
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Input:  {INPUT_DIR}")
print(f"Output: {OUTPUT_DIR}")
print()

# ── Page classifier ───────────────────────────────────────────────────────────

def classify_page(img_bgr):
    """
    Returns one of: 'DARK_BG' | 'FULL_BLEED' | 'TEXT_LEFT' | 'SKIP'
    """
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat  = hsv[:,:,1]

    bg = float(np.median([gray[10,10], gray[10,W-10], gray[H-10,10], gray[H-10,W-10]]))
    if bg < 80:
        return 'DARK_BG', bg

    WHITE = bg - 20
    content_pct = (gray < WHITE).mean() * 100
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
            # Only count text lines in narrow zone (max 200px) to avoid art content
            text_zone = gray[:, :min(strip_end, 200)]
            soby  = np.abs(np.diff(text_zone.astype(float), axis=0))
            text_lines = int((soby.max(axis=1) > 15).sum())
            if text_lines > 400:
                return 'TEXT_LEFT', bg

    # Bilateral binder margins with no text column = centered art panel
    if leftmost_sat < 35 and rightmost_sat < 35:
        return 'FULL_BLEED', bg

    return 'FULL_BLEED', bg


# Pages that auto-classification gets wrong — override here
PAGE_OVERRIDES = {
    # Jury page: irregular scattered portrait layout, not reconstructable as panels
    'Spectrum 6_Page_006.jpg': 'SKIP',
    # Student competition + article: too mixed, art panels embedded in text columns
    'Spectrum 6_Page_014.jpg': 'SKIP',
}


# ── Panel detection ───────────────────────────────────────────────────────────

def detect_panels(img_bgr, page_type, bg_level):
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    WHITE = bg_level - 20

    if page_type == 'FULL_BLEED':
        # Whole page is one panel — just find the content bbox
        content = (gray < WHITE).astype(np.uint8)
        rows = np.where(content.any(axis=1))[0]
        cols = np.where(content.any(axis=0))[0]
        if len(rows) and len(cols):
            return [(int(cols[0]), int(rows[0]),
                     int(cols[-1]-cols[0]), int(rows[-1]-rows[0]))]
        return [(0, 0, W, H)]

    if page_type == 'DARK_BG':
        bg_thresh = 50
        candidate = (gray <= bg_thresh).astype(np.uint8)
        bg_mask = np.zeros((H, W), np.uint8)
        for (y, x) in [(0,0),(0,W-1),(H-1,0),(H-1,W-1)]:
            if candidate[y, x]:
                sm = np.zeros((H+2, W+2), np.uint8)
                flood = candidate.copy()
                cv2.floodFill(flood, sm, (x, y), 2)
                bg_mask |= (flood == 2).astype(np.uint8)
        content = (1 - bg_mask).astype(np.uint8) * 255
        kernel  = np.ones((40,40), np.uint8)
        closed  = cv2.morphologyEx(content, cv2.MORPH_CLOSE, kernel)
        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            if w*h > 0.05*W*H and 0.1 < w/h < 10:
                boxes.append((x,y,w,h))
        return sorted(boxes, key=lambda b:(b[1],b[0])) or [(0,0,W,H)]

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


# ── Main ──────────────────────────────────────────────────────────────────────

files = sorted(f for f in os.listdir(INPUT_DIR)
               if f.lower().endswith('.jpg') or f.lower().endswith('.jpeg'))
print(f'Found {len(files)} images...\n')

for fname in files:
    path = os.path.join(INPUT_DIR, fname)
    img  = cv2.imread(path)
    if img is None: continue
    img  = straighten_text(img)        # STEP 0: deskew + dewarp text
    H, W = img.shape[:2]

    page_type, bg = classify_page(img)
    if fname in PAGE_OVERRIDES:
        page_type = PAGE_OVERRIDES[fname]
    print(f'{fname}: {page_type}', end='')

    if page_type == 'SKIP':
        print(' — skipped')
        continue

    dark_bg   = page_type == 'DARK_BG'

    # Sample background color from page margins
    H, W = img.shape[:2]
    border_mask = np.zeros((H, W), dtype=bool)
    bw = max(30, int(W * 0.05))
    bh = max(30, int(H * 0.05))
    border_mask[:bh, :]  = True
    border_mask[-bh:, :] = True
    border_mask[:, :bw]  = True
    border_mask[:, -bw:] = True
    border_pixels = img[border_mask]
    bg_color = tuple(int(x) for x in np.median(border_pixels, axis=0))

    canvas    = np.full((H, W, 3), bg_color, dtype=np.uint8)

    boxes = detect_panels(img, page_type, bg)
    print(f' -> {len(boxes)} panels')

    for (bx, by, bw, bh) in boxes:
        clean, (cx0,cy0,cx1,cy1) = clean_crop(img, bx, by, bw, bh, bg)
        cH, cW = clean.shape[:2]
        dy0=max(0,cy0); dy1=min(H,cy0+cH)
        dx0=max(0,cx0); dx1=min(W,cx0+cW)
        sy0=dy0-cy0;    sy1=sy0+(dy1-dy0)
        sx0=dx0-cx0;    sx1=sx0+(dx1-dx0)
        if dy1>dy0 and dx1>dx0:
            canvas[dy0:dy1, dx0:dx1] = clean[sy0:sy1, sx0:sx1]

    stem    = fname.replace('.jpg','')
    outpath = os.path.join(OUTPUT_DIR, f'{stem}_reconstructed.jpg')
    cv2.imwrite(outpath, canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])

print(f'\nDone. Reconstructed pages saved to:\n  {OUTPUT_DIR}')
input('\nPress Enter to close...')

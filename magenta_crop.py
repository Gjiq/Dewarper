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

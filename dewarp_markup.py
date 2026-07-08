#!/usr/bin/env python3
"""Magenta markup editor — draw guides on a page and export a pixel-aligned
<page>_mag.jpg that the engine reads as manual guides.

Tools:
  * Skew Box  -- a magenta quad with four DRAGGABLE corners (up to 5 per page).
                 The engine perspective-rectifies the content inside each quad
                 from its corners (deskew), so the box can be dragged onto rotated
                 or perspective-skewed art.
  * Free Line -- freehand strokes: warp / text-bowing guides (dewarp areas).
  * Fill      -- a bucket that fills the region ENCLOSED by free lines. Dropped
                 anywhere not inside a closed loop of lines, it does nothing.

Strokes are stored in ORIGINAL image coordinates and composited at full
resolution, so the markup stays pixel-aligned with the clean scan. The render and
flood helpers are pure functions (unit-testable headless).
"""
import base64
import os

# Pen colour: BGR inside magenta_crop's detected band (HSV ~163/240/245).
PEN_BGR = (145, 14, 245)
PEN_HEX = "#f50e91"
MAX_QUADS = 5


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def render_markup(orig_bgr, strokes, pen_bgr=PEN_BGR):
    """Composite strokes (ORIGINAL coords) onto a copy of the full-res original.
    Stroke dicts:
      {"tool":"skewbox","pts":[4x(x,y)],"w":int}  -> closed magenta quad outline
      {"tool":"freeline","pts":[(x,y)...],"w":int} -> open magenta polyline
      {"tool":"fill","pts":[(x,y)...]}             -> filled magenta polygon
    (legacy aliases box/line/freehand are also accepted.)
    """
    import cv2
    import numpy as np
    out = orig_bgr.copy()
    for s in strokes:
        tool = s.get("tool")
        pts = s.get("pts") or []
        w = int(s.get("w", 8))
        if tool in ("skewbox",) and len(pts) >= 4:
            arr = np.array([[int(x), int(y)] for x, y in pts[:4]], np.int32)
            cv2.polylines(out, [arr], True, pen_bgr, w)
        elif tool == "box" and len(pts) >= 2:                       # legacy rect
            (x0, y0), (x1, y1) = pts[0], pts[-1]
            cv2.rectangle(out, (int(min(x0, x1)), int(min(y0, y1))),
                          (int(max(x0, x1)), int(max(y0, y1))), pen_bgr, w)
        elif tool in ("freeline", "freehand") and len(pts) >= 2:
            arr = np.array([[int(x), int(y)] for x, y in pts], np.int32)
            cv2.polylines(out, [arr], False, pen_bgr, w)
        elif tool == "line" and len(pts) >= 2:                      # legacy straight
            cv2.line(out, (int(pts[0][0]), int(pts[0][1])),
                     (int(pts[-1][0]), int(pts[-1][1])), pen_bgr, w)
        elif tool == "fill" and len(pts) >= 3:
            arr = np.array([[int(x), int(y)] for x, y in pts], np.int32)
            cv2.fillPoly(out, [arr], pen_bgr)
    return out


def enclosed_region(line_pts_list, seed_xy, shape, line_w=3):
    """Given freehand lines (lists of (x,y) points), decide whether `seed_xy` sits
    inside a region CLOSED by those lines. Returns the enclosed region's contour as
    a list of (x,y) points, or None if the seed is not enclosed (flood escapes to
    the image border). All coordinates in the same space as `shape` (H, W)."""
    import cv2
    import numpy as np
    H, W = shape[:2]
    wall = np.zeros((H, W), np.uint8)
    for pts in line_pts_list:
        if len(pts) >= 2:
            arr = np.array([[int(x), int(y)] for x, y in pts], np.int32)
            cv2.polylines(wall, [arr], False, 255, max(1, int(line_w)))
    sx, sy = int(seed_xy[0]), int(seed_xy[1])
    if not (0 <= sx < W and 0 <= sy < H) or wall[sy, sx]:
        return None
    free = (wall == 0).astype(np.uint8)                 # 1 where paintable
    ff = np.zeros((H + 2, W + 2), np.uint8)
    filled = free.copy()
    cv2.floodFill(filled, ff, (sx, sy), 2)              # mark reachable region as 2
    region = (filled == 2).astype(np.uint8)
    # touches the border -> the seed was NOT enclosed
    if region[0, :].any() or region[-1, :].any() or region[:, 0].any() or region[:, -1].any():
        return None
    cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    return [(int(x), int(y)) for x, y in c.reshape(-1, 2)]


def mag_path_for(image_path):
    d = os.path.dirname(image_path)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(d, f"{stem}_mag.jpg")


def save_markup(image_path, strokes, quality=95):
    """Render strokes onto the original, write <stem>_mag.jpg beside it. Returns the
    path, or None if nothing drawn / read failed."""
    import cv2
    if not strokes:
        return None
    orig = cv2.imread(image_path)
    if orig is None:
        return None
    out = render_markup(orig, strokes)
    path = mag_path_for(image_path)
    cv2.imwrite(path, out, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return path


# ---------------------------------------------------------------------------
# GUI editor (Tkinter). Imported lazily; pure helpers above stay headless.
# ---------------------------------------------------------------------------

def open_markup(parent, image_path, on_saved=None, max_side=880):
    import tkinter as tk
    from tkinter import ttk, messagebox
    import cv2

    orig = cv2.imread(image_path)
    if orig is None:
        messagebox.showerror("Markup", f"Could not open:\n{image_path}")
        return
    # Re-marking a page starts fresh: drop any previous magenta version so the old
    # markup is not left in the input folder / queue.
    for _old in (mag_path_for(image_path),
                 os.path.splitext(image_path)[0] + "_magenta.jpg"):
        try:
            if os.path.isfile(_old):
                os.remove(_old)
        except OSError:
            pass
    H, W = orig.shape[:2]
    scale = min(1.0, float(max_side) / max(H, W))
    dw, dh = max(1, int(W * scale)), max(1, int(H * scale))
    disp = cv2.resize(orig, (dw, dh), interpolation=cv2.INTER_AREA) if scale < 1 else orig
    ok, buf = cv2.imencode(".png", disp)
    photo_data = base64.b64encode(buf.tobytes()) if ok else None

    win = tk.Toplevel(parent)
    win.title(f"Markup — {os.path.basename(image_path)}")
    win.transient(parent)
    try:
        import sys as _sys
        _base = getattr(_sys, "_MEIPASS", None) or os.path.dirname(os.path.abspath(__file__))
        _icon = tk.PhotoImage(file=os.path.join(_base, "icon.png"))
        win.iconphoto(False, _icon)
        win._icon = _icon
    except Exception:
        pass

    tool = tk.StringVar(value="pen")
    penw = tk.IntVar(value=3)
    rects = []     # each: {"pts":[[x,y]*4], "handles":[ids], "line":id, "delx":id, "cx","cy"}
    lines = []     # each: {"disp":[(x,y)...], "item":id}
    fills = []     # each: {"disp":[(x,y)...], "item":id, "delx":id, "cx","cy"}

    def o(cx, cy):                         # display -> original coords
        return (min(max(cx, 0), dw) / scale, min(max(cy, 0), dh) / scale)

    def ow(w):                             # display width -> original width
        return max(4, int(round(w / max(scale, 1e-6))))

    HR = 6          # corner handle half-size
    DELR = 11       # delete-x hit radius

    bar = ttk.Frame(win, padding=6)
    bar.pack(fill="x")
    add_btn = ttk.Button(bar, text="+ Rectangle")
    add_btn.pack(side="left")
    ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=6)
    ttk.Label(bar, text="Tool:").pack(side="left")
    for lbl, val in [("Pen", "pen"), ("Eraser", "eraser"), ("Fill", "fill")]:
        ttk.Radiobutton(bar, text=lbl, value=val, variable=tool).pack(side="left")
    ttk.Label(bar, text="  Pen:").pack(side="left")
    ttk.Spinbox(bar, from_=1, to=12, width=3, textvariable=penw).pack(side="left")

    cv = tk.Canvas(win, width=dw, height=dh, background="#333",
                   highlightthickness=0, cursor="crosshair")
    cv.pack(padx=6, pady=6)
    photo = tk.PhotoImage(data=photo_data) if photo_data else None
    if photo is not None:
        cv.create_image(0, 0, anchor="nw", image=photo)
        cv._bg = photo

    foot = ttk.Frame(win, padding=6)
    foot.pack(fill="x")
    status = ttk.Label(foot, text="")
    status.pack(side="left")

    def _update_status():
        status.configure(
            text=f"{len(rects)}/{MAX_QUADS} rectangle(s), {len(lines)} line(s), "
                 f"{len(fills)} fill(s).  Rectangle over art = dewarp, over text = "
                 f"deskew.  Drag corners; click a \u2715 to delete.")

    # ---- rectangles -------------------------------------------------------
    def _draw_rect(r):
        for i in r.get("handles", []):
            cv.delete(i)
        for k in ("line", "delx"):
            if r.get(k):
                cv.delete(r[k])
        flat = [c for p in r["pts"] for c in p]
        r["line"] = cv.create_polygon(*flat, outline=PEN_HEX, fill="", width=2)
        r["handles"] = [cv.create_rectangle(x - HR, y - HR, x + HR, y + HR,
                        fill="white", outline=PEN_HEX, width=2) for x, y in r["pts"]]
        cx = sum(p[0] for p in r["pts"]) / 4.0
        cy = sum(p[1] for p in r["pts"]) / 4.0
        r["cx"], r["cy"] = cx, cy
        r["delx"] = cv.create_text(cx, cy, text="\u2715", fill="#d00000",
                                   font=("TkDefaultFont", 13, "bold"))

    def add_rect():
        if len(rects) >= MAX_QUADS:
            status.configure(text=f"Limit {MAX_QUADS} rectangles per page.")
            return
        n = len(rects)
        off = 18 * n
        x0, y0 = 0.22 * dw + off, 0.22 * dh + off
        x1, y1 = min(dw - 4, 0.72 * dw + off), min(dh - 4, 0.72 * dh + off)
        r = {"pts": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]}
        _draw_rect(r)
        rects.append(r)
        _update_status()

    add_btn.configure(command=add_rect)

    def _delete_rect(r):
        for i in r.get("handles", []):
            cv.delete(i)
        for k in ("line", "delx"):
            if r.get(k):
                cv.delete(r[k])
        if r in rects:
            rects.remove(r)
        _update_status()

    def _delete_fill(f):
        for k in ("item", "delx"):
            if f.get(k):
                cv.delete(f[k])
        if f in fills:
            fills.remove(f)
        _update_status()

    def _hit(x, y):
        for r in rects:                              # delete-x first
            if abs(r["cx"] - x) <= DELR and abs(r["cy"] - y) <= DELR:
                return ("rectdel", r)
        for f in fills:
            if abs(f["cx"] - x) <= DELR and abs(f["cy"] - y) <= DELR:
                return ("filldel", f)
        for r in rects:                              # corner handles
            for ci, (hx, hy) in enumerate(r["pts"]):
                if abs(hx - x) <= HR + 2 and abs(hy - y) <= HR + 2:
                    return ("corner", r, ci)
        return None

    def _erase_at(x, y, rad=12):
        r2 = rad * rad
        for ln in list(lines):
            if any((px - x) ** 2 + (py - y) ** 2 <= r2 for px, py in ln["disp"]):
                cv.delete(ln["item"])
                lines.remove(ln)
        _update_status()

    def _do_fill(x, y):
        reg = enclosed_region([ln["disp"] for ln in lines], (x, y),
                              (dh, dw), line_w=3)
        if reg is None:
            status.configure(text="Fill did nothing — click inside a closed set "
                                  "of Pen lines.")
            return
        flat = [c for p in reg for c in p]
        item = cv.create_polygon(*flat, fill=PEN_HEX, outline=PEN_HEX,
                                 stipple="gray50")
        delx = cv.create_text(x, y, text="\u2715", fill="black",
                              font=("TkDefaultFont", 13, "bold"))
        fills.append({"disp": reg, "item": item, "delx": delx,
                      "cx": float(x), "cy": float(y)})
        _update_status()

    drag = {"mode": None, "rect": None, "corner": None, "pts": [], "item": None}

    def on_press(e):
        hit = _hit(e.x, e.y)
        if hit:
            if hit[0] == "rectdel":
                _delete_rect(hit[1])
                return
            if hit[0] == "filldel":
                _delete_fill(hit[1])
                return
            drag.update(mode="corner", rect=hit[1], corner=hit[2])
            return
        t = tool.get()
        if t == "pen":
            drag.update(mode="line", pts=[(e.x, e.y)], item=None)
        elif t == "eraser":
            drag.update(mode="erase")
            _erase_at(e.x, e.y)
        else:                                        # fill on release
            drag.update(mode="fill")

    def on_move(e):
        m = drag["mode"]
        if m == "corner":
            r = drag["rect"]
            r["pts"][drag["corner"]] = [e.x, e.y]
            _draw_rect(r)
        elif m == "line":
            drag["pts"].append((e.x, e.y))
            if drag["item"]:
                cv.delete(drag["item"])
            flat = [c for p in drag["pts"] for c in p]
            if len(flat) >= 4:
                drag["item"] = cv.create_line(*flat, fill=PEN_HEX, width=2)
        elif m == "erase":
            _erase_at(e.x, e.y)

    def on_release(e):
        m = drag["mode"]
        if m == "line":
            if len(drag["pts"]) >= 2:
                flat = [c for p in drag["pts"] for c in p]
                item = cv.create_line(*flat, fill=PEN_HEX, width=2)
                lines.append({"disp": list(drag["pts"]), "item": item})
            if drag["item"]:
                cv.delete(drag["item"])
        elif m == "fill":
            _do_fill(e.x, e.y)
        drag["mode"] = None
        _update_status()

    cv.bind("<ButtonPress-1>", on_press)
    cv.bind("<B1-Motion>", on_move)
    cv.bind("<ButtonRelease-1>", on_release)

    # ---- export / buttons -------------------------------------------------
    def collect_strokes():
        strokes = []
        for r in rects:
            strokes.append({"tool": "skewbox",
                            "pts": [o(x, y) for x, y in r["pts"]], "w": ow(penw.get())})
        for ln in lines:
            strokes.append({"tool": "freeline",
                            "pts": [o(x, y) for x, y in ln["disp"]], "w": ow(penw.get())})
        for f in fills:
            strokes.append({"tool": "fill", "pts": [o(x, y) for x, y in f["disp"]]})
        return strokes

    def undo():
        if fills:
            _delete_fill(fills[-1])
        elif lines:
            cv.delete(lines.pop()["item"])
            _update_status()
        elif rects:
            _delete_rect(rects[-1])

    def clear():
        while fills:
            _delete_fill(fills[-1])
        while lines:
            cv.delete(lines.pop()["item"])
        while rects:
            _delete_rect(rects[-1])
        _update_status()

    def save():
        strokes = collect_strokes()
        if not strokes:
            messagebox.showinfo("Markup", "Nothing drawn yet.")
            return
        if not rects and not fills:
            if not messagebox.askyesno(
                "Markup", "No rectangle or fill region — the engine needs at least one "
                          "closed region to treat this as art/text markup. Save anyway?"):
                return
        path = save_markup(image_path, strokes)
        if path and on_saved:
            on_saved(path)
        win.destroy()

    ttk.Button(foot, text="Undo", command=undo).pack(side="right", padx=2)
    ttk.Button(foot, text="Clear", command=clear).pack(side="right", padx=2)
    ttk.Button(foot, text="Cancel", command=win.destroy).pack(side="right", padx=2)
    ttk.Button(foot, text="Apply", command=save).pack(side="right", padx=8)
    _update_status()
    return win

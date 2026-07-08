#!/usr/bin/env python3
"""Magenta markup editor — draw art/text guides on a page and export a pixel-aligned
<page>_guide.jpg the pipeline reads as manual guides (plus a <page>_guide.json sidecar
holding the editable strokes so the markup can be re-opened and adjusted).

Tools (one shared thickness applies to whichever is active):
  Box    magenta rectangle with 4 draggable corners (up to 5). Over art -> dewarp;
         over text -> deskew (decided by content).
  Line   a single straight magenta line.
  Pen    freehand magenta line (along text rows = warp guide).
  Eraser drag over a line / box / fill to delete it.
  Fill   flood-fill the area enclosed by lines (nothing if not enclosed); black x deletes.

render_markup / enclosed_region are pure and unit-testable headless.
"""
import base64
import json
import os

PEN_BGR = (145, 14, 245)      # in magenta_crop's detected band
PEN_HEX = "#f50e91"
MAX_QUADS = 5


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def render_markup(orig_bgr, strokes, pen_bgr=PEN_BGR):
    """Composite strokes (ORIGINAL coords) onto a copy of the full-res original."""
    import cv2
    import numpy as np
    out = orig_bgr.copy()
    for s in strokes:
        tool = s.get("tool")
        pts = s.get("pts") or []
        w = int(s.get("w", 8))
        if tool == "skewbox" and len(pts) >= 4:
            arr = np.array([[int(x), int(y)] for x, y in pts[:4]], np.int32)
            cv2.polylines(out, [arr], True, pen_bgr, w)
        elif tool in ("freeline", "freehand") and len(pts) >= 2:
            arr = np.array([[int(x), int(y)] for x, y in pts], np.int32)
            cv2.polylines(out, [arr], False, pen_bgr, w)
        elif tool == "line" and len(pts) >= 2:
            cv2.line(out, (int(pts[0][0]), int(pts[0][1])),
                     (int(pts[-1][0]), int(pts[-1][1])), pen_bgr, w)
        elif tool == "fill" and len(pts) >= 3:
            arr = np.array([[int(x), int(y)] for x, y in pts], np.int32)
            cv2.fillPoly(out, [arr], pen_bgr)
    return out


def enclosed_region(line_pts_list, seed_xy, shape, line_w=3):
    """Contour of the region CLOSED by the given lines around seed_xy, or None if the
    seed is not enclosed (flood escapes to the border)."""
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
    free = (wall == 0).astype(np.uint8)
    ff = np.zeros((H + 2, W + 2), np.uint8)
    filled = free.copy()
    cv2.floodFill(filled, ff, (sx, sy), 2)
    region = (filled == 2).astype(np.uint8)
    if region[0, :].any() or region[-1, :].any() or region[:, 0].any() or region[:, -1].any():
        return None
    cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    return [(int(x), int(y)) for x, y in c.reshape(-1, 2)]


def guide_path_for(image_path):
    d = os.path.dirname(image_path)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(d, f"{stem}_guide.jpg")


def guide_json_for(image_path):
    d = os.path.dirname(image_path)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(d, f"{stem}_guide.json")


# legacy alias (older callers)
def mag_path_for(image_path):
    return guide_path_for(image_path)


def save_markup(image_path, strokes, quality=95):
    """Render strokes onto the original -> <stem>_guide.jpg, and write the editable
    strokes to <stem>_guide.json. Returns the jpg path (or None if nothing/failed)."""
    import cv2
    if not strokes:
        return None
    orig = cv2.imread(image_path)
    if orig is None:
        return None
    out = render_markup(orig, strokes)
    jpg = guide_path_for(image_path)
    cv2.imwrite(jpg, out, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    try:
        with open(guide_json_for(image_path), "w", encoding="utf-8") as f:
            json.dump(strokes, f)
    except OSError:
        pass
    return jpg


def load_strokes(image_path):
    """Editable strokes previously saved for this page, or [] if none."""
    p = guide_json_for(image_path)
    if not os.path.isfile(p):
        return []
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _pt_in_poly(x, y, poly):
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and \
                (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def _seg_t(px, py, ax, ay, bx, by):
    """Clamped parameter (0..1) of the projection of (px,py) onto segment A-B."""
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-9:
        return 0.0
    t = ((px - ax) * dx + (py - ay) * dy) / L2
    return max(0.0, min(1.0, t))


def _quad_bezier(a, c, b, n=22):
    """Sample a quadratic Bezier A -> (control C) -> B into n+1 points."""
    out = []
    for i in range(n + 1):
        t = i / n
        u = 1.0 - t
        out.append((u * u * a[0] + 2 * u * t * c[0] + t * t * b[0],
                    u * u * a[1] + 2 * u * t * c[1] + t * t * b[1]))
    return out


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
    H, W = orig.shape[:2]
    scale = min(1.0, float(max_side) / max(H, W))
    dw, dh = max(1, int(W * scale)), max(1, int(H * scale))
    disp = cv2.resize(orig, (dw, dh), interpolation=cv2.INTER_AREA) if scale < 1 else orig
    ok, buf = cv2.imencode(".png", disp)
    photo_data = base64.b64encode(buf.tobytes()) if ok else None

    win = tk.Toplevel(parent)
    win.title(f"Markup — {os.path.basename(image_path)}")
    win.transient(parent)
    win.minsize(360, 320)
    try:
        import sys as _sys
        _base = getattr(_sys, "_MEIPASS", None) or os.path.dirname(os.path.abspath(__file__))
        _icon = tk.PhotoImage(file=os.path.join(_base, "icon.png"))
        win.iconphoto(False, _icon)
        win._icon = _icon
    except Exception:
        pass

    tool = tk.StringVar(value="box")
    thick = tk.IntVar(value=5)        # ORIGINAL-px magenta thickness (5..25 step 5)
    rects = []     # {"pts":[[x,y]*4], "w", "handles":[ids], "line":id}
    lines = []     # {"kind":"line"|"pen", "disp":[(x,y)...], "w", "item":id}
    fills = []     # {"disp":[(x,y)...], "item":id, "delx":id, "cx","cy"}

    def o(cx, cy):                     # display -> original
        return (min(max(cx, 0), dw) / scale, min(max(cy, 0), dh) / scale)

    def d(ox, oy):                     # original -> display
        return (ox * scale, oy * scale)

    def dispw():                       # current thickness as a display line width
        return max(1, int(round(thick.get() * scale)))

    HR = 6

    # ---- toolbar (top) ----------------------------------------------------
    bar = ttk.Frame(win, padding=6)
    bar.pack(side="top", fill="x")
    ttk.Label(bar, text="Tool:").pack(side="left")
    for lbl, val in [("Box", "box"), ("Line", "line"), ("Pen", "pen"),
                     ("Bow", "bow"), ("Eraser", "eraser"), ("Fill", "fill")]:
        ttk.Radiobutton(bar, text=lbl, value=val, variable=tool).pack(side="left")
    ttk.Spinbox(bar, from_=5, to=25, increment=5, width=4,
                textvariable=thick).pack(side="left", padx=(8, 0))

    # ---- action buttons (top-right, so a resize never hides them) ---------
    act = ttk.Frame(win, padding=(6, 0, 6, 4))
    act.pack(side="top", fill="x")

    # ---- canvas (fills remaining space) -----------------------------------
    cv = tk.Canvas(win, width=dw, height=dh, background="#333",
                   highlightthickness=0, cursor="crosshair")
    cv.pack(side="top", fill="both", expand=True, padx=6, pady=6)
    photo = tk.PhotoImage(data=photo_data) if photo_data else None
    if photo is not None:
        cv.create_image(0, 0, anchor="nw", image=photo)
        cv._bg = photo

    status = ttk.Label(win, anchor="w", padding=(8, 2))
    status.pack(side="bottom", fill="x")

    def _update_status():
        status.configure(
            text=f"{len(rects)} box(es), {len(lines)} line(s), "
                 f"{len(fills)} fill(s).  Box over art=dewarp, over text=deskew.  "
                 f"Eraser deletes lines & boxes.  Fill: click inside a box or line loop.")

    # ---- rectangles -------------------------------------------------------
    def _draw_rect(r):
        for i in r.get("handles", []):
            cv.delete(i)
        if r.get("line"):
            cv.delete(r["line"])
        flat = [c for p in r["pts"] for c in p]
        r["line"] = cv.create_polygon(*flat, outline=PEN_HEX, fill="",
                                      width=max(1, int(round(r["w"] * scale))))
        r["handles"] = [cv.create_rectangle(x - HR, y - HR, x + HR, y + HR,
                        fill="white", outline=PEN_HEX, width=2) for x, y in r["pts"]]

    def _delete_rect(r):
        for i in r.get("handles", []):
            cv.delete(i)
        if r.get("line"):
            cv.delete(r["line"])
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
        for f in fills:
            if abs(f["cx"] - x) <= 11 and abs(f["cy"] - y) <= 11:
                return ("filldel", f)
        for r in rects:
            for ci, (hx, hy) in enumerate(r["pts"]):
                if abs(hx - x) <= HR + 2 and abs(hy - y) <= HR + 2:
                    return ("corner", r, ci)
        return None

    def _erase_at(x, y):
        rad = max(8, dispw() + 4)
        r2 = rad * rad
        for ln in list(lines):
            if any((px - x) ** 2 + (py - y) ** 2 <= r2 for px, py in ln["disp"]):
                cv.delete(ln["item"])
                lines.remove(ln)
        for r in list(rects):
            if _pt_in_poly(x, y, r["pts"]) or \
                    any((hx - x) ** 2 + (hy - y) ** 2 <= r2 for hx, hy in r["pts"]):
                _delete_rect(r)
        for f in list(fills):
            if _pt_in_poly(x, y, f["disp"]):
                _delete_fill(f)
        _update_status()

    def _do_fill(x, y):
        walls = [ln["disp"] for ln in lines]            # pen + straight lines
        for r in rects:                                 # rectangles are closed walls
            walls.append(r["pts"] + [r["pts"][0]])
        reg = enclosed_region(walls, (x, y), (dh, dw), line_w=3)
        if reg is None:
            status.configure(text="Fill did nothing — click inside a box or an area "
                                  "closed by lines.")
            return
        flat = [c for p in reg for c in p]
        item = cv.create_polygon(*flat, fill=PEN_HEX, outline=PEN_HEX, stipple="gray50")
        delx = cv.create_text(x, y, text="\u2715", fill="black",
                              font=("TkDefaultFont", 13, "bold"))
        fills.append({"disp": reg, "item": item, "delx": delx,
                      "cx": float(x), "cy": float(y)})
        _update_status()

    drag = {"mode": None, "rect": None, "corner": None, "pts": [], "item": None,
            "start": None, "a1": None, "a2": None, "t0": 0.5}

    pending = {"line": None, "anchors": [], "circles": []}

    def _clear_pending():
        for c in pending["circles"]:
            cv.delete(c)
        pending["line"] = None
        pending["anchors"] = []
        pending["circles"] = []

    def _nearest_straight_line(x, y, thr=12):
        best, bestd = None, thr
        for ln in lines:
            if ln.get("kind") != "line":
                continue
            (ax, ay), (bx, by) = ln["disp"][0], ln["disp"][-1]
            t = _seg_t(x, y, ax, ay, bx, by)
            px, py = ax + t * (bx - ax), ay + t * (by - ay)
            dd = ((px - x) ** 2 + (py - y) ** 2) ** 0.5
            if dd < bestd:
                bestd, best = dd, (ln, (px, py))
        return best

    def on_press(e):
        hit = _hit(e.x, e.y)
        if hit:
            if hit[0] == "filldel":
                _delete_fill(hit[1])
                return
            drag.update(mode="corner", rect=hit[1], corner=hit[2])
            return
        t = tool.get()
        if t == "box":
            drag.update(mode="newbox", start=(e.x, e.y), item=None)
        elif t == "line":
            drag.update(mode="line", start=(e.x, e.y), item=None)
        elif t == "pen":
            drag.update(mode="pen", pts=[(e.x, e.y)], item=None)
        elif t == "eraser":
            drag.update(mode="erase")
            _erase_at(e.x, e.y)
        elif t == "bow":
            if len(pending["anchors"]) < 2:
                seg = _nearest_straight_line(e.x, e.y)
                if seg is None:
                    status.configure(text="Bow: click on a straight Line to drop an anchor.")
                    return
                ln, pt = seg
                if pending["line"] is None:
                    pending["line"] = ln
                elif ln is not pending["line"]:
                    status.configure(text="Bow: both anchors must be on the same line.")
                    return
                pending["anchors"].append(pt)
                pending["circles"].append(cv.create_oval(
                    pt[0] - 4, pt[1] - 4, pt[0] + 4, pt[1] + 4,
                    outline=PEN_HEX, fill="white", width=2))
                status.configure(text=("Bow: drag between the two anchors to bend it."
                                       if len(pending["anchors"]) == 2
                                       else "Bow: click a second spot on the same line."))
                drag["mode"] = None
            else:
                a1, a2 = pending["anchors"]
                t0 = min(0.9, max(0.1, _seg_t(e.x, e.y, a1[0], a1[1], a2[0], a2[1])))
                drag.update(mode="bow", a1=a1, a2=a2, t0=t0, item=None)
        else:
            drag.update(mode="fill")

    def on_move(e):
        m = drag["mode"]
        if m == "corner":
            r = drag["rect"]
            r["pts"][drag["corner"]] = [e.x, e.y]
            _draw_rect(r)
        elif m == "newbox":
            if drag["item"]:
                cv.delete(drag["item"])
            x0, y0 = drag["start"]
            drag["item"] = cv.create_rectangle(x0, y0, e.x, e.y,
                                               outline=PEN_HEX, width=dispw())
        elif m == "line":
            if drag["item"]:
                cv.delete(drag["item"])
            x0, y0 = drag["start"]
            drag["item"] = cv.create_line(x0, y0, e.x, e.y, fill=PEN_HEX, width=dispw())
        elif m == "pen":
            drag["pts"].append((e.x, e.y))
            if drag["item"]:
                cv.delete(drag["item"])
            flat = [c for p in drag["pts"] for c in p]
            if len(flat) >= 4:
                drag["item"] = cv.create_line(*flat, fill=PEN_HEX, width=dispw())
        elif m == "erase":
            _erase_at(e.x, e.y)
        elif m == "bow":
            a1, a2, t0 = drag["a1"], drag["a2"], drag["t0"]
            u = 1.0 - t0
            cx = (e.x - u * u * a1[0] - t0 * t0 * a2[0]) / (2 * u * t0)
            cy = (e.y - u * u * a1[1] - t0 * t0 * a2[1]) / (2 * u * t0)
            curve = _quad_bezier(a1, (cx, cy), a2)
            if drag["item"]:
                cv.delete(drag["item"])
            flat = [c for p in curve for c in p]
            drag["item"] = cv.create_line(*flat, fill=PEN_HEX, width=dispw(), smooth=True)

    def on_release(e):
        m = drag["mode"]
        if m == "newbox":
            if drag["item"]:
                cv.delete(drag["item"])
            x0, y0 = drag["start"]
            if abs(x0 - e.x) < 8 or abs(y0 - e.y) < 8:
                drag["mode"] = None
                return
            xa, xb = sorted((x0, e.x))
            ya, yb = sorted((y0, e.y))
            r = {"pts": [[xa, ya], [xb, ya], [xb, yb], [xa, yb]], "w": thick.get()}
            _draw_rect(r)
            rects.append(r)
        elif m == "line":
            if drag["item"]:
                cv.delete(drag["item"])
            x0, y0 = drag["start"]
            if abs(x0 - e.x) >= 4 or abs(y0 - e.y) >= 4:
                item = cv.create_line(x0, y0, e.x, e.y, fill=PEN_HEX, width=dispw())
                lines.append({"kind": "line", "disp": [(x0, y0), (e.x, e.y)],
                              "w": thick.get(), "item": item})
        elif m == "pen":
            if len(drag["pts"]) >= 2:
                flat = [c for p in drag["pts"] for c in p]
                item = cv.create_line(*flat, fill=PEN_HEX, width=dispw())
                lines.append({"kind": "pen", "disp": list(drag["pts"]),
                              "w": thick.get(), "item": item})
            if drag["item"]:
                cv.delete(drag["item"])
        elif m == "fill":
            _do_fill(e.x, e.y)
        elif m == "bow":
            if drag["item"]:
                cv.delete(drag["item"])
            a1, a2, t0 = drag["a1"], drag["a2"], drag["t0"]
            u = 1.0 - t0
            cx = (e.x - u * u * a1[0] - t0 * t0 * a2[0]) / (2 * u * t0)
            cy = (e.y - u * u * a1[1] - t0 * t0 * a2[1]) / (2 * u * t0)
            curve = _quad_bezier(a1, (cx, cy), a2)
            ln = pending["line"]
            if ln is not None and ln in lines:
                P0, P1 = ln["disp"][0], ln["disp"][-1]
                # keep anchors in line order; the curve runs a1 -> a2
                if _seg_t(a1[0], a1[1], P0[0], P0[1], P1[0], P1[1]) > \
                        _seg_t(a2[0], a2[1], P0[0], P0[1], P1[0], P1[1]):
                    curve = curve[::-1]
                pre = [P0] if (P0[0] - curve[0][0]) ** 2 + (P0[1] - curve[0][1]) ** 2 > 4 else []
                post = [P1] if (P1[0] - curve[-1][0]) ** 2 + (P1[1] - curve[-1][1]) ** 2 > 4 else []
                newpts = pre + curve + post
                cv.delete(ln["item"])
                ln["kind"] = "pen"           # now a freehand-style warp guide (polyline)
                ln["disp"] = newpts
                flat = [c for p in newpts for c in p]
                ln["item"] = cv.create_line(*flat, fill=PEN_HEX,
                                            width=max(1, int(round(ln["w"] * scale))),
                                            smooth=True)
            _clear_pending()
            status.configure(text="Bow applied.")
        drag["mode"] = None
        _update_status()

    cv.bind("<ButtonPress-1>", on_press)
    cv.bind("<B1-Motion>", on_move)
    cv.bind("<ButtonRelease-1>", on_release)

    # ---- load existing guide strokes (re-open, do NOT reset) --------------
    def _load_existing():
        for s in load_strokes(image_path):
            t = s.get("tool")
            w = int(s.get("w", thick.get()))
            pts = s.get("pts") or []
            if t == "skewbox" and len(pts) >= 4:
                r = {"pts": [list(d(x, y)) for x, y in pts[:4]], "w": w}
                _draw_rect(r)
                rects.append(r)
            elif t == "line" and len(pts) >= 2:
                dd = [d(*pts[0]), d(*pts[-1])]
                flat = [c for p in dd for c in p]
                item = cv.create_line(*flat, fill=PEN_HEX, width=max(1, int(round(w * scale))))
                lines.append({"kind": "line", "disp": dd, "w": w, "item": item})
            elif t in ("freeline", "freehand") and len(pts) >= 2:
                dd = [d(x, y) for x, y in pts]
                flat = [c for p in dd for c in p]
                item = cv.create_line(*flat, fill=PEN_HEX, width=max(1, int(round(w * scale))))
                lines.append({"kind": "pen", "disp": dd, "w": w, "item": item})
            elif t == "fill" and len(pts) >= 3:
                dd = [d(x, y) for x, y in pts]
                flat = [c for p in dd for c in p]
                item = cv.create_polygon(*flat, fill=PEN_HEX, outline=PEN_HEX, stipple="gray50")
                cx = sum(p[0] for p in dd) / len(dd)
                cy = sum(p[1] for p in dd) / len(dd)
                delx = cv.create_text(cx, cy, text="\u2715", fill="black",
                                      font=("TkDefaultFont", 13, "bold"))
                fills.append({"disp": dd, "item": item, "delx": delx, "cx": cx, "cy": cy})
        _update_status()

    # ---- export / actions -------------------------------------------------
    def collect_strokes():
        strokes = []
        for r in rects:
            strokes.append({"tool": "skewbox",
                            "pts": [o(x, y) for x, y in r["pts"]], "w": r["w"]})
        for ln in lines:
            strokes.append({"tool": "line" if ln["kind"] == "line" else "freeline",
                            "pts": [o(x, y) for x, y in ln["disp"]], "w": ln["w"]})
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
        path = save_markup(image_path, strokes)
        if path and on_saved:
            on_saved(path)
        win.destroy()

    ttk.Button(act, text="Apply", command=save).pack(side="right", padx=(6, 0))
    ttk.Button(act, text="Cancel", command=win.destroy).pack(side="right", padx=2)
    ttk.Button(act, text="Clear", command=clear).pack(side="right", padx=2)
    ttk.Button(act, text="Undo", command=undo).pack(side="right", padx=2)

    _load_existing()
    return win

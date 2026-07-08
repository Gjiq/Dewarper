#!/usr/bin/env python3
"""Dewarp — desktop GUI front-end.

A thin window over the same engine the command line uses (reconstruct.py via
dewarp_cli.py). It lets you:

  * pick an input folder of .jpg scans and an output folder,
  * see every page and, for any the classifier gets wrong, pin its type
    (Auto / FULL_BLEED / DARK_BG / TEXT / SIDE_TEXT / SKIP) — these are written
    to a dewarp_overrides.json the engine reads,
  * run the batch with a live log + progress bar,
  * and afterwards spot pages that came out SKIP (likely mis-classifications)
    so you can fix them and re-run.

Standard-library Tkinter only (plus OpenCV, already a pipeline dependency, for
thumbnails). No behavioural change to the engine — the GUI just drives it.
"""
import base64
import os
import queue
import subprocess
import sys
import tempfile
import threading

# ---------------------------------------------------------------------------
# Pure helpers (no Tk) — unit-testable on their own.
# ---------------------------------------------------------------------------

# The toolbar type buttons + the right-click Page-type menu use this one list.
TYPE_BUTTONS = [("Auto", "Auto"), ("Full Art", "FULL_ART"),
                ("Multi Art", "MULTI_ART"), ("Dark BG", "DARK_BG"),
                ("Text", "TEXT"), ("Side Text", "SIDE_TEXT")]
PAGE_TYPES = ["Auto", "FULL_BLEED", "FULL_ART", "MULTI_ART", "DARK_BG",
              "TEXT", "SIDE_TEXT", "SKIP"]
# Output JPEG quality presets (label -> value). Higher = larger files, closer to
# the original scan size.
QUALITY_PRESETS = [("Standard (smaller files)", 95),
                   ("High", 98),
                   ("Original quality (full colour, matches source)", 100)]
_DONE_TYPES = {"FULL_BLEED", "DARK_BG", "TEXT", "SIDE_TEXT", "TEXT_LEFT",
               "TEXT_RIGHT", "SKIP"}
_IMG_EXT = (".jpg", ".jpeg")


def list_pages(folder):
    """Sorted list of page image basenames in a folder. Excludes magenta markup
    sidecars (<stem>_guide.jpg etc.) — they pair with their original at run
    time but are not themselves pages to process."""
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    out = []
    for n in names:
        if not n.lower().endswith(_IMG_EXT):
            continue
        stem = os.path.splitext(n)[0].lower()
        if stem.endswith("_mag") or stem.endswith("_magenta") or stem.endswith("_guide"):
            continue
        out.append(n)
    return sorted(out)


def build_overrides(rows):
    """rows: iterable of (name, type_label). Returns {name: TYPE} for every row
    whose type is not 'Auto'."""
    out = {}
    for name, label in rows:
        if label and label != "Auto":
            out[name] = label
    return out


def app_dir():
    """Folder the app runs from (works both from source and when frozen)."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def asset_path(name):
    """Path to a bundled asset (icon.png), from PyInstaller's _MEIPASS when frozen."""
    base = getattr(sys, "_MEIPASS", None) or app_dir()
    return os.path.join(base, name)


def _engine_argv():
    """Base command that runs the engine, frozen-aware.
    Frozen: the sibling `dewarp` executable. From source: python dewarp_cli.py."""
    if getattr(sys, "frozen", False):
        exe = "dewarp.exe" if os.name == "nt" else "dewarp"
        return [os.path.join(app_dir(), exe)]
    return [sys.executable, os.path.join(app_dir(), "dewarp_cli.py")]


def build_command(inp, out, cached=False, overrides_path=None):
    """Full subprocess argv to reconstruct `inp` into `out`."""
    cmd = _engine_argv() + ["--input", inp, "--output", out]
    if cached:
        cmd.append("--cached-profile")
    if overrides_path:
        cmd += ["--overrides", overrides_path]
    return cmd


def parse_page_line(line):
    """Parse one engine stdout line. Returns (name, TYPE) for a processed page,
    (name, 'DONE') for a resume-skip, or None. TYPE is upper-case."""
    line = line.rstrip("\n")
    if ": " not in line:
        return None
    name, rest = line.split(": ", 1)
    name = name.strip()
    if not name.lower().endswith(_IMG_EXT):
        return None
    if rest.startswith("already done"):
        return (name, "DONE")
    tok = rest.strip().split()
    if tok:
        t = tok[0].strip().upper()
        if t in _DONE_TYPES:
            return (name, t)
    return None


def open_in_file_manager(path):
    """Open a folder in the OS file manager. Best-effort, cross-platform."""
    try:
        if sys.platform.startswith("darwin"):
            subprocess.Popen(["open", path])
        elif os.name == "nt":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception:
        pass


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

def _make_thumb_png_b64(path, max_side=460):
    """Load an image, downscale, return base64 PNG bytes (for tk.PhotoImage)."""
    import cv2
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    sc = min(1.0, float(max_side) / max(h, w))
    if sc < 1.0:
        img = cv2.resize(img, (max(1, int(w * sc)), max(1, int(h * sc))),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return None
    return base64.b64encode(buf.tobytes())


def run_gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    root = tk.Tk()
    root.title("Dewarp — scan cleanup")
    try:
        _appicon = tk.PhotoImage(file=asset_path("icon.png"))
        root.iconphoto(True, _appicon)
        root._appicon = _appicon
    except Exception:
        pass
    root.geometry("980x680")
    root.minsize(820, 560)

    style = ttk.Style()
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    ACCENT = "#2d6cdf"
    style.configure("Run.TButton", font=("TkDefaultFont", 11, "bold"),
                    padding=8)
    style.configure("Treeview", rowheight=24)

    state = {
        "input": tk.StringVar(),
        "output": tk.StringVar(),
        "cached": tk.BooleanVar(value=False),
        "quality": tk.StringVar(value=QUALITY_PRESETS[0][0]),
        "proc": None,
        "q": queue.Queue(),
        "total": 0,
        "done": 0,
        "thumb": None,     # keep a ref so Tk doesn't GC the preview
        "running": False,
    }

    # ---- top: folders + options -----------------------------------------
    top = ttk.Frame(root, padding=(12, 12, 12, 6))
    top.pack(fill="x")
    top.columnconfigure(1, weight=1)

    ttk.Label(top, text="Input folder").grid(row=0, column=0, sticky="w")
    in_entry = ttk.Entry(top, textvariable=state["input"])
    in_entry.grid(row=0, column=1, sticky="ew", padx=6)

    ttk.Label(top, text="Output folder").grid(row=1, column=0, sticky="w",
                                               pady=(6, 0))
    out_entry = ttk.Entry(top, textvariable=state["output"])
    out_entry.grid(row=1, column=1, sticky="ew", padx=6, pady=(6, 0))

    def choose_input():
        d = filedialog.askdirectory(title="Choose a folder of page scans")
        if d:
            state["input"].set(d)
            if not state["output"].get():
                state["output"].set(d.rstrip("/\\") + "_reconstructed")
            populate(d)

    def choose_output():
        d = filedialog.askdirectory(title="Choose an output folder")
        if d:
            state["output"].set(d)

    ttk.Button(top, text="Browse…", command=choose_input).grid(row=0, column=2)
    ttk.Button(top, text="Browse…", command=choose_output).grid(row=1, column=2,
                                                                 pady=(6, 0))
    ttk.Checkbutton(top, text="Reuse cached paper profile (faster re-runs)",
                    variable=state["cached"]).grid(row=2, column=1, sticky="w",
                                                    padx=6, pady=(6, 0))
    qframe = ttk.Frame(top)
    qframe.grid(row=3, column=1, sticky="w", padx=6, pady=(4, 0))
    ttk.Label(qframe, text="Output quality:").pack(side="left")
    qbox = ttk.Combobox(qframe, textvariable=state["quality"], state="readonly",
                        width=28, values=[lbl for lbl, _ in QUALITY_PRESETS])
    qbox.pack(side="left", padx=(6, 0))

    # ---- middle: page list (left) + preview (right) ----------------------
    mid = ttk.Panedwindow(root, orient="horizontal")
    mid.pack(fill="both", expand=True, padx=12, pady=6)

    left = ttk.Frame(mid)
    bulk_bar = ttk.Frame(left)
    bulk_bar.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))
    tree = ttk.Treeview(left, columns=("type", "opts"), selectmode="extended",
                        show="tree headings", height=16)
    tree.heading("#0", text="Page")
    tree.heading("type", text="Type")
    tree.heading("opts", text="Options")
    tree.column("#0", width=230, anchor="w")
    tree.column("type", width=130, anchor="center")
    tree.column("opts", width=130, anchor="center")
    vsb = ttk.Scrollbar(left, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=vsb.set)
    tree.grid(row=1, column=0, sticky="nsew")
    vsb.grid(row=1, column=1, sticky="ns")
    left.rowconfigure(1, weight=1)
    left.columnconfigure(0, weight=1)
    tree.tag_configure("skip", background="#ffe1e1")
    tree.tag_configure("override", foreground=ACCENT)
    mid.add(left, weight=3)

    right = ttk.Frame(mid)
    ttk.Label(right, text="Preview").pack(anchor="w")
    preview = ttk.Label(right, relief="groove", anchor="center",
                        text="\n\n(select a page)")
    preview.pack(fill="both", expand=True, pady=(4, 0))
    mid.add(right, weight=2)

    # ---- bottom: run + progress + log ------------------------------------
    bottom = ttk.Frame(root, padding=(12, 6, 12, 12))
    bottom.pack(fill="x")
    bottom.columnconfigure(1, weight=1)

    run_btn = ttk.Button(bottom, text="Reconstruct", style="Run.TButton")
    run_btn.grid(row=0, column=0, rowspan=2, padx=(0, 10))

    prog = ttk.Progressbar(bottom, mode="determinate")
    prog.grid(row=0, column=1, sticky="ew")
    status = ttk.Label(bottom, text="Choose an input folder to begin.")
    status.grid(row=1, column=1, sticky="w", pady=(2, 0))
    open_btn = ttk.Button(bottom, text="Open output",
                          command=lambda: open_in_file_manager(state["output"].get()))
    open_btn.grid(row=0, column=2, rowspan=2, padx=(10, 0))

    log = tk.Text(root, height=8, wrap="none", state="disabled",
                  font=("TkFixedFont", 9), background="#1e1e1e",
                  foreground="#d6d6d6")
    log.pack(fill="both", expand=False, padx=12, pady=(0, 12))

    def logline(s):
        log.configure(state="normal")
        log.insert("end", s + "\n")
        log.see("end")
        log.configure(state="disabled")

    # ---- page list population + type editing -----------------------------
    def populate(folder):
        tree.delete(*tree.get_children())
        pages = list_pages(folder)
        for name in pages:
            stem = os.path.splitext(name)[0]
            has_mag = os.path.isfile(os.path.join(folder, stem + "_guide.jpg")) or \
                os.path.isfile(os.path.join(folder, stem + "_mag.jpg"))
            label = ("✎ " + name) if has_mag else name
            tree.insert("", "end", iid=name, text=label, values=("Auto", ""))
        status.configure(text=f"{len(pages)} page(s) found. "
                              f"Set a type on any page the classifier gets wrong, "
                              f"then Reconstruct.")
        prog.configure(value=0, maximum=max(1, len(pages)))
        if pages:                                   # show the first page's preview
            tree.selection_set(pages[0])
            tree.focus(pages[0])
            selected_preview()

    def selected_preview(_evt=None):
        sel = tree.selection()
        if not sel:
            return
        name = sel[0]
        path = os.path.join(state["input"].get(), name)

        def work():
            data = _make_thumb_png_b64(path)

            def show():
                if not data:
                    preview.configure(image="", text="\n\n(cannot preview)")
                    return
                img = tk.PhotoImage(data=data)
                state["thumb"] = img            # keep ref
                preview.configure(image=img, text="")
            root.after(0, show)
        threading.Thread(target=work, daemon=True).start()

    tree.bind("<<TreeviewSelect>>", selected_preview)

    # per-row extra settings beyond the Type column
    row_opts = {}      # iid -> {"rotate": 0|90|180|270, "decurl": True/False}

    def _opts_text(iid):
        o = row_opts.get(iid, {})
        parts = []
        if o.get("rotate"):
            parts.append(f"\u21bb{o['rotate']}\u00b0")
        if o.get("decurl", True) is False:
            parts.append("no-curl")
        if o.get("inline_crop", True) is False:
            parts.append("no-crop")
        return " \u00b7 ".join(parts)

    def _refresh_row(iid):
        tree.set(iid, "opts", _opts_text(iid))
        typ = tree.set(iid, "type")
        tags = [t for t in tree.item(iid, "tags") if t not in ("override", "skip")]
        if typ not in ("", "Auto") or row_opts.get(iid):
            tags.append("override")
        tree.item(iid, tags=tags)

    def set_row_type(iid, val):
        tree.set(iid, "type", val)
        _refresh_row(iid)

    def _set_opt(iid, key, val, default):
        o = row_opts.setdefault(iid, {"rotate": 0, "decurl": True, "inline_crop": True})
        o[key] = val
        if o.get("rotate", 0) == 0 and o.get("decurl", True) is True \
                and o.get("inline_crop", True) is True:
            row_opts.pop(iid, None)
        _refresh_row(iid)

    def _target_rows():
        return list(tree.selection()) or list(tree.get_children())

    # ---- drag-to-select ----
    _dragsel = {"start": None}

    def on_tree_press(e):
        if not state["running"]:
            r = tree.identify_row(e.y)
            _dragsel["start"] = r or None

    def on_tree_drag(e):
        start = _dragsel.get("start")
        row = tree.identify_row(e.y)
        if not start or not row:
            return
        kids = list(tree.get_children())
        try:
            i0, i1 = kids.index(start), kids.index(row)
        except ValueError:
            return
        lo, hi = sorted((i0, i1))
        tree.selection_set(kids[lo:hi + 1])

    tree.bind("<ButtonPress-1>", on_tree_press, add="+")
    tree.bind("<B1-Motion>", on_tree_drag, add="+")

    # ---- right-click context menu ----
    def _apply(fn, *a):
        rows = _target_rows()
        for iid in rows:
            fn(iid, *a)
        status.configure(text=f"Updated {len(rows)} page(s).")
        update_highlights()

    menu = tk.Menu(tree, tearoff=0)
    _tmenu = tk.Menu(menu, tearoff=0)
    for _lbl, _val in TYPE_BUTTONS:
        _tmenu.add_command(label=_lbl,
                           command=lambda v=_val: _apply(set_row_type, v))
    menu.add_cascade(label="Page type", menu=_tmenu)
    _cmenu = tk.Menu(menu, tearoff=0)
    _cmenu.add_command(label="On", command=lambda: _apply(_set_opt, "decurl", True, True))
    _cmenu.add_command(label="Off", command=lambda: _apply(_set_opt, "decurl", False, True))
    menu.add_cascade(label="De-curl", menu=_cmenu)
    _imenu = tk.Menu(menu, tearoff=0)
    _imenu.add_command(label="On", command=lambda: _apply(_set_opt, "inline_crop", True, True))
    _imenu.add_command(label="Off", command=lambda: _apply(_set_opt, "inline_crop", False, True))
    menu.add_cascade(label="Inline crop", menu=_imenu)
    menu.add_separator()
    menu.add_command(label="Select all",
                     command=lambda: tree.selection_set(tree.get_children()))
    menu.add_command(label="\u270e Mark up\u2026", command=lambda: markup_selected())

    def on_tree_menu(e):
        if state["running"]:
            return
        row = tree.identify_row(e.y)
        if row and row not in tree.selection():
            tree.selection_set(row)
        try:
            menu.tk_popup(e.x_root, e.y_root)
        finally:
            menu.grab_release()

    tree.bind("<Button-3>", on_tree_menu)
    tree.bind("<Button-2>", on_tree_menu)

    # ---- bulk type buttons (act on selection, or all pages if none selected) --
    def bulk_set(val):
        rows = _target_rows()
        for iid in rows:
            set_row_type(iid, val)
        scope = f"{len(tree.selection())} selected" if tree.selection() else f"all {len(rows)}"
        status.configure(text=f"Set {scope} page(s) \u2192 {val}.")
        update_highlights()

    ACTIVE_BG = "#2d6cdf"
    ttk.Label(bulk_bar, text="Type:").pack(side="left", padx=(0, 4))
    type_btns = {}
    for _lbl, _val in TYPE_BUTTONS:
        b = tk.Button(bulk_bar, text=_lbl, width=8,
                      command=lambda v=_val: bulk_set(v))
        b.pack(side="left", padx=1)
        type_btns[_val] = b
    _DEF_BG = type_btns["Auto"].cget("background")
    _DEF_FG = type_btns["Auto"].cget("foreground")
    _DEF_ABG = type_btns["Auto"].cget("activebackground")

    def _decurl_toggle():
        rows = _target_rows()
        cur = {row_opts.get(i, {}).get("decurl", True) for i in rows}
        newv = False if cur == {True} else True
        for i in rows:
            _set_opt(i, "decurl", newv, True)
        update_highlights()

    def _inlinecrop_toggle():
        rows = _target_rows()
        cur = {row_opts.get(i, {}).get("inline_crop", True) for i in rows}
        newv = False if cur == {True} else True
        for i in rows:
            _set_opt(i, "inline_crop", newv, True)
        update_highlights()

    ttk.Separator(bulk_bar, orient="vertical").pack(side="left", fill="y", padx=8)
    decurl_btn = tk.Button(bulk_bar, text="De-curl", width=7, command=_decurl_toggle)
    decurl_btn.pack(side="left", padx=1)
    inlinecrop_btn = tk.Button(bulk_bar, text="Inline crop", width=9,
                               command=_inlinecrop_toggle)
    inlinecrop_btn.pack(side="left", padx=1)

    def mark_row_done(image_path):
        name = os.path.basename(image_path)
        # image_path is the ORIGINAL page; find its row and flag it
        if tree.exists(name) and not tree.item(name, "text").startswith("✎"):
            tree.item(name, text="✎ " + name)
        status.configure(text=f"Markup saved for {name} — used automatically on "
                              f"the next Reconstruct.")
        update_highlights()

    def markup_selected():
        sel = tree.selection()
        if not sel:
            messagebox.showinfo("Markup", "Select a page to mark up first.")
            return
        name = sel[0]
        path = os.path.join(state["input"].get(), name)
        # Re-marking a page re-opens its existing _guide for editing (does not reset).
        try:
            import dewarp_markup
            dewarp_markup.open_markup(root, path, on_saved=mark_row_done)
        except Exception as exc:                 # noqa: BLE001
            messagebox.showerror("Markup", f"Could not open editor:\n{exc}")

    mk_btn = tk.Button(bulk_bar, text="\u270e Mark up\u2026", width=11,
                       command=markup_selected)
    mk_btn.pack(side="left", padx=(12, 2))

    def update_highlights(_e=None):
        sel = tree.selection()
        types = {tree.set(i, "type") for i in sel} if sel else set()
        common = next(iter(types)) if len(types) == 1 else None
        for val, b in type_btns.items():
            on = (common == val)
            b.configure(bg=ACTIVE_BG if on else _DEF_BG,
                        fg="white" if on else _DEF_FG,
                        activebackground=ACTIVE_BG if on else _DEF_ABG)

        def _all_on(key):                            # True when whole selection is ON
            vals = {row_opts.get(i, {}).get(key, True) for i in sel} if sel else set()
            return len(vals) == 1 and True in vals

        for b, key in ((decurl_btn, "decurl"), (inlinecrop_btn, "inline_crop")):
            on = _all_on(key)                        # blue = ON (the default state)
            b.configure(bg=ACTIVE_BG if on else _DEF_BG,
                        fg="white" if on else _DEF_FG,
                        activebackground=ACTIVE_BG if on else _DEF_ABG)
        marked = bool(sel) and all(tree.item(i, "text").startswith("\u270e") for i in sel)
        mk_btn.configure(bg=ACTIVE_BG if marked else _DEF_BG,
                         fg="white" if marked else _DEF_FG,
                         activebackground=ACTIVE_BG if marked else _DEF_ABG)

    tree.bind("<<TreeviewSelect>>", update_highlights, add="+")

    # ---- per-page overrides (shared by Reconstruct and Save project) ----------
    def gather_overrides():
        ov = {}
        for iid in tree.get_children():
            typ = tree.set(iid, "type")
            o = row_opts.get(iid)
            has_type = typ not in ("", "Auto")
            if not has_type and not o:
                continue
            if o and (o.get("rotate") or o.get("decurl", True) is False
                      or o.get("inline_crop", True) is False):
                entry = {}
                if has_type:
                    entry["type"] = typ
                if o.get("rotate"):
                    entry["rotate"] = o["rotate"]
                if o.get("decurl", True) is False:
                    entry["decurl"] = False
                if o.get("inline_crop", True) is False:
                    entry["inline_crop"] = False
                ov[iid] = entry
            elif has_type:
                ov[iid] = typ
        return ov

    # ---- save / load project --------------------------------------------------
    def save_project():
        import json
        qval = dict(QUALITY_PRESETS).get(state["quality"].get(), 95)
        proj = {
            "dewarp_project": 1,
            "input": state["input"].get(),
            "output": state["output"].get(),
            "quality": qval,
            "cached_profile": bool(state["cached"].get()),
            "pages": gather_overrides(),
        }
        path = filedialog.asksaveasfilename(
            title="Save project", defaultextension=".dewarp",
            filetypes=[("Dewarp project", "*.dewarp"), ("JSON", "*.json"),
                       ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(proj, f, indent=2, ensure_ascii=False)
            status.configure(text=f"Project saved: {os.path.basename(path)} "
                                  f"({len(proj['pages'])} page setting(s)).")
        except Exception as exc:                 # noqa: BLE001
            messagebox.showerror("Save project", str(exc))

    def load_project():
        import json
        path = filedialog.askopenfilename(
            title="Load project",
            filetypes=[("Dewarp project", "*.dewarp"), ("JSON", "*.json"),
                       ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                proj = json.load(f)
        except Exception as exc:                 # noqa: BLE001
            messagebox.showerror("Load project", f"Could not read project:\n{exc}")
            return
        if not isinstance(proj, dict) or "pages" not in proj:
            messagebox.showerror("Load project", "Not a Dewarp project file.")
            return
        inp = proj.get("input", "")
        state["input"].set(inp)
        state["output"].set(proj.get("output", ""))
        state["cached"].set(bool(proj.get("cached_profile", False)))
        qv = proj.get("quality", 95)
        for lbl, v in QUALITY_PRESETS:
            if v == qv:
                state["quality"].set(lbl)
                break
        if inp and os.path.isdir(inp):
            populate(inp)
        row_opts.clear()
        missing = 0
        for name, val in proj.get("pages", {}).items():
            if not tree.exists(name):
                missing += 1
                continue
            if isinstance(val, dict):
                if val.get("type"):
                    set_row_type(name, val["type"])
                if val.get("rotate"):
                    _set_opt(name, "rotate", int(val["rotate"]), 0)
                if val.get("decurl", True) is False:
                    _set_opt(name, "decurl", False, True)
                if val.get("inline_crop", True) is False:
                    _set_opt(name, "inline_crop", False, True)
            else:
                set_row_type(name, str(val))
        update_highlights()
        msg = f"Project loaded: {os.path.basename(path)}."
        if not inp or not os.path.isdir(inp):
            msg += "  (Input folder not found — set it and re-open the project.)"
        elif missing:
            msg += f"  ({missing} saved page(s) not in this folder.)"
        status.configure(text=msg)

    ttk.Button(qframe, text="Save project\u2026", command=save_project).pack(
        side="left", padx=(16, 2))
    ttk.Button(qframe, text="Load project\u2026", command=load_project).pack(
        side="left", padx=2)

    # ---- run --------------------------------------------------------------
    def set_running(on):
        state["running"] = on
        run_btn.configure(text="Stop" if on else "Reconstruct")
        for w in (in_entry, out_entry):
            w.configure(state="disabled" if on else "normal")

    def start_run():
        inp = state["input"].get().strip()
        out = state["output"].get().strip()
        if not inp or not os.path.isdir(inp):
            messagebox.showerror("Dewarp", "Please choose a valid input folder.")
            return
        if not out:
            out = inp.rstrip("/\\") + "_reconstructed"
            state["output"].set(out)
        os.makedirs(out, exist_ok=True)

        ov = gather_overrides()
        ov_path = None
        if ov:
            import json
            fd, ov_path = tempfile.mkstemp(prefix="dewarp_ov_", suffix=".json")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(ov, f)

        for iid in tree.get_children():             # clear prior skip marks
            tags = [t for t in tree.item(iid, "tags") if t != "skip"]
            tree.item(iid, tags=tags)

        cmd = build_command(inp, out, state["cached"].get(), ov_path)
        qval = dict(QUALITY_PRESETS).get(state["quality"].get(), 95)
        env = dict(os.environ, PYTHONUNBUFFERED="1",
                   DEWARP_JPEG_QUALITY=str(qval))
        state["done"] = 0
        state["total"] = len(tree.get_children()) or 1
        prog.configure(value=0, maximum=state["total"])
        logline("$ " + " ".join(cmd))
        set_running(True)

        def worker():
            try:
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True,
                                     bufsize=1, env=env)
                state["proc"] = p
                for line in p.stdout:
                    state["q"].put(("line", line.rstrip("\n")))
                p.wait()
                state["q"].put(("done", p.returncode))
            except Exception as exc:                 # noqa: BLE001
                state["q"].put(("line", f"ERROR: {exc}"))
                state["q"].put(("done", -1))
            finally:
                if ov_path:
                    try:
                        os.remove(ov_path)
                    except OSError:
                        pass

        threading.Thread(target=worker, daemon=True).start()

    def stop_run():
        p = state.get("proc")
        if p and p.poll() is None:
            p.terminate()
        logline("(stopped)")

    def on_run_click():
        if state["running"]:
            stop_run()
        else:
            start_run()

    run_btn.configure(command=on_run_click)

    # ---- poll subprocess output -----------------------------------------
    def pump():
        try:
            while True:
                kind, payload = state["q"].get_nowait()
                if kind == "line":
                    logline(payload)
                    parsed = parse_page_line(payload)
                    if parsed:
                        name, t = parsed
                        state["done"] += 1
                        prog.configure(value=state["done"])
                        status.configure(
                            text=f"Processing… {state['done']}/{state['total']}  "
                                 f"({name})")
                        if t == "SKIP" and tree.exists(name):
                            tags = list(tree.item(name, "tags")) + ["skip"]
                            tree.item(name, tags=tags)
                elif kind == "done":
                    set_running(False)
                    rc = payload
                    skips = [iid for iid in tree.get_children()
                             if "skip" in tree.item(iid, "tags")]
                    if rc == 0:
                        msg = f"Finished — {state['done']} page(s)."
                        if skips:
                            msg += (f"  {len(skips)} page(s) were skipped "
                                    f"(highlighted) — pin a type and re-run if any "
                                    f"is real art.")
                        status.configure(text=msg)
                    else:
                        status.configure(text=f"Stopped/failed (exit {rc}). "
                                              f"See log below.")
        except queue.Empty:
            pass
        root.after(120, pump)

    root.after(120, pump)

    # Optional: a folder passed on the command line or dragged onto the app.
    for a in sys.argv[1:]:
        if os.path.isdir(a):
            state["input"].set(os.path.abspath(a))
            if not state["output"].get():
                state["output"].set(os.path.abspath(a).rstrip("/\\") + "_reconstructed")
            populate(state["input"].get())
            break

    root.mainloop()


if __name__ == "__main__":
    run_gui()

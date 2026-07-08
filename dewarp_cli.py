#!/usr/bin/env python3
"""Command-line / drag-and-drop front-end for the Dewarp Pipeline.

Ways to launch the compiled `dewarp` executable:

  * Drag a FOLDER of page scans onto the app  -> processes that folder; results go
    to a sibling folder "<folder>_reconstructed".
  * Drag a SINGLE page file onto the app       -> reconstructs it; the result
    "<name>_reconstructed.jpg" is written NEXT TO the original.
  * Drag SEVERAL loose page files onto the app -> reconstructs them together;
    results collect in "<folder-of-first-file>/dewarp_reconstructed".
  * Double-click the app                       -> asks you for a folder or file.
  * From a terminal:
        dewarp ./pages                 (a folder)
        dewarp page1.jpg page2.jpg     (loose files)
        dewarp -i ./pages -o ./out     (explicit in/out)

Magenta-marked pages supply box geometry and are reconstructed from the clean
original -- drop the marked page and its clean original together.
"""
import argparse
import os
import shutil
import sys
import tempfile

_IMG_EXT = ('.jpg', '.jpeg')


def _pause(friendly: bool) -> None:
    """Keep a double-clicked / drag-dropped console window open so messages stay readable."""
    if friendly:
        try:
            input("\nPress Enter to close this window...")
        except EOFError:
            pass


def _clean(s: str) -> str:
    return os.path.abspath(os.path.expanduser(s.strip().strip('"').strip("'")))


def _stage(files, folders):
    """Gather the chosen image files (and every image inside any chosen folder) into
    one fresh temp dir, so reconstruct.py's directory scan + magenta pairing work
    unchanged. Files are hard-linked when possible (instant, no copy) and only copied
    as a cross-drive fallback. Returns the temp dir path."""
    staging = tempfile.mkdtemp(prefix='dewarp_in_')
    srcs = list(files)
    for fo in folders:
        for nm in sorted(os.listdir(fo)):
            if nm.lower().endswith(_IMG_EXT):
                srcs.append(os.path.join(fo, nm))
    counts = {}
    for src in srcs:
        base = os.path.basename(src)
        if base in counts:                       # basename collision across folders
            counts[base] += 1
            stem, ext = os.path.splitext(base)
            base = f"{stem}__{counts[base]}{ext}"
        else:
            counts[base] = 0
        dst = os.path.join(staging, base)
        try:
            os.link(src, dst)                    # cheap: same filesystem, no copy
        except OSError:
            shutil.copy2(src, dst)               # cross-drive / no hard-link support
    return staging


def main() -> None:
    p = argparse.ArgumentParser(
        prog="dewarp",
        description="Reconstruct scanned art-annual pages (deskew, crop, whiten / "
                    "keep-black, retain text). Drag a folder, a single page, or "
                    "several pages onto the app.")
    p.add_argument("paths", nargs="*",
                   help="a folder, a page file, or several page files "
                        "(you can drag them onto the app)")
    p.add_argument("-i", "--input", dest="input_flag", metavar="DIR",
                   help="a folder of page scans (same as dragging a folder)")
    p.add_argument("-o", "--output", dest="output_flag", metavar="DIR",
                   help="where to write results")
    p.add_argument("--cached-profile", action="store_true",
                   help="reuse a cached paper profile if present")
    args = p.parse_args()

    out = args.output_flag
    paths = list(args.paths)
    if args.input_flag:
        paths.insert(0, args.input_flag)

    # "friendly" = launched by double-click or drag-and-drop (no -i/-o flags): prompt
    # when empty, auto-name the output, and pause at the end so the window stays open.
    friendly = (args.output_flag is None and args.input_flag is None)

    if not paths:
        friendly = True
        print("Dewarp Pipeline")
        print("---------------")
        entry = input("Paste a folder (or a page file), then press Enter:\n> ")
        if entry.strip():
            paths = [entry]

    files, folders = [], []
    for raw in paths:
        pth = _clean(raw)
        if os.path.isdir(pth):
            folders.append(pth)
        elif os.path.isfile(pth):
            if pth.lower().endswith(_IMG_EXT):
                files.append(pth)
            else:
                print(f"Skipping (not a .jpg page): {pth}")
        else:
            print(f"Skipping (not found): {pth}")

    if not files and not folders:
        print("Nothing to process.")
        return _pause(friendly)

    temp_dir = None
    if len(folders) == 1 and not files:
        # single folder -> feed it straight to the pipeline (unchanged behavior)
        inp = folders[0]
        if not out:
            out = inp.rstrip("/\\") + "_reconstructed"
    else:
        # one or many loose files (optionally mixed with folders) -> stage to a temp dir
        temp_dir = _stage(files, folders)
        if not out:
            if len(files) == 1 and not folders:
                out = os.path.dirname(files[0]) or "."          # result beside the original
            else:
                anchor = os.path.dirname(files[0]) if files else folders[0]
                out = os.path.join(anchor or ".", "dewarp_reconstructed")
        inp = temp_dir

    out = os.path.abspath(os.path.expanduser(out))
    os.makedirs(out, exist_ok=True)

    os.environ["DEWARP_INPUT"] = inp
    os.environ["DEWARP_OUTPUT"] = out
    if args.cached_profile:
        os.environ["DEWARP_CACHED_PROFILE"] = "1"
    os.environ.pop("DEWARP_IMPORT", None)

    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    n_in = len([f for f in os.listdir(inp) if f.lower().endswith(_IMG_EXT)])
    print(f"\ninput : {inp}  ({n_in} page(s))")
    print(f"output: {out}\n")

    ok = False
    try:
        import reconstruct  # noqa: F401  (importing runs the full reconstruction)
        ok = True
    except Exception as exc:  # keep the window open on failure
        import traceback
        traceback.print_exc()
        print(f"\nERROR: {exc}")
    finally:
        if temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

    if ok:
        print("\nFinished. Your reconstructed pages are in:")
        print(f"  {out}")
    _pause(friendly)


if __name__ == "__main__":
    main()

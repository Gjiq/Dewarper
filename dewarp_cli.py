#!/usr/bin/env python3
"""Command-line / drag-and-drop front-end for the Dewarp Pipeline.

Three ways to launch the compiled `dewarp` executable:

  * Drag a folder of page scans onto the app  -> it processes that folder and
    writes results to a sibling folder "<folder>_reconstructed".
  * Double-click the app                       -> it asks you for the folder.
  * From a terminal                            -> dewarp -i ./pages -o ./out

Magenta-marked pages (<page>_mag.jpg) supply box geometry and are reconstructed
from the clean original.
"""
import argparse
import os
import sys


def _pause(friendly: bool) -> None:
    """Keep a double-clicked console window open so messages stay readable."""
    if friendly:
        try:
            input("\nPress Enter to close this window...")
        except EOFError:
            pass


def main() -> None:
    argv = sys.argv[1:]

    p = argparse.ArgumentParser(
        prog="dewarp",
        description="Reconstruct scanned art-annual pages (deskew, crop, whiten / "
                    "keep-black, retain text).",
    )
    p.add_argument("folder", nargs="?",
                   help="folder of page scans (you can drag it onto the app)")
    p.add_argument("-i", "--input", dest="input_flag", metavar="DIR",
                   help="same as the folder argument")
    p.add_argument("-o", "--output", dest="output_flag", metavar="DIR",
                   help="where to write results (default: <folder>_reconstructed)")
    p.add_argument("--cached-profile", action="store_true",
                   help="reuse a cached paper profile if present")
    args = p.parse_args()

    inp = args.input_flag or args.folder
    out = args.output_flag

    # "friendly" = launched by double-click or by dragging a folder onto the app,
    # i.e. not a normal terminal invocation with -i/-o. In that mode we prompt,
    # auto-name the output, and pause at the end.
    friendly = (not argv) or (args.folder is not None and args.input_flag is None
                              and args.output_flag is None)

    if not inp:
        friendly = True
        print("Dewarp Pipeline")
        print("---------------")
        inp = input("Paste the folder that holds your page scans, then press Enter:\n> ")
        inp = inp.strip().strip('"').strip("'")

    if not inp:
        print("No folder given -- nothing to do.")
        return _pause(friendly)

    inp = os.path.abspath(os.path.expanduser(inp))
    if not os.path.isdir(inp):
        print(f"That is not a folder:\n  {inp}")
        return _pause(friendly)

    if not out:
        out = inp.rstrip("/\\") + "_reconstructed"
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

    print(f"\ninput : {inp}")
    print(f"output: {out}\n")

    try:
        import reconstruct  # noqa: F401  (importing runs the full reconstruction)
    except Exception as exc:  # keep the window open on failure
        import traceback
        traceback.print_exc()
        print(f"\nERROR: {exc}")
        return _pause(friendly)

    print("\nFinished. Your reconstructed pages are in:")
    print(f"  {out}")
    _pause(friendly)


if __name__ == "__main__":
    main()

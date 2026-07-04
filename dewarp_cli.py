#!/usr/bin/env python3
"""Command-line front-end for the Dewarp Pipeline.

Wraps ``reconstruct.py`` so the tool takes normal command-line arguments
(``--input`` / ``--output``) instead of ``DEWARP_INPUT`` / ``DEWARP_OUTPUT``
environment variables. This is the entry point the compiled executable uses.

Examples
--------
    dewarp --input ./pages --output ./out
    dewarp -i ./pages -o ./out --cached-profile
"""
import argparse
import os
import sys


def main() -> None:
    p = argparse.ArgumentParser(
        prog="dewarp",
        description="Reconstruct scanned art-annual pages: deskew, deep-crop, "
                    "flat-field whiten (or keep-black for dark pages) and retain "
                    "page text. Magenta-marked pages (<page>_mag.jpg) supply box "
                    "geometry and are reconstructed from the clean original.",
    )
    p.add_argument("-i", "--input", required=True, metavar="DIR",
                   help="folder of page scans (*.jpg), including any *_mag.jpg markups")
    p.add_argument("-o", "--output", required=True, metavar="DIR",
                   help="folder to write <page>_reconstructed.jpg / _text.jpg into")
    p.add_argument("--cached-profile", action="store_true",
                   help="reuse a cached paper profile if one is present "
                        "(sets DEWARP_CACHED_PROFILE=1)")
    args = p.parse_args()

    in_dir = os.path.abspath(args.input)
    out_dir = os.path.abspath(args.output)
    if not os.path.isdir(in_dir):
        p.error(f"input folder does not exist: {in_dir}")
    os.makedirs(out_dir, exist_ok=True)

    # reconstruct.py reads these at import time and runs the batch driver on
    # import, so they must be set *before* it is imported below.
    os.environ["DEWARP_INPUT"] = in_dir
    os.environ["DEWARP_OUTPUT"] = out_dir
    if args.cached_profile:
        os.environ["DEWARP_CACHED_PROFILE"] = "1"
    os.environ.pop("DEWARP_IMPORT", None)  # ensure the batch actually runs

    # Allow running both as a plain script and as a frozen (PyInstaller) binary.
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    print(f"input : {in_dir}")
    print(f"output: {out_dir}\n")

    import reconstruct  # noqa: F401  (importing runs the full reconstruction)


if __name__ == "__main__":
    main()

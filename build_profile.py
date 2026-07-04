#!/usr/bin/env python3
"""
build_profile.py -- OPTIONAL: pre-build / inspect the off-white paper profile for
a folder of images.

The pipeline rebuilds the profile automatically over the staged batch on every
run, so this script is NOT required. It is handy only when you want to see the
s_max/dv numbers for a set of files ahead of time, or pre-write paper_profile.json.
The profile (s_max/dv envelope) is most stable when built over the WHOLE batch you
intend to process together, so point this at all of those files at once.

Usage:
    python3 build_profile.py <dir-or-glob> [<dir-or-glob> ...]
    python3 build_profile.py /home/claude/work/input
    python3 build_profile.py '/path/*.jpg'
    python3 build_profile.py                 # defaults to ./_PAGES if present

Writes paper_profile.json next to this script, stamped with the page count and
build time.
"""
import sys
import os
import glob
import deskew_crop

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'paper_profile.json')


def collect(args):
    if not args:
        args = [os.path.join(HERE, '_PAGES')]
    paths = []
    for a in args:
        if os.path.isdir(a):
            paths += glob.glob(os.path.join(a, '*.jpg'))
            paths += glob.glob(os.path.join(a, '*.jpeg'))
            paths += glob.glob(os.path.join(a, '*.JPG'))
        else:
            paths += glob.glob(a)
    return sorted(p for p in paths if p.lower().endswith(('.jpg', '.jpeg')))


def main(argv):
    paths = collect(argv)
    if not paths:
        print('build_profile: no jpegs found; nothing to do.')
        return 1
    prof = deskew_crop.write_paper_profile(paths, OUT)
    print(f"paper profile built over {prof['n_pages']} files -> {OUT}")
    print(f"  s_max={prof['s_max']}  dv={prof['dv']}  ({prof['built_at']})")
    if prof['n_pages'] < 20:
        print('  NOTE: few files -- the envelope is most stable over a larger '
              'batch. Build over all the files you will process together.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))

# Dewarp Pipeline

Reconstruction tools for scanned fantasy/sci-fi **art-annual pages** (built against
*Spectrum* volumes). It deskews, deep-crops and flat-field-whitens each printed
artwork so it sits axis-aligned on its original ground — whitened paper for paper
pages, kept **black** for art-on-black pages — with all page text retained.

> **Assisted, not fully autonomous.** Ordinary single plates run close to hands-off,
> but the hard cases (mixed media, heavy curl, two-page spreads, edge-case
> classification) are driven by a manual **magenta-markup** workflow and benefit from
> a human eyeball on the final result. See *Honest limitations* below.

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Requires Python 3.9+ and OpenCV / NumPy / SciPy.

## Usage

### Batch reconstruction

Using the command-line front-end (recommended; this is what the compiled binary runs):

```bash
python dewarp_cli.py --input ./pages --output ./out
# or the compiled executable:
dewarp --input ./pages --output ./out          # add --cached-profile to reuse a profile
```

Equivalently, via environment variables:

```bash
DEWARP_INPUT=/path/to/pages DEWARP_OUTPUT=/path/to/out python reconstruct.py
```
Each page is classified (`FULL_BLEED | TEXT | TEXT_LEFT | DARK_BG | SKIP`) and the
matching reconstruction is written as `<page>_reconstructed.jpg` (or `_text.jpg`).

### Magenta-markup override (manual guides)
Draw magenta guides on a **pixel-aligned copy** of a scan (name it `<page>_mag.jpg`)
and stage it next to the clean original. A closed box = an art plate; an open box or
a stack of horizontal lines over a text column = a **warp guide** (the strokes trace
how the text is skewed/curled, not a crop boundary). Boxes supply geometry only — the
output is rebuilt from the clean original's pixels.

## Files
| file | role |
|------|------|
| `dewarp_cli.py` | command-line front-end (`--input/--output`); compiled-binary entry point |
| `reconstruct.py` | classify + reconstruct each page (env-var batch driver) |
| `deskew_crop.py` | deskew / deep-crop / rectify / compositing |
| `dewarp_text.py` | text-page curl correction + flat-field whitening |
| `text_blocks.py` | paragraph/credit-block detection & straightening |
| `magenta_crop.py` | magenta box detection, magenta-guided reconstruction |
| `magenta_text.py` | magenta text-warp guides (stacked, both polarities) |
| `dark_art.py` | art-on-black (keep-black) reconstruction |
| `page_dewarp.py` | page-level curl pre-pass |
| `build_profile.py` | paper-profile helper |
| `START_HERE.txt` | full design notes + per-version changelog |

## Honest limitations
- The classifier has a recurring failure mode (vivid full-bleed paintings read as
  blank/decorative and get dropped to `SKIP`); it's patched per-page via overrides,
  not by loosening the classifier, so new volumes still need spot checks.
- Guided output is only as accurate as the drawn guide.
- Verify outputs visually — automated metrics catch gross failures, not fine detail.

## ⚠️ Copyright
The *Spectrum* page scans and reconstructed images are **copyrighted artwork** and are
**not** part of this repository (`.gitignore` blocks all image formats). This repo is
the **code only**. Do not commit scans or outputs; keep them out of version control.

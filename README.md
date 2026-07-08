# Dewarp

Clean up **scanned pages** — books, comics, art annuals, magazines, documents. It
deskews each page, deep-crops and flat-field-whitens the paper (or keeps a **black**
ground on art-on-black pages), splits pages that hold several images, gently removes
binding curl, and leaves the page text in place. One finished JPEG per page.

> **Assisted, not fully autonomous.** Ordinary pages run close to hands-off, but the
> hard cases (mixed media, heavy curl, two-page spreads, edge-case classification) are
> helped by a manual **magenta-markup** workflow and benefit from a human eyeball on the
> result. See *Honest limitations* below.

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Requires Python 3.9+ and OpenCV / NumPy / SciPy.

## Usage

### Desktop app (GUI)

For a point-and-click workflow, run the GUI:

```bash
python dewarp_gui.py            # or the compiled dewarp-gui executable
```

Pick an input folder of `.jpg` scans and an output folder, then **Reconstruct**.

**Save project… / Load project…** store the whole setup — input/output folders, output
quality, cached-profile flag, and every page's type and settings — in a small `.dewarp`
text file (JSON) you can reopen later to pick up exactly where you left off. Markup
`_mag.jpg` files live in the input folder, so they're restored automatically.
Every page is listed; for any the classifier gets wrong you can set its type in the
**Type** column (Auto / FULL_BLEED / DARK_BG / TEXT / SIDE_TEXT / SKIP) — those
selections are written to an overrides file the engine reads, no file-editing needed.
The toolbar's **Type** buttons — **Auto / Full Art / Multi Art / Dark BG / Text / Side Text** — plus **De-curl** and **Inline crop** toggles act on the selected rows (or all pages if none are selected) and **highlight blue** to show the current state: Auto, De-curl and Inline crop are on by default, so they read blue until you change them. Select multiple rows (drag, or Ctrl/Shift-click) and a button applies to all of them. **Right-click** gives the same actions plus **Select all**, or open the markup editor. An **Output quality** selector trades file size against fidelity, and
**✎ Mark up…** opens the drawing editor (below) for the selected page. A live log and
progress bar show the run, and pages
that came out `SKIP` are highlighted afterwards so you can pin a type and re-run if one
was real art. You can also pass or drag a folder onto it: `dewarp-gui ./pages`.

### Batch reconstruction

Command-line front-end (this is what the compiled binary runs):

```bash
python dewarp_cli.py --input ./pages --output ./out
# or the compiled executable:
dewarp --input ./pages --output ./out          # add --cached-profile to reuse a profile
```

Drag-and-drop also works: drop a folder, a single page, or several loose pages onto the
`dewarp` app. Equivalently, via environment variables:

```bash
DEWARP_INPUT=/path/to/pages DEWARP_OUTPUT=/path/to/out python reconstruct.py
```

Inputs are `.jpg` page scans. Each page is classified
(`FULL_BLEED | TEXT | SIDE_TEXT | DARK_BG | SKIP`) and **one file is written per page**,
named for what the page turned out to be:

| output | written for |
|--------|-------------|
| `<page>_reconstructed.jpg` | art pages — full-bleed, art-on-black, and text-beside-art (`FULL_BLEED`, `DARK_BG`, `SIDE_TEXT`) |
| `<page>_text.jpg` | full text / credit / body-copy pages (`TEXT`) |
| `<page>_skipped.jpg` | blank leaves and skipped pages (`SKIP`), written back so every input page has an output |

A `paper_profile.json` (the per-batch paper-tone envelope) is also written to the
working area; it is a run artifact, not a deliverable.

### What it handles automatically

Beyond per-page deskew, deep-crop and flat-field whitening, no markup needed:

- **Multi-image pages** — a page holding several pictures is split into its individual
  images (each deskewed on its own), then composited back onto the one page.
- **Stacked-image split + overlap resolution** — two vertically-stacked images that
  would otherwise fuse into one oversized box are separated, and overlaps reconciled.
- **Page-curl pre-pass** — binding curl is measured from clean image edges and gently
  removed, with a sanity cap that no-ops a degenerate field rather than tearing the page.
- **Art-on-black keyline deskew** — pages with a printed frame on a dark ground are
  squared to that keyline, with the black background preserved.
- **Per-batch paper profile** — paper tone is profiled across the whole folder so
  whitening is consistent; `--cached-profile` reuses a saved profile (faster, and keeps
  tone identical across resumed runs). Build one profile **per batch** — don't reuse a
  profile across unrelated scan sets.

### Resuming a run

Re-running over the same output folder **skips pages that already have a
`_reconstructed`/`_text` output** (`already done — skipped`) and continues where it left
off — handy for large batches or interrupted runs. (`_skipped.jpg` pages are
re-evaluated on a re-run, since they have no reconstructed output.)

### Output size / quality

Reconstructions keep the **original pixel resolution**; only the JPEG quality differs, which is why files come out smaller than the source scans. Raise it with `--quality N` (1-100, default 95) or the GUI's **Output quality** selector. The top option, **Original quality (100)**, encodes at full quality with **no chroma subsampling (4:4:4)** — cv2's default even at q100 uses 4:2:0, which softens colour and halves file size, so this option is what actually matches the source scan's fidelity. Example: `dewarp -i ./pages -o ./out --quality 100`.

### Fixing a mis-classified page (overrides)

Most pages need nothing. When the classifier gets a specific page wrong (e.g. a vivid
full-page painting read as blank and dropped to `SKIP`), pin that page's type in an
**external overrides file** — overrides are data for the job, never edits to the code:

- Pass it explicitly: `dewarp -i ./pages --overrides fixes.json`, **or**
- Drop a `dewarp_overrides.json` (or `.txt`) into the input folder — picked up
  automatically.

Format — JSON:

```json
{ "cover.jpg": "FULL_BLEED", "blank_leaf.jpg": "SKIP" }
```

or plain text (`#` starts a comment):

```
cover.jpg      = FULL_BLEED
blank_leaf.jpg : SKIP
```

Each value may be a page type, or an object with per-page settings — `{"type": "FULL_ART", "rotate": 90, "decurl": false}` — where `rotate` is 90/180/270° (clockwise, for sideways scans) and `decurl:false` turns off the page-curl pass for that page. Valid types: `FULL_BLEED | FULL_ART | MULTI_ART | TEXT | SIDE_TEXT | DARK_BG | SKIP`. `FULL_ART` forces one panel; `MULTI_ART` forces the multi-picture split; plain `FULL_BLEED` auto-decides. `SIDE_TEXT` is a text column beside the art, either side (`TEXT_LEFT`/`TEXT_RIGHT` are accepted aliases). Unknown types are ignored
with a warning. A starter template is in `presets/dewarp_overrides.example.txt`, and the
pins used for the *Spectrum* art-annual runs are in `presets/spectrum_overrides.json`
(pass it with `--overrides` to reproduce those).

### Marking up a page (draw the guides)

Select a page and click **✎ Mark up…** to open the editor. One thickness box (5–25 px) applies to whatever tool is active:

- **Box** — drag to make a magenta rectangle with four draggable corners (as many as you like). Over art it's dewarped; over text it's deskewed (decided by content).
- **Line** — a single straight magenta line.
- **Pen** — freehand line; along the text rows these are warp guides that straighten a bowed column.
- **Bow** — click two spots on a straight **Line** to drop anchors, then drag between them to bend it into a curve (apex at where you start dragging, so it can be left- or right-heavy) — handy for tracing a bowed row of text.
- **Eraser** — drag over a line or a box to delete it.
- **Fill** — click inside a box, or inside an area closed by lines, to fill it solid; a black ✕ in the fill deletes it. (Lines needn't meet perfectly — the engine groups open edges into regions.)

On **Apply** the marks are composited onto a full-resolution copy and saved as `<page>_guide.jpg` next to the original (with a small `_guide.json` of the editable strokes). The next **Reconstruct** pairs it and rebuilds from the clean pixels. Marked pages show a ✎; opening Mark up again **re-opens** that guide so you can adjust it (it no longer resets).

### Magenta-markup override (manual guides)
Draw magenta guides on a **pixel-aligned copy** of a scan (name it `<page>_mag.jpg`)
and stage it next to the clean original. A closed box = one image; an open box or a
stack of horizontal lines over a text column = a **warp guide** (the strokes trace how
the text is skewed/curled, not a crop boundary). Boxes supply geometry only — the output
is rebuilt from the clean original's pixels. If you're handed a magenta page **without**
its clean original, stop and get the original first — the marked-up copy isn't shippable.

## Files
| file | role |
|------|------|
| `dewarp_gui.py` | desktop GUI front-end (folder pickers, type overrides, markup, live progress) |
| `dewarp_markup.py` | in-app markup editor (skew-box quads, free lines, enclosed-region fill) |
| `dewarp_cli.py` | command-line / drag-drop front-end (`--input/--output/--overrides`); compiled-binary entry point |
| `reconstruct.py` | classify + reconstruct each page (env-var batch driver); loads external overrides |
| `deskew_crop.py` | deskew / deep-crop / rectify / compositing |
| `dewarp_text.py` | text-page curl correction + flat-field whitening |
| `text_blocks.py` | paragraph/credit-block detection & straightening |
| `magenta_crop.py` | magenta box detection, magenta-guided reconstruction |
| `magenta_text.py` | magenta text-warp guides (stacked, both polarities) |
| `dark_art.py` | art-on-black (keep-black) reconstruction + keyline deskew |
| `page_dewarp.py` | page-level curl pre-pass |
| `build_profile.py` | paper-profile helper |
| `presets/` | override presets: an example template + the *Spectrum* pin set |
| `HOW_TO_USE.txt` | plain-language run instructions (ships next to the exe) |
| `START_HERE.txt` | full design notes + per-version changelog |

## Honest limitations
- The classifier has a recurring failure mode (vivid full-page images read as
  blank/decorative and get dropped to `SKIP`); it's fixed per-page via an overrides
  file, not by loosening the classifier — so a new scan set still warrants a spot check.
- Guided output is only as accurate as the drawn guide.
- Verify outputs visually — automated metrics catch gross failures, not fine detail.
- Inputs are `.jpg`; camera-angle perspective unwarping is out of scope (this targets
  flatbed/overhead scans, not photos taken at an angle).

## ⚠️ Copyright
Page scans and reconstructed images may be **copyrighted**. They are **not** part of
this repository (`.gitignore` blocks all image formats) — this repo is the **code only**.
Don't commit scans or outputs; keep them out of version control, and only process
material you have the right to.

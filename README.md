# Pharmaceutical Packaging Artwork Comparison Application

Offline desktop tool for comparing "original" vs "suspect" pharmaceutical packaging
artwork (PDF/image) at pixel, text, graphics, and barcode/Braille level. See
`SRS_Artwork_Comparison_App .docx` for requirements and `Architecture_Document.md`
for the technical design this codebase follows.

It never uses the network: OCR, barcode decoding and PDF rendering are all local, and
`--self-test` (below) proves it by blocking every network attempt while it runs.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e ".[dev]"
```

**Tesseract OCR** (used for text/number comparison) is a separate native program that
pip cannot install. Install it (Windows: the UB-Mannheim installer) and the application
finds it by itself -- in `TESSERACT_CMD`, next to the application, on `PATH`, or in the
usual `C:\Program Files\Tesseract-OCR` folder. Without it everything else still works,
and the results page says plainly that text could not be compared.

## Run

```bash
python -m app.main
```

Load an original and a suspect file (PDF, PNG, JPEG, TIFF, BMP), optionally use
**Edit Artwork Area** on either to crop, lift the artwork from its sheet, place an outline, or
erase regions, choose a mode, and press **Compare**. The comparison runs in a separate process, so the window
stays responsive and **Cancel** stops it at once.

| Mode | Compares |
| --- | --- |
| Automatic | Text, colour and graphics of two digital files (barcodes and Braille as pixels) |
| Scan vs Digital | Where *ink* is, tolerating a scan's softness, noise, tilt and colour cast |
| Barcode / Pharmacode / QR | Only the decoded symbols, grouped by type with a pass/fail mark each |
| Braille | Only the coloured Braille dots: missing, added, moved |

Viewing: side-by-side, flicker, overlay and grayscale views from one aligned pair, with
scroll lock; pinch or Ctrl + wheel to zoom about the cursor; PDFs stay sharp at any zoom.
In the crop dialog, drag any edge or corner of the selection to resize it, drag inside to
move it, Space + drag (or middle-drag) to pan, and confirm the crop when saving.

## Comparing only the artwork

Packaging files are usually a *sheet*: the carton's die-cut outline with the printed panels
inside it and, around it, a title, a colour legend, crop marks, notes. Only what is inside the
outline is artwork. In **Edit Artwork Area**:

* **Lift Subject** -- press and hold on the artwork (a ring fills while you hold). Only the area
  inside its closed border line is lifted; the flaps, which are the colour of the paper, are in, and
  the title, legend, crop marks and notes are out. For a PDF its vector lines are used, in any
  colour (a dieline is often a spot colour); otherwise the dark lines of the picture. A photograph
  or scan of artwork on a surface is lifted by colour and background instead. *Method* forces one
  or the other; *Auto-detect artwork* lifts the largest closed border without a press.
  **+ Keep / − Drop** paint corrections over what the outline missed or wrongly took.
* **Save cutout...** writes the lifted artwork as a lossless PNG with a transparent background.
* **Place Outline** is for a file with no clear border of its own (a photo, a plain scan): lay the
  other file's lifted outline (or a saved cutout) over it, **Auto-fit** snaps it onto the artwork by
  matching what is printed there, drag / rotate / scale to adjust, then **Trim to outline**.

The comparison then covers only that area, and the results page dims everything left out (toggle
**Show compared area**). A lifted or placed outline masks the comparison but never registers the
files -- the artwork's own features do, so the reported alignment residual stays honest.

## Configuration

An optional `config.json` (in the working directory, beside the executable, or in
`%LOCALAPPDATA%\ArtworkCompare`) overrides any subset of the defaults in
`app/config.py`. The ones QA is most likely to touch:

```json
{
  "alignment": {"residual_threshold_px": 2.0},
  "clustering": {"distance_px": 30},
  "runtime": {"max_file_size_mb": 250, "max_pdf_pages": 100,
              "max_render_megapixels": 160, "worker_recycle_after_jobs": 20,
              "job_timeout_s": 600},
  "subject": {"hold_ms": 350, "border_dark_level": 115, "border_min_area_fraction": 0.08}
}
```

Oversized input is refused before anything large is allocated, with a message saying
which limit and how to change it.

## Test

```bash
pytest
```

Real artwork is confidential and never committed. To also check against real files, put
them in `sample test files/` (git-ignored) with a `pairs.json` manifest; see
`tests/test_real_artwork.py`. Expected answers are derived from the pixels, and scans are
simulated (`tests/scan_simulator.py`), so nothing is tuned to any one file.

`tests/golden_pairs/` is where hand-verified original/suspect artwork fixture pairs go
for the regression suite (Architecture §9).

## Package a single offline executable

```bash
pyinstaller app/packaging/pyinstaller.spec --noconfirm --clean
dist\ArtworkCompare.exe --self-test --report self-test.txt
```

The self-test starts the real worker process, renders a PDF, runs a comparison and reads a
Pharmacode with all network access blocked in both processes; exit status 0 (and
`PASSED` in the report) means the build works. To ship OCR inside the executable, place a
Tesseract install at `vendor/tesseract` before building. `ARTWORK_ONEDIR=1` builds a
folder instead of one file (faster to start).

## Known limits and open items

* **Text comparison needs Tesseract**, and has only been exercised against the synthetic
  suite where it is installed.
* **Scans:** detection thresholds are measured from each image (noise, softness,
  registration), but have been validated on *simulated* scans, not a range of real
  scanners. Fine print softer than the scan can hold is reported as unverifiable rather
  than compared.
* **Lifting the artwork** needs a *closed* outline. A break wider than about 5 px in a die-cut
  line, or an outline that is not dark and not a PDF stroke, falls back to colour (or reports
  that no border was found); Keep / Drop then correct it by hand. Lifting has been checked on
  synthetic sheets and photographs and on the sample carton, not on a wide range of real files.
* **Pharmacode** needs four or more bars (value 15 upward) to be read; vertical symbols
  are read from the top.
* **SRS section 7 items still open** and configurable rather than decided: the clustering
  distance, the severity tiers and colours, whether results must be exportable as an audit
  report, the Braille dot colour convention, batch mode, and accounts / audit trail.

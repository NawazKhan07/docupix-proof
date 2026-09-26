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
**Edit Artwork Area** on either to crop, lift the artwork from its sheet, place the other file's lifted
artwork over it, or erase regions, choose a mode, and press **Compare**. The comparison runs in a separate process, so the window
stays responsive and **Cancel** stops it at once.

| Mode | Compares |
| --- | --- |
| Automatic | Text, colour and graphics of two digital files (barcodes and Braille as pixels); finds the artwork itself if neither file was cropped or lifted |
| Scan vs Digital | What is *printed* -- above all its text, and barcode bars as marks -- not its colour; graphics on request; see below |
| Barcode / Pharmacode / QR | Only the decoded symbols, grouped by type with a pass/fail mark each |
| Braille | Only the coloured Braille dots (red or magenta by default, and round and dot-sized): missing, added, moved |
| Series (repeated artwork) | One artwork (cropped or lifted in the Original) against every repeat of it found on a sheet in the Suspect |
| Leaflet (multi-page) | Every page of a multi-page document, open or closed, in any language the reader has been given; see below |

**Automatic** finds differences in the *pixels* and only then reads them. Where neither file was cropped or
lifted, the artwork itself is found first on each sheet (a closed border, or the die-cut outline) and only it
is compared; the results page says so. The two aligned files are then compared pixel by pixel at full size,
tile by tile (`detectors/local_diff.py`), each tile registered on its own (a carton bent by its folds is off
by a different amount in different places); a pair that registers *exactly* (residual 0.25 px or less)
forgives nothing, so a tolerance never hides a `0` become a `1`. Every difference found is then checked by
what is inside its box: if the two files' marks (their ink) coincide and their average colour is the same, it
is the same artwork drawn a little differently -- one file softer than the other, a picture of the type
against live type, a fraction of a pixel of registration -- and it is dropped. (On one real carton pair this
removed 14 of 19 false "colour changes": hairlines round letters that were the same in both files.) A die
line or fold line one file draws (or draws in another colour) and the other does not is ignored too, wherever
it lies on the edge of the compared area or has nothing printed beside it -- a cut/crease line, whether drawn
as one long stroke or a dashed one. Reading never *starts* a finding: the words in each finding's own box are
read in both files, alone, and if they differ the finding says so ("'36' -> '24'", "'Feed' missing from
suspect"). Where the words read the *same* in both files, what differs about how they are set is still said --
font colour, weight, size, a different typeface, or the same text moved a little -- from the pixels of the
words themselves, never from the reading. A graphic that is missing here and added a short distance away is
one finding, that it moved, not two. Two readings of a whole page group their lines differently, and comparing
them reported areas that had not changed; that cannot happen when the pixels are the same. A finding is
**described**, not ranked: what kind it is -- text, colour, graphics, barcode, Braille -- decides its colour on
the page and in the list, and what happened to it (changed / moved / added / removed) is said in words.

A few more things make Automatic mode both finer and quieter than a plain overlay:

* **Exact borders.** The pixels the local comparison found different give each finding an exact border: the box
  is fitted to them (with a 1-2 px halo), so it does not sit a few pixels off the change, and it does not grow to
  cover a whole area unless the whole area really is what changed. A **recoloured area** (a panel, a tint) is its
  own finding, at its own exact edge, and what is written on it is findings of their own -- so if a whole panel's
  colour changed, that is one tight box, and a digit that also changed inside it is a second, separate finding,
  not excluded from the report.
* **Colour profiles are not changes.** Two exports of one artwork often render a solid ink a few dE apart across
  the whole page (a wordmark's pink or a black word). Where the marks coincide, a colour difference up to
  `detection.residue_max_ink_colour_delta` (5) is the same artwork (more for a thin box, where a moved edge shifts
  the average: contrast x 2 px / width), and a *shape* finding over the same marks yields to the colour comparison
  up to `residue_max_shape_colour_delta` (8).
* **Findings are not chained together just because they are near each other.** Two differences a few millimetres
  apart are two findings, even down a whole side of a carton; the only exceptions are the bars of a barcode or the
  dots of Braille (joined by distance), the letters and words of one changed line of type, and the same place found
  by two detectors. Where several small ones really are one -- the bars of a barcode drawn as marks, say -- merge
  them by hand (below).
* **Text on its side is read**: a finding of up to 100 px where nothing reads upright is read again turned a
  quarter either way (a serial number down a flap), with the window enlarged 4x for type that small.

**Scan vs Digital** compares **text** (and barcode bars as marks) and ignores colour completely, because a scan
changes every colour. Each file (inside the area you lifted) is turned into *ink on paper*: every background
colour -- a tint, a gradient, the scan's cream paper, uneven lighting -- comes out white, and everything
printed on it comes out black, whatever its colour (cyan, yellow and orange lettering included). A solid dark
bar or block stays a black block with its lettering white; whether a region is solid is a smooth function of
how dark it looks, so a panel a few levels lighter in a scan is a little less black, never the opposite colour.
Then:

* **Each picture is levelled before it is read.** Alignment corrects the scan's overall rotation, but a fraction
  of a degree is often left over, and that alone is enough to throw off Tesseract's OWN line segmentation on
  packed, repeating print (a batch/expiry stamp, a dense table): lines merge into their neighbours, or a whole
  line comes back as a scrambled run of words from several real lines. The residual tilt is measured (a rotation
  search that scores how sharply the picture's rows alternate between full of ink and empty -- sharpest when
  level) and the picture is turned that far just for the one whole-page reading; every word it returns is turned
  straight back, so nothing else -- matching, ink agreement, the boxes shown -- ever knows it happened. A picture
  already close to level, or with too little ink to measure, is left untouched (`detection.deskew_enabled`,
  `deskew_max_angle_deg`).
* **Text** is read (Tesseract) from both pictures *in full*, side by side (each page is cut into strips at
  blank rows and the strips are read one to a core), with lettering reversed out of a dark bar read the right
  way round and hairline typefaces thickened for reading. The words of the two files are matched **by what
  they say and where they are**, allowing for how far a scan sits from its digital file *at that place* (a
  scanned carton is bent and can be tens of pixels out). Only lines where words do not match are read again --
  one window per line, stacked into a single picture -- and a line counts as different only if none of its
  readings (the window, the page reading, both together) agree. Case, punctuation and l/i/1 or o/0 confusions
  are ignored, and a number must match exactly, so "500 mg" against "250 mg" is one change showing both
  values. A difference that is probably the reading's, not the artwork's, is listed but ranked down and hidden
  by default (**Show possible misreads** brings it back): a word the scan reads differently although its ink
  matches (or the two files' ink pictures correlate closely throughout its whole box, warp along the scan and
  all), a word read shakily on the digital side, a word "missing" or "new" where the other file has ink. A line
  whose ink is in the scan but too soft to read is reported as *unreadable*; a line whose ink is gone is
  *removed*; text only the scan has is *added*. Text on its side is not read on a whole page (upright it is junk).
* **Barcodes** are not decoded. Where the digital file has one, each is registered on its own against the scan
  (scans are bent by their fold lines) and its bars are compared as marks: a bar added, missing or moved is
  reported as a physical difference at the barcode.
* **Graphics** are not compared by default: a scan changes every colour, blurs every edge and never registers
  exactly, and comparing the pictures gave dozens of false differences (fold lines, the scanner's own edge,
  every dark block) round the few real ones. Tick **Also compare graphics** under the mode to include logos
  and other artwork (or set `detection.scan_compare_graphics` in `config.json`).
* The results page shows the two **black-and-white** pictures (untick *Black && white* for the colour
  files). These are what was read for text.
* A typical A4 or carton pair takes 3-4.5 s (it took 19-25 s when every line was read twice, one Tesseract
  call each).
* **What the words are read from was measured, not assumed.** Both files can be read from other pictures than
  the ink picture -- the file's own grey (`gray`), grey with the paper divided out (`flat`), the darkest channel
  (`darkest`), local contrast (`clahe`) or a local threshold (`adaptive`) -- chosen per file with
  `detection.scan_reading_digital` and `scan_reading_scan`, and lines that differ can also be read from a second
  picture (`scan_reading_alternate`). On two real digital/scan pairs and simulated light, typical, harsh and
  *dark* scans of them, with known text edits painted into the digital file (digits, letters, removed words),
  none of the others beat the ink picture: on clean scans they were level (the file's own grey found about 7%
  more edits and about twice the critical false findings); on a harsh scan (of 48 edits) the ink picture found 42,
  the flattened grey 32 and the file's own grey 12; on a dark scan the ink picture found 36, the flattened grey 3 and
  the file's own grey none. So removing the background removal makes a dark scan *worse*, not better. Only reading
  twice and keeping the findings that both readings agree on cut the false ones by about half, at twice the time
  (10 s, not 5), so it is not the default. The ink picture stays.
* **A difference whose ink pictures are, in fact, the same picture is not a difference.** Beyond matching what a
  line *says*, the finding's two ink pictures are correlated directly (`text_diff.ink_correlation`): each
  letter-wide window re-settles its own local shift within a few pixels, so a scan's warp along a long line of
  type does not count against it. Where they correlate closely throughout, the reading -- not the artwork -- is
  what differs, and the finding is ranked down and hidden by default rather than shown as a real change. On two
  real digital/scan pairs this cut the false differences shown by default (a reader's inconsistency presented as
  if it were a real one) from two or three a run to close to none, without losing a genuine one-letter change,
  which still leaves a window of poor correlation however well the rest of the line reads.

Viewing: side-by-side, flicker, overlay and grayscale views from one aligned pair, with
scroll lock; pinch or Ctrl + wheel to zoom about the cursor; PDFs stay sharp at any zoom.
In the crop dialog, drag any edge or corner of the selection to resize it, drag inside to
move it, Space + drag (or middle-drag) to pan, and confirm the crop when saving.

## How differences are grouped and mapped

Every detector reports small *candidates*; they are grouped into the *findings* you review, each with a box in
both files. Two are joined only where they are really one thing: the bars of a barcode and the dots of Braille
(joined by distance), the letters and words of one changed line of type that adjoin along it, and the same
place found by two detectors (a new logo is both a colour and a graphics difference). Nothing is joined because
it is merely *near* another -- a digit changed inside a recoloured panel is not part of the panel's finding,
and two differences a few millimetres apart are two findings, however far down the carton that chain would
otherwise run. Words that moved are grouped only with words that moved by about the same amount. All of these
limits are in `config.json` (`clustering`, `detection`).

**Borders.** A finding's box is the union of its members', and members come from reduced copies of the page (each
axis scaled by its own factor: a reduced picture's width and height are whole numbers) and from tolerances of a few
pixels. Automatic mode fits each border to the pixels that differ at full size where enough of the box is backed by
them (a faint tint change or a barcode keeps its box; a recoloured area keeps its own exact edge). Differences the
deep local comparison finds are grouped by their own pixels; a group with a big box is grouped again at half the
reach so that a small difference is not lost in the box of a drawing of thin lines that happens to surround it.

**Merging findings.** Where several small differences are really one -- the bars of a barcode drawn as marks,
say, rather than decoded and compared as one symbol -- press **Merge findings** above the page and drag a
rectangle round them (the findings mostly inside it are merged). This is a manual override for the odd case
the automatic grouping above does not cover; it is not needed, and not meant, for an ordinary change of text,
graphics or colour, which are already found as one finding each. A merge becomes one finding with the members
of all, a box covering them in both files, and the lowest number (the numbering does not shift). **Undo merge**
splits the last one; merging is logged, and a merged finding can be ignored like any other.

## Comparing only the artwork

Packaging files are usually a *sheet*: the carton's die-cut outline with the printed panels
inside it and, around it, a title, a colour legend, crop marks, notes. Only what is inside the
outline is artwork. In **Edit Artwork Area**:

* **Lift Subject** -- press and hold on the artwork (a ring fills while you hold). Only the area
  inside its closed border line is lifted; the flaps, which are the colour of the paper, are in, and
  the title, legend, crop marks and notes are out. Measurement lines and arrowheads hanging off the
  outline are shaved off. For a PDF its vector lines are used, in any colour (a dieline is often a
  spot colour); otherwise the dark lines of the picture. A photograph of artwork on a surface is
  lifted by colour and background instead. There is no minimum size for a pressed point: a border
  found where you actually pointed is kept however small it is (only *Auto-detect artwork*, which
  finds the largest closed border on the whole page without a press, ignores a small one -- there is
  nothing else to prefer it over). *Method* forces one or the other. **+ Keep / - Drop** paint corrections over what the
  outline missed or wrongly took (they lift again). **Edit outline** reshapes the outline by hand: drag a
  corner to move it, drag an *edge* to push it out (taking in what the lift left) or in (dropping what it
  took), double-click an edge to add a corner, right-click a corner to remove it, Ctrl+Z to undo; an outline
  with many corners is simplified to at most 100 so the handles are usable, and what is saved (and offered
  to the other file) is the outline as you left it -- so do this after Keep / Drop. **Save cutout...** writes
  the lifted artwork as a lossless PNG with a transparent background.
* **Place Subject** lays the artwork lifted from the *other* file over this one, so this file is
  trimmed to the same area. It is the way to handle a **scan or photo**: a scan of a die-cut sheet has
  no outline to find (its flaps are the colour of the paper), and colour alone lifts only the printed
  panels. The flow: lift the artwork in the Original (Lift Subject, Apply & Save); open Edit Artwork
  Area on the Suspect, and the banner at the top offers it -- **Place it on this file**. It arrives
  already snapped onto the artwork (it matches what is printed, so a scan turned 90 degrees is fine);
  drag, rotate or scale it to adjust, then **Trim to outline**. **Open cutout...** uses a cutout saved
  earlier instead. If a colour lift on the Suspect is much smaller than the Original's artwork, the
  banner says so and offers the same thing.
* **Crop** draws a rectangle (drag its edges or corners to resize it, its inside to move it).
* **Erase** selects a rectangle the same way a crop is selected; **Erase selected area** (or Enter)
  leaves it out of the comparison, and any number can be erased. **Undo last** and **Clear all**
  bring them back.

The comparison then works on the two artworks alone: each file is cleared to white outside its own
area (and where anything was erased), the two are aligned, and then compared. Where the two areas
differ, what only one of them contains is compared too, so a flap or panel one file lacks is not
hidden. A file with no area of its own takes the other's. The line the outline was drawn along is not
compared either -- a scan of the printed sheet has none -- so comparing starts a line's width inside it.
The results page shows the compared artwork alone on a white background, with a dashed outline
(untick **Show compared area only** to see each whole page). A lifted or placed outline masks the comparison but never registers the
files -- the artwork's own features do, so the reported alignment residual stays honest.

## Comparing a sheet printed with several repeats of one artwork

**Series (repeated artwork)** mode is for a press sheet or multi-up proof: the Original has *one* copy of the
artwork (crop or lift it as usual); the Suspect is left alone -- it is a whole sheet with several copies of it,
found automatically (`app/engine/series.py`). The scale is settled once, from the strongest match anywhere on
the sheet, then every repeat at that scale is found and compared against the Original on its own, so one badly
printed repeat is not blamed on its neighbours. All the findings come back together, each saying which repeat
(of how many) it is in; the Suspect pane shows the whole sheet, so a finding sits where its repeat actually is,
and the Original pane shows the one artwork every repeat was measured against. Flicker, overlay and grayscale
views are unavailable here (the two panes are not the same picture warped onto each other) -- side by side
still works, with the same zoom, markers and merge as any other mode.

## Comparing a multi-page document

**Leaflet (multi-page)** mode compares a leaflet or pack insert page by page -- open or closed (a live PDF
with its own text, or one outlined to pure vector paths or a picture, with none), and in any language: text is
only ever *read* to describe a difference the pixels already found, never to decide one, so a page in a script
nobody has installed a Tesseract language for is still compared correctly, just not captioned with its words
(`detection.ocr_language`, default `eng`; join several with `+`, e.g. `eng+fra`). A page needs no border or
die-line -- the whole page is the content, and it is compared exactly as Automatic mode compares a single pair.
Pages are paired by what is actually on them (their words, where enough can be read; their picture otherwise),
not by position, so a page inserted or removed does not throw every page after it out of step; a page with
nothing alike in the other file is its own finding, that it is missing or added. All the findings come back as
one list: each file's pages are stacked into one tall picture with a label at each page's own top, so a finding
is shown on its own page and the ordinary viewer works on it unchanged. Comparison time scales with how big and
how many the pages are (the same adaptive-DPI budget as everywhere else), so a handful of very large pages
takes longer than the single-page NFR target.

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
  "subject": {"hold_ms": 350, "border_dark_level": 115, "border_min_area_fraction": 0.08},
  "detection": {"ocr_language": "eng"},
  "leaflet": {"language": "eng"}
}
```

**Nothing refuses a file.** `max_file_size_mb` and `max_pdf_pages` are 0 (off) by default; set them only if a
site policy needs a hard limit. `max_render_megapixels` (default 100) is not a limit either: it is the most
pixels a page is *held* at, and a page too big to fit it -- a 2 m shipper carton at 300 DPI -- is rendered at
whatever lower DPI does fit, with a note on the results page saying so, rather than refused or asked to be
resized. This is why the same real 332-megapixel shipper file that once produced a "not accepted" pop-up now
just compares, a little softer, with the note explaining why.

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

* **Text comparison needs Tesseract** (the UB-Mannheim Windows installer, default folder; `pip install tesseract`
  is an unrelated package), and, for a language other than English, that language's tessdata installed alongside
  it (`detection.ocr_language` / `leaflet.language`). Without either, Automatic mode still finds every pixel
  difference (it just cannot say what the words were, or say them in that language) and Scan vs Digital reports
  that text was not compared. Words are read from the finding's own box, so a change to text set as a picture,
  over a gradient, or in white on a colour can be found in the pixels and reported as a graphics change rather
  than a text change if the reader cannot read it. Text turned on its side is not read on a whole page in Scan
  vs Digital (in Automatic mode a small finding's own box is read turned too).
* **Scans:** detection thresholds are measured from each image (noise, softness,
  registration), but have been validated on *simulated* scans, not a range of real
  scanners. Fine print softer than the scan can hold is reported as unverifiable rather
  than compared. A false difference from a reader's inconsistency, not the artwork, is ranked down and hidden
  by default (**Show possible misreads** on the findings list brings it back) rather than removed outright.
* **Lifting the artwork** needs a *closed* outline. A break wider than about 5 px in a die-cut
  line, or an outline that is not dark and not a PDF stroke, falls back to colour (or reports
  that no border was found); Keep / Drop then correct it by hand. On a scan the outline is usually
  not there and colour finds only the printed panels: use **Place Subject** with the digital file's
  lifted artwork instead. Lifting has been checked on synthetic sheets and photographs, the sample
  cartons and two real scans, not on a wide range of real files.
* **Series mode** settles on one scale for every repeat and does not look for rotation: a sheet whose repeats
  are printed at more than one size, or turned relative to each other, needs the repeats compared one pair at a
  time in Automatic mode instead.
* **Leaflet mode** pairs pages by content (their words, or failing that their picture); two pages that share
  almost all their wording (a heavily boilerplated pair of sections) could in principle be paired with each
  other rather than left unmatched -- not seen on the real files tried, but worth checking by eye on a leaflet
  whose sections are unusually alike.
* **Pharmacode** needs four or more bars (value 15 upward) to be read; vertical symbols
  are read from the top.
* **SRS section 7 items still open** and configurable rather than decided: the clustering
  distance, whether results must be exportable as an audit report, the Braille dot colour
  convention, batch mode, and accounts / audit trail.

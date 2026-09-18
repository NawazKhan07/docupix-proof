# Claude Code Build Prompt
## Pharmaceutical Packaging Artwork Comparison Desktop App

Paste this into Claude Code as your project brief. It references the SRS and Architecture Document — attach both files (or paste their content) alongside this prompt so Claude Code has the full requirement and design context.

---

You are building a **production-grade, offline desktop application** in Python for comparing pharmaceutical packaging artwork files (original vs. suspect) at pixel, text, graphics, and barcode/Braille level. Full requirements are in the attached **Software Requirements Specification**; the intended technical design is in the attached **Architecture Document**. Follow both; where they are silent, ask me before assuming.

### 1. Project setup

- Initialize a Python 3.11+ project with `pyproject.toml`, a virtual environment, and this dependency set: `PySide6`, `PyMuPDF`, `opencv-python`, `numpy`, `pyzbar`, `zxing-cpp` (or document a fallback if unavailable on this platform), `pytesseract` (+ note the local Tesseract binary/language-data requirement), `scikit-image`, `scikit-learn` (or a lightweight custom clustering if you'd rather avoid the dependency — your call, but explain the tradeoff), `pyinstaller` for packaging.
- Set up the module layout exactly as proposed in the Architecture Document's §10 ("Suggested Module Layout"), adjusting only if you find a concrete reason to — explain any deviation.
- Add `pytest` and a `tests/` tree with a `golden_pairs/` fixtures folder (empty for now, I will add sample artwork files later).
- Set up logging (Python `logging`, structured, one log per comparison job) from day one — I need pipeline-stage timing and memory visibility per the Architecture Document §7 and §9.

### 2. Build order (please work in this sequence, and pause for my review at the end of each phase before continuing)

**Phase 1 — Skeleton & file I/O**
- Main window (PySide6) with Upload panel (Original / Suspect), Mode selector (Automatic / Scan-vs-Digital / Barcode-Pharmacode-QR / Braille), and a disabled Compare button until both files are loaded.
- `file_loader.py`: load PDF via PyMuPDF and images via Pillow/OpenCV; render PDFs at a working DPI; no lossy recompression. Verify a PDF and a PNG both load and preview at full quality when zoomed (FR-9.1 in the SRS).

**Phase 2 — Artwork isolation tools**
- `interactive_canvas.py`: implement rectangular crop first, with Save Crop triggering auto-rotate/auto-align (Architecture §3.3.3). Then implement the click-to-trace dieline polygon extractor (graph walk along connected vector paths / contour pixels back to the start point — Architecture §3.3.2). Then the freehand erase-mask tool.
- Enforce that everything outside the resolved artwork boundary is masked out centrally, before any detector runs (SRS FR-2.1).

**Phase 3 — Alignment**
- Implement the dieline-to-dieline alignment path and the feature-matching fallback path (Architecture §3.3.3) using OpenCV (`findHomography` + RANSAC, ORB/AKAZE features).
- Implement the **alignment quality gate**: compute and log a registration residual error; surface a "low-confidence alignment" warning in the UI if it exceeds a configurable threshold. Make the threshold a config value, not a hard-coded constant.

**Phase 4 — Detectors**
- Implement detectors one at a time, each as an independent, testable module returning `DiffCandidate` objects in the shared aligned coordinate space: text/OCR diff, colour diff (Lab/CIEDE2000, skipped when `compare_color=False`), graphics/structural diff (SSIM), barcode/pharmacode/QR diff (pyzbar + zxing-cpp), Braille dot diff.
- Wire up Scan-vs-Digital mode's preprocessor (background-to-white, text-to-black binarization, `compare_color=False` flag threaded through).

**Phase 5 — Diff aggregation & UI**
- `clustering.py`: cluster raw diff candidates into bounding regions (SRS FR-7.3) with a configurable clustering distance/IoU threshold.
- `severity.py`: rule-based severity tagging with a colour per tier (make the rule table easy to edit).
- `coordinate_projection.py`: project each cluster's bbox into both the original's and the suspect's native coordinates via the inverse alignment transform, so a diff is always shown at the corresponding place in both files (SRS FR-7.4 — this is a hard correctness requirement, add a unit test asserting projected boxes round-trip correctly).
- Diff List panel with category, severity colour, and a per-item Ignore control (soft state, not deletion).
- Viewer: implement all four modes (side-by-side, flicker, overlay, grayscale-with-colour-diffs) as re-renderings of one shared aligned image pair, plus the scroll-lock sync toggle and zoom controls with a visible zoom percentage.

**Phase 6 — Barcode/Braille result panels**
- Barcode/Pharmacode/QR mode: grouped-by-type results panel with per-component pass (green tick) / fail (red cross) as specified in SRS FR-10.2.
- Braille mode: dot-pattern match panel.
- Default (neither mode selected): confirm barcode/Braille regions are still visually pixel-compared but never decoded (SRS FR-12.1).

**Phase 7 — Concurrency, reliability, packaging**
- Move the comparison pipeline into a `ProcessPoolExecutor` worker (Architecture §5), with worker recycling after N jobs and job cancellation support.
- Add the file-size/page-count sanity checks and clear user-facing errors for oversized inputs (Architecture §7).
- Package with PyInstaller into a single offline executable; verify no runtime network calls occur (spot-check with a network-disabled run).

### 3. Non-negotiable constraints (do not relax these without telling me)

- The app must never call out to the network at runtime.
- Comparison must complete in 3–5 seconds on a typical artwork file at the working resolution described in the Architecture Document.
- A diff must always be rendered at the same corresponding location in both the original and suspect views — treat any drift here as a bug, not a tolerance.
- No lossy re-encoding of uploaded files at any pipeline stage.
- Log the alignment residual error on every job; do not silently claim perfect alignment.

### 4. Open items to flag back to me, not guess at

Pull these directly from the SRS "Open Items" section and check with me before finalizing: the numeric clustering threshold, the final severity tier list and colours, whether comparison results need to be exportable as an audit report, the Braille dot colour convention in our source files, whether batch/queued comparison is needed, and whether user accounts/audit trail are in scope.

### 5. Testing expectations

- Unit tests per detector module using small synthetic image fixtures (e.g. two near-identical generated images with one deliberate text/colour/graphic change) so tests don't depend on real artwork files yet.
- An integration test that runs a full `ComparisonJob` end-to-end against a synthetic pair and asserts at least one `DiffCluster` is produced with correct coordinates in both coordinate spaces.
- A regression test asserting the alignment quality gate correctly flags a deliberately misaligned pair as low-confidence.

Start with Phase 1. Show me the project scaffold and the file-loader test results before moving to Phase 2.

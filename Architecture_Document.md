# Architecture Document
## Pharmaceutical Packaging Artwork Comparison Application

Version 1.0 — September 2026
Companion document to: *Software Requirements Specification — Pharmaceutical Packaging Artwork Comparison Application*

---

## 1. Architectural Goals & Drivers

| Driver | Source requirement | Architectural implication |
|---|---|---|
| 3–5 second comparison time | NFR-1 | Multi-resolution pipeline, coarse-to-fine alignment, avoid full-resolution processing until the final diff pass |
| No crashes/leaks on repeated sequential comparisons | NFR-2 | Worker-process isolation, explicit resource release, bounded caches |
| Exact (100%) registration after crop/polygon extraction | NFR-3 | Deterministic alignment pipeline with a measurable, logged error metric (see §9) |
| Fully offline | NFR-4 | No network calls anywhere in the dependency chain; all OCR/decoding models bundled locally |
| Full-fidelity rendering at any zoom | FR-9.1 | Vector-first PDF rendering (re-rasterize per zoom level), never a single fixed-resolution bitmap |
| Diffs must line up 1:1 between original and suspect views | FR-7.4 | A single shared coordinate space, with both views being re-projections of one alignment transform |

---

## 2. High-Level Architecture

The application is a **layered desktop architecture**: a Qt/PySide UI shell, a stateless-per-job **Comparison Engine** running in a separate process/thread pool, and a set of pluggable **Detectors** that each engine job invokes. This separation is what keeps the UI responsive and prevents a bad file from crashing the whole app (NFR-2).

```mermaid
flowchart TB
    subgraph UI["Presentation Layer (PySide6 / Qt Widgets)"]
        A[Upload Panel] --> B[Mode Selector]
        B --> C[Crop / Dieline-Polygon / Erase Tools]
        C --> D[Compare Button]
        E[Viewer: Side-by-side / Flicker / Overlay / Grayscale]
        F[Diff List Panel: cluster, severity, ignore]
        G[Barcode/Braille Result Panel]
    end

    subgraph ORCH["Orchestration Layer (Job Controller)"]
        H[Comparison Job Manager]
        I[Job Queue - QThreadPool / multiprocessing]
    end

    subgraph ENGINE["Comparison Engine (worker process)"]
        J[File Loader & Renderer]
        K[Artwork Isolation: Dieline Detect / Polygon Extract / Manual Crop]
        L[Alignment: Feature Matching + Homography]
        M[Preprocessor: Scan-normalize (bg-white, text-black)]
        N[Detectors]
        O[Diff Aggregator: cluster + severity]
    end

    subgraph DET["Detectors (pluggable)"]
        N1[Text/OCR Diff]
        N2[Colour/Pixel Diff]
        N3[Graphics/Structural Diff]
        N4[Barcode/Pharmacode/QR Diff]
        N5[Braille Dot Diff]
    end

    D --> H --> I --> J --> K --> L --> M --> N
    N --> N1 & N2 & N3 & N4 & N5
    N1 & N2 & N3 & N4 & N5 --> O --> F
    O --> E
    N4 --> G
    N5 --> G
```

---

## 3. Component Breakdown

### 3.1 Presentation Layer (PySide6)
- **Upload Panel** — file pickers for Original/Suspect; validates format, hands raw bytes to the File Loader.
- **Mode Selector** — Automatic (default) / Scan-vs-Digital / Barcode-Pharmacode-QR / Braille. Mode is a parameter object passed into the job, not a code fork in the UI.
- **Crop / Dieline-Polygon / Erase Tools** — a shared `InteractiveCanvas` widget (QGraphicsView-based) that supports three tool modes: rectangular crop, click-to-trace polygon (dieline follower), and freehand erase mask. Emits normalized (0–1) coordinates so results are resolution-independent.
- **Viewer** — a single `ComparisonViewport` component with four render strategies (side-by-side, flicker timer, alpha-blended overlay, grayscale-with-colour-diff-mask) driven by the same underlying aligned image pair — this guarantees FR-7.4 (diffs always at the same place in both views) because both views are drawn from one shared transform, never independently re-aligned.
- **Diff List Panel** — data-bound to the `DiffSet` result object; supports per-item Ignore (soft-delete flag, not deletion, so it's auditable).
- **Barcode/Braille Result Panel** — tabular pass/fail view per FR-10.2.

### 3.2 Orchestration Layer
- **Comparison Job Manager**: builds a `ComparisonJob` (files, mode, crop/polygon/erase geometry, options) and submits it to a worker pool. Runs in the main process but off the UI thread.
- **Job Queue**: `QThreadPool` for lightweight steps; a separate **subprocess per comparison** (via `multiprocessing.Process` or `concurrent.futures.ProcessPoolExecutor`) for the actual OpenCV/PyMuPDF-heavy work. This is the key defense for NFR-2 — if a malformed file segfaults OpenCV/PyMuPDF, only the worker process dies; the UI reports a clean error and the process pool respawns. Each job gets a fresh worker recycled after N jobs to bound long-run memory growth.

### 3.3 Comparison Engine (runs inside the worker process)

1. **File Loader & Renderer**
   - PDFs: PyMuPDF (`fitz`), rendered at a *dynamic* DPI — render once at a working resolution (~300 DPI) for detection, and re-render on-demand at higher DPI per current zoom level for display (FR-9.1). Vector content is never rasterized once-and-cached at a single fixed size.
   - Images: loaded via Pillow/OpenCV, colour-profile preserved, no re-compression on load.

2. **Artwork Isolation**
   - *Dieline detector*: for digital/vector sources, parse PDF vector paths (PyMuPDF `page.get_drawings()`) to find closed-loop paths; for raster-only sources, use OpenCV contour detection (Canny + `findContours`) tuned to the characteristic thin, high-contrast dieline stroke.
   - *Click-to-trace polygon extraction* (FR-4.1/4.2): given a user click near a path segment, do a graph walk along connected vector path segments (or connected contour pixels) back to the start point; return the enclosed polygon. This is a **graph-topology traversal**, not a segmentation-only approach — it directly satisfies "follow the border and loop back."
   - *Manual crop* (FR-3.1): simple rectangular ROI from the `InteractiveCanvas`.
   - *Region erase* (FR-5.1): freehand mask painted onto an alpha channel, applied before any detector sees the image.
   - *Furniture exclusion*: everything outside the resolved artwork boundary (dieline polygon or manual crop) is hard-masked out before any detector runs — this is enforced once, centrally, so no detector can accidentally compare furniture (FR-2.1, FR-6 scope).

3. **Alignment**
   - Primary path: **dieline-to-dieline** — if both artwork boundaries are extracted as polygons, align by matching the polygon boundaries via similarity/affine transform fitted with `cv2.estimateAffinePartial2D` or a full homography (`cv2.findHomography`) over shape correspondence, refined with ORB/SIFT feature matching inside the artwork region for sub-pixel correction.
   - Fallback path (one file has no dieline, FR-2.3): OpenCV feature matching (ORB/AKAZE keypoints + `cv2.findHomography` with RANSAC) between the dielined artwork region and the whole non-dielined image, to locate and warp the corresponding region.
   - Auto-rotate on crop save (FR-3.2): estimate rotation via minimum-area-rect of the artwork mask or via feature-matching angle recovery, then warp-correct before displaying the "saved" crop back to the user.
   - **Alignment quality gate** (supports NFR-3): every alignment computes a residual registration error (mean reprojection error of matched keypoints, in pixels). If the residual exceeds a configured epsilon, the job is flagged "low-confidence alignment" and surfaced to the user rather than silently proceeding — this makes the "100%, no exceptions" requirement testable instead of an unverifiable absolute (see SRS §7 open item).

4. **Preprocessor — Scan vs Digital mode** (FR-8.2 in SRS numbering / mode logic)
   - Background flattening to white (adaptive threshold / `cv2.threshold` with Otsu, or background-subtraction if scan has vignetting).
   - Foreground text/numerals normalized to pure black (binarization).
   - A `compare_color=False` flag is threaded through the Diff Aggregator so colour-diff detectors are skipped entirely in this mode, not just hidden in the UI.

5. **Detectors** (each returns a list of `DiffCandidate{bbox, type, confidence}` in the shared aligned coordinate space)
   - **Text/OCR Diff**: OCR both aligned artwork regions (Tesseract via `pytesseract`, offline), diff at token level (a text-diff algorithm, e.g. `difflib`/Levenshtein on word sequences) to classify changed / moved text and numbers.
   - **Colour/Pixel Diff**: per-pixel or per-superpixel colour delta (CIEDE2000 in Lab space rather than raw RGB, for perceptually meaningful colour-change detection); skipped when `compare_color=False`.
   - **Graphics/Structural Diff**: SSIM (`skimage.metrics.structural_similarity`) or contour-shape diffing on non-text regions to catch logo/graphic changes and moves.
   - **Barcode/Pharmacode/QR Diff**: `pyzbar` (ZBar) for 1D/2D codes; ZXing (via `zxing-cpp` Python binding or subprocess CLI) as a secondary decoder for symbologies ZBar misses (notably some pharmacode variants); decoded values compared string-for-string.
   - **Braille Dot Diff**: colour-blob detection (`cv2.SimpleBlobDetector` or connected-components on the known dot colour/hue range) producing a dot-position matrix per file, diffed by nearest-neighbour matching.

6. **Diff Aggregator**
   - **Clustering** (FR-7.3): spatially cluster raw `DiffCandidate` boxes of the same type using density-based clustering (`DBSCAN` on box centroids/IoU-linkage) so a moved sentence or a run of missing barcode bars becomes one bounding region, not N.
   - **Severity assignment**: rule table mapping diff type + magnitude → severity tier (e.g. any barcode/pharmacode mismatch = Critical; colour ΔE above a threshold = Critical; minor sub-pixel graphic shift = Low), each tier mapped to a display colour.
   - **Coordinate binding**: every cluster stores its bbox in the *shared aligned space*, then is re-projected into the original file's native coordinates AND the suspect file's native coordinates via the inverse of each file's alignment transform — this is what guarantees FR-7.4.
   - **Ignore state**: cluster objects carry a `status: active|ignored` flag rather than being deleted, for traceability.

### 3.4 Data Model (core objects)

```
ComparisonJob(original_file, suspect_file, mode, isolation_geometry, options)
ArtworkRegion(source_file, polygon_or_bbox, alignment_transform, registration_error)
DiffCandidate(bbox, type, confidence, subtype)
DiffCluster(id, bbox_in_original, bbox_in_suspect, type, severity, status, members[])
ComparisonResult(diff_clusters[], barcode_results[], braille_results[], alignment_quality)
```

---

## 4. Technology Stack (confirmed / recommended)

| Layer | Technology | Notes |
|---|---|---|
| UI | **PySide6** (Qt6) | `QGraphicsView`/`QGraphicsScene` for the interactive canvas and viewers; native look, mature |
| PDF rendering | **PyMuPDF (fitz)** | Fast, high-fidelity, gives vector path access for dieline detection |
| Image ops / CV | **OpenCV (opencv-python)** | Alignment, contour/feature detection, colour-space conversion |
| Numerics | **NumPy** | Backing array format shared across OpenCV/Pillow/PyMuPDF |
| OCR | **Tesseract via pytesseract** | Fully offline; bundle language data locally |
| Barcode/QR | **pyzbar** + **zxing-cpp** (secondary decoder) | Two decoders improve pharmacode/QR recall |
| Structural similarity | **scikit-image** | SSIM for graphics diffing |
| Clustering | **scikit-learn (DBSCAN)** or a lightweight custom IoU-clustering | Keep dependency footprint small if scikit-learn is heavy |
| Packaging | **PyInstaller** (or Briefcase) | Single offline executable, no internet dependency at runtime |
| Concurrency | **`concurrent.futures.ProcessPoolExecutor`** | Worker-process isolation per comparison job |

> All of the above run fully offline once installed — no runtime network calls, satisfying NFR-4.

---

## 5. Process & Concurrency Model

- **Main process**: Qt event loop + UI only. Never runs OpenCV/PyMuPDF work synchronously on the UI thread.
- **Worker pool**: 1–2 persistent worker processes (configurable), each handling one `ComparisonJob` at a time. Workers are recycled after a configurable number of jobs (e.g. 20) to bound memory creep from native-library allocations that Python's GC won't reclaim (directly targets the "app lags/crashes after multiple sequential comparisons" complaint in NFR-2).
- **Cancellation**: each job is cancellable; on cancel, the worker process is terminated and replaced rather than relying on cooperative cancellation of OpenCV calls (which don't support interruption well).
- **Backpressure**: only one job enqueued at a time in the MVP (the UI disables Compare while a job runs) — simplest way to avoid resource contention; can be relaxed to a bounded queue later.

---

## 6. Performance Strategy (targeting the 3–5s NFR-1)

1. Run detection at a **working resolution** (e.g. long-edge 2000–3000px) rather than full print resolution; only re-render at full/display resolution for the on-screen viewer.
2. Restrict all detector work to the masked artwork bounding box, never the full page canvas.
3. Coarse-to-fine alignment: cheap global homography first, refined locally only inside regions where a difference is suspected.
4. Cache the rendered original/suspect bitmaps and the alignment transform for the duration of the session so switching viewer modes or zoom levels doesn't re-run the full pipeline.
5. Vectorize colour-diff and SSIM computations with NumPy/OpenCV (avoid per-pixel Python loops).

---

## 7. Reliability & Memory Strategy (targeting NFR-2)

- Process-per-job isolation (§5) contains native-library crashes.
- Explicit `del` / context-managed release of large NumPy arrays and PyMuPDF `Document`/`Page` objects after each stage.
- Worker recycling after N jobs.
- File-size/page-count sanity checks before processing, with a clear user-facing error rather than an uncontrolled allocation.
- Structured logging of each pipeline stage's duration and peak memory (via `tracemalloc` or `psutil`) during development/QA to catch regressions before they reach users.

---

## 8. Security & Offline Posture

- No HTTP client libraries invoked at runtime; dependency licenses/models for OCR and barcode decoding are bundled at build time.
- No telemetry/crash-reporting network calls unless explicitly added and disclosed later.
- All temporary files written to a per-session temp directory and cleaned up on exit.

---

## 9. Testability Additions (recommended, addressing SRS §7 open items)

- **Alignment residual metric** logged and surfaced per job (see §3.3.3) so "100% alignment" becomes a measurable, regression-testable number instead of an unverifiable claim.
- **Golden-file regression suite**: a fixed set of original/suspect artwork pairs with hand-verified expected diffs, run in CI against the detector pipeline.
- **Clustering threshold** exposed as a configuration value (not hard-coded) so QA can tune false-merge vs. false-split behaviour.

---

## 10. Suggested Module Layout

```
app/
  ui/
    main_window.py
    upload_panel.py
    mode_selector.py
    interactive_canvas.py        # crop / dieline-trace / erase
    comparison_viewport.py       # side-by-side / flicker / overlay / grayscale
    diff_list_panel.py
    barcode_braille_panel.py
  engine/
    job_manager.py
    worker.py                    # entry point run inside each subprocess
    file_loader.py               # PyMuPDF / Pillow loading, DPI-aware rendering
    isolation/
      dieline_detector.py
      polygon_tracer.py
      crop_tool.py
      erase_mask.py
    alignment/
      feature_matcher.py
      homography.py
      quality_gate.py
    preprocessing/
      scan_normalizer.py
    detectors/
      text_diff.py
      color_diff.py
      graphics_diff.py
      barcode_diff.py
      braille_diff.py
    aggregation/
      clustering.py
      severity.py
      coordinate_projection.py
  models/
    job.py
    diff.py
    result.py
  packaging/
    pyinstaller.spec
tests/
  golden_pairs/
  test_alignment.py
  test_detectors.py
  test_clustering.py
```

---

## 11. Key Risks Carried Into Design

| Risk | Mitigation in this architecture |
|---|---|
| "100% alignment" is not literally achievable for degraded scans | Quality-gate + logged residual error; surfaced to user rather than hidden |
| OCR/graphics diff false positives on font-rendering noise between digital and scanned sources | Scan-vs-Digital mode disables colour compare and binarizes both sources before OCR/diff, reducing noise |
| Native library crash (OpenCV/PyMuPDF) taking down the whole app | Process-per-job isolation |
| Memory growth over a long session | Worker recycling + explicit resource teardown |
| Dieline detection failing on unusual artwork | Three-tier fallback: dieline → feature-matching alignment → manual crop |

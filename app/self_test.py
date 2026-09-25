"""`--self-test`: check that an installed copy of the application works, offline.

Run against the packaged executable (or a source checkout) it exercises what a
build can silently get wrong -- the worker process starting, PDF rendering, the
comparison pipeline, symbol decoding -- with the offline guard on in both this
process and the worker, so a single attempt to reach the network fails the test.
No display is needed. Exit status 0 means everything passed. A windowed
executable has no console, so the report can also be written to a file
(`--self-test --report FILE`).
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

from app import offline_guard


def _synthetic_pair(folder: Path) -> tuple[Path, Path, Path]:
    """(original.png, suspect.png with one recoloured block, page.pdf)."""
    import cv2
    import pymupdf as fitz
    import numpy as np
    from PIL import Image

    image = np.full((400, 600, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (40, 30), (560, 370), (0, 0, 0), 3)
    cv2.putText(image, "PARACETAMOL 500mg", (70, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2)
    cv2.rectangle(image, (70, 200), (240, 290), (200, 30, 40), -1)
    cv2.circle(image, (430, 240), 50, (30, 90, 200), -1)
    changed = image.copy()
    cv2.rectangle(changed, (70, 200), (240, 290), (40, 160, 60), -1)

    original, suspect = folder / "original.png", folder / "suspect.png"
    Image.fromarray(image).save(original)
    Image.fromarray(changed).save(suspect)

    pdf = folder / "page.pdf"
    doc = fitz.open()
    page = doc.new_page(width=300, height=200)
    page.draw_rect(fitz.Rect(20, 20, 280, 180), color=(0, 0, 0), width=2)
    doc.save(pdf)
    doc.close()
    return original, suspect, pdf


def _run_job(request, timeout_s: float = 120.0):
    """Run one job through the real worker process; return (outcome or None, error or None)."""
    from PySide6.QtCore import QEventLoop, QTimer

    from app.config import AppConfig
    from app.ui.job_controller import JobController

    controller = JobController(AppConfig().runtime)
    result: dict = {}
    loop = QEventLoop()
    controller.completed.connect(lambda outcome: (result.update(outcome=outcome), loop.quit()))
    controller.failed.connect(lambda kind, message: (result.update(error=(kind, message)), loop.quit()))
    QTimer.singleShot(int(timeout_s * 1000), loop.quit)
    try:
        controller.start(request)
        loop.exec()
    finally:
        controller.shutdown()
    return result.get("outcome"), result.get("error", None if "outcome" in result else ("timeout", "no answer"))


def run(report_path: Path | None = None) -> int:
    os.environ[offline_guard.GUARD_ENV] = "1"  # the worker process reads this too
    offline_guard.install()

    from PySide6.QtCore import QCoreApplication

    from app.config import AppConfig
    from app.engine.file_loader import load_artwork
    from app.engine.ocr_locator import configure_ocr
    from app.engine.worker_process import ComparisonRequest
    from app.models.job import ComparisonMode

    app = QCoreApplication.instance() or QCoreApplication(sys.argv)
    del app  # kept alive by Qt for the event loops below
    checks: list[tuple[str, bool, str]] = []
    lines: list[str] = []

    def say(line: str) -> None:
        lines.append(line)
        print(line, flush=True)

    def record(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))
        say(f"  [{'ok' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")

    say("Artwork Comparison self-test (network access is blocked)")
    with tempfile.TemporaryDirectory(prefix="artwork-selftest-") as tmp:
        original, suspect, pdf = _synthetic_pair(Path(tmp))

        try:
            with load_artwork(pdf) as artwork:
                rendered = artwork.render()
            record("PDF rendering", rendered.ndim == 3 and rendered.shape[0] > 100, f"{rendered.shape[1]}x{rendered.shape[0]} px")
        except Exception as exc:
            record("PDF rendering", False, f"{type(exc).__name__}: {exc}")

        # Lifting the artwork: the die-cut line of a PDF page, and a picture by colour (GrabCut).
        try:
            import cv2
            import numpy as np

            from app.config import SubjectConfig
            from app.engine.isolation.subject_lifter import lift_subject

            with load_artwork(pdf) as artwork:
                page_image = artwork.render()
                by_border = lift_subject(page_image, (page_image.shape[1] / 2, page_image.shape[0] / 2),
                                         SubjectConfig(), page=artwork.pdf_page(0), dpi=300, method="border")
            scene = np.full((600, 800, 3), 150, np.uint8)
            cv2.rectangle(scene, (200, 150), (600, 450), (200, 60, 50), -1)
            cv2.circle(scene, (400, 300), 60, (240, 240, 240), -1)
            by_colour = lift_subject(scene, (300, 200), SubjectConfig(), method="colour")
            ok = by_border.source == "border" and 0.4 < by_border.area_fraction < 0.9 and 0.2 < by_colour.area_fraction < 0.35
            record("lifting the artwork", ok, f"border line {100 * by_border.area_fraction:.0f}%, colour {100 * by_colour.area_fraction:.0f}%")
        except Exception as exc:
            record("lifting the artwork", False, f"{type(exc).__name__}: {exc}")

        started = time.perf_counter()
        outcome, error = _run_job(ComparisonRequest(original, suspect, ComparisonMode.AUTOMATIC, AppConfig()))
        if outcome is None:
            record("comparison in the worker process", False, f"{error}")
        else:
            record(
                "comparison in the worker process",
                bool(outcome.clusters),
                f"{len(outcome.clusters)} finding(s), alignment {outcome.alignment.residual_error_px:.2f} px, "
                f"{time.perf_counter() - started:.1f} s",
            )

        # Comparing only inside a chosen area (each file cleared to white outside it, then aligned and
        # compared): the change is found inside it, and gone once its rectangle is erased.
        try:
            from dataclasses import replace

            from app.models.isolation import IsolationGeometry, NormalizedPolygon, NormalizedRect

            area = IsolationGeometry(
                boundary=NormalizedPolygon([(0.05, 0.05), (0.95, 0.05), (0.95, 0.95), (0.05, 0.95)]),
                boundary_origin="lifted",
                edge_inset_norm=0.005,
            )
            erased = replace(area, erase_regions=[NormalizedRect(0.1, 0.45, 0.45, 0.8)])
            inside, error = _run_job(ComparisonRequest(original, suspect, ComparisonMode.AUTOMATIC, AppConfig(), area, area))
            hidden, error2 = _run_job(ComparisonRequest(original, suspect, ComparisonMode.AUTOMATIC, AppConfig(), erased, erased))
            ok = inside is not None and bool(inside.clusters) and hidden is not None and not hidden.clusters
            record("comparing inside a chosen area", ok, f"{len(inside.clusters) if inside else error} finding(s) inside, "
                   f"{len(hidden.clusters) if hidden else error2} once erased")
        except Exception as exc:
            record("comparing inside a chosen area", False, f"{type(exc).__name__}: {exc}")

        # Scan vs Digital: each file as ink on paper (background removed, print black), and no barcode decoding.
        try:
            scan_outcome, scan_error = _run_job(
                ComparisonRequest(original, suspect, ComparisonMode.SCAN_VS_DIGITAL, AppConfig())
            )
            ink = scan_outcome.ink_original if scan_outcome is not None else None
            ok = ink is not None and ink.ndim == 2 and bool((ink == 255).mean() > 0.5) and bool((ink < 60).any())
            record("black-and-white ink pictures (Scan vs Digital)", ok, "ready" if ok else f"{scan_error}")
        except Exception as exc:
            record("black-and-white ink pictures (Scan vs Digital)", False, f"{type(exc).__name__}: {exc}")

        # Automatic mode finds differences in the pixels and drops what is only how a softer file draws the same
        # artwork: a softened copy of the page has NO finding, and the same page with a digit changed has one.
        try:
            import cv2
            import numpy as np
            from PIL import Image

            def _line(text):
                page = np.full((220, 900, 3), 255, np.uint8)
                cv2.putText(page, text, (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (20, 20, 20), 4, cv2.LINE_AA)
                return page

            same, softer, edited = (Path(tmp) / name for name in ("same.png", "softer.png", "edited.png"))
            Image.fromarray(_line("Take 500 mg daily")).save(same)
            Image.fromarray(cv2.GaussianBlur(_line("Take 500 mg daily"), (0, 0), 1.3)).save(softer)
            Image.fromarray(cv2.GaussianBlur(_line("Take 250 mg daily"), (0, 0), 1.3)).save(edited)
            softened, error = _run_job(ComparisonRequest(same, softer, ComparisonMode.AUTOMATIC, AppConfig()))
            changed, error2 = _run_job(ComparisonRequest(same, edited, ComparisonMode.AUTOMATIC, AppConfig()))
            ok = softened is not None and not softened.clusters and changed is not None and len(changed.clusters) == 1
            record(
                "pixel comparison ignores a softer copy",
                ok,
                f"{len(softened.clusters) if softened else error} finding(s) on a softened copy, "
                f"{len(changed.clusters) if changed else error2} with a digit changed",
            )
        except Exception as exc:
            record("pixel comparison ignores a softer copy", False, f"{type(exc).__name__}: {exc}")

        # The deep local comparison: a 0 become a 1 in a serial number set in tiny type, on a pair that registers
        # exactly, is one small finding at that digit (and nothing else on the sheet).
        try:
            import cv2
            import numpy as np
            from PIL import Image

            def _carton(serial):
                page = np.full((900, 1400, 3), 235, np.uint8)
                for row in range(12):
                    cv2.putText(page, f"Store below 25 C in a dry place, protect from light {row}", (60, 90 + row * 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2, cv2.LINE_AA)
                cv2.rectangle(page, (900, 100), (1300, 400), (30, 60, 200), -1)
                cv2.circle(page, (1100, 650), 120, (200, 120, 30), -1)
                cv2.putText(page, serial, (300, 800), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 0, 0), 1, cv2.LINE_AA)
                return page

            first, second = Path(tmp) / "serial_a.png", Path(tmp) / "serial_b.png"
            Image.fromarray(_carton("SN EMJ3TLBE204300")).save(first)
            Image.fromarray(_carton("SN EMJ3TLBE204301")).save(second)
            found, error = _run_job(ComparisonRequest(first, second, ComparisonMode.AUTOMATIC, AppConfig()))
            ok = found is not None and len(found.clusters) == 1 and found.clusters[0].bbox_in_original.width < 30
            record(
                "small change found by the local comparison",
                ok,
                f"{len(found.clusters) if found else error} finding(s) for a digit changed in tiny type",
            )
        except Exception as exc:
            record("small change found by the local comparison", False, f"{type(exc).__name__}: {exc}")

        # Series mode: one artwork against a sheet of three repeats, one of them recoloured -- found, on its own
        # repeat, and the other two are silent.
        try:
            import cv2
            import numpy as np
            from PIL import Image

            from app.models.isolation import IsolationGeometry, NormalizedRect

            def _artwork(colour=(20, 60, 150)):
                # Enough distinct detail (a heading, a coloured bar, a circle, barcode-like bars, small print) that
                # three copies of it are told apart cleanly from an unrelated crop at another scale: three near-blank
                # cells differing only in one bar's colour proved too alike for that, and found nothing.
                cell = np.full((260, 200, 3), 250, np.uint8)
                cv2.rectangle(cell, (2, 2), (197, 257), (90, 90, 90), 2)
                cv2.putText(cell, "SPAS TABLETS", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
                cv2.rectangle(cell, (20, 35), (180, 70), colour, -1)
                cv2.circle(cell, (55, 130), 30, (200, 40, 40), -1)
                cv2.putText(cell, "500 mg", (95, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
                for i, bar in enumerate((3, 1, 2, 1, 3, 2, 1, 3, 2, 1, 2, 3)):
                    x = 20 + i * 13
                    cv2.rectangle(cell, (x, 180), (x + max(2, bar), 220), (20, 20, 20), -1)
                cv2.putText(cell, "Batch AB1234  Exp 07/2028", (12, 244), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1, cv2.LINE_AA)
                return cell

            gap = 20
            sheet = np.full((260 + 2 * gap, 3 * 200 + 4 * gap, 3), 235, np.uint8)
            for column, colour in enumerate([(20, 60, 150), (20, 60, 150), (150, 20, 20)]):  # repeat 3 recoloured
                x = gap + column * (200 + gap)
                sheet[gap : gap + 260, x : x + 200] = _artwork(colour)
            page = np.full((260 + 2 * gap, 200 + 2 * gap, 3), 235, np.uint8)
            page[gap : gap + 260, gap : gap + 200] = _artwork()

            series_original, series_suspect = Path(tmp) / "series_original.png", Path(tmp) / "series_suspect.png"
            Image.fromarray(page).save(series_original)
            Image.fromarray(sheet).save(series_suspect)
            geometry = IsolationGeometry(boundary=NormalizedRect(gap / page.shape[1], gap / page.shape[0], (gap + 200) / page.shape[1], (gap + 260) / page.shape[0]))
            series_outcome, series_error = _run_job(
                ComparisonRequest(series_original, series_suspect, ComparisonMode.SERIES, AppConfig(), original_geometry=geometry)
            )
            ok = (
                series_outcome is not None
                and any("Repeat 3 of 3" in c.detail for c in series_outcome.clusters)
                and not any("Repeat 1 of 3" in c.detail or "Repeat 2 of 3" in c.detail for c in series_outcome.clusters)
            )
            record(
                "series mode (repeated artwork)",
                ok,
                f"{len(series_outcome.clusters) if series_outcome else series_error} finding(s)",
            )
        except Exception as exc:
            record("series mode (repeated artwork)", False, f"{type(exc).__name__}: {exc}")

        # Leaflet mode: a two-page document, one page changed -- found on its own page, the other page silent.
        try:
            from PIL import Image

            def _leaflet_page(heading, edited=False):
                page = np.full((500, 700, 3), 250, np.uint8)
                cv2.rectangle(page, (0, 0), (700, 80), (40, 90, 150), -1)
                cv2.putText(page, heading, (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(page, "Body text for this section of the leaflet", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1, cv2.LINE_AA)
                if edited:
                    cv2.rectangle(page, (25, 130), (500, 165), (250, 250, 250), -1)
                    cv2.putText(page, "This body text was changed", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1, cv2.LINE_AA)
                return page

            leaflet_original, leaflet_suspect = Path(tmp) / "leaflet_original.pdf", Path(tmp) / "leaflet_suspect.pdf"
            original_leaflet_pages = [Image.fromarray(_leaflet_page("INDICATIONS")), Image.fromarray(_leaflet_page("DOSAGE"))]
            suspect_leaflet_pages = [Image.fromarray(_leaflet_page("INDICATIONS")), Image.fromarray(_leaflet_page("DOSAGE", edited=True))]
            original_leaflet_pages[0].save(leaflet_original, save_all=True, append_images=original_leaflet_pages[1:])
            suspect_leaflet_pages[0].save(leaflet_suspect, save_all=True, append_images=suspect_leaflet_pages[1:])
            leaflet_outcome, leaflet_error = _run_job(ComparisonRequest(leaflet_original, leaflet_suspect, ComparisonMode.LEAFLET, AppConfig()))
            ok = (
                leaflet_outcome is not None
                and bool(leaflet_outcome.clusters)
                and all(c.detail.startswith("Page 2:") for c in leaflet_outcome.clusters)
            )
            record(
                "leaflet mode (multi-page)",
                ok,
                f"{len(leaflet_outcome.clusters) if leaflet_outcome else leaflet_error} finding(s)",
            )
        except Exception as exc:
            record("leaflet mode (multi-page)", False, f"{type(exc).__name__}: {exc}")

        # Symbol decoding: a Pharmacode drawn here must be read back by the bundled decoders.
        try:
            import cv2
            import numpy as np

            from app.config import DetectionConfig
            from app.engine.detectors import barcode_diff, pharmacode

            page = np.full((300, 400, 3), 255, np.uint8)
            x = 100
            for thick in reversed(pharmacode.encode(2731)):
                size = 10 if thick else 4
                cv2.rectangle(page, (x, 80), (x + size - 1, 220), (30, 30, 30), -1)
                x += size + 8
            readings = barcode_diff.decode_symbols(page, DetectionConfig())
            values = [reading.value for reading in readings]
            record("symbol decoding", "2731" in values, f"read {values}")
        except Exception as exc:
            record("symbol decoding", False, f"{type(exc).__name__}: {exc}")

    tesseract = configure_ocr()
    record("OCR engine (informational)", True, str(tesseract) if tesseract else "not found: text comparison will be reported as unavailable")
    if tesseract is not None:
        # An engine that is found must also work: read a line of text back from a picture.
        try:
            import cv2
            import numpy as np
            import pytesseract

            page = np.full((160, 900, 3), 255, np.uint8)
            cv2.putText(page, "PARACETAMOL 500 mg", (30, 105), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (10, 10, 10), 4, cv2.LINE_AA)
            text = pytesseract.image_to_string(page, config="--psm 7").strip()
            record("reading text (Tesseract)", "PARACETAMOL" in text.upper(), f"read {text!r}")
        except Exception as exc:
            record("reading text (Tesseract)", False, f"{type(exc).__name__}: {exc}")

        # A changed number is found by Scan vs Digital, in the worker, with both values in the finding.
        try:
            import cv2
            import numpy as np
            from PIL import Image

            with tempfile.TemporaryDirectory(prefix="artwork-selftest-scan-") as scan_tmp:
                def _leaflet(figure):
                    page = np.full((900, 1300, 3), 255, np.uint8)
                    cv2.rectangle(page, (60, 60), (1240, 840), (226, 238, 246), -1)
                    for row, text in enumerate(("PARACETAMOL 500 mg TABLETS", f"Each tablet contains {figure} mg")):
                        cv2.putText(page, text, (110, 200 + row * 130), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (20, 20, 20), 3, cv2.LINE_AA)
                    return page

                digital, scanned = Path(scan_tmp) / "digital.png", Path(scan_tmp) / "scanned.png"
                Image.fromarray(_leaflet(500)).save(digital)
                Image.fromarray(cv2.GaussianBlur(_leaflet(250), (0, 0), 1.2)).save(scanned)
                found, error = _run_job(ComparisonRequest(digital, scanned, ComparisonMode.SCAN_VS_DIGITAL, AppConfig()))
                ok = found is not None and any("500" in c.detail and "250" in c.detail for c in found.clusters)
                record(
                    "text change found in Scan vs Digital",
                    ok,
                    "; ".join(c.detail for c in found.clusters)[:100] if found else f"{error}",
                )
        except Exception as exc:
            record("text change found in Scan vs Digital", False, f"{type(exc).__name__}: {exc}")

    failed = [name for name, ok, _detail in checks if not ok]
    say("PASSED" if not failed else f"FAILED: {', '.join(failed)}")
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if not failed else 1

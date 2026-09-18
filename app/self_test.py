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

    failed = [name for name, ok, _detail in checks if not ok]
    say("PASSED" if not failed else f"FAILED: {', '.join(failed)}")
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if not failed else 1

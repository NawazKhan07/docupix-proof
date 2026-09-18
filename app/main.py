"""Application entry point."""

from __future__ import annotations

import multiprocessing
import sys

from app import offline_guard


def main(argv: list[str] | None = None) -> int:
    # A packaged executable starts each comparison worker by re-running itself;
    # this hands those child processes to the multiprocessing machinery instead
    # of opening another application window. It does nothing in a normal run.
    multiprocessing.freeze_support()
    offline_guard.install_if_requested()
    argv = sys.argv if argv is None else argv

    if "--self-test" in argv[1:]:
        from pathlib import Path

        from app.self_test import run

        report = Path(argv[argv.index("--report") + 1]) if "--report" in argv else None
        return run(report)

    from PySide6.QtWidgets import QApplication

    from app.engine.ocr_locator import configure_ocr
    from app.logging_config import configure_logging
    from app.ui.main_window import MainWindow
    from app.ui.theme import apply_theme

    configure_logging()
    configure_ocr()
    app = QApplication(argv)
    apply_theme(app)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

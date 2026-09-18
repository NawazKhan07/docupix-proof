"""Application entry point."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from app.logging_config import configure_logging
from app.ui.main_window import MainWindow
from app.ui.theme import apply_theme


def main() -> int:
    configure_logging()
    app = QApplication(sys.argv)
    apply_theme(app)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

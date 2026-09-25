"""Finding the Tesseract OCR engine, which is a separate native program.

`pytesseract` only calls a `tesseract` executable; it does not contain one. The
packaged application ships one beside itself (so nothing needs installing), and
a source checkout uses whatever is on the machine. Either way it is located
here, once, rather than relying on PATH: on Windows the Tesseract installer
does not always put itself there, and a comparison that silently skips text
because OCR "is not installed" when it is installed is a poor failure.

Search order: TESSERACT_CMD (explicit) -> the copy bundled with the
application -> PATH -> the usual Windows install folders.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path

import pytesseract

from app import paths

logger = logging.getLogger(__name__)

EXPLICIT_ENV = "TESSERACT_CMD"
_EXECUTABLE = "tesseract.exe" if sys.platform == "win32" else "tesseract"
_BUNDLED_FOLDER = "tesseract"
_WINDOWS_FOLDERS = (
    r"C:\Program Files\Tesseract-OCR",
    r"C:\Program Files (x86)\Tesseract-OCR",
)


def _candidates() -> list[Path]:
    found: list[Path] = []
    explicit = os.environ.get(EXPLICIT_ENV)
    if explicit:
        found.append(Path(explicit))
    found.append(paths.bundle_dir() / _BUNDLED_FOLDER / _EXECUTABLE)
    on_path = shutil.which("tesseract")
    if on_path:
        found.append(Path(on_path))
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        folders = list(_WINDOWS_FOLDERS) + ([str(Path(local) / "Programs" / "Tesseract-OCR")] if local else [])
        found.extend(Path(folder) / _EXECUTABLE for folder in folders)
    return found


def find_tesseract() -> Path | None:
    """The Tesseract executable to use, or None if there is none."""
    return next((candidate for candidate in _candidates() if candidate.is_file()), None)


def available_languages() -> frozenset[str]:
    """The Tesseract language codes that can actually be read (their tessdata is installed) -- empty if Tesseract
    itself is not available. `configure_ocr()` should have run first, or this asks whatever `tesseract` PATH finds."""
    try:
        return frozenset(pytesseract.get_languages(config=""))
    except Exception:  # noqa: BLE001 -- no engine, or a build too old for --list-langs: nothing is available
        return frozenset()


def usable_language(requested: str) -> tuple[str, bool]:
    """(the language to actually ask Tesseract for, whether that is what was requested).

    `requested` may join several with "+" (Tesseract's own syntax): whichever of them are installed are kept, in the
    order given. None installed falls back to English; if that is not installed either, `requested` is returned
    regardless (there is nothing better to try, and Tesseract will say so itself)."""
    wanted = [code for code in requested.split("+") if code]
    installed = available_languages()
    if not installed:
        return requested, True
    usable = [code for code in wanted if code in installed]
    if usable:
        return "+".join(usable), usable == wanted
    return ("eng" if "eng" in installed else requested), False


def configure_ocr() -> Path | None:
    """Point pytesseract at the Tesseract found. Safe to call more than once, in any process."""
    executable = find_tesseract()
    if executable is None:
        logger.warning("Tesseract OCR was not found; text and number comparison will be unavailable")
        return None

    pytesseract.pytesseract.tesseract_cmd = str(executable)
    # A bundled copy keeps its language data beside it.
    tessdata = executable.parent / "tessdata"
    if tessdata.is_dir() and "TESSDATA_PREFIX" not in os.environ:
        os.environ["TESSDATA_PREFIX"] = str(tessdata)
    logger.info("using Tesseract OCR", extra={"path": str(executable)})
    return executable

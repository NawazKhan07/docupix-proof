"""Tesseract is a separate program: where the app looks for it, and in what order."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import pytesseract

from app import paths
from app.engine import ocr_locator

EXE = "tesseract.exe" if sys.platform == "win32" else "tesseract"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """Nothing on this machine may influence the search; the pytesseract setting is restored."""
    monkeypatch.delenv(ocr_locator.EXPLICIT_ENV, raising=False)
    monkeypatch.delenv("TESSDATA_PREFIX", raising=False)
    monkeypatch.setattr(ocr_locator.shutil, "which", lambda name: None)
    monkeypatch.setattr(ocr_locator, "_WINDOWS_FOLDERS", ())
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setattr(paths, "bundle_dir", lambda: tmp_path / "bundle")
    monkeypatch.setattr(pytesseract.pytesseract, "tesseract_cmd", "tesseract")


def _fake(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / EXE
    path.write_text("")
    return path


def test_nothing_found_leaves_the_setting_alone_and_says_so():
    assert ocr_locator.find_tesseract() is None
    assert ocr_locator.configure_ocr() is None
    assert pytesseract.pytesseract.tesseract_cmd == "tesseract"


def test_the_copy_bundled_with_the_application_is_used(tmp_path):
    bundled = _fake(tmp_path / "bundle" / "tesseract")
    assert ocr_locator.configure_ocr() == bundled
    assert pytesseract.pytesseract.tesseract_cmd == str(bundled)


def test_the_bundled_copy_brings_its_language_data(tmp_path, monkeypatch):
    _fake(tmp_path / "bundle" / "tesseract")
    (tmp_path / "bundle" / "tesseract" / "tessdata").mkdir()
    ocr_locator.configure_ocr()
    assert Path(__import__("os").environ["TESSDATA_PREFIX"]) == tmp_path / "bundle" / "tesseract" / "tessdata"


def test_an_explicit_setting_beats_the_bundled_copy(tmp_path, monkeypatch):
    _fake(tmp_path / "bundle" / "tesseract")
    mine = _fake(tmp_path / "mine")
    monkeypatch.setenv(ocr_locator.EXPLICIT_ENV, str(mine))
    assert ocr_locator.find_tesseract() == mine


def test_the_bundled_copy_beats_whatever_is_on_the_path(tmp_path, monkeypatch):
    bundled = _fake(tmp_path / "bundle" / "tesseract")
    on_path = _fake(tmp_path / "elsewhere")
    monkeypatch.setattr(ocr_locator.shutil, "which", lambda name: str(on_path))
    assert ocr_locator.find_tesseract() == bundled


def test_a_tesseract_on_the_path_is_used_when_nothing_is_bundled(tmp_path, monkeypatch):
    on_path = _fake(tmp_path / "elsewhere")
    monkeypatch.setattr(ocr_locator.shutil, "which", lambda name: str(on_path))
    assert ocr_locator.find_tesseract() == on_path


def test_an_installer_that_did_not_update_the_path_is_still_found(tmp_path, monkeypatch):
    if sys.platform != "win32":
        pytest.skip("Windows install folders")
    installed = _fake(tmp_path / "Program Files" / "Tesseract-OCR")
    monkeypatch.setattr(ocr_locator, "_WINDOWS_FOLDERS", (str(installed.parent),))
    assert ocr_locator.find_tesseract() == installed


def test_a_bad_explicit_path_is_not_trusted(tmp_path, monkeypatch):
    monkeypatch.setenv(ocr_locator.EXPLICIT_ENV, str(tmp_path / "nope" / EXE))
    assert ocr_locator.find_tesseract() is None

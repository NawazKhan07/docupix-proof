# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build for the Artwork Comparison application (Build Prompt Phase 7).

Produces one self-contained executable that runs with no Python installed and no
network. Build from the project root:

    pyinstaller app/packaging/pyinstaller.spec --noconfirm --clean

Then check it (no console needed; the exit status and report file say how it went):

    dist\\ArtworkCompare.exe --self-test --report self-test.txt

Set ARTWORK_ONEDIR=1 for a folder build (starts faster, easier to debug) instead of
a single file. To ship OCR with the application, put a Tesseract install (the
folder holding tesseract.exe, its DLLs and a `tessdata` folder with eng.traineddata)
at vendor/tesseract before building; it is bundled and found automatically at run
time (see app/engine/ocr_locator.py). Without it the application still runs and
says clearly that text comparison is unavailable.
"""

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

ROOT = Path(SPECPATH).resolve().parent.parent
ONEDIR = bool(os.environ.get("ARTWORK_ONEDIR"))
NAME = "ArtworkCompare"

# -- what the frozen program needs that static analysis cannot see -------------------------
hiddenimports = (
    collect_submodules("app")  # the worker process imports its target by name
    + collect_submodules("skimage")  # scikit-image loads most of its submodules lazily
    + ["zxingcpp", "pytesseract"]
)

binaries = (
    collect_dynamic_libs("pyzbar")  # libzbar and its runtime, which pyzbar loads by path
    + collect_dynamic_libs("zxingcpp")
)

datas = collect_data_files("skimage")  # includes the stubs its lazy loader reads

# An optional bundled OCR engine.
tesseract = ROOT / "vendor" / "tesseract"
if tesseract.is_dir():
    datas.append((str(tesseract), "tesseract"))

# Nothing here is used at run time; leaving them out keeps the executable smaller.
excludes = [
    "tkinter", "matplotlib", "IPython", "jupyter", "notebook", "pytest", "sphinx",
    "PyQt5", "PyQt6", "PySide2",  # only PySide6 is used; a second Qt binding would clash
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtQml", "PySide6.QtQuick",
    "PySide6.Qt3DCore", "PySide6.QtMultimedia", "PySide6.QtCharts", "PySide6.QtDataVisualization",
    "PySide6.QtPdf", "PySide6.QtPdfWidgets", "PySide6.QtSql", "PySide6.QtNetwork",  # no network, by design
]

a = Analysis(
    [str(ROOT / "app" / "main.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

if ONEDIR:
    exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name=NAME, console=False)
    coll = COLLECT(exe, a.binaries, a.datas, name=NAME)
else:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name=NAME,
        console=False,  # a window application; --self-test writes its report to a file
        upx=False,  # compressing the Qt/OpenCV DLLs saves little and trips antivirus scanners
    )

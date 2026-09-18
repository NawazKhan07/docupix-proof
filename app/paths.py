"""Where the application keeps things, whether run from source or as a packaged executable.

Run from a checkout, logs and config sit in the working directory, as they
always have. Run as a packaged executable the working directory is whatever the
shortcut says -- often somewhere unwritable -- so logs go to a per-user folder
and config is also looked for beside the executable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_FOLDER = "ArtworkCompare"
HOME_ENV = "ARTWORK_COMPARE_HOME"  # overrides the per-user folder (also used by tests)


def is_frozen() -> bool:
    """True when running as a PyInstaller executable."""
    return bool(getattr(sys, "frozen", False))


def bundle_dir() -> Path:
    """Where bundled resources (e.g. the OCR engine) live."""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent.parent


def executable_dir() -> Path:
    return Path(sys.executable).parent if is_frozen() else Path.cwd()


def user_data_dir() -> Path:
    """The per-user folder for logs and settings."""
    override = os.environ.get(HOME_ENV)
    if override:
        return Path(override)
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share")
    return Path(base) / APP_FOLDER


def log_dir() -> Path:
    if is_frozen() or os.environ.get(HOME_ENV):
        return user_data_dir() / "logs"
    return Path("logs")


def config_candidates(filename: str) -> list[Path]:
    """Places a config file may be, most specific first."""
    places = [Path(filename)]
    if is_frozen():
        places.append(executable_dir() / filename)
    places.append(user_data_dir() / filename)
    return places

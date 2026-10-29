"""A PDF page's stroked vector paths, read once.

Finding the artwork's border, dividing it into panels by its folds and finding its die lines each want the page's
stroked paths -- its lines -- and nothing else. PyMuPDF's `get_drawings` turns every point of every path on the page
into a Python object first: on a pack insert with a few thousand filled shapes that is millions of points and several
seconds, three times over. Here the page's paths are read in their raw form (`get_cdrawings`, ~4x faster), only the
stroked ones are converted -- exactly as `get_drawings` would convert them -- and the result is kept for the next
caller asking about the same page.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict

import pymupdf as fitz

# The PDF library is not safe to call from two threads at once (two files, or a leaflet's pages, are handled side by
# side); re-entrant, since a caller holding it may ask for a page's paths.
PDF_LOCK = threading.RLock()

_KEPT_PAGES = 8
_kept: OrderedDict[tuple, tuple[int, list[dict]]] = OrderedDict()
_STROKED = ("s", "fs")
_PATH_KEYS = ("closePath", "fill", "color", "width", "lineCap", "lineJoin", "dashes", "stroke_opacity", "fill_opacity", "even_odd")


def _page_key(page) -> tuple:
    document = page.parent
    name = getattr(document, "name", "") or ""
    try:
        stamp = os.stat(name).st_mtime_ns if name else None
    except OSError:
        stamp = None
    return id(document), name, stamp, page.number


def _converted(path: dict) -> dict:
    """One raw path as `get_drawings` gives it: its points, rectangles and quads as PyMuPDF objects."""
    path["rect"] = fitz.Rect(path["rect"])
    items = []
    for item in path["items"]:
        command, rest = item[0], item[1:]
        if command == "re":
            items.append(("re", fitz.Rect(rest[0]).normalize(), rest[1]))
        elif command == "qu":
            items.append(("qu", fitz.Quad(rest[0])))
        else:
            items.append(tuple([command] + [fitz.Point(point) for point in rest]))
    path["items"] = items
    for key in _PATH_KEYS:
        path.setdefault(key, None)
    return path


def stroked_paths(page) -> tuple[int, list[dict]]:
    """(how many paths the page has in all, its stroked paths as `get_drawings` gives them)."""
    key = _page_key(page)
    with PDF_LOCK:
        if key in _kept:
            _kept.move_to_end(key)
            return _kept[key]
        raw = page.get_cdrawings()
        found = (len(raw), [_converted(path) for path in raw if path.get("type") in _STROKED])
        _kept[key] = found
        while len(_kept) > _KEPT_PAGES:
            _kept.popitem(last=False)
        return found

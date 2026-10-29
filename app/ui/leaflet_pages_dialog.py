"""Leaflet mode's Edit dialog: the whole file at once, every page, one under the other.

A leaflet is compared page by page, each page trimmed to the leaflet found on it (`app.engine.leaflet`). Here the
reviewer sees every page of one file and can, on any of them:

* **erase** areas -- a job sheet's print details, a note, a colour bar -- which are then left out of that page's
  comparison, on top of the leaflet area found automatically; and
* set a page's **area** by hand -- a rectangle on the page -- which is compared instead of the area found on it.

The pages are shown reduced (a pack insert's page is 40-100 MP at full size); what is drawn is kept in shares of each
page, so it means the same at any size. The result is one `IsolationGeometry` per page that has anything on it.
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.engine.file_loader import LoadedArtwork
from app.models.isolation import IsolationGeometry, IsolationTool, NormalizedRect
from app.ui.flow_layout import flow_widget
from app.ui.interactive_canvas import InteractiveCanvas

_PAGE_LONG_EDGE_PX = 2200  # each page is shown no bigger than this along its long edge
_PAGE_PREVIEW_DPI = 150.0  # ...rendered at no more than this first (a PDF), then reduced
_LABEL_BAND_PX = 44  # the band above each page, with its number (at least this tall)
_BAND_COLOUR = (60, 64, 72)
_PAD_COLOUR = (235, 235, 230)
_SCREEN_SHARE = 0.85

PageAreas = dict[int, IsolationGeometry]


def _page_picture(artwork: LoadedArtwork, index: int) -> np.ndarray:
    """One page, reduced for showing."""
    if artwork.kind == "pdf":
        picture = artwork.render(dpi=min(_PAGE_PREVIEW_DPI, artwork.working_dpi(index)), page=index)
    else:
        picture = artwork.render(page=index)
    scale = min(1.0, _PAGE_LONG_EDGE_PX / max(picture.shape[:2]))
    if scale < 1.0:
        picture = cv2.resize(picture, (max(1, round(picture.shape[1] * scale)), max(1, round(picture.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    return picture


def stack_pages(pictures: list[np.ndarray]) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Every page one under the other, each under a band with its number. Returns the picture and each page's
    (x, y, width, height) in it."""
    width = max(picture.shape[1] for picture in pictures)
    # the page numbers readable with the whole file fitted in the window: sized to the pages' width
    scale = max(0.8, width / 700)
    band_px = max(_LABEL_BAND_PX, round(34 * scale))
    parts, places, y = [], [], 0
    for number, picture in enumerate(pictures, start=1):
        band = np.full((band_px, width, 3), _BAND_COLOUR, np.uint8)
        cv2.putText(band, f"Page {number}", (round(12 * scale), band_px - round(9 * scale)), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), max(2, round(2 * scale)), cv2.LINE_AA)
        parts.append(band)
        y += band_px
        pad = width - picture.shape[1]
        parts.append(picture if pad == 0 else np.pad(picture, ((0, 0), (0, pad), (0, 0)), constant_values=_PAD_COLOUR[0]))
        places.append((0, y, picture.shape[1], picture.shape[0]))
        y += picture.shape[0]
    return np.vstack(parts), places


def to_pages(region: NormalizedRect, size: tuple[int, int], places: list[tuple[int, int, int, int]]) -> dict[int, NormalizedRect]:
    """A rectangle drawn on the stacked pages (in shares of the whole stack, `size` = (width, height)) as the part of it
    on each page it covers, in shares of that page."""
    width, height = size
    x0, y0, x1, y1 = region.x0 * width, region.y0 * height, region.x1 * width, region.y1 * height
    found = {}
    for index, (px, py, pw, ph) in enumerate(places):
        left, top, right, bottom = max(x0, px), max(y0, py), min(x1, px + pw), min(y1, py + ph)
        if right - left >= 1 and bottom - top >= 1:
            found[index] = NormalizedRect((left - px) / pw, (top - py) / ph, (right - px) / pw, (bottom - py) / ph)
    return found


def to_stack(index: int, region: NormalizedRect, size: tuple[int, int], places: list[tuple[int, int, int, int]]) -> NormalizedRect:
    """`to_pages` the other way: a rectangle in shares of page `index`, in shares of the whole stack."""
    width, height = size
    px, py, pw, ph = places[index]
    return NormalizedRect(
        (px + region.x0 * pw) / width, (py + region.y0 * ph) / height, (px + region.x1 * pw) / width, (py + region.y1 * ph) / height
    )


class LeafletPagesDialog(QDialog):
    """See the module's account. `areas` are what was set on this file's pages before (kept, and shown again)."""

    def __init__(self, artwork: LoadedArtwork, title: str, parent=None, areas: PageAreas | None = None):
        super().__init__(parent)
        self.setWindowTitle(f"Edit {title} — every page")
        self._artwork = artwork
        self._result: PageAreas | None = None
        QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self._stack, self._places = stack_pages([_page_picture(artwork, i) for i in range(artwork.page_count)])
        finally:
            QGuiApplication.restoreOverrideCursor()
        self._size = (self._stack.shape[1], self._stack.shape[0])
        self._page_areas: dict[int, NormalizedRect] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        tools, tool_row = flow_widget(self, spacing=12)
        self.erase_tool_button = QPushButton("Erase areas")
        self.area_tool_button = QPushButton("Set a page's area")
        group = QButtonGroup(self)
        for button in (self.erase_tool_button, self.area_tool_button):
            button.setCheckable(True)
            button.setProperty("variant", "segment")
            group.addButton(button)
            tool_row.addWidget(button)
        self.erase_tool_button.setChecked(True)
        self.zoom_out_button = QPushButton("−")
        self.zoom_label = QLabel("100%")
        self.zoom_in_button = QPushButton("+")
        self.fit_button = QPushButton("Fit")
        for button in (self.zoom_out_button, self.zoom_in_button):
            button.setFixedWidth(36)
            button.setProperty("variant", "icon")
        self.zoom_label.setMinimumWidth(48)
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        zoom = QWidget()
        zoom_row = QHBoxLayout(zoom)
        zoom_row.setContentsMargins(0, 0, 0, 0)
        zoom_row.setSpacing(4)
        for widget in (self.zoom_out_button, self.zoom_label, self.zoom_in_button, self.fit_button):
            zoom_row.addWidget(widget)
        tool_row.addWidget(zoom)
        layout.addWidget(tools)

        # the tools' own buttons: one row, what it offers changing with the tool
        self.erase_apply_button = QPushButton("Erase selected area")
        self.erase_apply_button.setProperty("variant", "primary")
        self.erase_apply_button.setToolTip("Leave the selected rectangle out of the comparison, on every page it covers (Enter).")
        self.erase_undo_button = QPushButton("Undo last")
        self.erase_clear_button = QPushButton("Clear all")
        self.area_set_button = QPushButton("Use as the page's area")
        self.area_set_button.setProperty("variant", "primary")
        self.area_set_button.setToolTip("Compare only this rectangle of the page it is on, instead of the leaflet found there.")
        self.area_clear_button = QPushButton("Clear page areas")
        self.area_clear_button.setToolTip("Every page is compared by the leaflet found on it again.")
        options, option_row = flow_widget(self, spacing=8)
        for button in (self.erase_apply_button, self.erase_undo_button, self.erase_clear_button, self.area_set_button, self.area_clear_button):
            option_row.addWidget(button)
        self.status_label = QLabel("")
        self.status_label.setProperty("secondary", "true")
        self.status_label.setWordWrap(True)
        layout.addWidget(options)
        layout.addWidget(self.status_label)

        self.canvas = InteractiveCanvas(self)
        self.canvas.set_image(self._stack)
        layout.addWidget(self.canvas, stretch=1)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        save = self.buttons.button(QDialogButtonBox.StandardButton.Save)
        save.setText("Apply && Save")
        save.setProperty("variant", "primary")
        layout.addWidget(self.buttons)

        for index, geometry in sorted((areas or {}).items()):
            if index >= len(self._places):
                continue
            if isinstance(geometry.boundary, NormalizedRect):
                self._page_areas[index] = geometry.boundary
            for region in geometry.erase_regions:
                if isinstance(region, NormalizedRect):
                    self.canvas.add_erase_region(to_stack(index, region, self._size, self._places))
        self._show_page_areas()

        self.erase_tool_button.clicked.connect(lambda: self._select_tool(IsolationTool.ERASE))
        self.area_tool_button.clicked.connect(lambda: self._select_tool(IsolationTool.CROP))
        self.erase_apply_button.clicked.connect(self.canvas.apply_erase_selection)
        self.erase_undo_button.clicked.connect(self.canvas.undo_erase)
        self.erase_clear_button.clicked.connect(self.canvas.clear_erase_regions)
        self.area_set_button.clicked.connect(self._set_page_area)
        self.area_clear_button.clicked.connect(self._clear_page_areas)
        self.canvas.eraseSelectionChanged.connect(lambda _rect: self._update_states())
        self.canvas.eraseRegionsChanged.connect(self._update_states)
        self.canvas.cropRectChanged.connect(lambda _rect: self._update_states())
        self.canvas.zoomChanged.connect(lambda zoom: self.zoom_label.setText(f"{round(zoom * 100)}%"))
        self.zoom_in_button.clicked.connect(self.canvas.zoom_in)
        self.zoom_out_button.clicked.connect(self.canvas.zoom_out)
        self.fit_button.clicked.connect(self.canvas.fit_to_view)
        self.buttons.accepted.connect(self._on_save)
        self.buttons.rejected.connect(self.reject)

        self._select_tool(IsolationTool.ERASE)
        self._size_to_screen()

    # -- tools --------------------------------------------------------------------------------------------------------
    def _select_tool(self, tool: IsolationTool) -> None:
        self.canvas.set_tool(tool)
        (self.erase_tool_button if tool is IsolationTool.ERASE else self.area_tool_button).setChecked(True)
        erasing = tool is IsolationTool.ERASE
        for button in (self.erase_apply_button, self.erase_undo_button, self.erase_clear_button):
            button.setVisible(erasing)
        for button in (self.area_set_button, self.area_clear_button):
            button.setVisible(not erasing)
        self._update_states()

    def _update_states(self) -> None:
        erased = len(self.canvas.erase_regions())
        self.erase_apply_button.setEnabled(self.canvas.erase_selection() is not None)
        self.erase_undo_button.setEnabled(erased > 0)
        self.erase_clear_button.setEnabled(erased > 0)
        self.area_set_button.setEnabled(self.canvas.crop_rect() is not None)
        self.area_clear_button.setEnabled(bool(self._page_areas))
        if self.canvas.tool is IsolationTool.ERASE:
            hint = "Drag a rectangle over anything to leave out of the comparison, on any page, then press Erase selected area."
        else:
            hint = "Drag a rectangle on a page, then press Use as the page's area: only that part of the page is compared."
        facts = []
        if erased:
            facts.append(f"{erased} area{'s' if erased != 1 else ''} erased")
        if self._page_areas:
            facts.append("own area on page " + ", ".join(str(i + 1) for i in sorted(self._page_areas)))
        self.status_label.setText(hint + (f"  ({'; '.join(facts)})" if facts else ""))

    def _set_page_area(self) -> None:
        drawn = self.canvas.crop_rect()
        if drawn is None:
            return
        parts = to_pages(drawn, self._size, self._places)
        if not parts:
            return
        # the page it mostly lies on
        index = max(parts, key=lambda i: (parts[i].x1 - parts[i].x0) * (parts[i].y1 - parts[i].y0) * self._places[i][2] * self._places[i][3])
        self._page_areas[index] = parts[index]
        self.canvas.clear_crop()
        self._show_page_areas()
        self._update_states()

    def _clear_page_areas(self) -> None:
        self._page_areas = {}
        self._show_page_areas()
        self._update_states()

    def _show_page_areas(self) -> None:
        self.canvas.set_marked_areas([to_stack(i, r, self._size, self._places) for i, r in sorted(self._page_areas.items())])

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self.canvas.erase_selection() is not None:
            self.canvas.apply_erase_selection()
            return
        super().keyPressEvent(event)

    def _size_to_screen(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            self.resize(1000, 750)
            return
        available = screen.availableGeometry()
        self.resize(max(900, int(available.width() * _SCREEN_SHARE)), max(650, int(available.height() * _SCREEN_SHARE)))

    # -- result -------------------------------------------------------------------------------------------------------
    def _on_save(self) -> None:
        erased: dict[int, list[NormalizedRect]] = {}
        for region in self.canvas.erase_regions():
            for index, part in to_pages(region, self._size, self._places).items():
                erased.setdefault(index, []).append(part)
        result: PageAreas = {}
        for index in sorted(set(erased) | set(self._page_areas)):
            result[index] = IsolationGeometry(boundary=self._page_areas.get(index), erase_regions=erased.get(index, []), boundary_origin="manual")
        self._result = result
        self.accept()

    def result_areas(self) -> PageAreas | None:
        """What was set on each page (only pages with anything on them); None until saved."""
        return self._result

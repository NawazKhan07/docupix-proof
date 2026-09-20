"""The comparison viewer (SRS §3.8, Architecture §3.1).

One component with four render strategies -- side by side, flicker, overlay,
grayscale-with-colour-differences -- all driven by the same aligned image
pair, so a difference is at the same place in every view (FR-7.4). Both
panes draw markers from the same shared-space box and share one zoom, so
nothing here can put a finding in two different places.
"""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np
from PySide6.QtCore import QPointF, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.engine.file_loader import LoadedArtwork
from app.engine.tile_renderer import (
    TileRect,
    is_vector,
    render_aligned_vector_tile,
    render_raster_tile,
    render_vector_tile,
)
from app.logging_config import StageTimer
from app.models.diff import BoundingBox, DiffCluster
from app.ui.composition import (
    ANIMATED_MODES,
    SINGLE_PANE_MODES,
    VIEW_MODE_LABELS,
    ViewMode,
    compose_grayscale,
    compose_overlay,
)
from app.ui.pane_view import ImageStack, PaneView

logger = logging.getLogger(__name__)

ZOOM_STEP = 1.25
MIN_ZOOM = 0.05
MAX_ZOOM = 8.0
FOCUS_MAX_ZOOM = 3.0  # how far "jump to finding" will zoom in by itself
FIT_MARGIN_PX = 4
FLICKER_INTERVAL_MS = 600

# FR-9.1: above 100% the 300 DPI raster is being magnified, so the visible
# region is re-rendered at screen resolution instead.
DETAIL_MIN_ZOOM = 1.0
DETAIL_MARGIN_PX = 96  # render a little beyond the viewport so short pans stay sharp
DETAIL_DEBOUNCE_MS = 120  # wait for zoom/scroll to settle before paying for a render
DETAIL_MAX_TILE_PX = 6000  # bound memory on very large or high-DPI displays

_PANE_CAPTIONS = {
    ViewMode.SIDE_BY_SIDE: ("Original", "Suspect (aligned to original)"),
    ViewMode.FLICKER: ("Flicker: Original / Suspect", ""),
    ViewMode.OVERLAY: ("Overlay: Original + Suspect", ""),
    ViewMode.GRAYSCALE: ("Grayscale: differences in colour", ""),
}


class ComparisonViewport(QWidget):
    markerClicked = Signal(int)  # cluster id
    zoomChanged = Signal(float)
    viewModeChanged = Signal(object)  # ViewMode
    mergeRegionSelected = Signal(object)  # BoundingBox (shared space): the reviewer drew a rectangle to merge findings in
    undoMergeRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mode = ViewMode.SIDE_BY_SIDE
        self._zoom = 1.0
        self._fit_mode = True
        self._scroll_lock = True
        self._syncing_scroll = False
        self._phase = 0
        self._selected_id: int | None = None
        self._marker_clusters: list[DiffCluster] = []
        self._original_rgb: np.ndarray | None = None
        self._suspect_rgb: np.ndarray | None = None
        # Scan vs Digital also has each file as ink on paper (black and white); it is what is shown by default.
        self._colour_pair: tuple[np.ndarray, np.ndarray] | None = None
        self._ink_pair: tuple[np.ndarray, np.ndarray] | None = None
        self._showing_ink = False
        self._sources: tuple[LoadedArtwork | None, LoadedArtwork | None, np.ndarray | None] = (None, None, None)
        self._stacks: dict[str, ImageStack] = {}
        self._original_artwork: LoadedArtwork | None = None
        self._suspect_artwork: LoadedArtwork | None = None
        self._alignment_transform: np.ndarray | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addLayout(self._build_toolbar())
        layout.addWidget(self.merge_row)

        self.original_pane = PaneView(self)
        self.suspect_pane = PaneView(self)
        self._original_caption = QLabel()
        self._suspect_caption = QLabel()
        for caption in (self._original_caption, self._suspect_caption):
            caption.setProperty("secondary", "true")

        self._suspect_holder = self._holder(self._suspect_caption, self.suspect_pane)
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(self._holder(self._original_caption, self.original_pane))
        splitter.addWidget(self._suspect_holder)
        splitter.setChildrenCollapsible(False)
        # Two panes showing the same page should start out the same width;
        # left to size hints, the first came out less than half the second.
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([1000, 1000])
        layout.addWidget(splitter, stretch=1)

        self._timer = QTimer(self)
        self._timer.setInterval(FLICKER_INTERVAL_MS)
        self._timer.timeout.connect(self._on_tick)

        self._detail_timer = QTimer(self)
        self._detail_timer.setSingleShot(True)
        self._detail_timer.setInterval(DETAIL_DEBOUNCE_MS)
        self._detail_timer.timeout.connect(self.update_detail_now)

        for pane in (self.original_pane, self.suspect_pane):
            pane.markerClicked.connect(self.markerClicked)
            pane.regionSelected.connect(self._on_region_selected)
            pane.zoomStepRequested.connect(self._on_wheel_zoom)
            pane.zoomRequested.connect(self._on_zoom_requested)
        self.original_pane.resized.connect(self._on_pane_resized)
        self._connect_scroll_sync()
        for pane in (self.original_pane, self.suspect_pane):
            pane.horizontalScrollBar().valueChanged.connect(self._schedule_detail)
            pane.verticalScrollBar().valueChanged.connect(self._schedule_detail)
        self._apply_mode_layout()

    # -- construction --------------------------------------------------------
    def _build_toolbar(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)

        self._mode_buttons: dict[ViewMode, QPushButton] = {}
        group = QButtonGroup(self)
        group.setExclusive(True)
        for mode in ViewMode:
            button = QPushButton(VIEW_MODE_LABELS[mode])
            button.setCheckable(True)
            button.setProperty("variant", "segment")
            group.addButton(button)
            button.clicked.connect(lambda _checked=False, m=mode: self.set_view_mode(m))
            row.addWidget(button)
            self._mode_buttons[mode] = button
        self._mode_buttons[ViewMode.SIDE_BY_SIDE].setChecked(True)

        row.addSpacing(16)
        self.scroll_lock_check = QCheckBox("Scroll lock")
        self.scroll_lock_check.setChecked(True)
        self.scroll_lock_check.toggled.connect(self.set_scroll_lock)
        row.addWidget(self.scroll_lock_check)
        self.ink_check = QCheckBox("Black && white")
        self.ink_check.setToolTip(
            "Show each file as ink on paper: background colours removed, everything printed in black. "
            "This is the picture that was read for text and compared. Untick for the colour originals."
        )
        self.ink_check.setChecked(True)
        self.ink_check.setVisible(False)  # only in Scan vs Digital
        self.ink_check.toggled.connect(self._on_ink_toggled)
        row.addWidget(self.ink_check)
        self.compared_area_check = QCheckBox("Show compared area only")
        self.compared_area_check.setToolTip(
            "Erase everything the comparison left out (white background) and outline what it covered. "
            "Untick to see the whole page."
        )
        self.compared_area_check.setChecked(True)
        self.compared_area_check.setVisible(False)  # only when part of the page was left out
        self.compared_area_check.toggled.connect(self._on_compared_area_toggled)
        row.addWidget(self.compared_area_check)

        row.addSpacing(16)
        self.merge_button = QPushButton("Merge findings")
        self.merge_button.setCheckable(True)
        self.merge_button.setProperty("variant", "segment")
        self.merge_button.setToolTip(
            "Where several small differences are really one (the bars of a barcode, the letters of a changed word): "
            "drag a rectangle round them on the page and they become one finding."
        )
        self.merge_button.setMinimumWidth(self.merge_button.sizeHint().width())  # the label is never clipped
        # What the merge tool has to say (and Undo) sits on a slim row of its own, only while there is something on it.
        self.undo_merge_button = QPushButton("Undo merge")
        self.undo_merge_button.setToolTip("Split the last merged finding back into the findings it was made of.")
        self.undo_merge_button.setVisible(False)
        self.merge_notice = QLabel("")
        self.merge_notice.setProperty("secondary", "true")
        self.merge_row = QWidget()
        merge_layout = QHBoxLayout(self.merge_row)
        merge_layout.setContentsMargins(0, 0, 0, 0)
        merge_layout.setSpacing(8)
        merge_layout.addWidget(self.merge_notice)
        merge_layout.addWidget(self.undo_merge_button)
        merge_layout.addStretch(1)
        self.merge_row.setVisible(False)
        self.merge_button.toggled.connect(self.set_merge_mode)
        self.undo_merge_button.clicked.connect(self.undoMergeRequested)
        for widget in (self.merge_button,):
            row.addWidget(widget)

        row.addStretch(1)
        self.zoom_out_button = QPushButton("−")
        self.zoom_in_button = QPushButton("+")
        self.fit_button = QPushButton("Reset")
        self.zoom_label = QLabel("100%")
        self.zoom_label.setMinimumWidth(52)
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        for button in (self.zoom_out_button, self.zoom_in_button):
            button.setFixedWidth(36)
            button.setProperty("variant", "icon")
        self.zoom_out_button.clicked.connect(self.zoom_out)
        self.zoom_in_button.clicked.connect(self.zoom_in)
        self.fit_button.clicked.connect(self.fit_to_view)
        for widget in (self.zoom_out_button, self.zoom_label, self.zoom_in_button, self.fit_button):
            row.addWidget(widget)
        return row

    def _holder(self, caption: QLabel, pane: PaneView) -> QFrame:
        holder = QFrame(self)
        box = QVBoxLayout(holder)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(4)
        box.addWidget(caption)
        box.addWidget(pane, stretch=1)
        return holder

    def _connect_scroll_sync(self) -> None:
        pairs = (
            (self.original_pane.horizontalScrollBar(), self.suspect_pane.horizontalScrollBar()),
            (self.original_pane.verticalScrollBar(), self.suspect_pane.verticalScrollBar()),
        )
        for first, second in pairs:
            first.valueChanged.connect(lambda value, other=second: self._sync_scroll(other, value))
            second.valueChanged.connect(lambda value, other=first: self._sync_scroll(other, value))

    # -- images --------------------------------------------------------------
    def set_images(
        self,
        original_rgb: np.ndarray,
        aligned_suspect_rgb: np.ndarray,
        ink_pair: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        """Load the aligned pair every mode is drawn from (FR-7.4).

        `ink_pair` is the same pair as ink on paper (grey images, 255 = background): when given, it is what
        is shown, with a checkbox to see the colour originals instead.
        """
        self._colour_pair = (original_rgb, aligned_suspect_rgb)
        self._ink_pair = (
            (cv2.cvtColor(ink_pair[0], cv2.COLOR_GRAY2RGB), cv2.cvtColor(ink_pair[1], cv2.COLOR_GRAY2RGB))
            if ink_pair is not None
            else None
        )
        self._showing_ink = self._ink_pair is not None
        self.ink_check.blockSignals(True)
        self.ink_check.setChecked(self._showing_ink)
        self.ink_check.blockSignals(False)
        self.ink_check.setVisible(self._ink_pair is not None)
        self._selected_id = None
        self._fit_mode = True
        self._phase = 0
        self._load_active_pair()
        self._apply_fit()

    def _load_active_pair(self) -> None:
        """Make the pair being shown (ink or colour) the one every mode is drawn from."""
        original, suspect = (self._ink_pair if self._showing_ink and self._ink_pair else self._colour_pair)
        self._original_rgb, self._suspect_rgb = original, suspect
        self._stacks = {"original": ImageStack(original), "suspect": ImageStack(suspect)}
        # Zoomed-in tiles are re-rendered from a PDF's vector data; the ink pictures exist only as pixels.
        original_source, suspect_source, transform = self._sources
        if self._showing_ink:
            original_source = suspect_source = transform = None
        self._original_artwork, self._suspect_artwork, self._alignment_transform = original_source, suspect_source, transform
        for pane in (self.original_pane, self.suspect_pane):
            pane.clear_detail()
        self._rebuild_for_mode()
        self._schedule_detail()

    def _on_ink_toggled(self, checked: bool) -> None:
        if self._ink_pair is None or checked == self._showing_ink:
            return
        self._showing_ink = checked
        self._load_active_pair()

    @property
    def showing_ink(self) -> bool:
        return self._showing_ink

    def set_sources(
        self,
        original: LoadedArtwork | None,
        suspect: LoadedArtwork | None,
        alignment_transform: np.ndarray | None,
    ) -> None:
        """Give the viewer the files themselves, so zoomed-in views can be re-rendered.

        Without these (or for image files, which have no vector data) the
        viewer still works: it falls back to resampling the 300 DPI pair.
        """
        self._sources = (original, suspect, alignment_transform)
        if self._showing_ink:
            original = suspect = alignment_transform = None
        self._original_artwork = original
        self._suspect_artwork = suspect
        self._alignment_transform = alignment_transform
        self._schedule_detail()

    def release_sources(self) -> None:
        """Drop the file references, e.g. because the files were closed or replaced."""
        # Set the fields directly: going through `set_sources` would schedule
        # a render, which is exactly what releasing is meant to prevent.
        self._detail_timer.stop()
        self._sources = (None, None, None)
        self._original_artwork = None
        self._suspect_artwork = None
        self._alignment_transform = None
        for pane in (self.original_pane, self.suspect_pane):
            pane.clear_detail()

    @property
    def has_images(self) -> bool:
        return self._original_rgb is not None

    # -- modes ---------------------------------------------------------------
    @property
    def view_mode(self) -> ViewMode:
        return self._mode

    def set_view_mode(self, mode: ViewMode) -> None:
        if mode is self._mode and self._mode_buttons[mode].isChecked():
            return
        self._mode = mode
        self._mode_buttons[mode].setChecked(True)
        self._phase = 0
        self._apply_mode_layout()
        self._rebuild_for_mode()
        self.viewModeChanged.emit(mode)
        if self._fit_mode:
            self._apply_fit()
        self._schedule_detail()

    def _apply_mode_layout(self) -> None:
        single = self._mode in SINGLE_PANE_MODES
        self._suspect_holder.setVisible(not single)
        original_caption, suspect_caption = _PANE_CAPTIONS[self._mode]
        self._original_caption.setText(original_caption)
        self._suspect_caption.setText(suspect_caption)
        self.scroll_lock_check.setEnabled(not single)  # nothing to lock with one pane

        if self._mode in ANIMATED_MODES and self.isVisible():
            self._timer.start()
        else:
            self._timer.stop()

    def _rebuild_for_mode(self) -> None:
        """Set each pane's images for the current mode, then redraw markers."""
        if not self.has_images:
            return

        for key in ("overlay", "gray0", "gray1"):
            self._stacks.pop(key, None)  # composites are rebuilt per mode

        if self._mode is ViewMode.SIDE_BY_SIDE:
            self.original_pane.set_stacks({"original": self._stacks["original"]})
            self.suspect_pane.set_stacks({"suspect": self._stacks["suspect"]})
        elif self._mode is ViewMode.FLICKER:
            self.original_pane.set_stacks(
                {"original": self._stacks["original"], "suspect": self._stacks["suspect"]}
            )
            self.original_pane.show_stack("original")
        elif self._mode is ViewMode.OVERLAY:
            self._stacks["overlay"] = ImageStack(
                compose_overlay(self._original_rgb, self._suspect_rgb)
            )
            self.original_pane.set_stacks({"overlay": self._stacks["overlay"]})
        else:  # GRAYSCALE
            self._build_grayscale_stacks()

        self._refresh_markers()

    def _build_grayscale_stacks(self) -> None:
        boxes = [cluster.bbox_in_original for cluster in self._marker_clusters]
        for phase in (0, 1):
            self._stacks[f"gray{phase}"] = ImageStack(
                compose_grayscale(self._original_rgb, self._suspect_rgb, boxes, phase)
            )
        self.original_pane.set_stacks(
            {"gray0": self._stacks["gray0"], "gray1": self._stacks["gray1"]}
        )
        self.original_pane.show_stack(f"gray{self._phase}")

    def _on_tick(self) -> None:
        self._phase ^= 1
        if self._mode is ViewMode.FLICKER:
            self.original_pane.show_stack("original" if self._phase == 0 else "suspect")
        elif self._mode is ViewMode.GRAYSCALE:
            self.original_pane.show_stack(f"gray{self._phase}")

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._mode in ANIMATED_MODES:
            self._timer.start()
        self._schedule_detail()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._timer.stop()  # no point animating a view nobody can see
        self._detail_timer.stop()

    @property
    def is_animating(self) -> bool:
        return self._timer.isActive()

    # -- merging findings --------------------------------------------------------
    def set_merge_mode(self, on: bool) -> None:
        """While on, dragging on the page draws a rectangle to merge the findings in, instead of panning."""
        for pane in (self.original_pane, self.suspect_pane):
            pane.set_region_selection(on)
        if self.merge_button.isChecked() != on:
            self.merge_button.blockSignals(True)
            self.merge_button.setChecked(on)
            self.merge_button.blockSignals(False)
        self.merge_notice.setText("Drag a rectangle round the findings to merge." if on else "")
        self._show_merge_row()

    @property
    def merge_mode(self) -> bool:
        return self.merge_button.isChecked()

    def set_can_undo_merge(self, can: bool) -> None:
        self.undo_merge_button.setVisible(can)
        self._show_merge_row()

    def show_merge_notice(self, text: str) -> None:
        self.merge_notice.setText(text)
        self._show_merge_row()

    def _show_merge_row(self) -> None:
        """The merge row is there while it has a message or an Undo to offer, and out of the way otherwise."""
        self.merge_row.setVisible(bool(self.merge_notice.text()) or not self.undo_merge_button.isHidden())

    def _on_region_selected(self, rect) -> None:
        self.set_merge_mode(False)  # one rectangle at a time: a stray drag afterwards should pan, not merge
        self.mergeRegionSelected.emit(BoundingBox(rect.left(), rect.top(), rect.right(), rect.bottom()))

    # -- markers -------------------------------------------------------------
    def set_marker_clusters(self, clusters: list[DiffCluster]) -> None:
        """The findings to draw (the caller filters out ignored/hidden ones)."""
        self._marker_clusters = list(clusters)
        if self._mode is ViewMode.GRAYSCALE and self.has_images:
            self._build_grayscale_stacks()  # the coloured regions follow the findings
        self._refresh_markers()
        self._schedule_detail()

    def set_compared_area(self, polygons: list[list[tuple[float, float]]] | None) -> None:
        """Show what the comparison covered (None when it covered the whole page)."""
        self.compared_area_check.setVisible(polygons is not None)
        for pane in (self.original_pane, self.suspect_pane):
            pane.set_compared_area(polygons)
            pane.show_compared_area(self.compared_area_check.isChecked())

    def _on_compared_area_toggled(self, checked: bool) -> None:
        for pane in (self.original_pane, self.suspect_pane):
            pane.show_compared_area(checked)

    def set_selected(self, cluster_id: int | None) -> None:
        self._selected_id = cluster_id
        self._refresh_markers()

    def _refresh_markers(self) -> None:
        for pane in self._visible_panes():
            pane.set_markers(self._marker_clusters, self._selected_id)

    def _visible_panes(self) -> list[PaneView]:
        if self._mode is ViewMode.SIDE_BY_SIDE:
            return [self.original_pane, self.suspect_pane]
        return [self.original_pane]

    def focus_cluster(self, cluster: DiffCluster) -> None:
        """Select a finding and bring it into view in every visible pane."""
        self.set_selected(cluster.id)
        self.focus_box(cluster.bbox_in_original)

    def focus_box(self, box: BoundingBox) -> None:
        """Bring a region (a finding, a decoded symbol) into view in every visible pane."""
        pane = self.original_pane
        viewport = pane.viewport()
        if box.width > 0 and box.height > 0 and viewport.width() > 0:
            # Frame the finding with room around it, but never zoom *out*.
            fits = min(viewport.width() / (box.width * 3), viewport.height() / (box.height * 3))
            target = min(FOCUS_MAX_ZOOM, MAX_ZOOM, fits)
            if target > self._zoom:
                self.set_zoom(target)
        for visible in self._visible_panes():
            visible.center_on_box(box)

    # -- zoom (FR-8.6) -------------------------------------------------------
    @property
    def zoom(self) -> float:
        return self._zoom

    @property
    def zoom_percent(self) -> int:
        return round(self._zoom * 100)

    def set_zoom(self, zoom: float, from_fit: bool = False, anchor: QPointF | None = None) -> None:
        """Zoom both panes together, and show the resulting percentage.

        `anchor` is a position in a pane's viewport that stays put (a pinch's
        centre, the cursor); both panes are the same size, so it applies to both.
        """
        self._zoom = max(MIN_ZOOM, min(MAX_ZOOM, zoom))
        if not from_fit:
            self._fit_mode = False
        if anchor is None:
            self.original_pane.set_zoom(self._zoom)
            self.suspect_pane.set_zoom(self._zoom)
        else:
            # Each pane repositions itself about the anchor. The scroll lock must
            # stay out of it meanwhile: one pane's move, mirrored onto the other
            # before that one has zoomed, would be clamped to its old range.
            self._syncing_scroll = True
            try:
                self.original_pane.set_zoom(self._zoom, anchor)
                self.suspect_pane.set_zoom(self._zoom, anchor)
            finally:
                self._syncing_scroll = False
            if self._scroll_lock:
                self._sync_scroll(self.suspect_pane.horizontalScrollBar(), self.original_pane.horizontalScrollBar().value())
                self._sync_scroll(self.suspect_pane.verticalScrollBar(), self.original_pane.verticalScrollBar().value())
        self.zoom_label.setText(f"{self.zoom_percent}%")
        self.zoom_in_button.setEnabled(self._zoom < MAX_ZOOM)
        self.zoom_out_button.setEnabled(self._zoom > MIN_ZOOM)
        self.zoomChanged.emit(self._zoom)
        self._schedule_detail()

    def zoom_in(self) -> None:
        self.set_zoom(self._zoom * ZOOM_STEP)

    def zoom_out(self) -> None:
        self.set_zoom(self._zoom / ZOOM_STEP)

    def fit_to_view(self) -> None:
        """Reset: show the whole artwork (FR-8.6's reset control)."""
        self._fit_mode = True
        self._apply_fit()

    def _apply_fit(self) -> None:
        width, height = self.original_pane.image_size
        viewport = self.original_pane.viewport()
        if width <= 0 or height <= 0 or viewport.width() <= 0 or viewport.height() <= 0:
            return  # not laid out yet; the pane's resize event will call back in
        fit = min(
            (viewport.width() - FIT_MARGIN_PX) / width,
            (viewport.height() - FIT_MARGIN_PX) / height,
        )
        self.set_zoom(fit, from_fit=True)

    def _on_pane_resized(self) -> None:
        if self._fit_mode and self.has_images:
            self._apply_fit()

    def _on_wheel_zoom(self, direction: int) -> None:
        self.zoom_in() if direction > 0 else self.zoom_out()

    def _on_zoom_requested(self, factor: float, anchor: QPointF) -> None:
        """A pinch or Ctrl + wheel: scale continuously about the fingers / cursor."""
        self.set_zoom(self._zoom * factor, anchor=anchor)

    # -- full-quality detail when zoomed in (FR-9.1) ---------------------------
    def _schedule_detail(self, *_args) -> None:
        if self.has_images and self.isVisible():
            self._detail_timer.start()  # restarts the wait, so a flurry is one render

    @property
    def has_detail(self) -> bool:
        return self.original_pane.has_detail or self.suspect_pane.has_detail

    def update_detail_now(self) -> None:
        """Render the visible region at screen resolution, or drop it if zoomed out.

        Called after zoom and scrolling settle. A render that fails must not
        take the viewer down mid-review, so failures are logged and the view
        simply keeps showing the magnified base image.
        """
        if not self.has_images or not self.isVisible():
            return
        if self._zoom <= DETAIL_MIN_ZOOM + 1e-9:
            for pane in (self.original_pane, self.suspect_pane):
                pane.clear_detail()
            return

        for pane in self._visible_panes():
            rect = self._tile_rect_for(pane)
            if rect is None:
                pane.clear_detail()
                continue
            try:
                with StageTimer(logger, "viewer_detail_tile"):
                    tiles = self._tiles_for(pane, rect)
            except Exception:  # a native PDF renderer at a UI boundary; see docstring
                logger.exception("could not render a full-quality tile")
                pane.clear_detail()
                continue
            pane.set_detail(tiles, (rect.x, rect.y), self._zoom)

    def _tile_rect_for(self, pane: PaneView) -> TileRect | None:
        x0, y0, x1, y1 = pane.visible_device_rect()
        image_width, image_height = pane.image_size
        x0, y0 = max(0, x0 - DETAIL_MARGIN_PX), max(0, y0 - DETAIL_MARGIN_PX)
        x1 = min(math.ceil(image_width * self._zoom), x1 + DETAIL_MARGIN_PX)
        y1 = min(math.ceil(image_height * self._zoom), y1 + DETAIL_MARGIN_PX)
        if x1 <= x0 or y1 <= y0:
            return None
        return TileRect(x0, y0, min(x1 - x0, DETAIL_MAX_TILE_PX), min(y1 - y0, DETAIL_MAX_TILE_PX))

    def _original_tile(self, rect: TileRect) -> np.ndarray:
        if is_vector(self._original_artwork):
            return render_vector_tile(self._original_artwork, self._zoom, rect)
        return render_raster_tile(self._original_rgb, self._zoom, rect)

    def _suspect_tile(self, rect: TileRect) -> np.ndarray:
        if is_vector(self._suspect_artwork) and self._alignment_transform is not None:
            return render_aligned_vector_tile(
                self._suspect_artwork, self._alignment_transform, self._zoom, rect
            )
        return render_raster_tile(self._suspect_rgb, self._zoom, rect)

    def _tiles_for(self, pane: PaneView, rect: TileRect) -> dict[str, np.ndarray]:
        """The tile(s) this pane needs, keyed like the pane's own images."""
        if self._mode is ViewMode.SIDE_BY_SIDE:
            if pane is self.original_pane:
                return {"original": self._original_tile(rect)}
            return {"suspect": self._suspect_tile(rect)}

        original, suspect = self._original_tile(rect), self._suspect_tile(rect)
        if self._mode is ViewMode.FLICKER:
            return {"original": original, "suspect": suspect}
        if self._mode is ViewMode.OVERLAY:
            return {"overlay": compose_overlay(original, suspect)}

        zoom = self._zoom
        boxes = [
            BoundingBox(
                b.x0 * zoom - rect.x, b.y0 * zoom - rect.y, b.x1 * zoom - rect.x, b.y1 * zoom - rect.y
            )
            for b in (cluster.bbox_in_original for cluster in self._marker_clusters)
        ]
        return {f"gray{phase}": compose_grayscale(original, suspect, boxes, phase) for phase in (0, 1)}

    # -- scroll lock (FR-8.5) --------------------------------------------------
    @property
    def scroll_lock(self) -> bool:
        return self._scroll_lock

    def set_scroll_lock(self, enabled: bool) -> None:
        self._scroll_lock = enabled
        if self.scroll_lock_check.isChecked() != enabled:
            self.scroll_lock_check.setChecked(enabled)
        if enabled:  # re-couple the second pane to wherever the first one is
            self._sync_scroll(self.suspect_pane.horizontalScrollBar(),
                              self.original_pane.horizontalScrollBar().value())
            self._sync_scroll(self.suspect_pane.verticalScrollBar(),
                              self.original_pane.verticalScrollBar().value())

    def _sync_scroll(self, other, value: int) -> None:
        if not self._scroll_lock or self._syncing_scroll or other.value() == value:
            return
        self._syncing_scroll = True
        try:
            other.setValue(value)
        finally:
            self._syncing_scroll = False

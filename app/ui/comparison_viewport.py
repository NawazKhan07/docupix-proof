"""The comparison viewer (SRS §3.8, Architecture §3.1).

One component with four render strategies -- side by side, flicker, overlay,
grayscale-with-colour-differences -- all driven by the same aligned image
pair, so a difference is at the same place in every view (FR-7.4). Both
panes draw markers from the same shared-space box and share one zoom, so
nothing here can put a finding in two different places.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
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

from app.models.diff import DiffCluster
from app.ui.composition import (
    ANIMATED_MODES,
    SINGLE_PANE_MODES,
    VIEW_MODE_LABELS,
    ViewMode,
    compose_grayscale,
    compose_overlay,
)
from app.ui.pane_view import ImageStack, PaneView

ZOOM_STEP = 1.25
MIN_ZOOM = 0.05
MAX_ZOOM = 8.0
FOCUS_MAX_ZOOM = 3.0  # how far "jump to finding" will zoom in by itself
FIT_MARGIN_PX = 4
FLICKER_INTERVAL_MS = 600

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
        self._stacks: dict[str, ImageStack] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addLayout(self._build_toolbar())

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

        for pane in (self.original_pane, self.suspect_pane):
            pane.markerClicked.connect(self.markerClicked)
            pane.zoomStepRequested.connect(self._on_wheel_zoom)
        self.original_pane.resized.connect(self._on_pane_resized)
        self._connect_scroll_sync()
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
    def set_images(self, original_rgb: np.ndarray, aligned_suspect_rgb: np.ndarray) -> None:
        """Load the aligned pair every mode is drawn from (FR-7.4)."""
        self._original_rgb = original_rgb
        self._suspect_rgb = aligned_suspect_rgb
        self._stacks = {
            "original": ImageStack(original_rgb),
            "suspect": ImageStack(aligned_suspect_rgb),
        }
        self._selected_id = None
        self._fit_mode = True
        self._phase = 0
        self._rebuild_for_mode()
        self._apply_fit()

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

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._timer.stop()  # no point animating a view nobody can see

    @property
    def is_animating(self) -> bool:
        return self._timer.isActive()

    # -- markers -------------------------------------------------------------
    def set_marker_clusters(self, clusters: list[DiffCluster]) -> None:
        """The findings to draw (the caller filters out ignored/hidden ones)."""
        self._marker_clusters = list(clusters)
        if self._mode is ViewMode.GRAYSCALE and self.has_images:
            self._build_grayscale_stacks()  # the coloured regions follow the findings
        self._refresh_markers()

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
        box = cluster.bbox_in_original
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

    def set_zoom(self, zoom: float, from_fit: bool = False) -> None:
        """Zoom both panes together, and show the resulting percentage."""
        self._zoom = max(MIN_ZOOM, min(MAX_ZOOM, zoom))
        if not from_fit:
            self._fit_mode = False
        self.original_pane.set_zoom(self._zoom)
        self.suspect_pane.set_zoom(self._zoom)
        self.zoom_label.setText(f"{self.zoom_percent}%")
        self.zoom_in_button.setEnabled(self._zoom < MAX_ZOOM)
        self.zoom_out_button.setEnabled(self._zoom > MIN_ZOOM)
        self.zoomChanged.emit(self._zoom)

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

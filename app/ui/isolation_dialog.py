"""Dialog wrapping InteractiveCanvas so one file's artwork area can be chosen from the upload
panel: crop it, lift it from its background, lay the other file's lifted artwork over it, or erase parts.

**Lift Subject** is how the artwork is separated from everything else on the sheet. Press and
hold on the artwork and only what is inside its border is lifted -- for a packaging sheet, the
carton inside its die-cut line, and none of the title, colour legend, crop marks or notes
around it. The outline can be corrected with Keep / Drop strokes, saved as a transparent PNG
cutout, and used as the area to compare.

**Place Subject** is for a file with no clear border of its own (a photograph, a plain scan):
the artwork lifted from the other file is laid over it, snapped into place by matching what is
printed (Auto-fit), adjusted by hand if need be, and this file is trimmed to that outline. When the
other file has a lifted subject, the dialog offers it at once in a banner, so this route does not have
to be found.

**Erase** selects a rectangle the way a crop is selected (drag it, adjust it by its edges, move it),
and Erase selected area leaves it out of the comparison. Any number of areas can be erased.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.config import AppConfig
from app.engine.alignment.aligner import AlignmentError
from app.engine.file_loader import WORKING_DPI, LoadedArtwork
from app.engine.isolation import crop_tool
from app.engine.isolation.cutout import Cutout, CutoutError
from app.engine.isolation.placement import Placement, fit_placement
from app.engine.isolation.subject_lifter import HintStroke, LiftedSubject, NoSubjectFoundError, lift_subject
from app.models.isolation import (
    IsolationGeometry,
    IsolationTool,
    NormalizedPolygon,
    NormalizedRect,
)
from app.ui.interactive_canvas import InteractiveCanvas

_STATUS_TEXT = {
    IsolationTool.CROP: "Drag on the page to draw a crop. Drag its edges or corners to resize it, or its inside to move it.",
    IsolationTool.LIFT_SUBJECT: (
        "Press and hold on the artwork to lift it: only the area inside its border is taken, "
        "not the text, colour legend or marks around it. Use Keep / Drop to correct the outline."
    ),
    IsolationTool.PLACE_OUTLINE: (
        "Place Subject lays the artwork lifted from the other file over this one, so this file is trimmed to the "
        "same area: Auto-fit matches what is printed, you can drag it into place, then press Trim to outline."
    ),
    IsolationTool.ERASE: (
        "Drag on the page to select an area to leave out, adjust it by its edges, then press Erase selected "
        "area (or Enter). Erased areas are not compared."
    ),
}
_NAVIGATION_HINT = "Pinch or Ctrl + wheel to zoom  ·  two-finger scroll, Space + drag or middle-drag to pan"
_SCREEN_SHARE = 0.85  # how much of the screen the dialog may take, so the whole page can be seen
_MM_PER_INCH = 25.4
_METHODS = (("Automatic", "auto"), ("Border line", "border"), ("Colour", "colour"))
_BRUSH_UNIT = 1 / 1000  # the brush slider is in thousandths of the page's long edge
_PLACED_MARGIN_PX = 1.5  # a placed outline is only as exact as the fit: start comparing this far further in
# A colour lift much smaller than the artwork already lifted from the other file has probably caught
# only the printed part of a scan (a pale flap is close to the paper it lies on).
_SMALLER_THAN_REFERENCE = 0.7


@dataclass
class SubjectReference:
    """The other file's lifted subject, offered here to be laid over this file."""

    cutout: Cutout
    source_name: str


class IsolationDialog(QDialog):
    """Lets the user crop, lift the artwork from its background, place an outline, or erase, on one file."""

    def __init__(
        self,
        artwork: LoadedArtwork,
        title: str,
        parent=None,
        reference: SubjectReference | None = None,
        config: AppConfig | None = None,
        partner_title: str | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(f"Isolate Artwork — {title}")
        self._size_to_screen()
        self._artwork = artwork
        self._config = config or AppConfig()
        self._rendered = artwork.render()
        self._reference = reference
        self._partner_title = partner_title or "the other file"
        self._result: Optional[IsolationGeometry] = None
        self._result_cutout: Optional[Cutout] = None

        # What has been lifted, and the corrections made to it.
        self._subject: LiftedSubject | None = None
        self._subject_seed: tuple[float, float] | None = None
        self._hints: list[HintStroke] = []
        # The region that will be compared: a lifted outline, or a placed one; and how far inside its
        # edge (image pixels) comparing starts.
        self._outline: tuple[NormalizedPolygon, str] | None = None
        self._outline_inset_px = 0.0
        # Shown above the page when the other file's lifted artwork can be laid over this one.
        self._banner_hint: str | None = None
        # The cutout laid over the page.
        self._cutout: Cutout | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        tool_row = QHBoxLayout()
        tool_row.setSpacing(6)
        self.crop_button = QPushButton("Crop")
        self.lift_button = QPushButton("Lift Subject")
        self.place_button = QPushButton("Place Subject")
        self.place_button.setToolTip(
            "Lay the artwork lifted from the other file over this one, so this file is trimmed to the same area."
        )
        self.erase_button = QPushButton("Erase")
        self._tool_buttons = (
            (self.crop_button, IsolationTool.CROP),
            (self.lift_button, IsolationTool.LIFT_SUBJECT),
            (self.place_button, IsolationTool.PLACE_OUTLINE),
            (self.erase_button, IsolationTool.ERASE),
        )
        for button, _tool in self._tool_buttons:
            button.setCheckable(True)
            button.setProperty("variant", "segment")
            tool_row.addWidget(button)
        self.crop_button.setChecked(True)
        tool_row.addStretch(1)

        # Zoom controls: the page opens fitted, and these (or a pinch / Ctrl + wheel) change that.
        self.zoom_out_button = QPushButton("−")
        self.zoom_label = QLabel("100%")
        self.zoom_in_button = QPushButton("+")
        self.fit_button = QPushButton("Fit")
        self.actual_button = QPushButton("100%")
        for button in (self.zoom_out_button, self.zoom_in_button):
            button.setProperty("variant", "icon")
        for button in (self.fit_button, self.actual_button):
            button.setProperty("variant", "segment")
        self.zoom_label.setMinimumWidth(48)
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        for widget in (self.zoom_out_button, self.zoom_label, self.zoom_in_button, self.fit_button, self.actual_button):
            tool_row.addWidget(widget)
        layout.addLayout(tool_row)

        layout.addWidget(self._build_reference_banner())

        # The options of the current tool.
        self.options = QStackedWidget()
        self.options.addWidget(QWidget())  # crop has none
        self.options.addWidget(self._build_lift_options())
        self.options.addWidget(self._build_place_options())
        self.options.addWidget(self._build_erase_options())
        layout.addWidget(self.options)

        self.canvas = InteractiveCanvas(self)
        self.canvas.set_image(self._rendered)
        self.canvas.set_hold_timing(self._config.subject.hold_ms, self._config.subject.hold_slop_px)
        self._on_brush_size_changed(self.brush_slider.value())
        layout.addWidget(self.canvas, stretch=1)

        info_row = QHBoxLayout()
        self.status_label = QLabel(_STATUS_TEXT[IsolationTool.CROP])
        self.status_label.setProperty("role", "statusbar")
        self.status_label.setWordWrap(True)
        self.crop_readout = QLabel("")
        self.crop_readout.setProperty("role", "statusbar")
        info_row.addWidget(self.status_label, stretch=1)
        info_row.addWidget(self.crop_readout)
        layout.addLayout(info_row)

        hint = QLabel(_NAVIGATION_HINT)
        hint.setProperty("secondary", "true")
        layout.addWidget(hint)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        save_button = self.buttons.button(QDialogButtonBox.StandardButton.Save)
        save_button.setText("Apply && Save")
        save_button.setProperty("variant", "primary")
        self.reset_button = self.buttons.addButton("Reset selection", QDialogButtonBox.ButtonRole.ResetRole)
        layout.addWidget(self.buttons)

        for button, tool in self._tool_buttons:
            button.clicked.connect(lambda _checked=False, t=tool: self._select_tool(t))
        self.canvas.subjectRequested.connect(self._on_subject_requested)
        self.canvas.hintPainted.connect(self._on_hint_painted)
        self.canvas.outlineEdited.connect(self._on_outline_edited)
        self.canvas.overlayChanged.connect(self._sync_placement_controls)
        self.canvas.eraseSelectionChanged.connect(self._on_erase_selection_changed)
        self.canvas.eraseRegionsChanged.connect(self._update_option_states)
        self.canvas.cropRectAdjusting.connect(self._show_crop_readout)
        self.canvas.cropRectChanged.connect(self._on_crop_changed)
        self.canvas.zoomChanged.connect(self._show_zoom)
        self.zoom_in_button.clicked.connect(self.canvas.zoom_in)
        self.zoom_out_button.clicked.connect(self.canvas.zoom_out)
        self.fit_button.clicked.connect(self.canvas.fit_to_view)
        self.actual_button.clicked.connect(self.canvas.zoom_actual)
        self.reset_button.clicked.connect(self._on_reset)
        self.buttons.accepted.connect(self._on_save_clicked)
        self.buttons.rejected.connect(self.reject)
        self._show_zoom(self.canvas.zoom)
        self._show_options(0)
        self._update_option_states()

    # -- building the option bars ---------------------------------------------------------
    def _build_lift_options(self) -> QWidget:
        bar = QWidget()
        row = QHBoxLayout(bar)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        self.method_combo = QComboBox()
        for label, value in _METHODS:
            self.method_combo.addItem(label, value)
        self.method_combo.setToolTip(
            "Automatic looks for a closed border line first (a die-cut outline), then falls back to colour. "
            "Border line and Colour force one method."
        )
        self.autodetect_button = QPushButton("Auto-detect artwork")
        self.autodetect_button.setToolTip("Lift the largest closed border on the page, without pressing on it.")
        self.keep_button = QPushButton("+ Keep")
        self.drop_button = QPushButton("− Drop")
        self.edit_button = QPushButton("Edit outline")
        for button in (self.keep_button, self.drop_button, self.edit_button):
            button.setCheckable(True)
            button.setProperty("variant", "segment")
        self.edit_button.setToolTip(
            "Reshape the outline by hand. Drag a corner to move it; drag an edge to push it out (taking in what the "
            "lift left) or in (dropping what it took); double-click an edge to add a corner; right-click a corner to "
            "remove it; Ctrl+Z undoes. Do this after Keep / Drop, which lift again."
        )
        self.keep_button.setToolTip("Paint over a part of the artwork the outline missed.")
        self.drop_button.setToolTip("Paint over something the outline wrongly took.")
        self.brush_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_slider.setRange(1, 100)
        self.brush_slider.setValue(12)
        self.brush_slider.setMaximumWidth(120)
        self.brush_slider.setToolTip("Brush size")
        self.save_cutout_button = QPushButton("Save cutout…")
        self.save_cutout_button.setToolTip("Save the lifted artwork as a PNG with a transparent background.")

        for widget in (QLabel("Method"), self.method_combo, self.autodetect_button, self.keep_button, self.drop_button,
                       QLabel("Brush"), self.brush_slider, self.edit_button):
            row.addWidget(widget)
        row.addStretch(1)
        row.addWidget(self.save_cutout_button)

        self.autodetect_button.clicked.connect(lambda: self._run_lift(None))
        self.keep_button.toggled.connect(lambda on: self._set_brush("keep", on))
        self.drop_button.toggled.connect(lambda on: self._set_brush("drop", on))
        self.edit_button.toggled.connect(lambda on: self._set_brush("edit", on))
        self.brush_slider.valueChanged.connect(self._on_brush_size_changed)
        self.save_cutout_button.clicked.connect(self._save_cutout)
        return bar

    def _build_reference_banner(self) -> QWidget:
        """A strip above the page offering the other file's lifted artwork, so the way to place it is in plain sight."""
        self.reference_banner = QFrame()
        self.reference_banner.setProperty("role", "banner")
        row = QHBoxLayout(self.reference_banner)
        row.setContentsMargins(12, 8, 12, 8)
        row.setSpacing(12)
        self.reference_label = QLabel()
        self.reference_label.setWordWrap(True)
        self.reference_button = QPushButton("Place it on this file")
        self.reference_button.setProperty("variant", "primary")
        row.addWidget(self.reference_label, stretch=1)
        row.addWidget(self.reference_button)
        self.reference_button.clicked.connect(self._place_reference_from_banner)
        return self.reference_banner

    def _default_banner_text(self) -> str:
        name = self._reference.source_name if self._reference else self._partner_title
        return (
            f"The artwork lifted from {name} is ready. Lay it over this file to trim this file to the same "
            "area — the way to compare a scan or photo that has no clear border of its own."
        )

    def _build_place_options(self) -> QWidget:
        panel = QWidget()
        column = QVBoxLayout(panel)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)

        # Says, in words, whether there is anything to place -- a greyed-out button says nothing.
        self.place_hint_label = QLabel()
        self.place_hint_label.setWordWrap(True)
        self.place_hint_label.setProperty("secondary", "true")
        column.addWidget(self.place_hint_label)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        column.addLayout(row)

        name = self._reference.source_name if self._reference else self._partner_title
        self.use_reference_button = QPushButton(f"Use artwork lifted from {name}")
        self.use_reference_button.setEnabled(self._reference is not None)
        self.use_reference_button.setToolTip(
            "Lay the artwork lifted from the other file over this one."
            if self._reference is not None
            else f"Nothing has been lifted from {self._partner_title} yet."
        )
        self.open_cutout_button = QPushButton("Open cutout…")
        self.open_cutout_button.setToolTip("Use a cutout saved earlier (a PNG with a transparent background).")
        self.autofit_button = QPushButton("Auto-fit")
        self.autofit_button.setToolTip("Snap the outline onto the artwork by matching what is printed on it.")
        self.rotation_spin = QDoubleSpinBox()
        self.rotation_spin.setRange(-180.0, 180.0)
        self.rotation_spin.setSingleStep(0.1)
        self.rotation_spin.setDecimals(2)
        self.rotation_spin.setSuffix("°")
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(2.0, 800.0)
        self.scale_spin.setSingleStep(0.5)
        self.scale_spin.setDecimals(1)
        self.scale_spin.setSuffix("%")
        self.scale_spin.setValue(100.0)
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(10, 100)
        self.opacity_slider.setValue(60)
        self.opacity_slider.setMaximumWidth(110)
        self.opacity_slider.setToolTip("How see-through the outline is")
        self.trim_button = QPushButton("Trim to outline")
        self.trim_button.setProperty("variant", "primary")

        for widget in (self.use_reference_button, self.open_cutout_button, self.autofit_button, QLabel("Rotate"),
                       self.rotation_spin, QLabel("Scale"), self.scale_spin, QLabel("Opacity"), self.opacity_slider):
            row.addWidget(widget)
        row.addStretch(1)
        row.addWidget(self.trim_button)

        self.use_reference_button.clicked.connect(self._use_reference)
        self.open_cutout_button.clicked.connect(self._open_cutout)
        self.autofit_button.clicked.connect(self._autofit)
        self.rotation_spin.valueChanged.connect(self._apply_placement_controls)
        self.scale_spin.valueChanged.connect(self._apply_placement_controls)
        self.opacity_slider.valueChanged.connect(lambda value: self.canvas.set_overlay_opacity(value / 100))
        self.trim_button.clicked.connect(self._trim_to_outline)
        return panel

    def _build_erase_options(self) -> QWidget:
        bar = QWidget()
        row = QHBoxLayout(bar)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        self.erase_apply_button = QPushButton("Erase selected area")
        self.erase_apply_button.setProperty("variant", "primary")
        self.erase_apply_button.setToolTip("Leave the selected rectangle out of the comparison (Enter).")
        self.erase_undo_button = QPushButton("Undo last")
        self.erase_undo_button.setToolTip("Bring back the area erased last.")
        self.erase_clear_button = QPushButton("Clear all")
        self.erase_clear_button.setToolTip("Bring back every erased area.")
        self.erase_count_label = QLabel("")
        self.erase_count_label.setProperty("secondary", "true")

        for widget in (self.erase_apply_button, self.erase_undo_button, self.erase_clear_button):
            row.addWidget(widget)
        row.addWidget(self.erase_count_label)
        row.addStretch(1)

        self.erase_apply_button.clicked.connect(self._apply_erase_selection)
        self.erase_undo_button.clicked.connect(lambda: self.canvas.undo_erase())
        self.erase_clear_button.clicked.connect(lambda: self.canvas.clear_erase_regions())
        return bar

    def _apply_erase_selection(self) -> None:
        self.canvas.apply_erase_selection()

    def _size_to_screen(self) -> None:
        """Open large enough that the whole page can be seen, whatever the screen."""
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            self.resize(1000, 750)
            return
        available = screen.availableGeometry()
        self.resize(
            max(900, int(available.width() * _SCREEN_SHARE)), max(650, int(available.height() * _SCREEN_SHARE))
        )

    # -- small displays -----------------------------------------------------------------------
    def _show_zoom(self, zoom: float) -> None:
        self.zoom_label.setText(f"{round(zoom * 100)}%")

    def _say(self, text: str) -> None:
        self.status_label.setText(text)

    def _show_crop_readout(self, rect: NormalizedRect | None) -> None:
        """Live size of the selection, so it can be judged without saving."""
        size = self.canvas.crop_size_px()
        if rect is None or size is None:
            self.crop_readout.setText("")
            return
        self.crop_readout.setText(self._size_text("Crop", *size))

    def _size_text(self, label: str, width_px: int, height_px: int) -> str:
        page_height, page_width = self._rendered.shape[:2]
        share = 100 * width_px * height_px / (page_width * page_height)
        text = f"{label} {width_px:,} × {height_px:,} px"
        if self._artwork.kind == "pdf":  # a PDF is rendered at a known DPI, so real size is known
            to_mm = _MM_PER_INCH / WORKING_DPI
            text += f" ({width_px * to_mm:.1f} × {height_px * to_mm:.1f} mm)"
        return f"{text} · {share:.0f}% of the page"

    def _show_outline_readout(self, polygon: NormalizedPolygon | None) -> None:
        if polygon is None:
            self.crop_readout.setText("")
            return
        height, width = self._rendered.shape[:2]
        points = np.array([(x * width, y * height) for x, y in polygon.points], np.float32)
        x0, y0 = points.min(axis=0)
        x1, y1 = points.max(axis=0)
        share = 100 * cv2.contourArea(points.reshape(-1, 1, 2)) / (width * height)
        text = self._size_text("Outline", round(x1 - x0), round(y1 - y0))
        self.crop_readout.setText(text.rsplit(" · ", 1)[0] + f" · {share:.0f}% of the page")

    def _update_option_states(self) -> None:
        self.save_cutout_button.setEnabled(self._subject is not None)
        has_overlay = self.canvas.has_overlay
        for widget in (self.autofit_button, self.rotation_spin, self.scale_spin, self.opacity_slider, self.trim_button):
            widget.setEnabled(has_overlay)

        self.erase_apply_button.setEnabled(self.canvas.erase_selection() is not None)
        applied = len(self.canvas.erase_regions())
        self.erase_undo_button.setEnabled(applied > 0)
        self.erase_clear_button.setEnabled(applied > 0)
        self.erase_count_label.setText(f"{applied} area{'s' if applied != 1 else ''} erased" if applied else "")

        if self._reference is None:
            self.place_hint_label.setText(
                f"Nothing has been lifted from {self._partner_title} yet. Open Edit Artwork Area on it, use Lift Subject "
                "and Apply & Save, then come back here — or open a cutout saved earlier."
            )
        else:
            self.place_hint_label.setText(
                f"The artwork lifted from {self._reference.source_name} can be laid over this file: press the first "
                "button, and Auto-fit matches what is printed."
            )
        show_banner = self._reference is not None and not has_overlay and (self._outline is None or self._banner_hint is not None)
        self.reference_banner.setVisible(show_banner)
        if show_banner:
            self.reference_label.setText(self._banner_hint or self._default_banner_text())

    # -- choosing a tool -------------------------------------------------------------------------
    def _select_tool(self, tool: IsolationTool) -> None:
        self.canvas.set_tool(tool)
        for button, button_tool in self._tool_buttons:
            button.setChecked(button_tool == tool)
        self._show_options({IsolationTool.LIFT_SUBJECT: 1, IsolationTool.PLACE_OUTLINE: 2, IsolationTool.ERASE: 3}.get(tool, 0))
        if tool != IsolationTool.LIFT_SUBJECT:
            self._set_brush(None, True)
        self._say(_STATUS_TEXT[tool])
        if tool == IsolationTool.PLACE_OUTLINE and self._reference is not None and self._cutout is None:
            self._use_reference()  # there is something to place: place it, rather than wait for a second click

    def _show_options(self, index: int) -> None:
        """Show one tool's options, and give the page all the height the other tools' taller options would leave blank."""
        self.options.setCurrentIndex(index)
        for position in range(self.options.count()):
            page = self.options.widget(position)
            page.setSizePolicy(
                QSizePolicy.Policy.Preferred,
                QSizePolicy.Policy.Preferred if position == index else QSizePolicy.Policy.Ignored,
            )
        self.options.updateGeometry()

    def _on_erase_selection_changed(self, rect: NormalizedRect | None) -> None:
        self._update_option_states()
        size = self.canvas.erase_selection_size_px()
        if rect is None or size is None:
            self.crop_readout.setText("")
            return
        self.crop_readout.setText(self._size_text("Erase", *size))

    def _on_crop_changed(self, rect: NormalizedRect | None) -> None:
        """A crop and an outline are two ways of saying the same thing: the last one drawn is the one used."""
        self._show_crop_readout(rect)
        if rect is not None and self._outline is not None:
            self._clear_outline()

    # -- lifting the artwork --------------------------------------------------------------------
    def _set_brush(self, mode: str | None, on: bool) -> None:
        """Keep, Drop and Edit outline are exclusive; with none on, pressing and holding lifts.

        `mode=None` switches all off (used when another tool is chosen)."""
        buttons = {"keep": self.keep_button, "drop": self.drop_button, "edit": self.edit_button}
        for name, button in buttons.items():
            if mode is None or (on and name != mode):
                button.blockSignals(True)
                button.setChecked(False)
                button.blockSignals(False)
        self.canvas.set_brush(self._current_brush_mode())
        self.canvas.set_outline_editing(self.edit_button.isChecked())

    def _current_brush_mode(self) -> str | None:
        if self.keep_button.isChecked():
            return "keep"
        if self.drop_button.isChecked():
            return "drop"
        return None

    def _on_outline_edited(self, polygon: NormalizedPolygon) -> None:
        """The outline was reshaped by hand: it is the outline now, and what would be lifted for the other file."""
        if self._outline is None:
            return
        self._outline = (polygon, self._outline[1])
        self._show_outline_readout(polygon)
        if self._subject is not None:
            height, width = self._rendered.shape[:2]
            corners = np.array([(x * width, y * height) for x, y in polygon.points], np.float32)
            mask = np.zeros((height, width), np.uint8)
            cv2.fillPoly(mask, [np.round(corners).astype(np.int32)], 1)
            self._subject = replace(self._subject, mask=mask.astype(bool), polygon=[(float(x), float(y)) for x, y in corners])
        self._update_option_states()
        self._say("Outline reshaped by hand. Apply & Save uses it as it is now.")

    def _on_brush_size_changed(self, value: int) -> None:
        height, width = self._rendered.shape[:2]
        self.canvas.set_brush(self.canvas.brush_mode, max(2.0, value * _BRUSH_UNIT * max(height, width)))

    def _on_subject_requested(self, x_norm: float, y_norm: float) -> None:
        height, width = self._rendered.shape[:2]
        self._hints = []  # a new press is a new subject
        self.canvas.clear_hint_strokes()
        self._run_lift((x_norm * width, y_norm * height))

    def _run_lift(self, seed: tuple[float, float] | None) -> None:
        """Lift the artwork at `seed` (or the largest closed border, with none), applying any corrections."""
        is_pdf = self._artwork.kind == "pdf"
        QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            subject = lift_subject(
                self._rendered,
                seed,
                self._config.subject,
                hints=self._hints,
                page=self._artwork.pdf_page(0) if is_pdf else None,
                dpi=WORKING_DPI if is_pdf else None,
                method=self.method_combo.currentData(),
            )
        except NoSubjectFoundError as exc:
            self._say(str(exc))
            return
        finally:
            QGuiApplication.restoreOverrideCursor()

        self._subject, self._subject_seed = subject, seed
        self._banner_hint = self._partial_lift_hint(subject)
        self._set_outline(subject.normalized_polygon(), "lifted", subject.edge_inset_px)
        how = "inside its closed border line" if subject.source == "border" else "by colour, from its background"
        follow_up = ""
        if self._reference is None:
            follow_up = f" Then open Edit Artwork Area on {self._partner_title} and place this artwork over it."
            if subject.source == "segmented":
                follow_up += (
                    " On a scan the paler parts (flaps the colour of the paper) can be missed: if "
                    f"{self._partner_title} has a clear border, lift it there first and use Place Subject here instead."
                )
        self._say(
            f"Lifted the artwork {how}: {100 * subject.area_fraction:.0f}% of the page. Press and hold elsewhere to lift "
            f"something else, or use Keep / Drop to correct it, then Apply & Save.{follow_up}"
        )

    def _partial_lift_hint(self, subject: LiftedSubject) -> str | None:
        """Advice when a lift by colour caught much less than the artwork lifted from the other file.

        A scan of a die-cut sheet is the usual cause: its flaps are the colour of the paper they lie on, so
        colour finds only the printed panels. Laying the other file's outline over it does not depend on that.
        """
        if self._reference is None or subject.source != "segmented":
            return None
        expected = float(self._reference.cutout.mask.sum())
        if expected <= 0 or float(subject.mask.sum()) >= _SMALLER_THAN_REFERENCE * expected:
            return None
        share = 100 * float(subject.mask.sum()) / expected
        return (
            f"This lift is only {share:.0f}% of the artwork lifted from {self._reference.source_name}: on a scan the outline "
            "is often too faint to follow. Lay that artwork over this file instead — Auto-fit matches what is printed."
        )

    def _on_hint_painted(self, points: list, radius_px: float, keep: bool) -> None:
        if self._subject is None:
            self.canvas.clear_hint_strokes()
            self._say("Press and hold on the artwork first, then use Keep / Drop to correct the outline.")
            return
        self._hints.append(HintStroke([(float(x), float(y)) for x, y in points], float(radius_px), keep))
        self._run_lift(self._subject_seed)
        self.canvas.clear_hint_strokes()  # it is part of the outline now

    def _set_outline(self, polygon: NormalizedPolygon, origin: str, inset_px: float = 0.0) -> None:
        self._outline = (polygon, origin)
        self._outline_inset_px = inset_px
        self.canvas.clear_crop()
        self._show_crop_readout(None)
        self.canvas.show_polygon(polygon)
        self._show_outline_readout(polygon)
        self._update_option_states()

    def _clear_outline(self) -> None:
        self._outline = None
        self._outline_inset_px = 0.0
        self._banner_hint = None
        self.canvas.clear_polygon()
        self._show_outline_readout(None)
        self._update_option_states()

    def _save_cutout(self) -> None:
        if self._subject is None:
            return
        suggested = str(Path(self._artwork.path).with_name(f"{Path(self._artwork.path).stem}_cutout.png"))
        chosen, _ = QFileDialog.getSaveFileName(self, "Save cutout", suggested, "PNG image (*.png)")
        if not chosen:
            return
        saved = Cutout.from_subject(self._rendered, self._subject).save_png(chosen)
        self._say(f"Saved the cutout to {saved}.")

    # -- placing another file's outline ----------------------------------------------------------
    def _use_reference(self) -> None:
        if self._reference is not None:
            self._set_cutout(self._reference.cutout, f"the artwork lifted from {self._reference.source_name}")

    def _place_reference_from_banner(self) -> None:
        """The banner's button: switch to Place Subject with the other file's artwork laid over this one."""
        if self._reference is None:
            return
        self._banner_hint = None
        if self._outline is not None and self._outline[1] == "lifted":
            # The partial lift it was offered instead of: forget it, so it is not saved as this file's artwork.
            self._subject, self._subject_seed, self._hints = None, None, []
            self.canvas.clear_hint_strokes()
            self._clear_outline()
        self._select_tool(IsolationTool.PLACE_OUTLINE)

    def _open_cutout(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(self, "Open cutout", "", "PNG image (*.png)")
        if not chosen:
            return
        try:
            cutout = Cutout.load_png(chosen, self._config.subject)
        except CutoutError as exc:
            self._say(str(exc))
            return
        self._set_cutout(cutout, Path(chosen).name)

    def _set_cutout(self, cutout: Cutout, name: str) -> None:
        self._cutout = cutout
        self._banner_hint = None
        self.canvas.set_overlay(cutout)
        self.canvas.set_overlay_opacity(self.opacity_slider.value() / 100)
        self._update_option_states()
        self._autofit(announce=f"Placed {name}. ")

    def _autofit(self, announce: str = "") -> None:
        if self._cutout is None:
            return
        QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            fit = fit_placement(self._cutout, self._rendered, self._config.alignment)
        except AlignmentError:
            self._say(announce + "The artwork could not be matched automatically. Drag the outline into place, "
                      "or set its rotation and scale, then press Trim to outline.")
            return
        finally:
            QGuiApplication.restoreOverrideCursor()

        self.canvas.set_overlay_placement(fit.placement)
        confidence = " Low confidence: check it by eye before trimming." if fit.alignment.low_confidence else ""
        self._say(
            f"{announce}Fitted by matching the artwork (residual {fit.alignment.residual_error_px:.2f} px). "
            f"Adjust it if needed, then press Trim to outline.{confidence}"
        )

    def _sync_placement_controls(self) -> None:
        placement = self.canvas.overlay_placement()
        self._update_option_states()
        if placement is None:
            return
        for spin, value in ((self.rotation_spin, placement.rotation_deg), (self.scale_spin, placement.scale * 100.0)):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)

    def _apply_placement_controls(self, _value: float = 0.0) -> None:
        placement = self.canvas.overlay_placement()
        if placement is not None:
            self.canvas.set_overlay_placement(
                Placement(placement.center, self.rotation_spin.value(), self.scale_spin.value() / 100.0)
            )

    def _trim_to_outline(self) -> None:
        points = self.canvas.overlay_polygon_px()
        if not points:
            return
        height, width = self._rendered.shape[:2]
        polygon = NormalizedPolygon(points=[(x / width, y / height) for x, y in points])
        placement = self.canvas.overlay_placement()
        scale = placement.scale if placement is not None else 1.0
        inset = (self._cutout.edge_inset_px * scale if self._cutout is not None else 0.0) + _PLACED_MARGIN_PX
        self._set_outline(polygon, "placed", inset)
        self._say("Trimmed to the placed outline: only the area inside it will be compared. Apply & Save to keep it.")

    # -- resetting and saving ------------------------------------------------------------------------
    def _on_reset(self) -> None:
        """Clear whatever the current tool has drawn, to start that selection again."""
        tool = self.canvas.tool
        if tool == IsolationTool.CROP:
            self.canvas.clear_crop()
            self._show_crop_readout(None)
        elif tool == IsolationTool.LIFT_SUBJECT:
            self._subject, self._subject_seed, self._hints = None, None, []
            self.canvas.clear_hint_strokes()
            if self._outline is not None and self._outline[1] == "lifted":
                self._clear_outline()
            self._update_option_states()
        elif tool == IsolationTool.PLACE_OUTLINE:
            self.canvas.clear_overlay()
            self._cutout = None
            if self._outline is not None and self._outline[1] == "placed":
                self._clear_outline()
            self._update_option_states()
        else:
            self.canvas.clear_erase_selection()
            self.canvas.clear_erase_regions()
            self.crop_readout.setText("")
        self._say(_STATUS_TEXT[tool])

    def _on_save_clicked(self) -> None:
        """Save button: confirm what discards the rest of the page, and catch a placed outline left untrimmed."""
        if self.canvas.has_overlay and (self._outline is None or self._outline[1] != "placed"):
            answer = QMessageBox.question(
                self,
                "Trim to the placed outline?",
                "You placed an outline over this file but did not trim to it. Trim this file to it before saving?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Yes,
            )
            if answer == QMessageBox.StandardButton.Cancel:
                return
            if answer == QMessageBox.StandardButton.Yes:
                self._trim_to_outline()

        if self.canvas.erase_selection() is not None:
            answer = QMessageBox.question(
                self,
                "Erase the selected area?",
                "You selected an area to erase but did not press Erase selected area. Leave it out of the comparison?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Yes,
            )
            if answer == QMessageBox.StandardButton.Cancel:
                return
            if answer == QMessageBox.StandardButton.Yes:
                self.canvas.apply_erase_selection()

        if self._outline is not None and self._outline[1] == "placed":
            question = "Only the area inside the placed outline will be compared; the rest of the page is ignored."
        elif self._outline is None and self.canvas.crop_rect() is not None:
            question = f"{self.crop_readout.text()}\n\nOnly this area will be compared; the rest of the page is ignored."
        else:
            question = None
        if question is not None:
            answer = QMessageBox.question(
                self,
                "Apply this area?",
                question,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self._on_save()

    def _on_save(self) -> None:
        boundary: NormalizedRect | NormalizedPolygon | None
        rotation_deg = 0.0
        origin = "manual"
        if self._outline is not None:
            boundary, origin = self._outline
        elif self.canvas.crop_rect() is not None:
            boundary = self.canvas.crop_rect()
            cropped = crop_tool.crop_to_rect(self._rendered, boundary)
            _, rotation_deg = crop_tool.auto_rotate_align(cropped)
        else:
            boundary = None
        inset_norm = 0.0
        if isinstance(boundary, NormalizedPolygon) and self._outline is not None:
            inset_norm = self._outline_inset_px / max(self._rendered.shape[:2])
        self._result = IsolationGeometry(
            boundary=boundary,
            erase_regions=self.canvas.erase_regions(),
            rotation_deg=rotation_deg,
            boundary_origin=origin,
            edge_inset_norm=inset_norm,
        )
        # A lifted subject stays available to lay over the other file, whether or not it is this file's area.
        self._result_cutout = Cutout.from_subject(self._rendered, self._subject) if self._subject is not None else None
        self.accept()

    def result_geometry(self) -> Optional[IsolationGeometry]:
        return self._result

    def result_cutout(self) -> Optional[Cutout]:
        """The artwork lifted in this dialog, if any, for the other file to lay over itself."""
        return self._result_cutout

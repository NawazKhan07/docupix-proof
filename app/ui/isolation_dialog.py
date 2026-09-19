"""Dialog wrapping InteractiveCanvas so one file's artwork area can be chosen from the upload
panel: crop it, lift it from its background, lay another file's outline over it, or erase parts.

**Lift Subject** is how the artwork is separated from everything else on the sheet. Press and
hold on the artwork and only what is inside its border is lifted -- for a packaging sheet, the
carton inside its die-cut line, and none of the title, colour legend, crop marks or notes
around it. The outline can be corrected with Keep / Drop strokes, saved as a transparent PNG
cutout, and used as the area to compare.

**Place Outline** is for a file with no clear border of its own (a photograph, a plain scan):
the other file's lifted outline is laid over it, snapped into place by matching the artwork,
adjusted by hand if need be, and the file is trimmed to it.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
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
        "Lay the other file's outline over this one: Auto-fit matches the artwork, and you can drag it into "
        "place. Then press Trim to outline."
    ),
    IsolationTool.ERASE: "Drag to paint regions to exclude from comparison.",
}
_NAVIGATION_HINT = "Pinch or Ctrl + wheel to zoom  ·  two-finger scroll, Space + drag or middle-drag to pan"
_SCREEN_SHARE = 0.85  # how much of the screen the dialog may take, so the whole page can be seen
_MM_PER_INCH = 25.4
_METHODS = (("Automatic", "auto"), ("Border line", "border"), ("Colour", "colour"))
_BRUSH_UNIT = 1 / 1000  # the brush slider is in thousandths of the page's long edge


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
    ):
        super().__init__(parent)
        self.setWindowTitle(f"Isolate Artwork — {title}")
        self._size_to_screen()
        self._artwork = artwork
        self._config = config or AppConfig()
        self._rendered = artwork.render()
        self._reference = reference
        self._result: Optional[IsolationGeometry] = None
        self._result_cutout: Optional[Cutout] = None

        # What has been lifted, and the corrections made to it.
        self._subject: LiftedSubject | None = None
        self._subject_seed: tuple[float, float] | None = None
        self._hints: list[HintStroke] = []
        # The region that will be compared: a lifted outline, or a placed one.
        self._outline: tuple[NormalizedPolygon, str] | None = None
        # The cutout laid over the page.
        self._cutout: Cutout | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        tool_row = QHBoxLayout()
        tool_row.setSpacing(6)
        self.crop_button = QPushButton("Crop")
        self.lift_button = QPushButton("Lift Subject")
        self.place_button = QPushButton("Place Outline")
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

        # The options of the current tool.
        self.options = QStackedWidget()
        self.options.addWidget(QWidget())  # crop and erase have none
        self.options.addWidget(self._build_lift_options())
        self.options.addWidget(self._build_place_options())
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
        self.canvas.overlayChanged.connect(self._sync_placement_controls)
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
        for button in (self.keep_button, self.drop_button):
            button.setCheckable(True)
            button.setProperty("variant", "segment")
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
                       QLabel("Brush"), self.brush_slider):
            row.addWidget(widget)
        row.addStretch(1)
        row.addWidget(self.save_cutout_button)

        self.autodetect_button.clicked.connect(lambda: self._run_lift(None))
        self.keep_button.toggled.connect(lambda on: self._set_brush("keep", on))
        self.drop_button.toggled.connect(lambda on: self._set_brush("drop", on))
        self.brush_slider.valueChanged.connect(self._on_brush_size_changed)
        self.save_cutout_button.clicked.connect(self._save_cutout)
        return bar

    def _build_place_options(self) -> QWidget:
        bar = QWidget()
        row = QHBoxLayout(bar)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        name = self._reference.source_name if self._reference else "the other file"
        self.use_reference_button = QPushButton(f"Use outline from {name}")
        self.use_reference_button.setEnabled(self._reference is not None)
        self.use_reference_button.setToolTip("Lift the artwork in the other file first (Lift Subject), then use it here.")
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
        return bar

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

    # -- choosing a tool -------------------------------------------------------------------------
    def _select_tool(self, tool: IsolationTool) -> None:
        self.canvas.set_tool(tool)
        for button, button_tool in self._tool_buttons:
            button.setChecked(button_tool == tool)
        self.options.setCurrentIndex({IsolationTool.LIFT_SUBJECT: 1, IsolationTool.PLACE_OUTLINE: 2}.get(tool, 0))
        if tool != IsolationTool.LIFT_SUBJECT:
            self._set_brush(None, True)
        self._say(_STATUS_TEXT[tool])

    def _on_crop_changed(self, rect: NormalizedRect | None) -> None:
        """A crop and an outline are two ways of saying the same thing: the last one drawn is the one used."""
        self._show_crop_readout(rect)
        if rect is not None and self._outline is not None:
            self._clear_outline()

    # -- lifting the artwork --------------------------------------------------------------------
    def _set_brush(self, mode: str | None, on: bool) -> None:
        """Keep and Drop are exclusive brushes; with neither on, pressing and holding lifts.

        `mode=None` switches both off (used when another tool is chosen)."""
        if mode is None:
            for button in (self.keep_button, self.drop_button):
                button.blockSignals(True)
                button.setChecked(False)
                button.blockSignals(False)
        elif on:
            other = self.drop_button if mode == "keep" else self.keep_button
            other.blockSignals(True)
            other.setChecked(False)
            other.blockSignals(False)
        self.canvas.set_brush(self._current_brush_mode())

    def _current_brush_mode(self) -> str | None:
        if self.keep_button.isChecked():
            return "keep"
        if self.drop_button.isChecked():
            return "drop"
        return None

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
        self._set_outline(subject.normalized_polygon(), "lifted")
        how = "inside its closed border line" if subject.source == "border" else "by colour, from its background"
        self._say(
            f"Lifted the artwork {how}: {100 * subject.area_fraction:.0f}% of the page. Press and hold elsewhere to lift "
            "something else, or use Keep / Drop to correct it, then Apply & Save."
        )

    def _on_hint_painted(self, points: list, radius_px: float, keep: bool) -> None:
        if self._subject is None:
            self.canvas.clear_hint_strokes()
            self._say("Press and hold on the artwork first, then use Keep / Drop to correct the outline.")
            return
        self._hints.append(HintStroke([(float(x), float(y)) for x, y in points], float(radius_px), keep))
        self._run_lift(self._subject_seed)
        self.canvas.clear_hint_strokes()  # it is part of the outline now

    def _set_outline(self, polygon: NormalizedPolygon, origin: str) -> None:
        self._outline = (polygon, origin)
        self.canvas.clear_crop()
        self._show_crop_readout(None)
        self.canvas.show_polygon(polygon)
        self._show_outline_readout(polygon)
        self._update_option_states()

    def _clear_outline(self) -> None:
        self._outline = None
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
            self._set_cutout(self._reference.cutout, f"the outline from {self._reference.source_name}")

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
        self._set_outline(polygon, "placed")
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
            self.canvas.clear_erase_strokes()
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
        self._result = IsolationGeometry(
            boundary=boundary,
            erase_regions=self.canvas.erase_strokes(),
            rotation_deg=rotation_deg,
            boundary_origin=origin,
        )
        # A lifted subject stays available to lay over the other file, whether or not it is this file's area.
        self._result_cutout = Cutout.from_subject(self._rendered, self._subject) if self._subject is not None else None
        self.accept()

    def result_geometry(self) -> Optional[IsolationGeometry]:
        return self._result

    def result_cutout(self) -> Optional[Cutout]:
        """The artwork lifted in this dialog, if any, for the other file to lay over itself."""
        return self._result_cutout

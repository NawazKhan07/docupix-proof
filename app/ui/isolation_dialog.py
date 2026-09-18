"""Dialog wrapping InteractiveCanvas so one file's crop / dieline-trace /
erase-mask selection can be made from the upload panel (Build Prompt Phase 2).

Not part of the Architecture Document's §10 module list -- added because the
canvas needs somewhere to live with tool-switch buttons and a Save/Cancel
flow; everything it does beyond hosting the canvas is thin UI glue.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from app.engine.file_loader import WORKING_DPI, LoadedArtwork
from app.engine.isolation import crop_tool, polygon_tracer
from app.engine.isolation.polygon_tracer import NoDielineFoundError
from app.models.isolation import (
    IsolationGeometry,
    IsolationTool,
    NormalizedPolygon,
    NormalizedRect,
)
from app.ui.interactive_canvas import InteractiveCanvas
from app.ui.theme import set_variant

_STATUS_TEXT = {
    IsolationTool.CROP: "Drag on the page to draw a crop. Drag its edges or corners to resize it, or its inside to move it.",
    IsolationTool.POLYGON_TRACE: "Click a point on the dieline to trace it.",
    IsolationTool.ERASE: "Drag to paint regions to exclude from comparison.",
}
_NAVIGATION_HINT = "Pinch or Ctrl + wheel to zoom  ·  two-finger scroll, Space + drag or middle-drag to pan"
_SCREEN_SHARE = 0.85  # how much of the screen the dialog may take, so the whole page can be seen
_MM_PER_INCH = 25.4


class IsolationDialog(QDialog):
    """Lets the user crop, click-to-trace a dieline, or freehand-erase on one file."""

    def __init__(self, artwork: LoadedArtwork, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Isolate Artwork — {title}")
        self._size_to_screen()
        self._artwork = artwork
        self._rendered = artwork.render()
        self._result: Optional[IsolationGeometry] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        tool_row = QHBoxLayout()
        tool_row.setSpacing(6)
        self.crop_button = QPushButton("Crop")
        self.trace_button = QPushButton("Trace Dieline")
        self.erase_button = QPushButton("Erase")
        for button in (self.crop_button, self.trace_button, self.erase_button):
            button.setCheckable(True)
            button.setProperty("variant", "segment")
        self.crop_button.setChecked(True)
        tool_row.addWidget(self.crop_button)
        tool_row.addWidget(self.trace_button)
        tool_row.addWidget(self.erase_button)
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

        self.canvas = InteractiveCanvas(self)
        self.canvas.set_image(self._rendered)
        layout.addWidget(self.canvas, stretch=1)

        info_row = QHBoxLayout()
        self.status_label = QLabel(_STATUS_TEXT[IsolationTool.CROP])
        self.status_label.setProperty("role", "statusbar")
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

        self.crop_button.clicked.connect(lambda: self._select_tool(IsolationTool.CROP))
        self.trace_button.clicked.connect(lambda: self._select_tool(IsolationTool.POLYGON_TRACE))
        self.erase_button.clicked.connect(lambda: self._select_tool(IsolationTool.ERASE))
        self.canvas.pointClicked.connect(self._on_point_clicked)
        self.canvas.cropRectAdjusting.connect(self._show_crop_readout)
        self.canvas.cropRectChanged.connect(self._show_crop_readout)
        self.canvas.zoomChanged.connect(self._show_zoom)
        self.zoom_in_button.clicked.connect(self.canvas.zoom_in)
        self.zoom_out_button.clicked.connect(self.canvas.zoom_out)
        self.fit_button.clicked.connect(self.canvas.fit_to_view)
        self.actual_button.clicked.connect(self.canvas.zoom_actual)
        self.reset_button.clicked.connect(self._on_reset)
        self.buttons.accepted.connect(self._on_save_clicked)
        self.buttons.rejected.connect(self.reject)
        self._show_zoom(self.canvas.zoom)

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

    def _show_zoom(self, zoom: float) -> None:
        self.zoom_label.setText(f"{round(zoom * 100)}%")

    def _show_crop_readout(self, rect: NormalizedRect | None) -> None:
        """Live size of the selection, so it can be judged without saving."""
        size = self.canvas.crop_size_px()
        if rect is None or size is None:
            self.crop_readout.setText("")
            return
        width_px, height_px = size
        page_height, page_width = self._rendered.shape[:2]
        share = 100 * width_px * height_px / (page_width * page_height)
        text = f"Crop {width_px:,} × {height_px:,} px"
        if self._artwork.kind == "pdf":  # a PDF is rendered at a known DPI, so real size is known
            to_mm = _MM_PER_INCH / WORKING_DPI
            text += f" ({width_px * to_mm:.1f} × {height_px * to_mm:.1f} mm)"
        self.crop_readout.setText(f"{text} · {share:.0f}% of the page")

    def _on_reset(self) -> None:
        """Clear whatever the current tool has drawn, to start that selection again."""
        tool = self.canvas.tool
        if tool == IsolationTool.CROP:
            self.canvas.clear_crop()
            self._show_crop_readout(None)
        elif tool == IsolationTool.POLYGON_TRACE:
            self.canvas.clear_polygon()
        else:
            self.canvas.clear_erase_strokes()
        self.status_label.setText(_STATUS_TEXT[tool])

    def _on_save_clicked(self) -> None:
        """Save button: confirm a crop first, since it discards the rest of the page."""
        if self.canvas.polygon() is None and self.canvas.crop_rect() is not None:
            answer = QMessageBox.question(
                self,
                "Apply this crop?",
                f"{self.crop_readout.text()}\n\nOnly this area will be compared; the rest of the page is ignored.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self._on_save()

    def _select_tool(self, tool: IsolationTool) -> None:
        self.canvas.set_tool(tool)
        for button, button_tool in (
            (self.crop_button, IsolationTool.CROP),
            (self.trace_button, IsolationTool.POLYGON_TRACE),
            (self.erase_button, IsolationTool.ERASE),
        ):
            button.setChecked(button_tool == tool)
        self.status_label.setText(_STATUS_TEXT[tool])

    def _on_point_clicked(self, x_norm: float, y_norm: float) -> None:
        height, width = self._rendered.shape[:2]
        try:
            if self._artwork.kind == "pdf":
                page = self._artwork.pdf_page(0)
                # The click lands in rendered-pixel space; convert back to
                # the PDF's own point-space, which is what get_drawings() uses.
                scale_x = page.rect.width / width
                scale_y = page.rect.height / height
                click_pdf = (x_norm * width * scale_x, y_norm * height * scale_y)
                traced = polygon_tracer.trace_vector_dieline(page, click_pdf)
                points_norm = [
                    (px / page.rect.width, py / page.rect.height) for px, py in traced.points
                ]
            else:
                click_px = (x_norm * width, y_norm * height)
                traced = polygon_tracer.trace_raster_dieline(self._rendered, click_px)
                points_norm = [(px / width, py / height) for px, py in traced.points]
        except NoDielineFoundError as exc:
            QMessageBox.information(self, "No dieline found", str(exc))
            return
        self.canvas.show_polygon(NormalizedPolygon(points=points_norm))
        self.status_label.setText(f"Traced a {len(points_norm)}-point dieline polygon. Save to keep it.")

    def _on_save(self) -> None:
        boundary: NormalizedRect | NormalizedPolygon | None
        rotation_deg = 0.0
        if self.canvas.polygon() is not None:
            boundary = self.canvas.polygon()
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
        )
        self.accept()

    def result_geometry(self) -> Optional[IsolationGeometry]:
        return self._result

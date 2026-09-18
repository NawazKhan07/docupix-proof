"""Dialog wrapping InteractiveCanvas so one file's crop / dieline-trace /
erase-mask selection can be made from the upload panel (Build Prompt Phase 2).

Not part of the Architecture Document's §10 module list -- added because the
canvas needs somewhere to live with tool-switch buttons and a Save/Cancel
flow; everything it does beyond hosting the canvas is thin UI glue.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from app.engine.file_loader import LoadedArtwork
from app.engine.isolation import crop_tool, polygon_tracer
from app.engine.isolation.polygon_tracer import NoDielineFoundError
from app.models.isolation import (
    IsolationGeometry,
    IsolationTool,
    NormalizedPolygon,
    NormalizedRect,
)
from app.ui.interactive_canvas import InteractiveCanvas

_STATUS_TEXT = {
    IsolationTool.CROP: "Drag to draw a crop rectangle, then Save.",
    IsolationTool.POLYGON_TRACE: "Click a point on the dieline to trace it.",
    IsolationTool.ERASE: "Drag to paint regions to exclude from comparison.",
}


class IsolationDialog(QDialog):
    """Lets the user crop, click-to-trace a dieline, or freehand-erase on one file."""

    def __init__(self, artwork: LoadedArtwork, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Isolate Artwork — {title}")
        self.resize(800, 650)
        self._artwork = artwork
        self._rendered = artwork.render()
        self._result: Optional[IsolationGeometry] = None

        layout = QVBoxLayout(self)

        tool_row = QHBoxLayout()
        self.crop_button = QPushButton("Crop")
        self.trace_button = QPushButton("Trace Dieline")
        self.erase_button = QPushButton("Erase")
        for button in (self.crop_button, self.trace_button, self.erase_button):
            button.setCheckable(True)
        self.crop_button.setChecked(True)
        tool_row.addWidget(self.crop_button)
        tool_row.addWidget(self.trace_button)
        tool_row.addWidget(self.erase_button)
        layout.addLayout(tool_row)

        self.canvas = InteractiveCanvas(self)
        self.canvas.set_image(self._rendered)
        layout.addWidget(self.canvas, stretch=1)

        self.status_label = QLabel(_STATUS_TEXT[IsolationTool.CROP])
        layout.addWidget(self.status_label)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        layout.addWidget(self.buttons)

        self.crop_button.clicked.connect(lambda: self._select_tool(IsolationTool.CROP))
        self.trace_button.clicked.connect(lambda: self._select_tool(IsolationTool.POLYGON_TRACE))
        self.erase_button.clicked.connect(lambda: self._select_tool(IsolationTool.ERASE))
        self.canvas.pointClicked.connect(self._on_point_clicked)
        self.buttons.accepted.connect(self._on_save)
        self.buttons.rejected.connect(self.reject)

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

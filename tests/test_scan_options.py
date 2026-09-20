"""Scan vs Digital compares text; its graphics are an option the reviewer turns on."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QObject, Signal

from app.config import DetectionConfig
from app.models.job import ComparisonMode
from app.ui.main_window import MainWindow
from app.ui.mode_selector import ModeSelector


class _RecordingJobs(QObject):
    """Stands in for the job controller: keeps the request it was asked to run."""

    progressed = Signal(str)
    completed = Signal(object)
    failed = Signal(str, str)
    cancelled = Signal()
    busyChanged = Signal(bool)
    is_busy = False

    def __init__(self):
        super().__init__()
        self.requests = []

    def start(self, request):
        self.requests.append(request)
        return len(self.requests)

    def cancel(self):
        pass

    def warm_up(self):
        pass


def test_the_graphics_option_is_offered_only_for_scan_vs_digital(qapp):
    selector = ModeSelector()
    selector.show()
    assert not selector.scan_graphics.isVisible()

    selector.set_mode(ComparisonMode.SCAN_VS_DIGITAL)
    assert selector.scan_graphics.isVisible() and not selector.compare_scan_graphics

    selector.set_mode(ComparisonMode.AUTOMATIC)
    assert not selector.scan_graphics.isVisible()


def test_the_option_starts_off_because_a_scans_graphics_give_false_differences():
    assert DetectionConfig().scan_compare_graphics is False


def _pair(tmp_path: Path) -> tuple[Path, Path]:
    image = np.full((400, 600, 3), 255, dtype=np.uint8)
    cv2.putText(image, "PARACETAMOL 500mg", (70, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2)
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    Image.fromarray(image).save(a)
    Image.fromarray(image).save(b)
    return a, b


@pytest.mark.parametrize("ticked", [False, True])
def test_the_reviewers_choice_reaches_the_comparison(qapp, tmp_path, ticked):
    jobs = _RecordingJobs()
    window = MainWindow(jobs=jobs)
    a, b = _pair(tmp_path)
    window.upload_panel.original_slot.load_path(a)
    window.upload_panel.suspect_slot.load_path(b)
    window.mode_selector.set_mode(ComparisonMode.SCAN_VS_DIGITAL)
    window.mode_selector.scan_graphics.setChecked(ticked)

    window._on_compare()

    (request,) = jobs.requests
    assert request.mode is ComparisonMode.SCAN_VS_DIGITAL
    assert request.config.detection.scan_compare_graphics is ticked
    window.close()


def test_the_choice_does_not_change_other_modes(qapp, tmp_path):
    jobs = _RecordingJobs()
    window = MainWindow(jobs=jobs)
    a, b = _pair(tmp_path)
    window.upload_panel.original_slot.load_path(a)
    window.upload_panel.suspect_slot.load_path(b)
    window.mode_selector.set_mode(ComparisonMode.SCAN_VS_DIGITAL)
    window.mode_selector.scan_graphics.setChecked(True)
    window.mode_selector.set_mode(ComparisonMode.AUTOMATIC)

    window._on_compare()

    (request,) = jobs.requests
    assert request.mode is ComparisonMode.AUTOMATIC
    assert request.config.detection.scan_compare_graphics is False
    window.close()

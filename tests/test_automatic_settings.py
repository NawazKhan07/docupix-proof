"""Automatic mode's settings on the setup page: Food label, shown only for Automatic, carried into the comparison."""

from __future__ import annotations

from app.models.job import ComparisonMode
from test_leaflet_options import _loaded_window


def test_food_label_is_offered_for_automatic_only_and_reaches_the_comparison(qapp, tmp_path):
    window, jobs = _loaded_window(qapp, tmp_path, ComparisonMode.AUTOMATIC)
    window.show()
    settings = window.automatic_settings
    assert settings.isVisible() and not settings.food_label
    window._on_compare()
    assert jobs.requests[-1].config.detection.food_label is False  # off unless ticked

    settings.set_food_label(True)
    assert "Food label" in settings.label.text()
    jobs.is_busy = False
    window._on_compare()
    assert jobs.requests[-1].config.detection.food_label is True

    for mode in (ComparisonMode.SCAN_VS_DIGITAL, ComparisonMode.LEAFLET, ComparisonMode.SERIES):
        window.mode_selector.set_mode(mode)
        assert not settings.isVisible()
        jobs.is_busy = False
        window._on_compare()
        assert jobs.requests[-1].config.detection.food_label is False  # a food label is Automatic's alone
    window.mode_selector.set_mode(ComparisonMode.AUTOMATIC)
    assert settings.isVisible() and settings.food_label  # (kept while the window is open)
    window.close()

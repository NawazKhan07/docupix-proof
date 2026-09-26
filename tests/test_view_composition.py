from __future__ import annotations

import numpy as np

from app.models.diff import BoundingBox
from app.ui.composition import (
    ANIMATED_MODES,
    SINGLE_PANE_MODES,
    ViewMode,
    compose_flicker,
    compose_grayscale,
    compose_overlay,
)
from app.ui.pane_view import ImageStack


def _solid(colour, height: int = 60, width: int = 80) -> np.ndarray:
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = colour
    return image


def test_flicker_holds_unchanged_areas_still():
    """FR-8.2: only the flagged regions flicker, not the whole page."""
    original = _solid((200, 0, 0))
    result = compose_flicker(original, _solid((0, 0, 200)), [], phase=1)
    assert np.array_equal(result, original)


def test_flicker_alternates_the_two_files_inside_a_difference():
    original, suspect = _solid((200, 30, 40)), _solid((30, 160, 60))
    box = BoundingBox(10, 10, 30, 30)

    first = compose_flicker(original, suspect, [box], phase=0)
    second = compose_flicker(original, suspect, [box], phase=1)

    assert tuple(first[20, 20]) == (200, 30, 40)  # the original's colour...
    assert tuple(second[20, 20]) == (30, 160, 60)  # ...then the suspect's: it flickers
    assert np.array_equal(first[50:, 40:], original[50:, 40:])  # outside stays on the original
    assert np.array_equal(second[50:, 40:], original[50:, 40:])  # ...and does not flicker


def test_flicker_boxes_are_clipped_to_the_image():
    """A marker box straddling the edge must not crash or wrap around."""
    original, suspect = _solid((200, 30, 40)), _solid((30, 160, 60))
    result = compose_flicker(original, suspect, [BoundingBox(-20, -20, 500, 500)], phase=1)
    assert result.shape == original.shape


def test_overlay_blends_the_two_files_evenly():
    """FR-8.3: both files are visible at once."""
    blended = compose_overlay(_solid((200, 100, 0)), _solid((0, 100, 200)))
    assert np.all(np.abs(blended.astype(int) - np.array([100, 100, 100])) <= 1)


def test_overlay_of_identical_images_is_unchanged():
    image = _solid((90, 140, 200))
    assert np.array_equal(compose_overlay(image, image.copy()), image)


def test_grayscale_leaves_unchanged_areas_black_and_white():
    """FR-8.4: unchanged areas are rendered without colour."""
    original = _solid((200, 30, 40))
    result = compose_grayscale(original, original.copy(), [], phase=0)
    assert np.all(result[..., 0] == result[..., 1]) and np.all(result[..., 1] == result[..., 2])


def test_grayscale_keeps_colour_inside_a_difference_and_it_alternates():
    original, suspect = _solid((200, 30, 40)), _solid((30, 160, 60))
    box = BoundingBox(10, 10, 30, 30)

    first = compose_grayscale(original, suspect, [box], phase=0)
    second = compose_grayscale(original, suspect, [box], phase=1)

    assert tuple(first[20, 20]) == (200, 30, 40)  # the original's colour...
    assert tuple(second[20, 20]) == (30, 160, 60)  # ...then the suspect's: it flickers
    assert first[50, 70, 0] == first[50, 70, 1] == first[50, 70, 2]  # outside stays gray
    assert np.array_equal(first[50:, 40:], second[50:, 40:])  # and outside does not flicker


def test_grayscale_boxes_are_clipped_to_the_image():
    """A marker box straddling the edge must not crash or wrap around."""
    original, suspect = _solid((200, 30, 40)), _solid((30, 160, 60))
    result = compose_grayscale(original, suspect, [BoundingBox(-20, -20, 500, 500)], phase=0)
    assert result.shape == original.shape
    assert tuple(result[0, 0]) == (200, 30, 40)


def test_the_single_pane_modes_are_flicker_overlay_and_grayscale():
    assert SINGLE_PANE_MODES == {ViewMode.FLICKER, ViewMode.OVERLAY, ViewMode.GRAYSCALE}
    assert ANIMATED_MODES == {ViewMode.FLICKER, ViewMode.GRAYSCALE}  # overlay is static


# ---------------------------------------------------------------------------
# ImageStack: levels of detail
# ---------------------------------------------------------------------------


def test_full_size_and_magnified_views_use_the_full_size_level(qapp):
    stack = ImageStack(_solid((255, 255, 255), 3508, 2480))
    assert stack.level_for_zoom(1.0) == 0
    assert stack.level_for_zoom(4.0) == 0


def test_a_reduced_level_is_chosen_but_never_smaller_than_the_display(qapp):
    """Scaling a level *down* by <2x keeps quality; scaling it up would blur."""
    stack = ImageStack(_solid((255, 255, 255), 3508, 2480))
    assert stack.level_for_zoom(0.5) == 1
    assert stack.level_for_zoom(0.3) == 1  # 0.5-scale level shown at 0.6 of itself
    assert stack.level_for_zoom(0.25) == 2
    for zoom in (0.9, 0.5, 0.3, 0.13, 0.05):
        level_scale = 2.0 ** -stack.level_for_zoom(zoom)
        assert level_scale >= zoom * 0.999


def test_the_deepest_level_is_capped(qapp):
    stack = ImageStack(_solid((255, 255, 255), 400, 300))
    assert stack.level_for_zoom(0.0001) == stack.max_level
    assert max(400, 300) / 2**stack.max_level >= 64


def test_a_levels_pixmap_maps_back_to_the_full_page_size(qapp):
    """The pane scales each level back up by these factors, so scene
    coordinates -- and therefore every marker -- stay in full-size pixels."""
    stack = ImageStack(_solid((255, 255, 255), 1000, 700))
    for level in range(stack.max_level + 1):
        pixmap, scale_x, scale_y = stack.pixmap(level)
        assert abs(pixmap.width() * scale_x - 700) < 1e-6
        assert abs(pixmap.height() * scale_y - 1000) < 1e-6


def test_reduction_keeps_a_one_pixel_line(qapp):
    """Why levels exist: thin dielines and fine rules must survive reduction."""
    image = np.full((400, 400, 3), 255, dtype=np.uint8)
    image[:, 201] = 0  # a single 1 px vertical rule on an odd column
    stack = ImageStack(image)

    reduced = stack._array(1)

    assert reduced.shape[:2] == (200, 200)
    assert (reduced < 255).any()  # area averaging keeps it as a grey line
    assert reduced.min() < 200  # ...clearly visible, not a faint ghost

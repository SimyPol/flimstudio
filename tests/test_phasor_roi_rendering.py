from __future__ import annotations

import numpy as np

from flim_app.demo import create_demo_dataset, create_demo_irf
from flim_app.models import AnalysisSettings
from flim_app.phasor_roi import PhasorROI, combined_owner_mask, contains_array
from flim_app.processing import analyze_dataset
from flim_app.rendering import PHASOR_ROI_BACKGROUND, PHASOR_ROI_INVALID, phasor_roi_lifetime_image


def _demo_result_and_whole_image_roi(phasor_min_counts: float = 20.0):
    dataset = create_demo_dataset(size=12, time_bins=32, seed=4)
    irf = create_demo_irf(dataset.y_size, dataset.time_bins, dataset.dt_ns)
    settings = AnalysisSettings(
        method="phasor_phase",
        dt_ns=dataset.dt_ns,
        rep_rate_mhz=dataset.rep_rate_mhz,
        phasor_min_counts=phasor_min_counts,
    )
    result = analyze_dataset(dataset, settings, irf)
    # A rectangle spanning the whole valid (G, S) domain, so every pixel with
    # a finite phasor coordinate falls inside it -- makes the "coloured"
    # variants directly comparable to the underlying image with no partial
    # masking to account for.
    whole_roi = PhasorROI(label="ROI1", shape="rectangle", points_gs=[(-1.0, -1.0), (2.0, 2.0)], colour="#56B4E9")
    return result, whole_roi


def test_all_four_modes_return_correctly_shaped_rgb_images() -> None:
    result, roi = _demo_result_and_whole_image_roi()
    for mode in ("original", "composite", "coloured", "coloured_composite"):
        image = phasor_roi_lifetime_image(result, [roi], mode=mode)
        assert image.mode == "RGB"
        assert image.size == (result.lifetime_ns.shape[1], result.lifetime_ns.shape[0])


def test_coloured_is_a_flat_fill_regardless_of_brightness() -> None:
    result, roi = _demo_result_and_whole_image_roi()
    owner = combined_owner_mask([roi], result.g, result.s)
    inside = owner > 0
    assert inside.any(), "test ROI should cover at least some valid phasor pixels"
    rgb = np.asarray(phasor_roi_lifetime_image(result, [roi], mode="coloured").convert("RGB"))
    unique_inside = {tuple(pixel) for pixel in rgb[inside]}
    # "coloured" ignores brightness entirely -- every covered pixel is the
    # exact same flat colour.
    assert unique_inside == {(0x56, 0xB4, 0xE9)}


def test_coloured_composite_varies_with_brightness_but_keeps_the_roi_hue() -> None:
    result, roi = _demo_result_and_whole_image_roi()
    owner = combined_owner_mask([roi], result.g, result.s)
    inside = owner > 0
    rgb = np.asarray(phasor_roi_lifetime_image(result, [roi], mode="coloured_composite").convert("RGB"), dtype=np.float64)
    covered = rgb[inside]
    # Unlike "coloured", brightness should genuinely vary pixel to pixel
    # (this is the whole point of the composite-weighted variant)...
    assert len({tuple(pixel) for pixel in covered}) > 1
    # ...but every covered pixel's colour is still (up to 8-bit rounding) a
    # scalar multiple of the ROI's own (R, G, B) ratio -- i.e. still
    # recognisably "that ROI's colour", just dimmed by local brightness,
    # never hue-shifted. Compared via each channel's *reconstructed*
    # brightness fraction (pixel / base) rather than a strict cross-channel
    # ratio check, which is far more sensitive to independent per-channel
    # 8-bit quantization noise on small values than the underlying
    # brightness scaling actually is.
    base = np.array([0x56, 0xB4, 0xE9], dtype=np.float64)
    for pixel in covered[:: max(1, len(covered) // 20)]:  # sample, not every pixel -- this is not a slow test either way
        fractions = pixel / base
        assert np.ptp(fractions) < 0.03, f"channel fractions should agree within 8-bit rounding: {fractions}"


def test_invalid_pixels_distinct_from_unclassified_in_every_mode() -> None:
    """A pixel excluded from the phasor calc entirely (NaN G/S -- typically
    too few photons for the sidebar's phasor threshold) must render
    differently from a pixel that has a real phasor coordinate but simply
    doesn't fall inside any drawn ROI -- otherwise "this species is
    genuinely absent here" and "this pixel was never analysed" are
    indistinguishable on the highlight, which is exactly the ambiguity this
    feature exists to remove."""

    # A threshold picked to sit between this dataset's dim background and
    # its bright synthetic cells (see the demo module) -- guarantees a real
    # mix of both valid and below-threshold pixels to compare.
    result, _whole_roi = _demo_result_and_whole_image_roi(phasor_min_counts=500.0)
    invalid_mask = ~(np.isfinite(result.g) & np.isfinite(result.s))
    assert invalid_mask.any(), "test dataset should have at least one below-threshold pixel"
    assert (~invalid_mask).any(), "test dataset should have at least one valid pixel too"
    # A small ROI far from the phasor domain entirely -- guaranteed to
    # match nothing, so every *valid* pixel is "unclassified" and every
    # invalid one is still "invalid": exercises both non-matched states at
    # once, cleanly separated from any "matched" colour.
    empty_roi = PhasorROI(label="ROI1", shape="rectangle", points_gs=[(5.0, 5.0), (5.1, 5.1)], colour="#56B4E9")
    for mode in ("original", "composite", "coloured", "coloured_composite"):
        rgb = np.asarray(phasor_roi_lifetime_image(result, [empty_roi], mode=mode).convert("RGB"))
        invalid_colours = {tuple(pixel) for pixel in rgb[invalid_mask]}
        valid_colours = {tuple(pixel) for pixel in rgb[~invalid_mask]}
        assert invalid_colours == {PHASOR_ROI_INVALID}, f"mode={mode}: {invalid_colours}"
        assert valid_colours == {PHASOR_ROI_BACKGROUND}, f"mode={mode}: {valid_colours}"
        assert PHASOR_ROI_INVALID != PHASOR_ROI_BACKGROUND


def test_show_invalid_false_folds_invalid_back_into_background() -> None:
    """The "Flag below-threshold pixels" checkbox, unchecked: invalid
    pixels should render exactly like unclassified ones again -- the
    pre-this-feature behaviour, for a cleaner view when the distinction
    isn't currently wanted."""

    result, _whole_roi = _demo_result_and_whole_image_roi(phasor_min_counts=500.0)
    invalid_mask = ~(np.isfinite(result.g) & np.isfinite(result.s))
    assert invalid_mask.any()
    empty_roi = PhasorROI(label="ROI1", shape="rectangle", points_gs=[(5.0, 5.0), (5.1, 5.1)], colour="#56B4E9")
    for mode in ("original", "composite", "coloured", "coloured_composite"):
        rgb = np.asarray(
            phasor_roi_lifetime_image(result, [empty_roi], mode=mode, show_invalid=False).convert("RGB")
        )
        all_colours = {tuple(pixel) for pixel in rgb.reshape(-1, 3)}
        assert all_colours == {PHASOR_ROI_BACKGROUND}, f"mode={mode}: {all_colours}"


def test_additive_overlap_sums_colours_where_rois_fully_overlap() -> None:
    result, _whole_roi = _demo_result_and_whole_image_roi()
    green = PhasorROI(label="ROI1", shape="rectangle", points_gs=[(-1.0, -1.0), (2.0, 2.0)], colour="#00FF00")
    red = PhasorROI(label="ROI2", shape="rectangle", points_gs=[(-1.0, -1.0), (2.0, 2.0)], colour="#FF0000")
    owner = combined_owner_mask([green, red], result.g, result.s)
    inside = owner > 0
    assert inside.any()
    rgb = np.asarray(
        phasor_roi_lifetime_image(result, [green, red], mode="coloured", additive_overlap=True).convert("RGB")
    )
    # Fully overlapping green + red should add to yellow everywhere both
    # ROIs cover, not hide the green behind the last-drawn red.
    unique_inside = {tuple(pixel) for pixel in rgb[inside]}
    assert unique_inside == {(0xFF, 0xFF, 0x00)}


def test_additive_overlap_false_keeps_last_roi_wins_behaviour() -> None:
    result, _whole_roi = _demo_result_and_whole_image_roi()
    green = PhasorROI(label="ROI1", shape="rectangle", points_gs=[(-1.0, -1.0), (2.0, 2.0)], colour="#00FF00")
    red = PhasorROI(label="ROI2", shape="rectangle", points_gs=[(-1.0, -1.0), (2.0, 2.0)], colour="#FF0000")
    owner = combined_owner_mask([green, red], result.g, result.s)
    inside = owner > 0
    rgb = np.asarray(
        phasor_roi_lifetime_image(result, [green, red], mode="coloured", additive_overlap=False).convert("RGB")
    )
    # additive_overlap defaults to False -- the pre-existing winner-take-all
    # behaviour (last-drawn ROI wins) must be unchanged.
    unique_inside = {tuple(pixel) for pixel in rgb[inside]}
    assert unique_inside == {(0xFF, 0x00, 0x00)}


def test_additive_overlap_leaves_non_overlapping_rois_unmixed() -> None:
    result, _whole_roi = _demo_result_and_whole_image_roi()
    left = PhasorROI(label="ROI1", shape="rectangle", points_gs=[(-1.0, -1.0), (0.3, 2.0)], colour="#00FF00")
    right = PhasorROI(label="ROI2", shape="rectangle", points_gs=[(0.3, -1.0), (2.0, 2.0)], colour="#FF0000")
    left_mask = contains_array(left, result.g, result.s)
    right_mask = contains_array(right, result.g, result.s)
    assert left_mask.any() and right_mask.any()
    rgb = np.asarray(
        phasor_roi_lifetime_image(result, [left, right], mode="coloured", additive_overlap=True).convert("RGB")
    )
    assert {tuple(p) for p in rgb[left_mask]} == {(0x00, 0xFF, 0x00)}
    assert {tuple(p) for p in rgb[right_mask]} == {(0xFF, 0x00, 0x00)}


def test_coloured_composite_matches_plain_composite_brightness() -> None:
    """The brightest pixel under "coloured_composite" should correspond to
    the brightest pixel of the plain composite image -- confirms the
    brightness scaling is genuinely driven by composite_image's own output,
    not something disconnected from it."""

    from flim_app.rendering import composite_image

    result, roi = _demo_result_and_whole_image_roi()
    coloured_composite = np.asarray(
        phasor_roi_lifetime_image(result, [roi], mode="coloured_composite").convert("RGB"), dtype=np.float64
    )
    plain_composite = np.asarray(composite_image(result).convert("RGB"), dtype=np.float64)
    brightest_cc = np.unravel_index(np.argmax(coloured_composite.sum(axis=-1)), coloured_composite.shape[:2])
    brightest_plain = np.unravel_index(np.argmax(plain_composite.sum(axis=-1)), plain_composite.shape[:2])
    assert brightest_cc == brightest_plain

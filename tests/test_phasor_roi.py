from __future__ import annotations

import numpy as np
import pytest

from flim_app.phasor_roi import (
    PhasorROI,
    any_roi_mask,
    boundary_points_gs,
    centroid_gs,
    combined_owner_mask,
    contains_array,
    enabled_only,
    load_phasor_rois,
    next_phasor_roi_colour,
    rotate_roi,
    save_phasor_rois,
    suggest_rois_from_clusters,
)


def test_rectangle_membership() -> None:
    roi = PhasorROI(label="R1", shape="rectangle", points_gs=[(0.2, 0.1), (0.6, 0.4)], colour="#fff")
    assert roi.contains(0.4, 0.25)
    assert not roi.contains(0.0, 0.0)
    assert not roi.contains(0.7, 0.25)
    # Order of corners must not matter.
    reversed_roi = PhasorROI(label="R1", shape="rectangle", points_gs=[(0.6, 0.4), (0.2, 0.1)], colour="#fff")
    assert reversed_roi.contains(0.4, 0.25)


def test_ellipse_membership() -> None:
    roi = PhasorROI(label="E1", shape="ellipse", points_gs=[(0.0, 0.0), (1.0, 0.5)], colour="#fff")
    assert roi.contains(0.5, 0.25)  # centre
    assert not roi.contains(0.02, 0.02)  # near corner, outside the inscribed ellipse
    assert not roi.contains(-0.1, 0.25)


def test_freeform_triangle_membership() -> None:
    roi = PhasorROI(
        label="F1", shape="freeform",
        points_gs=[(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)], colour="#fff",
    )
    assert roi.contains(0.5, 0.3)
    assert not roi.contains(0.05, 0.9)
    assert not roi.contains(2.0, 2.0)


def test_nan_pixels_never_inside() -> None:
    roi = PhasorROI(label="R1", shape="rectangle", points_gs=[(-1.0, -1.0), (2.0, 2.0)], colour="#fff")
    g = np.array([0.5, np.nan])
    s = np.array([0.5, 0.5])
    mask = contains_array(roi, g, s)
    assert mask[0]
    assert not mask[1]


def test_combined_owner_mask_last_roi_wins_on_overlap() -> None:
    first = PhasorROI(label="A", shape="rectangle", points_gs=[(0.0, 0.0), (1.0, 1.0)], colour="#111")
    second = PhasorROI(label="B", shape="rectangle", points_gs=[(0.4, 0.4), (0.6, 0.6)], colour="#222")
    g = np.array([[0.1, 0.5]])
    s = np.array([[0.1, 0.5]])
    owner = combined_owner_mask([first, second], g, s)
    assert owner[0, 0] == 1  # only inside the first rectangle
    assert owner[0, 1] == 2  # inside both -- the later ROI wins
    assert np.array_equal(any_roi_mask([first, second], g, s), owner > 0)


def test_next_phasor_roi_colour_cycles() -> None:
    from flim_app.phasor_roi import PHASOR_ROI_COLOURS

    assert next_phasor_roi_colour(0) == PHASOR_ROI_COLOURS[0]
    assert next_phasor_roi_colour(len(PHASOR_ROI_COLOURS)) == PHASOR_ROI_COLOURS[0]


def test_rotate_rectangle_90_degrees_swaps_extents() -> None:
    # A wide, short rectangle centred at (0.5, 0.5): half-widths (0.3, 0.1).
    roi = PhasorROI(label="R1", shape="rectangle", points_gs=[(0.2, 0.4), (0.8, 0.6)], colour="#fff")
    assert roi.contains(0.75, 0.5)  # near the right edge of the wide box
    assert not roi.contains(0.5, 0.55 + 0.11)  # just outside the short dimension
    rotate_roi(roi, 90.0)
    assert roi.rotation_deg == 90.0
    # After a 90-degree turn the rectangle is now tall and narrow: the point
    # that used to be safely inside (out along G) is now outside, and a
    # point out along S (previously just outside) is now inside.
    assert not roi.contains(0.75, 0.5)
    assert roi.contains(0.5, 0.5 + 0.25)
    # Corners (points_gs) are untouched -- only rotation_deg changed.
    assert roi.points_gs == [(0.2, 0.4), (0.8, 0.6)]


def test_rotate_rectangle_360_is_identity() -> None:
    roi = PhasorROI(label="R1", shape="rectangle", points_gs=[(0.2, 0.4), (0.8, 0.6)], colour="#fff")
    rotate_roi(roi, 360.0)
    assert abs(roi.rotation_deg) < 1e-9


def test_rotate_freeform_bakes_into_vertices() -> None:
    roi = PhasorROI(
        label="F1", shape="freeform",
        points_gs=[(0.4, 0.5), (0.6, 0.5), (0.6, 0.7), (0.4, 0.7)], colour="#fff",
    )
    cg, cs = centroid_gs(roi)
    assert abs(cg - 0.5) < 1e-9 and abs(cs - 0.6) < 1e-9
    before = list(roi.points_gs)
    rotate_roi(roi, 90.0)
    assert roi.rotation_deg == 0.0  # freeform never uses this field
    assert roi.points_gs != before
    # Rotating 90 degrees about the centroid keeps every vertex the same
    # distance from that centroid.
    for (g0, s0), (g1, s1) in zip(before, roi.points_gs):
        d0 = ((g0 - cg) ** 2 + (s0 - cs) ** 2) ** 0.5
        d1 = ((g1 - cg) ** 2 + (s1 - cs) ** 2) ** 0.5
        assert abs(d0 - d1) < 1e-9


def test_boundary_points_gs_rectangle_matches_rotation() -> None:
    roi = PhasorROI(label="R1", shape="rectangle", points_gs=[(0.0, 0.0), (1.0, 0.5)], colour="#fff")
    unrotated = boundary_points_gs(roi)
    assert len(unrotated) == 4
    rotate_roi(roi, 45.0)
    rotated = boundary_points_gs(roi)
    assert rotated != unrotated
    cg, cs = centroid_gs(roi)
    for g, s in rotated:
        assert abs(((g - cg) ** 2 + (s - cs) ** 2) ** 0.5 - ((0.5) ** 2 + (0.25) ** 2) ** 0.5) < 1e-9


def test_boundary_points_gs_ellipse_has_many_segments() -> None:
    roi = PhasorROI(label="E1", shape="ellipse", points_gs=[(0.0, 0.0), (1.0, 1.0)], colour="#fff")
    points = boundary_points_gs(roi, segments=36)
    assert len(points) == 36
    cg, cs = 0.5, 0.5
    for g, s in points:
        assert abs(((g - cg) / 0.5) ** 2 + ((s - cs) / 0.5) ** 2 - 1.0) < 1e-9


def test_rotation_round_trips_through_save_load(tmp_path) -> None:
    roi = PhasorROI(label="R1", shape="rectangle", points_gs=[(0.2, 0.1), (0.6, 0.4)], colour="#56B4E9")
    rotate_roi(roi, 30.0)
    path = tmp_path / "rotated.json"
    save_phasor_rois([roi], path)
    loaded = load_phasor_rois(path)
    assert abs(loaded[0].rotation_deg - 30.0) < 1e-9


def test_enabled_defaults_true_and_filters() -> None:
    on = PhasorROI(label="R1", shape="rectangle", points_gs=[(0.0, 0.0), (0.5, 0.5)], colour="#111")
    assert on.enabled is True
    off = PhasorROI(label="R2", shape="rectangle", points_gs=[(0.0, 0.0), (0.5, 0.5)], colour="#222", enabled=False)
    assert enabled_only([on, off]) == [on]


def test_disabled_roi_excluded_from_owner_mask() -> None:
    on = PhasorROI(label="R1", shape="rectangle", points_gs=[(0.0, 0.0), (1.0, 1.0)], colour="#111")
    off = PhasorROI(label="R2", shape="rectangle", points_gs=[(0.0, 0.0), (1.0, 1.0)], colour="#222", enabled=False)
    g = np.array([0.5])
    s = np.array([0.5])
    # Both geometrically contain the point -- contains_array itself doesn't
    # know about "enabled" (that's a list-filtering concern, see
    # enabled_only), but the owner mask built from the filtered list must
    # never attribute a pixel to a switched-off ROI.
    assert on.contains(0.5, 0.5) and off.contains(0.5, 0.5)
    owner = combined_owner_mask(enabled_only([on, off]), g, s)
    assert owner[0] == 1  # only "on" survives filtering, so it's the sole (index-1) owner


def test_enabled_round_trips_through_save_load(tmp_path) -> None:
    rois = [
        PhasorROI(label="R1", shape="rectangle", points_gs=[(0.1, 0.1), (0.4, 0.3)], colour="#56B4E9", enabled=False),
    ]
    path = tmp_path / "disabled.json"
    save_phasor_rois(rois, path)
    loaded = load_phasor_rois(path)
    assert loaded[0].enabled is False


def test_save_load_round_trip(tmp_path) -> None:
    rois = [
        PhasorROI(label="R1", shape="rectangle", points_gs=[(0.1, 0.1), (0.4, 0.3)], colour="#56B4E9"),
        PhasorROI(label="F1", shape="freeform", points_gs=[(0.0, 0.0), (1.0, 0.0), (0.5, 0.9)], colour="#E69F00"),
    ]
    path = tmp_path / "rois.json"
    save_phasor_rois(rois, path)
    loaded = load_phasor_rois(path)
    assert len(loaded) == 2
    assert loaded[0].shape == "rectangle"
    assert loaded[0].points_gs == [(0.1, 0.1), (0.4, 0.3)]
    assert loaded[1].shape == "freeform"
    assert loaded[1].colour == "#E69F00"


def _two_blobs(seed: int = 0):
    rng = np.random.default_rng(seed)
    blob_a = rng.normal(loc=(0.2, 0.2), scale=0.02, size=(200, 2))
    blob_b = rng.normal(loc=(0.8, 0.6), scale=0.02, size=(200, 2))
    points = np.concatenate([blob_a, blob_b], axis=0)
    return points[:, 0], points[:, 1]


def test_suggest_rois_from_clusters_finds_two_separated_blobs() -> None:
    g, s = _two_blobs()
    rois = suggest_rois_from_clusters(g, s, n_clusters=2)
    assert len(rois) == 2
    assert all(roi.shape == "ellipse" for roi in rois)
    centroids = sorted(roi.centroid_gs() for roi in rois)
    assert centroids[0] == pytest.approx((0.2, 0.2), abs=0.02)
    assert centroids[1] == pytest.approx((0.8, 0.6), abs=0.02)
    # Each fitted ellipse should actually cover the bulk of its own blob,
    # not just land near the right mean.
    for roi in rois:
        cg, cs = roi.centroid_gs()
        nearby = np.hypot(g - cg, s - cs) < 0.05
        covered = contains_array(roi, g[nearby], s[nearby])
        assert covered.mean() > 0.9


def test_suggest_rois_from_clusters_orders_by_descending_weight() -> None:
    g, s = _two_blobs()
    # Twice as many points in blob A (the first 200) as blob B makes it the
    # heavier cluster regardless of which k-means label index it gets.
    weights = np.ones_like(g)
    rois = suggest_rois_from_clusters(g, s, n_clusters=2, weights=weights)
    first_centroid = rois[0].centroid_gs()
    assert first_centroid == pytest.approx((0.2, 0.2), abs=0.02)


def test_suggest_rois_from_clusters_ellipse_orientation_matches_covariance() -> None:
    # A cluster stretched along the G axis and barely spread in S: the
    # fitted ellipse's long axis should point along G (rotation near 0 or
    # 180 degrees), not S.
    rng = np.random.default_rng(1)
    g = rng.normal(loc=0.5, scale=0.05, size=500)
    s = rng.normal(loc=0.3, scale=0.005, size=500)
    roi = suggest_rois_from_clusters(g, s, n_clusters=1)[0]
    angle = roi.rotation_deg % 180.0
    assert angle < 10.0 or angle > 170.0


def _tilted_cloud(rng, n, centre, sd_major, sd_minor, angle_deg):
    import math

    a = math.radians(angle_deg)
    rotation = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return (rng.normal(size=(n, 2)) * [sd_major, sd_minor]) @ rotation.T + centre


def _angle_difference_deg(a: float, b: float) -> float:
    # An ellipse is unchanged by a 180 degree turn.
    return abs((a - b + 90.0) % 180.0 - 90.0)


def test_suggest_rois_from_clusters_follows_a_tilted_elongated_cloud() -> None:
    rng = np.random.default_rng(0)
    core = _tilted_cloud(rng, 4000, (0.30, 0.43), 0.045, 0.012, -30.0)
    # A sparse halo and tail of noisy pixels must not drag the fit off the
    # dense core -- this is what the robust core refit is for.
    tail = _tilted_cloud(rng, 300, (0.38, 0.36), 0.08, 0.02, -50.0)
    halo = rng.normal(loc=(0.28, 0.46), scale=0.05, size=(400, 2))
    points = np.concatenate([core, tail, halo])
    roi = suggest_rois_from_clusters(points[:, 0], points[:, 1], n_clusters=1)[0]
    assert _angle_difference_deg(roi.rotation_deg, 150.0) < 5.0
    assert roi.centroid_gs() == pytest.approx((0.30, 0.43), abs=0.01)
    (g0, s0), (g1, s1) = roi.points_gs
    assert (g1 - g0) / 2 == pytest.approx(2 * 0.045, rel=0.2)  # semi-major, sigma=2
    assert (s1 - s0) / 2 == pytest.approx(2 * 0.012, rel=0.25)  # semi-minor
    assert contains_array(roi, core[:, 0], core[:, 1]).mean() > 0.8


def test_suggest_rois_from_clusters_keeps_thin_tilted_cluster_distinct_from_round_blob() -> None:
    rng = np.random.default_rng(0)
    thin = _tilted_cloud(rng, 3000, (0.35, 0.40), 0.05, 0.008, -35.0)
    round_blob = rng.normal(loc=(0.30, 0.47), scale=0.03, size=(3000, 2))
    points = np.concatenate([thin, round_blob])
    rois = suggest_rois_from_clusters(points[:, 0], points[:, 1], n_clusters=2)
    assert len(rois) == 2
    thin_roi = min(rois, key=lambda roi: abs(roi.centroid_gs()[0] - 0.35))
    round_roi = min(rois, key=lambda roi: abs(roi.centroid_gs()[0] - 0.30))
    assert _angle_difference_deg(thin_roi.rotation_deg, 145.0) < 8.0
    assert contains_array(thin_roi, thin[:, 0], thin[:, 1]).mean() > 0.75
    assert contains_array(round_roi, round_blob[:, 0], round_blob[:, 1]).mean() > 0.75
    # Each ellipse should mostly avoid the other population.
    assert contains_array(thin_roi, round_blob[:, 0], round_blob[:, 1]).mean() < 0.25


def test_suggest_rois_from_clusters_weights_pull_fit_toward_heavy_points() -> None:
    # One cluster, two point masses far apart within it -- weighting one
    # side far more heavily should pull the fitted mean toward it.
    g = np.array([0.1] * 50 + [0.9] * 50)
    s = np.array([0.1] * 50 + [0.9] * 50)
    weights = np.array([1.0] * 50 + [1000.0] * 50)
    roi = suggest_rois_from_clusters(g, s, n_clusters=1, weights=weights)[0]
    cg, cs = roi.centroid_gs()
    assert cg > 0.8 and cs > 0.8


def test_suggest_rois_from_clusters_excludes_invalid_points() -> None:
    g, s = _two_blobs()
    g = g.copy()
    g[:100] = np.nan  # half of blob A becomes invalid
    rois = suggest_rois_from_clusters(g, s, n_clusters=2)
    assert len(rois) == 2
    centroids = sorted(roi.centroid_gs() for roi in rois)
    assert centroids[0] == pytest.approx((0.2, 0.2), abs=0.02)
    assert centroids[1] == pytest.approx((0.8, 0.6), abs=0.02)


def test_suggest_rois_from_clusters_raises_when_too_few_points() -> None:
    g = np.array([0.1, 0.2, np.nan])
    s = np.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="valid phasor point"):
        suggest_rois_from_clusters(g, s, n_clusters=5)

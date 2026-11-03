from __future__ import annotations

"""Phasor-plot ROI drawing.

Shapes are defined in true (G, S) phasor-coordinate space -- never in
rendered pixels, normalised image fractions, or anything tied to a
particular plot size or zoom level -- so a drawn ROI stays meaningful
across re-renders, and round-trips through save/load unchanged regardless
of what size the phasor plot happens to be drawn at later.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Literal

import numpy as np

PhasorShapeKind = Literal["rectangle", "ellipse", "freeform"]

# Fixed order requested by the user: ROI1 green, ROI2 red, ROI3 blue,
# ROI4 orange -- then continues with a few more clearly distinct colours
# for a 5th+ ROI. No longer the same palette as comparison points
# (points.py's POINT_COLOURS); that one keeps its own colour-blind-safe
# set independently.
PHASOR_ROI_COLOURS = (
    "#2CA02C",  # ROI1 -- green
    "#D62728",  # ROI2 -- red
    "#1F77B4",  # ROI3 -- blue
    "#FF7F0E",  # ROI4 -- orange
    "#9467BD",  # purple
    "#8C564B",  # brown
    "#E377C2",  # pink
    "#7F7F7F",  # grey
)


def next_phasor_roi_colour(existing_count: int) -> str:
    return PHASOR_ROI_COLOURS[existing_count % len(PHASOR_ROI_COLOURS)]


@dataclass(slots=True)
class PhasorROI:
    """One user-drawn region of interest in phasor (G, S) space.

    ``points_gs`` holds the shape's defining vertices in true G/S data
    coordinates:
    - "rectangle"/"ellipse": exactly two opposite corners of the
      *unrotated* bounding box -- ``rotation_deg`` (about the box's own
      centre) is applied on top for both hit-testing and rendering, so
      rotating one of these shapes never has to re-derive a new pair of
      corners.
    - "freeform": three or more polygon vertices, implicitly closed.
      Rotating a freeform ROI instead bakes the rotation directly into
      these vertices (see ``rotate_roi``) -- there is no separate
      unrotated reference frame worth keeping for an arbitrary polygon,
      so ``rotation_deg`` always stays 0 for this shape.
    """

    label: str
    shape: PhasorShapeKind
    points_gs: list[tuple[float, float]]
    colour: str
    rotation_deg: float = 0.0
    # Switched off without deleting: excluded from rendering, the lifetime
    # highlight, and rotation hit-testing (see gui.py's enabled_only calls
    # and _phasor_roi_hit_test), but still kept in the list -- and still
    # saved/loaded -- so it can be switched back on later.
    enabled: bool = True

    def bounds_gs(self) -> tuple[float, float, float, float]:
        g_values = [g for g, _ in self.points_gs]
        s_values = [s for _, s in self.points_gs]
        return min(g_values), min(s_values), max(g_values), max(s_values)

    def centroid_gs(self) -> tuple[float, float]:
        return centroid_gs(self)

    def contains(self, g: float, s: float) -> bool:
        return bool(contains_array(self, np.asarray([g]), np.asarray([s]))[0])

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "shape": self.shape,
            "points_gs": [[float(g), float(s)] for g, s in self.points_gs],
            "colour": self.colour,
            "rotation_deg": float(self.rotation_deg),
            "enabled": bool(self.enabled),
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "PhasorROI":
        shape = str(data["shape"])
        if shape not in ("rectangle", "ellipse", "freeform"):
            raise ValueError(f"Unknown phasor ROI shape: {shape!r}")
        points = [(float(g), float(s)) for g, s in data["points_gs"]]
        return PhasorROI(
            label=str(data.get("label", "ROI")),
            shape=shape,  # type: ignore[arg-type]
            points_gs=points,
            colour=str(data.get("colour", "#FFFFFF")),
            rotation_deg=float(data.get("rotation_deg", 0.0)),
            enabled=bool(data.get("enabled", True)),
        )


def centroid_gs(roi: PhasorROI) -> tuple[float, float]:
    if roi.shape == "freeform":
        g_values = [g for g, _ in roi.points_gs]
        s_values = [s for _, s in roi.points_gs]
        return sum(g_values) / len(g_values), sum(s_values) / len(s_values)
    (g0, s0), (g1, s1) = roi.points_gs[0], roi.points_gs[1]
    return (g0 + g1) / 2.0, (s0 + s1) / 2.0


def rotate_roi(roi: PhasorROI, delta_deg: float) -> None:
    """Rotate ``roi`` in place by ``delta_deg`` about its own centroid.

    Freeform vertices are physically rotated (there's no separate
    unrotated reference worth keeping for an arbitrary polygon).
    Rectangle/ellipse instead just accumulate the angle in
    ``rotation_deg`` -- their two corner points keep describing the
    unrotated box; ``contains_array``/``boundary_points_gs`` apply the
    angle on top of that whenever they're used.
    """

    if roi.shape == "freeform":
        cg, cs = centroid_gs(roi)
        angle = math.radians(delta_deg)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        roi.points_gs = [
            (
                cg + (g - cg) * cos_a - (s - cs) * sin_a,
                cs + (g - cg) * sin_a + (s - cs) * cos_a,
            )
            for g, s in roi.points_gs
        ]
    else:
        roi.rotation_deg = (roi.rotation_deg + delta_deg) % 360.0


def boundary_points_gs(roi: PhasorROI, segments: int = 72) -> list[tuple[float, float]]:
    """Concrete boundary vertices for any ROI shape, in (G, S) space,
    respecting rotation -- the single representation both the PIL phasor
    renderer and the interactive canvas overlay draw from, so an ellipse
    or a rotated rectangle never needs its own separate drawing code path.
    """

    if roi.shape == "freeform":
        return list(roi.points_gs)
    (g0, s0), (g1, s1) = roi.points_gs[0], roi.points_gs[1]
    g_lo, g_hi = sorted((g0, g1))
    s_lo, s_hi = sorted((s0, s1))
    cg, cs = (g_lo + g_hi) / 2.0, (s_lo + s_hi) / 2.0
    rg, rs = (g_hi - g_lo) / 2.0, (s_hi - s_lo) / 2.0
    angle = math.radians(roi.rotation_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)

    def to_world(lx: float, ly: float) -> tuple[float, float]:
        return (cg + lx * cos_a - ly * sin_a, cs + lx * sin_a + ly * cos_a)

    if roi.shape == "rectangle":
        return [to_world(dx, dy) for dx, dy in ((-rg, -rs), (rg, -rs), (rg, rs), (-rg, rs))]
    return [
        to_world(rg * math.cos(t), rs * math.sin(t))
        for t in np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
    ]


def contains_array(roi: PhasorROI, g: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Vectorised membership test.

    ``g``/``s`` are same-shape arrays (e.g. a whole image's phasor maps);
    returns a same-shape boolean mask. NaN coordinates (pixels excluded
    from the phasor calculation entirely) are always False, never
    accidentally "inside" a shape that happens to cover the plot origin.
    """

    g = np.asarray(g, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    finite = np.isfinite(g) & np.isfinite(s)
    if roi.shape in ("rectangle", "ellipse"):
        (g0, s0), (g1, s1) = roi.points_gs[0], roi.points_gs[1]
        g_lo, g_hi = sorted((g0, g1))
        s_lo, s_hi = sorted((s0, s1))
        cg, cs = (g_lo + g_hi) / 2.0, (s_lo + s_hi) / 2.0
        rg, rs = max((g_hi - g_lo) / 2.0, 1e-12), max((s_hi - s_lo) / 2.0, 1e-12)
        dg, ds = g - cg, s - cs
        if roi.rotation_deg:
            # Test in the shape's own unrotated frame: rotate the query
            # points *backwards* by the shape's rotation instead of
            # re-deriving a rotated box/ellipse formula.
            angle = math.radians(-roi.rotation_deg)
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            dg, ds = dg * cos_a - ds * sin_a, dg * sin_a + ds * cos_a
        if roi.shape == "rectangle":
            inside = (np.abs(dg) <= rg) & (np.abs(ds) <= rs)
        else:
            inside = (dg / rg) ** 2 + (ds / rs) ** 2 <= 1.0
    elif roi.shape == "freeform":
        inside = _point_in_polygon(g, s, roi.points_gs)
    else:
        raise ValueError(f"Unknown phasor ROI shape: {roi.shape!r}")
    return inside & finite


def _point_in_polygon(g: np.ndarray, s: np.ndarray, vertices: list[tuple[float, float]]) -> np.ndarray:
    """Vectorised even-odd (ray-casting) point-in-polygon test, closing the
    polygon back to its first vertex automatically."""

    n = len(vertices)
    inside = np.zeros(np.broadcast(g, s).shape, dtype=bool)
    if n < 3:
        return inside
    gj, sj = vertices[-1]
    for gi, si in vertices:
        crosses = (si > s) != (sj > s)
        # Guard the division for a horizontal edge (sj == si); such an
        # edge never itself flips the parity for a ray cast along s, so
        # the guarded value is never actually used where crosses is True.
        denom = np.where(sj - si == 0, 1.0, sj - si)
        g_intersect = gi + (s - si) / denom * (gj - gi)
        flip = crosses & (g < g_intersect)
        inside = np.where(flip, ~inside, inside)
        gj, sj = gi, si
    return inside


def enabled_only(rois: list[PhasorROI]) -> list[PhasorROI]:
    """The subset of ``rois`` currently switched on.

    Every place a ROI list drives rendering, the lifetime highlight, or
    hit-testing should filter through this first -- a ROI switched off
    stays in the full list (and still saves/loads) but otherwise behaves
    as if it doesn't exist.
    """

    return [roi for roi in rois if roi.enabled]


def combined_owner_mask(rois: list[PhasorROI], g: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Per-pixel owning ROI: 0 = none, else a 1-based index into ``rois``.

    Later ROIs in the list win where shapes overlap -- the most recently
    drawn region is the one a pixel "belongs to" for the coloured-regions
    display, matching how a newly drawn ROI visually sits "on top" of
    earlier ones in the phasor plot.
    """

    owner = np.zeros(np.broadcast(g, s).shape, dtype=np.int32)
    for index, roi in enumerate(rois, start=1):
        owner = np.where(contains_array(roi, g, s), index, owner)
    return owner


def any_roi_mask(rois: list[PhasorROI], g: np.ndarray, s: np.ndarray) -> np.ndarray:
    return combined_owner_mask(rois, g, s) > 0


# Two-degree-of-freedom chi-square 99% cutoff, and the factor by which
# trimming a Gaussian at that cutoff shrinks its covariance
# (P(chi2_4 <= c) / P(chi2_2 <= c)) -- divided back out so the trimmed fit
# stays an unbiased estimate of the core's own standard deviation.
_TRIM_D2 = 9.210340371976184
_TRIM_COV_SHRINK = 0.9530


def _weighted_mean_cov(points: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total = float(np.sum(weights))
    mean = (weights[:, None] * points).sum(axis=0) / total
    centered = points - mean
    cov = (weights[:, None] * centered).T @ centered / total
    return mean, cov


def _mahalanobis_sq(points: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    det = cov[0, 0] * cov[1, 1] - cov[0, 1] * cov[1, 0]
    inverse = np.array([[cov[1, 1], -cov[0, 1]], [-cov[1, 0], cov[0, 0]]]) / det
    centered = points - mean
    return np.einsum("ni,ij,nj->n", centered, inverse, centered)


def _fit_gaussian_mixture(
    points: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    variance_floor: float,
    max_iter: int = 200,
    tol: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted EM for a full-covariance 2-D Gaussian mixture, started from
    ``means``. Returns (means, covariances, per-point responsibilities)."""

    n, k = len(points), len(means)
    floor = variance_floor * np.eye(2)
    means = np.array(means, dtype=np.float64)
    nearest = ((points[:, None, :] - means[None, :, :]) ** 2).sum(axis=-1).argmin(axis=1)
    _, global_cov = _weighted_mean_cov(points, weights)
    covs = np.empty((k, 2, 2))
    for j in range(k):
        member = nearest == j
        if member.sum() >= 3:
            covs[j] = _weighted_mean_cov(points[member], weights[member])[1]
        else:
            covs[j] = global_cov / k
        covs[j] += floor
    mix = np.full(k, 1.0 / k)

    total_weight = float(np.sum(weights))
    previous = -np.inf
    resp = np.full((n, k), 1.0 / k)
    for _ in range(max_iter):
        log_p = np.empty((n, k))
        for j in range(k):
            det = covs[j][0, 0] * covs[j][1, 1] - covs[j][0, 1] * covs[j][1, 0]
            log_p[:, j] = (
                np.log(max(mix[j], 1e-300))
                - 0.5 * _mahalanobis_sq(points, means[j], covs[j])
                - 0.5 * np.log(det)
                - np.log(2.0 * np.pi)
            )
        peak = log_p.max(axis=1, keepdims=True)
        log_norm = peak + np.log(np.exp(log_p - peak).sum(axis=1, keepdims=True))
        resp = np.exp(log_p - log_norm)
        likelihood = float((weights * log_norm[:, 0]).sum() / total_weight)

        weighted_resp = weights[:, None] * resp
        component_weight = weighted_resp.sum(axis=0)
        for j in range(k):
            if component_weight[j] < 1e-9 * total_weight:
                continue
            means[j], covs[j] = _weighted_mean_cov(points, weighted_resp[:, j])
            covs[j] += floor
        mix = component_weight / component_weight.sum()
        if abs(likelihood - previous) < tol:
            break
        previous = likelihood
    return means, covs, resp


def suggest_rois_from_clusters(
    g: np.ndarray,
    s: np.ndarray,
    n_clusters: int,
    weights: np.ndarray | None = None,
    sigma: float = 2.0,
    seed: int = 0,
    max_points: int = 200_000,
) -> list[PhasorROI]:
    """Automatically propose ``n_clusters`` ellipse ROIs from a phasor cloud.

    Fits a full-covariance Gaussian mixture (weighted EM, started from
    k-means centres) to the valid (finite G, S) points, so each component
    can be elongated and tilted to follow the cloud's own shape -- plain
    k-means assumes round clusters and would instead slice an elongated
    cloud (typical of a dye whose phasor smears along a mixing line) into
    round chunks. Each component is then refit on its own core only:
    points beyond the 99% Mahalanobis radius are trimmed and the fit
    repeated, so a sparse halo or tail of noisy low-photon pixels cannot
    drag the ellipse's centre, size or orientation away from the dense
    core. The result becomes an ellipse ROI spanning ``sigma`` standard
    deviations along each principal axis, using the existing
    ellipse+rotation representation (no new ROI type).

    ``weights`` should be per-pixel photon counts where available -- a dim
    species' pixels are noisier and would otherwise count as much as a
    bright, reliable one. Clouds larger than ``max_points`` are randomly
    subsampled (seeded) for speed.

    Returns ROIs ordered by descending total component weight (most
    populated/most reliable first), so that a smaller, rarer population
    added after it into the caller's own ROI list wins any overlap once
    handed to ``combined_owner_mask``. Labels/colours are placeholders
    ("Cluster 1", "Cluster 2", ...) -- a caller appending these to an
    existing ROI list should renumber/recolour to avoid collisions.

    Raises ``ValueError`` if fewer valid points exist than ``n_clusters``,
    or if the initial k-means seeding fails (e.g. every valid point is
    identical).
    """

    from scipy.cluster.vq import kmeans2

    g = np.asarray(g, dtype=np.float64).ravel()
    s = np.asarray(s, dtype=np.float64).ravel()
    if g.shape != s.shape:
        raise ValueError(f"g and s must have the same shape; got {g.shape} and {s.shape}.")
    valid = np.isfinite(g) & np.isfinite(s)
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64).ravel()
        if weights.shape != g.shape:
            raise ValueError(f"weights must match g/s shape; got {weights.shape} vs {g.shape}.")
        valid &= np.isfinite(weights) & (weights > 0)
    points = np.stack([g[valid], s[valid]], axis=1)
    if len(points) < n_clusters:
        raise ValueError(
            f"Only {len(points)} valid phasor point(s) are available -- cannot form {n_clusters} cluster(s)."
        )
    point_weights = weights[valid] if weights is not None else np.ones(len(points))
    if len(points) > max_points:
        keep = np.random.default_rng(seed).choice(len(points), size=max_points, replace=False)
        points, point_weights = points[keep], point_weights[keep]
    point_weights = point_weights / point_weights.mean()

    if n_clusters == 1:
        initial_means = np.average(points, axis=0, weights=point_weights)[None, :]
    else:
        try:
            initial_means, _labels = kmeans2(points, n_clusters, minit="++", seed=seed)
        except Exception as exc:
            raise ValueError(f"Clustering failed: {exc}") from exc

    # A small variance floor (relative to the whole cloud's own spread)
    # keeps a near-degenerate component (few points, or points that are
    # nearly collinear) from collapsing into an unusably thin or
    # zero-size ellipse.
    span = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])), 1e-9)
    variance_floor = (span * 1e-3) ** 2
    floor = variance_floor * np.eye(2)

    means, covs, resp = _fit_gaussian_mixture(points, point_weights, initial_means, variance_floor)
    total_weight = float(np.sum(point_weights))

    fits: list[tuple[float, tuple[float, float], float, float, float]] = []
    for j in range(len(means)):
        component_weights = point_weights * resp[:, j]
        component_total = float(component_weights.sum())
        if component_total < 1e-4 * total_weight:
            continue
        mean, cov = means[j], covs[j]
        for _ in range(5):
            core = _mahalanobis_sq(points, mean, cov) <= _TRIM_D2
            core_weights = component_weights * core
            if core_weights.sum() < 1e-4 * total_weight:
                break
            mean, cov = _weighted_mean_cov(points, core_weights)
            cov = cov / _TRIM_COV_SHRINK + floor
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.clip(eigenvalues[order], variance_floor, None)
        major = eigenvectors[:, order][:, 0]
        angle_deg = math.degrees(math.atan2(major[1], major[0])) % 360.0
        rg = sigma * math.sqrt(eigenvalues[0])
        rs = sigma * math.sqrt(eigenvalues[1])
        fits.append((component_total, (float(mean[0]), float(mean[1])), rg, rs, angle_deg))

    if not fits:
        raise ValueError("Clustering produced no non-empty clusters.")

    fits.sort(key=lambda item: item[0], reverse=True)

    rois: list[PhasorROI] = []
    for index, (_weight, (cg, cs), rg, rs, angle_deg) in enumerate(fits):
        rois.append(
            PhasorROI(
                label=f"Cluster {index + 1}",
                shape="ellipse",
                points_gs=[(cg - rg, cs - rs), (cg + rg, cs + rs)],
                colour=next_phasor_roi_colour(index),
                rotation_deg=angle_deg,
            )
        )
    return rois


PHASOR_ROI_FILE_VERSION = 1


def save_phasor_rois(rois: list[PhasorROI], path: str | Path) -> None:
    payload = {"version": PHASOR_ROI_FILE_VERSION, "rois": [roi.to_dict() for roi in rois]}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_phasor_rois(path: str | Path) -> list[PhasorROI]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_rois = payload.get("rois", []) if isinstance(payload, dict) else payload
    return [PhasorROI.from_dict(entry) for entry in raw_rois]

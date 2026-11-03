from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .demo import create_demo_dataset, create_demo_irf
from .io import load_dataset
from .models import AnalysisSettings
from .processing import analyze_dataset
from .rendering import export_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyse time-resolved FLIM image cubes.")
    parser.add_argument("input", nargs="?", help="ICS/IDS, NPY/NPZ, TIFF, CSV, or TXT data file")
    parser.add_argument("--irf", help="Optional IRF dataset")
    parser.add_argument("--background", help="Optional background dataset")
    parser.add_argument("--output", default="flim_results", help="Output directory")
    parser.add_argument(
        "--method",
        choices=(
            "phasor_phase", "phasor_modulation", "mono_exponential", "moment",
            "flimlib_reconvolution", "flimlib_global",
        ),
        default="phasor_phase",
    )
    parser.add_argument("--dt-ns", type=float, help="Override time-bin width in ns")
    parser.add_argument("--rep-rate-mhz", type=float, help="Override laser repetition rate in MHz")
    parser.add_argument(
        "--min-photons",
        type=float,
        default=30.0,
        help="Legacy shared threshold used only when a method-specific threshold is negative",
    )
    parser.add_argument("--phasor-min-counts", type=float, default=1000.0)
    parser.add_argument("--fit-min-counts", type=float, default=1000.0)
    parser.add_argument("--background-counts", type=float, default=0.0)
    parser.add_argument(
        "--background-scale-mode",
        choices=("auto", "manual"),
        default="auto",
        help="Infer sample/background exposure ratio from loops/time, or use the manual multiplier",
    )
    parser.add_argument(
        "--background-multiplier",
        type=float,
        default=0.025,
        help="Manual background multiplier and fallback when auto metadata is unavailable",
    )
    parser.add_argument(
        "--spatial-bin",
        type=int,
        default=0,
        help="Spatial-bin radius level: 0=1x1, 1=3x3, 2=5x5, ...",
    )
    parser.add_argument("--smoothing-passes", type=int, default=0)
    parser.add_argument(
        "--interpolate-rows",
        help="Defined zero-based detector rows, e.g. '3,30,54' or '10-14'",
    )
    parser.add_argument(
        "--row-coordinate-mode",
        choices=("direct_y", "labview_mirrored"),
        default="direct_y",
        help="Interpret row IDs directly or with y=height-1-row (original LabVIEW convention)",
    )
    parser.add_argument("--row-radius", type=int, default=2, help="Number of good donor rows per side")
    parser.add_argument(
        "--interpolate-columns",
        help="Defined zero-based detector columns, e.g. '12,44,101-103'",
    )
    parser.add_argument("--column-radius", type=int, default=2)
    parser.add_argument("--phasor-heatmap-bins", type=int, default=400)
    parser.add_argument(
        "--labview-row-preset",
        action="store_true",
        help="Use the 19-row map saved in the supplied main VI (mirrored convention)",
    )
    parser.add_argument("--offset-bins", type=float, default=0.0)
    parser.add_argument("--lifetime-min", type=float, default=0.5)
    parser.add_argument("--lifetime-max", type=float, default=4.5)
    parser.add_argument("--lifetime-scale", choices=("auto", "manual"), default="manual")
    parser.add_argument("--lifetime-lut", help="Optional 768-byte ImageJ raw lifetime LUT")
    parser.add_argument("--invert-lifetime-lut", action="store_true")
    parser.add_argument(
        "--clamp-lifetime-lut",
        action="store_true",
        help="Clamp values outside the display range instead of removing them",
    )
    parser.add_argument("--histogram-intervals", type=int, default=50)
    parser.add_argument("--detailed-histogram-bins", type=int, default=1000)
    parser.add_argument("--fit-start-bin", type=int, default=-1)
    parser.add_argument("--fit-end-bin", type=int, default=-1)
    parser.add_argument(
        "--native-fit-bins",
        type=int,
        default=29,
        help="Leading TCSPC bins supplied to native FLIMLib (-1/0 uses all acquisition bins)",
    )
    parser.add_argument("--native-bin-width-ns", type=float, default=0.4)
    parser.add_argument(
        "--native-irf-baseline",
        choices=("none", "median", "tail_median"),
        default="median",
    )
    parser.add_argument("--native-dll", help="Path to flimfit_labview.dll")
    parser.add_argument("--model-order", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--fit-background-mode", choices=("free", "fixed"), default="free")
    parser.add_argument("--fit-background-value", type=float, default=0.0)
    parser.add_argument("--noise-model", type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument("--irf-mode", choices=("global", "per_pixel"), default="global")
    parser.add_argument(
        "--irf-integrate-per-column",
        action="store_true",
        help=(
            "Per-pixel IRF only: sum each column's IRF across every row into one high-SNR "
            "curve and repeat it down that column, for a sweeping linear-array detector "
            "(e.g. RA2) -- see processing.integrate_irf_per_column."
        ),
    )
    parser.add_argument("--global-iterations", type=int, default=4)
    parser.add_argument("--max-global-pixels", type=int, default=20_000)
    parser.add_argument("--initial-taus", type=float, nargs="*", default=())
    parser.add_argument("--demo", action="store_true", help="Analyse the bundled synthetic demo")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.demo:
        dataset = create_demo_dataset()
        irf = create_demo_irf(dataset.y_size, dataset.time_bins, dataset.dt_ns)
    elif args.input:
        dataset = load_dataset(args.input)
        irf = load_dataset(args.irf) if args.irf else None
    else:
        raise SystemExit("Provide an input file or use --demo.")
    background = load_dataset(args.background) if args.background else None
    from .row_interpolation import labview_main_preset, parse_row_spec

    if args.labview_row_preset:
        row_map = labview_main_preset()
    else:
        row_map = None
        if args.interpolate_rows:
            from .row_interpolation import RowInterpolationMap

            row_map = RowInterpolationMap(
                rows=parse_row_spec(args.interpolate_rows),
                radius=args.row_radius,
                coordinate_mode=args.row_coordinate_mode,
                name="CLI detector-row map",
            ).normalized()
    settings = AnalysisSettings(
        method=args.method,
        dt_ns=args.dt_ns if args.dt_ns is not None else dataset.dt_ns,
        rep_rate_mhz=args.rep_rate_mhz if args.rep_rate_mhz is not None else dataset.rep_rate_mhz,
        min_photons=args.min_photons,
        phasor_min_counts=args.phasor_min_counts,
        fit_min_counts=args.fit_min_counts,
        background_counts=args.background_counts,
        background_scale_mode=args.background_scale_mode,
        background_multiplier=args.background_multiplier,
        background_auto_fallback=args.background_multiplier,
        spatial_bin=args.spatial_bin,
        smoothing_passes=args.smoothing_passes,
        interpolate_rows=row_map is not None,
        interpolation_rows=row_map.rows if row_map else (),
        interpolation_radius=row_map.radius if row_map else args.row_radius,
        interpolation_coordinate_mode=row_map.coordinate_mode if row_map else args.row_coordinate_mode,
        interpolation_map_name=row_map.name if row_map else "",
        interpolation_expected_source_height=row_map.expected_source_height if row_map and row_map.expected_source_height else 0,
        interpolate_columns=bool(args.interpolate_columns),
        interpolation_columns=(
            parse_row_spec(args.interpolate_columns) if args.interpolate_columns else ()
        ),
        column_interpolation_radius=args.column_radius,
        phasor_heatmap_bins=args.phasor_heatmap_bins,
        timing_offset_bins=args.offset_bins,
        lifetime_min_ns=args.lifetime_min,
        lifetime_max_ns=args.lifetime_max,
        lifetime_scale_mode=args.lifetime_scale,
        lifetime_lut_path=args.lifetime_lut or "",
        lifetime_lut_inverse=args.invert_lifetime_lut,
        lifetime_lut_remove_out_of_range=not args.clamp_lifetime_lut,
        histogram_bins=args.histogram_intervals,
        detailed_histogram_bins=args.detailed_histogram_bins,
        fit_start_bin=args.fit_start_bin,
        fit_end_bin=args.fit_end_bin,
        native_fit_bins=args.native_fit_bins,
        native_bin_width_ns=args.native_bin_width_ns,
        native_irf_baseline_mode=args.native_irf_baseline,
        native_dll_path=args.native_dll or "",
        model_order=args.model_order,
        fit_background_mode=1 if args.fit_background_mode == "fixed" else 0,
        fit_background_value=args.fit_background_value,
        fit_noise_model=args.noise_model,
        fit_irf_mode=1 if args.irf_mode == "per_pixel" else 0,
        irf_integrate_per_column=bool(args.irf_integrate_per_column),
        global_iterations=args.global_iterations,
        max_global_pixels=args.max_global_pixels,
        initial_taus_ns=tuple(args.initial_taus),
    )
    result = analyze_dataset(
        dataset,
        settings,
        irf,
        background,
        lambda value, message: print(f"[{value:>5.0%}] {message}", file=sys.stderr),
    )
    files = export_result(result, Path(args.output))
    print(json.dumps({"summary": result.summary(), "files": [str(path) for path in files]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

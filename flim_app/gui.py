from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import json
import math
import os
import subprocess
import threading
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, VERTICAL, X, Y, filedialog, messagebox
import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk

from .demo import create_demo_dataset, create_demo_irf
from .global_model import GlobalModelError, TrainedGlobalModel, load_global_model
from .io import SUPPORTED_EXTENSIONS, load_dataset
from .models import AnalysisResult, AnalysisSettings, FLIMDataset
from .native_backend import NativeFitResult, fit_in_subprocess, find_native_dll
from .processing import (
    _prepare_background,
    analyze_dataset,
    estimate_synthetic_gaussian_irf,
    native_multiexponential_curve,
    phasor_lifetimes,
    resolve_background_multiplier,
    rolling_spatial_bin,
    shift_time_axis,
)
from .phasor_roi import (
    PhasorROI,
    boundary_points_gs,
    combined_owner_mask,
    contains_array,
    enabled_only,
    load_phasor_rois,
    next_phasor_roi_colour,
    rotate_roi,
    save_phasor_rois,
    suggest_rois_from_clusters,
)
from .points import MAX_SELECTED_POINTS, PointAnalysis, PointSelection, add_point, analyse_points, lifetime_label
from .training_gui import TrainingDialog
from .manual_model_gui import ManualModelDialog
from .leica_export_gui import LeicaExportDialog, resolve_leica_backend
from ._paths import bundled_leica_root, main_folder
from .settings_ini import (
    DEFAULT_INI_NAME,
    LoadedFilePaths,
    load_file_paths_ini,
    load_settings_ini,
    save_settings_ini,
)
from .row_interpolation import (
    DIRECT_Y,
    LABVIEW_MIRRORED,
    RowInterpolationMap,
    format_row_spec,
    labview_main_preset,
    load_row_map,
    parse_row_spec,
    save_row_map,
)
from .roi import (
    BoundsXYXY,
    ROIAnalysis,
    analyse_roi,
    bounds_to_normalised,
    clamp_bounds,
    normalised_to_bounds,
    normalised_to_pixel,
    pixel_to_normalised_center,
)
from .rendering import (
    ACCENT,
    AMBER,
    BG,
    GREEN,
    MUTED,
    PANEL,
    TEXT,
    chi_square_image,
    chi_square_stats,
    colourise,
    composite_image,
    export_point_selections,
    export_roi_analysis,
    export_result,
    histogram_plot,
    interpolation_map_image,
    irf_decay_overlay_plot,
    line_plot,
    lifetime_image,
    component_lifetime_image,
    component_amplitude_image,
    lifetime_display_range,
    phasor_gs_to_pixel,
    phasor_pixel_to_gs,
    phasor_plot,
    phasor_roi_lifetime_image,
    threshold_intensity_image,
)


SIDEBAR = "#0f1a2b"
PANEL_ALT = "#162338"
BORDER = "#263750"
RED = "#fb7185"

# The exact size the detailed Phasor tab's plot is always rendered at (see
# _refresh_phasors) -- phasor ROI drawing on that view needs this exact,
# fixed value to convert a click's canvas position into a true (G, S)
# phasor coordinate via phasor_pixel_to_gs, and back via phasor_gs_to_pixel
# when rendering a committed ROI's outline into a freshly exported plot.
PHASOR_DETAIL_SIZE = (940, 860)


class ImageView(ttk.Frame):
    DRAG_THRESHOLD_PX = 5
    MIN_ZOOM = 1.0
    MAX_ZOOM = 16.0
    ZOOM_STEP = 1.25

    def __init__(self, parent: tk.Misc, title: str, subtitle: str = "", selection_callback=None) -> None:
        super().__init__(parent, style="Panel.TFrame", padding=(10, 9))
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        self.title_label = ttk.Label(self, text=title, style="PanelTitle.TLabel")
        self.title_label.grid(row=0, column=0, sticky="w")
        self.subtitle_var = tk.StringVar(value=subtitle)
        ttk.Label(self, textvariable=self.subtitle_var, style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(0, 7))
        self.canvas = tk.Canvas(
            self,
            bg="#07101d",
            highlightthickness=0,
            cursor="crosshair" if selection_callback is not None else "arrow",
        )
        self.canvas.grid(row=2, column=0, sticky="nsew")
        self._source: Image.Image | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._display_box = (0, 0, 1, 1)
        self._view_box_norm = (0.0, 0.0, 1.0, 1.0)
        self._zoom = 1.0
        self._pan_center = (0.5, 0.5)
        self._zoom_badge_bbox: tuple[float, float, float, float] | None = None
        self._pan_drag_start_canvas: tuple[int, int] | None = None
        self._pan_drag_start_center: tuple[float, float] | None = None
        self._panning = False
        self._selection_callback = selection_callback
        self._selection: tuple[float, float, float, float] | None = None
        self._committed_selection: tuple[float, float, float, float] | None = None
        self._drag_start: tuple[float, float] | None = None
        self._press_canvas: tuple[int, int] | None = None
        self._dragging_roi = False
        self._points: list[tuple[float, float, str, str]] = []
        self._shapes: list[tuple[str, list[tuple[float, float]], str, str]] = []
        self._row_overlay: list[tuple[float, float]] = []
        self._column_overlay: list[tuple[float, float]] = []
        self._interactive = selection_callback is not None
        self.canvas.bind("<Configure>", self._render)
        self.canvas.bind("<ButtonPress-1>", self._selection_started)
        self.canvas.bind("<B1-Motion>", self._selection_dragged)
        self.canvas.bind("<ButtonRelease-1>", self._selection_finished)
        self.canvas.bind("<ButtonPress-3>", self._pan_or_clear_started)
        self.canvas.bind("<B3-Motion>", self._pan_or_clear_dragged)
        self.canvas.bind("<ButtonRelease-3>", self._pan_or_clear_finished)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", self._on_mouse_wheel)
        self.canvas.bind("<Button-5>", self._on_mouse_wheel)

    def set_image(self, image: Image.Image | None, subtitle: str | None = None) -> None:
        self._cancel_drag()
        self._source = image.copy() if image is not None else None
        if subtitle is not None:
            self.subtitle_var.set(subtitle)
        self._render()

    def set_selection(
        self,
        selection: tuple[float, float, float, float] | None,
        *,
        committed: bool = True,
    ) -> None:
        if selection is None:
            self._cancel_drag()
        self._selection = selection
        if committed:
            self._committed_selection = selection
        self._draw_selection()

    def set_points(self, points: list[tuple[float, float, str, str]]) -> None:
        self._points = list(points)
        self._draw_points()

    def set_shapes(self, shapes: list[tuple[str, list[tuple[float, float]], str, str]]) -> None:
        """Persistent shape overlays (phasor-plot ROI drawing; see gui.py's
        phasor-ROI controller). Each entry is ``(kind, points_norm, colour,
        label)``: ``kind`` is "rectangle"/"ellipse" (exactly two opposite
        bounding-box corners) or "freeform" (an arbitrary open polygon path,
        closed automatically); ``points_norm`` are (nx, ny) fractions of the
        full source image, same convention as everything else in this class.
        A pure function of this state -- callers own the shape list (add/
        undo/clear/live-preview-while-dragging) and just call this again.
        """

        self._shapes = list(shapes)
        self._draw_shapes()

    def set_row_overlay(self, bands: list[tuple[float, float]]) -> None:
        self._row_overlay = list(bands)
        self._draw_row_overlay()

    def set_column_overlay(self, bands: list[tuple[float, float]]) -> None:
        self._column_overlay = list(bands)
        self._draw_column_overlay()

    def set_interactive(self, interactive: bool) -> None:
        self._interactive = bool(interactive and self._selection_callback is not None)
        self.canvas.configure(cursor="crosshair" if self._interactive else "arrow")
        if not self._interactive:
            self._cancel_drag()

    def _cancel_drag(self) -> None:
        self._drag_start = None
        self._press_canvas = None
        self._dragging_roi = False
        self._selection = self._committed_selection
        self._pan_drag_start_canvas = None
        self._pan_drag_start_center = None
        self._panning = False
        try:
            if self.canvas.grab_current() is self.canvas:
                self.canvas.grab_release()
        except tk.TclError:
            pass

    # -- Zoom/pan geometry -------------------------------------------------
    # Everything below expresses positions as a fraction (0..1) of the full
    # source image, exactly as the rest of this class already did before
    # zoom existed. `_display_box` is the canvas-pixel box the *currently
    # visible crop* occupies; `_view_box_norm` is which fraction of the full
    # image that crop covers. At zoom == 1.0 the crop is the whole image, so
    # this is a strict superset of the previous (unzoomed) behaviour.

    def _clamp_center(self, cx: float, cy: float) -> tuple[float, float]:
        span = 1.0 / self._zoom
        if span >= 1.0:
            return 0.5, 0.5
        half = span / 2.0
        return (min(1.0 - half, max(half, cx)), min(1.0 - half, max(half, cy)))

    def _zoom_view_box(self) -> tuple[float, float, float, float]:
        span = 1.0 / self._zoom
        half = span / 2.0
        cx, cy = self._clamp_center(*self._pan_center)
        return (cx - half, cy - half, cx + half, cy + half)

    def _image_to_canvas(self, nx: float, ny: float) -> tuple[float, float]:
        x, y, width, height = self._display_box
        vx0, vy0, vx1, vy1 = self._view_box_norm
        span_x = max(vx1 - vx0, 1e-9)
        span_y = max(vy1 - vy0, 1e-9)
        return (x + (nx - vx0) / span_x * width, y + (ny - vy0) / span_y * height)

    def _canvas_to_image(self, event_x: float, event_y: float, clamp: bool = False) -> tuple[float, float] | None:
        x, y, width, height = self._display_box
        if width <= 0 or height <= 0:
            return None
        if not clamp and not (x <= event_x < x + width and y <= event_y < y + height):
            return None
        rel_x = min(1.0, max(0.0, (event_x - x) / width))
        rel_y = min(1.0, max(0.0, (event_y - y) / height))
        vx0, vy0, vx1, vy1 = self._view_box_norm
        return (vx0 + rel_x * (vx1 - vx0), vy0 + rel_y * (vy1 - vy0))

    def _point_in_zoom_badge(self, x: float, y: float) -> bool:
        if self._zoom_badge_bbox is None:
            return False
        x0, y0, x1, y1 = self._zoom_badge_bbox
        return x0 <= x <= x1 and y0 <= y <= y1

    def _reset_zoom(self) -> None:
        self._zoom = 1.0
        self._pan_center = (0.5, 0.5)
        self._render()

    def _on_mouse_wheel(self, event: tk.Event) -> None:
        if self._source is None or self._point_in_zoom_badge(event.x, event.y):
            return
        num = getattr(event, "num", None)
        if num == 4:
            direction = 1
        elif num == 5:
            direction = -1
        else:
            direction = 1 if getattr(event, "delta", 0) > 0 else -1
        new_zoom = self._zoom * (self.ZOOM_STEP if direction > 0 else 1.0 / self.ZOOM_STEP)
        new_zoom = min(self.MAX_ZOOM, max(self.MIN_ZOOM, new_zoom))
        if abs(new_zoom - self._zoom) < 1e-9:
            return
        anchor = self._canvas_to_image(event.x, event.y, clamp=True)
        self._zoom = new_zoom
        if anchor is not None:
            self._pan_center = self._clamp_center(*anchor)
        self._render()

    def _pan_or_clear_started(self, event: tk.Event) -> None:
        if self._source is None:
            return
        self._pan_drag_start_canvas = (int(event.x), int(event.y))
        self._pan_drag_start_center = self._pan_center
        self._panning = False

    def _pan_or_clear_dragged(self, event: tk.Event) -> None:
        if self._pan_drag_start_canvas is None or self._pan_drag_start_center is None:
            return
        dx = int(event.x) - self._pan_drag_start_canvas[0]
        dy = int(event.y) - self._pan_drag_start_canvas[1]
        if not self._panning and dx * dx + dy * dy < self.DRAG_THRESHOLD_PX**2:
            return
        self._panning = True
        _, _, width, height = self._display_box
        if width <= 0 or height <= 0:
            return
        vx0, vy0, vx1, vy1 = self._view_box_norm
        start_cx, start_cy = self._pan_drag_start_center
        new_cx = start_cx - (dx / width) * (vx1 - vx0)
        new_cy = start_cy - (dy / height) * (vy1 - vy0)
        self._pan_center = self._clamp_center(new_cx, new_cy)
        self._render()

    def _pan_or_clear_finished(self, event: tk.Event) -> None:
        was_panning = self._panning
        self._pan_drag_start_canvas = None
        self._pan_drag_start_center = None
        self._panning = False
        if not was_panning:
            self._selection_cleared(event)

    def _draw_zoom_badge(self, canvas_width: int) -> None:
        self.canvas.delete("zoom-badge")
        if self._zoom <= self.MIN_ZOOM + 1e-6:
            self._zoom_badge_bbox = None
            return
        text_id = self.canvas.create_text(
            canvas_width - 10,
            10,
            text=f"{self._zoom * 100:.0f}%  · click to reset",
            anchor="ne",
            fill="#07101d",
            font=("Segoe UI", 9, "bold"),
            tags="zoom-badge",
        )
        box = self.canvas.bbox(text_id)
        if box is None:
            self._zoom_badge_bbox = None
            return
        pad = 5
        bg_box = (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad)
        background = self.canvas.create_rectangle(*bg_box, fill="#facc15", outline="", tags="zoom-badge")
        self.canvas.tag_lower(background, text_id)
        self._zoom_badge_bbox = bg_box

    def _render(self, _event=None) -> None:
        if _event is not None:
            was_roi_drag = self._drag_start is not None
            if was_roi_drag or self._pan_drag_start_canvas is not None:
                self._cancel_drag()
            if was_roi_drag and self._selection_callback is not None:
                self._selection_callback("cancel_preview", None, False)
        self.canvas.delete("all")
        width = max(self.canvas.winfo_width(), 20)
        height = max(self.canvas.winfo_height(), 20)
        if self._source is None:
            self.canvas.create_text(
                width / 2,
                height / 2,
                text="No analysis yet",
                fill=MUTED,
                font=("Segoe UI", 11),
            )
            self._zoom_badge_bbox = None
            return
        available = (max(width - 8, 1), max(height - 8, 1))
        self._view_box_norm = self._zoom_view_box()
        vx0, vy0, vx1, vy1 = self._view_box_norm
        src_w, src_h = self._source.width, self._source.height
        left = min(src_w - 1, max(0, round(vx0 * src_w)))
        top = min(src_h - 1, max(0, round(vy0 * src_h)))
        right = min(src_w, max(left + 1, round(vx1 * src_w)))
        bottom = min(src_h, max(top + 1, round(vy1 * src_h)))
        cropped = self._source.crop((left, top, right, bottom))
        scale = min(available[0] / cropped.width, available[1] / cropped.height)
        rendered = cropped.resize(
            (max(1, int(cropped.width * scale)), max(1, int(cropped.height * scale))),
            Image.Resampling.NEAREST if scale > 1 else Image.Resampling.LANCZOS,
        )
        x = (width - rendered.width) // 2
        y = (height - rendered.height) // 2
        self._photo = ImageTk.PhotoImage(rendered)
        self.canvas.create_image(x, y, image=self._photo, anchor="nw")
        self._display_box = (x, y, rendered.width, rendered.height)
        self._draw_row_overlay()
        self._draw_column_overlay()
        self._draw_selection()
        self._draw_points()
        self._draw_shapes()
        self._draw_zoom_badge(width)

    def _normalised_position(self, event: tk.Event, clamp: bool = False) -> tuple[float, float] | None:
        if self._source is None:
            return None
        return self._canvas_to_image(event.x, event.y, clamp=clamp)

    def _selection_started(self, event: tk.Event) -> None:
        if self._point_in_zoom_badge(event.x, event.y):
            self._reset_zoom()
            return
        if self._selection_callback is None or not self._interactive:
            return
        point = self._normalised_position(event)
        if point is None:
            return
        try:
            self.canvas.grab_set()
        except tk.TclError:
            return
        self._drag_start = point
        self._press_canvas = (int(event.x), int(event.y))
        self._dragging_roi = False

    def _selection_dragged(self, event: tk.Event) -> None:
        if self._drag_start is None or self._press_canvas is None:
            return
        dx = int(event.x) - self._press_canvas[0]
        dy = int(event.y) - self._press_canvas[1]
        if not self._dragging_roi and dx * dx + dy * dy < self.DRAG_THRESHOLD_PX**2:
            return
        self._dragging_roi = True
        point = self._normalised_position(event, clamp=True)
        if point is None:
            return
        self._selection = (self._drag_start[0], self._drag_start[1], point[0], point[1])
        self._draw_selection()
        self._selection_callback("roi", self._selection, False)

    def _selection_finished(self, event: tk.Event) -> None:
        if self._drag_start is None or self._selection_callback is None:
            return
        point = self._normalised_position(event, clamp=True) or self._drag_start
        anchor = self._drag_start
        press_canvas = self._press_canvas
        release_moved = False
        if press_canvas is not None:
            dx = int(event.x) - press_canvas[0]
            dy = int(event.y) - press_canvas[1]
            release_moved = dx * dx + dy * dy >= self.DRAG_THRESHOLD_PX**2
        # Some window systems can deliver a release without an intermediate
        # motion event. Recheck the complete gesture here so it cannot be
        # mistaken for a point click.
        dragging_roi = self._dragging_roi or release_moved
        selection = (anchor[0], anchor[1], point[0], point[1])
        self._drag_start = None
        self._press_canvas = None
        self._dragging_roi = False
        try:
            self.canvas.grab_release()
        except tk.TclError:
            pass
        if dragging_roi:
            self._selection = selection
            self._committed_selection = selection
            self._draw_selection()
            self._selection_callback("roi", selection, True)
        else:
            self._draw_selection()
            self._selection_callback("point", anchor, True)

    def _selection_cleared(self, _event: tk.Event) -> None:
        if self._selection_callback is not None and self._interactive:
            self._cancel_drag()
            self._selection_callback("clear_roi", None, True)

    def _draw_selection(self) -> None:
        self.canvas.delete("roi-overlay")
        if self._selection is None or self._source is None:
            return
        x0n, y0n, x1n, y1n = self._selection
        cx0, cy0 = self._image_to_canvas(x0n, y0n)
        cx1, cy1 = self._image_to_canvas(x1n, y1n)
        x0, x1 = sorted((cx0, cx1))
        y0, y1 = sorted((cy0, cy1))
        if x1 - x0 < 3 and y1 - y0 < 3:
            self.canvas.create_line(x0 - 7, y0, x0 + 7, y0, fill="#facc15", width=2, tags="roi-overlay")
            self.canvas.create_line(x0, y0 - 7, x0, y0 + 7, fill="#facc15", width=2, tags="roi-overlay")
            return
        self.canvas.create_rectangle(
            x0,
            y0,
            x1,
            y1,
            outline="#facc15",
            width=3,
            dash=(7, 3),
            tags="roi-overlay",
        )
        label = self.canvas.create_text(
            x0 + 5,
            y0 + 5,
            text="ROI",
            anchor="nw",
            fill="#07101d",
            font=("Segoe UI", 9, "bold"),
            tags="roi-overlay",
        )
        box = self.canvas.bbox(label)
        if box:
            background = self.canvas.create_rectangle(*box, fill="#facc15", outline="", tags="roi-overlay")
            self.canvas.tag_lower(background, label)

    def _draw_row_overlay(self) -> None:
        self.canvas.delete("row-map-overlay")
        if self._source is None or not self._row_overlay:
            return
        x, _y, width, _height = self._display_box
        for y0n, y1n in self._row_overlay:
            _, y0 = self._image_to_canvas(0.0, min(y0n, y1n))
            _, y1 = self._image_to_canvas(0.0, max(y0n, y1n))
            if y1 - y0 < 2:
                centre = (y0 + y1) / 2
                self.canvas.create_line(
                    x, centre, x + width, centre,
                    fill="#f472b6", width=2, dash=(5, 3), tags="row-map-overlay",
                )
            else:
                self.canvas.create_rectangle(
                    x, y0, x + width, y1,
                    fill="#f472b6", stipple="gray25", outline="#f472b6", width=1,
                    tags="row-map-overlay",
                )
        self.canvas.tag_raise("roi-overlay")
        self.canvas.tag_raise("point-overlay")

    def _draw_column_overlay(self) -> None:
        self.canvas.delete("column-map-overlay")
        if self._source is None or not self._column_overlay:
            return
        _x, y, _width, height = self._display_box
        for x0n, x1n in self._column_overlay:
            x0, _ = self._image_to_canvas(min(x0n, x1n), 0.0)
            x1, _ = self._image_to_canvas(max(x0n, x1n), 0.0)
            if x1 - x0 < 2:
                centre = (x0 + x1) / 2
                self.canvas.create_line(
                    centre, y, centre, y + height,
                    fill="#c084fc", width=2, dash=(5, 3), tags="column-map-overlay",
                )
            else:
                self.canvas.create_rectangle(
                    x0, y, x1, y + height,
                    fill="#c084fc", stipple="gray25", outline="#c084fc", width=1,
                    tags="column-map-overlay",
                )
        self.canvas.tag_raise("roi-overlay")
        self.canvas.tag_raise("point-overlay")

    def _draw_points(self) -> None:
        self.canvas.delete("point-overlay")
        if self._source is None:
            return
        x, _y, width, _height = self._display_box
        for nx, ny, label, colour in self._points:
            px, py = self._image_to_canvas(nx, ny)
            self.canvas.create_oval(
                px - 8, py - 8, px + 8, py + 8,
                fill="#07101d", outline="white", width=1, tags="point-overlay",
            )
            self.canvas.create_oval(
                px - 5, py - 5, px + 5, py + 5,
                fill=colour, outline=colour, tags="point-overlay",
            )
            anchor = "ne" if px > x + width - 120 else "nw"
            tx = px - 9 if anchor == "ne" else px + 9
            self.canvas.create_text(
                tx, py - 9, text=label, anchor=anchor, fill=colour,
                font=("Segoe UI", 9, "bold"), tags="point-overlay",
            )
        self.canvas.tag_raise("point-overlay")
        self.canvas.tag_raise("zoom-badge")

    def _draw_shapes(self) -> None:
        self.canvas.delete("shape-overlay")
        if self._source is None:
            return
        for kind, points_norm, colour, label in self._shapes:
            canvas_points = [self._image_to_canvas(nx, ny) for nx, ny in points_norm]
            if len(canvas_points) < 2:
                continue
            if kind == "freeform":
                flat = [coord for point in canvas_points for coord in point]
                self.canvas.create_line(
                    *flat, canvas_points[0][0], canvas_points[0][1],
                    fill=colour, width=2, tags="shape-overlay", joinstyle="round",
                )
                anchor_x, anchor_y = min(canvas_points, key=lambda point: point[1])
            else:
                (cx0, cy0), (cx1, cy1) = canvas_points[0], canvas_points[1]
                x0, x1 = sorted((cx0, cx1))
                y0, y1 = sorted((cy0, cy1))
                if kind == "ellipse":
                    self.canvas.create_oval(x0, y0, x1, y1, outline=colour, width=2, tags="shape-overlay")
                else:
                    self.canvas.create_rectangle(x0, y0, x1, y1, outline=colour, width=2, tags="shape-overlay")
                anchor_x, anchor_y = x0, y0
            if label:
                self.canvas.create_text(
                    anchor_x + 4, anchor_y - 9, text=label, anchor="sw", fill=colour,
                    font=("Segoe UI", 9, "bold"), tags="shape-overlay",
                )
        self.canvas.tag_raise("shape-overlay")
        self.canvas.tag_raise("zoom-badge")


@dataclass(slots=True)
class LiveFitSelection:
    """A pooled single-point/region selection for the single-point-mode live preview.

    Deliberately much smaller than roi.py's ``ROIAnalysis`` -- the live
    preview (limited to Phasor and FLIMLib -- reconvolution) only ever
    needs the pooled/integrated decay for one selection, and must also
    work before any full analysis has been run, when no ``ROIAnalysis``
    can exist yet.
    """

    bounds_xyxy: BoundsXYXY
    selected_pixels: int
    summed_decay: np.ndarray


class FLIMStudioApp:
    METHOD_LABELS = {
        "Phasor — phase lifetime": "phasor_phase",
        "Phasor — modulation lifetime": "phasor_modulation",
        "Mono-exponential fit": "mono_exponential",
        "First moment": "moment",
        "FLIMLib — reconvolution": "flimlib_reconvolution",
        "FLIMLib — global shared τ": "flimlib_global",
        "FLIMLib — apply trained global model": "flimlib_apply_global",
    }
    NOISE_LABELS = {
        "Poisson MLE": 0,
        "Poisson fit weighting": 1,
        "Poisson data weighting": 2,
        "Constant weighting": 3,
    }
    FIT_BACKGROUND_LABELS = {"Fit / free background": 0, "Fixed background": 1}
    DISTRIBUTION_MODEL_LABELS = {
        "Multi-exponential (discrete τ)": 0,
        "Stretched exponential (distribution)": 1,
        "Multi-stretched exponential (independent components)": 2,
    }
    IRF_MODE_LABELS = {"Global IRF": 0, "Per-pixel IRF": 1}
    PHASOR_ROI_DISPLAY_MODES = {
        "Original lifetime": "original",
        "Composite image": "composite",
        "Coloured regions": "coloured",
        "Composite coloured regions": "coloured_composite",
    }
    ROW_COORDINATE_LABELS = {
        "Direct image Y (0-based)": DIRECT_Y,
        "Original LabVIEW detector row (mirrored)": LABVIEW_MIRRORED,
    }

    def __init__(self, root: tk.Tk, start_with_demo: bool = True) -> None:
        self.root = root
        self.root.title("FLIM Studio — Python analysis")
        self.root.geometry("1480x900")
        self.root.minsize(1160, 720)
        self.root.configure(bg=BG)
        self._configure_style()

        self.dataset: FLIMDataset | None = None
        self.irf: FLIMDataset | None = None
        self.background: FLIMDataset | None = None
        self.result: AnalysisResult | None = None
        # Raw (unfitted) spatially-binned intensity, kept for the pre-analysis
        # thresholding preview -- see _show_raw_intensity()/_refresh_secondary_map().
        # Rebuilt whenever the loaded dataset or the spatial bin radius changes,
        # so thresholding is usable before the first "Analyze" run, not just after.
        self._raw_binned_intensity: np.ndarray | None = None
        self.roi_bounds: BoundsXYXY | None = None
        self.roi_analysis: ROIAnalysis | None = None
        self.selected_points: list[PointSelection] = []
        self.point_analyses: list[PointAnalysis] = []
        self._lifetime_image: Image.Image | None = None
        self._composite_image: Image.Image | None = None
        self._component_images: list[Image.Image] = []
        self._amplitude_component_images: list[Image.Image] = []
        self.queue_paths: list[Path] = []
        # Queue browsing cache: results processed via "Process queue…" so
        # clicking a queue item shows an already-computed result instantly
        # instead of only loading the raw file. Bounded to
        # queue_cache_limit_var results, oldest-viewed evicted first (an
        # OrderedDict doubling as an LRU: touching an entry moves it to the
        # end via move_to_end, so popitem(last=False) always drops the
        # least-recently-viewed one).
        self.queue_results: OrderedDict[Path, tuple[FLIMDataset, AnalysisResult]] = OrderedDict()
        self.busy = False
        self.demo_defaults_active = False

        self.status_var = tk.StringVar(value="Ready")
        self.dataset_var = tk.StringVar(value="No dataset loaded")
        # Set once _load_default_ini_at_startup() runs, near the end of
        # __init__ -- guards load_demo()'s own hardcoded threshold/background
        # overrides (tuned for the synthetic demo data specifically) so a
        # user's own default.ini stays authoritative instead of being
        # partly clobbered by them a moment later. See settings_ini.py.
        self._default_ini_loaded = False
        self.settings_preset_status_var = tk.StringVar(value="Using built-in defaults.")
        self.leica_root_var = tk.StringVar(value="")
        self.leica_interrogator_status_var = tk.StringVar(
            value=(
                "Leica Interrogator: using FLIM Studio's built-in .lif reader "
                "(set a folder below only if you also want the full Interrogator GUI)"
                if bundled_leica_root() is not None
                else "Leica Interrogator: not set — .lif files can be queued but not opened"
            )
        )
        self.leica_watch_status_var = tk.StringVar(value="Not watching for exports.")
        # Plain attributes, not Tk variables -- internal watcher state only.
        self._leica_watch_folder: Path | None = None
        self._leica_watch_seen: set[Path] = set()
        self._leica_watch_after_id: str | None = None
        self.queue_cache_limit_var = tk.StringVar(value="8")
        self.queue_cache_status_var = tk.StringVar(value="No queue results cached yet.")
        # When enabled, running either queue action ("Process queue…" or
        # "Process & export queue") pools every processed file's own phasor
        # (G, S) map into one combined cloud for the Phasor tab -- a ROI
        # drawn against that pooled cloud reflects the whole queue at once,
        # not just whichever single file happens to be on screen. Each
        # entry is (dataset_name, g_array, s_array); kept independently of
        # queue_results (which discards results entirely for the "export"
        # path, and evicts under memory pressure for the "cached" path) so
        # accumulation survives either.
        self.phasor_accumulate_queue_var = tk.BooleanVar(value=False)
        self.phasor_accumulated_points: list[tuple[str, np.ndarray, np.ndarray]] = []
        self.phasor_accumulate_status_var = tk.StringVar(value="Phasor accumulation off.")
        self.irf_var = tk.StringVar(value="IRF: none")
        self.irf_source_var = tk.StringVar(value="Loaded file")
        self.synthetic_irf_fwhm_var = tk.StringVar(value="-1")
        self.synthetic_irf_summary_var = tk.StringVar(value="")
        self.background_var = tk.StringVar(value="Background: none")
        self.acquisition_var = tk.StringVar(value="Acquisition bins: load a dataset")
        self.background_scale_mode_var = tk.StringVar(value="Auto from loops / time")
        self.background_multiplier_var = tk.StringVar(value="0.025")
        self.background_clip_var = tk.BooleanVar(value=True)
        self.background_scale_summary_var = tk.StringVar(value="No background loaded")
        self.selection_var = tk.StringVar(
            value="Click adds a comparison point; drag selects an ROI. Right-click clears the ROI."
        )
        self.roi_pixels_var = tk.StringVar(value="—")
        self.roi_mean_tau_var = tk.StringVar(value="—")
        self.roi_median_tau_var = tk.StringVar(value="—")
        self.roi_aggregate_tau_var = tk.StringVar(value="—")
        self.roi_photons_var = tk.StringVar(value="—")
        self.roi_decay_mode_var = tk.StringVar(value="Summed processed signal")
        self.single_point_mode_var = tk.BooleanVar(value=False)
        self.live_fit_busy = False
        self._live_synthetic_irf_cache: tuple | None = None
        self.single_point_last_bounds: BoundsXYXY | None = None
        self.phasor_rois: list[PhasorROI] = []
        self.phasor_roi_shape_var = tk.StringVar(value="Rectangle")
        self.phasor_roi_display_mode_var = tk.StringVar(value="Original lifetime")
        self.phasor_roi_show_invalid_var = tk.BooleanVar(value=True)
        self.phasor_roi_additive_overlap_var = tk.BooleanVar(value=False)
        self.phasor_roi_summary_var = tk.StringVar(value="No phasor ROIs drawn yet.")
        self.phasor_roi_remove_number_var = tk.StringVar(value="1")
        self.phasor_cluster_count_var = tk.StringVar(value="3")
        self.phasor_cluster_sigma_var = tk.StringVar(value="2.0")
        # Live in-progress drawing/rotating state for the Phasor tab's own
        # mouse handlers (see _bind_phasor_roi_drawing) -- kept here rather
        # than on ImageView since it's specific to this one tab's shape
        # tools, not a general ImageView capability. _phasor_draw_path is
        # the points captured since the left-button press: exactly 2
        # (start, live end) for rectangle/ellipse, an accumulating hand-
        # drawn path for freeform.
        self._phasor_draw_path: list[tuple[float, float]] = []
        self._phasor_rotate_index: int | None = None
        self._phasor_rotate_last_angle_deg = 0.0
        self._phasor_rotate_preview_deg = 0.0
        self.map_mode_var = tk.StringVar(value="Lifetime")
        self.intensity_log_var = tk.BooleanVar(value=True)
        self.point_summary_var = tk.StringVar(value="Click a map to add up to 8 comparison points.")
        self.roi_summary_var = tk.StringVar(value="Histogram scope: whole image")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.valid_var = tk.StringVar(value="—")
        self.mean_tau_var = tk.StringVar(value="—")
        self.global_tau_var = tk.StringVar(value="—")
        self.max_counts_var = tk.StringVar(value="—")
        self.mean_chi_var = tk.StringVar(value="—")

        self.dt_var = tk.StringVar(value="0.390625")
        self.rep_var = tk.StringVar(value="80")
        self.method_var = tk.StringVar(value="Phasor — phase lifetime")
        self.background_counts_var = tk.StringVar(value="0")
        self.phasor_threshold_var = tk.StringVar(value="1000")
        self.fit_threshold_var = tk.StringVar(value="1000")
        self.start_bin_var = tk.StringVar(value="-1")
        self.end_bin_var = tk.StringVar(value="-1")
        self.native_fit_bins_var = tk.StringVar(value="29")
        self.native_bin_width_var = tk.StringVar(value="0.4")
        self.native_irf_baseline_var = tk.StringVar(value="Per-pixel median")
        # "Auto" by default: a mismatch here silently scales every
        # reconvolution-fit lifetime by native_bin_width_ns/dt_ns, which is
        # almost never what anyone wants -- see _native_bin_width_mode_changed
        # and AnalysisSettings.normalized()'s own <=0-means-auto fallback,
        # which this mode maps onto (passing -1.0 in Auto mode) rather than
        # duplicating.
        self.native_bin_width_mode_var = tk.StringVar(value="Auto")
        self.native_bin_width_summary_var = tk.StringVar(value="")
        self.native_fit_workers_var = tk.StringVar(value="1")
        self.bin_var = tk.StringVar(value="0")
        self.smooth_var = tk.StringVar(value="0")
        self.offset_var = tk.StringVar(value="0")
        self.life_min_var = tk.StringVar(value="0.5")
        self.life_max_var = tk.StringVar(value="4.5")
        self.lifetime_scale_mode_var = tk.StringVar(value="Manual")
        self.lifetime_scale_summary_var = tk.StringVar(value="Manual range: 0.5–4.5 ns")
        self.hist_bins_var = tk.StringVar(value="50")
        self.detailed_hist_bins_var = tk.StringVar(value="1000")
        self.lifetime_lut_path_var = tk.StringVar(value="")
        self.lifetime_lut_inverse_var = tk.BooleanVar(value=False)
        self.lifetime_lut_remove_var = tk.BooleanVar(value=True)
        self.lifetime_lut_summary_var = tk.StringVar(value="Reverse Rainbow (DM).lut — original LabVIEW LUT")
        self.interpolate_var = tk.BooleanVar(value=False)
        self.row_map_enabled_var = tk.BooleanVar(value=False)
        self.row_map_rows_var = tk.StringVar(value="")
        self.row_map_radius_var = tk.StringVar(value="2")
        self.row_map_coordinate_var = tk.StringVar(value="Direct image Y (0-based)")
        self.row_map_overlay_var = tk.BooleanVar(value=True)
        self.row_map_summary_var = tk.StringVar(value="Detector-row map: none defined")
        self.row_map_name = "Custom detector-row map"
        self.row_map_source = ""
        self.row_map_sha256 = ""
        self.row_map_expected_source_height: int | None = None
        self._loading_row_map_controls = False
        self.column_map_enabled_var = tk.BooleanVar(value=False)
        self.column_map_columns_var = tk.StringVar(value="")
        self.column_map_radius_var = tk.StringVar(value="2")
        self.column_map_overlay_var = tk.BooleanVar(value=True)
        self.column_map_summary_var = tk.StringVar(value="Detector-column map: none defined")
        self.phasor_heatmap_var = tk.BooleanVar(value=True)
        self.phasor_heatmap_bins_var = tk.StringVar(value="400")
        self.live_threshold_var = tk.StringVar(value="1000")
        self.live_threshold_summary_var = tk.StringVar(value="Run analysis to preview fit omissions")
        self.model_order_var = tk.StringVar(value="1 — mono")
        self.fit_distribution_model_var = tk.StringVar(value="Multi-exponential (discrete τ)")
        self.method_model_hint_var = tk.StringVar(value="")
        self.fit_background_mode_var = tk.StringVar(value="Fixed background")
        self.fit_background_value_var = tk.StringVar(value="0")
        self.noise_model_var = tk.StringVar(value="Poisson fit weighting")
        self.irf_mode_var = tk.StringVar(value="Per-pixel IRF")
        self.irf_integrate_per_column_var = tk.BooleanVar(value=False)
        self.global_iterations_var = tk.StringVar(value="4")
        self.max_global_pixels_var = tk.StringVar(value="20000")
        self.initial_taus_var = tk.StringVar(value="")
        detected_dll = find_native_dll()
        self.native_dll_var = tk.StringVar(value=str(detected_dll) if detected_dll else "")
        self.native_status_var = tk.StringVar(
            value="Native backend found" if detected_dll else "Native backend not found"
        )
        self.global_model_path_var = tk.StringVar(value="")
        self.global_model_status_var = tk.StringVar(value="No trained model loaded")
        self.trained_model: TrainedGlobalModel | None = None
        for variable in (self.background_scale_mode_var, self.background_multiplier_var):
            variable.trace_add("write", lambda *_args: self._update_background_scale_summary())
        # These three also feed the pre-analysis intensity preview's own
        # background subtraction (see _show_raw_intensity) -- keep it live
        # as the user tunes background settings, same as _on_spatial_bin_setting_changed
        # already does for the spatial bin radius (whose own guard this
        # reuses: a no-op once a real "Analyze" result exists).
        for variable in (
            self.background_scale_mode_var,
            self.background_multiplier_var,
            self.background_counts_var,
            self.background_clip_var,
        ):
            variable.trace_add("write", lambda *_args: self._on_spatial_bin_setting_changed())

        self._build_header()
        self._build_body()
        self._build_status_bar()
        self._bind_shortcuts()
        self._load_default_ini_at_startup()
        if start_with_demo:
            self.root.after(120, self.load_demo)

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", background=BG, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("TFrame", background=BG)
        style.configure("Sidebar.TFrame", background=SIDEBAR)
        style.configure("Panel.TFrame", background=PANEL, relief="flat")
        style.configure("TLabel", background=BG, foreground=TEXT)
        style.configure("Sidebar.TLabel", background=SIDEBAR, foreground=TEXT)
        style.configure("Section.TLabel", background=SIDEBAR, foreground="#7dd3fc", font=("Segoe UI Semibold", 10))
        style.configure("Title.TLabel", background=SIDEBAR, foreground=TEXT, font=("Segoe UI Semibold", 17))
        style.configure("PanelTitle.TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI Semibold", 12))
        style.configure("Muted.TLabel", background=PANEL, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Metric.TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI Semibold", 17))
        style.configure("MetricName.TLabel", background=PANEL, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("TButton", padding=(10, 7), background=PANEL_ALT, foreground=TEXT, bordercolor=BORDER)
        style.map("TButton", background=[("active", "#213451"), ("disabled", "#111827")])
        style.configure("Primary.TButton", background="#0369a1", foreground="white", bordercolor="#0ea5e9", font=("Segoe UI Semibold", 10))
        style.map("Primary.TButton", background=[("active", "#0284c7"), ("disabled", "#1e3a4b")])
        style.configure("TEntry", fieldbackground="#0a1423", foreground=TEXT, insertcolor=TEXT, bordercolor=BORDER)
        style.configure("TCombobox", fieldbackground="#0a1423", background=PANEL_ALT, foreground=TEXT, arrowcolor=TEXT)
        style.map("TCombobox", fieldbackground=[("readonly", "#0a1423")], foreground=[("readonly", TEXT)])
        # TSpinbox never got its own style rule (unlike TEntry/TCombobox
        # above) -- it silently fell back to the theme's default white
        # field while still inheriting this app's white "." foreground,
        # rendering every Spinbox's digits as white-on-white and
        # effectively invisible (the "ROI #" field, and the newer
        # "Clusters:"/"ROI size" fields, all share this one style).
        style.configure(
            "TSpinbox", fieldbackground="#0a1423", background=PANEL_ALT, foreground=TEXT,
            arrowcolor=TEXT, insertcolor=TEXT, bordercolor=BORDER,
        )
        style.map("TSpinbox", fieldbackground=[("readonly", "#0a1423")], foreground=[("readonly", TEXT)])
        style.configure("TCheckbutton", background=SIDEBAR, foreground=TEXT)
        style.map("TCheckbutton", background=[("active", SIDEBAR)])
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=SIDEBAR, foreground=MUTED, padding=(16, 9), borderwidth=0)
        style.map("TNotebook.Tab", background=[("selected", PANEL)], foreground=[("selected", TEXT)])
        style.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor="#07101d", bordercolor="#07101d")
        style.configure("Vertical.TScrollbar", background=PANEL_ALT, troughcolor=SIDEBAR, arrowcolor=TEXT)

    def _build_header(self) -> None:
        # No fixed height / pack_propagate(False) here on purpose: a hardcoded
        # pixel height clips the title+subtitle whenever the system's font
        # metrics/DPI scaling render them even slightly taller than expected
        # (this cut the subtitle off on at least one real machine). Letting
        # the frame size itself from its content is robust to that on any
        # display, at the cost of the header's height being font-dependent
        # rather than a fixed constant -- an unnoticeable difference in
        # practice, and worth it over silently clipped text.
        header = tk.Frame(self.root, bg=SIDEBAR)
        header.pack(fill=X)
        title_frame = tk.Frame(header, bg=SIDEBAR)
        title_frame.pack(side=LEFT, padx=24, pady=16)
        tk.Label(title_frame, text="FLIM Studio", bg=SIDEBAR, fg=TEXT, font=("Segoe UI Semibold", 20)).pack(anchor="w")
        tk.Label(
            title_frame,
            text="Time-resolved image processing & analysis",
            bg=SIDEBAR,
            fg=MUTED,
            font=("Segoe UI", 9),
        ).pack(anchor="w")
        actions = tk.Frame(header, bg=SIDEBAR)
        actions.pack(side=RIGHT, padx=22, pady=16)
        ttk.Button(actions, text="Open files", command=self.open_files).pack(side=LEFT, padx=4)
        ttk.Button(actions, text="Demo data", command=self.load_demo).pack(side=LEFT, padx=4)
        self.header_analyze = ttk.Button(actions, text="Run analysis", command=self.run_current, style="Primary.TButton")
        self.header_analyze.pack(side=LEFT, padx=(10, 4))
        ttk.Button(actions, text="Export", command=self.export_current).pack(side=LEFT, padx=4)

    def _build_body(self) -> None:
        body = ttk.Frame(self.root)
        body.pack(fill=BOTH, expand=True)
        sidebar_shell = ttk.Frame(body, style="Sidebar.TFrame", width=305)
        sidebar_shell.pack(side=LEFT, fill=Y)
        sidebar_shell.pack_propagate(False)
        self._build_sidebar(sidebar_shell)
        content = ttk.Frame(body, padding=(14, 12))
        content.pack(side=RIGHT, fill=BOTH, expand=True)
        self._build_content(content)
        self._method_model_hint_changed()
        self._native_bin_width_mode_changed()

    def _build_sidebar(self, parent: ttk.Frame) -> None:
        canvas = tk.Canvas(parent, bg=SIDEBAR, highlightthickness=0, width=286)
        # Referenced by _disable_combobox_wheel (and the two standalone
        # sidebar Comboboxes below it) so a scroll of the mouse wheel over
        # a dropdown still scrolls this panel instead of silently
        # cycling the dropdown's own value -- see that method's docstring.
        self._sidebar_canvas = canvas
        scrollbar = ttk.Scrollbar(parent, orient=VERTICAL, command=canvas.yview)
        inner = ttk.Frame(canvas, style="Sidebar.TFrame", padding=(16, 14))
        window = canvas.create_window((0, 0), window=inner, anchor="nw", width=285)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)
        inner.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window, width=e.width))

        def _on_sidebar_wheel(event: tk.Event) -> None:
            # bind_all (not just canvas.bind) so scrolling still works
            # while hovering any of the sidebar's own child widgets
            # (comboboxes, entries, ...) -- a plain canvas.bind only ever
            # fires for the canvas's own background, not its children.
            # The cost of bind_all is that it fires for *every* widget in
            # the whole application, not just this one, so the handler
            # itself checks the cursor is actually within the sidebar's
            # own on-screen bounds before scrolling it -- otherwise a
            # scroll anywhere else in the window (the image views, a
            # dialog, ...) would also scroll the sidebar underneath it.
            x0, y0 = canvas.winfo_rootx(), canvas.winfo_rooty()
            x1, y1 = x0 + canvas.winfo_width(), y0 + canvas.winfo_height()
            if x0 <= event.x_root < x1 and y0 <= event.y_root < y1:
                canvas.yview_scroll(int(-event.delta / 120), "units")

        canvas.bind_all("<MouseWheel>", _on_sidebar_wheel)

        ttk.Label(inner, text="SETTINGS PRESET", style="Section.TLabel").pack(anchor="w")
        preset_row_1 = ttk.Frame(inner, style="Sidebar.TFrame")
        preset_row_1.pack(fill=X, pady=(8, 3))
        ttk.Button(preset_row_1, text="Load…", command=self._load_settings_ini_dialog).pack(
            side=LEFT, fill=X, expand=True, padx=(0, 3)
        )
        ttk.Button(preset_row_1, text="Save as…", command=self._save_settings_ini_dialog).pack(
            side=LEFT, fill=X, expand=True, padx=3
        )
        preset_row_2 = ttk.Frame(inner, style="Sidebar.TFrame")
        preset_row_2.pack(fill=X, pady=(0, 3))
        ttk.Button(preset_row_2, text="Save as default", command=self._save_settings_as_default).pack(
            side=LEFT, fill=X, expand=True, padx=(0, 3)
        )
        ttk.Button(preset_row_2, text="Reset", command=self._reset_settings_to_defaults).pack(
            side=LEFT, fill=X, expand=True, padx=3
        )
        ttk.Label(
            inner,
            text=(
                f"Every choice below can be saved to/loaded from a .ini file. "
                f"\"{DEFAULT_INI_NAME}\", next to the app, loads automatically on startup if present."
            ),
            style="Sidebar.TLabel",
            wraplength=250,
            justify="left",
        ).pack(anchor="w", pady=(2, 0))
        ttk.Label(
            inner, textvariable=self.settings_preset_status_var, style="Sidebar.TLabel", wraplength=250
        ).pack(anchor="w", pady=(2, 10))
        ttk.Separator(inner).pack(fill=X, pady=(0, 8))

        ttk.Label(inner, text="DATA SOURCES", style="Section.TLabel").pack(anchor="w")
        ttk.Button(inner, text="Open decay files…", command=self.open_files).pack(fill=X, pady=(8, 4))
        ttk.Button(inner, text="Open a folder…", command=self.open_folder).pack(fill=X, pady=4)
        leica_row = ttk.Frame(inner, style="Sidebar.TFrame")
        leica_row.pack(fill=X, pady=(4, 0))
        ttk.Button(
            leica_row, text="Leica Interrogator…", command=self._choose_leica_interrogator
        ).pack(side=LEFT, fill=X, expand=True, padx=(0, 3))
        ttk.Button(
            leica_row, text="Full GUI…", command=self._launch_leica_full_gui
        ).pack(side=LEFT)
        ttk.Label(
            inner, textvariable=self.leica_interrogator_status_var, style="Sidebar.TLabel", wraplength=250
        ).pack(anchor="w", pady=(3, 3))
        leica_watch_row = ttk.Frame(inner, style="Sidebar.TFrame")
        leica_watch_row.pack(fill=X)
        ttk.Button(
            leica_watch_row, text="Watch for exports…", command=self._choose_leica_watch_folder
        ).pack(side=LEFT, fill=X, expand=True, padx=(0, 3))
        ttk.Button(leica_watch_row, text="Stop", command=self._stop_leica_watch).pack(side=LEFT)
        ttk.Label(
            inner, textvariable=self.leica_watch_status_var, style="Sidebar.TLabel", wraplength=250
        ).pack(anchor="w", pady=(3, 5))
        queue_frame = tk.Frame(inner, bg=SIDEBAR)
        queue_frame.pack(fill=X, pady=(5, 7))
        self.queue_list = tk.Listbox(
            queue_frame,
            height=4,
            bg="#091321",
            fg=TEXT,
            selectbackground="#075985",
            selectforeground="white",
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=BORDER,
            font=("Segoe UI", 9),
            selectmode=tk.EXTENDED,
        )
        self.queue_list.pack(fill=X)
        self.queue_list.bind("<<ListboxSelect>>", self._queue_selected)
        queue_edit_row = ttk.Frame(inner, style="Sidebar.TFrame")
        queue_edit_row.pack(fill=X, pady=(3, 0))
        ttk.Button(
            queue_edit_row, text="Remove selected", command=self._remove_selected_from_queue
        ).pack(side=LEFT, fill=X, expand=True, padx=(0, 3))
        ttk.Button(queue_edit_row, text="Clear queue", command=self._clear_queue).pack(side=LEFT)
        queue_process_row = ttk.Frame(inner, style="Sidebar.TFrame")
        queue_process_row.pack(fill=X, pady=(4, 0))
        ttk.Button(
            queue_process_row, text="Process queue…", command=self.process_queue_cached
        ).pack(side=LEFT, fill=X, expand=True, padx=(0, 3))
        ttk.Label(queue_process_row, text="Keep in memory", style="Sidebar.TLabel").pack(side=LEFT, padx=(3, 3))
        ttk.Entry(queue_process_row, textvariable=self.queue_cache_limit_var, width=4).pack(side=LEFT)
        ttk.Label(
            inner, textvariable=self.queue_cache_status_var, style="Sidebar.TLabel", wraplength=250
        ).pack(anchor="w", pady=(3, 0))
        phasor_accumulate_row = ttk.Frame(inner, style="Sidebar.TFrame")
        phasor_accumulate_row.pack(fill=X, pady=(4, 0))
        ttk.Checkbutton(
            phasor_accumulate_row,
            text="Accumulate phasor across queue",
            variable=self.phasor_accumulate_queue_var,
            command=self._phasor_accumulate_toggled,
        ).pack(side=LEFT)
        ttk.Button(
            phasor_accumulate_row, text="Clear", command=self._clear_phasor_accumulation, width=6
        ).pack(side=RIGHT)
        ttk.Label(
            inner, textvariable=self.phasor_accumulate_status_var, style="Sidebar.TLabel", wraplength=250
        ).pack(anchor="w", pady=(0, 5))
        ttk.Label(inner, textvariable=self.dataset_var, style="Sidebar.TLabel", wraplength=250).pack(anchor="w", pady=(0, 5))
        ref_row = ttk.Frame(inner, style="Sidebar.TFrame")
        ref_row.pack(fill=X, pady=2)
        ttk.Button(ref_row, text="Load IRF", command=self.open_irf).pack(side=LEFT, fill=X, expand=True, padx=(0, 3))
        ttk.Button(ref_row, text="Background", command=self.open_background).pack(side=LEFT, fill=X, expand=True, padx=(3, 0))
        ttk.Label(inner, textvariable=self.irf_var, style="Sidebar.TLabel", wraplength=250).pack(anchor="w", pady=(4, 0))
        irf_source_row = ttk.Frame(inner, style="Sidebar.TFrame")
        irf_source_row.pack(fill=X, pady=(3, 0))
        ttk.Label(irf_source_row, text="IRF source", style="Sidebar.TLabel").pack(side=LEFT)
        irf_source_combo = ttk.Combobox(
            irf_source_row,
            textvariable=self.irf_source_var,
            values=["Loaded file", "Synthetic (Gaussian)"],
            state="readonly",
            width=17,
        )
        irf_source_combo.pack(side=LEFT, padx=(6, 6))
        self._disable_combobox_wheel(irf_source_combo, self._sidebar_canvas)
        ttk.Label(irf_source_row, text="FWHM (ns, -1=auto)", style="Sidebar.TLabel").pack(side=LEFT)
        ttk.Entry(irf_source_row, textvariable=self.synthetic_irf_fwhm_var, width=6).pack(side=LEFT, padx=(4, 0))
        ttk.Label(
            inner,
            text=(
                "Synthetic (Gaussian): models the instrument response as a Gaussian pulse when no "
                "measured IRF is available or a loaded one doesn't match this dataset's timing -- "
                "an approximation (Becker 2005; O'Connor & Phillips 1984), not a substitute for a "
                "real IRF. FWHM auto-estimated from this decay's own rise time unless pinned above."
            ),
            style="Sidebar.TLabel",
            wraplength=250,
            justify="left",
        ).pack(anchor="w", pady=(2, 0))
        ttk.Label(
            inner, textvariable=self.synthetic_irf_summary_var, style="Sidebar.TLabel", wraplength=250
        ).pack(anchor="w", pady=(2, 0))
        ttk.Label(inner, textvariable=self.background_var, style="Sidebar.TLabel", wraplength=250).pack(anchor="w")
        ttk.Label(inner, textvariable=self.acquisition_var, style="Sidebar.TLabel", wraplength=250).pack(anchor="w", pady=(3, 12))

        ttk.Separator(inner).pack(fill=X, pady=3)
        ttk.Label(inner, text="TIMING & METHOD", style="Section.TLabel").pack(anchor="w", pady=(10, 3))
        self._field(inner, "Time-bin width (ns)", self.dt_var)
        self._field(inner, "Laser repetition (MHz)", self.rep_var)
        self._combo(inner, "Lifetime method", self.method_var, list(self.METHOD_LABELS))
        for variable in (self.method_var, self.model_order_var, self.fit_distribution_model_var):
            variable.trace_add("write", self._method_model_hint_changed)

        ttk.Label(inner, text="PREPROCESSING", style="Section.TLabel").pack(anchor="w", pady=(13, 3))
        self._combo(
            inner,
            "Background scaling",
            self.background_scale_mode_var,
            ["Auto from loops / time", "Manual multiplier"],
        )
        self._field(inner, "Background multiplier (manual / auto fallback)", self.background_multiplier_var)
        ttk.Label(
            inner,
            textvariable=self.background_scale_summary_var,
            style="Sidebar.TLabel",
            wraplength=250,
        ).pack(anchor="w", pady=(0, 3))
        self._field(inner, "Additional scalar background / time bin", self.background_counts_var)
        ttk.Checkbutton(
            inner,
            text="Clip negative bins to zero after subtraction",
            variable=self.background_clip_var,
        ).pack(anchor="w", pady=(2, 0))
        ttk.Label(
            inner,
            text="Untick when the background has the same exposure as the sample (multiplier ~1, e.g. "
            "Swarm): clipping noisy bins turns leftover noise into a positive bias, so hot rows stay visible.",
            style="Sidebar.TLabel",
            wraplength=250,
        ).pack(anchor="w", pady=(0, 3))
        threshold_row = ttk.Frame(inner, style="Sidebar.TFrame")
        threshold_row.pack(fill=X, pady=2)
        self._compact_field(
            threshold_row, "Phasor min total counts", self.phasor_threshold_var
        ).pack(side=LEFT, fill=X, expand=True, padx=(0, 3))
        self._compact_field(
            threshold_row, "Fit/native min total counts", self.fit_threshold_var
        ).pack(side=LEFT, fill=X, expand=True, padx=(3, 0))
        self._field(inner, "Timing offset (bins)", self.offset_var)
        self._combo(inner, "Spatial bin radius (0=1x1, 1=3x3, 2=5x5)", self.bin_var, ["0", "1", "2", "3", "4"])
        # Keeps the pre-analysis thresholding preview (intensity_view plus
        # the Intensity/Intensity-with-threshold secondary map) in step with
        # this setting -- before the first "Analyze" run there's no fitted
        # result to re-bin from, so _show_raw_intensity recomputes its own
        # raw binned intensity straight from the loaded dataset instead.
        # After a run this is a no-op: changing the radius only affects the
        # *next* run, same as every other analysis setting.
        self.bin_var.trace_add("write", self._on_spatial_bin_setting_changed)
        self._combo(inner, "Smoothing passes", self.smooth_var, ["0", "1", "2", "3"])
        ttk.Checkbutton(inner, text="Auto-detect extreme pixels", variable=self.interpolate_var).pack(anchor="w", pady=(6, 0))

        row_map_box = ttk.Frame(inner, style="Sidebar.TFrame")
        row_map_box.pack(fill=X, pady=(9, 0))
        ttk.Checkbutton(
            row_map_box,
            text="Apply defined detector-row map",
            variable=self.row_map_enabled_var,
        ).pack(anchor="w")
        self._field(row_map_box, "Row IDs (comma/ranges)", self.row_map_rows_var)
        self._combo(
            row_map_box,
            "Row coordinate convention",
            self.row_map_coordinate_var,
            list(self.ROW_COORDINATE_LABELS),
        )
        self._field(row_map_box, "Donor radius", self.row_map_radius_var)
        map_buttons = ttk.Frame(row_map_box, style="Sidebar.TFrame")
        map_buttons.pack(fill=X, pady=(3, 2))
        ttk.Button(map_buttons, text="Load map…", command=self._load_row_map).pack(side=LEFT, fill=X, expand=True)
        ttk.Button(map_buttons, text="Save map…", command=self._save_row_map).pack(side=LEFT, fill=X, expand=True, padx=3)
        ttk.Button(map_buttons, text="VI preset", command=self._use_labview_row_preset).pack(side=LEFT, fill=X, expand=True)
        map_buttons_2 = ttk.Frame(row_map_box, style="Sidebar.TFrame")
        map_buttons_2.pack(fill=X, pady=(1, 0))
        ttk.Button(map_buttons_2, text="Preview / validate", command=self._preview_row_map).pack(
            side=LEFT, fill=X, expand=True, padx=(0, 4)
        )
        ttk.Checkbutton(
            map_buttons_2,
            text="Show overlay",
            variable=self.row_map_overlay_var,
            command=self._row_overlay_changed,
        ).pack(side=LEFT)
        ttk.Label(
            row_map_box,
            textvariable=self.row_map_summary_var,
            style="Sidebar.TLabel",
            wraplength=250,
        ).pack(anchor="w", pady=(4, 0))
        for variable in (
            self.row_map_rows_var,
            self.row_map_radius_var,
            self.row_map_coordinate_var,
        ):
            variable.trace_add("write", self._row_map_control_edited)

        column_map_box = ttk.Frame(inner, style="Sidebar.TFrame")
        column_map_box.pack(fill=X, pady=(9, 0))
        ttk.Checkbutton(
            column_map_box,
            text="Apply defined detector-column map",
            variable=self.column_map_enabled_var,
        ).pack(anchor="w")
        self._field(column_map_box, "Column X positions (comma/ranges)", self.column_map_columns_var)
        self._field(column_map_box, "Column donor radius", self.column_map_radius_var)
        ttk.Checkbutton(
            column_map_box,
            text="Show column overlay",
            variable=self.column_map_overlay_var,
            command=self._column_overlay_changed,
        ).pack(anchor="w")
        ttk.Label(
            column_map_box,
            textvariable=self.column_map_summary_var,
            style="Sidebar.TLabel",
            wraplength=250,
        ).pack(anchor="w", pady=(3, 0))
        for variable in (self.column_map_columns_var, self.column_map_radius_var):
            variable.trace_add("write", self._column_map_control_edited)

        phasor_display = ttk.Frame(inner, style="Sidebar.TFrame")
        phasor_display.pack(fill=X, pady=(9, 0))
        ttk.Checkbutton(
            phasor_display,
            text="Phasor density heatmap",
            variable=self.phasor_heatmap_var,
            command=self._refresh_phasors,
        ).pack(side=LEFT)
        self._compact_field(
            phasor_display, "Heatmap bins", self.phasor_heatmap_bins_var
        ).pack(side=RIGHT, fill=X, expand=True, padx=(8, 0))

        ttk.Label(inner, text="FIT & DISPLAY", style="Section.TLabel").pack(anchor="w", pady=(13, 3))
        fit_row = ttk.Frame(inner, style="Sidebar.TFrame")
        fit_row.pack(fill=X, pady=2)
        self._compact_field(fit_row, "Fit start (-1 auto; VI 5)", self.start_bin_var).pack(side=LEFT, fill=X, expand=True, padx=(0, 3))
        self._compact_field(fit_row, "Fit stop (-1 auto; VI 24)", self.end_bin_var).pack(side=LEFT, fill=X, expand=True, padx=(3, 0))
        life_row = ttk.Frame(inner, style="Sidebar.TFrame")
        life_row.pack(fill=X, pady=2)
        self._compact_combo(
            life_row, "Lifetime scaling", self.lifetime_scale_mode_var, ["Auto", "Manual"]
        ).pack(side=LEFT, fill=X, expand=True, padx=(0, 3))
        self._compact_field(life_row, "Min τ (ns)", self.life_min_var).pack(side=LEFT, fill=X, expand=True, padx=(0, 3))
        self._compact_field(life_row, "Max τ (ns)", self.life_max_var).pack(side=LEFT, fill=X, expand=True, padx=(3, 0))
        ttk.Label(
            inner, textvariable=self.lifetime_scale_summary_var,
            style="Sidebar.TLabel", wraplength=250,
        ).pack(anchor="w", pady=(0, 2))
        for variable in (self.lifetime_scale_mode_var, self.life_min_var, self.life_max_var):
            variable.trace_add("write", self._lifetime_scale_changed)
        histogram_row = ttk.Frame(inner, style="Sidebar.TFrame")
        histogram_row.pack(fill=X, pady=2)
        self._compact_field(histogram_row, "Overview histogram intervals", self.hist_bins_var).pack(
            side=LEFT, fill=X, expand=True, padx=(0, 3)
        )
        self._compact_field(histogram_row, "Detailed histogram bins", self.detailed_hist_bins_var).pack(
            side=LEFT, fill=X, expand=True, padx=(3, 0)
        )
        lut_row = ttk.Frame(inner, style="Sidebar.TFrame")
        lut_row.pack(fill=X, pady=(5, 1))
        ttk.Button(lut_row, text="Choose lifetime LUT…", command=self._choose_lifetime_lut).pack(
            side=LEFT, fill=X, expand=True, padx=(0, 3)
        )
        ttk.Button(lut_row, text="Use VI LUT", command=self._use_default_lifetime_lut).pack(
            side=LEFT, fill=X, expand=True, padx=(3, 0)
        )
        ttk.Checkbutton(
            inner,
            text="Hide out-of-range lifetime-map pixels (display only)",
            variable=self.lifetime_lut_remove_var,
        ).pack(anchor="w")
        ttk.Checkbutton(inner, text="Invert lifetime LUT", variable=self.lifetime_lut_inverse_var).pack(anchor="w")
        ttk.Label(
            inner,
            textvariable=self.lifetime_lut_summary_var,
            style="Sidebar.TLabel",
            wraplength=250,
        ).pack(anchor="w", pady=(2, 0))

        ttk.Label(inner, text="NATIVE FLIMLIB FIT", style="Section.TLabel").pack(anchor="w", pady=(13, 3))
        ttk.Label(
            inner,
            textvariable=self.method_model_hint_var,
            style="Sidebar.TLabel",
            wraplength=250,
        ).pack(anchor="w", pady=(0, 4))
        self._field(
            inner,
            "Native input TCSPC bins (VI preset 29; -1 = all)",
            self.native_fit_bins_var,
        )
        self._combo(
            inner, "Native bin width mode", self.native_bin_width_mode_var, ["Auto", "Manual"]
        )
        native_time_row = ttk.Frame(inner, style="Sidebar.TFrame")
        native_time_row.pack(fill=X, pady=2)
        self._compact_field(
            native_time_row, "Native bin width (ns; manual only; VI 0.4)", self.native_bin_width_var
        ).pack(side=LEFT, fill=X, expand=True, padx=(0, 3))
        self._compact_combo(
            native_time_row,
            "IRF pedestal",
            self.native_irf_baseline_var,
            ["Per-pixel median", "Tail median", "None"],
        ).pack(side=LEFT, fill=X, expand=True, padx=(3, 0))
        ttk.Label(
            inner, textvariable=self.native_bin_width_summary_var, style="Sidebar.TLabel", wraplength=250
        ).pack(anchor="w", pady=(0, 3))
        for variable in (self.native_bin_width_mode_var, self.native_bin_width_var, self.dt_var):
            variable.trace_add("write", self._native_bin_width_mode_changed)
        self._field(
            inner,
            f"Reconvolution fit worker processes (1 = original behaviour; this machine has "
            f"{os.cpu_count() or 1} cores)",
            self.native_fit_workers_var,
        )
        ttk.Label(
            inner,
            text=(
                "Splits the image into this many row-bands, each fitted in its own subprocess "
                "concurrently -- only applies to \"FLIMLib — reconvolution\"; global/apply-model "
                "fits share state across pixels and always stay single-process."
            ),
            style="Sidebar.TLabel",
            wraplength=250,
            justify="left",
        ).pack(anchor="w", pady=(0, 3))
        self._model_order_combo = self._combo(
            inner, "Model order", self.model_order_var, ["1 — mono", "2 — bi", "3 — tri"]
        )
        self._combo(
            inner, "Distribution model", self.fit_distribution_model_var, list(self.DISTRIBUTION_MODEL_LABELS)
        )
        ttk.Label(
            inner,
            text=(
                "Stretched exponential fits one lifetime as a distribution "
                "(τ, h) instead of a discrete component -- see EcfUtil.c's "
                "GCI_stretchedexp: I(t) = Z + A·exp(-(t/τ)^(1/h)); h=1 is a "
                "plain exponential, h>1 a broader spread of rates. Forces "
                "Model order to 1 (always exactly one component). "
                "Multi-stretched exponential extends this to up to 3 "
                "independently-distributed (τ, h) components, summed "
                "together -- set Model order to how many. Both only apply "
                "to \"Train new model…\" and Lifetime method FLIMLib — "
                "global shared τ; multi-stretched also supports Auto-seed "
                "from phasor analysis in the Training dialog (§4.6) for a "
                "reliable starting guess with Model order 2-3, instead of "
                "typing Initial τ values by hand."
            ),
            style="Sidebar.TLabel",
            wraplength=250,
            justify="left",
        ).pack(anchor="w", pady=(0, 3))
        self._combo(inner, "Noise model", self.noise_model_var, list(self.NOISE_LABELS))
        self._combo(
            inner, "Fit background", self.fit_background_mode_var, list(self.FIT_BACKGROUND_LABELS)
        )
        self._field(inner, "Fixed fit background", self.fit_background_value_var)
        self._combo(inner, "IRF mode", self.irf_mode_var, list(self.IRF_MODE_LABELS))
        ttk.Checkbutton(
            inner,
            text="Integrate IRF per column (sweeping linear detector)",
            variable=self.irf_integrate_per_column_var,
        ).pack(anchor="w", pady=(3, 0))
        ttk.Label(
            inner,
            text=(
                "Per-pixel IRF only. For a scanning linear-array detector (e.g. RA2): sums each "
                "column's IRF across every row (sweep pass) into one high-SNR curve, then repeats "
                "it down that column -- matching the source LabVIEW tool's own IRF handling, and "
                "restoring the native fit's IRF-grouping speedup, which otherwise gets no sharing "
                "to exploit from noisy per-pixel-per-sweep IRF data. Leave off for a genuinely "
                "unique per-pixel IRF."
            ),
            style="Sidebar.TLabel",
            wraplength=250,
            justify="left",
        ).pack(anchor="w", pady=(0, 3))
        global_row = ttk.Frame(inner, style="Sidebar.TFrame")
        global_row.pack(fill=X, pady=2)
        self._compact_field(global_row, "Global iterations", self.global_iterations_var).pack(
            side=LEFT, fill=X, expand=True, padx=(0, 3)
        )
        self._compact_field(global_row, "Max pixels (0 = all)", self.max_global_pixels_var).pack(
            side=LEFT, fill=X, expand=True, padx=(3, 0)
        )
        ttk.Label(
            inner,
            text=(
                "Max pixels caps how many pixels the global/shared-τ fit and training use, purely "
                "for speed — it does not affect the per-pixel FLIMLib reconvolution method. 0 removes "
                "the cap entirely (every valid pixel from every training image); with more than one "
                "training image and a nonzero cap, the budget is split fairly across all of them, not "
                "spent entirely on whichever comes first."
            ),
            style="Sidebar.TLabel",
            wraplength=250,
            justify="left",
        ).pack(anchor="w", pady=(0, 3))
        self._field(inner, "Initial τ values (ns, comma separated)", self.initial_taus_var)
        dll_row = ttk.Frame(inner, style="Sidebar.TFrame")
        dll_row.pack(fill=X, pady=(3, 0))
        ttk.Button(dll_row, text="Choose fitting DLL…", command=self._choose_native_dll).pack(fill=X)
        ttk.Label(inner, textvariable=self.native_status_var, style="Sidebar.TLabel", wraplength=250).pack(
            anchor="w", pady=(3, 0)
        )

        ttk.Label(inner, text="GLOBAL MODEL", style="Section.TLabel").pack(anchor="w", pady=(13, 3))
        model_row = ttk.Frame(inner, style="Sidebar.TFrame")
        model_row.pack(fill=X, pady=2)
        ttk.Button(model_row, text="Load model…", command=self._choose_global_model).pack(
            side=LEFT, fill=X, expand=True, padx=(0, 3)
        )
        ttk.Button(model_row, text="Train new model…", command=self._open_training_dialog).pack(
            side=LEFT, fill=X, expand=True, padx=(3, 3)
        )
        ttk.Button(model_row, text="Set manually…", command=self._open_manual_model_dialog).pack(
            side=LEFT, fill=X, expand=True, padx=(3, 0)
        )
        ttk.Label(
            inner, textvariable=self.global_model_status_var, style="Sidebar.TLabel", wraplength=250
        ).pack(anchor="w", pady=(3, 0))

        self.sidebar_analyze = ttk.Button(inner, text="Run current analysis", command=self.run_current, style="Primary.TButton")
        self.sidebar_analyze.pack(fill=X, pady=(16, 5))
        ttk.Button(inner, text="Process & export queue", command=self.run_batch).pack(fill=X, pady=4)
        ttk.Button(inner, text="Export current result", command=self.export_current).pack(fill=X, pady=4)

    def _field(self, parent: tk.Misc, label: str, variable: tk.Variable) -> None:
        frame = ttk.Frame(parent, style="Sidebar.TFrame")
        frame.pack(fill=X, pady=3)
        ttk.Label(frame, text=label, style="Sidebar.TLabel").pack(anchor="w")
        ttk.Entry(frame, textvariable=variable).pack(fill=X, pady=(2, 0))

    def _disable_combobox_wheel(self, combo: ttk.Combobox, canvas: tk.Canvas | None = None) -> None:
        """Stops a scroll of the mouse wheel over this dropdown from
        silently cycling its selected value.

        ttk.Combobox has this baked into its own default widget-class
        bindings (Tk, not something this app added) -- scrolling while
        merely hovering one, with no click and no keyboard focus, changes
        which value is selected. Binding at the *instance* level lets us
        intervene before that class-level default runs (Tk processes a
        widget's own bindings before its class's), and returning "break"
        stops the class default from running at all -- but "break" would
        also stop the sidebar's own bind_all mouse-wheel scroll handler
        from ever seeing the event, so when `canvas` is given (this combo
        lives inside that scrollable area) this replicates that same
        scroll here instead of just swallowing the event -- scrolling
        over a dropdown still scrolls the panel, it just never changes
        the dropdown's own value. A combo outside any scrollable area
        (canvas=None) simply has the value-cycling suppressed outright.
        """

        def _on_wheel(event: tk.Event) -> str:
            if canvas is not None:
                canvas.yview_scroll(int(-event.delta / 120), "units")
            return "break"

        combo.bind("<MouseWheel>", _on_wheel)
        combo.bind("<Button-4>", _on_wheel)
        combo.bind("<Button-5>", _on_wheel)

    def _combo(self, parent: tk.Misc, label: str, variable: tk.Variable, values: list[str]) -> ttk.Combobox:
        frame = ttk.Frame(parent, style="Sidebar.TFrame")
        frame.pack(fill=X, pady=3)
        ttk.Label(frame, text=label, style="Sidebar.TLabel").pack(anchor="w")
        combo = ttk.Combobox(frame, textvariable=variable, values=values, state="readonly")
        combo.pack(fill=X, pady=(2, 0))
        self._disable_combobox_wheel(combo, self._sidebar_canvas)
        return combo

    def _compact_field(self, parent: tk.Misc, label: str, variable: tk.Variable) -> ttk.Frame:
        frame = ttk.Frame(parent, style="Sidebar.TFrame")
        ttk.Label(frame, text=label, style="Sidebar.TLabel").pack(anchor="w")
        ttk.Entry(frame, textvariable=variable, width=8).pack(fill=X, pady=(2, 0))
        return frame

    def _compact_combo(
        self, parent: tk.Misc, label: str, variable: tk.Variable, values: list[str]
    ) -> ttk.Frame:
        frame = ttk.Frame(parent, style="Sidebar.TFrame")
        ttk.Label(frame, text=label, style="Sidebar.TLabel").pack(anchor="w")
        combo = ttk.Combobox(frame, textvariable=variable, values=values, state="readonly", width=12)
        combo.pack(fill=X, pady=(2, 0))
        self._disable_combobox_wheel(combo, self._sidebar_canvas)
        return frame

    def _choose_native_dll(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose flimfit_labview.dll", filetypes=[("Windows DLL", "*.dll"), ("All files", "*.*")]
        )
        if path:
            self.native_dll_var.set(path)
            self.native_status_var.set(f"Native backend: {Path(path).name}")

    def _choose_leica_interrogator(self) -> None:
        folder = filedialog.askdirectory(title="Select the Leica Interrogator Desktop folder")
        if not folder:
            return
        root = Path(folder)
        venv_python = root / ".venv" / "Scripts" / "python.exe"
        converter = root / "leica_flim_converter"
        if not converter.is_dir():
            messagebox.showwarning(
                "Doesn't look right",
                f"{folder}\n\nNo \"leica_flim_converter\" folder found inside it -- pick the "
                "\"Leica Interrogator Desktop\" folder itself, not a subfolder.",
            )
            return
        self.leica_root_var.set(str(root))
        if venv_python.is_file():
            self.leica_interrogator_status_var.set(f"Leica Interrogator: {root.name}")
        else:
            self.leica_interrogator_status_var.set(
                f"Leica Interrogator: {root.name} — run install_packages.bat there first "
                "(no .venv found yet)"
            )

    def _launch_leica_full_gui(self) -> None:
        """Launch Interrogator's own PySide6 app, for manual preview/export.

        The quick in-app picker (opened by clicking a queued .lif -- see
        _open_in_leica_interrogator) covers the common case without a
        second window; this is the fallback for when you want Interrogator's
        own full preview before exporting anything.
        """

        root = self._require_leica_root()
        if root is None:
            return
        pythonw = root / ".venv" / "Scripts" / "pythonw.exe"
        if not pythonw.is_file():
            messagebox.showerror(
                "Leica Interrogator not installed",
                f"{pythonw}\n\nnot found -- run install_packages.bat in that folder first.",
            )
            return
        try:
            subprocess.Popen([str(pythonw), "-m", "leica_flim_converter"], cwd=str(root))
        except OSError as exc:
            messagebox.showerror("Could not launch Leica Interrogator", str(exc))
            return
        self.status_var.set("Opened the full Leica Interrogator GUI.")

    def _require_leica_root(self) -> Path | None:
        raw = self.leica_root_var.get().strip()
        if not raw:
            messagebox.showinfo(
                "Leica Interrogator not set",
                "Choose the Leica Interrogator Desktop folder first (sidebar: "
                "\"Leica Interrogator…\").",
            )
            return None
        return Path(raw)

    def _open_in_leica_interrogator(self, path: Path) -> None:
        """Scan a raw .lif with a leica_flim_converter and let the user
        pick which datasets inside it to export straight into FLIM
        Studio's queue -- see leica_export_gui.LeicaExportDialog.

        FLIM Studio doesn't decode .lif itself; this runs a
        leica_flim_converter package -- an externally configured
        Interrogator install if one is set, otherwise the copy bundled
        inside FLIM Studio itself (see resolve_leica_backend) -- as a
        separate process (see leica_worker.py) rather than in FLIM
        Studio's own process, for the same crash-isolation reason native
        fits run in a subprocess.
        """

        external_root = Path(self.leica_root_var.get().strip()) if self.leica_root_var.get().strip() else None
        root, command_prefix = resolve_leica_backend(external_root)
        if root is None:
            messagebox.showinfo(
                "Leica .lif reader not available",
                "This build has no built-in .lif reader, and no external \"Leica Interrogator "
                "Desktop\" folder is configured. Choose one first (sidebar: \"Leica Interrogator…\").",
            )
            return
        LeicaExportDialog(self, path, root, command_prefix)

    LEICA_WATCH_POLL_MS = 2000

    def _choose_leica_watch_folder(self) -> None:
        folder = filedialog.askdirectory(title="Choose the folder Leica Interrogator exports .ics files into")
        if not folder:
            return
        self._start_leica_watch(Path(folder))

    def _start_leica_watch(self, folder: Path) -> None:
        """Poll ``folder`` for newly-exported .ics files and queue them automatically.

        This is deliberately simple polling (no filesystem-event library
        dependency) rather than a true watch -- checking a directory listing
        every couple of seconds is cheap and plenty responsive for "I just
        clicked export in another program", and needs nothing beyond the
        standard library.
        """

        if self._leica_watch_after_id is not None:
            self.root.after_cancel(self._leica_watch_after_id)
            self._leica_watch_after_id = None
        self._leica_watch_folder = folder
        # Anything already in the queue (or already sitting in the folder
        # from before watching started) is a pre-existing file, not a new
        # export -- only files that show up *after* this point should be
        # auto-added.
        self._leica_watch_seen = set(self.queue_paths) | self._scan_leica_watch_folder(folder)
        self.leica_watch_status_var.set(f"Watching for new exports in: {folder}")
        self._poll_leica_exports()

    def _stop_leica_watch(self) -> None:
        if self._leica_watch_after_id is not None:
            self.root.after_cancel(self._leica_watch_after_id)
            self._leica_watch_after_id = None
        self._leica_watch_folder = None
        self.leica_watch_status_var.set("Not watching for exports.")

    @staticmethod
    def _scan_leica_watch_folder(folder: Path) -> set[Path]:
        try:
            found = {p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".ics", ".ids"}}
        except OSError:
            return set()
        # Match open_folder()'s rule: don't treat a .ids as its own entry
        # when its .ics header is present alongside it.
        return {
            p for p in found
            if p.suffix.lower() != ".ids" or not p.with_suffix(".ics").exists()
        }

    def _poll_leica_exports(self) -> None:
        if self._leica_watch_folder is None:
            return
        found = self._scan_leica_watch_folder(self._leica_watch_folder)
        new_paths = sorted(found - self._leica_watch_seen)
        if new_paths:
            self._leica_watch_seen.update(new_paths)
            self._add_to_queue(new_paths)
            self.status_var.set(
                f"{len(new_paths)} new Leica Interrogator export(s) added to the queue "
                f"({', '.join(p.name for p in new_paths[:3])}"
                f"{', …' if len(new_paths) > 3 else ''})"
            )
        self._leica_watch_after_id = self.root.after(self.LEICA_WATCH_POLL_MS, self._poll_leica_exports)

    def _add_to_queue(self, paths: list[Path]) -> None:
        """Append newly-discovered files to the existing queue without
        touching anything already queued or cached (unlike ``_set_queue``,
        which replaces the whole queue)."""

        existing = set(self.queue_paths)
        added = [path for path in paths if path not in existing]
        if not added:
            return
        self.queue_paths.extend(added)
        self._refresh_queue_labels()

    def _choose_global_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose a trained global model",
            filetypes=[("FLIM Studio global model", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            model = load_global_model(path)
        except GlobalModelError as exc:
            messagebox.showerror("Could not load model", str(exc))
            return
        self._activate_trained_model(model, path)
        self.status_var.set(f"Loaded trained model: {model.name}")

    def _use_trained_model(self, model: TrainedGlobalModel, path: str) -> None:
        """Wire a just-trained (and saved) model into Application mode."""

        self._activate_trained_model(model, path)
        self.status_var.set(f"Using trained model: {model.name}")

    @staticmethod
    def _label_for_value(labels: dict[str, object], value: object) -> str | None:
        for label, candidate in labels.items():
            if candidate == value:
                return label
        return None

    def _activate_trained_model(self, model: TrainedGlobalModel, path: str) -> None:
        """Load a trained model and mirror its fitting conditions in the sidebar.

        ``analyze_dataset`` overrides these fields from the model regardless
        of what the sidebar shows when applying (see the
        ``flimlib_apply_global`` branch) -- syncing the displayed controls
        here keeps the sidebar truthful about what a run will actually use,
        rather than silently diverging from it.
        """

        self.trained_model = model
        self.global_model_path_var.set(path)
        self.global_model_status_var.set(model.summary())

        order_label = {1: "1 — mono", 2: "2 — bi", 3: "3 — tri"}.get(model.model_order)
        if order_label is not None:
            self.model_order_var.set(order_label)
        distribution_label = self._label_for_value(
            self.DISTRIBUTION_MODEL_LABELS, getattr(model, "fit_distribution_model", 0)
        )
        if distribution_label is not None:
            self.fit_distribution_model_var.set(distribution_label)
        noise_label = self._label_for_value(self.NOISE_LABELS, model.noise_model)
        if noise_label is not None:
            self.noise_model_var.set(noise_label)
        background_label = self._label_for_value(self.FIT_BACKGROUND_LABELS, model.background_mode)
        if background_label is not None:
            self.fit_background_mode_var.set(background_label)
        self.fit_background_value_var.set(f"{model.background_value:g}")
        irf_label = self._label_for_value(self.IRF_MODE_LABELS, model.irf_mode)
        if irf_label is not None:
            self.irf_mode_var.set(irf_label)
        self.irf_integrate_per_column_var.set(bool(model.irf_integrate_per_column))
        self.fit_threshold_var.set(f"{model.fit_min_counts:g}")
        self.start_bin_var.set(str(model.fit_start_bin))
        self.end_bin_var.set(str(model.fit_end_bin))
        self.native_fit_bins_var.set(str(model.native_fit_bins))
        # Manual, not Auto: a trained model's bin width is a fixed condition
        # baked in at training time (analyze_dataset's flimlib_apply_global
        # branch always uses trained_model.native_bin_width_ns regardless of
        # this UI either way) -- showing "Auto" here would misleadingly
        # suggest it still tracks Time-bin width, which it no longer does.
        self.native_bin_width_mode_var.set("Manual")
        self.native_bin_width_var.set(f"{model.native_bin_width_ns:g}")
        baseline_label = self._label_for_value(
            {"Per-pixel median": "median", "Tail median": "tail_median", "None": "none"},
            model.native_irf_baseline_mode,
        )
        if baseline_label is not None:
            self.native_irf_baseline_var.set(baseline_label)

        for label, value in self.METHOD_LABELS.items():
            if value == "flimlib_apply_global":
                self.method_var.set(label)
                break

    def _open_training_dialog(self) -> None:
        if self.busy:
            return
        TrainingDialog(self)

    def _open_manual_model_dialog(self) -> None:
        if self.busy:
            return
        ManualModelDialog(self)

    def _choose_lifetime_lut(self) -> None:
        if self.busy:
            return
        path = filedialog.askopenfilename(
            title="Choose a 256-colour lifetime LUT",
            filetypes=[("ImageJ raw LUT", "*.lut"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            from .lut import load_imagej_lut

            load_imagej_lut(path)
        except Exception as exc:
            messagebox.showerror("Unable to load lifetime LUT", str(exc))
            return
        self.lifetime_lut_path_var.set(path)
        self.lifetime_lut_summary_var.set(f"{Path(path).name} — external 256-colour LUT")

    def _use_default_lifetime_lut(self) -> None:
        self.lifetime_lut_path_var.set("")
        self.lifetime_lut_inverse_var.set(False)
        self.lifetime_lut_remove_var.set(True)
        self.lifetime_lut_summary_var.set("Reverse Rainbow (DM).lut — original LabVIEW LUT")

    def _background_settings_preview(self) -> AnalysisSettings:
        mode = "auto" if self.background_scale_mode_var.get().startswith("Auto") else "manual"
        multiplier = float(self.background_multiplier_var.get())
        try:
            background_counts = float(self.background_counts_var.get())
        except ValueError:
            background_counts = 0.0
        return AnalysisSettings(
            background_scale_mode=mode,
            background_multiplier=multiplier,
            background_auto_fallback=multiplier,
            background_counts=background_counts,
            background_clip_negative=bool(self.background_clip_var.get()),
        )

    def _update_background_scale_summary(self) -> None:
        if self.background is None:
            self.background_scale_summary_var.set("No background loaded")
            return
        if self.dataset is None:
            self.background_scale_summary_var.set("Background loaded; load a sample to resolve scaling")
            return
        try:
            multiplier, details = resolve_background_multiplier(
                self.dataset,
                self.background,
                self._background_settings_preview(),
            )
            basis = str(details.get("basis", "manual")).replace("_", " ")
            sample_value = details.get("sample_normalization_value")
            background_value = details.get("background_normalization_value")
            ratio = (
                f" ({sample_value:g}/{background_value:g})"
                if isinstance(sample_value, (int, float)) and isinstance(background_value, (int, float))
                else ""
            )
            self.background_scale_summary_var.set(f"Applied ×{multiplier:g} from {basis}{ratio}")
        except Exception as exc:
            self.background_scale_summary_var.set(f"Background scale needs attention: {exc}")

    def _native_bin_width_mode_changed(self, *_args) -> None:
        """Keep the summary line under Native bin width truthful about what
        a reconvolution/global fit will actually use.

        In Auto mode the manual entry is ignored entirely -- _settings()
        passes -1.0, which AnalysisSettings.normalized() already resolves
        to the current Time-bin width -- so this only ever *reports* that
        value here, it never writes back into native_bin_width_var itself
        (same pattern as lifetime/background auto-scaling elsewhere: the
        manual field stays exactly what was last typed into it, ready to
        use immediately if switched to Manual, rather than being silently
        overwritten while inactive).
        """

        auto = self.native_bin_width_mode_var.get().startswith("Auto")
        if auto:
            try:
                dt_ns = float(self.dt_var.get())
                if not np.isfinite(dt_ns) or dt_ns <= 0:
                    raise ValueError
            except ValueError:
                self.native_bin_width_summary_var.set("Enter a valid Time-bin width above to resolve this.")
                return
            self.native_bin_width_summary_var.set(f"Using {dt_ns:g} ns — matches Time-bin width above.")
        else:
            try:
                manual_ns = float(self.native_bin_width_var.get())
                if not np.isfinite(manual_ns) or manual_ns <= 0:
                    raise ValueError
            except ValueError:
                self.native_bin_width_summary_var.set("Enter a positive manual bin width above.")
                return
            self.native_bin_width_summary_var.set(
                f"Using the manual value above ({manual_ns:g} ns) — may not match Time-bin width."
            )

    def _row_map_from_controls(self) -> RowInterpolationMap:
        coordinate_mode = self.ROW_COORDINATE_LABELS[self.row_map_coordinate_var.get()]
        rows = parse_row_spec(self.row_map_rows_var.get(), index_base=0)
        return RowInterpolationMap(
            rows=rows,
            radius=int(self.row_map_radius_var.get()),
            coordinate_mode=coordinate_mode,
            name=self.row_map_name,
            source_path=self.row_map_source,
            sha256=self.row_map_sha256,
            expected_source_height=self.row_map_expected_source_height,
        ).normalized()

    def _set_row_map_controls(self, mapping: RowInterpolationMap) -> None:
        mapping = mapping.normalized()
        self._loading_row_map_controls = True
        try:
            self.row_map_rows_var.set(format_row_spec(mapping.rows))
            self.row_map_radius_var.set(str(mapping.radius))
            label = next(
                name for name, value in self.ROW_COORDINATE_LABELS.items() if value == mapping.coordinate_mode
            )
            self.row_map_coordinate_var.set(label)
            self.row_map_name = mapping.name
            self.row_map_source = mapping.source_path
            self.row_map_sha256 = mapping.sha256
            self.row_map_expected_source_height = mapping.expected_source_height
        finally:
            self._loading_row_map_controls = False
        self._update_row_map_summary(mapping)

    def _row_map_control_edited(self, *_args) -> None:
        if not self._loading_row_map_controls:
            # A manual edit creates a custom, height-relative map. File hashes
            # and the original preset's fixed 512-row geometry no longer
            # describe the edited values and must not remain sticky.
            self.row_map_name = "Custom detector-row map"
            self.row_map_source = ""
            self.row_map_sha256 = ""
            self.row_map_expected_source_height = None
        self._update_row_map_summary()

    def _update_row_map_summary(self, mapping: RowInterpolationMap | None = None) -> None:
        try:
            mapping = mapping or self._row_map_from_controls()
            if not mapping.rows:
                self.row_map_summary_var.set("Detector-row map: none defined")
                return
            resolved_note = ""
            if self.dataset is not None:
                resolved = mapping.resolve_rows(self.dataset.y_size)
                resolved_note = f" • image Y {format_row_spec(resolved)}"
            convention = "mirrored LabVIEW IDs" if mapping.coordinate_mode == LABVIEW_MIRRORED else "direct image Y"
            self.row_map_summary_var.set(
                f"{len(mapping.rows)} rows • {convention} • radius {mapping.radius}{resolved_note}"
            )
        except Exception as exc:
            self.row_map_summary_var.set(f"Map needs attention: {exc}")

    def _use_labview_row_preset(self) -> None:
        mapping = labview_main_preset()
        self._set_row_map_controls(mapping)
        self.row_map_enabled_var.set(False)
        if self.dataset is not None:
            try:
                mapping.resolve_rows(self.dataset.y_size)
            except ValueError as exc:
                messagebox.showwarning(
                    "Preset does not match this image",
                    f"The original preset is for a 512-row acquisition and remains disabled.\n\n{exc}",
                )
                return
        self.status_var.set("Loaded the original 19-row LabVIEW detector preset (still disabled).")

    def _load_row_map(self) -> None:
        if self.busy:
            return
        path = filedialog.askopenfilename(
            title="Load detector-row map",
            filetypes=[("Row map", "*.json *.csv *.txt"), ("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            mapping = load_row_map(path)
            if self.dataset is not None:
                mapping.resolve_rows(self.dataset.y_size)
            self._set_row_map_controls(mapping)
            self.status_var.set(f"Loaded detector-row map {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("Unable to load row map", str(exc))

    def _save_row_map(self) -> None:
        if self.busy:
            return
        try:
            mapping = self._row_map_from_controls()
            if not mapping.rows:
                raise ValueError("Define at least one row before saving the map.")
        except Exception as exc:
            messagebox.showerror("Invalid row map", str(exc))
            return
        path = filedialog.asksaveasfilename(
            title="Save detector-row map",
            defaultextension=".json",
            filetypes=[("JSON row map", "*.json"), ("CSV rows", "*.csv"), ("Text rows", "*.txt")],
        )
        if not path:
            return
        try:
            save_row_map(path, mapping)
            loaded = load_row_map(path)
            self._set_row_map_controls(loaded)
            self.status_var.set(f"Saved detector-row map {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("Unable to save row map", str(exc))

    def _preview_row_map(self) -> None:
        if self.dataset is None:
            messagebox.showinfo("No data", "Load a decay dataset before previewing a detector-row map.")
            return
        try:
            mapping = self._row_map_from_controls()
            resolved = mapping.resolve_rows(self.dataset.y_size)
            if not resolved:
                raise ValueError("The detector-row map is empty.")
        except Exception as exc:
            messagebox.showerror("Invalid row map", str(exc))
            self._update_row_map_summary()
            return
        preview = tk.Toplevel(self.root)
        preview.title("Detector-row interpolation map")
        preview.geometry("760x620")
        preview.configure(bg=BG)
        view = ImageView(
            preview,
            "Detector-row interpolation map",
            (
                f"{len(resolved)} source rows; magenta = corrected; "
                f"entered {mapping.coordinate_mode}, resolved image Y shown"
            ),
        )
        view.pack(fill=BOTH, expand=True, padx=12, pady=12)
        view.set_image(
            interpolation_map_image(
                (self.dataset.y_size, self.dataset.x_size),
                resolved,
                base_intensity=self.dataset.decay.astype(np.float64).sum(axis=-1),
            )
        )
        ttk.Button(preview, text="Close", command=preview.destroy).pack(pady=(0, 12))
        self._update_row_map_summary(mapping)

    def _row_overlay_changed(self) -> None:
        self._sync_map_overlays()

    def _column_map_control_edited(self, *_args) -> None:
        try:
            columns = parse_row_spec(self.column_map_columns_var.get(), index_base=0)
            radius = max(1, int(self.column_map_radius_var.get()))
            if self.dataset is not None and columns and max(columns) >= self.dataset.x_size:
                raise ValueError(f"valid X range is 0..{self.dataset.x_size - 1}")
            self.column_map_summary_var.set(
                "Detector-column map: none defined"
                if not columns
                else f"{len(columns)} zero-based columns; donor radius {radius}"
            )
        except Exception as exc:
            self.column_map_summary_var.set(f"Column map needs attention: {exc}")

    def _column_overlay_changed(self) -> None:
        self._sync_map_overlays()

    def _build_content(self, parent: ttk.Frame) -> None:
        metrics = ttk.Frame(parent)
        metrics.pack(fill=X, pady=(0, 10))
        metric_items = [
            ("VALID PIXELS", self.valid_var),
            ("VALID-PIXEL τ MEAN", self.mean_tau_var),
            ("REPRESENTATIVE τ", self.global_tau_var),
            ("MAX PROCESSED COUNTS", self.max_counts_var),
            ("MEAN χ² (valid px)", self.mean_chi_var),
        ]
        last_column = len(metric_items) - 1
        for column, (name, variable) in enumerate(metric_items):
            card = ttk.Frame(metrics, style="Panel.TFrame", padding=(15, 9))
            card.grid(
                row=0, column=column, sticky="ew",
                padx=(0 if column == 0 else 5, 0 if column == last_column else 5),
            )
            metrics.columnconfigure(column, weight=1)
            ttk.Label(card, text=name, style="MetricName.TLabel").pack(anchor="w")
            ttk.Label(card, textvariable=variable, style="Metric.TLabel").pack(anchor="w")

        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=BOTH, expand=True)
        overview = ttk.Frame(self.notebook, padding=4)
        decay_tab = ttk.Frame(self.notebook, padding=4)
        irf_decay_tab = ttk.Frame(self.notebook, padding=4)
        phasor_tab = ttk.Frame(self.notebook, padding=4)
        histogram_tab = ttk.Frame(self.notebook, padding=4)
        multi_tab = ttk.Frame(self.notebook, padding=4)
        metadata_tab = ttk.Frame(self.notebook, padding=4)
        self.notebook.add(overview, text="Overview")
        self.notebook.add(decay_tab, text="Decay & fit")
        self.notebook.add(irf_decay_tab, text="IRF & Decay")
        self.notebook.add(phasor_tab, text="Phasor")
        self.notebook.add(histogram_tab, text="Histogram")
        self.notebook.add(multi_tab, text="Multi-exponential")
        self.notebook.add(metadata_tab, text="Metadata")

        overview.rowconfigure(0, weight=1, minsize=250)
        overview.rowconfigure(1, weight=1, minsize=250)
        overview.columnconfigure(0, weight=1, minsize=220)
        overview.columnconfigure(1, weight=1, minsize=220)
        self.intensity_view = ImageView(overview, "Intensity", "Click: add point • Drag: ROI", self._map_gesture)
        secondary_panel = ttk.Frame(overview, style="Panel.TFrame")
        secondary_panel.rowconfigure(1, weight=1)
        secondary_panel.columnconfigure(0, weight=1)
        secondary_toolbar = ttk.Frame(secondary_panel, style="Panel.TFrame", padding=(10, 6))
        secondary_toolbar.grid(row=0, column=0, sticky="ew")
        ttk.Label(secondary_toolbar, text="SECOND MAP", style="PanelTitle.TLabel").pack(side=LEFT)
        ttk.Checkbutton(
            secondary_toolbar,
            text="Log-scale intensity",
            variable=self.intensity_log_var,
            command=self._intensity_scale_changed,
        ).pack(side=LEFT, padx=(14, 0))
        map_combo = ttk.Combobox(
            secondary_toolbar,
            textvariable=self.map_mode_var,
            values=["Intensity", "Intensity with threshold", "Lifetime", "Composite"],
            state="readonly",
            width=13,
        )
        map_combo.pack(side=RIGHT)
        map_combo.bind("<<ComboboxSelected>>", self._map_mode_changed)
        # Not inside the scrollable sidebar -- no canvas to redirect a
        # wheel-scroll to, just suppress the value-cycling outright (see
        # _disable_combobox_wheel).
        self._disable_combobox_wheel(map_combo, canvas=None)
        self.map_mode_combo = map_combo
        threshold_controls = ttk.Frame(secondary_toolbar, style="Panel.TFrame")
        threshold_controls.pack(side=RIGHT, padx=(0, 10))
        ttk.Label(threshold_controls, text="Preview threshold", style="Muted.TLabel").pack(side=LEFT)
        self.live_threshold_scale = ttk.Scale(
            threshold_controls,
            from_=0,
            to=5000,
            orient="horizontal",
            command=self._live_threshold_changed,
            length=120,
        )
        self.live_threshold_scale.set(1000)
        self.live_threshold_scale.pack(side=LEFT, padx=5)
        ttk.Entry(threshold_controls, textvariable=self.live_threshold_var, width=7).pack(side=LEFT)
        ttk.Button(
            threshold_controls,
            text="Use for fit",
            command=self._apply_live_threshold_to_fit,
        ).pack(side=LEFT, padx=(5, 0))
        self.live_threshold_var.trace_add("write", self._live_threshold_entry_changed)
        self.secondary_view = ImageView(
            secondary_panel, "Lifetime", "Click: add point • Drag: ROI", self._map_gesture
        )
        self.secondary_view.grid(row=1, column=0, sticky="nsew")
        self.intensity_view.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        secondary_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

        inspector = ttk.Frame(overview)
        inspector.rowconfigure(1, weight=1)
        inspector.columnconfigure(0, weight=1)
        inspector.columnconfigure(1, weight=1)
        inspector.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(10, 0))

        point_toolbar = ttk.Frame(inspector, style="Panel.TFrame", padding=(10, 5))
        point_toolbar.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(point_toolbar, textvariable=self.point_summary_var, style="Muted.TLabel").pack(side=LEFT)
        ttk.Checkbutton(
            point_toolbar,
            text="Single point mode (live fit preview)",
            variable=self.single_point_mode_var,
            command=self._single_point_mode_toggled,
        ).pack(side=LEFT, padx=(14, 0))
        ttk.Button(
            point_toolbar,
            text="Re-fit at current settings",
            command=self._refresh_single_point_live_fit,
        ).pack(side=LEFT, padx=(6, 0))
        ttk.Button(point_toolbar, text="Undo", command=self.undo_point).pack(side=RIGHT, padx=(5, 0))
        ttk.Button(point_toolbar, text="Clear points", command=self.clear_points).pack(side=RIGHT, padx=(5, 0))
        ttk.Button(point_toolbar, text="Clear ROI", command=self.clear_roi).pack(side=RIGHT)

        # Selected-point decays keeps its original half-width column; phasor
        # and histogram now sit together in the other half instead of being
        # swapped in and out of it by a scope selector.
        decay_panel = ttk.Frame(inspector, style="Panel.TFrame")
        decay_panel.rowconfigure(1, weight=1)
        decay_panel.columnconfigure(0, weight=1)
        self.roi_decay_view = ImageView(
            decay_panel, "Selected-point decays", "Click either map to add comparison points"
        )
        self.roi_decay_view.grid(row=1, column=0, sticky="nsew")
        decay_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 5))

        scope_panel = ttk.Frame(inspector)
        scope_panel.rowconfigure(0, weight=1)
        scope_panel.columnconfigure(0, weight=1, minsize=140)
        scope_panel.columnconfigure(1, weight=1, minsize=140)
        scope_panel.grid(row=1, column=1, sticky="nsew", padx=(5, 0))
        self.overview_phasor_view = ImageView(
            scope_panel, "Selected phasors", "First-harmonic phasor cloud"
        )
        self.overview_phasor_view.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self.overview_histogram_view = ImageView(
            scope_panel, "Lifetime histogram", "Distribution of valid pixels"
        )
        self.overview_histogram_view.grid(row=0, column=1, sticky="nsew", padx=(4, 0))

        decay_tab.rowconfigure(0, weight=1)
        decay_tab.columnconfigure(0, weight=1)
        self.decay_view = ImageView(decay_tab, "ROI decay detail", "Drag on an Overview map to select a region")
        self.decay_view.grid(row=0, column=0, sticky="nsew")

        irf_decay_tab.rowconfigure(0, weight=1)
        irf_decay_tab.columnconfigure(0, weight=1)
        self.irf_decay_view = ImageView(
            irf_decay_tab, "IRF & Decay", "Whole-image mean decay and the IRF used for this run, both peak-normalised"
        )
        self.irf_decay_view.grid(row=0, column=0, sticky="nsew")

        phasor_tab.rowconfigure(2, weight=1)
        phasor_tab.columnconfigure(0, weight=1)
        phasor_tab.columnconfigure(1, weight=1)

        phasor_roi_toolbar = ttk.Frame(phasor_tab, style="Panel.TFrame", padding=(10, 6))
        phasor_roi_toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        ttk.Label(phasor_roi_toolbar, text="PHASOR ROI", style="PanelTitle.TLabel").pack(side=LEFT)
        ttk.Label(phasor_roi_toolbar, text="Shape:", style="Muted.TLabel").pack(side=LEFT, padx=(16, 4))
        shape_combo = ttk.Combobox(
            phasor_roi_toolbar,
            textvariable=self.phasor_roi_shape_var,
            values=["Rectangle", "Ellipse", "Freeform"],
            state="readonly",
            width=10,
        )
        shape_combo.pack(side=LEFT)
        self._disable_combobox_wheel(shape_combo, canvas=None)
        ttk.Label(phasor_roi_toolbar, text="Show on lifetime map as:", style="Muted.TLabel").pack(
            side=LEFT, padx=(16, 4)
        )
        display_combo = ttk.Combobox(
            phasor_roi_toolbar,
            textvariable=self.phasor_roi_display_mode_var,
            values=["Original lifetime", "Composite image", "Coloured regions", "Composite coloured regions"],
            state="readonly",
            width=24,
        )
        display_combo.pack(side=LEFT)
        self._disable_combobox_wheel(display_combo, canvas=None)
        display_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_phasor_roi_lifetime_view())
        ttk.Checkbutton(
            phasor_roi_toolbar,
            text="Flag below-threshold pixels",
            variable=self.phasor_roi_show_invalid_var,
            command=self._refresh_phasor_roi_lifetime_view,
        ).pack(side=LEFT, padx=(10, 0))
        ttk.Checkbutton(
            phasor_roi_toolbar,
            text="Show overlap (add colours)",
            variable=self.phasor_roi_additive_overlap_var,
            command=self._refresh_phasor_roi_lifetime_view,
        ).pack(side=LEFT, padx=(10, 0))
        ttk.Label(phasor_roi_toolbar, textvariable=self.phasor_roi_summary_var, style="Muted.TLabel").pack(
            side=LEFT, padx=(16, 0)
        )
        ttk.Button(phasor_roi_toolbar, text="Clear all ROIs", command=self._clear_phasor_rois).pack(
            side=RIGHT, padx=(6, 0)
        )
        ttk.Button(phasor_roi_toolbar, text="Load ROIs…", command=self._load_phasor_rois_dialog).pack(
            side=RIGHT, padx=(6, 0)
        )
        ttk.Button(phasor_roi_toolbar, text="Save ROIs…", command=self._save_phasor_rois_dialog).pack(
            side=RIGHT, padx=(6, 0)
        )
        ttk.Button(phasor_roi_toolbar, text="Remove", command=self._remove_phasor_roi_by_number).pack(
            side=RIGHT, padx=(2, 6)
        )
        ttk.Button(phasor_roi_toolbar, text="Toggle on/off", command=self._toggle_phasor_roi_by_number).pack(
            side=RIGHT, padx=(2, 2)
        )
        self.phasor_roi_remove_spin = ttk.Spinbox(
            phasor_roi_toolbar, from_=1, to=1, textvariable=self.phasor_roi_remove_number_var, width=4,
        )
        self.phasor_roi_remove_spin.pack(side=RIGHT)
        ttk.Label(phasor_roi_toolbar, text="ROI #", style="Muted.TLabel").pack(side=RIGHT, padx=(6, 2))

        phasor_cluster_row = ttk.Frame(phasor_tab, style="Panel.TFrame", padding=(10, 0))
        phasor_cluster_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        ttk.Label(phasor_cluster_row, text="Clusters:", style="Muted.TLabel").pack(side=LEFT, padx=(0, 4))
        cluster_spin = ttk.Spinbox(
            phasor_cluster_row, from_=1, to=8, textvariable=self.phasor_cluster_count_var, width=4,
        )
        cluster_spin.pack(side=LEFT)
        ttk.Label(phasor_cluster_row, text="ROI size (σ):", style="Muted.TLabel").pack(side=LEFT, padx=(10, 4))
        cluster_sigma_spin = ttk.Spinbox(
            phasor_cluster_row,
            from_=0.5, to=5.0, increment=0.1,
            textvariable=self.phasor_cluster_sigma_var, width=5,
        )
        cluster_sigma_spin.pack(side=LEFT)
        ttk.Button(
            phasor_cluster_row, text="Suggest ROIs (clustering)", command=self._suggest_phasor_roi_clusters
        ).pack(side=LEFT, padx=(8, 0))
        ttk.Label(
            phasor_cluster_row,
            text="Fits a photon-weighted Gaussian mixture to the phasor cloud and draws a tilted ellipse "
            "following each cluster's shape, as an alternative to drawing ROIs by hand.",
            style="Muted.TLabel",
        ).pack(side=LEFT, padx=(10, 0))

        self.phasor_view = ImageView(
            phasor_tab, "Phasor plot",
            "Drag: rectangle/ellipse/freeform ROI • Right-click-drag on a ROI to rotate it",
        )
        self.phasor_view.grid(row=2, column=0, sticky="nsew", padx=(0, 5))
        self._bind_phasor_roi_drawing(self.phasor_view)

        self.phasor_roi_lifetime_view = ImageView(
            phasor_tab, "Lifetime map — phasor ROI highlight", "Draw a phasor ROI to highlight it here"
        )
        self.phasor_roi_lifetime_view.grid(row=2, column=1, sticky="nsew", padx=(5, 0))

        histogram_tab.rowconfigure(0, weight=1)
        histogram_tab.columnconfigure(0, weight=1)
        self.histogram_view = ImageView(histogram_tab, "Lifetime histogram", "Distribution of valid pixels")
        self.histogram_view.grid(row=0, column=0, sticky="nsew")

        multi_tab.rowconfigure(1, weight=1)
        multi_tab.columnconfigure(0, weight=1)
        self.multi_summary_var = tk.StringVar(value="Run a native bi- or tri-exponential fit to view components.")
        ttk.Label(multi_tab, textvariable=self.multi_summary_var, style="Muted.TLabel").grid(
            row=0, column=0, sticky="w", padx=8, pady=(4, 8)
        )
        self.multi_grid = ttk.Frame(multi_tab)
        self.multi_grid.grid(row=1, column=0, sticky="nsew")
        for row in range(3):
            self.multi_grid.rowconfigure(row, weight=1)
        for column in range(2):
            self.multi_grid.columnconfigure(column, weight=1)
        self.multi_tau_views: list[ImageView] = []
        self.multi_amplitude_views: list[ImageView] = []
        for component in range(3):
            tau_view = ImageView(self.multi_grid, f"tau{component + 1}", "Native lifetime component")
            amp_view = ImageView(self.multi_grid, f"A{component + 1}", "Local fitted amplitude")
            tau_view.grid(row=component, column=0, sticky="nsew", padx=(0, 4), pady=3)
            amp_view.grid(row=component, column=1, sticky="nsew", padx=(4, 0), pady=3)
            self.multi_tau_views.append(tau_view)
            self.multi_amplitude_views.append(amp_view)

        metadata_tab.rowconfigure(0, weight=1)
        metadata_tab.columnconfigure(0, weight=1)
        self.metadata_text = tk.Text(
            metadata_tab,
            bg=PANEL,
            fg=TEXT,
            insertbackground=TEXT,
            selectbackground="#075985",
            borderwidth=0,
            padx=18,
            pady=16,
            font=("Cascadia Mono", 10),
            wrap="word",
        )
        self.metadata_text.grid(row=0, column=0, sticky="nsew")

    def _build_status_bar(self) -> None:
        bar = tk.Frame(self.root, bg="#07101d", height=34)
        bar.pack(fill=X)
        bar.pack_propagate(False)
        tk.Label(bar, textvariable=self.status_var, bg="#07101d", fg=MUTED, font=("Segoe UI", 9)).pack(
            side=LEFT, padx=16
        )
        ttk.Progressbar(bar, variable=self.progress_var, maximum=1.0, length=230).pack(side=RIGHT, padx=16, pady=9)

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Control-o>", lambda _e: self.open_files())
        self.root.bind("<Control-r>", lambda _e: self.run_current())
        self.root.bind("<Control-e>", lambda _e: self.export_current())
        self.root.bind("<Escape>", lambda _e: self.clear_roi())

    def _map_views(self) -> tuple[ImageView, ImageView]:
        return self.intensity_view, self.secondary_view

    def _sync_map_interactivity(self) -> None:
        """Maps are clickable once there's a committed result -- or, so single
        point mode can do its job of running *before* "Run analysis", whenever
        single point mode is enabled and a dataset is loaded even without one.
        Call this after anything that changes busy/result/single-point-mode/
        dataset-loaded state; it's the single source of truth other call
        sites used to duplicate inline (and disagree with).
        """

        interactive = not self.busy and (
            self.result is not None or (self.single_point_mode_var.get() and self.dataset is not None)
        )
        for view in self._map_views():
            view.set_interactive(interactive)

    @staticmethod
    def _file_types() -> list[tuple[str, str]]:
        return [
            ("FLIM data + Leica raw", "*.ics *.ids *.npy *.npz *.tif *.tiff *.csv *.txt *.lif"),
            ("ICS image cytometry", "*.ics *.ids"),
            ("Leica raw (.lif)", "*.lif"),
            ("NumPy", "*.npy *.npz"),
            ("TIFF stack", "*.tif *.tiff"),
            ("All files", "*.*"),
        ]

    def open_files(self) -> None:
        if self.busy:
            return
        paths = filedialog.askopenfilenames(title="Open FLIM decay files", filetypes=self._file_types())
        if not paths:
            return
        self._set_queue([Path(path) for path in paths])
        self.queue_list.selection_set(0)
        self.load_path(self.queue_paths[0])

    def open_folder(self) -> None:
        if self.busy:
            return
        folder = filedialog.askdirectory(title="Select a folder containing FLIM data")
        if not folder:
            return
        # .lif is not a loadable format for load_dataset() -- it's queued
        # here purely so it can be handed off to Leica Interrogator (see
        # load_path()) rather than being invisible/unselectable in a folder
        # that mixes raw .lif acquisitions with already-exported .ics.
        suffixes = set(SUPPORTED_EXTENSIONS) | {".lif"}
        paths = sorted(path for path in Path(folder).iterdir() if path.is_file() and path.suffix.lower() in suffixes)
        # Do not queue .ids companions when their .ics header is present.
        paths = [path for path in paths if path.suffix.lower() != ".ids" or not path.with_suffix(".ics").exists()]
        if not paths:
            messagebox.showinfo("No data found", "The selected folder contains no supported FLIM files.")
            return
        self._set_queue(paths)
        self.queue_list.selection_set(0)
        self.load_path(paths[0])

    def _set_queue(self, paths: list[Path]) -> None:
        self.queue_paths = list(paths)
        self.queue_results.clear()
        self._refresh_queue_labels()
        self.queue_cache_status_var.set("No queue results cached yet.")

    def _remove_selected_from_queue(self) -> None:
        """Remove whichever queue entries are currently selected (ctrl/shift
        -click to select more than one first). Their cached results, if any,
        are dropped too; nothing on disk is touched."""

        if self.busy:
            return
        selection = self.queue_list.curselection()
        if not selection:
            messagebox.showinfo(
                "Nothing selected", "Select one or more queue entries first (ctrl/shift-click for more than one)."
            )
            return
        removed = {self.queue_paths[index] for index in selection}
        self.queue_paths = [path for path in self.queue_paths if path not in removed]
        for path in removed:
            self.queue_results.pop(path, None)
        self.queue_list.selection_clear(0, END)
        self._refresh_queue_labels()
        self.queue_cache_status_var.set(
            f"{len(self.queue_results)}/{len(self.queue_paths)} results cached (● in the list)"
        )
        self.status_var.set(f"Removed {len(removed)} file(s) from the queue.")

    def _clear_queue(self) -> None:
        if self.busy:
            return
        if not self.queue_paths:
            return
        self._set_queue([])
        self.status_var.set("Queue cleared.")

    def _refresh_queue_labels(self) -> None:
        """Redraw the queue list, marking which entries have a cached result."""

        selection = self.queue_list.curselection()
        self.queue_list.delete(0, END)
        for path in self.queue_paths:
            if path.suffix.lower() == ".lif":
                marker = "◇ "  # raw Leica file -- opens in Interrogator, not analysed here
            elif path in self.queue_results:
                marker = "● "
            else:
                marker = "   "
            self.queue_list.insert(END, f"{marker}{path.name}")
        if selection:
            self.queue_list.selection_set(selection[0])

    def _queue_selected(self, _event=None) -> None:
        selection = self.queue_list.curselection()
        # A multi-selection (ctrl/shift-click, e.g. picking several entries
        # to remove -- see _remove_selected_from_queue) shouldn't try to
        # load/show every entry it passes through; only act on a plain
        # single-entry selection, the normal "click one to view it" case.
        if len(selection) != 1 or self.busy:
            return
        path = self.queue_paths[int(selection[0])]
        if path in self.queue_results:
            self._show_cached_result(path)
        else:
            self.load_path(path)

    def _show_cached_result(self, path: Path) -> None:
        dataset, result = self.queue_results.pop(path)
        self.queue_results[path] = (dataset, result)  # touch: mark most-recently-viewed
        self.dataset = dataset
        self.result = result
        self.dt_var.set(f"{dataset.dt_ns:.8g}")
        self.rep_var.set(f"{dataset.rep_rate_mhz:.8g}")
        self.dataset_var.set(f"{dataset.name}  •  {dataset.y_size}×{dataset.x_size}×{dataset.time_bins}")
        temporal_extent = dataset.time_bins * dataset.dt_ns
        self.acquisition_var.set(
            f"Acquisition: {dataset.time_bins} TCSPC bins • {dataset.dt_ns:g} ns/bin • "
            f"{temporal_extent:g} ns period"
        )
        self.progress_var.set(1)
        self.status_var.set(f"Showing cached result: {dataset.name}")
        self._refresh_result_views()
        self._set_busy(False)

    def _cache_queue_result(
        self, path: Path, dataset: FLIMDataset, result: AnalysisResult, limit: int
    ) -> None:
        if path in self.queue_results:
            del self.queue_results[path]
        self.queue_results[path] = (dataset, result)
        evicted = 0
        while len(self.queue_results) > max(1, limit):
            self.queue_results.popitem(last=False)
            evicted += 1
        self._refresh_queue_labels()
        note = f" ({evicted} older result{'s' if evicted != 1 else ''} dropped to stay under the cap)" if evicted else ""
        self.queue_cache_status_var.set(
            f"{len(self.queue_results)}/{len(self.queue_paths)} results cached (● in the list){note}"
        )

    def process_queue_cached(self) -> None:
        """Run analysis across the whole queue and keep results in memory to browse.

        Unlike "Process & export queue" (which writes each result to disk and
        discards it), this keeps up to ``queue_cache_limit_var`` results in
        memory so clicking a ● entry in the queue list shows it instantly
        with no re-run. Older results are evicted least-recently-viewed
        first once the cap is exceeded.
        """

        if self.busy:
            return
        if not self.queue_paths:
            messagebox.showinfo("Queue is empty", "Open multiple files or a folder first.")
            return
        try:
            settings = self._settings()
        except ValueError as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return
        try:
            limit = int(self.queue_cache_limit_var.get())
            if limit < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Invalid cache limit", "\"Keep in memory\" must be a whole number of at least 1."
            )
            return
        paths = list(self.queue_paths)
        irf, background = self.irf, self.background
        accumulate = bool(self.phasor_accumulate_queue_var.get())
        if accumulate:
            # Fresh per run, so a stale cloud from an earlier, unrelated
            # queue never silently lingers into this one.
            self.phasor_accumulated_points = []
            self._update_phasor_accumulate_status()
        self._set_busy(True)

        def worker() -> None:
            failures: list[str] = []
            last_path: Path | None = None
            for index, path in enumerate(paths):
                base = index / len(paths)
                if path.suffix.lower() == ".lif":
                    failures.append(f"{path.name}: raw Leica file — open it in Interrogator and export .ics first")
                    continue
                try:
                    self._progress(base, f"Queue {index + 1}/{len(paths)} — loading {path.name}")
                    dataset = load_dataset(path)
                    local_settings = AnalysisSettings(
                        **{**asdict(settings), "dt_ns": dataset.dt_ns, "rep_rate_mhz": dataset.rep_rate_mhz}
                    )
                    result = analyze_dataset(
                        dataset,
                        local_settings,
                        irf,
                        background,
                        lambda value, message, base=base: self._progress(
                            base + value / len(paths), f"{path.name}: {message}"
                        ),
                    )
                except Exception as exc:
                    failures.append(f"{path.name}: {exc}")
                    continue
                last_path = path
                if accumulate:
                    self.root.after(
                        0,
                        lambda name=path.name, g=result.g.copy(), s=result.s.copy(): self._append_phasor_accumulation(
                            name, g, s
                        ),
                    )
                self.root.after(
                    0,
                    lambda path=path, dataset=dataset, result=result: self._cache_queue_result(
                        path, dataset, result, limit
                    ),
                )
            self.root.after(0, lambda: self._queue_processing_finished(failures, len(paths), last_path))

        threading.Thread(target=worker, daemon=True, name="flim-queue-cache").start()

    def _queue_processing_finished(
        self, failures: list[str], attempted: int, last_path: Path | None
    ) -> None:
        # Same reasoning as _batch_finished's matching guard: only step in
        # when nothing at all was on screen to begin with, so a real
        # accumulated phasor cloud doesn't compute successfully and then
        # have nowhere to actually show itself.
        if self.result is None and last_path is not None and last_path in self.queue_results:
            self._show_cached_result(last_path)
        self._set_busy(False)
        if self.phasor_accumulate_queue_var.get() and self.result is not None:
            self._refresh_phasors()
        completed = attempted - len(failures)
        self.status_var.set(f"Queue processed — {completed}/{attempted} cached, click a ● entry to view")
        if failures:
            messagebox.showwarning(
                "Queue processed with warnings",
                f"Processed {completed} of {attempted} datasets.\n\n" + "\n".join(failures[:12]),
            )

    def load_path(self, path: Path) -> None:
        if self.busy:
            return
        if path.suffix.lower() == ".lif":
            self._open_in_leica_interrogator(path)
            return
        try:
            self.status_var.set(f"Loading {path.name}…")
            self.root.update_idletasks()
            dataset = load_dataset(path)
        except Exception as exc:
            messagebox.showerror("Unable to load data", f"{path.name}\n\n{exc}")
            self.status_var.set("Load failed")
            return
        if self.demo_defaults_active:
            if not self._default_ini_loaded:
                # Same reasoning as load_demo()'s matching guard: these are
                # this app's own hardcoded baseline, not necessarily what a
                # loaded default.ini asked for -- leave its values alone.
                self.background_counts_var.set("0")
                self.phasor_threshold_var.set("1000")
                self.fit_threshold_var.set("1000")
            self.demo_defaults_active = False
        self._set_dataset(dataset)

    def _set_dataset(self, dataset: FLIMDataset) -> None:
        self.dataset = dataset
        self.result = None
        self.roi_bounds = None
        self.roi_analysis = None
        self.selected_points = []
        self.point_analyses = []
        self.dt_var.set(f"{dataset.dt_ns:.8g}")
        self.rep_var.set(f"{dataset.rep_rate_mhz:.8g}")
        self.dataset_var.set(f"{dataset.name}  •  {dataset.y_size}×{dataset.x_size}×{dataset.time_bins}")
        temporal_extent = dataset.time_bins * dataset.dt_ns
        self.acquisition_var.set(
            f"Acquisition: {dataset.time_bins} TCSPC bins • {dataset.dt_ns:g} ns/bin • "
            f"{temporal_extent:g} ns period"
        )
        self._update_background_scale_summary()
        self.status_var.set(f"Loaded {dataset.name}")
        self.progress_var.set(0)
        self._show_raw_intensity()
        self._update_row_map_summary()
        self._column_map_control_edited()
        # Not just set_interactive(False): single point mode needs the maps
        # clickable even for a freshly loaded, not-yet-analysed dataset.
        self._sync_map_interactivity()
        self._update_metadata(dataset.metadata)

    def load_demo(self) -> None:
        if self.busy:
            return
        dataset = create_demo_dataset()
        self._set_queue([])
        self.irf = create_demo_irf(dataset.y_size, dataset.time_bins, dataset.dt_ns)
        self.background = None
        self.irf_var.set("IRF: synthetic calibration")
        self.background_var.set("Background: none")
        if not self._default_ini_loaded:
            # Tuned for this specific synthetic demo dataset's photon-count
            # scale -- skipped when a user's own default.ini was loaded at
            # startup, so it stays authoritative instead of being partly
            # overridden by demo-specific values a moment after applying it.
            self.background_counts_var.set("2")
            self.phasor_threshold_var.set("300")
            self.fit_threshold_var.set("300")
        self.demo_defaults_active = True
        self._set_dataset(dataset)
        self.run_current()

    def open_irf(self) -> None:
        if self.busy:
            return
        path = filedialog.askopenfilename(title="Open IRF calibration", filetypes=self._file_types())
        if not path:
            return
        try:
            self._load_irf_path(path)
            self.status_var.set("IRF calibration loaded")
        except Exception as exc:
            messagebox.showerror("Unable to load IRF", str(exc))

    def _load_irf_path(self, path: str) -> None:
        """Core of open_irf(), factored out so restoring a settings
        preset's remembered IRF path (see _restore_loaded_file_paths) can
        reuse the exact same loading/validation, not a re-implementation
        of it. Raises on failure; callers decide how to surface that."""

        loaded_irf = load_dataset(path)
        if self.dataset is not None and loaded_irf.time_bins != self.dataset.time_bins:
            raise ValueError(
                f"IRF has {loaded_irf.time_bins} time bins, but the sample has "
                f"{self.dataset.time_bins}."
            )
        if self.dataset is not None and not np.isclose(
            loaded_irf.dt_ns, self.dataset.dt_ns, rtol=1e-6, atol=1e-9
        ):
            raise ValueError(
                f"IRF time-bin width is {loaded_irf.dt_ns:g} ns, but the sample uses "
                f"{self.dataset.dt_ns:g} ns."
            )
        self.irf = loaded_irf
        self.irf_var.set(f"IRF: {self.irf.name} ({self.irf.time_bins} bins)")

    def open_background(self) -> None:
        if self.busy:
            return
        path = filedialog.askopenfilename(title="Open background acquisition", filetypes=self._file_types())
        if not path:
            return
        try:
            self._load_background_path(path)
            self.status_var.set("Background acquisition loaded")
        except Exception as exc:
            messagebox.showerror("Unable to load background", str(exc))

    def _load_background_path(self, path: str) -> None:
        """Core of open_background() -- see _load_irf_path's docstring,
        same reasoning."""

        loaded_background = load_dataset(path)
        if self.dataset is not None and loaded_background.time_bins != self.dataset.time_bins:
            raise ValueError(
                f"Background has {loaded_background.time_bins} time bins, but the sample has "
                f"{self.dataset.time_bins}."
            )
        if self.dataset is not None and not np.isclose(
            loaded_background.dt_ns, self.dataset.dt_ns, rtol=1e-6, atol=1e-9
        ):
            raise ValueError(
                f"Background time-bin width is {loaded_background.dt_ns:g} ns, but the sample "
                f"uses {self.dataset.dt_ns:g} ns."
            )
        self.background = loaded_background
        self.background_var.set(f"Background: {self.background.name} ({self.background.time_bins} bins)")
        self._update_background_scale_summary()
        self._on_spatial_bin_setting_changed()  # refresh the pre-analysis intensity preview's own subtraction

    def _current_loaded_file_paths(self) -> LoadedFilePaths:
        """Snapshot of which decay/IRF/background files are currently
        loaded, for save_settings_ini's ``file_paths=``. Empty for
        anything unset or not actually loaded from a real file (e.g. the
        synthetic demo IRF, or a dataset built in-memory some other way)
        -- both have no ``source_path``."""

        def source(dataset: FLIMDataset | None) -> str:
            return str(dataset.source_path) if dataset is not None and dataset.source_path else ""

        return LoadedFilePaths(
            decay_path=source(self.dataset),
            irf_path=source(self.irf),
            background_path=source(self.background),
        )

    def _restore_loaded_file_paths(self, file_paths: LoadedFilePaths) -> list[str]:
        """The inverse of _current_loaded_file_paths: (re)loads whichever
        of decay/IRF/background a settings preset remembered, in that
        order (IRF/background validate their time-bin count/width against
        ``self.dataset``, so the decay must load first for that check to
        mean anything). An empty path or a load failure (moved/deleted
        file, mismatched time bins, ...) is skipped with a warning rather
        than aborting the rest -- restoring what's still possible beats
        restoring nothing over one stale path. Returns those warnings
        (empty list if everything requested loaded cleanly, including
        the trivial case of no paths at all)."""

        warnings: list[str] = []
        if file_paths.decay_path:
            path = Path(file_paths.decay_path)
            if path.is_file():
                try:
                    self.load_path(path)
                except Exception as exc:
                    warnings.append(f"Decay file '{path.name}' could not be reloaded: {exc}")
            else:
                warnings.append(f"Decay file no longer found: {file_paths.decay_path}")
        if file_paths.irf_path:
            path = Path(file_paths.irf_path)
            if path.is_file():
                try:
                    self._load_irf_path(str(path))
                except Exception as exc:
                    warnings.append(f"IRF file '{path.name}' could not be reloaded: {exc}")
            else:
                warnings.append(f"IRF file no longer found: {file_paths.irf_path}")
        if file_paths.background_path:
            path = Path(file_paths.background_path)
            if path.is_file():
                try:
                    self._load_background_path(str(path))
                except Exception as exc:
                    warnings.append(f"Background file '{path.name}' could not be reloaded: {exc}")
            else:
                warnings.append(f"Background file no longer found: {file_paths.background_path}")
        return warnings

    def _show_raw_intensity(self) -> None:
        if self.dataset is None:
            return
        cube = self.dataset.decay.astype(np.float64)
        multiplier = 0.0
        background_applied = False
        if self.background is not None:
            try:
                bg_settings = self._background_settings_preview()
                bg_data = (
                    self.background.decay if isinstance(self.background, FLIMDataset) else self.background
                )
                multiplier, _ = resolve_background_multiplier(self.dataset, self.background, bg_settings)
                prepared_background = _prepare_background(bg_data, cube)
                # Mirrors analyze_dataset()'s own background-subtraction
                # formula exactly (see processing.py): subtract-then-clip
                # *per time bin*, not a subtraction of already-time-summed
                # totals below -- clipping is non-linear with respect to
                # that order, so doing it after summing would give a
                # different (wrong) answer whenever the background exceeds
                # the signal in some bins but not overall. This is the
                # whole point of doing this here at all: so the threshold
                # you pick against this preview is the threshold the real
                # "Analyze" run will actually apply, not one against a
                # differently-scaled, unsubtracted intensity.
                cube = cube - multiplier * prepared_background - bg_settings.background_counts
                if bg_settings.background_clip_negative:
                    cube = np.clip(cube, 0.0, None)
                background_applied = True
            except Exception:
                # Any background-resolution problem (mismatched shapes, bad
                # metadata, ...) -- fall back to the raw cube rather than
                # blocking the whole pre-analysis preview over it; the real
                # "Analyze" run surfaces the same error properly, with a
                # dialog, when it's actually run.
                cube = self.dataset.decay.astype(np.float64)
        raw = cube.sum(axis=-1)
        try:
            radius = int(self.bin_var.get())
        except ValueError:
            radius = 0
        # Sum-then-bin rather than bin-then-sum: rolling_spatial_bin's window
        # sum and the time-axis sum above are both plain sums, so the two
        # orders are numerically identical -- doing it this way runs the
        # O(pixels) integral-image pass on a single 2-D image instead of the
        # full (height, width, time_bins) cube, which matters here since this
        # reruns on every spatial-bin-radius change, not just once per load.
        # (The background subtraction above still has to happen on the full
        # cube first, per-bin -- see the comment there.) Kept in sync with
        # what analyze_dataset() itself does spatially (see processing.py)
        # so this pre-analysis preview lines up with the real processed
        # intensity map once "Analyze" actually runs.
        binned = rolling_spatial_bin(raw[..., None], radius)[..., 0]
        self._raw_binned_intensity = binned
        log_scale = bool(self.intensity_log_var.get())
        window = 2 * radius + 1
        bin_note = "no spatial binning" if radius == 0 else f"spatial bin radius {radius} ({window}x{window})"
        projection_label = f"Background-subtracted (×{multiplier:g})" if background_applied else "Raw"
        self.intensity_view.set_image(
            # Signed (unclipped) background subtraction can leave a pixel with
            # a slightly negative net total; show it as zero rather than
            # feeding a negative into the log scale.
            colourise(np.clip(binned, 0.0, None), "inferno", log=log_scale),
            f"{projection_label} photon-count projection, {bin_note} "
            f"({'log' if log_scale else 'linear'} scale)",
        )
        self.live_threshold_scale.configure(to=max(5000.0, float(np.nanpercentile(binned, 99.5))))
        self._refresh_secondary_map()
        for view in self._map_views():
            view.set_selection(None)
            view.set_points([])
            view.set_row_overlay([])
        raw_time = (np.arange(self.dataset.time_bins) + 0.5) * self.dataset.dt_ns
        raw_sum = self.dataset.decay.astype(np.float64).sum(axis=(0, 1))
        raw_series = [("Whole-image summed decay", raw_sum, ACCENT)]
        raw_plot = line_plot(raw_time, raw_series, title="Raw whole-image decay", y_label="Raw counts / bin")
        raw_plot_detailed = line_plot(
            raw_time, raw_series, size=(1200, 680), title="Raw whole-image decay", y_label="Raw counts / bin"
        )
        self.roi_decay_view.set_image(raw_plot, "Run analysis to enable ROI selection")
        self.decay_view.set_image(raw_plot_detailed, "Run analysis to enable ROI selection")
        self.irf_decay_view.set_image(None)
        self.phasor_view.set_image(None)
        self.histogram_view.set_image(None)
        self.overview_phasor_view.set_image(None)
        self.overview_histogram_view.set_image(None)
        self.valid_var.set("—")
        self.mean_tau_var.set("—")
        self.global_tau_var.set("—")
        self.max_counts_var.set(f"{np.nanmax(binned):,.0f}")
        self.mean_chi_var.set("—")
        self.selection_var.set("Run analysis, then drag on a map to select a region.")
        self.point_summary_var.set("Run analysis, then click a map to add comparison points.")
        self.roi_summary_var.set("Histogram scope: whole image")
        for variable in (
            self.roi_pixels_var,
            self.roi_mean_tau_var,
            self.roi_median_tau_var,
            self.roi_aggregate_tau_var,
            self.roi_photons_var,
        ):
            variable.set("—")

    def _on_spatial_bin_setting_changed(self, *_args) -> None:
        # Only meaningful pre-analysis, where _show_raw_intensity is the
        # thing computing _raw_binned_intensity in the first place -- once a
        # real result exists its maps stay whatever they were as of the last
        # "Analyze" run, same as any other setting change.
        if self.result is not None or self.dataset is None or self.busy:
            return
        self._show_raw_intensity()

    def _settings(self) -> AnalysisSettings:
        try:
            initial_taus = tuple(
                float(value.strip()) for value in self.initial_taus_var.get().split(",") if value.strip()
            )
            if self.row_map_enabled_var.get():
                row_map = self._row_map_from_controls()
                if not row_map.rows:
                    raise ValueError("Row interpolation is enabled but no detector rows are defined.")
                if self.dataset is not None:
                    row_map.resolve_rows(self.dataset.y_size)
            else:
                # A disabled, partly edited map must be inert. Preserve valid
                # editor values for settings provenance when possible, without
                # allowing malformed dormant text to block unrelated analysis.
                try:
                    row_map = self._row_map_from_controls()
                except Exception:
                    row_map = RowInterpolationMap()
            columns = parse_row_spec(self.column_map_columns_var.get(), index_base=0)
            if self.column_map_enabled_var.get():
                if not columns:
                    raise ValueError("Column interpolation is enabled but no detector columns are defined.")
                if self.dataset is not None and max(columns) >= self.dataset.x_size:
                    raise ValueError(
                        f"Detector column {max(columns)} is outside 0..{self.dataset.x_size - 1}."
                    )
            return AnalysisSettings(
                method=self.METHOD_LABELS[self.method_var.get()],
                dt_ns=float(self.dt_var.get()),
                rep_rate_mhz=float(self.rep_var.get()),
                background_counts=float(self.background_counts_var.get()),
                background_scale_mode=(
                    "auto" if self.background_scale_mode_var.get().startswith("Auto") else "manual"
                ),
                background_multiplier=float(self.background_multiplier_var.get()),
                background_auto_fallback=float(self.background_multiplier_var.get()),
                background_clip_negative=bool(self.background_clip_var.get()),
                phasor_min_counts=float(self.phasor_threshold_var.get()),
                fit_min_counts=float(self.fit_threshold_var.get()),
                fit_start_bin=int(self.start_bin_var.get()),
                fit_end_bin=int(self.end_bin_var.get()),
                native_fit_bins=int(self.native_fit_bins_var.get()),
                native_bin_width_ns=(
                    -1.0 if self.native_bin_width_mode_var.get().startswith("Auto")
                    else float(self.native_bin_width_var.get())
                ),
                native_irf_baseline_mode={
                    "Per-pixel median": "median",
                    "Tail median": "tail_median",
                    "None": "none",
                }[self.native_irf_baseline_var.get()],
                native_fit_workers=int(self.native_fit_workers_var.get()),
                spatial_bin=int(self.bin_var.get()),
                smoothing_passes=int(self.smooth_var.get()),
                timing_offset_bins=float(self.offset_var.get()),
                interpolate_outliers=bool(self.interpolate_var.get()),
                interpolate_rows=bool(self.row_map_enabled_var.get()),
                interpolation_rows=row_map.rows,
                interpolation_radius=row_map.radius,
                interpolation_coordinate_mode=row_map.coordinate_mode,
                interpolation_map_name=row_map.name,
                interpolation_map_source=row_map.source_path,
                interpolation_map_sha256=row_map.sha256,
                interpolation_expected_source_height=row_map.expected_source_height or 0,
                interpolate_columns=bool(self.column_map_enabled_var.get()),
                interpolation_columns=columns,
                column_interpolation_radius=int(self.column_map_radius_var.get()),
                phasor_heatmap_bins=int(self.phasor_heatmap_bins_var.get()),
                lifetime_scale_mode=(
                    "auto" if self.lifetime_scale_mode_var.get().startswith("Auto") else "manual"
                ),
                lifetime_min_ns=float(self.life_min_var.get()),
                lifetime_max_ns=float(self.life_max_var.get()),
                histogram_bins=int(self.hist_bins_var.get()),
                detailed_histogram_bins=int(self.detailed_hist_bins_var.get()),
                lifetime_lut_path=self.lifetime_lut_path_var.get(),
                lifetime_lut_inverse=bool(self.lifetime_lut_inverse_var.get()),
                lifetime_lut_remove_out_of_range=bool(self.lifetime_lut_remove_var.get()),
                irf_source=(
                    "synthetic_gaussian"
                    if self.irf_source_var.get().startswith("Synthetic")
                    else "file"
                ),
                synthetic_irf_fwhm_ns=float(self.synthetic_irf_fwhm_var.get()),
                max_global_pixels=int(self.max_global_pixels_var.get()),
                model_order=int(self.model_order_var.get().split()[0]),
                fit_distribution_model=self.DISTRIBUTION_MODEL_LABELS[self.fit_distribution_model_var.get()],
                fit_background_mode=self.FIT_BACKGROUND_LABELS[self.fit_background_mode_var.get()],
                fit_background_value=float(self.fit_background_value_var.get()),
                fit_noise_model=self.NOISE_LABELS[self.noise_model_var.get()],
                fit_irf_mode=self.IRF_MODE_LABELS[self.irf_mode_var.get()],
                irf_integrate_per_column=bool(self.irf_integrate_per_column_var.get()),
                global_iterations=int(self.global_iterations_var.get()),
                initial_taus_ns=initial_taus,
                native_dll_path=self.native_dll_var.get(),
                global_model_path=self.global_model_path_var.get(),
            )
        except (ValueError, KeyError) as exc:
            raise ValueError(f"One or more analysis settings is invalid: {exc}") from exc

    def _load_default_ini_at_startup(self) -> None:
        """Auto-load default.ini from the app's main folder, if present.

        Runs once near the end of __init__, before the demo dataset's own
        deferred (.after()) load -- see the _default_ini_loaded guards in
        load_demo()/load_path(). Never raises: a bad or missing file just
        leaves the built-in hardcoded defaults in place, reported on the
        status line rather than as a startup-blocking dialog.
        """

        path = main_folder() / DEFAULT_INI_NAME
        if not path.is_file():
            return
        try:
            settings = load_settings_ini(path)
            warnings = self._apply_settings_to_ui(settings)
        except Exception as exc:
            self.settings_preset_status_var.set(f"Could not load {DEFAULT_INI_NAME}: {exc}")
            return
        self._default_ini_loaded = True
        try:
            file_paths = load_file_paths_ini(path)
            warnings = warnings + self._restore_loaded_file_paths(file_paths)
        except Exception:
            pass  # a settings-only preset (or a corrupt files section) -- not fatal to startup
        message = f"Loaded {DEFAULT_INI_NAME} from {path.parent}."
        if warnings:
            message += f" {len(warnings)} item(s) skipped -- see status bar."
            self.status_var.set("; ".join(warnings[:3]))
        self.settings_preset_status_var.set(message)

    def _load_settings_ini_dialog(self) -> None:
        if self.busy:
            return
        path = filedialog.askopenfilename(
            title="Load FLIM Studio settings",
            filetypes=[("FLIM Studio settings", "*.ini"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            settings = load_settings_ini(path)
            warnings = self._apply_settings_to_ui(settings)
        except Exception as exc:
            messagebox.showerror("Could not load settings", str(exc))
            return
        try:
            file_paths = load_file_paths_ini(path)
            warnings = warnings + self._restore_loaded_file_paths(file_paths)
        except Exception as exc:
            warnings = warnings + [f"Could not read remembered file paths: {exc}"]
        name = Path(path).name
        self.settings_preset_status_var.set(f"Loaded settings from {name}.")
        self.status_var.set(f"Settings loaded from {name}.")
        if warnings:
            messagebox.showwarning(
                "Some settings could not be restored",
                f"{name} loaded, but:\n\n" + "\n".join(f"- {item}" for item in warnings),
            )

    def _save_settings_ini_dialog(self) -> None:
        try:
            settings = self._settings()
        except ValueError as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return
        path = filedialog.asksaveasfilename(
            title="Save FLIM Studio settings",
            defaultextension=".ini",
            filetypes=[("FLIM Studio settings", "*.ini"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            save_settings_ini(settings, path, file_paths=self._current_loaded_file_paths())
        except OSError as exc:
            messagebox.showerror("Could not save settings", str(exc))
            return
        name = Path(path).name
        self.settings_preset_status_var.set(f"Saved settings to {name}.")
        self.status_var.set(f"Settings saved to {name}.")

    def _save_settings_as_default(self) -> None:
        try:
            settings = self._settings()
        except ValueError as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return
        path = main_folder() / DEFAULT_INI_NAME
        try:
            save_settings_ini(settings, path, file_paths=self._current_loaded_file_paths())
        except OSError as exc:
            messagebox.showerror("Could not save default settings", str(exc))
            return
        self._default_ini_loaded = True
        self.settings_preset_status_var.set(
            f"Saved as {DEFAULT_INI_NAME} ({path.parent}) -- used automatically on every startup."
        )
        self.status_var.set(f"Saved current settings as {DEFAULT_INI_NAME}.")

    def _reset_settings_to_defaults(self) -> None:
        try:
            self._apply_settings_to_ui(AnalysisSettings())
        except Exception as exc:
            messagebox.showerror("Could not reset settings", str(exc))
            return
        self.settings_preset_status_var.set("Reset to FLIM Studio's built-in defaults.")
        self.status_var.set("Settings reset to built-in defaults.")

    def _apply_settings_to_ui(self, settings: AnalysisSettings) -> list[str]:
        """The inverse of _settings(): push every AnalysisSettings field
        back onto its sidebar control.

        Best-effort: an unrecognised label for a combo-backed field (a
        hand-edited .ini with a typo, or one saved by a future version with
        a choice this one doesn't know) is skipped and reported in the
        returned warning list rather than aborting the whole restore over
        one bad field. Does not touch anything about the currently loaded
        dataset/IRF/background/queue -- see settings_ini.py's docstring for
        why those aren't part of this at all.
        """

        warnings: list[str] = []

        def apply_label(labels: dict[str, object], value: object, field_name: str, var: tk.StringVar) -> None:
            label = self._label_for_value(labels, value)
            if label is None:
                warnings.append(f"Unrecognised {field_name} {value!r} -- left as {var.get()!r}.")
                return
            var.set(label)

        apply_label(self.METHOD_LABELS, settings.method, "method", self.method_var)
        self.dt_var.set(f"{settings.dt_ns:g}")
        self.rep_var.set(f"{settings.rep_rate_mhz:g}")
        self.background_counts_var.set(f"{settings.background_counts:g}")
        apply_label(
            {"Auto from loops / time": "auto", "Manual multiplier": "manual"},
            settings.background_scale_mode, "background scaling mode", self.background_scale_mode_var,
        )
        self.background_multiplier_var.set(f"{settings.background_multiplier:g}")
        self.background_clip_var.set(bool(settings.background_clip_negative))
        self.phasor_threshold_var.set(f"{settings.phasor_min_counts:g}")
        self.fit_threshold_var.set(f"{settings.fit_min_counts:g}")
        self.start_bin_var.set(str(int(settings.fit_start_bin)))
        self.end_bin_var.set(str(int(settings.fit_end_bin)))
        self.native_fit_bins_var.set(str(int(settings.native_fit_bins)))
        if settings.native_bin_width_ns > 0:
            self.native_bin_width_mode_var.set("Manual")
            self.native_bin_width_var.set(f"{settings.native_bin_width_ns:g}")
        else:
            self.native_bin_width_mode_var.set("Auto")
        apply_label(
            {"Per-pixel median": "median", "Tail median": "tail_median", "None": "none"},
            settings.native_irf_baseline_mode, "native IRF pedestal mode", self.native_irf_baseline_var,
        )
        self.native_fit_workers_var.set(str(max(1, min(int(settings.native_fit_workers), 32))))
        self.bin_var.set(str(max(0, min(int(settings.spatial_bin), 4))))
        self.smooth_var.set(str(max(0, min(int(settings.smoothing_passes), 3))))
        self.offset_var.set(f"{settings.timing_offset_bins:g}")
        self.interpolate_var.set(bool(settings.interpolate_outliers))

        self.row_map_enabled_var.set(bool(settings.interpolate_rows))
        try:
            self._set_row_map_controls(
                RowInterpolationMap(
                    rows=tuple(int(value) for value in settings.interpolation_rows),
                    radius=int(settings.interpolation_radius),
                    coordinate_mode=settings.interpolation_coordinate_mode,
                    name=settings.interpolation_map_name or "Custom detector-row map",
                    source_path=settings.interpolation_map_source,
                    sha256=settings.interpolation_map_sha256,
                    expected_source_height=settings.interpolation_expected_source_height or None,
                )
            )
        except Exception as exc:
            warnings.append(f"Detector-row map settings could not be applied ({exc}); left unchanged.")

        self.column_map_enabled_var.set(bool(settings.interpolate_columns))
        self.column_map_columns_var.set(format_row_spec(settings.interpolation_columns))
        self.column_map_radius_var.set(str(int(settings.column_interpolation_radius)))

        self.phasor_heatmap_bins_var.set(str(int(settings.phasor_heatmap_bins)))
        apply_label(
            {"Auto": "auto", "Manual": "manual"}, settings.lifetime_scale_mode,
            "lifetime scaling mode", self.lifetime_scale_mode_var,
        )
        self.life_min_var.set(f"{settings.lifetime_min_ns:g}")
        self.life_max_var.set(f"{settings.lifetime_max_ns:g}")
        self.hist_bins_var.set(str(int(settings.histogram_bins)))
        self.detailed_hist_bins_var.set(str(int(settings.detailed_histogram_bins)))

        lut_path = str(settings.lifetime_lut_path).strip()
        if lut_path:
            try:
                from .lut import load_imagej_lut

                load_imagej_lut(lut_path)
            except Exception as exc:
                warnings.append(f"Lifetime LUT {lut_path!r} could not be loaded ({exc}); using the built-in LUT.")
                lut_path = ""
        self.lifetime_lut_path_var.set(lut_path)
        self.lifetime_lut_summary_var.set(
            f"{Path(lut_path).name} — external 256-colour LUT" if lut_path
            else "Reverse Rainbow (DM).lut — original LabVIEW LUT"
        )
        self.lifetime_lut_inverse_var.set(bool(settings.lifetime_lut_inverse))
        self.lifetime_lut_remove_var.set(bool(settings.lifetime_lut_remove_out_of_range))

        self.irf_source_var.set(
            "Synthetic (Gaussian)" if settings.irf_source == "synthetic_gaussian" else "Loaded file"
        )
        self.synthetic_irf_fwhm_var.set(f"{settings.synthetic_irf_fwhm_ns:g}")

        self.model_order_var.set(
            {1: "1 — mono", 2: "2 — bi", 3: "3 — tri"}.get(int(settings.model_order), "1 — mono")
        )
        apply_label(
            self.DISTRIBUTION_MODEL_LABELS, getattr(settings, "fit_distribution_model", 0),
            "distribution model", self.fit_distribution_model_var,
        )
        apply_label(self.NOISE_LABELS, settings.fit_noise_model, "noise model", self.noise_model_var)
        apply_label(
            self.FIT_BACKGROUND_LABELS, settings.fit_background_mode, "fit background mode",
            self.fit_background_mode_var,
        )
        self.fit_background_value_var.set(f"{settings.fit_background_value:g}")
        apply_label(self.IRF_MODE_LABELS, settings.fit_irf_mode, "IRF mode", self.irf_mode_var)
        self.irf_integrate_per_column_var.set(bool(settings.irf_integrate_per_column))
        self.global_iterations_var.set(str(int(settings.global_iterations)))
        self.max_global_pixels_var.set(str(int(settings.max_global_pixels)))
        self.initial_taus_var.set(", ".join(f"{value:g}" for value in settings.initial_taus_ns))

        if str(settings.native_dll_path).strip():
            self.native_dll_var.set(settings.native_dll_path)
            self.native_status_var.set(f"Native backend: {Path(settings.native_dll_path).name}")

        model_path = str(settings.global_model_path).strip()
        self.global_model_path_var.set(model_path)
        if model_path:
            try:
                model = load_global_model(model_path)
            except Exception as exc:
                warnings.append(f"Global model {model_path!r} could not be loaded ({exc}).")
                self.trained_model = None
                self.global_model_status_var.set("No trained model loaded")
            else:
                # Deliberately not _activate_trained_model(): that also
                # overwrites method/model-order/noise/background/IRF-mode
                # fields *from the model*, which would silently override
                # whatever this same settings file just asked those to be.
                # analyze_dataset() re-reads this same path from disk at run
                # time regardless of self.trained_model, so this is only
                # for the status line here, not for run-time correctness.
                self.trained_model = model
                self.global_model_status_var.set(model.summary())
        else:
            self.trained_model = None
            self.global_model_status_var.set("No trained model loaded")

        return warnings

    def _set_busy(self, value: bool) -> None:
        self.busy = value
        state = "disabled" if value else "normal"
        self.header_analyze.configure(state=state)
        self.sidebar_analyze.configure(state=state)
        self._sync_map_interactivity()

    def _progress(self, value: float, message: str) -> None:
        self.root.after(0, lambda: (self.progress_var.set(value), self.status_var.set(message)))

    def run_current(self) -> None:
        if self.busy:
            return
        if self.dataset is None:
            messagebox.showinfo("No data", "Open a FLIM decay cube first.")
            return
        try:
            settings = self._settings()
        except ValueError as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return
        dataset, irf, background = self.dataset, self.irf, self.background
        self._set_busy(True)
        self.status_var.set("Starting analysis…")
        self.progress_var.set(0)

        def worker() -> None:
            try:
                result = analyze_dataset(dataset, settings, irf, background, self._progress)
            except Exception as exc:
                details = traceback.format_exc()
                self.root.after(0, lambda exc=exc, details=details: self._analysis_failed(exc, details))
                return
            self.root.after(0, lambda: self._analysis_finished(result))

        threading.Thread(target=worker, daemon=True, name="flim-analysis").start()

    def _analysis_finished(self, result: AnalysisResult) -> None:
        self.result = result
        self.progress_var.set(1)
        self.status_var.set(f"Analysis complete — {result.dataset_name}")
        self._refresh_result_views()
        self._set_busy(False)

    def _analysis_failed(self, exc: Exception, details: str) -> None:
        self._set_busy(False)
        self.progress_var.set(0)
        self.status_var.set("Analysis failed")
        messagebox.showerror("Analysis failed", f"{exc}\n\nTechnical details:\n{details[-1800:]}")

    def _refresh_result_views(self) -> None:
        if self.result is None:
            return
        result = self.result
        synthetic_irf = result.metadata.get("synthetic_irf")
        if synthetic_irf:
            source_note = {
                "manual_override": "pinned",
                "rise_time_10_90": "from this decay's 10-90% rise time",
                "fallback_bin_width": "fallback -- rise too short/flat to measure",
            }.get(synthetic_irf.get("estimation_method"), synthetic_irf.get("estimation_method"))
            self.synthetic_irf_summary_var.set(
                f"Synthetic IRF used: FWHM {synthetic_irf['fwhm_ns']:.3g} ns, "
                f"centred {synthetic_irf['center_ns']:.3g} ns ({source_note})"
            )
        else:
            self.synthetic_irf_summary_var.set("")
        self.roi_bounds = None
        self.roi_analysis = None
        self.selected_points = []
        self.point_analyses = []
        self._refresh_intensity_view()
        self._lifetime_image = lifetime_image(result)
        self._composite_image = composite_image(result)
        self._chi_square_image = chi_square_image(result)
        self._component_images = []
        self._amplitude_component_images = []
        components = result.fit_diagnostics.get("tau_components_ns")
        if components is not None and np.asarray(components).ndim == 3:
            self._component_images = [
                component_lifetime_image(result, index)
                for index in range(np.asarray(components).shape[0])
            ]
            amplitudes = result.fit_diagnostics.get("amplitude_components")
            if amplitudes is not None and np.asarray(amplitudes).shape == np.asarray(components).shape:
                self._amplitude_component_images = [
                    component_amplitude_image(result, index)
                    for index in range(np.asarray(components).shape[0])
                ]
        choices = ["Intensity", "Intensity with threshold", "Lifetime", "Composite", "Chi-square"]
        if len(self._component_images) > 1:
            choices[1:1] = [f"Lifetime tau{index + 1}" for index in range(len(self._component_images))]
        self.map_mode_combo.configure(values=choices)
        if self.map_mode_var.get() not in choices:
            self.map_mode_var.set("Lifetime")
        active_threshold = float(result.metadata.get("count_thresholds", {}).get("active_min_counts", 0.0))
        self.live_threshold_var.set(f"{active_threshold:g}")
        self.live_threshold_scale.configure(to=max(5000.0, float(np.nanpercentile(result.intensity, 99.5))))
        self.live_threshold_scale.set(active_threshold)
        self._refresh_secondary_map()
        self._update_lifetime_scale_summary()
        self._refresh_multiexponential_views()
        self.irf_decay_view.set_image(irf_decay_overlay_plot(result, size=(1200, 680)))
        summary = result.summary()
        self.valid_var.set(f"{summary['valid_pixels']:,}")
        self.mean_tau_var.set(
            f"{summary['mean_lifetime_ns']:.3f} ns" if summary["mean_lifetime_ns"] is not None else "—"
        )
        self.global_tau_var.set(
            f"{summary['global_tau_ns']:.3f} ns" if summary["global_tau_ns"] is not None else "—"
        )
        self.max_counts_var.set(f"{summary['max_intensity']:,.0f}")
        chi_mean, _chi_median, chi_count = chi_square_stats(result)
        self.mean_chi_var.set(f"{chi_mean:.3f} ({chi_count:,} px)" if chi_mean is not None else "—")
        for view in self._map_views():
            view.set_selection(None)
            view.set_points([])
        self._update_roi_details(None)
        self._refresh_point_display()
        self._update_metadata({"dataset": result.metadata, "analysis": summary})

    def _refresh_intensity_view(self) -> None:
        if self.result is None:
            return
        log_scale = bool(self.intensity_log_var.get())
        self.intensity_view.set_image(
            colourise(self.result.intensity, "inferno", log=log_scale),
            f"{'Log-scaled' if log_scale else 'Linear-scaled'} photon counts",
        )

    def _intensity_scale_changed(self) -> None:
        if self.result is None:
            self._show_raw_intensity()
            return
        self._refresh_intensity_view()
        if self.map_mode_var.get() in {"Intensity", "Intensity with threshold"}:
            self._refresh_secondary_map()

    def _refresh_secondary_map_pre_analysis(self) -> None:
        """Secondary-map preview before the first "Analyze" run.

        Only Intensity and Intensity-with-threshold make sense here --
        Lifetime/Composite/Chi-square genuinely need a fit. This lets
        thresholding be picked *before* running analysis (not just after,
        which previously left it unusable until the first run), using
        photon counts summed over time and spatially binned exactly as
        analyze_dataset() will bin them (see _show_raw_intensity) --
        including that same background subtraction (multiplier/scale mode/
        scalar counts-per-bin, all read live from the sidebar) when a
        background is loaded, so the threshold you pick here is the
        threshold the real run will actually apply, not one against a
        differently-scaled, unsubtracted intensity. Still only an
        approximation of the real post-fit map in other respects: the
        timing offset and row/column interpolation aren't applied here,
        only after a real run.
        """

        mode = self.map_mode_var.get()
        if self.dataset is None or self._raw_binned_intensity is None or mode not in (
            "Intensity", "Intensity with threshold",
        ):
            self.secondary_view.set_image(None, "Run analysis to calculate lifetime")
            return
        intensity = self._raw_binned_intensity
        log_scale = bool(self.intensity_log_var.get())
        try:
            radius = int(self.bin_var.get())
        except ValueError:
            radius = 0
        window = 2 * radius + 1
        bin_note = "no spatial binning" if radius == 0 else f"spatial bin radius {radius} ({window}x{window})"
        if mode == "Intensity":
            self.secondary_view.title_label.configure(text="Intensity")
            self.secondary_view.set_image(
                colourise(intensity, "inferno", log=log_scale),
                f"Raw photon counts, {bin_note} ({'log' if log_scale else 'linear'} scale) — run analysis for "
                "the processed map",
            )
            return
        try:
            threshold = max(0.0, float(self.live_threshold_var.get()))
        except ValueError:
            threshold = 0.0
        omitted = int(np.count_nonzero(intensity < threshold))
        total = int(intensity.size)
        self.secondary_view.title_label.configure(text="Intensity with threshold")
        self.secondary_view.set_image(
            threshold_intensity_image(intensity, threshold, log=log_scale),
            (
                f"Preview on raw counts, {bin_note} — red = omitted below {threshold:g}; "
                f"{omitted:,}/{total:,} pixels omitted ({'log' if log_scale else 'linear'} scale). "
                "Run analysis for the processed threshold map."
            ),
        )

    def _refresh_secondary_map(self) -> None:
        if self.result is None:
            self._refresh_secondary_map_pre_analysis()
            return
        mode = self.map_mode_var.get()
        if mode == "Intensity":
            log_scale = bool(self.intensity_log_var.get())
            self.secondary_view.title_label.configure(text="Intensity")
            self.secondary_view.set_image(
                colourise(self.result.intensity, "inferno", log=log_scale),
                f"Processed intensity before the display threshold ({'log' if log_scale else 'linear'} scale)",
            )
        elif mode == "Composite":
            self.secondary_view.title_label.configure(text="Intensity × lifetime composite")
            self.secondary_view.set_image(self._composite_image, "LabVIEW HSV lifetime × linear intensity")
        elif mode == "Chi-square":
            scope_bounds = self.roi_bounds
            chi_mean, chi_median, chi_count = chi_square_stats(self.result, scope_bounds)
            scope_label = "whole image" if scope_bounds is None else "selected ROI"
            self.secondary_view.title_label.configure(text="Reduced χ² (goodness of fit)")
            self.secondary_view.set_image(
                self._chi_square_image,
                (
                    f"{scope_label}: mean {chi_mean:.3f}, median {chi_median:.3f} ({chi_count:,} px); "
                    "~1 is a good fit, higher is worse"
                    if chi_mean is not None
                    else f"{scope_label}: no finite χ² values"
                ),
            )
        elif mode == "Intensity with threshold":
            try:
                threshold = max(0.0, float(self.live_threshold_var.get()))
            except ValueError:
                threshold = 0.0
            threshold_signal = np.asarray(
                self.result.fit_diagnostics.get("threshold_signal_map", self.result.intensity),
                dtype=np.float64,
            )
            omitted = int(np.count_nonzero(threshold_signal < threshold))
            total = int(threshold_signal.size)
            basis = str(
                self.result.metadata.get("count_thresholds", {}).get(
                    "preview_signal_basis", "all processed TCSPC bins"
                )
            )
            log_scale = bool(self.intensity_log_var.get())
            self.secondary_view.title_label.configure(text="Intensity with threshold")
            self.secondary_view.set_image(
                threshold_intensity_image(threshold_signal, threshold, log=log_scale),
                (
                    f"Red = omitted below {threshold:g}; {omitted:,}/{total:,} pixels omitted; {basis} "
                    f"({'log' if log_scale else 'linear'} scale)"
                ),
            )
        elif mode.startswith("Lifetime tau"):
            index = int(mode.replace("Lifetime tau", "")) - 1
            if 0 <= index < len(self._component_images):
                self.secondary_view.title_label.configure(text=f"Native lifetime component tau{index + 1}")
                self.secondary_view.set_image(
                    self._component_images[index],
                    "FLIMLib component map; components sorted longest-to-shortest",
                )
        else:
            label = lifetime_label(self.result)
            life_min, life_max = lifetime_display_range(self.result)
            scale_label = (
                "auto 1st–99th percentile"
                if self.result.settings.lifetime_scale_mode == "auto"
                else "manual"
            )
            self.secondary_view.title_label.configure(text=f"Lifetime map — {label} (ns)")
            self.secondary_view.set_image(
                self._lifetime_image,
                (
                    f"{life_min:g}–{life_max:g} ns ({scale_label}) • "
                    f"{Path(self.result.settings.lifetime_lut_path).name if self.result.settings.lifetime_lut_path else 'Reverse Rainbow (DM)'}"
                ),
            )
        self._sync_map_overlays()

    def _update_lifetime_scale_summary(self) -> None:
        if self.result is None:
            self.lifetime_scale_summary_var.set(
                f"Manual range: {self.life_min_var.get()}–{self.life_max_var.get()} ns"
            )
            return
        lo, hi = lifetime_display_range(self.result)
        if self.result.settings.lifetime_scale_mode == "auto":
            self.lifetime_scale_summary_var.set(
                f"Auto display range: {lo:.3g}–{hi:.3g} ns (1st–99th percentiles)"
            )
        else:
            self.lifetime_scale_summary_var.set(f"Manual display range: {lo:g}–{hi:g} ns")

    def _method_model_hint_changed(self, *_args) -> None:
        """Warn when Model order is set for bi/tri but the method won't use it.

        Model order only takes effect for the two FLIMLib native methods; it
        is silently ignored otherwise, which previously made bi/tri-exponential
        fitting appear to do nothing.
        """

        current_method = self.METHOD_LABELS.get(self.method_var.get(), "")
        is_native_method = current_method in {
            "flimlib_reconvolution",
            "flimlib_global",
            "flimlib_apply_global",
        }
        distribution_model = self.DISTRIBUTION_MODEL_LABELS.get(self.fit_distribution_model_var.get())
        # Only the single-component stretched exponential (1) forces Model
        # order to 1 -- multi-stretched (2) is exactly what Model order
        # 2-3 there means (that many independently-distributed components),
        # so it must keep the combo free, unlike single-component stretched.
        stretched = distribution_model == 1
        any_stretched = distribution_model in (1, 2)
        model_order_combo = getattr(self, "_model_order_combo", None)
        if stretched:
            if self.model_order_var.get() != "1 — mono":
                self.model_order_var.set("1 — mono")
            if model_order_combo is not None:
                model_order_combo.configure(state="disabled")
        elif model_order_combo is not None:
            model_order_combo.configure(state="readonly")
        order_text = self.model_order_var.get()
        order = int(order_text.split()[0]) if order_text.split() else 1
        if any_stretched and current_method not in {"flimlib_global"}:
            self.method_model_hint_var.set(
                "Distribution model (stretched exponential) is used by \"Train new model…\" "
                "regardless of Lifetime method below, and by Lifetime method "
                "FLIMLib — global shared τ. It has no effect on reconvolution or "
                "apply-trained-model analysis."
            )
        elif current_method == "flimlib_apply_global":
            self.method_model_hint_var.set(
                f"Model order {order_text} comes from the loaded trained model, not this control."
                if self.trained_model is not None
                else "Load or train a global model first (Global model section below)."
            )
        elif order > 1 and not is_native_method:
            self.method_model_hint_var.set(
                f"⚠ Model order {order_text} only applies to the FLIMLib native "
                "methods. Set Lifetime method to FLIMLib — reconvolution, "
                "FLIMLib — global shared τ, or FLIMLib — apply trained global model."
            )
        elif order > 1:
            self.method_model_hint_var.set(f"Native fit will use {order_text} components.")
        else:
            self.method_model_hint_var.set("")

    def _lifetime_scale_changed(self, *_args) -> None:
        mode = "auto" if self.lifetime_scale_mode_var.get().startswith("Auto") else "manual"
        try:
            life_min = float(self.life_min_var.get())
            life_max = float(self.life_max_var.get())
            if not np.isfinite(life_min) or not np.isfinite(life_max) or life_max <= life_min:
                raise ValueError
        except ValueError:
            self.lifetime_scale_summary_var.set("Enter an increasing finite manual lifetime range.")
            return
        if self.result is None:
            self.lifetime_scale_summary_var.set(
                "Automatic range will be calculated after analysis."
                if mode == "auto"
                else f"Manual display range: {life_min:g}–{life_max:g} ns"
            )
            return
        # Scaling is display-only: it must not rerun or alter fitted data.
        self.result.settings.lifetime_scale_mode = mode
        self.result.settings.lifetime_min_ns = life_min
        self.result.settings.lifetime_max_ns = life_max
        self._lifetime_image = lifetime_image(self.result)
        self._composite_image = composite_image(self.result)
        components = self.result.fit_diagnostics.get("tau_components_ns")
        if components is not None and np.asarray(components).ndim == 3:
            self._component_images = [
                component_lifetime_image(self.result, index)
                for index in range(np.asarray(components).shape[0])
            ]
        self._update_lifetime_scale_summary()
        self._refresh_secondary_map()
        self._refresh_multiexponential_views()
        self._refresh_histograms()

    def _refresh_multiexponential_views(self) -> None:
        if not hasattr(self, "multi_tau_views"):
            return
        if self.result is None:
            self.multi_summary_var.set("Run a native bi- or tri-exponential fit to view components.")
            for view in (*self.multi_tau_views, *self.multi_amplitude_views):
                view.set_image(None)
            return
        components = self.result.fit_diagnostics.get("tau_components_ns")
        amplitudes = self.result.fit_diagnostics.get("amplitude_components")
        if components is None or amplitudes is None or np.asarray(components).ndim != 3:
            self.multi_summary_var.set(
                "No multi-exponential maps are available. Select native FLIMLib bi- or tri-exponential fitting and run analysis."
            )
            for view in (*self.multi_tau_views, *self.multi_amplitude_views):
                view.set_image(None, "Unavailable for this analysis")
            return
        component_values = np.asarray(components, dtype=np.float64)
        amplitude_values = np.asarray(amplitudes, dtype=np.float64)
        order = min(component_values.shape[0], amplitude_values.shape[0], 3)
        status = np.asarray(self.result.fit_diagnostics.get("fit_status", self.result.valid_mask)) == 1
        successful = int(np.count_nonzero(status))
        shared = self.result.fit_diagnostics.get("global_taus_ns")
        shared_text = ""
        if shared is not None and np.asarray(shared).size:
            shared_text = " Shared taus: " + ", ".join(
                f"tau{index + 1}={value:.3g} ns"
                for index, value in enumerate(np.asarray(shared, dtype=float).ravel()[:order])
            ) + "."
        model_name = {2: "Bi-exponential", 3: "Tri-exponential"}.get(order, "Mono-exponential")
        self.multi_summary_var.set(
            f"{model_name}: {successful:,} successful pixel fits. Components are ordered longest-to-shortest; "
            "A maps are local fitted onset amplitudes. The main lifetime map is the amplitude-weighted mean."
            + shared_text
        )
        for index in range(3):
            tau_view = self.multi_tau_views[index]
            amplitude_view = self.multi_amplitude_views[index]
            if index < order:
                tau_lo, tau_hi = lifetime_display_range(self.result, component_values[index])
                tau_view.title_label.configure(text=f"Lifetime component tau{index + 1}")
                tau_view.set_image(
                    self._component_images[index],
                    f"{tau_lo:.3g}–{tau_hi:.3g} ns; longest-to-shortest component order",
                )
                amplitude_view.title_label.configure(text=f"Amplitude A{index + 1}")
                amplitude_view.set_image(
                    self._amplitude_component_images[index],
                    "Local fitted onset amplitude; logarithmic display scale",
                )
            else:
                tau_view.title_label.configure(text=f"Lifetime component tau{index + 1}")
                amplitude_view.title_label.configure(text=f"Amplitude A{index + 1}")
                tau_view.set_image(None, "Not present in this fitted model")
                amplitude_view.set_image(None, "Not present in this fitted model")

    def _map_mode_changed(self, _event=None) -> None:
        self._refresh_secondary_map()

    def _live_threshold_changed(self, value: str) -> None:
        threshold = max(0.0, float(value))
        formatted = f"{threshold:.0f}"
        if self.live_threshold_var.get() != formatted:
            self.live_threshold_var.set(formatted)

    def _live_threshold_entry_changed(self, *_args) -> None:
        # Usable before analysis too, not just after -- see
        # _refresh_secondary_map_pre_analysis. Only bail when there's no
        # dataset loaded at all, i.e. nothing to preview a threshold against.
        if self.dataset is None:
            return
        try:
            threshold = max(0.0, float(self.live_threshold_var.get()))
        except ValueError:
            return
        if abs(float(self.live_threshold_scale.get()) - threshold) > 0.5:
            self.live_threshold_scale.set(threshold)
        if self.map_mode_var.get() == "Intensity with threshold":
            self._refresh_secondary_map()

    def _apply_live_threshold_to_fit(self) -> None:
        try:
            threshold = max(0.0, float(self.live_threshold_var.get()))
        except ValueError:
            self.status_var.set("Enter a finite threshold before applying it to the fit.")
            return
        method = self.METHOD_LABELS.get(self.method_var.get(), "")
        if method in {"phasor_phase", "phasor_modulation"}:
            self.phasor_threshold_var.set(f"{threshold:g}")
            target = "phasor"
        else:
            self.fit_threshold_var.set(f"{threshold:g}")
            target = "fit/native"
        self.status_var.set(
            f"{target.capitalize()} threshold set to {threshold:g}. Run analysis to recalculate maps."
        )

    def _live_fit_pixel_shape(self) -> tuple[int, int] | None:
        """(height, width) for map click/drag coordinate conversion.

        From the committed result if one exists; otherwise straight from
        the loaded raw dataset, so single point mode's map gestures still
        resolve to real pixels before "Run analysis" has ever been used.
        """

        if self.result is not None:
            return self.result.intensity.shape
        if self.dataset is not None:
            return (self.dataset.y_size, self.dataset.x_size)
        return None

    def _map_gesture(self, kind: str, payload, committed: bool = True) -> None:
        if self.result is None:
            # Every other map gesture (points, ordinary ROI inspection)
            # needs a committed analysis; single point mode deliberately
            # doesn't -- it's meant to run before "Run analysis" too.
            if self.single_point_mode_var.get() and kind in ("point", "roi") and committed:
                shape = self._live_fit_pixel_shape()
                if shape is None:
                    return
                if kind == "point":
                    x, y = normalised_to_pixel(payload, shape)
                    self._run_single_point_live_fit((x, y, x + 1, y + 1))
                else:
                    self._run_single_point_live_fit(normalised_to_bounds(payload, shape))
            return
        if kind == "cancel_preview":
            self._sync_map_overlays()
            return
        if kind == "point" and committed:
            x, y = normalised_to_pixel(payload, self.result.intensity.shape)
            if self.single_point_mode_var.get():
                self._run_single_point_live_fit((x, y, x + 1, y + 1))
                return
            try:
                updated, added = add_point(self.selected_points, x, y)
            except ValueError as exc:
                self.status_var.set(str(exc))
                return
            self.selected_points = updated
            if not added:
                self.status_var.set(f"Point x={x}, y={y} is already selected.")
            self._refresh_point_display()
            return
        if kind == "clear_roi":
            self.clear_roi()
            return
        if kind == "roi":
            self._roi_selected(payload, committed)

    def _roi_selected(
        self,
        selection: tuple[float, float, float, float] | None,
        committed: bool = True,
    ) -> None:
        if self.result is None:
            return
        if selection is None:
            self.clear_roi()
            return
        bounds = normalised_to_bounds(selection, self.result.intensity.shape)
        normalised = bounds_to_normalised(bounds, self.result.intensity.shape)
        for view in self._map_views():
            view.set_selection(normalised, committed=committed)
        if committed:
            self.roi_bounds = bounds
            if self.single_point_mode_var.get():
                self._run_single_point_live_fit(bounds)
            else:
                self._update_roi_details(bounds)
        else:
            x0, y0, x1, y1 = bounds
            self.selection_var.set(
                f"ROI preview: x=[{x0},{x1}), y=[{y0},{y1}) — release to calculate"
            )

    def clear_roi(self) -> None:
        self.roi_bounds = None
        for view in self._map_views():
            view.set_selection(None)
        if self.result is not None:
            self._update_roi_details(None)
            self.roi_analysis = None

    def undo_point(self) -> None:
        if self.selected_points:
            self.selected_points.pop()
            self._refresh_point_display()

    def clear_points(self) -> None:
        self.selected_points = []
        self.point_analyses = []
        self._refresh_point_display()

    def _sync_map_overlays(self) -> None:
        if self.result is None:
            return
        roi = bounds_to_normalised(self.roi_bounds, self.result.intensity.shape) if self.roi_bounds else None
        analysed_by_id = {point.selection.identifier: point for point in self.point_analyses}
        markers = [
            (
                *pixel_to_normalised_center((point.x, point.y), self.result.intensity.shape),
                (
                    f"{point.identifier} {analysed_by_id[point.identifier].lifetime_label}="
                    f"{analysed_by_id[point.identifier].map_tau_ns:.2f} ns"
                    if point.identifier in analysed_by_id and analysed_by_id[point.identifier].valid
                    else f"{point.identifier} excluded"
                ),
                point.colour,
            )
            for point in self.selected_points
        ]
        row_bands: list[tuple[float, float]] = []
        if self.row_map_overlay_var.get():
            coverage = self.result.fit_diagnostics.get("interpolated_result_fraction")
            if coverage is not None:
                coverage_array = np.asarray(coverage)
                if coverage_array.shape == self.result.intensity.shape:
                    height = coverage_array.shape[0]
                    active_rows = np.flatnonzero(np.any(coverage_array > 0, axis=1))
                    row_bands = [(float(y) / height, float(y + 1) / height) for y in active_rows]
        column_bands: list[tuple[float, float]] = []
        if self.column_map_overlay_var.get():
            coverage = self.result.fit_diagnostics.get("interpolated_column_result_fraction")
            if coverage is not None:
                coverage_array = np.asarray(coverage)
                if coverage_array.shape == self.result.intensity.shape:
                    width = coverage_array.shape[1]
                    active_columns = np.flatnonzero(np.any(coverage_array > 0, axis=0))
                    column_bands = [
                        (float(x) / width, float(x + 1) / width) for x in active_columns
                    ]
        for view in self._map_views():
            view.set_selection(roi)
            view.set_points(markers)
            view.set_row_overlay(row_bands)
            view.set_column_overlay(column_bands)

    def _refresh_point_display(self) -> None:
        if self.result is None:
            return
        self.point_analyses = analyse_points(self.result, self.selected_points)
        self._sync_map_overlays()
        if self.point_analyses:
            series = [
                (point.selection.identifier, point.decay, point.selection.colour)
                for point in self.point_analyses
            ]
            plot = line_plot(
                self.result.time_ns,
                series,
                size=(920, 420),
                title=f"Selected-point processed decays (N={len(series)})",
                y_label="Processed signal / bin",
            )
            self.roi_decay_view.set_image(
                plot,
                f"{len(series)} selected points; colours match map markers; "
                f"{self.point_analyses[0].lifetime_label} values are printed on the maps",
            )
            self.point_summary_var.set(
                f"{len(series)}/{MAX_SELECTED_POINTS} points • colours match map markers and decay curves"
            )
        else:
            self.point_summary_var.set("Click either map to add up to 8 comparison points.")
            self._update_roi_decay_plot(self.roi_bounds)
        self._refresh_histograms()
        self._refresh_phasors()

    def _phasor_provenance_text(self) -> str:
        if self.result is None:
            return "First-harmonic phasor cloud"
        if bool(self.result.metadata.get("irf_loaded")):
            return "IRF-corrected first-harmonic cloud"
        return "Uncorrected first-harmonic cloud (no IRF)"

    def _refresh_phasors(self) -> None:
        if self.result is None:
            return
        override = self._phasor_accumulated_gs()
        if override is not None:
            # A pooled cloud spans several (differently shaped) images at
            # once -- "selected ROI" bounds belong to one image's own pixel
            # grid and can't mean anything here, so this always shows the
            # combined whole-queue cloud regardless of self.roi_bounds.
            names = ", ".join(name for name, _, _ in self.phasor_accumulated_points[:4])
            more = "" if len(self.phasor_accumulated_points) <= 4 else f" +{len(self.phasor_accumulated_points) - 4} more"
            subtitle = (
                f"{self._phasor_provenance_text()} — accumulated across "
                f"{len(self.phasor_accumulated_points)} queue file(s) ({names}{more})"
            )
            bounds_for_plot = None
        else:
            scope = "whole image" if self.roi_bounds is None else "selected ROI"
            available = sum(point.phasor_valid for point in self.point_analyses)
            subtitle = (
                f"{self._phasor_provenance_text()} — {scope}; "
                f"{available}/{len(self.point_analyses)} selected phasors available"
            )
            bounds_for_plot = self.roi_bounds
        # Switched-off ROIs stay in self.phasor_rois (so they can be
        # switched back on, and still save/load) but are otherwise invisible.
        visible_rois = enabled_only(self.phasor_rois)
        # Two separate renders (both square -- G/S must stay physically
        # proportional either way) rather than one small bitmap stretched
        # to fill both the compact Overview panel and the full Phasor tab.
        overview_image = phasor_plot(
            self.result,
            bounds=bounds_for_plot,
            points=self.point_analyses,
            heatmap=bool(self.phasor_heatmap_var.get()),
            heatmap_bins=int(self.result.settings.phasor_heatmap_bins),
            rois=visible_rois,
            g_s_override=override,
        )
        self.overview_phasor_view.set_image(overview_image, subtitle)
        detailed_image = phasor_plot(
            self.result,
            size=PHASOR_DETAIL_SIZE,
            bounds=bounds_for_plot,
            points=self.point_analyses,
            heatmap=bool(self.phasor_heatmap_var.get()),
            heatmap_bins=int(self.result.settings.phasor_heatmap_bins),
            rois=visible_rois,
            g_s_override=override,
        )
        self.phasor_view.set_image(detailed_image, subtitle)
        self._refresh_phasor_roi_lifetime_view()

    def _phasor_accumulate_toggled(self) -> None:
        if self.phasor_accumulate_queue_var.get():
            self.status_var.set(
                "Phasor accumulation on -- run “Process queue…” or “Process & export "
                "queue” to pool every file's phasor cloud together."
            )
        else:
            self.status_var.set("Phasor accumulation off -- the phasor plot shows only the current file again.")
        self._update_phasor_accumulate_status()
        if self.result is not None:
            self._refresh_phasors()

    def _clear_phasor_accumulation(self) -> None:
        if not self.phasor_accumulated_points:
            return
        self.phasor_accumulated_points = []
        self._update_phasor_accumulate_status()
        if self.result is not None:
            self._refresh_phasors()
        self.status_var.set("Cleared the accumulated phasor cloud.")

    def _append_phasor_accumulation(self, name: str, g: np.ndarray, s: np.ndarray) -> None:
        self.phasor_accumulated_points.append((name, g, s))
        self._update_phasor_accumulate_status()

    def _update_phasor_accumulate_status(self) -> None:
        if not self.phasor_accumulate_queue_var.get():
            self.phasor_accumulate_status_var.set("Phasor accumulation off.")
            return
        if not self.phasor_accumulated_points:
            self.phasor_accumulate_status_var.set("Phasor accumulation on -- nothing pooled yet.")
            return
        total_points = sum(
            int(np.count_nonzero(np.isfinite(g) & np.isfinite(s))) for _, g, s in self.phasor_accumulated_points
        )
        self.phasor_accumulate_status_var.set(
            f"Pooled {len(self.phasor_accumulated_points)} file(s), {total_points:,} valid phasor points."
        )

    def _phasor_accumulated_gs(self) -> tuple[np.ndarray, np.ndarray] | None:
        if not self.phasor_accumulate_queue_var.get() or not self.phasor_accumulated_points:
            return None
        g_all = np.concatenate([np.asarray(g).ravel() for _, g, _ in self.phasor_accumulated_points])
        s_all = np.concatenate([np.asarray(s).ravel() for _, _, s in self.phasor_accumulated_points])
        return g_all, s_all

    # -- Phasor ROI drawing --------------------------------------------------
    # Draw a region of interest directly on the phasor plot (rectangle,
    # ellipse, or freeform polygon) and see, live, which pixels on the
    # lifetime map have a phasor coordinate inside it. Deliberately
    # bypasses ImageView's own single-rectangle/point click handling (see
    # _bind_phasor_roi_drawing) with a dedicated state machine here, since
    # multi-shape, multi-ROI drawing is a fundamentally different
    # interaction from anywhere else this app uses ImageView.

    def _phasor_norm_to_gs(self, point_norm: tuple[float, float]) -> tuple[float, float]:
        nx, ny = point_norm
        px, py = nx * PHASOR_DETAIL_SIZE[0], ny * PHASOR_DETAIL_SIZE[1]
        return phasor_pixel_to_gs(px, py, PHASOR_DETAIL_SIZE)

    def _phasor_gs_to_norm(self, g: float, s: float) -> tuple[float, float]:
        px, py = phasor_gs_to_pixel(g, s, PHASOR_DETAIL_SIZE)
        return px / PHASOR_DETAIL_SIZE[0], py / PHASOR_DETAIL_SIZE[1]

    def _bind_phasor_roi_drawing(self, view: "ImageView") -> None:
        canvas = view.canvas
        canvas.configure(takefocus=1)
        canvas.bind("<ButtonPress-1>", lambda event: self._phasor_roi_press(view, event))
        canvas.bind("<B1-Motion>", lambda event: self._phasor_roi_drag(view, event))
        canvas.bind("<ButtonRelease-1>", lambda event: self._phasor_roi_release(view, event))
        canvas.bind("<Escape>", lambda _event: self._phasor_roi_cancel(view))
        # Right-click rotates whichever existing ROI is under the cursor;
        # right-click/drag *not* over one falls straight through to
        # ImageView's own pan (drag) / clear-ROI (plain click) behaviour,
        # completely unchanged -- see the "no hit" branches below.
        canvas.bind("<ButtonPress-3>", lambda event: self._phasor_roi_rotate_press(view, event))
        canvas.bind("<B3-Motion>", lambda event: self._phasor_roi_rotate_drag(view, event))
        canvas.bind("<ButtonRelease-3>", lambda event: self._phasor_roi_rotate_release(view, event))

    def _phasor_roi_press(self, view: "ImageView", event: "tk.Event") -> None:
        if self.result is None:
            return
        view.canvas.focus_set()
        point = view._normalised_position(event, clamp=True)
        if point is None:
            return
        # A fresh path for every gesture: exactly (start, live end) for
        # rectangle/ellipse, an accumulating hand-drawn outline for
        # freeform -- see _phasor_roi_drag.
        self._phasor_draw_path = [self._phasor_norm_to_gs(point)]
        self._render_phasor_roi_preview(view)

    def _phasor_roi_drag(self, view: "ImageView", event: "tk.Event") -> None:
        if not self._phasor_draw_path:
            return
        point = view._normalised_position(event, clamp=True)
        if point is None:
            return
        gs = self._phasor_norm_to_gs(point)
        if self.phasor_roi_shape_var.get() == "Freeform":
            # Freeform is drawn by continuous drag, like a pen -- keep
            # every point along the path (with a minimum spacing so a
            # fast drag doesn't pile up thousands of near-duplicate
            # vertices; ~1% of the plot's G/S span is still plenty of
            # outline resolution for a hand-drawn shape).
            last = self._phasor_draw_path[-1]
            if (gs[0] - last[0]) ** 2 + (gs[1] - last[1]) ** 2 >= 0.01 ** 2:
                self._phasor_draw_path.append(gs)
        elif len(self._phasor_draw_path) == 1:
            self._phasor_draw_path.append(gs)
        else:
            self._phasor_draw_path[-1] = gs
        self._render_phasor_roi_preview(view)

    def _phasor_roi_release(self, view: "ImageView", event: "tk.Event") -> None:
        if not self._phasor_draw_path:
            return
        path = self._phasor_draw_path
        self._phasor_draw_path = []
        self._render_phasor_roi_preview(view)  # clears the drag preview
        tool = self.phasor_roi_shape_var.get()
        if tool == "Freeform":
            if len(path) < 3:
                self.status_var.set("Freeform phasor ROI needs a longer drag -- draw a bigger loop.")
                return
            self._commit_phasor_roi("freeform", path)
            return
        if len(path) < 2:
            return
        start_gs, end_gs = path[0], path[-1]
        if abs(end_gs[0] - start_gs[0]) < 1e-4 and abs(end_gs[1] - start_gs[1]) < 1e-4:
            # Too small to be a deliberate drag -- a plain click, discard
            # rather than committing a zero-size ROI.
            return
        shape = "rectangle" if tool == "Rectangle" else "ellipse"
        self._commit_phasor_roi(shape, [start_gs, end_gs])

    def _phasor_roi_cancel(self, view: "ImageView") -> None:
        if self._phasor_draw_path:
            self._phasor_draw_path = []
            self._render_phasor_roi_preview(view)
            self.status_var.set("Phasor ROI drawing cancelled.")
        elif self._phasor_rotate_index is not None:
            self._phasor_rotate_index = None
            self._phasor_rotate_preview_deg = 0.0
            view.set_shapes([])
            self.status_var.set("Phasor ROI rotation cancelled.")

    def _render_phasor_roi_preview(self, view: "ImageView") -> None:
        """Canvas-only live preview of the shape still being drawn.

        Already-committed ROIs are baked directly into the rendered phasor
        plot image itself (see _refresh_phasors/_commit_phasor_roi), so
        this only ever needs to draw the one in-progress path, not the
        whole list -- cheap enough to redo on every mouse movement.
        """

        preview: list[tuple[str, list[tuple[float, float]], str, str]] = []
        if len(self._phasor_draw_path) >= 2:
            tool = self.phasor_roi_shape_var.get()
            kind = "freeform" if tool == "Freeform" else ("rectangle" if tool == "Rectangle" else "ellipse")
            preview.append((
                kind,
                [self._phasor_gs_to_norm(g, s) for g, s in self._phasor_draw_path],
                "#facc15", "",
            ))
        view.set_shapes(preview)

    def _phasor_roi_hit_test(self, gs: tuple[float, float]) -> int | None:
        """Index of the topmost (most recently drawn), switched-on ROI containing ``gs``, if any.

        A switched-off ROI is invisible, so right-click-drag correctly
        passes through it to whatever's underneath (or falls through to
        pan) rather than grabbing something the user can't currently see.
        """

        for index in range(len(self.phasor_rois) - 1, -1, -1):
            roi = self.phasor_rois[index]
            if roi.enabled and roi.contains(*gs):
                return index
        return None

    def _phasor_roi_rotate_press(self, view: "ImageView", event: "tk.Event") -> None:
        point = view._normalised_position(event, clamp=True)
        if point is None or self.result is None:
            view._pan_or_clear_started(event)
            return
        gs = self._phasor_norm_to_gs(point)
        hit_index = self._phasor_roi_hit_test(gs)
        if hit_index is None:
            # Not over any ROI -- fall through to ImageView's own
            # right-drag pan / plain-right-click clear-ROI, unchanged.
            view._pan_or_clear_started(event)
            return
        self._phasor_rotate_index = hit_index
        roi = self.phasor_rois[hit_index]
        cg, cs = roi.centroid_gs()
        self._phasor_rotate_last_angle_deg = math.degrees(math.atan2(gs[1] - cs, gs[0] - cg))
        self._phasor_rotate_preview_deg = 0.0
        self.status_var.set(f"Rotating phasor ROI {roi.label} -- drag to rotate, release to finish.")

    def _phasor_roi_rotate_drag(self, view: "ImageView", event: "tk.Event") -> None:
        if self._phasor_rotate_index is None:
            view._pan_or_clear_dragged(event)
            return
        point = view._normalised_position(event, clamp=True)
        if point is None:
            return
        roi = self.phasor_rois[self._phasor_rotate_index]
        cg, cs = roi.centroid_gs()
        gs = self._phasor_norm_to_gs(point)
        angle_deg = math.degrees(math.atan2(gs[1] - cs, gs[0] - cg))
        self._phasor_rotate_preview_deg += angle_deg - self._phasor_rotate_last_angle_deg
        self._phasor_rotate_last_angle_deg = angle_deg
        # Cheap canvas-only live preview (a rotated copy of the ROI) --
        # the real ROI/plot image isn't touched until release, exactly
        # like drawing a brand new shape only bakes in on commit.
        preview_roi = deepcopy(roi)
        rotate_roi(preview_roi, self._phasor_rotate_preview_deg)
        boundary = [self._phasor_gs_to_norm(g, s) for g, s in boundary_points_gs(preview_roi)]
        view.set_shapes([("freeform", boundary, "#facc15", roi.label)])

    def _phasor_roi_rotate_release(self, view: "ImageView", event: "tk.Event") -> None:
        if self._phasor_rotate_index is None:
            view._pan_or_clear_finished(event)
            return
        roi = self.phasor_rois[self._phasor_rotate_index]
        applied_deg = self._phasor_rotate_preview_deg
        rotate_roi(roi, applied_deg)
        self._phasor_rotate_index = None
        self._phasor_rotate_preview_deg = 0.0
        view.set_shapes([])
        if self.result is not None:
            self._refresh_phasors()
        # Freeform bakes rotation straight into its vertices and always
        # keeps rotation_deg at 0 (see phasor_roi.rotate_roi) -- report the
        # angle just applied there instead of that always-zero field, or
        # this would misleadingly read "rotated ... to 0.0°" every time.
        if roi.shape == "freeform":
            self.status_var.set(f"Rotated phasor ROI {roi.label} by {applied_deg:.1f}°.")
        else:
            self.status_var.set(f"Rotated phasor ROI {roi.label} to {roi.rotation_deg:.1f}°.")

    def _commit_phasor_roi(self, shape: str, points_gs: list[tuple[float, float]]) -> None:
        colour = next_phasor_roi_colour(len(self.phasor_rois))
        label = f"ROI{len(self.phasor_rois) + 1}"
        self.phasor_rois.append(PhasorROI(label=label, shape=shape, points_gs=points_gs, colour=colour))
        self._update_phasor_roi_summary()
        if self.result is not None:
            self._refresh_phasors()
        self.phasor_view.set_shapes([])
        self.status_var.set(f"Added phasor ROI {label} ({shape}).")

    def _suggest_phasor_roi_clusters(self) -> None:
        """Alternative to hand-drawing: fit ellipse ROIs to clusters found
        automatically in the current phasor cloud (see
        ``phasor_roi.suggest_rois_from_clusters``). Uses exactly the same
        (G, S) source the phasor plot itself is currently showing -- the
        pooled queue-wide cloud when "Accumulate phasor across queue" is
        on, otherwise the current result restricted to the selected ROI
        if one is active -- so "what you see is what gets clustered"."""

        if self.result is None:
            messagebox.showwarning("No result", "Run an analysis before suggesting phasor ROIs.")
            return
        try:
            n_clusters = int(self.phasor_cluster_count_var.get())
        except ValueError:
            messagebox.showerror("Invalid cluster count", "Enter a whole number of clusters.")
            return
        if n_clusters < 1:
            messagebox.showerror("Invalid cluster count", "Cluster count must be at least 1.")
            return
        try:
            sigma = float(self.phasor_cluster_sigma_var.get())
        except ValueError:
            messagebox.showerror("Invalid ROI size", "Enter a numeric ROI size (σ), e.g. 2.0.")
            return
        if sigma <= 0.0:
            messagebox.showerror("Invalid ROI size", "ROI size (σ) must be greater than 0.")
            return

        override = self._phasor_accumulated_gs()
        if override is not None:
            g, s = override
            # The pooled cloud only ever stores (name, g, s) per file (see
            # _append_phasor_accumulation) -- no per-pixel intensity survives
            # the pooling, so clustering here is necessarily unweighted.
            weights = None
        elif self.roi_bounds is None:
            g, s = self.result.g, self.result.s
            weights = self.result.intensity
        else:
            x0, y0, x1, y1 = self.roi_bounds
            g = self.result.g[y0:y1, x0:x1]
            s = self.result.s[y0:y1, x0:x1]
            weights = self.result.intensity[y0:y1, x0:x1]

        try:
            suggested = suggest_rois_from_clusters(g, s, n_clusters, weights=weights, sigma=sigma)
        except ValueError as exc:
            messagebox.showerror("Could not suggest ROIs", str(exc))
            return

        start = len(self.phasor_rois)
        for offset, roi in enumerate(suggested):
            roi.label = f"Cluster{start + offset + 1}"
            roi.colour = next_phasor_roi_colour(start + offset)
            self.phasor_rois.append(roi)

        self._update_phasor_roi_summary()
        self._refresh_phasors()
        found = len(suggested)
        shortfall = "" if found == n_clusters else f" ({n_clusters - found} cluster(s) had no members and were skipped)"
        self.status_var.set(f"Suggested {found} phasor ROI(s) from clustering{shortfall}.")

    def _phasor_roi_number_from_field(self, action: str) -> int | None:
        """Validate the shared "ROI #" field, used by both Toggle and Remove.

        Returns a valid 1-based index into ``self.phasor_rois``, or ``None``
        with an explanatory status message already set.
        """

        if not self.phasor_rois:
            self.status_var.set(f"No phasor ROIs to {action}.")
            return None
        try:
            number = int(self.phasor_roi_remove_number_var.get())
        except ValueError:
            self.status_var.set(f"Enter a whole ROI number to {action} (e.g. 1 for ROI1).")
            return None
        if not (1 <= number <= len(self.phasor_rois)):
            self.status_var.set(f"No ROI numbered {number} — there are {len(self.phasor_rois)} phasor ROI(s).")
            return None
        return number

    def _toggle_phasor_roi_by_number(self) -> None:
        number = self._phasor_roi_number_from_field("toggle")
        if number is None:
            return
        roi = self.phasor_rois[number - 1]
        roi.enabled = not roi.enabled
        self._update_phasor_roi_summary()
        if self.result is not None:
            self._refresh_phasors()
        self.status_var.set(f"Phasor ROI {roi.label} switched {'on' if roi.enabled else 'off'}.")

    def _remove_phasor_roi_by_number(self) -> None:
        number = self._phasor_roi_number_from_field("remove")
        if number is None:
            return
        removed = self.phasor_rois.pop(number - 1)
        # Keep every remaining ROI's own colour (so the user doesn't have
        # to re-learn which colour is which after a removal), but renumber
        # labels back to a gap-free 1..N sequence.
        for index, roi in enumerate(self.phasor_rois, start=1):
            roi.label = f"ROI{index}"
        self._update_phasor_roi_summary()
        if self.result is not None:
            self._refresh_phasors()
        self.status_var.set(f"Removed phasor ROI {removed.label}.")

    def _clear_phasor_rois(self) -> None:
        if not self.phasor_rois:
            return
        self.phasor_rois = []
        self._phasor_draw_path = []
        self._phasor_rotate_index = None
        self.phasor_view.set_shapes([])
        self._update_phasor_roi_summary()
        if self.result is not None:
            self._refresh_phasors()
        self.status_var.set("Cleared all phasor ROIs.")

    def _save_phasor_rois_dialog(self) -> None:
        if not self.phasor_rois:
            messagebox.showinfo("No phasor ROIs", "Draw at least one phasor ROI before saving.")
            return
        path = filedialog.asksaveasfilename(
            title="Save phasor ROIs", defaultextension=".json", filetypes=[("Phasor ROI JSON", "*.json")]
        )
        if not path:
            return
        try:
            save_phasor_rois(self.phasor_rois, path)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        self.status_var.set(f"Saved {len(self.phasor_rois)} phasor ROI(s) to {path}")

    def _load_phasor_rois_dialog(self) -> None:
        path = filedialog.askopenfilename(title="Load phasor ROIs", filetypes=[("Phasor ROI JSON", "*.json")])
        if not path:
            return
        try:
            loaded = load_phasor_rois(path)
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))
            return
        self.phasor_rois = loaded
        self.phasor_view.set_shapes([])
        self._update_phasor_roi_summary()
        if self.result is not None:
            self._refresh_phasors()
        self.status_var.set(f"Loaded {len(loaded)} phasor ROI(s) from {path}")

    def _update_phasor_roi_summary(self) -> None:
        count = len(self.phasor_rois)
        self.phasor_roi_remove_spin.configure(to=max(count, 1))
        try:
            current = int(self.phasor_roi_remove_number_var.get())
        except ValueError:
            current = 1
        # Keep the spinner pointing at a still-valid number -- clamp down
        # to the new last ROI after a removal, rather than leaving it
        # stuck referencing one that no longer exists.
        self.phasor_roi_remove_number_var.set(str(min(max(current, 1), max(count, 1))))
        if not count:
            self.phasor_roi_summary_var.set("No phasor ROIs drawn yet.")
            return
        labels = ", ".join(roi.label if roi.enabled else f"{roi.label} (off)" for roi in self.phasor_rois)
        self.phasor_roi_summary_var.set(f"{count} phasor ROI(s): {labels}")

    def _refresh_phasor_roi_lifetime_view(self) -> None:
        if self.result is None:
            return
        mode = self.PHASOR_ROI_DISPLAY_MODES.get(self.phasor_roi_display_mode_var.get(), "original")
        visible_rois = enabled_only(self.phasor_rois)
        show_invalid = bool(self.phasor_roi_show_invalid_var.get())
        additive_overlap = bool(self.phasor_roi_additive_overlap_var.get())
        image = phasor_roi_lifetime_image(
            self.result, visible_rois, mode=mode, show_invalid=show_invalid, additive_overlap=additive_overlap
        )
        # Split "below the phasor threshold" from "valid, just unmatched" --
        # both used to render identically, which hid e.g. a dim species
        # (fewer photons/pixel, more likely excluded by the threshold)
        # looking like it was simply absent from regions where it was
        # really just never analysed at all. See phasor_roi_lifetime_image's
        # own PHASOR_ROI_INVALID vs PHASOR_ROI_BACKGROUND colours, and its
        # show_invalid flag -- the "Flag below-threshold pixels" checkbox.
        owner = combined_owner_mask(visible_rois, self.result.g, self.result.s)
        invalid = int(np.count_nonzero(~(np.isfinite(self.result.g) & np.isfinite(self.result.s))))
        matched = int(np.count_nonzero(owner > 0))
        unclassified = int(owner.size - matched - invalid)
        invalid_note = (
            f"{invalid:,} below phasor threshold (dark red)"
            if show_invalid
            else f"{invalid:,} below phasor threshold (not flagged -- shown as background)"
        )
        counts_note = f"{matched:,} matched (colour), {unclassified:,} unclassified (background), {invalid_note}"
        if additive_overlap and mode in ("coloured", "coloured_composite") and len(visible_rois) > 1:
            per_roi_masks = [contains_array(roi, self.result.g, self.result.s) for roi in visible_rois]
            overlap_count = int(np.count_nonzero(np.sum(per_roi_masks, axis=0) > 1))
            counts_note += f"; {overlap_count:,} pixels in overlapping ROIs shown as added colours"
        if visible_rois:
            off_note = "" if len(visible_rois) == len(self.phasor_rois) else (
                f"; {len(self.phasor_rois) - len(visible_rois)} switched off"
            )
            subtitle = (
                f"{len(visible_rois)} phasor ROI(s) — {self.phasor_roi_display_mode_var.get().lower()}"
                f"{off_note} — {counts_note}"
            )
        elif self.phasor_rois:
            subtitle = f"All phasor ROIs are switched off — {counts_note}"
        else:
            subtitle = f"Draw a phasor ROI to highlight it here — {counts_note}"
        self.phasor_roi_lifetime_view.set_image(image, subtitle)

    def _refresh_histograms(self) -> None:
        if self.result is None:
            return
        overview_image = histogram_plot(
            self.result,
            bounds=self.roi_bounds,
            points=self.point_analyses,
            bins=self.result.settings.histogram_bins,
        )
        detailed_image = histogram_plot(
            self.result,
            size=(1200, 680),
            bounds=self.roi_bounds,
            points=self.point_analyses,
            bins=self.result.settings.detailed_histogram_bins,
        )
        scope = "whole image" if self.roi_bounds is None else "selected ROI"
        subtitle = f"Valid-pixel distribution — {scope}; coloured rules are selected points"
        self.overview_histogram_view.set_image(overview_image, subtitle)
        self.histogram_view.set_image(detailed_image, subtitle)

    def _roi_display_mode_changed(self, _event=None) -> None:
        if self.result is not None:
            self._update_roi_details(self.roi_bounds)

    @staticmethod
    def _tau_text(value: float) -> str:
        return f"{value:.3f} ns" if np.isfinite(value) else "—"

    def _update_roi_details(self, bounds: BoundsXYXY | None) -> None:
        if self.result is None:
            return
        result = self.result
        roi = analyse_roi(result, bounds)
        self.roi_analysis = roi if bounds is not None else None
        x0, y0, x1, y1 = roi.bounds_xyxy
        is_full_image = bounds is None
        region_name = "Whole image" if is_full_image else (f"Pixel ({x0}, {y0})" if roi.selected_pixels == 1 else "ROI")
        bin_factor = int(result.metadata.get("spatial_transform", {}).get("spatial_bin_width", 1))
        source_note = ""
        if bin_factor > 1:
            source_note = (
                f"; source x=[{x0 * bin_factor},{x1 * bin_factor}), "
                f"y=[{y0 * bin_factor},{y1 * bin_factor})"
            )
        self.selection_var.set(
            f"{region_name}: x=[{x0},{x1}), y=[{y0},{y1}) — "
            f"{roi.width}×{roi.height} processed px{source_note}"
        )
        self.roi_summary_var.set(
            "Histogram scope: whole image"
            if bounds is None
            else f"ROI {roi.width}×{roi.height}: {roi.valid_pixels}/{roi.selected_pixels} valid"
        )
        self.roi_pixels_var.set(
            f"{roi.valid_pixels:,} / {roi.selected_pixels:,} ({roi.valid_fraction:.0%})"
        )
        self.roi_mean_tau_var.set(
            f"{roi.pixel_tau_mean_ns:.3f} ± {roi.pixel_tau_std_ns:.3f} ns"
            if np.isfinite(roi.pixel_tau_mean_ns)
            else "—"
        )
        self.roi_median_tau_var.set(
            f"{roi.pixel_tau_median_ns:.3f} [{roi.pixel_tau_q1_ns:.3f}, {roi.pixel_tau_q3_ns:.3f}] ns"
            if np.isfinite(roi.pixel_tau_median_ns)
            else "—"
        )
        self.roi_aggregate_tau_var.set(self._tau_text(roi.aggregate_tau_ns))
        self.roi_photons_var.set(f"{roi.total_photons:,.0f}")

        self._update_roi_decay_plot(
            bounds,
            roi=roi,
            update_overview=not self.point_analyses,
        )
        self._refresh_phasors()
        self._refresh_histograms()
        if self.map_mode_var.get() == "Chi-square":
            self._refresh_secondary_map()

    def _update_roi_decay_plot(
        self,
        bounds: BoundsXYXY | None,
        *,
        roi: ROIAnalysis | None = None,
        update_overview: bool = True,
    ) -> None:
        if self.result is None:
            return
        result = self.result
        roi = roi if roi is not None else analyse_roi(result, bounds)
        x0, y0, x1, y1 = roi.bounds_xyxy
        region_name = "Whole image" if bounds is None else "ROI"

        show_mean = self.roi_decay_mode_var.get() == "Mean per processed pixel"
        data = roi.mean_decay if show_mean else roi.summed_decay
        divisor = max(roi.selected_pixels, 1) if show_mean else 1
        data_label = "Mean signal / processed pixel" if show_mean else "ROI summed processed signal"
        y_label = "Mean signal / processed pixel / bin" if show_mean else "Processed signal / bin"
        title = f"{region_name} decay — x={x0}:{x1}, y={y0}:{y1}"
        series = [(data_label, data, ACCENT)]
        if roi.native_fitted_decay is not None:
            native_fit = np.asarray(roi.native_fitted_decay, dtype=np.float64) / divisor
            series.append((roi.native_fit_label or "Native fit", native_fit, AMBER))
            fit_note = f"{roi.native_fit_label}; IRF-reconvolved"
        else:
            mono_fit = roi.fitted_decay / divisor
            series.append(("Mono tail guide", mono_fit, GREEN))
            fit_note = (
                f"diagnostic mono-tail guide τ={self._tau_text(roi.aggregate_tau_ns)}, "
                f"bins [{roi.guide_fit_start_bin},{roi.guide_fit_end_bin}); no IRF reconvolution"
            )
        subtitle = f"{roi.selected_pixels:,} selected; {roi.valid_pixels:,} valid map lifetimes; {fit_note}"
        # Two separate renders, not one bitmap stretched to fit both --
        # the compact Overview panel and the full Decay & fit tab are very
        # different sizes, and a single small render just looks blurred/
        # blocky once ImageView enlarges it to fill the bigger pane.
        if update_overview:
            compact_plot = line_plot(result.time_ns, series, size=(920, 360), y_label=y_label, title=title)
            self.roi_decay_view.set_image(compact_plot, subtitle)
        detailed_plot = line_plot(result.time_ns, series, size=(1200, 680), y_label=y_label, title=title)
        self.decay_view.set_image(detailed_plot, subtitle)

    def _single_point_mode_toggled(self) -> None:
        """Flip between the precomputed-map inspector and the live single-decay preview.

        Off (default): clicking/dragging a map behaves as it always has --
        comparison points are added, or an ROI is measured against the maps
        from the last committed ``analyze_dataset`` run. On: the same
        gestures instead trigger a fresh live ANALYSIS of just that
        selection, using the sidebar's current (not yet applied) fit
        parameters -- this is a tuning aid: try a parameter, click a spot,
        see the result, adjust, repeat, then run the full analysis once
        happy. Deliberately limited to Phasor (phase/modulation) and
        FLIMLib -- reconvolution, and works even before "Run analysis" has
        ever been used (see ``_live_fit_selection``) -- the whole point is
        to help pick parameters *before* committing to a full run.
        """

        # Maps are otherwise non-interactive until a full analysis has run
        # (see _sync_map_interactivity) -- single point mode is the one
        # case that needs clicks to reach a freshly loaded, un-analysed
        # dataset too, so toggling it must update this immediately rather
        # than waiting for the next busy/dataset-load event.
        self._sync_map_interactivity()
        if self.single_point_mode_var.get():
            if self.dataset is None:
                self.status_var.set("Single point mode: load a dataset first, then click a map.")
            else:
                self.status_var.set(
                    "Single point mode: click a map to live-analyse that pixel, or drag to "
                    "live-analyse a region -- works even before Run analysis. Limited to "
                    "Phasor and FLIMLib -- reconvolution, using the sidebar's current settings."
                )
        else:
            self.status_var.set("Single point mode off -- selections show the committed analysis again.")
            if self.result is not None:
                self._update_roi_details(self.roi_bounds)

    def _refresh_single_point_live_fit(self) -> None:
        """Re-run the live preview on whatever was last selected, at the sidebar's current settings.

        Lets the user change a parameter (Model order, Noise model, Lifetime
        method, ...) and see its effect without re-clicking/re-dragging the
        same spot on the map -- the "beside the enable checkbox" button
        alongside clicking a map, which still updates the preview live too.
        """

        if not self.single_point_mode_var.get():
            self.status_var.set("Single point mode: enable the checkbox first, then click/drag a map.")
            return
        if self.single_point_last_bounds is None:
            self.status_var.set("Single point mode: click or drag a map first, then Re-fit updates it.")
            return
        self._run_single_point_live_fit(self.single_point_last_bounds)

    def _live_fit_selection(
        self, bounds: BoundsXYXY, settings: AnalysisSettings
    ) -> tuple[LiveFitSelection, np.ndarray] | None:
        """Pool ``bounds`` into one decay plus a matching time axis.

        From the committed result's own already-processed decay if one
        exists (background/spatial-corrected, exact); otherwise straight
        from the raw loaded dataset, background-subtracted the same way the
        pre-analysis intensity preview already does (see
        ``_show_raw_intensity``) -- so single point mode works even before
        "Run analysis" has ever been used. Returns ``None`` (a status
        message is already set) if there's nothing to analyse yet.
        """

        if self.result is not None:
            result = self.result
            try:
                roi = analyse_roi(result, bounds)
            except ValueError as exc:
                self.status_var.set(str(exc))
                return None
            selection = LiveFitSelection(
                roi.bounds_xyxy, roi.selected_pixels, np.asarray(roi.summed_decay, dtype=np.float64)
            )
            n_bins = result.time_ns.size
        else:
            if self.dataset is None:
                self.status_var.set("Single point mode: load a dataset first.")
                return None
            try:
                x0, y0, x1, y1 = clamp_bounds(bounds, (self.dataset.y_size, self.dataset.x_size))
            except ValueError as exc:
                self.status_var.set(str(exc))
                return None
            sub = np.asarray(self.dataset.decay[y0:y1, x0:x1, :], dtype=np.float64)
            if self.background is not None:
                try:
                    bg_data = (
                        self.background.decay if isinstance(self.background, FLIMDataset) else self.background
                    )
                    multiplier, _ = resolve_background_multiplier(self.dataset, self.background, settings)
                    bg_array = np.asarray(bg_data)
                    if bg_array.shape == self.dataset.decay.shape:
                        # A per-pixel background acquisition: subtract the
                        # matching pixels, as analyze_dataset does -- not the
                        # whole-image mean trace _prepare_background would
                        # fall back to for a smaller selection (which also
                        # re-averages the entire cube on every click).
                        prepared_background = np.asarray(bg_array[y0:y1, x0:x1], dtype=np.float32)
                    else:
                        prepared_background = _prepare_background(bg_data, sub)
                    # Subtract-then-clip *per time bin*, matching
                    # analyze_dataset's own formula (see
                    # _show_raw_intensity's identical comment) -- summing
                    # pixels first would give a different, wrong answer
                    # whenever the background exceeds the signal in some
                    # bins but not overall.
                    sub = sub - multiplier * prepared_background - settings.background_counts
                    if settings.background_clip_negative:
                        sub = np.clip(sub, 0.0, None)
                except Exception:
                    # Any background-resolution problem -- fall back to the
                    # raw, unsubtracted selection rather than blocking the
                    # whole live preview over it.
                    sub = np.asarray(self.dataset.decay[y0:y1, x0:x1, :], dtype=np.float64)
            selection = LiveFitSelection(
                (x0, y0, x1, y1), int((x1 - x0) * (y1 - y0)), sub.sum(axis=(0, 1))
            )
            n_bins = self.dataset.time_bins

        time_ns = (np.arange(n_bins, dtype=np.float64) + 0.5) * settings.dt_ns
        return selection, time_ns

    def _run_single_point_live_fit(self, bounds: BoundsXYXY) -> None:
        """Analyse just the selected pixel(s) live, using the sidebar's *current* settings.

        Deliberately limited to Phasor (phase/modulation) and FLIMLib --
        reconvolution -- the only two methods that make sense on a single
        pooled decay with no whole-image context and no dependency on a
        prior committed analysis.
        """

        if self.live_fit_busy:
            return
        self.single_point_last_bounds = bounds
        try:
            settings = self._settings()
        except ValueError as exc:
            self.status_var.set(str(exc))
            return
        method = settings.method
        if method not in ("phasor_phase", "phasor_modulation", "flimlib_reconvolution"):
            self.status_var.set(
                "Single point mode only supports Phasor (phase/modulation) and "
                f"FLIMLib -- reconvolution; '{self.method_var.get()}' isn't live-previewed."
            )
            return

        built = self._live_fit_selection(bounds, settings)
        if built is None:
            return
        selection, time_ns = built

        shape = self._live_fit_pixel_shape()
        if shape is not None:
            normalised = bounds_to_normalised(selection.bounds_xyxy, shape)
            for view in self._map_views():
                view.set_selection(normalised)

        decay = np.asarray(selection.summed_decay, dtype=np.float64)
        if method in ("phasor_phase", "phasor_modulation"):
            self._live_fit_phasor(selection, settings, decay, time_ns)
        else:
            self._live_fit_native(selection, settings, decay, time_ns)

    @staticmethod
    def _live_fit_region_label(selection: LiveFitSelection) -> str:
        return "Point" if selection.selected_pixels == 1 else "Region"

    def _live_fit_render(
        self,
        selection: LiveFitSelection,
        time_ns: np.ndarray,
        method_label: str,
        series: list,
        summary_text: str,
        status_text: str,
    ) -> None:
        """Shared renderer for every live single-point/region preview branch."""

        x0, y0, x1, y1 = selection.bounds_xyxy
        region_label = self._live_fit_region_label(selection)
        self.selection_var.set(
            f"{region_label} live preview: x=[{x0},{x1}), y=[{y0},{y1}) -- {method_label}"
        )
        self.roi_summary_var.set(f"LIVE PREVIEW (not the committed analysis): {summary_text}")
        title = f"Live {method_label} preview -- x={x0}:{x1}, y={y0}:{y1}"
        subtitle = f"LIVE PREVIEW using current sidebar settings, not the committed analysis; {summary_text}"
        compact_plot = line_plot(time_ns, series, size=(920, 360), y_label="Processed signal / bin", title=title)
        self.roi_decay_view.set_image(compact_plot, subtitle)
        detailed_plot = line_plot(time_ns, series, size=(1200, 680), y_label="Processed signal / bin", title=title)
        self.decay_view.set_image(detailed_plot, subtitle)
        self.status_var.set(status_text)

    def _live_fit_phasor(
        self, selection: LiveFitSelection, settings: AnalysisSettings, decay: np.ndarray, time_ns: np.ndarray
    ) -> None:
        irf_trace = self._live_fit_generic_irf_trace(settings)
        cube = decay.reshape(1, 1, -1)
        try:
            g, s, tau_phase, tau_mod = phasor_lifetimes(
                cube, settings.dt_ns, settings.rep_rate_mhz, settings.phasor_min_counts, irf_trace
            )
        except ValueError as exc:
            # Typically a loaded IRF whose time-bin count doesn't match this dataset.
            self.status_var.set(f"Live phasor preview: {exc} Load a matching IRF or use the synthetic IRF.")
            return
        g_val, s_val = float(g[0, 0]), float(s[0, 0])
        tau_phase_val, tau_mod_val = float(tau_phase[0, 0]), float(tau_mod[0, 0])
        is_phase = settings.method == "phasor_phase"
        tau = tau_phase_val if is_phase else tau_mod_val
        method_label = f"Phasor -- {'phase' if is_phase else 'modulation'} lifetime"
        if not np.isfinite(tau):
            summary = (
                f"below the phasor photon threshold ({settings.phasor_min_counts:g} counts) or an "
                f"invalid phasor point; G={g_val:.4f}, S={s_val:.4f}"
            )
        else:
            summary = (
                f"tau={self._tau_text(tau)}; G={g_val:.4f}, S={s_val:.4f} (phase tau="
                f"{self._tau_text(tau_phase_val)}, modulation tau={self._tau_text(tau_mod_val)}); "
                f"no fitted curve -- phasor is a coordinate, not a curve-shape fit"
            )
        series = [("Selected decay (integrated)", decay, ACCENT)]
        self._live_fit_render(
            selection, time_ns, method_label, series, summary, "Live single-point phasor preview updated."
        )

    def _live_fit_generic_irf_trace(self, settings: AnalysisSettings) -> np.ndarray | None:
        """The IRF for a live phasor preview, as a plain 1-D trace, NOT baseline-corrected.

        Follows the sidebar's *current* IRF source. The last committed
        analysis' own IRF is reused when it was made with the same source
        (and, for the synthetic Gaussian, the same FWHM setting) -- it has
        already been through that run's preprocessing. Otherwise a
        synthetic IRF is estimated afresh from the loaded data, or the raw
        loaded IRF dataset is used -- so this works even before "Run
        analysis" has ever been used, and never silently keeps calibrating
        against an IRF the sidebar no longer asks for. Deliberately not
        baseline-corrected here: ``phasor_lifetimes`` already does its own
        median-baseline correction internally, and correcting again here
        would double up on it; ``_live_fit_native_irf_trace`` below adds
        its own correction for the native fit, which does need one.
        """

        if self.result is not None and self._result_irf_matches(self.result, settings):
            trace = self.result.fit_diagnostics.get("irf_trace")
            if trace is not None:
                return np.asarray(trace, dtype=np.float64)
        if settings.irf_source == "synthetic_gaussian":
            return self._live_fit_synthetic_irf_trace(settings)
        if self.irf is None:
            return None
        raw = np.asarray(self.irf.decay, dtype=np.float64)
        return raw.mean(axis=(0, 1)) if raw.ndim == 3 else raw

    @staticmethod
    def _result_irf_matches(result: AnalysisResult, settings: AnalysisSettings) -> bool:
        if result.metadata.get("irf_source") != settings.irf_source:
            return False
        if settings.irf_source == "synthetic_gaussian":
            return float(result.settings.synthetic_irf_fwhm_ns) == float(settings.synthetic_irf_fwhm_ns)
        return True

    def _live_fit_synthetic_irf_trace(self, settings: AnalysisSettings) -> np.ndarray | None:
        """Synthetic Gaussian IRF for the current sidebar settings, without a committed analysis.

        ``analyze_dataset`` estimates it from the mean decay of the fully
        preprocessed cube. That mean decay is rebuilt here from the loaded
        dataset with the same background subtraction (including the clip
        option) and timing shift -- row/column/outlier interpolation,
        smoothing and binning barely move a whole-image mean, so this is a
        faithful stand-in. Cached on everything that feeds it, so moving
        the selection around doesn't redo a whole-cube pass each click.
        """

        dataset = self.dataset
        if dataset is None:
            return None
        params = (
            settings.background_scale_mode, settings.background_multiplier, settings.background_counts,
            settings.background_clip_negative, settings.timing_offset_bins,
            settings.synthetic_irf_fwhm_ns, settings.dt_ns,
        )
        cache = self._live_synthetic_irf_cache
        if cache is not None and cache[0] is dataset and cache[1] is self.background and cache[2] == params:
            return cache[3]

        decay = dataset.decay
        height = decay.shape[0]
        prepared: np.ndarray | float = 0.0
        multiplier = 0.0
        if self.background is not None:
            try:
                bg_data = self.background.decay if isinstance(self.background, FLIMDataset) else self.background
                multiplier, _ = resolve_background_multiplier(dataset, self.background, settings)
                prepared = _prepare_background(bg_data, decay)
            except Exception:
                prepared, multiplier = 0.0, 0.0
        total = np.zeros(decay.shape[-1], dtype=np.float64)
        for y0 in range(0, height, 64):
            block = np.asarray(decay[y0:y0 + 64], dtype=np.float32)
            if isinstance(prepared, np.ndarray) and prepared.ndim == 3 and prepared.shape[0] == height:
                reference = prepared[y0:y0 + 64]
            else:
                reference = prepared
            block = block - multiplier * reference - settings.background_counts
            if settings.background_clip_negative:
                block = np.clip(block, 0.0, None)
            total += block.sum(axis=(0, 1), dtype=np.float64)
        mean_decay = (total / float(decay.shape[0] * decay.shape[1])).astype(np.float32)
        mean_decay = shift_time_axis(mean_decay.reshape(1, 1, -1), settings.timing_offset_bins)
        time_ns = (np.arange(mean_decay.shape[-1], dtype=np.float64) + 0.5) * settings.dt_ns
        trace, _metadata = estimate_synthetic_gaussian_irf(mean_decay, time_ns, settings.synthetic_irf_fwhm_ns)
        trace = np.asarray(trace, dtype=np.float64)
        self._live_synthetic_irf_cache = (dataset, self.background, params, trace)
        return trace

    def _live_fit_native_irf_trace(self, settings: AnalysisSettings) -> np.ndarray | None:
        """The IRF trace to feed a live native fit and its curve reconstruction.

        Prefers the exact IRF the last committed analysis' own native fit
        used (``native_irf_trace`` -- already baseline-corrected under that
        run's own settings). Otherwise falls back to
        ``_live_fit_generic_irf_trace`` (works pre-analysis too),
        baseline-corrected live with the sidebar's *current* setting
        instead of whatever the last run used.
        """

        if self.result is not None and self._result_irf_matches(self.result, settings):
            native_trace = self.result.fit_diagnostics.get("native_irf_trace")
            if native_trace is not None:
                return np.asarray(native_trace, dtype=np.float64)
        raw_trace = self._live_fit_generic_irf_trace(settings)
        if raw_trace is None:
            return None
        if settings.native_irf_baseline_mode == "median":
            baseline = float(np.median(raw_trace))
        elif settings.native_irf_baseline_mode == "tail_median":
            tail_bins = min(8, raw_trace.size)
            baseline = float(np.median(raw_trace[-tail_bins:]))
        else:
            baseline = 0.0
        return np.clip(raw_trace - baseline, 0.0, None)

    def _live_fit_native(
        self, selection: LiveFitSelection, settings: AnalysisSettings, decay: np.ndarray, time_ns: np.ndarray
    ) -> None:
        """Run a native FLIMLib reconvolution fit on just this pooled selection.

        Works even before "Run analysis" has ever run (see
        ``_live_fit_native_irf_trace``) -- the fit itself still needs the
        native subprocess (crash-isolated DLL call), so this is the one
        live-preview branch that runs in a background thread.
        """

        irf_trace = self._live_fit_native_irf_trace(settings)
        if irf_trace is None:
            self.status_var.set("Single point mode needs an IRF -- load one in the sidebar first.")
            return
        try:
            # .normalized() resolves the sidebar's "Auto" native bin width
            # (stored as the sentinel -1.0) to a real value the same way
            # analyze_dataset's own cfg does -- the raw settings from
            # _settings() alone are not enough for the native fit call.
            # This selection's own decay length stands in for the dataset's
            # time-bin count, whether or not a full analysis has run.
            resolved = settings.normalized(decay.size)
        except ValueError as exc:
            self.status_var.set(str(exc))
            return

        # Mirror analyze_dataset's own native-fit bin-window trimming
        # exactly: native_fit_bins (< 0 means "all") caps how many LEADING
        # time bins the native fit itself sees -- e.g. gating out a second
        # laser period in the raw trace -- independent of how many bins the
        # pooled decay/IRF otherwise carry. Skipping this previously fed
        # the *whole* untrimmed trace to the DLL: wrong fit window, and for
        # datasets with many more raw bins than native_fit_bins, a much
        # slower/less stable fit that could look like a hang.
        native_input_bins = (
            decay.size if resolved.native_fit_bins < 0 else min(resolved.native_fit_bins, decay.size)
        )
        if irf_trace.size < native_input_bins:
            self.status_var.set(
                f"Single point mode: the available IRF has fewer time bins ({irf_trace.size}) than "
                f"the configured native fit window ({native_input_bins}) -- run a full analysis once "
                "to refresh it."
            )
            return

        x0, y0, x1, y1 = selection.bounds_xyxy
        region_label = self._live_fit_region_label(selection)
        self.selection_var.set(f"{region_label} live preview: x=[{x0},{x1}), y=[{y0},{y1}) -- fitting...")
        self.status_var.set("Running live single-point fit...")

        fit_decay = decay[:native_input_bins].reshape(1, 1, -1)
        fit_irf = np.asarray(irf_trace[:native_input_bins], dtype=np.float64)
        native_options = {
            "bin_width_ns": resolved.native_bin_width_ns,
            "model_order": resolved.model_order,
            "background_mode": resolved.fit_background_mode,
            "background_value": resolved.fit_background_value,
            "noise_model": resolved.fit_noise_model,
            # A single, already-selected point/region deliberately ignores
            # the whole-image validity threshold -- the user explicitly
            # picked this exact data to inspect.
            "min_photons": 0.0,
            "irf_mode": resolved.fit_irf_mode,
        }
        self.live_fit_busy = True

        def worker() -> None:
            try:
                fit_result = fit_in_subprocess(
                    "pixels", fit_decay, fit_irf,
                    dll_path=resolved.native_dll_path or None,
                    **native_options,
                )
            except Exception as exc:  # noqa: BLE001 -- surfaced to the user, never crashes the app
                self.root.after(0, lambda exc=exc: self._live_fit_failed(exc))
                return
            self.root.after(
                0,
                # fit_irf (not the untrimmed irf_trace) -- must match what
                # was actually fitted, for the curve reconstruction below.
                lambda: self._live_fit_native_finished(selection, resolved, fit_result, fit_irf, time_ns),
            )

        threading.Thread(target=worker, daemon=True).start()

    def _live_fit_failed(self, exc: Exception) -> None:
        self.live_fit_busy = False
        self.status_var.set(f"Live single-point fit failed: {exc}")

    def _live_fit_native_finished(
        self,
        selection: LiveFitSelection,
        settings: AnalysisSettings,
        fit_result: NativeFitResult,
        irf_trace: np.ndarray,
        time_ns: np.ndarray,
    ) -> None:
        self.live_fit_busy = False
        x0, y0, x1, y1 = selection.bounds_xyxy
        region_label = self._live_fit_region_label(selection)
        status = int(np.asarray(fit_result.fit_status, dtype=np.float64).reshape(-1)[0])
        if status != 1:
            self.selection_var.set(
                f"{region_label} live preview: x=[{x0},{x1}), y=[{y0},{y1}) -- FLIMLib reconvolution "
                f"did not converge"
            )
            self.status_var.set("Live single-point fit did not converge for this selection.")
            return

        tau_components = np.asarray(fit_result.tau_components_ns, dtype=np.float64).reshape(-1)
        amplitudes = np.asarray(fit_result.amplitude_components, dtype=np.float64).reshape(-1)
        background = float(np.asarray(fit_result.background, dtype=np.float64).reshape(-1)[0])
        chi_square = float(np.asarray(fit_result.chi_square, dtype=np.float64).reshape(-1)[0])
        order = tau_components.size
        curve = native_multiexponential_curve(
            time_ns.size,
            settings.native_bin_width_ns,
            irf_trace,
            tau_components.reshape(order, 1),
            amplitudes.reshape(order, 1),
            np.array([background], dtype=np.float64),
        )

        sort_order = np.argsort(tau_components)
        tau_sorted = tau_components[sort_order]
        amp_sorted = amplitudes[sort_order]
        component_text = "; ".join(
            f"tau{i + 1}={tau:.3f} ns (A={amp:.3g})" for i, (tau, amp) in enumerate(zip(tau_sorted, amp_sorted))
        )
        model_name = {1: "Mono", 2: "Bi", 3: "Tri"}.get(order, f"{order}-component")
        summary = f"{component_text}; chi2={chi_square:.3f}; bg={background:.3g}"
        series = [
            ("Selected decay (integrated)", selection.summed_decay, ACCENT),
            (f"Live {model_name}-exp fit", curve, AMBER),
        ]
        self._live_fit_render(
            selection, time_ns, f"FLIMLib reconvolution ({model_name.lower()}-exponential)", series, summary,
            "Live single-point fit preview updated.",
        )

    def _update_metadata(self, metadata: dict) -> None:
        self.metadata_text.configure(state="normal")
        self.metadata_text.delete("1.0", END)
        self.metadata_text.insert("1.0", json.dumps(metadata, indent=2, default=str))
        self.metadata_text.configure(state="disabled")

    def export_current(self) -> None:
        if self.busy:
            return
        if self.result is None:
            messagebox.showinfo("No result", "Run an analysis before exporting.")
            return
        folder = filedialog.askdirectory(title="Choose an output folder")
        if not folder:
            return
        try:
            written = export_result(self.result, folder)
            if self.roi_bounds is not None and self.roi_analysis is not None:
                written.extend(export_roi_analysis(self.result, self.roi_analysis, folder))
            if self.point_analyses:
                written.extend(
                    export_point_selections(
                        self.result,
                        self.point_analyses,
                        folder,
                        active_roi_bounds=self.roi_bounds,
                    )
                )
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))
            return
        self.status_var.set(f"Exported {len(written)} files to {folder}")
        messagebox.showinfo("Export complete", f"Saved {len(written)} analysis files to:\n{folder}")

    def run_batch(self) -> None:
        if self.busy:
            return
        if not self.queue_paths:
            messagebox.showinfo("Queue is empty", "Open multiple files or a folder first.")
            return
        folder = filedialog.askdirectory(title="Choose a folder for batch results")
        if not folder:
            return
        try:
            settings = self._settings()
        except ValueError as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return
        paths = list(self.queue_paths)
        irf, background = self.irf, self.background
        accumulate = bool(self.phasor_accumulate_queue_var.get())
        if accumulate:
            # Fresh per run, so a stale cloud from an earlier, unrelated
            # queue never silently lingers into this one.
            self.phasor_accumulated_points = []
            self._update_phasor_accumulate_status()
        self._set_busy(True)

        def worker() -> None:
            failures: list[str] = []
            last_result: AnalysisResult | None = None
            last_dataset: FLIMDataset | None = None
            for index, path in enumerate(paths):
                base = index / len(paths)
                if path.suffix.lower() == ".lif":
                    failures.append(f"{path.name}: raw Leica file — open it in Interrogator and export .ics first")
                    continue
                try:
                    self._progress(base, f"Batch {index + 1}/{len(paths)} — loading {path.name}")
                    dataset = load_dataset(path)
                    local_settings = AnalysisSettings(
                        **{**asdict(settings), "dt_ns": dataset.dt_ns, "rep_rate_mhz": dataset.rep_rate_mhz}
                    )
                    last_result = analyze_dataset(
                        dataset,
                        local_settings,
                        irf,
                        background,
                        lambda value, message, base=base: self._progress(
                            base + value / len(paths), f"{path.name}: {message}"
                        ),
                    )
                    last_dataset = dataset
                    export_result(last_result, folder)
                    if accumulate:
                        self.root.after(
                            0,
                            lambda name=path.name, g=last_result.g.copy(), s=last_result.s.copy(): (
                                self._append_phasor_accumulation(name, g, s)
                            ),
                        )
                except Exception as exc:
                    failures.append(f"{path.name}: {exc}")
            self.root.after(
                0, lambda: self._batch_finished(last_result, last_dataset, failures, folder, len(paths))
            )

        threading.Thread(target=worker, daemon=True, name="flim-batch").start()

    def _batch_finished(
        self,
        result: AnalysisResult | None,
        dataset: FLIMDataset | None,
        failures: list[str],
        folder: str,
        attempted: int,
    ) -> None:
        self.progress_var.set(1)
        # Batch output is written directly to disk. Keep the current on-screen
        # dataset/result paired instead of replacing it with an unnamed last
        # queue item -- UNLESS nothing at all was on screen to begin with
        # (e.g. the user queued files and ran a batch without ever viewing
        # any single result first), in which case there's nothing to
        # preserve and leaving self.result at None would otherwise make
        # phasor accumulation compute a real pooled cloud that never
        # actually gets shown anywhere.
        if self.result is None and result is not None:
            self.result = result
            if dataset is not None:
                self.dataset = dataset
            self._refresh_result_views()
        self._set_busy(False)
        if self.phasor_accumulate_queue_var.get() and self.result is not None:
            self._refresh_phasors()
        completed = attempted - len(failures)
        self.status_var.set(f"Batch complete — {completed}/{attempted} datasets exported")
        if failures:
            messagebox.showwarning(
                "Batch complete with warnings",
                f"Exported {completed} of {attempted} datasets to {folder}.\n\n" + "\n".join(failures[:12]),
            )
        else:
            messagebox.showinfo("Batch complete", f"Exported {completed} datasets to:\n{folder}")

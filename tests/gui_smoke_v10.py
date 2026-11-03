from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
import tkinter as tk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flim_app.gui import FLIMStudioApp
from flim_app.demo import create_demo_dataset, create_demo_irf
from flim_app.models import AnalysisSettings
from flim_app.processing import analyze_dataset
from flim_app.rendering import PANEL, phasor_plot


def _wheel_event(x: int, y: int, delta: int) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, delta=delta, num=None)


def _button_event(x: int, y: int) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y)


def main() -> None:
    root = tk.Tk()
    root.withdraw()
    try:
        app = FLIMStudioApp(root, start_with_demo=False)

        # Model order alone must not silently do nothing: previously nothing
        # in the UI indicated that "Model order" only takes effect for the
        # FLIMLib native methods, which made bi/tri fitting look broken.
        app.model_order_var.set("2 — bi")
        assert "only applies" in app.method_model_hint_var.get(), app.method_model_hint_var.get()
        app.method_var.set("FLIMLib — reconvolution")
        assert app.method_model_hint_var.get() == "Native fit will use 2 — bi components."
        app.model_order_var.set("1 — mono")
        assert app.method_model_hint_var.get() == ""

        # A real native bi-exponential analysis must show an actual
        # reconvolved decay curve, not just component maps.
        app.model_order_var.set("2 — bi")
        dataset = create_demo_dataset(size=8, time_bins=32, seed=95)
        app.irf = create_demo_irf(dataset.y_size, dataset.time_bins, dataset.dt_ns)
        app._set_dataset(dataset)
        settings = AnalysisSettings(
            method="flimlib_reconvolution",
            fit_min_counts=0,
            model_order=2,
            native_fit_bins=29,
        )
        app.result = analyze_dataset(dataset, settings, app.irf)
        app._refresh_result_views()
        app._update_roi_details(None)

        assert app.result.fit_diagnostics.get("native_irf_trace") is not None
        assert app.roi_analysis is None  # whole-image view, no ROI drag yet
        subtitle = app.decay_view.subtitle_var.get()
        assert "Native Bi-exponential" in subtitle, subtitle
        assert app.decay_view._source is not None

        # Zoom must default to "fit whole image" and behave identically to
        # the pre-zoom rendering unless the user actually zooms.
        view = app.decay_view
        view.canvas.configure(width=400, height=300)
        view._render()
        assert view._zoom == 1.0
        assert view._view_box_norm == (0.0, 0.0, 1.0, 1.0)
        assert view._zoom_badge_bbox is None

        # Wheel-zoom in, centred at the canvas centre.
        view._on_mouse_wheel(_wheel_event(200, 150, 120))
        assert view._zoom > 1.0
        assert view._zoom_badge_bbox is not None
        vx0, vy0, vx1, vy1 = view._view_box_norm
        assert (vx1 - vx0) < 1.0 and (vy1 - vy0) < 1.0

        # Right-drag pans; the view centre must move.
        centre_before = view._pan_center
        view._pan_or_clear_started(_button_event(200, 150))
        view._pan_or_clear_dragged(_button_event(260, 190))
        assert view._panning is True
        assert view._pan_center != centre_before
        view._pan_or_clear_finished(_button_event(260, 190))
        assert view._panning is False

        # Clicking the zoom badge resets to the default fit view.
        bx0, by0, bx1, by1 = view._zoom_badge_bbox
        badge_point = ((bx0 + bx1) / 2, (by0 + by1) / 2)
        view._selection_started(_button_event(*badge_point))
        assert view._zoom == 1.0
        assert view._pan_center == (0.5, 0.5)
        assert view._zoom_badge_bbox is None

        # A plain (non-dragging) right click on an interactive view must
        # still behave like the pre-zoom "clear ROI" gesture, not a pan.
        app.intensity_view.canvas.configure(width=300, height=300)
        app.intensity_view._render()
        app.intensity_view._pan_or_clear_started(_button_event(50, 50))
        app.intensity_view._pan_or_clear_finished(_button_event(50, 50))
        assert app.roi_bounds is None

        # Chi-square must be visible both as a whole-image metric and as a
        # selectable, ROI-aware secondary map.
        assert app.mean_chi_var.get() != "—", app.mean_chi_var.get()
        combo_choices = app.map_mode_combo.cget("values")
        assert "Chi-square" in combo_choices, combo_choices
        app.map_mode_var.set("Chi-square")
        app._refresh_secondary_map()
        assert "χ²" in app.secondary_view.title_label.cget("text")
        whole_image_subtitle = app.secondary_view.subtitle_var.get()
        assert "whole image" in whole_image_subtitle, whole_image_subtitle
        assert "mean" in whole_image_subtitle, whole_image_subtitle
        app._roi_selected((0.0, 0.0, 0.5, 0.5), committed=True)
        roi_subtitle = app.secondary_view.subtitle_var.get()
        assert "selected ROI" in roi_subtitle, roi_subtitle
        app.clear_roi()

        # The phasor plot must use this app's dark theme, not LabVIEW's grey
        # window chrome -- a corner pixel (outside the black plot rectangle)
        # should match the panel background exactly.
        phasor_image = phasor_plot(app.result)
        corner_rgb = phasor_image.getpixel((2, 2))
        expected_rgb = tuple(int(PANEL[i : i + 2], 16) for i in (1, 3, 5))
        assert corner_rgb == expected_rgb, corner_rgb

        # Phasor and Histogram are separate full-width tabs again (each its
        # own dedicated panel), not merged into one shared tab.
        assert app.phasor_view.master is not app.histogram_view.master
        assert int(app.phasor_view.grid_info()["column"]) == 0
        assert int(app.histogram_view.grid_info()["column"]) == 0

        root.update_idletasks()
        print("GUI_V10_OK", app.result.intensity.shape)
    finally:
        root.destroy()


if __name__ == "__main__":
    main()

from __future__ import annotations

import sys
from pathlib import Path
import tkinter as tk

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flim_app.gui import FLIMStudioApp
from flim_app.demo import create_demo_dataset, create_demo_irf
from flim_app.processing import analyze_dataset
from flim_app.rendering import component_amplitude_image, component_lifetime_image


def main() -> None:
    root = tk.Tk()
    root.withdraw()
    try:
        app = FLIMStudioApp(root, start_with_demo=False)
        dataset = create_demo_dataset(size=60, time_bins=32, seed=109)
        app.irf = create_demo_irf(dataset.y_size, dataset.time_bins, dataset.dt_ns)
        app._set_dataset(dataset)
        app.result = analyze_dataset(dataset, app._settings(), app.irf)
        app._refresh_result_views()
        assert app.result is not None
        assert app.bin_var.get() == "0"
        assert app.result.metadata["spatial_transform"]["spatial_bin_width"] == 1
        assert app.phasor_heatmap_var.get() is True
        app.map_mode_var.set("Intensity with threshold")
        app._refresh_secondary_map()
        assert "threshold" in app.secondary_view.title_label.cget("text").lower()
        app.live_threshold_var.set("750")
        app._apply_live_threshold_to_fit()
        assert app.phasor_threshold_var.get() == "750"
        app.column_map_columns_var.set("3, 7-8")
        app.column_map_enabled_var.set(True)
        app.column_map_radius_var.set("2")
        settings = app._settings()
        assert settings.interpolation_columns == (3, 7, 8)
        assert settings.spatial_bin == 0
        app.lifetime_scale_mode_var.set("Auto")
        app._lifetime_scale_changed()
        assert app.result.settings.lifetime_scale_mode == "auto"
        assert "Auto display range" in app.lifetime_scale_summary_var.get()
        shape = app.result.intensity.shape
        app.result.fit_diagnostics["tau_components_ns"] = np.stack(
            (np.full(shape, 3.2), np.full(shape, 0.8))
        )
        app.result.fit_diagnostics["amplitude_components"] = np.stack(
            (np.full(shape, 80.0), np.full(shape, 25.0))
        )
        app.result.fit_diagnostics["fit_status"] = np.ones(shape)
        app._component_images = [
            component_lifetime_image(app.result, index) for index in range(2)
        ]
        app._amplitude_component_images = [
            component_amplitude_image(app.result, index) for index in range(2)
        ]
        app._refresh_multiexponential_views()
        assert "Bi-exponential" in app.multi_summary_var.get()
        assert app.multi_tau_views[0]._source is not None
        assert app.multi_tau_views[1]._source is not None
        assert app.multi_tau_views[2]._source is None
        assert app.multi_amplitude_views[0]._source is not None
        root.update_idletasks()
        print("GUI_V09_OK", app.result.intensity.shape)
    finally:
        root.destroy()


if __name__ == "__main__":
    main()

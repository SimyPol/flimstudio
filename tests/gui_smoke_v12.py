from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flim_app.gui import FLIMStudioApp
from flim_app.demo import create_demo_dataset
from flim_app.models import AnalysisSettings
from flim_app.processing import analyze_dataset


def main() -> None:
    root = tk.Tk()
    root.withdraw()
    try:
        app = FLIMStudioApp(root, start_with_demo=False)

        paths = [Path(f"fixture_{seed}.npy") for seed in (1, 2, 3)]
        app._set_queue(paths)
        assert app.queue_results == {}
        assert list(app.queue_list.get(0, tk.END)) == [f"   {p.name}" for p in paths]

        pairs = []
        for seed, path in zip((1, 2, 3), paths):
            dataset = create_demo_dataset(size=6, time_bins=16, seed=seed)
            result = analyze_dataset(dataset, AnalysisSettings(method="phasor_phase", min_photons=0), None)
            pairs.append((path, dataset, result))

        # Cache with a limit of 2: adding a 3rd result must evict the oldest
        # (first) one, not the most recently viewed.
        app.queue_cache_limit_var.set("2")
        for path, dataset, result in pairs:
            app._cache_queue_result(path, dataset, result, limit=2)

        assert len(app.queue_results) == 2, app.queue_results.keys()
        assert paths[0] not in app.queue_results, "oldest result should have been evicted"
        assert paths[1] in app.queue_results and paths[2] in app.queue_results

        labels = list(app.queue_list.get(0, tk.END))
        assert labels[0] == f"   {paths[0].name}"  # evicted: no marker
        assert labels[1] == f"● {paths[1].name}"
        assert labels[2] == f"● {paths[2].name}"

        # Viewing paths[1] (the older of the two cached results) must touch
        # it to most-recently-viewed, so a subsequent cache insert evicts
        # paths[2] instead, not paths[1], despite paths[2] having been
        # cached more recently in wall-clock terms.
        app.queue_list.selection_clear(0, tk.END)
        app.queue_list.selection_set(1)
        app._queue_selected()
        assert app.dataset is pairs[1][1]
        assert app.result is pairs[1][2]
        assert app.status_var.get().startswith("Showing cached result")

        dataset4 = create_demo_dataset(size=6, time_bins=16, seed=4)
        result4 = analyze_dataset(dataset4, AnalysisSettings(method="phasor_phase", min_photons=0), None)
        path4 = Path("fixture_4.npy")
        app.queue_paths.append(path4)
        app._cache_queue_result(path4, dataset4, result4, limit=2)
        assert paths[1] in app.queue_results, "recently-viewed result must survive eviction"
        assert paths[2] not in app.queue_results, "untouched older result should be evicted instead"
        assert path4 in app.queue_results

        # Intensity log/linear toggle: defaults on, and flipping it changes
        # both the primary Intensity panel and (when selected) the second
        # map's rendering and subtitle wording.
        assert app.intensity_log_var.get() is True
        app._intensity_scale_changed()
        subtitle_log = app.intensity_view.subtitle_var.get()
        assert "log" in subtitle_log.lower(), subtitle_log
        app.intensity_log_var.set(False)
        app._intensity_scale_changed()
        subtitle_linear = app.intensity_view.subtitle_var.get()
        assert "linear" in subtitle_linear.lower(), subtitle_linear
        app.map_mode_var.set("Intensity")
        app._refresh_secondary_map()
        assert "linear" in app.secondary_view.subtitle_var.get().lower()
        app.intensity_log_var.set(True)
        app._intensity_scale_changed()
        assert "log" in app.secondary_view.subtitle_var.get().lower()

        root.update_idletasks()
        print("GUI_V12_OK")
    finally:
        root.destroy()


if __name__ == "__main__":
    main()

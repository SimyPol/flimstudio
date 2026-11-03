from __future__ import annotations

import time
import tkinter as tk
import traceback
from types import SimpleNamespace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flim_app.gui import FLIMStudioApp
from flim_app.row_interpolation import LABVIEW_MIRRORED, labview_main_preset


def point_event(view, x_fraction: float, y_fraction: float):
    x, y, width, height = view._display_box
    return SimpleNamespace(
        x=int(x + width * x_fraction),
        y=int(y + height * y_fraction),
    )


def main() -> None:
    root = tk.Tk()
    root.geometry("1160x720")
    root.withdraw()
    app = FLIMStudioApp(root, start_with_demo=True)
    deadline = time.time() + 60
    failures: list[tuple[BaseException, str]] = []

    def exercise() -> None:
        try:
            assert int(app.intensity_view.grid_info()["row"]) == 0
            assert int(app.intensity_view.grid_info()["column"]) == 0
            assert int(app.secondary_view.master.grid_info()["row"]) == 0
            assert int(app.secondary_view.master.grid_info()["column"]) == 1
            assert int(app.roi_decay_view.master.grid_info()["row"]) == 1
            assert int(app.roi_decay_view.master.grid_info()["column"]) == 0
            assert int(app.overview_phasor_view.master.grid_info()["row"]) == 1
            assert int(app.overview_phasor_view.master.grid_info()["column"]) == 1

            app._set_row_map_controls(labview_main_preset())
            preset = app._row_map_from_controls()
            assert len(preset.rows) == 19
            assert preset.coordinate_mode == LABVIEW_MIRRORED
            assert preset.expected_source_height == 512
            assert app.row_map_enabled_var.get() is False

            # Exercise a valid direct-Y map on the 160-row demo and verify the
            # processed-grid overlay survives secondary-map toggling.
            app.row_map_rows_var.set("3, 30, 54")
            app.row_map_coordinate_var.set("Direct image Y (0-based)")
            app.row_map_radius_var.set("2")
            assert app._row_map_from_controls().expected_source_height is None
            app.row_map_enabled_var.set(True)
            app.run_current()
            root.after(25, poll)
            return

        except BaseException as exc:
            failures.append((exc, traceback.format_exc()))
            root.destroy()

    def exercise_after_row_analysis() -> None:
        try:
            assert app.result is not None
            assert app.result.metadata["row_interpolation"]["applied"] is True
            assert app.result.metadata["row_interpolation"]["canonical_target_rows_y"] == [3, 30, 54]
            assert len(app.intensity_view._row_overlay) == 3
            assert app.intensity_view._row_overlay == app.secondary_view._row_overlay
            assert app.overview_phasor_view._source is not None
            assert app.overview_histogram_view._source is not None

            app.intensity_view.canvas.configure(width=400, height=300)
            app.intensity_view._render()
            for x_fraction, y_fraction in ((0.25, 0.25), (0.55, 0.48), (0.78, 0.72)):
                event = point_event(app.intensity_view, x_fraction, y_fraction)
                app.intensity_view._selection_started(event)
                app.intensity_view._selection_finished(event)
            assert len(app.selected_points) == 3
            assert len(app.point_analyses) == 3
            assert len(app.intensity_view._points) == 3
            assert app.intensity_view._points == app.secondary_view._points
            assert all("ns" in marker[2] for marker in app.intensity_view._points)
            assert all(point.phasor_valid for point in app.point_analyses)
            point_phasor_image = app.overview_phasor_view._source.tobytes()

            start = point_event(app.intensity_view, 0.1, 0.15)
            finish = point_event(app.intensity_view, 0.65, 0.70)
            app.intensity_view._selection_started(start)
            app.intensity_view._selection_dragged(finish)
            app.intensity_view._selection_finished(finish)
            assert app.roi_bounds is not None
            assert app.intensity_view._selection == app.secondary_view._selection
            assert "selected ROI" in app.overview_phasor_view.subtitle_var.get()
            assert app.overview_phasor_view._source.tobytes() != point_phasor_image

            # Phasor and histogram are shown together now, not toggled by a
            # scope selector -- both must always be present with their own
            # fixed titles.
            assert app.overview_phasor_view.title_label.cget("text") == "Selected phasors"
            assert app.overview_histogram_view.title_label.cget("text") == "Lifetime histogram"
            assert "selected ROI" in app.overview_histogram_view.subtitle_var.get()

            app.map_mode_var.set("Composite")
            app._map_mode_changed()
            assert "composite" in app.secondary_view.title_label.cget("text").lower()
            assert len(app.secondary_view._points) == 3

            print(
                "GUI_V06_OK",
                app.roi_bounds,
                [(point.identifier, point.x, point.y) for point in app.selected_points],
                app.intensity_view.grid_info()["row"],
                app.secondary_view.master.grid_info()["column"],
                app.roi_decay_view.master.grid_info()["row"],
                app.overview_phasor_view.master.grid_info()["column"],
            )
        except BaseException as exc:
            failures.append((exc, traceback.format_exc()))
        finally:
            root.destroy()

    def poll() -> None:
        if app.result is not None and not app.busy:
            if app.result.metadata.get("row_interpolation", {}).get("applied"):
                exercise_after_row_analysis()
            else:
                exercise()
        elif time.time() >= deadline:
            failures.append((TimeoutError("Demo analysis timed out"), ""))
            root.destroy()
        else:
            root.after(25, poll)

    root.after(25, poll)
    root.mainloop()
    if failures:
        exc, details = failures[0]
        raise AssertionError(details or str(exc)) from exc


if __name__ == "__main__":
    main()

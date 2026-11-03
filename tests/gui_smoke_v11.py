from __future__ import annotations

import sys
import tempfile
import tkinter as tk
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flim_app.gui import FLIMStudioApp
from flim_app.training_gui import TrainingDialog
from flim_app import training_gui as training_gui_module


def _write_npy(directory: Path, name: str, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    cube = rng.integers(0, 50, size=(4, 4, 16)).astype(np.float32)
    path = directory / name
    np.save(path, cube)
    return path


def main() -> None:
    root = tk.Tk()
    root.withdraw()
    try:
        app = FLIMStudioApp(root, start_with_demo=False)
        dialog = TrainingDialog(app)
        dialog.withdraw()

        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            decay_a = _write_npy(directory, "decay_a.npy", 1)
            decay_b = _write_npy(directory, "decay_b.npy", 2)
            shared_irf = _write_npy(directory, "shared_irf.npy", 3)

            # Shared IRF: chosen once, then reused silently for every image
            # added afterwards -- no per-image IRF dialog should fire.
            # _add_image() itself now uses askopenfilenames (plural,
            # multi-select) for the decay files, not askopenfilename --
            # both need mocking, since they're genuinely different
            # filedialog functions, not just a renamed one.
            call_log: list[str] = []

            def fake_askopenfilename(*_args, **_kwargs):
                call_log.append("askopenfilename")
                return str(shared_irf)

            def fake_askopenfilenames(*_args, **_kwargs):
                call_log.append("askopenfilenames")
                return (str(decay_a), str(decay_b))

            def fake_askyesno(*_args, **_kwargs):
                call_log.append("askyesno")
                return False  # no background for these images

            training_gui_module.filedialog.askopenfilename = fake_askopenfilename
            training_gui_module.filedialog.askopenfilenames = fake_askopenfilenames
            training_gui_module.messagebox.askyesno = fake_askyesno

            dialog._choose_shared_irf()
            assert dialog.shared_irf is not None
            assert "shared_irf.npy" in dialog.shared_irf_label_var.get()
            assert call_log == ["askopenfilename"]

            # One multi-select picker call adds both decay files at once --
            # the point of the multi-select feature this now exercises.
            dialog._add_image()
            assert len(dialog.items) == 2
            # One multi-select decay-files prompt plus the earlier shared-IRF
            # prompt = 1 askopenfilename + 1 askopenfilenames; critically, no
            # extra prompts were made for IRF per image, and a background
            # prompt (askyesno) fired once per image since no shared
            # background was set.
            assert call_log.count("askopenfilename") == 1, call_log
            assert call_log.count("askopenfilenames") == 1, call_log
            assert call_log.count("askyesno") == 2, call_log
            assert dialog.items[0].irf is dialog.shared_irf
            assert dialog.items[1].irf is dialog.shared_irf
            row_values = dialog.tree.item(dialog.tree.get_children()[0], "values")
            assert row_values[2].endswith("(shared)"), row_values

            dialog._clear_shared_irf()
            assert dialog.shared_irf is None
            assert "None set" in dialog.shared_irf_label_var.get()

        dialog.destroy()
        root.update_idletasks()
        print("GUI_V11_OK")
    finally:
        root.destroy()


if __name__ == "__main__":
    main()

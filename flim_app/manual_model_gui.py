from __future__ import annotations

"""Manual global-model dialog: fix the shared lifetime(s) directly, with
no search/training run at all.

For when you already know the values you want -- literature, a separate
reference measurement, a previous fit -- and just want to fit
amplitude/background per pixel against them via "FLIMLib — apply trained
global model" (``flimlib_apply_global``), without running
``flim_train_global_model_v1``'s search at all. Deliberately reuses the
main window's current native-fit settings (model order is implied by how
many taus you type; noise model, background mode, IRF mode, ... come
from the sidebar) rather than duplicating that whole control set here --
matches how the Train dialog itself already works.
"""

import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING

import numpy as np

from .global_model import TrainedGlobalModel, build_trained_model, new_model_name, save_global_model
from .rendering import BG

if TYPE_CHECKING:
    from .gui import FLIMStudioApp


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]', "_", name).strip()
    return cleaned or "global_model"


class ManualModelDialog(tk.Toplevel):
    def __init__(self, app: "FLIMStudioApp") -> None:
        super().__init__(app.root)
        self.app = app
        self.title("Set Global Model Manually")
        self.geometry("560x360")
        self.configure(bg=BG)
        self.transient(app.root)

        self.taus_var = tk.StringVar(value="")
        self.model_name_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Enter 1-3 lifetimes (ns), then Create model.")
        self.model: TrainedGlobalModel | None = None
        self.saved_path: str | None = None

        self._build()

    def _build(self) -> None:
        ttk.Label(self, text="Set global model manually", style="PanelTitle.TLabel").pack(
            anchor="w", padx=14, pady=(12, 2)
        )
        ttk.Label(
            self,
            text=(
                "Fix the shared lifetime(s) directly instead of training/searching for them -- "
                "useful when you already know the values (literature, a reference measurement, a "
                "previous fit) and just want to fit amplitude/background per pixel against them "
                "(\"FLIMLib — apply trained global model\"). Uses the main window's current "
                "native-fit settings (Noise model, Background mode, IRF mode, ...) the same way "
                "\"Train new model…\" does -- set those in the sidebar first. No fitting or search "
                "runs here at all; the values you type are used exactly as entered."
            ),
            style="Muted.TLabel",
            wraplength=520,
            justify="left",
        ).pack(anchor="w", padx=14, pady=(0, 10))

        taus_row = ttk.Frame(self)
        taus_row.pack(fill=tk.X, padx=14, pady=(0, 6))
        ttk.Label(
            taus_row, text="Global τ values (ns, comma separated, 1-3)", style="Sidebar.TLabel"
        ).pack(anchor="w")
        ttk.Entry(taus_row, textvariable=self.taus_var).pack(fill=tk.X, pady=(2, 0))

        name_row = ttk.Frame(self)
        name_row.pack(fill=tk.X, padx=14, pady=(0, 10))
        ttk.Label(name_row, text="Model name", style="Sidebar.TLabel").pack(anchor="w")
        ttk.Entry(name_row, textvariable=self.model_name_var).pack(fill=tk.X, pady=(2, 0))

        action_row = ttk.Frame(self)
        action_row.pack(fill=tk.X, padx=14, pady=(0, 6))
        ttk.Button(
            action_row, text="Create model", command=self._create, style="Primary.TButton"
        ).pack(side=tk.LEFT)
        self.save_button = ttk.Button(
            action_row, text="Save model…", command=self._save_model, state="disabled"
        )
        self.save_button.pack(side=tk.LEFT, padx=(6, 0))
        self.use_button = ttk.Button(
            action_row, text="Use this model now", command=self._use_now, state="disabled"
        )
        self.use_button.pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(action_row, text="Close", command=self.destroy).pack(side=tk.RIGHT)

        ttk.Label(
            self, textvariable=self.status_var, style="Muted.TLabel", wraplength=520, justify="left"
        ).pack(anchor="w", padx=14, pady=(4, 12))

    def _parse_taus(self) -> tuple[float, ...]:
        raw = self.taus_var.get().strip()
        if not raw:
            raise ValueError("Enter at least one lifetime (ns).")
        values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
        if not (1 <= len(values) <= 3):
            raise ValueError("Enter 1 to 3 lifetimes, comma separated.")
        if any(not np.isfinite(value) or value <= 0 for value in values):
            raise ValueError("Every lifetime must be a finite, positive number of nanoseconds.")
        return values

    def _create(self) -> None:
        try:
            taus = self._parse_taus()
        except ValueError as exc:
            messagebox.showerror("Invalid lifetimes", str(exc), parent=self)
            return
        try:
            settings = self.app._settings()
        except ValueError as exc:
            messagebox.showerror("Invalid settings", str(exc), parent=self)
            return

        model_order = len(taus)
        # Longest-to-shortest -- matches this app's own sort convention
        # for global taus elsewhere (e.g. train_global_model's output).
        taus_sorted = tuple(sorted(taus, reverse=True))
        # Only used to clamp fit_start_bin/fit_end_bin/native_fit_bins in
        # .normalized() below; harmless if it doesn't match whatever image
        # this model eventually gets applied to -- analyze_dataset's own
        # apply-model branch re-clamps native_fit_bins etc. against the
        # real data's own bin count at that point regardless (see
        # native_input_bins = min(cfg.native_fit_bins, cube.shape[-1])).
        time_bins = self.app.dataset.time_bins if self.app.dataset is not None else 4096
        cfg = settings.normalized(time_bins)
        model_name = self.model_name_var.get().strip() or new_model_name(model_order)

        model = build_trained_model(
            name=model_name,
            global_taus_ns=np.asarray(taus_sorted, dtype=np.float64),
            # NaN throughout: no fit was run, so there's no real
            # objective/iteration-count/pixel-count to report -- build_trained_model
            # already turns a non-finite stat into None rather than a
            # fabricated number (see its own stat() helper).
            global_stats=np.full(10, np.nan),
            model_order=model_order,
            dt_ns=cfg.dt_ns,
            background_mode=cfg.fit_background_mode,
            background_value=cfg.fit_background_value,
            noise_model=cfg.fit_noise_model,
            fit_min_counts=cfg.fit_min_counts,
            fit_start_bin=cfg.fit_start_bin,
            fit_end_bin=cfg.fit_end_bin,
            native_fit_bins=cfg.native_fit_bins,
            native_bin_width_ns=cfg.native_bin_width_ns,
            native_irf_baseline_mode=cfg.native_irf_baseline_mode,
            irf_mode=cfg.fit_irf_mode,
            irf_integrate_per_column=cfg.irf_integrate_per_column,
            spatial_bin=cfg.spatial_bin,
            timing_offset_bins=cfg.timing_offset_bins,
            background_scale_mode=cfg.background_scale_mode,
            background_multiplier=cfg.background_multiplier,
            background_auto_fallback=cfg.background_auto_fallback,
            background_counts=cfg.background_counts,
            global_iterations=0,
            max_global_pixels=cfg.max_global_pixels,
            initial_taus_ns=(),
            dll_sha256="",
            training_images=(),
        )
        self.model = model
        self.saved_path = None
        self.save_button.configure(state="normal")
        self.use_button.configure(state="disabled")
        taus_text = ", ".join(f"{tau:.4g} ns" for tau in taus_sorted)
        self.status_var.set(f"Created \"{model_name}\": {taus_text} (no data fitted). Save, then Use.")

    def _save_model(self) -> None:
        if self.model is None:
            return
        path = filedialog.asksaveasfilename(
            title="Save global model",
            defaultextension=".json",
            filetypes=[("FLIM Studio global model", "*.json")],
            initialfile=f"{_safe_filename(self.model.name)}.json",
            parent=self,
        )
        if not path:
            return
        try:
            save_global_model(self.model, path)
        except OSError as exc:
            messagebox.showerror("Could not save model", str(exc), parent=self)
            return
        self.saved_path = path
        self.status_var.set(f"Saved: {path}")
        self.use_button.configure(state="normal")

    def _use_now(self) -> None:
        if self.model is None or not self.saved_path:
            return
        self.app._use_trained_model(self.model, self.saved_path)
        self.destroy()

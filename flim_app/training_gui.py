from __future__ import annotations

"""Training-mode dialog: build a multi-image training set and train one
shared global lifetime model (``flim_train_global_model_v1``).

Deliberately reuses the main window's current native-fit settings (model
order, noise model, background mode, IRF mode, global iterations, max
pixels) rather than duplicating that whole control set here -- set those in
the sidebar first, then open this dialog to add training images and train.
"""

import re
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

from .global_model import TrainedGlobalModel, new_model_name, save_global_model
from .io import load_dataset
from .models import FLIMDataset
from .processing import TrainingImageInput, train_global_model
from .rendering import BG

if TYPE_CHECKING:
    from .gui import FLIMStudioApp


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]', "_", name).strip()
    return cleaned or "global_model"


class TrainingDialog(tk.Toplevel):
    def __init__(self, app: "FLIMStudioApp") -> None:
        super().__init__(app.root)
        self.app = app
        self.title("Train Global Lifetime Model")
        self.geometry("780x580")
        self.configure(bg=BG)
        self.transient(app.root)

        self.items: list[TrainingImageInput] = []
        self.model_name_var = tk.StringVar(value="")
        # Third seeding mode alongside unseeded (the default: nothing set
        # here, native multi-start) and manual (sidebar's own "Initial τ
        # values" field) -- see processing.train_global_model's own
        # phasor_seed docstring and RECONSTRUCTION_NOTES.md for the full
        # method. Lives in THIS dialog, not the sidebar, since it's a
        # property of a specific training run, not a general fit setting.
        self.phasor_seed_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Add training images, then click Train.")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.busy = False
        self.trained_model: TrainedGlobalModel | None = None
        self.saved_path: str | None = None

        # Optional shared IRF/background, used for every image added while
        # set instead of prompting per image -- matches the DLL's irfMode 0
        # ("one shared IRF for all images"). Set the sidebar's IRF mode to
        # "Global IRF" to actually train under that mode; this only controls
        # what this dialog asks for while building the training set.
        self.shared_irf: FLIMDataset | None = None
        self.shared_irf_path = ""
        self.shared_irf_label_var = tk.StringVar(value="None set — asked for each image")
        self.shared_background: FLIMDataset | None = None
        self.shared_background_path = ""
        self.shared_background_label_var = tk.StringVar(value="None set — asked for each image")

        self._build()

    def _build(self) -> None:
        ttk.Label(self, text="Global lifetime training", style="PanelTitle.TLabel").pack(
            anchor="w", padx=14, pady=(12, 2)
        )
        ttk.Label(
            self,
            text=(
                "Add one row per training image. Each needs an IRF (and optionally a background) — "
                "set a shared one below to reuse it for every image, or leave unset to be asked per "
                "image. \"Add image…\" accepts multiple files at once (ctrl/shift-click in the file "
                "picker) — with a shared IRF/background set above, that adds every selected file "
                "straight in, with no further prompts. Training uses the main window's current "
                "native-fit settings — Model order, Noise model, Background mode, IRF mode, Global "
                "iterations, Max pixels — so set those in the sidebar first. After training, \"Px "
                "used/total\" shows how many pixels of each image actually went into the fit — every "
                "row should show a non-trivial share, not just the first one or two."
            ),
            style="Muted.TLabel",
            wraplength=740,
            justify="left",
        ).pack(anchor="w", padx=14, pady=(0, 8))

        shared_row = ttk.Frame(self)
        shared_row.pack(fill=tk.X, padx=14, pady=(0, 8))
        shared_row.columnconfigure(1, weight=1)
        ttk.Label(shared_row, text="Shared IRF", style="Sidebar.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            shared_row, textvariable=self.shared_irf_label_var, style="Muted.TLabel"
        ).grid(row=0, column=1, sticky="w", padx=(8, 8))
        ttk.Button(shared_row, text="Choose…", command=self._choose_shared_irf).grid(row=0, column=2)
        ttk.Button(shared_row, text="Clear", command=self._clear_shared_irf).grid(row=0, column=3, padx=(4, 0))
        ttk.Label(shared_row, text="Shared background", style="Sidebar.TLabel").grid(
            row=1, column=0, sticky="w", pady=(4, 0)
        )
        ttk.Label(
            shared_row, textvariable=self.shared_background_label_var, style="Muted.TLabel"
        ).grid(row=1, column=1, sticky="w", padx=(8, 8), pady=(4, 0))
        ttk.Button(shared_row, text="Choose…", command=self._choose_shared_background).grid(
            row=1, column=2, pady=(4, 0)
        )
        ttk.Button(shared_row, text="Clear", command=self._clear_shared_background).grid(
            row=1, column=3, padx=(4, 0), pady=(4, 0)
        )

        columns = ("name", "decay", "irf", "background", "selected")
        self.tree = ttk.Treeview(self, columns=columns, show="headings", height=8)
        for column, label, width in (
            ("name", "Image", 150),
            ("decay", "Decay file", 190),
            ("irf", "IRF file", 190),
            ("background", "Background file", 150),
            ("selected", "Px used/total", 110),
        ):
            self.tree.heading(column, text=label)
            self.tree.column(column, width=width, anchor="w")
        self.tree.pack(fill=tk.BOTH, expand=True, padx=14)

        button_row = ttk.Frame(self)
        button_row.pack(fill=tk.X, padx=14, pady=(6, 10))
        ttk.Button(button_row, text="Add image(s)…", command=self._add_image).pack(side=tk.LEFT)
        ttk.Button(button_row, text="Remove selected", command=self._remove_selected).pack(
            side=tk.LEFT, padx=(6, 0)
        )
        ttk.Button(button_row, text="Clear all", command=self._clear_all).pack(side=tk.LEFT, padx=(6, 0))

        name_row = ttk.Frame(self)
        name_row.pack(fill=tk.X, padx=14, pady=(0, 6))
        ttk.Label(name_row, text="Model name", style="Sidebar.TLabel").pack(side=tk.LEFT)
        ttk.Entry(name_row, textvariable=self.model_name_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0)
        )

        ttk.Checkbutton(
            self,
            text="Auto-seed from phasor analysis (multi-stretched distribution model)",
            variable=self.phasor_seed_var,
        ).pack(anchor="w", padx=14, pady=(0, 0))
        ttk.Label(
            self,
            text=(
                "Only applies to the multi-stretched distribution model (sidebar's Distribution "
                "model, Model order ≥ 2) and only when the sidebar's Initial τ values field is "
                "left blank -- an explicit Initial τ values entry always wins over this. Derives a "
                "starting (τ, h) guess for each component automatically from this training set's "
                "own phasor structure, instead of the native fit's own blind multi-start search -- "
                "reliably finds all the components without needing any manually-typed guess. Best "
                "effort: falls back to the ordinary unseeded search if a usable seed can't be "
                "derived (e.g. too few valid pixels)."
            ),
            style="Muted.TLabel",
            wraplength=740,
            justify="left",
        ).pack(anchor="w", padx=14, pady=(2, 6))

        action_row = ttk.Frame(self)
        action_row.pack(fill=tk.X, padx=14, pady=(0, 6))
        self.train_button = ttk.Button(
            action_row, text="Train", command=self._run_training, style="Primary.TButton"
        )
        self.train_button.pack(side=tk.LEFT)
        self.save_button = ttk.Button(
            action_row, text="Save model…", command=self._save_model, state="disabled"
        )
        self.save_button.pack(side=tk.LEFT, padx=(6, 0))
        self.use_button = ttk.Button(
            action_row, text="Use this model now", command=self._use_now, state="disabled"
        )
        self.use_button.pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(action_row, text="Close", command=self.destroy).pack(side=tk.RIGHT)

        ttk.Progressbar(self, variable=self.progress_var, maximum=1.0).pack(
            fill=tk.X, padx=14, pady=(4, 4)
        )
        ttk.Label(
            self, textvariable=self.status_var, style="Muted.TLabel", wraplength=740, justify="left"
        ).pack(anchor="w", padx=14, pady=(0, 12))

    def _choose_shared_irf(self) -> None:
        path = filedialog.askopenfilename(
            title="Select the shared IRF for every training image",
            filetypes=self.app._file_types(),
            parent=self,
        )
        if not path:
            return
        try:
            self.shared_irf = load_dataset(path)
        except Exception as exc:
            messagebox.showerror("Could not load IRF", str(exc), parent=self)
            return
        self.shared_irf_path = path
        self.shared_irf_label_var.set(f"{Path(path).name} ({self.shared_irf.time_bins} bins)")

    def _clear_shared_irf(self) -> None:
        self.shared_irf = None
        self.shared_irf_path = ""
        self.shared_irf_label_var.set("None set — asked for each image")

    def _choose_shared_background(self) -> None:
        path = filedialog.askopenfilename(
            title="Select the shared background for every training image",
            filetypes=self.app._file_types(),
            parent=self,
        )
        if not path:
            return
        try:
            self.shared_background = load_dataset(path)
        except Exception as exc:
            messagebox.showerror("Could not load background", str(exc), parent=self)
            return
        self.shared_background_path = path
        self.shared_background_label_var.set(f"{Path(path).name} ({self.shared_background.time_bins} bins)")

    def _clear_shared_background(self) -> None:
        self.shared_background = None
        self.shared_background_path = ""
        self.shared_background_label_var.set("None set — asked for each image")

    def _add_image(self) -> None:
        # Multi-select (ctrl/shift-click in the file picker): with a shared
        # IRF/background set above, every selected file adds straight in
        # with no further prompts -- this is what makes "load several
        # images at once" actually one click instead of N. Without a
        # shared IRF/background, each file below still gets its own
        # interactive IRF/background prompt (there's no other sensible way
        # to pair per-image reference data), same as adding one at a time
        # ever did -- multi-select there just saves re-opening this same
        # file picker from the same folder N times.
        filetypes = self.app._file_types()
        decay_paths = filedialog.askopenfilenames(
            title="Select one or more decay images", filetypes=filetypes, parent=self
        )
        if not decay_paths:
            return
        skipped: list[tuple[str, str]] = []
        for decay_path in decay_paths:
            reason = self._add_one_training_image(decay_path, filetypes)
            if reason is not None:
                skipped.append((Path(decay_path).name, reason))
        if skipped:
            if len(skipped) == len(decay_paths) == 1:
                # Exactly what a single failed add always showed.
                messagebox.showerror("Could not add training image", skipped[0][1], parent=self)
            else:
                detail = "\n".join(f"- {name}: {reason}" for name, reason in skipped)
                messagebox.showwarning(
                    "Some images were not added",
                    f"{len(decay_paths) - len(skipped)} of {len(decay_paths)} added.\n\n{detail}",
                    parent=self,
                )
        self.status_var.set(f"{len(self.items)} training image(s) added.")

    def _add_one_training_image(self, decay_path: str, filetypes: list[tuple[str, str]]) -> str | None:
        """Add a single training-set row for ``decay_path``.

        Returns ``None`` on success (the row is already in ``self.items``/
        the tree), or a short reason string if it was skipped/failed --
        the caller batches these into one combined message rather than
        popping a dialog per file, so adding many files at once (the
        point of this) doesn't turn into a wall of interruptions.
        """

        if self.shared_irf is not None:
            irf = self.shared_irf
            irf_label = f"{Path(self.shared_irf_path).name} (shared)"
        else:
            irf_path = filedialog.askopenfilename(
                title=f"Select the IRF for {Path(decay_path).name}", filetypes=filetypes, parent=self
            )
            if not irf_path:
                return "no IRF provided"
            try:
                irf = load_dataset(irf_path)
            except Exception as exc:
                return f"could not load IRF ({exc})"
            irf_label = Path(irf_path).name

        if self.shared_background is not None:
            background: FLIMDataset | None = self.shared_background
            background_label = f"{Path(self.shared_background_path).name} (shared)"
        else:
            background = None
            background_label = "—"
            if messagebox.askyesno(
                "Background", f"Add a background acquisition for {Path(decay_path).name}?", parent=self
            ):
                background_path = filedialog.askopenfilename(
                    title=f"Select the background for {Path(decay_path).name}",
                    filetypes=filetypes, parent=self,
                )
                if background_path:
                    try:
                        background = load_dataset(background_path)
                    except Exception as exc:
                        return f"could not load background ({exc})"
                    background_label = Path(background_path).name

        try:
            dataset = load_dataset(decay_path)
            if irf.time_bins != dataset.time_bins:
                raise ValueError(
                    f"IRF has {irf.time_bins} time bins, but the image has {dataset.time_bins}."
                )
            if background is not None and background.time_bins != dataset.time_bins:
                raise ValueError(
                    f"Background has {background.time_bins} time bins, but the image has "
                    f"{dataset.time_bins}."
                )
        except Exception as exc:
            return str(exc)

        item = TrainingImageInput(dataset=dataset, irf=irf, background=background, name=dataset.name)
        self.items.append(item)
        self.tree.insert(
            "",
            tk.END,
            values=(dataset.name, Path(decay_path).name, irf_label, background_label, "—"),
        )
        return None

    def _remove_selected(self) -> None:
        selection = self.tree.selection()
        if not selection:
            return
        indices = sorted((self.tree.index(entry) for entry in selection), reverse=True)
        for index in indices:
            del self.items[index]
        for entry in selection:
            self.tree.delete(entry)
        self.status_var.set(f"{len(self.items)} training image(s) remaining.")

    def _clear_all(self) -> None:
        self.items = []
        for entry in self.tree.get_children():
            self.tree.delete(entry)
        self.status_var.set("Add training images, then click Train.")

    def _set_busy(self, value: bool) -> None:
        self.busy = value
        self.train_button.configure(state="disabled" if value else "normal")

    def _run_training(self) -> None:
        if self.busy:
            return
        if not self.items:
            messagebox.showinfo(
                "No training images", "Add at least one training image first.", parent=self
            )
            return
        try:
            settings = self.app._settings()
        except ValueError as exc:
            messagebox.showerror("Invalid settings", str(exc), parent=self)
            return
        model_name = self.model_name_var.get().strip() or new_model_name(settings.model_order)
        items = list(self.items)
        # Snapshotted together with `items` above, not read fresh from
        # self.tree once training finishes -- nothing here disables
        # Add/Remove/Clear during a run (only the Train button itself is),
        # so the tree's rows could have changed by the time this training
        # run actually completes. Matched 1:1 with `items`/`model.training_images`
        # by shared insertion order.
        row_ids = list(self.tree.get_children())
        self._set_busy(True)
        self.progress_var.set(0.0)
        self.status_var.set("Starting training…")
        self.save_button.configure(state="disabled")
        self.use_button.configure(state="disabled")

        def progress(value: float, message: str) -> None:
            self.after(0, lambda: (self.progress_var.set(value), self.status_var.set(message)))

        phasor_seed = bool(self.phasor_seed_var.get())

        def worker() -> None:
            try:
                model = train_global_model(items, settings, progress, name=model_name, phasor_seed=phasor_seed)
            except Exception as exc:
                details = traceback.format_exc()
                self.after(0, lambda exc=exc, details=details: self._training_failed(exc, details))
                return
            self.after(0, lambda: self._training_finished(model, row_ids))

        threading.Thread(target=worker, daemon=True, name="flim-global-train").start()

    def _training_finished(self, model: TrainedGlobalModel, row_ids: list[str]) -> None:
        self._set_busy(False)
        self.trained_model = model
        self.saved_path = None
        self.progress_var.set(1.0)
        self.status_var.set(model.summary())
        self.save_button.configure(state="normal")
        # Per-image pixel counts actually used for this run -- the direct
        # way to confirm every training image contributed pixels rather
        # than the fit silently running on only the first one or two (see
        # the per-image mask subsampling in processing.train_global_model).
        for row_id, provenance in zip(row_ids, model.training_images):
            if self.tree.exists(row_id):
                self.tree.set(
                    row_id, "selected", f"{provenance.selected_pixels:,}/{provenance.total_pixels:,}"
                )

    def _training_failed(self, exc: Exception, details: str) -> None:
        self._set_busy(False)
        self.progress_var.set(0.0)
        self.status_var.set("Training failed")
        messagebox.showerror(
            "Training failed", f"{exc}\n\nTechnical details:\n{details[-1800:]}", parent=self
        )

    def _save_model(self) -> None:
        if self.trained_model is None:
            return
        path = filedialog.asksaveasfilename(
            title="Save trained global model",
            defaultextension=".json",
            filetypes=[("FLIM Studio global model", "*.json")],
            initialfile=f"{_safe_filename(self.trained_model.name)}.json",
            parent=self,
        )
        if not path:
            return
        try:
            save_global_model(self.trained_model, path)
        except OSError as exc:
            messagebox.showerror("Could not save model", str(exc), parent=self)
            return
        self.saved_path = path
        self.status_var.set(f"Saved: {path}")
        self.use_button.configure(state="normal")

    def _use_now(self) -> None:
        if self.trained_model is None or not self.saved_path:
            return
        self.app._use_trained_model(self.trained_model, self.saved_path)
        self.destroy()

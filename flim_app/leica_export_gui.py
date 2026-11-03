from __future__ import annotations

"""Quick picker for converting a .lif straight into FLIM Studio's queue.

Scans the .lif with a ``leica_flim_converter`` package (via
``leica_worker.py``, run as a separate process -- see that module's
docstring for why) and lets the user pick which of the datasets inside it
to export, exports the chosen ones to .ics, and adds the results straight
to the main window's queue. No separate Interrogator GUI window is involved
in this path -- see the sidebar's "Open full Interrogator GUI..." button
for that instead.

``resolve_leica_backend()`` decides *which* ``leica_flim_converter`` and
*which* interpreter to run it under: an externally configured Interrogator
install (its own venv) when the user has set one, otherwise the copy
vendored inside FLIM Studio itself (see ``flim_app/leica_bundled/``) run
under FLIM Studio's own interpreter -- which is what lets a packaged .exe
read .lif files with no separate install at all.
"""

import json
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING, Any, Callable

from ._paths import app_root, bundled_leica_root, is_frozen
from .rendering import BG

if TYPE_CHECKING:
    from .gui import FLIMStudioApp


def _disable_combobox_wheel(combo: ttk.Combobox) -> None:
    """Stops a scroll of the mouse wheel over this dropdown from silently
    cycling its selected value -- ttk.Combobox has this baked into its
    own default widget-class bindings (Tk, not something this app
    added). Binding at the instance level intervenes before that
    class-level default runs and "break" stops it from running at all;
    this dialog has nothing scrollable underneath a dropdown to redirect
    the scroll to (see gui.FLIMStudioApp._disable_combobox_wheel for the
    version that does, in the main window's scrollable sidebar), so
    simply suppressing the value-cycling is enough here."""

    def _on_wheel(_event: tk.Event) -> str:
        return "break"

    combo.bind("<MouseWheel>", _on_wheel)
    combo.bind("<Button-4>", _on_wheel)
    combo.bind("<Button-5>", _on_wheel)


# Resolved via app_root() (not Path(__file__).with_name(...)) so this keeps
# working in a packaged build: this module itself is compiled into FLIM
# Studio's own frozen archive there, so its __file__ doesn't point at a
# real file on disk for an *external* interpreter to run -- app_root()
# instead resolves to wherever this project's .spec ships loose data files
# (see flim_studio.spec's `datas`, which includes this script for exactly
# this reason), which is a real, findable path either way.
WORKER_SCRIPT = app_root() / "leica_worker.py"
SCAN_TIMEOUT_S = 120

# fast/balanced/max map to leica_flim_converter.ics.GZIP_LEVEL_PRESETS --
# duplicated here rather than imported, since this runs in FLIM Studio's
# own process while that package only exists in a worker subprocess.
GZIP_LEVEL_CHOICES = {"Fast": 1, "Balanced": 4, "Max": 6}

ProgressCallback = Callable[[str, float, float], None]

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _venv_python(interrogator_root: Path) -> Path:
    return interrogator_root / ".venv" / "Scripts" / "python.exe"


def resolve_leica_backend(external_root: Path | None) -> tuple[Path, list[str]] | tuple[None, None]:
    """Pick which ``leica_flim_converter`` to use and how to invoke its
    worker, returning ``(root, command_prefix)`` -- ``command_prefix`` is
    everything before ``[str(root), mode, ...]`` in the worker command line.

    ``external_root`` (an explicitly configured "Leica Interrogator
    Desktop" folder) always wins when set, since it's required anyway for
    the "Full GUI..." button and keeps decoding on the exact environment
    Interrogator's own decoder is built/tested against. Otherwise falls
    back to FLIM Studio's own bundled copy, if this build includes one.
    Returns ``(None, None)`` when neither is available.
    """

    if external_root is not None:
        return external_root, [str(_venv_python(external_root)), str(WORKER_SCRIPT)]
    bundled = bundled_leica_root()
    if bundled is not None:
        if is_frozen():
            # No standalone leica_worker.py file on disk in a packaged
            # build -- self-relaunch this same exe instead (see
            # run_flim_studio.py's --leica-worker branch).
            return bundled, [sys.executable, "--leica-worker"]
        return bundled, [sys.executable, str(WORKER_SCRIPT)]
    return None, None


def _run_worker(command_prefix: list[str], root: Path, args: list[str], timeout: float) -> dict[str, Any]:
    """Run leica_worker.py (via ``command_prefix``) and parse its JSON.

    Raises RuntimeError with a message ready to show the user directly on
    any failure -- missing interpreter, a non-JSON crash, a timeout, or an
    explicit ``{"error": ...}`` from the worker itself.
    """

    interpreter = Path(command_prefix[0])
    if not interpreter.is_file():
        raise RuntimeError(
            f"Could not find the Python interpreter needed to read .lif files:\n{interpreter}\n\n"
            "If you configured an external \"Leica Interrogator Desktop\" folder in the "
            "sidebar, run install_packages.bat there first -- or clear that folder to fall "
            "back to FLIM Studio's built-in .lif reader instead."
        )
    try:
        completed = subprocess.run(
            [*command_prefix, str(root), *args],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=_NO_WINDOW,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"The .lif reader took too long to respond ({timeout:g}s timeout).") from exc
    except OSError as exc:
        raise RuntimeError(f"Could not launch {interpreter}:\n{exc}") from exc

    stdout = (completed.stdout or "").strip()
    if not stdout:
        detail = (completed.stderr or "").strip()[-2000:]
        raise RuntimeError(
            "Leica Interrogator's converter produced no output"
            + (f":\n\n{detail}" if detail else f" (exit code {completed.returncode}).")
        )
    try:
        payload = json.loads(stdout.splitlines()[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Could not parse Leica Interrogator's output:\n\n{stdout[-2000:]}") from exc
    if "error" in payload:
        raise RuntimeError(str(payload["error"]))
    return payload


def _stream_export(
    command_prefix: list[str], root: Path, args: list[str], on_progress: ProgressCallback,
    on_process_started: Callable[[subprocess.Popen], None],
) -> dict[str, Any]:
    """Like ``_run_worker``, but streams ``{"progress": ...}`` lines to
    ``on_progress`` as they arrive rather than waiting for the whole
    subprocess to finish.

    Exporting a large multi-plane series can legitimately take minutes;
    without this, the dialog looks exactly the same whether it's working
    normally or genuinely stuck. ``on_process_started`` hands the caller
    the live ``Popen`` so a "Close" click can terminate it rather than
    leaving it running in the background after the dialog is gone.
    """

    interpreter = Path(command_prefix[0])
    if not interpreter.is_file():
        raise RuntimeError(
            f"Could not find the Python interpreter needed to read .lif files:\n{interpreter}\n\n"
            "If you configured an external \"Leica Interrogator Desktop\" folder in the "
            "sidebar, run install_packages.bat there first -- or clear that folder to fall "
            "back to FLIM Studio's built-in .lif reader instead."
        )
    # -u (unbuffered stdout) only makes sense when command_prefix is a real
    # "python.exe script.py"-style invocation -- the frozen self-relaunch
    # case (command_prefix ending in the --leica-worker *flag*, not a
    # script path) is this same exe answering its own argv parsing, where
    # an extra "-u" would just become a bogus positional argument. That
    # case doesn't need it anyway: leica_worker.py's _emit() already calls
    # flush=True on every line that matters for this protocol; -u was only
    # ever backup-safety for stray output from elsewhere in the import
    # chain, not what makes progress streaming itself work.
    argv = list(command_prefix)
    if argv[-1] != "--leica-worker":
        # Must sit right after the interpreter and before the script path
        # ("python -u script.py ...") -- appending it at the end instead
        # would make it leica_worker.py's first positional argument
        # (interrogator_root) rather than an interpreter flag.
        argv.insert(1, "-u")
    argv.extend([str(root), *args])
    try:
        process = subprocess.Popen(
            argv,
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            creationflags=_NO_WINDOW,
        )
    except OSError as exc:
        raise RuntimeError(f"Could not launch {interpreter}:\n{exc}") from exc
    on_process_started(process)

    final_payload: dict[str, Any] | None = None
    raw_lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        line = line.strip()
        if not line:
            continue
        raw_lines.append(line)
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue  # ignore stray non-JSON noise on stdout
        if "progress" in payload:
            info = payload["progress"]
            on_progress(str(info.get("message", "")), float(info.get("completed", 0)), float(info.get("total", 1)))
        else:
            final_payload = payload  # the results/error line, expected last

    returncode = process.wait()
    stderr = (process.stderr.read() if process.stderr else "").strip()[-2000:]
    if final_payload is None:
        raise RuntimeError(
            "Leica Interrogator's converter produced no result"
            + (f":\n\n{stderr}" if stderr else f" (exit code {returncode}).")
        )
    if "error" in final_payload:
        message = final_payload["error"]
        if not message:
            # The worker is only ever supposed to emit a real string here
            # (see leica_worker.py) -- an empty/None value means something
            # unanticipated happened. Surface everything captured rather
            # than a bare "None", since that's otherwise not actionable.
            message = (
                "Leica Interrogator reported a failure with no message "
                f"(exit code {returncode})."
                + (f"\n\nstderr:\n{stderr}" if stderr else "")
                + (f"\n\nLast output line(s):\n" + "\n".join(raw_lines[-5:]) if raw_lines else "")
            )
        raise RuntimeError(str(message))
    return final_payload


class LeicaExportDialog(tk.Toplevel):
    def __init__(
        self, app: "FLIMStudioApp", lif_path: Path, root: Path, command_prefix: list[str],
    ) -> None:
        super().__init__(app.root)
        self.app = app
        self.lif_path = lif_path
        self.interrogator_root = root
        self.command_prefix = command_prefix
        self.datasets: list[dict[str, Any]] = []
        self.busy = False
        self._process: subprocess.Popen | None = None
        self._cancelled = False

        self.title(f"Export from {lif_path.name}")
        self.geometry("760x560")
        self.configure(bg=BG)
        self.transient(app.root)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.output_dir_var = tk.StringVar(value=str(lif_path.parent / "exported"))
        self.bins_var = tk.StringVar(value="128")
        self.gzip_level_var = tk.StringVar(value="Fast")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.status_var = tk.StringVar(value="Scanning .lif for FLIM datasets…")

        self._build()
        threading.Thread(target=self._scan, daemon=True, name="leica-scan").start()

    def _build(self) -> None:
        ttk.Label(self, text=f"Datasets found in {self.lif_path.name}", style="PanelTitle.TLabel").pack(
            anchor="w", padx=14, pady=(12, 2)
        )
        ttk.Label(
            self,
            text="Pick which ones to export as .ics — exported files are added to the queue automatically.",
            style="Muted.TLabel",
            wraplength=720,
            justify="left",
        ).pack(anchor="w", padx=14, pady=(0, 8))

        self.listbox = tk.Listbox(
            self,
            selectmode=tk.EXTENDED,
            bg="#091321",
            fg="#dce7f5",
            selectbackground="#075985",
            selectforeground="white",
            font=("Segoe UI", 9),
            activestyle="none",
        )
        self.listbox.pack(fill=tk.BOTH, expand=True, padx=14)

        select_row = ttk.Frame(self)
        select_row.pack(fill=tk.X, padx=14, pady=(6, 6))
        ttk.Button(select_row, text="Select all", command=self._select_all).pack(side=tk.LEFT)
        ttk.Button(select_row, text="Select none", command=self._select_none).pack(side=tk.LEFT, padx=(6, 0))

        options_row = ttk.Frame(self)
        options_row.pack(fill=tk.X, padx=14, pady=(0, 6))
        ttk.Label(options_row, text="Output folder", style="Sidebar.TLabel").pack(side=tk.LEFT)
        ttk.Entry(options_row, textvariable=self.output_dir_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 6)
        )
        ttk.Button(options_row, text="Browse…", command=self._browse_output).pack(side=tk.LEFT)
        ttk.Label(options_row, text="Bins", style="Sidebar.TLabel").pack(side=tk.LEFT, padx=(12, 4))
        bins_combo = ttk.Combobox(
            options_row,
            textvariable=self.bins_var,
            values=["32", "64", "128", "256", "512", "1024", "2048", "4096"],
            state="readonly",
            width=6,
        )
        bins_combo.pack(side=tk.LEFT)
        _disable_combobox_wheel(bins_combo)
        ttk.Label(options_row, text="Compression level", style="Sidebar.TLabel").pack(side=tk.LEFT, padx=(12, 4))
        gzip_combo = ttk.Combobox(
            options_row,
            textvariable=self.gzip_level_var,
            values=list(GZIP_LEVEL_CHOICES),
            state="readonly",
            width=9,
        )
        gzip_combo.pack(side=tk.LEFT)
        _disable_combobox_wheel(gzip_combo)

        action_row = ttk.Frame(self)
        action_row.pack(fill=tk.X, padx=14, pady=(0, 6))
        self.export_button = ttk.Button(
            action_row, text="Export selected", command=self._run_export, style="Primary.TButton", state="disabled"
        )
        self.export_button.pack(side=tk.LEFT)
        ttk.Button(action_row, text="Close", command=self._on_close).pack(side=tk.RIGHT)

        self.progress_bar = ttk.Progressbar(self, variable=self.progress_var, maximum=1.0)
        self.progress_bar.pack(fill=tk.X, padx=14, pady=(2, 4))

        ttk.Label(self, textvariable=self.status_var, style="Muted.TLabel", wraplength=720, justify="left").pack(
            anchor="w", padx=14, pady=(0, 12)
        )

    def _browse_output(self) -> None:
        folder = filedialog.askdirectory(
            title="Choose where to save the exported .ics files", parent=self
        )
        if folder:
            self.output_dir_var.set(folder)

    def _select_all(self) -> None:
        self.listbox.selection_set(0, tk.END)

    def _select_none(self) -> None:
        self.listbox.selection_clear(0, tk.END)

    def _set_busy(self, value: bool) -> None:
        self.busy = value
        self.export_button.configure(state="disabled" if value else "normal")

    def _dispatch(self, callback: Callable[[], None]) -> None:
        """Run ``callback`` on the Tk thread, but only if this dialog is
        still around -- a background scan/export thread can finish (or
        error out) after the user already closed the window, and touching
        destroyed widgets from a queued ``after()`` callback raises."""

        def guarded() -> None:
            try:
                if self.winfo_exists():
                    callback()
            except tk.TclError:
                pass

        self.after(0, guarded)

    def _scan(self) -> None:
        try:
            payload = _run_worker(
                self.command_prefix, self.interrogator_root, ["scan", str(self.lif_path)], SCAN_TIMEOUT_S
            )
        except RuntimeError as exc:
            message = str(exc)  # see the matching comment in _run_export -- same reason
            self._dispatch(lambda: self._scan_failed(message))
            return
        self._dispatch(lambda: self._scan_finished(payload.get("datasets", [])))

    def _scan_finished(self, datasets: list[dict[str, Any]]) -> None:
        self.datasets = datasets
        if not datasets:
            self.status_var.set("No FLIM datasets were found in this file.")
            return
        for dataset in datasets:
            dims = f"{dataset['width']}×{dataset['height']}"
            if dataset.get("plane_count", 1) > 1:
                dims += f" × {dataset['plane_count']} planes"
            channel = dataset.get("detector_name") or dataset.get("channel_name") or ""
            photons = dataset.get("total_photons")
            photon_text = f" • {photons:,} photons" if isinstance(photons, int) else ""
            label = f"{dataset['display_name']} — {dims}{(' — ' + channel) if channel else ''}{photon_text}"
            self.listbox.insert(tk.END, label)
        self._select_all()
        self.export_button.configure(state="normal")
        self.status_var.set(f"{len(datasets)} dataset(s) found — {len(datasets)} selected by default.")

    def _scan_failed(self, message: str) -> None:
        self.status_var.set("Scan failed.")
        messagebox.showerror("Could not scan .lif", message, parent=self)

    def _run_export(self) -> None:
        if self.busy:
            return
        selection = self.listbox.curselection()
        if not selection:
            messagebox.showinfo("Nothing selected", "Select at least one dataset to export.", parent=self)
            return
        selected_keys = [self.datasets[index]["key"] for index in selection]
        output_dir = self.output_dir_var.get().strip()
        if not output_dir:
            messagebox.showinfo("Choose a folder", "Choose an output folder first.", parent=self)
            return
        try:
            bins = int(self.bins_var.get())
        except ValueError:
            bins = 128
        gzip_level = GZIP_LEVEL_CHOICES.get(self.gzip_level_var.get(), 1)
        self._cancelled = False
        self._set_busy(True)
        self.progress_var.set(0.0)
        self.status_var.set(f"Exporting {len(selected_keys)} dataset(s)…")

        def on_progress(message: str, completed: float, total: float) -> None:
            fraction = completed / total if total else 0.0
            self._dispatch(lambda: (self.progress_var.set(fraction), self.status_var.set(message)))

        def on_process_started(process: subprocess.Popen) -> None:
            self._process = process

        def worker() -> None:
            try:
                payload = _stream_export(
                    self.command_prefix,
                    self.interrogator_root,
                    ["export", str(self.lif_path), output_dir, json.dumps(selected_keys), str(bins), str(gzip_level)],
                    on_progress,
                    on_process_started,
                )
            except RuntimeError as exc:
                # Capture the message as a plain string *now*, synchronously,
                # inside the except block. `except X as exc:` variables are
                # auto-deleted by Python the instant this block ends -- a
                # lambda that instead closes over `exc` itself and only runs
                # later (once Tkinter's event loop gets to it, via
                # self._dispatch) hits a NameError at that point, because
                # `exc` no longer exists by then. A plain str already
                # computed here has no such lifetime problem.
                message = str(exc)
                self._process = None
                if self._cancelled:
                    self._dispatch(lambda: (self.status_var.set("Export cancelled."), self._set_busy(False)))
                else:
                    self._dispatch(lambda: self._export_failed(message))
                return
            self._process = None
            self._dispatch(lambda: self._export_finished(payload.get("results", [])))

        threading.Thread(target=worker, daemon=True, name="leica-export").start()

    def _on_close(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._cancelled = True
            self._process.terminate()
        self.destroy()

    def _export_finished(self, results: list[dict[str, Any]]) -> None:
        self._set_busy(False)
        exported = [Path(r["path"]) for r in results if r.get("status") == "exported" and r.get("path")]
        skipped = [r for r in results if r.get("status") == "skipped"]
        failed = [r for r in results if r.get("status") == "failed"]
        if exported:
            self.app._add_to_queue(exported)
        summary = f"{len(exported)} exported"
        if skipped:
            summary += f", {len(skipped)} skipped (already existed)"
        if failed:
            summary += f", {len(failed)} failed"
        self.status_var.set(f"{summary}. Exported files were added to the main queue.")
        if failed:
            detail = "\n".join(f"- {item.get('dataset', '?')}: {item.get('error', 'unknown error')}" for item in failed[:8])
            messagebox.showwarning("Some datasets failed to export", detail, parent=self)

    def _export_failed(self, message: str) -> None:
        self._set_busy(False)
        self.status_var.set("Export failed.")
        messagebox.showerror("Export failed", message, parent=self)

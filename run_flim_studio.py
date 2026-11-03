from __future__ import annotations

import argparse
import sys


def main() -> None:
    # Frozen (PyInstaller) builds have no standalone native_worker.py file
    # on disk to spawn -- this same executable is what native_backend.py
    # relaunches for every native fit instead, via this hidden flag (see
    # native_backend.py's _run_native_subprocess). This must stay the very
    # first thing main() does and stay import-light: no tkinter/GUI
    # imports before it, so relaunching for a fit is fast and never
    # touches a display. Running from source, this branch is simply never
    # taken -- fit_in_subprocess spawns native_worker.py directly instead.
    if len(sys.argv) > 1 and sys.argv[1] == "--native-worker":
        from flim_app.native_worker import main as worker_main

        raise SystemExit(worker_main(sys.argv[2:]))

    # Same reasoning as --native-worker above, for the bundled Leica .lif
    # reader (see leica_worker.py and leica_export_gui.resolve_leica_backend):
    # a packaged build has no standalone leica_worker.py file on disk for an
    # external interpreter to run when there's no separately-installed
    # Interrogator venv to run it under -- this exe answers to its own
    # --leica-worker flag instead, relaunched as a fresh, isolated process.
    if len(sys.argv) > 1 and sys.argv[1] == "--leica-worker":
        from flim_app.leica_worker import main as leica_worker_main

        raise SystemExit(leica_worker_main(sys.argv[2:]))

    import tkinter as tk

    from flim_app.gui import FLIMStudioApp

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--empty", action="store_true")
    args, _ = parser.parse_known_args()
    root = tk.Tk()
    FLIMStudioApp(root, start_with_demo=not args.empty)
    root.mainloop()


if __name__ == "__main__":
    main()

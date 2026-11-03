from __future__ import annotations

import subprocess
from pathlib import Path
import sys


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dll = root / "native" / "flimfit_labview.dll"
    worker = root / "tests" / "native_smoke_worker.py"
    completed = subprocess.run(
        [sys.executable, str(worker), str(dll)],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"Native subprocess failed with {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    if '"global"' not in completed.stdout or '"noise_3"' not in completed.stdout:
        raise AssertionError(f"Native smoke output was incomplete:\n{completed.stdout}")
    print(completed.stdout)
    print("PASS native subprocess smoke test")


if __name__ == "__main__":
    main()

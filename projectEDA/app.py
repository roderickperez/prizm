from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def main() -> int:
    root = Path(__file__).resolve().parent
    entry = root / "panel" / "ProjectEDA_main.py"
    cmd = [
        sys.executable,
        "-m",
        "panel",
        "serve",
        str(entry),
        "--autoreload",
        "--port",
        "5017",
    ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
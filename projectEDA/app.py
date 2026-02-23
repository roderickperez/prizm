from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--test", action="store_true", help="Run Panel UI in test mode without Petrel data")
    args, _ = parser.parse_known_args()

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

    if args.test:
        cmd.extend(["--args", "--test"])

    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
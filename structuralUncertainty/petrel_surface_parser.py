from __future__ import annotations

from pathlib import Path
import re

import numpy as np


def parse_petrel_surface_file(file_path: str | Path) -> dict[str, object]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Petrel surface file not found: {path}")

    header_lines: list[str] = []
    data_rows: list[tuple[float, float, float]] = []
    in_header = False
    after_header = False

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue

            if line == "BEGIN HEADER":
                in_header = True
                continue
            if line == "END HEADER":
                in_header = False
                after_header = True
                continue
            if in_header:
                header_lines.append(line)
                continue

            if not after_header:
                continue

            if line.startswith("ATTRIBUTES") or line.startswith("END ATTRIBUTES"):
                continue

            tokens = line.split()
            if len(tokens) < 3:
                continue
            try:
                x_val = float(tokens[0])
                y_val = float(tokens[1])
                z_val = float(tokens[2])
            except ValueError:
                continue
            data_rows.append((x_val, y_val, z_val))

    if not data_rows:
        raise ValueError(f"No numeric XYZ rows found in: {path}")

    arr = np.asarray(data_rows, dtype=np.float64)
    x = arr[:, 0]
    y = arr[:, 1]
    z = arr[:, 2]

    grid_ny = None
    grid_nx = None
    for line in header_lines:
        if line.lower().startswith("grid_size"):
            match = re.search(r"(\d+)\s*x\s*(\d+)", line, flags=re.IGNORECASE)
            if match:
                grid_ny = int(match.group(1))
                grid_nx = int(match.group(2))
                break

    x_unique = np.unique(x)
    y_unique = np.unique(y)
    x_unique.sort()
    y_unique.sort()

    z_grid = np.full((len(y_unique), len(x_unique)), np.nan, dtype=np.float32)
    x_index = {val: idx for idx, val in enumerate(x_unique.tolist())}
    y_index = {val: idx for idx, val in enumerate(y_unique.tolist())}

    for x_val, y_val, z_val in data_rows:
        z_grid[y_index[y_val], x_index[x_val]] = np.float32(z_val)

    if np.isnan(z_grid).any():
        valid = np.isfinite(z_grid)
        fill_value = float(np.nanmean(z_grid)) if np.any(valid) else 0.0
        z_grid = np.where(valid, z_grid, fill_value).astype(np.float32)

    metadata = {
        "grid_size_header": (grid_ny, grid_nx),
        "grid_size_detected": (len(y_unique), len(x_unique)),
        "x_min": float(np.min(x_unique)),
        "x_max": float(np.max(x_unique)),
        "y_min": float(np.min(y_unique)),
        "y_max": float(np.max(y_unique)),
        "z_min": float(np.min(z_grid)),
        "z_max": float(np.max(z_grid)),
    }

    return {
        "x": x_unique.astype(np.float32),
        "y": y_unique.astype(np.float32),
        "z": z_grid,
        "metadata": metadata,
    }


def to_panel_payload(parsed: dict[str, object]) -> dict[str, list]:
    return {
        "x": np.asarray(parsed["x"], dtype=float).tolist(),
        "y": np.asarray(parsed["y"], dtype=float).tolist(),
        "z": np.asarray(parsed["z"], dtype=float).tolist(),
    }

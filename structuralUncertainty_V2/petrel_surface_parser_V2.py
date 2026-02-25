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
    rc_rows: list[tuple[int, int, float, float, float]] = []
    in_header = False
    after_header = False

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue

            if line.startswith("#"):
                header_lines.append(line.lstrip("#").strip())
                after_header = True
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
                # Allow plain numeric files with no explicit header blocks.
                pass

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
            if len(tokens) >= 5:
                try:
                    col_idx = int(float(tokens[3]))
                    row_idx = int(float(tokens[4]))
                    rc_rows.append((row_idx, col_idx, x_val, y_val, z_val))
                except Exception:
                    pass

    if not data_rows:
        raise ValueError(f"No numeric XYZ rows found in: {path}")

    arr = np.asarray(data_rows, dtype=np.float64)
    x = arr[:, 0]
    y = arr[:, 1]

    grid_ny = None
    grid_nx = None
    for line in header_lines:
        if line.lower().startswith("grid_size"):
            match = re.search(r"(\d+)\s*x\s*(\d+)", line, flags=re.IGNORECASE)
            if match:
                grid_ny = int(match.group(1))
                grid_nx = int(match.group(2))
                break

    use_row_col = False
    if rc_rows:
        rc = np.asarray(rc_rows, dtype=float)
        row_vals = rc[:, 0].astype(int)
        col_vals = rc[:, 1].astype(int)
        unique_rows = np.unique(row_vals)
        unique_cols = np.unique(col_vals)
        if len(unique_rows) > 1 and len(unique_cols) > 1:
            # Prefer row/column reconstruction when available (EarthVision/Petrel exports)
            use_row_col = True

    if use_row_col:
        rc = np.asarray(rc_rows, dtype=float)
        row_vals = rc[:, 0].astype(int)
        col_vals = rc[:, 1].astype(int)
        unique_rows = np.unique(row_vals)
        unique_cols = np.unique(col_vals)
        row_to_i = {r: i for i, r in enumerate(sorted(unique_rows.tolist()))}
        col_to_j = {c: j for j, c in enumerate(sorted(unique_cols.tolist()))}

        z_grid = np.full((len(unique_rows), len(unique_cols)), np.nan, dtype=np.float32)
        x_grid = np.full((len(unique_rows), len(unique_cols)), np.nan, dtype=np.float64)
        y_grid = np.full((len(unique_rows), len(unique_cols)), np.nan, dtype=np.float64)

        for row_idx, col_idx, x_val, y_val, z_val in rc_rows:
            i = row_to_i[int(row_idx)]
            j = col_to_j[int(col_idx)]
            z_grid[i, j] = np.float32(z_val)
            x_grid[i, j] = float(x_val)
            y_grid[i, j] = float(y_val)

        x_unique = np.nanmean(x_grid, axis=0)
        y_unique = np.nanmean(y_grid, axis=1)
        x_unique = np.where(np.isfinite(x_unique), x_unique, np.nan)
        y_unique = np.where(np.isfinite(y_unique), y_unique, np.nan)
        # Fallback if some columns/rows were sparse
        if np.isnan(x_unique).any():
            finite_x = np.nanmean(x_grid)
            x_unique = np.nan_to_num(x_unique, nan=float(finite_x))
        if np.isnan(y_unique).any():
            finite_y = np.nanmean(y_grid)
            y_unique = np.nan_to_num(y_unique, nan=float(finite_y))
        x_unique = np.asarray(x_unique, dtype=np.float32)
        y_unique = np.asarray(y_unique, dtype=np.float32)
    else:
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

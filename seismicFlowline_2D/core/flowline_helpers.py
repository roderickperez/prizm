import numpy as np


def line_shape_feature(line: np.ndarray, n_samples: int = 64) -> np.ndarray:
    if line.shape[0] < 2:
        return np.zeros((n_samples + 2,), dtype=np.float32)

    sort_idx = np.argsort(line[:, 0])
    x = line[sort_idx, 0]
    y = line[sort_idx, 1]
    unique_x, unique_idx = np.unique(x, return_index=True)
    unique_y = y[unique_idx]

    if unique_x.shape[0] < 2:
        return np.zeros((n_samples + 2,), dtype=np.float32)

    x_grid = np.linspace(float(unique_x[0]), float(unique_x[-1]), n_samples)
    y_interp = np.interp(x_grid, unique_x, unique_y).astype(np.float32)
    y_mean = float(np.mean(y_interp))
    y_std = float(np.std(y_interp))
    y_norm = (y_interp - y_mean) / (y_std + 1e-6)
    x_span = float(unique_x[-1] - unique_x[0])
    return np.concatenate([y_norm, np.array([y_mean, x_span], dtype=np.float32)], axis=0)


def extend_line_to_full_width(line: np.ndarray, width: int, height: int) -> np.ndarray:
    if line.shape[0] < 2:
        return line.astype(np.float32)

    sort_idx = np.argsort(line[:, 0])
    x_vals = line[sort_idx, 0]
    y_vals = line[sort_idx, 1]
    unique_x, unique_idx = np.unique(x_vals, return_index=True)
    unique_y = y_vals[unique_idx]

    if unique_x.shape[0] < 2:
        return line.astype(np.float32)

    x_full = np.arange(0, width, dtype=np.float32)
    y_full = np.interp(x_full, unique_x, unique_y, left=unique_y[0], right=unique_y[-1]).astype(np.float32)
    y_full = np.clip(y_full, 0, max(0, height - 1))
    return np.column_stack([x_full, y_full]).astype(np.float32)


def build_segment_arrays(lines: list[np.ndarray], values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if not lines or values.size == 0:
        return None

    seg_x0: list[np.ndarray] = []
    seg_y0: list[np.ndarray] = []
    seg_x1: list[np.ndarray] = []
    seg_y1: list[np.ndarray] = []
    seg_v: list[np.ndarray] = []

    for line, value in zip(lines, values):
        if line.shape[0] < 2:
            continue
        seg_x0.append(line[:-1, 0])
        seg_y0.append(line[:-1, 1])
        seg_x1.append(line[1:, 0])
        seg_y1.append(line[1:, 1])
        seg_v.append(np.full(line.shape[0] - 1, float(value), dtype=np.float32))

    if not seg_x0:
        return None

    return (
        np.concatenate(seg_x0),
        np.concatenate(seg_y0),
        np.concatenate(seg_x1),
        np.concatenate(seg_y1),
        np.concatenate(seg_v),
    )

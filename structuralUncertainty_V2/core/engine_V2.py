from __future__ import annotations

from functools import lru_cache
from typing import Callable

import numpy as np
from scipy import fft as sp_fft
from scipy.ndimage import gaussian_filter, label, map_coordinates

try:
    import torch
except Exception:
    torch = None

ProgressCb = Callable[[int, int], None] | None


def _safe_workers() -> int:
    try:
        import os

        raw = os.getenv("STRUCTURAL_FFT_WORKERS", "1")
        return max(1, int(raw))
    except Exception:
        return 1


def build_surfaces(
    surface_mode: str,
    x: np.ndarray,
    y: np.ndarray,
    major_sigma: float = 2600.0,
    minor_sigma: float = 1200.0,
    rotation_deg: float = 30.0,
    twt_base_ms: float = -1850.0,
    twt_amp_ms: float = 260.0,
    vel_base: float = 3000.0,
    vel_amp: float = 120.0,
) -> tuple[np.ndarray, np.ndarray]:
    x_grid, y_grid = np.meshgrid(x, y)
    x0 = float(np.mean(x))
    y0 = float(np.mean(y))

    if surface_mode == "Elongated / Ellipsoidal":
        theta = np.deg2rad(rotation_deg)
        xr = (x_grid - x0) * np.cos(theta) + (y_grid - y0) * np.sin(theta)
        yr = -(x_grid - x0) * np.sin(theta) + (y_grid - y0) * np.cos(theta)

        twt_ms = twt_base_ms + twt_amp_ms * np.exp(-((xr**2) / (2 * max(major_sigma, 1.0) ** 2) + (yr**2) / (2 * max(minor_sigma, 1.0) ** 2)))

        vel = vel_base + vel_amp * np.exp(-(((x_grid - x0) ** 2) / (2 * (major_sigma * 1.2) ** 2) + ((y_grid - y0) ** 2) / (2 * (minor_sigma * 1.2) ** 2)))
        return twt_ms.astype(np.float32), vel.astype(np.float32)

    twt_ms = twt_base_ms + twt_amp_ms * np.exp(-(((x_grid - x0) ** 2 + (y_grid - y0) ** 2) / (2 * 2000.0**2)))
    vel = vel_base + vel_amp * np.exp(-(((x_grid - (x0 + 1500)) ** 2 + (y_grid - (y0 - 800)) ** 2) / (2 * 3500.0**2)))
    return twt_ms.astype(np.float32), vel.astype(np.float32)


@lru_cache(maxsize=24)
def _covariance_grid(shape: tuple[int, int], dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = shape
    y = np.fft.fftfreq(ny, d=1.0 / ny) * dy * ny
    x = np.fft.fftfreq(nx, d=1.0 / nx) * dx * nx
    return np.meshgrid(x, y)


@lru_cache(maxsize=64)
def _sqrt_spectrum_numpy(
    shape: tuple[int, int],
    dx: float,
    dy: float,
    model: str,
    range_val: float,
    nugget: float,
    sill: float,
) -> np.ndarray:
    x_grid, y_grid = _covariance_grid(shape, dx, dy)
    h = np.sqrt(x_grid**2 + y_grid**2)

    safe_range = max(float(range_val), 1e-6)
    safe_sill = max(float(sill), 1e-6)

    if model == "Gaussian":
        covariance = safe_sill * np.exp(-3.0 * (h / safe_range) ** 2)
    elif model == "Exponential":
        covariance = safe_sill * np.exp(-3.0 * h / safe_range)
    else:
        ratio = h / safe_range
        covariance = np.where(h <= safe_range, safe_sill * (1.0 - (1.5 * ratio - 0.5 * ratio**3)), 0.0)

    spectrum = np.abs(sp_fft.fft2(covariance, workers=_safe_workers()))
    return np.sqrt(spectrum).astype(np.float32, copy=False)


def _generate_field_numpy(
    shape: tuple[int, int],
    dx: float,
    dy: float,
    model: str,
    range_val: float,
    nugget: float,
    sill: float,
    rng: np.random.Generator,
) -> np.ndarray:
    safe_nugget = max(float(nugget), 0.0)

    sqrt_spectrum = _sqrt_spectrum_numpy(
        shape=shape,
        dx=dx,
        dy=dy,
        model=model,
        range_val=float(range_val),
        nugget=float(nugget),
        sill=float(sill),
    )
    white_noise = rng.normal(0.0, 1.0, shape)
    field_fft = sp_fft.fft2(white_noise, workers=_safe_workers())
    spatial_field = np.real(sp_fft.ifft2(field_fft * sqrt_spectrum, workers=_safe_workers()))

    spatial_std = float(np.std(spatial_field))
    if spatial_std > 0:
        spatial_field = spatial_field / spatial_std

    if safe_nugget > 0:
        spatial_field += rng.normal(0.0, np.sqrt(safe_nugget), shape)

    total_std = float(np.std(spatial_field))
    if total_std > 0:
        spatial_field = spatial_field / total_std

    return spatial_field.astype(np.float32)


def _generate_field_torch(
    shape: tuple[int, int],
    dx: float,
    dy: float,
    model: str,
    range_val: float,
    nugget: float,
    sill: float,
) -> np.ndarray:
    if torch is None:
        raise RuntimeError("torch unavailable")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    safe_range = max(float(range_val), 1e-6)
    safe_sill = max(float(sill), 1e-6)
    safe_nugget = max(float(nugget), 0.0)

    sqrt_spectrum_np = _sqrt_spectrum_numpy(
        shape=shape,
        dx=dx,
        dy=dy,
        model=model,
        range_val=safe_range,
        nugget=safe_nugget,
        sill=safe_sill,
    )
    sqrt_spectrum = torch.tensor(sqrt_spectrum_np, device=device, dtype=torch.float32)
    white_noise = torch.randn(shape, device=device, dtype=torch.float32)
    spatial_field = torch.real(torch.fft.ifft2(torch.fft.fft2(white_noise) * sqrt_spectrum))

    spatial_std = float(torch.std(spatial_field).item())
    if spatial_std > 0:
        spatial_field = spatial_field / spatial_std

    if safe_nugget > 0:
        spatial_field = spatial_field + torch.randn(shape, device=device, dtype=torch.float32) * np.sqrt(safe_nugget)

    total_std = float(torch.std(spatial_field).item())
    if total_std > 0:
        spatial_field = spatial_field / total_std

    return spatial_field.detach().cpu().numpy().astype(np.float32)


def _compute_covariance_matrix(h: np.ndarray, model: str, range_val: float, sill: float, nugget: float) -> np.ndarray:
    safe_range = max(float(range_val), 1e-6)
    safe_sill = max(float(sill), 1e-6)
    total_var = safe_sill + max(float(nugget), 0.0)

    if model == "Gaussian":
        cov = safe_sill * np.exp(-3.0 * (h / safe_range) ** 2)
    elif model == "Exponential":
        cov = safe_sill * np.exp(-3.0 * h / safe_range)
    else:
        ratio = h / safe_range
        cov = np.where(h <= safe_range, safe_sill * (1.0 - (1.5 * ratio - 0.5 * ratio**3)), 0.0)

    cov = np.where(h < 1e-6, total_var, cov)
    return cov / total_var


def generate_velocity_depth_stacks(
    twt_ms: np.ndarray,
    base_velocity: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    n_maps: int,
    model: str,
    range_val: float,
    sill: float,
    nugget: float,
    smooth_sigma: float,
    velocity_std: float,
    dx: float = 100.0,
    dy: float = 100.0,
    seed: int = 42,
    use_torch: bool = False,
    progress_cb: ProgressCb = None,
    well_x: list[float] | None = None,
    well_y: list[float] | None = None,
    well_velocity: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    stack = np.empty((n_maps, *twt_ms.shape), dtype=np.float32)

    conditioning_weights = None
    well_indices = None
    s_target = None

    if well_x and well_y and well_velocity and len(well_x) > 0:
        n_wells = len(well_x)
        w_x = np.array(well_x, dtype=float)
        w_y = np.array(well_y, dtype=float)
        w_v = np.array(well_velocity, dtype=float)

        ww_dist = np.sqrt((w_x[:, None] - w_x[None, :]) ** 2 + (w_y[:, None] - w_y[None, :]) ** 2)
        C_ww = _compute_covariance_matrix(ww_dist, model, range_val, sill, nugget)

        X, Y = np.meshgrid(x_coords, y_coords)
        C_xw = np.zeros((n_wells, Y.shape[0], X.shape[1]), dtype=float)
        well_indices = []

        for w in range(n_wells):
            h_xw = np.sqrt((X - w_x[w]) ** 2 + (Y - w_y[w]) ** 2)
            C_xw[w] = _compute_covariance_matrix(h_xw, model, range_val, sill, nugget)

            w_idx_x = int(np.argmin(np.abs(x_coords - w_x[w])))
            w_idx_y = int(np.argmin(np.abs(y_coords - w_y[w])))
            well_indices.append((w_idx_y, w_idx_x))

        C_xw_flat = C_xw.reshape(n_wells, -1)
        try:
            weights_flat = np.linalg.solve(C_ww, C_xw_flat)
        except np.linalg.LinAlgError:
            C_ww = C_ww + np.eye(n_wells) * 1e-4
            weights_flat = np.linalg.solve(C_ww, C_xw_flat)

        conditioning_weights = weights_flat.reshape(n_wells, Y.shape[0], X.shape[1])

        s_target = np.zeros(n_wells, dtype=float)
        for w in range(n_wells):
            w_idx_y, w_idx_x = well_indices[w]
            v_base_well = float(base_velocity[w_idx_y, w_idx_x])
            s_target[w] = (w_v[w] - v_base_well) / velocity_std if velocity_std > 0 else 0.0

    for idx in range(n_maps):
        if use_torch and torch is not None:
            field = _generate_field_torch(twt_ms.shape, dx, dy, model, range_val, nugget, sill)
        else:
            field = _generate_field_numpy(twt_ms.shape, dx, dy, model, range_val, nugget, sill, rng)

        if smooth_sigma > 0:
            filtered = gaussian_filter(field, sigma=smooth_sigma)
            std_val = float(np.std(filtered))
            if std_val > 0:
                filtered = filtered / std_val
            field = filtered.astype(np.float32)

        if conditioning_weights is not None and well_indices is not None and s_target is not None:
            s_sim = np.zeros(len(well_indices), dtype=float)
            for w, (w_idx_y, w_idx_x) in enumerate(well_indices):
                s_sim[w] = float(field[w_idx_y, w_idx_x])

            diff = s_target - s_sim
            conditioning_field = np.tensordot(diff, conditioning_weights, axes=([0], [0]))
            field = field + conditioning_field.astype(np.float32)

        stack[idx] = field
        if progress_cb is not None:
            progress_cb(idx + 1, n_maps)

    velocity_stack = base_velocity[None, :, :] + stack * velocity_std
    velocity_stack = np.maximum(velocity_stack, 1200.0)

    avg_velocity_map = np.mean(velocity_stack, axis=0)
    final_depth_map = (twt_ms * avg_velocity_map) / 2000.0
    depth_stack = (twt_ms[None, :, :] * velocity_stack) / 2000.0

    return velocity_stack, depth_stack, avg_velocity_map, final_depth_map


def get_trap_and_spill(depth_map: np.ndarray, step: float) -> tuple[float, np.ndarray, float]:
    step_val = max(float(step), 0.1)
    is_negative_domain = bool(np.nanmax(depth_map) <= 0.0)

    if is_negative_domain:
        crest_idx = int(np.nanargmax(depth_map))
    else:
        crest_idx = int(np.nanargmin(depth_map))

    crest_y, crest_x = np.unravel_index(crest_idx, depth_map.shape)
    crest_depth = float(depth_map[crest_y, crest_x])

    previous_closed_mask = np.zeros_like(depth_map, dtype=bool)
    spill_depth = crest_depth

    if is_negative_domain:
        current_level = crest_depth - step_val
        min_depth = float(np.nanmin(depth_map))
        cond = lambda level: level >= min_depth
        flood_fn = lambda arr, level: arr >= level
        update_fn = lambda level: level - step_val
    else:
        current_level = crest_depth + step_val
        max_depth = float(np.nanmax(depth_map))
        cond = lambda level: level <= max_depth
        flood_fn = lambda arr, level: arr <= level
        update_fn = lambda level: level + step_val

    while cond(current_level):
        flooded = flood_fn(depth_map, current_level)
        labeled, _ = label(flooded)
        crest_label = labeled[crest_y, crest_x]

        if crest_label == 0:
            break

        crest_polygon = labeled == crest_label
        touches_edge = (
            np.any(crest_polygon[0, :])
            or np.any(crest_polygon[-1, :])
            or np.any(crest_polygon[:, 0])
            or np.any(crest_polygon[:, -1])
        )

        if touches_edge:
            break

        previous_closed_mask = crest_polygon
        spill_depth = current_level
        current_level = update_fn(current_level)

    return float(spill_depth), previous_closed_mask, crest_depth


def compute_trap_statistics(
    depth_stack: np.ndarray,
    contour_step: float,
    thickness_mean: float,
    thickness_std: float,
    cell_area: float = 10000.0,
    progress_cb: ProgressCb = None,
) -> dict[str, np.ndarray]:
    n_maps = depth_stack.shape[0]

    rng = np.random.default_rng(42)

    trap_masks = np.zeros_like(depth_stack, dtype=bool)
    spill_depths = np.zeros(n_maps, dtype=float)
    crest_depths = np.zeros(n_maps, dtype=float)
    areas = np.zeros(n_maps, dtype=float)
    thickness_total = np.zeros(n_maps, dtype=float)
    grv = np.zeros(n_maps, dtype=float)

    for map_idx in range(n_maps):
        depth_map = depth_stack[map_idx]
        spill_depth, trap_mask, crest_depth = get_trap_and_spill(depth_map, contour_step)
        is_negative_domain = bool(np.nanmax(depth_map) <= 0.0)

        spill_depths[map_idx] = spill_depth
        crest_depths[map_idx] = crest_depth
        trap_masks[map_idx] = trap_mask

        if not np.any(trap_mask):
            if progress_cb is not None:
                progress_cb(map_idx + 1, n_maps)
            continue

        area = float(np.sum(trap_mask) * cell_area)
        areas[map_idx] = area
        thickness_total[map_idx] = max(0.0, (crest_depth - spill_depth) if is_negative_domain else (spill_depth - crest_depth))

        if is_negative_domain:
            height_to_spill = depth_map[trap_mask] - spill_depth
        else:
            height_to_spill = spill_depth - depth_map[trap_mask]

        height_to_spill = np.clip(height_to_spill, a_min=0.0, a_max=None)
        total_volume = float(np.sum(height_to_spill) * cell_area)

        sampled_thickness = max(10.0, float(rng.normal(thickness_mean, thickness_std)))
        relative_depth = height_to_spill - sampled_thickness
        base_volume = float(np.sum(np.clip(relative_depth, a_min=0.0, a_max=None)) * cell_area)

        grv[map_idx] = max(0.0, total_volume - base_volume)
        if progress_cb is not None:
            progress_cb(map_idx + 1, n_maps)

    return {
        "trap_masks": trap_masks,
        "spill_depths": spill_depths,
        "crest_depths": crest_depths,
        "areas": areas,
        "thickness_total": thickness_total,
        "grv": grv,
    }


def line_through_center(x_values: np.ndarray, y_values: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
    y_min, y_max = float(np.min(y_values)), float(np.max(y_values))
    center_x, center_y = float(np.mean(x_values)), float(np.mean(y_values))

    theta = np.deg2rad(angle_deg)
    length = np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)
    dx = np.cos(theta) * length / 2.0
    dy = np.sin(theta) * length / 2.0

    xs = np.array([center_x - dx, center_x + dx], dtype=float)
    ys = np.array([center_y - dy, center_y + dy], dtype=float)
    return xs, ys


def _extract_line_values(data_2d: np.ndarray, x_values: np.ndarray, y_values: np.ndarray, angle_deg: float, n_samples: int = 260) -> tuple[np.ndarray, np.ndarray]:
    center_x, center_y = float(np.mean(x_values)), float(np.mean(y_values))
    x_span = float(np.max(x_values) - np.min(x_values))
    y_span = float(np.max(y_values) - np.min(y_values))
    half_len = 0.5 * np.sqrt(x_span**2 + y_span**2)

    # Structural section is perpendicular to displayed dashed cut line
    phi = np.deg2rad((angle_deg + 90.0) % 360.0)
    t = np.linspace(-half_len, half_len, n_samples)
    x_line = center_x + t * np.cos(phi)
    y_line = center_y + t * np.sin(phi)

    x_idx = np.interp(x_line, x_values, np.arange(len(x_values)))
    y_idx = np.interp(y_line, y_values, np.arange(len(y_values)))

    values = map_coordinates(data_2d, [y_idx, x_idx], order=1, mode="nearest")
    distance = t + half_len
    return distance, values


def extract_section_stack(depth_stack: np.ndarray, final_depth_map: np.ndarray, x_values: np.ndarray, y_values: np.ndarray, angle_deg: float, n_samples: int = 260) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distance, final_section = _extract_line_values(final_depth_map, x_values, y_values, angle_deg, n_samples=n_samples)

    center_x, center_y = float(np.mean(x_values)), float(np.mean(y_values))
    x_span = float(np.max(x_values) - np.min(x_values))
    y_span = float(np.max(y_values) - np.min(y_values))
    half_len = 0.5 * np.sqrt(x_span**2 + y_span**2)

    phi = np.deg2rad((angle_deg + 90.0) % 360.0)
    t = np.linspace(-half_len, half_len, n_samples)
    x_line = center_x + t * np.cos(phi)
    y_line = center_y + t * np.sin(phi)

    x_idx = np.interp(x_line, x_values, np.arange(len(x_values)))
    y_idx = np.interp(y_line, y_values, np.arange(len(y_values)))

    n_maps = depth_stack.shape[0]
    map_idx = np.broadcast_to(np.arange(n_maps, dtype=float)[:, None], (n_maps, n_samples))
    y_idx_2d = np.broadcast_to(y_idx, (n_maps, n_samples))
    x_idx_2d = np.broadcast_to(x_idx, (n_maps, n_samples))

    section_stack = map_coordinates(
        depth_stack,
        [map_idx, y_idx_2d, x_idx_2d],
        order=1,
        mode="nearest",
    ).astype(np.float32, copy=False)

    return distance, final_section, section_stack

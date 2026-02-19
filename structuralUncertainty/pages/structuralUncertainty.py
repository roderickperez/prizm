import ctypes
import logging
import os
import platform
import sys
import time
from functools import lru_cache
from pathlib import Path

import holoviews as hv
import hvplot.pandas
import hvplot.xarray
import numpy as np
import pandas as pd
import panel as pn
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from bokeh.models import HoverTool
from scipy.ndimage import gaussian_filter, label


# ==============================================================================
#  WINDOWS DLL COMPATIBILITY FIX
# ==============================================================================
def apply_dll_fix() -> None:
    if platform.system() != "Windows":
        return

    try:
        current_venv = sys.prefix
        torch_lib_path = os.path.join(current_venv, "Lib", "site-packages", "torch", "lib")
        user_lib_path = os.path.expanduser(r"~\py_pkgs\torch\lib")

        for lib_path in [torch_lib_path, user_lib_path]:
            if not os.path.exists(lib_path):
                continue

            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(lib_path)
                except Exception:
                    pass

            for dll in ["libiomp5md.dll", "c10.dll", "torch_python.dll"]:
                dll_file = os.path.join(lib_path, dll)
                if os.path.exists(dll_file):
                    try:
                        ctypes.CDLL(dll_file)
                    except Exception:
                        pass
    except Exception:
        pass


apply_dll_fix()


# ==============================================================================
#  PATH RESOLUTION & THEME IMPORTS
# ==============================================================================
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from shared.ui.omv_theme import (
        BLUE_OMV_COLOR,
        DARK_BLUE_OMV_COLOR,
        NEON_OMV_COLOR,
        docs_button_html,
        get_content_text_color,
        get_dark_select_stylesheets,
        get_extension_raw_css,
        get_main_outer_background,
        get_neon_button_stylesheets,
        get_slider_stylesheets,
        is_dark_mode_from_state,
    )
except ImportError:
    BLUE_OMV_COLOR = "#005A9B"
    DARK_BLUE_OMV_COLOR = "#003056"
    NEON_OMV_COLOR = "#00E5FF"

    def docs_button_html(_url: str) -> str:
        return ""

    def get_content_text_color(dark: bool) -> str:
        return "white" if dark else "black"

    def get_dark_select_stylesheets(_dark: bool) -> list[str]:
        return []

    def get_extension_raw_css(_dark: bool) -> list[str]:
        return []

    def get_main_outer_background(dark: bool) -> str:
        return "#121212" if dark else "#F4F4F4"

    def get_neon_button_stylesheets() -> list[str]:
        return []

    def get_slider_stylesheets() -> list[str]:
        return []

    def is_dark_mode_from_state() -> bool:
        return False


# ==============================================================================
#  APP CONFIG
# ==============================================================================
APP_TITLE = "Structural Uncertainty Evaluation"
DOCUMENTATION_URL = "https://example.com/docs"

ASSETS_DIR = ROOT_DIR / "images" / "OMV_brandLogoPack" / "OMV_brandLogoPack"
LOGO_PATH = ASSETS_DIR / "OMV_logo_Neon_Small.png"
FAVICON_PATH = ASSETS_DIR / "OMV_logo_Blue_Small.png"

APP_DIR = Path(__file__).resolve().parent
LOG_DIR = APP_DIR / "logs"

is_dark_mode = is_dark_mode_from_state()


# ==============================================================================
#  APP LOGGER (KEEP LAST 10 SESSIONS)
# ==============================================================================
def setup_structural_logger(log_dir: Path, logger_name: str = "structural_uncertainty_app") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"structural_uncertainty_{time.strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    existing_logs = sorted(log_dir.glob("structural_uncertainty_*.log"), key=lambda p: p.stat().st_mtime)
    while len(existing_logs) > 10:
        oldest = existing_logs.pop(0)
        try:
            oldest.unlink()
        except OSError:
            pass

    logger.info("Structural Uncertainty logger initialized")
    logger.info("Log file: %s", log_file)
    return logger


logger = setup_structural_logger(LOG_DIR)


# ==============================================================================
#  PANEL EXTENSION & CUSTOM CSS
# ==============================================================================
custom_css = f"""
.dashboard-grid {{
    display: grid !important;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) minmax(0, 1fr) !important;
    grid-template-rows: minmax(0, 1fr) minmax(0, 1fr) !important;
    grid-template-areas:
        \"map section hists\"
        \"realization iso hists\";
    gap: 10px !important;
    height: calc(100vh - 150px) !important;
    overflow: hidden !important;
}}
.plot-box {{
    background-color: {'#1E1E1E' if is_dark_mode else 'white'};
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 10px;
    min-height: 0;
    overflow: hidden;
    display: flex;
    flex-direction: column;
}}
.plot-wrapper {{
    flex: 1;
    min-height: 0;
    overflow: hidden;
}}
.plot-wrapper .bk,
.plot-wrapper .bk-root,
.plot-wrapper .bk-panel-models-layout-Column,
.plot-wrapper .bk-panel-models-layout-Row,
.plot-wrapper .bk-panel-models-layout-Tabs,
.plot-wrapper .bk-panel-models-markup-HTML,
.plot-wrapper .bk-panel-models-markup-Markdown,
.plot-wrapper .bk-Column,
.plot-wrapper .bk-Tabs {{
    height: 100% !important;
    min-height: 0 !important;
}}
.hist-container {{
    display: flex;
    flex-direction: column;
    height: 100%;
    gap: 6px;
}}
.hist-container .bk,
.hist-container .bk-root,
.hist-container .bk-Column,
.hist-container .bk-panel-models-layout-Column {{
    min-height: 0 !important;
}}
.plot-title {{
    color: {DARK_BLUE_OMV_COLOR};
    margin-top: 0;
    margin-bottom: 6px;
    font-weight: bold;
    font-size: 1.05em;
}}
.omv-run-btn,
.omv-run-btn.bk-btn,
button.omv-run-btn,
.omv-run-btn .bk-btn,
.omv-run-btn button.bk-btn {{
    background: {NEON_OMV_COLOR} !important;
    color: {DARK_BLUE_OMV_COLOR} !important;
    border-color: {NEON_OMV_COLOR} !important;
    font-weight: 600 !important;
}}
"""

pn.extension("tabulator", raw_css=get_extension_raw_css(is_dark_mode) + [custom_css])


# ==============================================================================
#  THEME VARS & STYLESHEETS
# ==============================================================================
section_header_background = NEON_OMV_COLOR
section_body_background = DARK_BLUE_OMV_COLOR if is_dark_mode else "white"
section_text_color = get_content_text_color(is_dark_mode)
select_stylesheets = get_dark_select_stylesheets(is_dark_mode)
slider_stylesheets = get_slider_stylesheets()


# ==============================================================================
#  SYNTHETIC INPUTS: TWT + DETERMINISTIC VELOCITY
# ==============================================================================
def create_synthetic_twt_and_velocity() -> tuple[xr.DataArray, xr.DataArray]:
    x = np.linspace(0, 10000, 100)
    y = np.linspace(0, 10000, 100)
    x_grid, y_grid = np.meshgrid(x, y)

    twt_ms = 4000 - 700 * np.exp(-(((x_grid - 5000) ** 2 + (y_grid - 5000) ** 2) / (2 * 2000 ** 2)))
    vel_det = 3000 + 120 * np.exp(-(((x_grid - 6500) ** 2 + (y_grid - 4200) ** 2) / (2 * 3500 ** 2)))

    twt = xr.DataArray(twt_ms, coords=[y, x], dims=["y", "x"], name="TWT")
    velocity = xr.DataArray(vel_det, coords=[y, x], dims=["y", "x"], name="Velocity")
    return twt, velocity


base_twt, base_velocity = create_synthetic_twt_and_velocity()
y_section_value = 5000
y_section_idx = int(np.abs(base_twt.y.values - y_section_value).argmin())
deterministic_depth = (base_twt.values * base_velocity.values) / 2000.0


# ==============================================================================
#  GEOSTATISTICAL SIMULATION CORE
# ==============================================================================
@lru_cache(maxsize=24)
def _covariance_grid(shape: tuple[int, int], dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = shape
    y = np.fft.fftfreq(ny, d=1.0 / ny) * dy * ny
    x = np.fft.fftfreq(nx, d=1.0 / nx) * dx * nx
    return np.meshgrid(x, y)


def generate_geostatistical_realization(
    shape: tuple[int, int],
    dx: float,
    dy: float,
    model: str,
    range_val: float,
    nugget: float,
    sill: float,
    rng: np.random.Generator,
) -> np.ndarray:
    x_grid, y_grid = _covariance_grid(shape, dx, dy)
    h = np.sqrt(x_grid**2 + y_grid**2)

    safe_range = max(float(range_val), 1e-6)
    safe_sill = max(float(sill), 1e-6)
    safe_nugget = max(float(nugget), 0.0)

    if model == "Gaussian":
        covariance = safe_sill * np.exp(-3.0 * (h / safe_range) ** 2)
    elif model == "Exponential":
        covariance = safe_sill * np.exp(-3.0 * h / safe_range)
    else:
        ratio = h / safe_range
        covariance = np.where(h <= safe_range, safe_sill * (1.0 - (1.5 * ratio - 0.5 * ratio**3)), 0.0)

    spectrum = np.abs(np.fft.fft2(covariance))
    white_noise = rng.normal(0.0, 1.0, shape)
    field_fft = np.fft.fft2(white_noise)
    spatial_field = np.real(np.fft.ifft2(field_fft * np.sqrt(spectrum)))

    spatial_std = float(np.std(spatial_field))
    if spatial_std > 0:
        spatial_field = spatial_field / spatial_std

    if safe_nugget > 0:
        spatial_field += rng.normal(0.0, np.sqrt(safe_nugget), shape)

    total_std = float(np.std(spatial_field))
    if total_std > 0:
        spatial_field = spatial_field / total_std

    return spatial_field.astype(np.float32)


@lru_cache(maxsize=6)
def _normalized_realization_stack(
    n_maps: int,
    model: str,
    range_val: float,
    sill: float,
    nugget: float,
    smooth_sigma: float,
) -> np.ndarray:
    logger.info(
        "Generating normalized realizations | n_maps=%s model=%s range=%.1f sill=%.3f nugget=%.3f smooth=%.2f",
        n_maps,
        model,
        range_val,
        sill,
        nugget,
        smooth_sigma,
    )

    rng = np.random.default_rng(42)
    stack = np.empty((n_maps, *base_twt.shape), dtype=np.float32)

    for map_idx in range(n_maps):
        field = generate_geostatistical_realization(
            shape=base_twt.shape,
            dx=100.0,
            dy=100.0,
            model=model,
            range_val=range_val,
            nugget=nugget,
            sill=sill,
            rng=rng,
        )

        if smooth_sigma > 0:
            filtered = gaussian_filter(field, sigma=smooth_sigma)
            filtered_std = float(np.std(filtered))
            if filtered_std > 0:
                filtered = filtered / filtered_std
            field = filtered.astype(np.float32)

        stack[map_idx] = field

    return stack


def get_velocity_and_depth_stacks() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    normalized = _normalized_realization_stack(
        sld_n_maps.value,
        sld_slope.value,
        round(float(sld_range.value), 3),
        round(float(sld_sill.value), 6),
        round(float(sld_nugget.value), 6),
        round(float(sld_smooth.value), 3),
    )

    velocity_stack = base_velocity.values[None, :, :] + normalized * sld_std_dev.value
    velocity_stack = np.maximum(velocity_stack, 1200.0)

    avg_velocity_map = np.mean(velocity_stack, axis=0)
    final_depth_map = (base_twt.values * avg_velocity_map) / 2000.0
    depth_stack = (base_twt.values[None, :, :] * velocity_stack) / 2000.0

    return velocity_stack, depth_stack, avg_velocity_map, final_depth_map


def get_trap_and_spill(depth_map: np.ndarray, step: float) -> tuple[float, np.ndarray, float]:
    crest_idx = int(np.argmin(depth_map))
    crest_y, crest_x = np.unravel_index(crest_idx, depth_map.shape)
    crest_depth = float(depth_map[crest_y, crest_x])

    current_level = crest_depth + max(step, 0.1)
    previous_closed_mask = np.zeros_like(depth_map, dtype=bool)
    spill_depth = crest_depth

    max_depth = float(np.max(depth_map))
    while current_level <= max_depth:
        flooded = depth_map <= current_level
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
        current_level += max(step, 0.1)

    return float(spill_depth), previous_closed_mask, crest_depth


def compute_trap_statistics(
    depth_stack: np.ndarray,
    contour_step: float,
    thickness_mean: float,
    thickness_std: float,
) -> dict[str, np.ndarray]:
    n_maps = depth_stack.shape[0]
    cell_area = 100.0 * 100.0

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

        spill_depths[map_idx] = spill_depth
        crest_depths[map_idx] = crest_depth
        trap_masks[map_idx] = trap_mask

        if not np.any(trap_mask):
            continue

        area = float(np.sum(trap_mask) * cell_area)
        areas[map_idx] = area
        thickness_total[map_idx] = max(0.0, spill_depth - crest_depth)

        total_volume = float(np.sum((spill_depth - depth_map[trap_mask])) * cell_area)

        sampled_thickness = max(10.0, float(rng.normal(thickness_mean, thickness_std)))
        base_depth_map = depth_map + sampled_thickness
        base_mask = trap_mask & (base_depth_map < spill_depth)

        if np.any(base_mask):
            base_volume = float(np.sum((spill_depth - base_depth_map[base_mask])) * cell_area)
        else:
            base_volume = 0.0

        grv[map_idx] = max(0.0, total_volume - base_volume)

    return {
        "trap_masks": trap_masks,
        "spill_depths": spill_depths,
        "crest_depths": crest_depths,
        "areas": areas,
        "thickness_total": thickness_total,
        "grv": grv,
    }


def _build_report_payload() -> dict[str, np.ndarray | pd.DataFrame | float]:
    _, depth_stack, avg_velocity_map, final_depth_map = get_velocity_and_depth_stacks()
    trap_stats = compute_trap_statistics(depth_stack, sld_c_inc.value, sld_thick_mean.value, sld_thick_std.value)

    arr_thick = np.asarray(trap_stats["thickness_total"])
    arr_area = np.asarray(trap_stats["areas"]) / 1e6
    arr_grv = np.asarray(trap_stats["grv"]) / 1e9
    arr_stoiip = (
        np.asarray(trap_stats["grv"])
        * sld_ntg.value
        * sld_poro.value
        * (1.0 - sld_sw.value)
        / sld_fvf.value
        * 6.2898
        / 1e6
    )

    probability = trap_stats["trap_masks"].mean(axis=0) * 100.0

    summary_df = pd.DataFrame(
        {
            "Metric": ["Thickness (m)", "Area (km²)", "GRV (km³)", "STOIIP (MMbbls)"],
            "P90": [np.percentile(arr_thick, 10), np.percentile(arr_area, 10), np.percentile(arr_grv, 10), np.percentile(arr_stoiip, 10)],
            "P50": [np.percentile(arr_thick, 50), np.percentile(arr_area, 50), np.percentile(arr_grv, 50), np.percentile(arr_stoiip, 50)],
            "P10": [np.percentile(arr_thick, 90), np.percentile(arr_area, 90), np.percentile(arr_grv, 90), np.percentile(arr_stoiip, 90)],
            "Mean": [np.mean(arr_thick), np.mean(arr_area), np.mean(arr_grv), np.mean(arr_stoiip)],
        }
    )

    realization_df = pd.DataFrame(
        {
            "Realization": np.arange(1, depth_stack.shape[0] + 1),
            "SpillDepth(m)": trap_stats["spill_depths"],
            "CrestDepth(m)": trap_stats["crest_depths"],
            "Thickness(m)": arr_thick,
            "Area(km2)": arr_area,
            "GRV(km3)": arr_grv,
            "STOIIP(MMbbls)": arr_stoiip,
        }
    )

    return {
        "depth_stack": depth_stack,
        "avg_velocity_map": avg_velocity_map,
        "final_depth_map": final_depth_map,
        "trap_stats": trap_stats,
        "probability": probability,
        "arr_thick": arr_thick,
        "arr_area": arr_area,
        "arr_grv": arr_grv,
        "arr_stoiip": arr_stoiip,
        "summary_df": summary_df,
        "realization_df": realization_df,
    }


def _plot_hist_with_lines(ax, data: np.ndarray, title: str, xlabel: str, color: str) -> None:
    ax.hist(data, bins=20, color=color, edgecolor="black", alpha=0.75)
    if len(data) > 0:
        p90 = float(np.percentile(data, 10))
        p50 = float(np.percentile(data, 50))
        p10 = float(np.percentile(data, 90))
        mean_val = float(np.mean(data))
        ax.axvline(p90, color="red", linestyle="--", linewidth=1.4)
        ax.axvline(p50, color="green", linestyle="--", linewidth=1.4)
        ax.axvline(p10, color="blue", linestyle="--", linewidth=1.4)
        ax.axvline(mean_val, color="black", linestyle="--", linewidth=1.2)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Freq")
    ax.grid(alpha=0.25)


def _build_dashboard_figure(report_title: str, payload: dict) -> plt.Figure:
    depth_stack = payload["depth_stack"]
    avg_velocity_map = payload["avg_velocity_map"]
    final_depth_map = payload["final_depth_map"]
    probability = payload["probability"]
    arr_thick = payload["arr_thick"]
    arr_area = payload["arr_area"]
    arr_grv = payload["arr_grv"]
    arr_stoiip = payload["arr_stoiip"]

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(report_title, fontsize=15, fontweight="bold")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.0], height_ratios=[1.0, 1.0], wspace=0.42, hspace=0.3)

    extent = [base_twt.x.values.min(), base_twt.x.values.max(), base_twt.y.values.min(), base_twt.y.values.max()]

    ax_twt = fig.add_subplot(gs[0, 0])
    twt_img = ax_twt.imshow(base_twt.values, origin="lower", extent=extent, cmap=sel_cmap.value, aspect="auto")
    ax_twt.contour(base_twt.x.values, base_twt.y.values, base_twt.values, levels=sld_contours.value, colors="black", alpha=0.35, linewidths=0.7)
    ax_twt.axhline(y_section_value, color="red", linestyle="--", linewidth=1.4)
    ax_twt.set_title("Input Deterministic TWT Map (ms)", fontsize=10, fontweight="bold")
    ax_twt.set_xlabel("x")
    ax_twt.set_ylabel("y")
    fig.colorbar(twt_img, ax=ax_twt, fraction=0.046, pad=0.02)

    ax_section = fig.add_subplot(gs[0, 1])
    x_coords = base_twt.x.values
    section_stack = depth_stack[:, y_section_idx, :]
    for row in section_stack:
        ax_section.plot(x_coords, row, color="red", alpha=0.2, linewidth=0.7)
    ax_section.plot(x_coords, final_depth_map[y_section_idx, :], color="black", linewidth=2.2, label="Final Depth from Mean AV")
    deterministic_spill, _, _ = get_trap_and_spill(final_depth_map, sld_c_inc.value)
    ax_section.axhline(deterministic_spill, color="blue", linestyle="--", linewidth=1.5, label="Deterministic Spill")
    top_y, bottom_y = sld_y_range.value
    ax_section.set_ylim(bottom_y, top_y)
    ax_section.set_title(f"Structural Section (y={y_section_value}m)", fontsize=10, fontweight="bold")
    ax_section.set_xlabel("Distance (m)")
    ax_section.set_ylabel("Depth (m)")
    ax_section.yaxis.labelpad = 2
    ax_section.legend(loc="upper right", fontsize=8)
    ax_section.grid(alpha=0.25)

    hist_gs = gs[:, 2].subgridspec(4, 1, hspace=0.45)
    _plot_hist_with_lines(fig.add_subplot(hist_gs[0, 0]), arr_thick, "Crest-to-Spill Thick.", "Thickness (m)", "#8888ff")
    _plot_hist_with_lines(fig.add_subplot(hist_gs[1, 0]), arr_area, "Closure Area", "Area (km²)", "#ff8888")
    _plot_hist_with_lines(fig.add_subplot(hist_gs[2, 0]), arr_grv, "GRV Dist.", "GRV (km³)", "#88ff88")
    _plot_hist_with_lines(fig.add_subplot(hist_gs[3, 0]), arr_stoiip, "STOIIP Dist.", "STOIIP (MMbbls)", "#ffaa00")

    ax_depth = fig.add_subplot(gs[1, 0])
    depth_img = ax_depth.imshow(final_depth_map, origin="lower", extent=extent, cmap=sel_cmap.value, aspect="auto")
    ax_depth.axhline(y_section_value, color="red", linestyle="--", linewidth=1.4)
    ax_depth.contour(base_twt.x.values, base_twt.y.values, final_depth_map, levels=[deterministic_spill], colors="red", linestyles="--", linewidths=1.8)
    ax_depth.set_title("Final Depth Map = (TWT ms × AV)/2000", fontsize=10, fontweight="bold")
    ax_depth.set_xlabel("x")
    ax_depth.set_ylabel("y")
    fig.colorbar(depth_img, ax=ax_depth, fraction=0.046, pad=0.02)

    ax_prob = fig.add_subplot(gs[1, 1])
    prob_img = ax_prob.imshow(probability, origin="lower", extent=extent, cmap="rainbow", vmin=0, vmax=100, aspect="auto")
    ax_prob.contour(base_twt.x.values, base_twt.y.values, probability, levels=10, colors="black", alpha=0.35, linewidths=0.7)
    ax_prob.set_title(f"Isoprobability Map (Isolated Trap, N={depth_stack.shape[0]})", fontsize=10, fontweight="bold")
    ax_prob.set_xlabel("x")
    ax_prob.set_ylabel("y")
    fig.colorbar(prob_img, ax=ax_prob, fraction=0.046, pad=0.02)

    return fig


def _build_pdf_tables_page(report_title: str, payload: dict) -> plt.Figure:
    summary_df = payload["summary_df"]
    realization_df = payload["realization_df"]

    summary_display = summary_df.copy()
    for col in ["P90", "P50", "P10", "Mean"]:
        if col in summary_display.columns:
            summary_display[col] = summary_display[col].astype(float).round(3)

    realization_display = realization_df.copy()
    numeric_cols = realization_display.select_dtypes(include=[np.number]).columns
    realization_display[numeric_cols] = realization_display[numeric_cols].round(3)

    fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27))
    fig.suptitle(f"{report_title} — Tables", fontsize=14, fontweight="bold")

    axes[0].axis("off")
    summary_tbl = axes[0].table(
        cellText=summary_display.values,
        colLabels=summary_display.columns,
        loc="center",
        cellLoc="center",
    )
    summary_tbl.auto_set_font_size(False)
    summary_tbl.set_fontsize(9)
    summary_tbl.scale(1.0, 1.45)
    axes[0].set_title("Summary Statistics (P90 / P50 / P10 / Mean)", fontsize=11, fontweight="bold", pad=10)

    axes[1].axis("off")
    head_df = realization_display.head(20).copy()
    detail_tbl = axes[1].table(
        cellText=head_df.values,
        colLabels=head_df.columns,
        loc="center",
        cellLoc="center",
    )
    detail_tbl.auto_set_font_size(False)
    detail_tbl.set_fontsize(8)
    detail_tbl.scale(1.0, 1.2)
    axes[1].set_title("Per-Realization Results (first 20 rows)", fontsize=11, fontweight="bold", pad=10)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def _build_pdf_parameters_page(report_title: str) -> plt.Figure:
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_subplot(111)
    ax.axis("off")

    lines = [
        f"{report_title} — Parameters and Configuration",
        "",
        "Input & Realizations",
        f"- Surface: {sel_surface.value}",
        f"- Number of realizations: {sld_n_maps.value}",
        f"- Velocity Std. Dev. (m/s): {sld_std_dev.value}",
        f"- Smoothing sigma: {sld_smooth.value}",
        f"- Contour step (spill search, m): {sld_c_inc.value}",
        "",
        "Variogram",
        f"- Model type: {sld_slope.value}",
        f"- Range (m): {sld_range.value}",
        f"- Sill: {sld_sill.value}",
        f"- Nugget: {sld_nugget.value}",
        "",
        "Volumetrics",
        f"- Thickness mean (m): {sld_thick_mean.value}",
        f"- Thickness std. dev. (m): {sld_thick_std.value}",
        f"- N/G: {sld_ntg.value}",
        f"- Porosity: {sld_poro.value}",
        f"- Water saturation (Sw): {sld_sw.value}",
        f"- FVF: {sld_fvf.value}",
        "",
        "Equations",
        "- Depth(m) = TWT(ms) × AV(m/s) / 2000",
        "- HIIP = (GRV × N/G × Φ × (1 - Sw)) / FVF",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=11)

    return fig


# ==============================================================================
#  UI WIDGETS
# ==============================================================================
sel_surface = pn.widgets.Select(
    name="Select Surface",
    options=["Synthetic TWT Input"],
    value="Synthetic TWT Input",
    stylesheets=select_stylesheets,
)
sld_contours = pn.widgets.IntSlider(
    name="Contours Range Number",
    start=5,
    end=50,
    value=20,
    stylesheets=slider_stylesheets,
)
sel_cmap = pn.widgets.Select(
    name="Colormap",
    options=["viridis", "plasma", "terrain", "rainbow"],
    value="terrain",
    stylesheets=select_stylesheets,
)
btn_generate_surf = pn.widgets.Button(
    name="Generate / Update TWT Input",
    css_classes=["omv-run-btn"],
    stylesheets=get_neon_button_stylesheets(),
    sizing_mode="stretch_width",
)

sld_c_inc = pn.widgets.FloatSlider(
    name="Contour Search Step (m)",
    start=1,
    end=20,
    value=5,
    stylesheets=slider_stylesheets,
)
chk_eliminate = pn.widgets.Checkbox(name="Eliminate outside closure", value=False)
chk_close_poly = pn.widgets.Checkbox(name="Show closure contour (red)", value=True)
btn_update_culm = pn.widgets.Button(
    name="Update Closure / Culmination",
    css_classes=["omv-run-btn"],
    stylesheets=get_neon_button_stylesheets(),
    sizing_mode="stretch_width",
)

sld_n_maps = pn.widgets.IntSlider(
    name="Number of Velocity Realizations",
    start=5,
    end=250,
    value=50,
    stylesheets=slider_stylesheets,
)
sld_std_dev = pn.widgets.FloatSlider(
    name="Velocity Std. Dev. (m/s)",
    start=0.0,
    end=300.0,
    value=43.0,
    stylesheets=slider_stylesheets,
)
sld_smooth = pn.widgets.FloatSlider(
    name="Smoothing (Sigma)",
    start=0.0,
    end=10.0,
    value=2.0,
    step=0.5,
    stylesheets=slider_stylesheets,
)
sld_y_range = pn.widgets.RangeSlider(
    name="Depth Y-Axis Range",
    start=0,
    end=10000,
    value=(4500, 6500),
    step=100,
    stylesheets=slider_stylesheets,
)
sld_map_show = pn.widgets.IntSlider(
    name="Velocity Realization to Highlight",
    start=1,
    end=50,
    value=25,
    stylesheets=slider_stylesheets,
)
btn_update_depth_maps = pn.widgets.Button(
    name="Update Velocity & Depth Realizations",
    css_classes=["omv-run-btn"],
    stylesheets=get_neon_button_stylesheets(),
    sizing_mode="stretch_width",
)


@pn.depends(sld_n_maps, watch=True)
def _update_map_slider(n_maps_val: int) -> None:
    sld_map_show.end = n_maps_val
    if sld_map_show.value > n_maps_val:
        sld_map_show.value = n_maps_val


sld_range = pn.widgets.FloatSlider(
    name="Range (m)",
    start=500,
    end=5000,
    value=1500,
    stylesheets=slider_stylesheets,
)
sld_sill = pn.widgets.FloatSlider(
    name="Sill (Variance)",
    start=0.1,
    end=2.0,
    value=1.0,
    stylesheets=slider_stylesheets,
)
sld_nugget = pn.widgets.FloatSlider(
    name="Nugget",
    start=0.0,
    end=0.5,
    value=0.0,
    stylesheets=slider_stylesheets,
)
sld_slope = pn.widgets.Select(
    name="Model Type",
    options=["Gaussian", "Spherical", "Exponential"],
    value="Gaussian",
    stylesheets=select_stylesheets,
)
btn_update_vario = pn.widgets.Button(
    name="Update Variogram",
    css_classes=["omv-run-btn"],
    stylesheets=get_neon_button_stylesheets(),
    sizing_mode="stretch_width",
)

sld_ntg = pn.widgets.FloatSlider(
    name="Net-to-Gross Ratio (N/G)",
    start=0.01,
    end=1.0,
    value=0.8,
    step=0.01,
    stylesheets=slider_stylesheets,
)
sld_poro = pn.widgets.FloatSlider(
    name="Porosity (Φ)",
    start=0.01,
    end=0.4,
    value=0.2,
    step=0.01,
    stylesheets=slider_stylesheets,
)
sld_sw = pn.widgets.FloatSlider(
    name="Water Saturation (Sw)",
    start=0.01,
    end=1.0,
    value=0.3,
    step=0.01,
    stylesheets=slider_stylesheets,
)
sld_fvf = pn.widgets.FloatSlider(
    name="Formation Volume Factor (FVF)",
    start=1.0,
    end=2.0,
    value=1.2,
    step=0.05,
    stylesheets=slider_stylesheets,
)
sld_thick_mean = pn.widgets.FloatSlider(
    name="Thickness Mean (m)",
    start=10,
    end=200,
    value=102,
    step=1,
    stylesheets=slider_stylesheets,
)
sld_thick_std = pn.widgets.FloatSlider(
    name="Thickness Std. Dev. (m)",
    start=0,
    end=100,
    value=37,
    step=1,
    stylesheets=slider_stylesheets,
)
btn_update_volumetrics = pn.widgets.Button(
    name="Run Monte Carlo Volumetrics",
    css_classes=["omv-run-btn"],
    stylesheets=get_neon_button_stylesheets(),
    sizing_mode="stretch_width",
)

txt_report_header = pn.widgets.TextInput(name="Plot Header", value="Structural Uncertainty Evaluation")
txt_report_filename = pn.widgets.TextInput(name="File Name", value="structural_uncertainty_report")
txt_report_location = pn.widgets.TextInput(name="Location", value=str(APP_DIR / "reports"))
sel_report_type = pn.widgets.Select(name="Type", options=[".png", ".jpeg", ".pdf"], value=".pdf", stylesheets=select_stylesheets)
btn_export_report = pn.widgets.Button(
    name="Generate Report",
    css_classes=["omv-run-btn"],
    stylesheets=get_neon_button_stylesheets(),
    sizing_mode="stretch_width",
)
report_status = pn.pane.Markdown("Ready to export report.")


# ==============================================================================
#  PLOTTING FUNCTIONS
# ==============================================================================
@pn.depends(btn_generate_surf)
def plot_top_left_input_twt_map(*_args):
    logger.info("Rendering input TWT map")

    da = base_twt.copy()
    base_plot = da.hvplot.image(
        cmap=sel_cmap.value,
        title="Input Deterministic TWT Map (ms)",
        colorbar=True,
    )
    contours = da.hvplot.contour(levels=sld_contours.value, color="black", alpha=0.5)
    cross_section = hv.HLine(y_section_value).opts(color="red", line_width=2, line_dash="dashed")
    return (base_plot * contours * cross_section).opts(toolbar="above", aspect=None, data_aspect=None)


@pn.depends(btn_update_depth_maps, btn_update_culm, btn_update_vario)
def plot_top_right_section(*_args):
    logger.info("Rendering structural section with stochastic depth realizations")

    _, depth_stack, _, final_depth_map = get_velocity_and_depth_stacks()

    x_coords = base_twt.x.values
    base_section = final_depth_map[y_section_idx, :]
    realization_section = depth_stack[:, y_section_idx, :]

    base_curve = hv.Curve(
        (x_coords, base_section),
        kdims=["Distance_x"],
        vdims=["Depth_Z"],
        label="Final Depth from Mean AV",
    ).opts(color="black", line_width=3)

    selected_idx = min(max(sld_map_show.value - 1, 0), realization_section.shape[0] - 1)
    realization_curves = [
        hv.Curve((x_coords, realization_section[i]), label="Stochastic Depth Realizations").opts(
            color="red",
            line_width=0.6,
            alpha=0.25,
        )
        for i in range(realization_section.shape[0])
        if i != selected_idx
    ]
    selected_curve = hv.Curve(
        (x_coords, realization_section[selected_idx]),
        kdims=["Distance_x"],
        vdims=["Depth_Z"],
        label=f"Selected Realization #{selected_idx + 1}",
    ).opts(color="red", line_width=2.3, alpha=1.0)

    deterministic_spill, _, _ = get_trap_and_spill(final_depth_map, sld_c_inc.value)
    spill_line = hv.HLine(deterministic_spill, label="Deterministic Spill Point").opts(color="blue", line_dash="dashed", line_width=2)

    top_y, bottom_y = sld_y_range.value

    return (hv.Overlay(realization_curves + [selected_curve]) * base_curve * spill_line).opts(
        toolbar="above",
        title=f"Structural Section (y={y_section_value}m)",
        ylim=(bottom_y, top_y),
        xlabel="Distance (m)",
        ylabel="Depth (m)",
        shared_axes=False,
        show_legend=True,
        legend_position="top_right",
    )


@pn.depends(btn_update_depth_maps, btn_update_culm, btn_update_vario)
def plot_bottom_left_final_depth_from_av(*_args):
    logger.info("Rendering average AV map and final depth map")

    _, _, avg_velocity_map, final_depth_map = get_velocity_and_depth_stacks()

    av_da = xr.DataArray(avg_velocity_map, coords=base_twt.coords, dims=base_twt.dims, name="AV")
    depth_da = xr.DataArray(final_depth_map, coords=base_twt.coords, dims=base_twt.dims, name="Depth")

    av_plot = av_da.hvplot.image(cmap="viridis", colorbar=True, title="Average Velocity Map (AV, m/s)").opts(
        aspect=None,
        data_aspect=None,
        toolbar="above",
    )
    depth_plot = depth_da.hvplot.image(cmap=sel_cmap.value, colorbar=True, title="Final Depth Map = (TWT ms × AV)/2000").opts(
        aspect=None,
        data_aspect=None,
        toolbar="above",
    )
    overlay = depth_plot * hv.HLine(y_section_value).opts(color="red", line_width=2, line_dash="dashed")

    deterministic_spill, deterministic_mask, _ = get_trap_and_spill(final_depth_map, sld_c_inc.value)

    if chk_eliminate.value and np.any(deterministic_mask):
        masked_depth = depth_da.where(deterministic_mask)
        overlay = masked_depth.hvplot.image(cmap=sel_cmap.value, colorbar=True, title="Final Depth Map (Isolated Closure)")

    if chk_close_poly.value:
        red_contour = depth_da.hvplot.contour(levels=[deterministic_spill], cmap=["red"]).opts(line_width=3, line_dash="dashed")
        overlay = overlay * red_contour

    return pn.Tabs(
        ("Average AV", pn.panel(av_plot, sizing_mode="stretch_both")),
        ("Final Depth", pn.panel(overlay.opts(aspect=None, data_aspect=None), sizing_mode="stretch_both")),
        dynamic=False,
        tabs_location="above",
        sizing_mode="stretch_both",
    )


@pn.depends(btn_update_vario)
def plot_bottom_left_variogram(*_args):
    h = np.linspace(0, 5000, 200)
    if sld_slope.value == "Gaussian":
        gamma = sld_nugget.value + sld_sill.value * (1 - np.exp(-(h**2) / ((sld_range.value / 1.732) ** 2)))
    elif sld_slope.value == "Exponential":
        gamma = sld_nugget.value + sld_sill.value * (1 - np.exp(-(3 * h) / sld_range.value))
    else:
        gamma = np.where(
            h <= sld_range.value,
            sld_nugget.value + sld_sill.value * (1.5 * (h / sld_range.value) - 0.5 * (h / sld_range.value) ** 3),
            sld_nugget.value + sld_sill.value,
        )

    df = pd.DataFrame({"Distance": h, "Semivariance": gamma})
    curve = df.hvplot.line(x="Distance", y="Semivariance", color="teal", line_width=2)
    sill_line = hv.HLine(sld_nugget.value + sld_sill.value).opts(color="red", line_width=1)
    range_line = hv.VLine(sld_range.value).opts(color="red", line_width=1)

    return (curve * sill_line * range_line).opts(
        toolbar="above",
        xlim=(0, None),
        ylim=(0, None),
        title="Theoretical Variogram",
    )


@pn.depends(btn_update_depth_maps, btn_update_culm, btn_update_vario, btn_update_volumetrics)
def plot_bottom_right_isoprobability(*_args):
    logger.info("Rendering isoprobability map")

    _, depth_stack, _, _ = get_velocity_and_depth_stacks()
    n_maps = depth_stack.shape[0]

    trap_stats = compute_trap_statistics(depth_stack, sld_c_inc.value, sld_thick_mean.value, sld_thick_std.value)
    count_array = trap_stats["trap_masks"].sum(axis=0).astype(float)
    prob_array = (count_array / n_maps) * 100

    ds_prob = xr.Dataset(
        {
            "Probability": (["y", "x"], prob_array),
            "ClosureCount": (["y", "x"], count_array),
            "TotalMaps": (["y", "x"], np.full_like(count_array, n_maps)),
        },
        coords={"x": base_twt.x.values, "y": base_twt.y.values},
    )

    custom_hover = HoverTool(
        tooltips=[
            ("X, Y", "$x{0,0}, $y{0,0}"),
            ("Probability", "@image{0.0}%"),
            ("Closure Detail", "In @ClosureCount maps of @TotalMaps this node is within isolated trap"),
        ]
    )

    hv_ds = hv.Dataset(ds_prob)
    img = hv_ds.to(hv.Image, kdims=["x", "y"], vdims=["Probability", "ClosureCount", "TotalMaps"]).opts(
        cmap="rainbow",
        clim=(0, 100),
        colorbar=True,
        tools=[custom_hover],
    )
    contours = ds_prob["Probability"].hvplot.contour(levels=10, color="black", line_width=1, alpha=0.7)

    return (img * contours).opts(
        toolbar="above",
        aspect=None,
        data_aspect=None,
        title=f"Isoprobability Map (Isolated Trap, N={n_maps})",
    )


def create_hist_with_stats(data: np.ndarray, name: str, color: str, title: str):
    if len(data) == 0 or np.all(data == 0):
        return hv.Curve([]).opts(title=title)

    mean_val = float(np.mean(data))
    p90_val = float(np.percentile(data, 10))
    p50_val = float(np.percentile(data, 50))
    p10_val = float(np.percentile(data, 90))

    counts, _ = np.histogram(data, bins=20)
    max_y = float(counts.max())
    x_offset = float((data.max() - data.min()) * 0.02)

    hist = pd.DataFrame({name: data}).hvplot.hist(
        name,
        bins=20,
        color=color,
        title=title,
        ylabel="Freq",
    )

    l_mean = hv.VLine(mean_val, label="Mean").opts(color="black", line_dash="dashed", line_width=1.5)
    l_p90 = hv.VLine(p90_val, label="P90").opts(color="red", line_dash="dashed", line_width=1.5)
    l_p50 = hv.VLine(p50_val, label="P50").opts(color="green", line_dash="dashed", line_width=1.5)
    l_p10 = hv.VLine(p10_val, label="P10").opts(color="blue", line_dash="dashed", line_width=1.5)

    t_mean = hv.Text(mean_val + x_offset, max_y * 0.95, f"Mean: {mean_val:.2f}", halign="left", valign="top").opts(
        color="black",
        text_font_size="8pt",
        text_font_style="bold",
    )
    t_p90 = hv.Text(p90_val + x_offset, max_y * 0.80, f"P90: {p90_val:.2f}", halign="left", valign="top").opts(
        color="red",
        text_font_size="8pt",
        text_font_style="bold",
    )
    t_p50 = hv.Text(p50_val + x_offset, max_y * 0.95, f"P50: {p50_val:.2f}", halign="left", valign="top").opts(
        color="green",
        text_font_size="8pt",
        text_font_style="bold",
    )
    t_p10 = hv.Text(p10_val + x_offset, max_y * 0.80, f"P10: {p10_val:.2f}", halign="left", valign="top").opts(
        color="blue",
        text_font_size="8pt",
        text_font_style="bold",
    )

    return (hist * l_p90 * l_p50 * l_p10 * l_mean * t_p90 * t_p50 * t_p10 * t_mean).opts(
        show_legend=True,
        legend_position="top_right",
        shared_axes=False,
    )


def _export_report(_event) -> None:
    header = txt_report_header.value.strip() or "Structural Uncertainty Evaluation"
    base_name = txt_report_filename.value.strip() or "structural_uncertainty_report"
    extension = sel_report_type.value
    out_dir = Path(txt_report_location.value.strip() or str(APP_DIR / "reports"))

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{base_name}{extension}"

        payload = _build_report_payload()

        if extension in {".png", ".jpeg"}:
            fig = _build_dashboard_figure(header, payload)
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        else:
            with PdfPages(output_path) as pdf:
                fig_dashboard = _build_dashboard_figure(header, payload)
                pdf.savefig(fig_dashboard, bbox_inches="tight")
                plt.close(fig_dashboard)

                fig_tables = _build_pdf_tables_page(header, payload)
                pdf.savefig(fig_tables, bbox_inches="tight")
                plt.close(fig_tables)

                fig_params = _build_pdf_parameters_page(header)
                pdf.savefig(fig_params, bbox_inches="tight")
                plt.close(fig_params)

        report_status.object = f"✅ Report exported: {output_path}"
        logger.info("Report exported: %s", output_path)
    except Exception as exc:
        report_status.object = f"❌ Export failed: {exc}"
        logger.exception("Report export failed")


btn_export_report.on_click(_export_report)


@pn.depends(btn_update_volumetrics, btn_update_depth_maps, btn_update_culm, btn_update_vario)
def plot_third_column_volumetrics(*_args):
    logger.info("Computing volumetrics distribution")

    _, depth_stack, _, _ = get_velocity_and_depth_stacks()

    trap_stats = compute_trap_statistics(depth_stack, sld_c_inc.value, sld_thick_mean.value, sld_thick_std.value)
    arr_thick = np.asarray(trap_stats["thickness_total"])
    arr_area = np.asarray(trap_stats["areas"]) / 1e6
    arr_grv = np.asarray(trap_stats["grv"]) / 1e9
    arr_stoiip = (
        np.asarray(trap_stats["grv"])
        * sld_ntg.value
        * sld_poro.value
        * (1.0 - sld_sw.value)
        / sld_fvf.value
        * 6.2898
        / 1e6
    )

    p1 = create_hist_with_stats(arr_thick, "Thickness (m)", "#8888ff", "Crest-to-Spill Thick.")
    p2 = create_hist_with_stats(arr_area, "Area (km²)", "#ff8888", "Closure Area")
    p3 = create_hist_with_stats(arr_grv, "GRV (km³)", "#88ff88", "GRV Dist.")
    p4 = create_hist_with_stats(arr_stoiip, "STOIIP (MMbbls)", "#ffaa00", "STOIIP Dist.")

    hist_grid = pn.GridSpec(nrows=4, ncols=1, sizing_mode="stretch_both", margin=0)
    hist_grid[0, 0] = pn.panel(p1, sizing_mode="stretch_both")
    hist_grid[1, 0] = pn.panel(p2, sizing_mode="stretch_both")
    hist_grid[2, 0] = pn.panel(p3, sizing_mode="stretch_both")
    hist_grid[3, 0] = pn.panel(p4, sizing_mode="stretch_both")
    return pn.Column(hist_grid, css_classes=["hist-container"], sizing_mode="stretch_both", margin=0)


# ==============================================================================
#  LAYOUT DEFINITION
# ==============================================================================
def wrap_plot(title: str, plot_func, area_name: str) -> pn.Column:
    pane = pn.panel(plot_func, sizing_mode="stretch_both", css_classes=["plot-wrapper"])
    return pn.Column(
        pn.pane.HTML(f"<div class='plot-title'>{title}</div>"),
        pane,
        css_classes=["plot-box"],
        sizing_mode="stretch_both",
        styles={"grid-area": area_name},
    )


main_grid = pn.Column(
    wrap_plot("Input Time Surface (TWT)", plot_top_left_input_twt_map, "map"),
    wrap_plot("Structural Section (Depth Realizations)", plot_top_right_section, "section"),
    wrap_plot("AV Mean and Final Depth", plot_bottom_left_final_depth_from_av, "realization"),
    wrap_plot("Isoprobability Map", plot_bottom_right_isoprobability, "iso"),
    wrap_plot("Monte Carlo Distributions", plot_third_column_volumetrics, "hists"),
    css_classes=["dashboard-grid"],
    sizing_mode="stretch_both",
)


main_content = pn.Column(
    main_grid,
    sizing_mode="stretch_both",
    margin=0,
    styles={
        "height": "100%",
        "overflow": "hidden",
        "background": get_main_outer_background(is_dark_mode),
        "color": section_text_color if is_dark_mode else "inherit",
    },
)


def create_sidebar_card(title: str, *widgets, collapsed: bool = True) -> pn.Card:
    return pn.Card(
        *widgets,
        title=title,
        collapsed=collapsed,
        hide_header=False,
        sizing_mode="stretch_width",
        header_background=section_header_background,
        active_header_background=section_header_background,
        header_color=section_text_color,
        styles={"background": section_body_background, "color": section_text_color},
        margin=(0, 0, 12, 0),
    )


card_1 = create_sidebar_card("Input Selection", sel_surface, sld_contours, sel_cmap, btn_generate_surf, collapsed=False)
card_2 = create_sidebar_card("Trap Detection Rules", sld_c_inc, chk_eliminate, chk_close_poly, btn_update_culm)
card_3 = create_sidebar_card(
    "Velocity Realizations",
    sld_n_maps,
    sld_std_dev,
    sld_smooth,
    sld_y_range,
    sld_map_show,
    btn_update_depth_maps,
)

vario_pane = pn.panel(plot_bottom_left_variogram, sizing_mode="stretch_width", min_height=250)
card_4 = create_sidebar_card("Variogram Parameters", sld_slope, sld_range, sld_sill, sld_nugget, btn_update_vario, vario_pane)

formula_text = pn.pane.Markdown("*Formula: Depth(m) = TWT(ms) × AV(m/s) / 2000; HIIP = (GRV × N/G × Φ × (1 - Sw)) / FVF*")
card_5 = create_sidebar_card("GRV and STOIIP", formula_text, sld_thick_mean, sld_thick_std, sld_ntg, sld_poro, sld_sw, sld_fvf, btn_update_volumetrics)
card_6 = create_sidebar_card(
    "Report",
    txt_report_header,
    txt_report_filename,
    txt_report_location,
    sel_report_type,
    btn_export_report,
    report_status,
)
card_7 = create_sidebar_card("Export to Petrel", pn.Spacer(height=10))

sidebar_items = [card_1, card_4, card_2, card_3, card_5, card_6, card_7]


template_kwargs = dict(
    title=APP_TITLE,
    accent_base_color=BLUE_OMV_COLOR,
    header_background=DARK_BLUE_OMV_COLOR,
    main_layout=None,
    main_max_width="",
    sidebar=sidebar_items,
    main=[main_content],
    header=[
        pn.Row(
            pn.Spacer(sizing_mode="stretch_width"),
            pn.pane.HTML(docs_button_html(DOCUMENTATION_URL)),
            sizing_mode="stretch_width",
            margin=0,
        )
    ],
)

if LOGO_PATH.exists():
    template_kwargs["logo"] = str(LOGO_PATH)
if FAVICON_PATH.exists():
    template_kwargs["favicon"] = str(FAVICON_PATH)

logger.info("Workflow aligned: deterministic TWT -> AV realizations -> depth conversion")
pn.template.FastListTemplate(**template_kwargs).servable()

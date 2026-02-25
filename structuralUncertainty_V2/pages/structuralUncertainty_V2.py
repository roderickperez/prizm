# -*- coding: utf-8 -*-
import ctypes
import json
import logging
import os
import platform
import sys
import threading
import time
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
APP_DIR = Path(__file__).resolve().parent
STRUCTURAL_DIR = APP_DIR.parent
ROOT_DIR = STRUCTURAL_DIR.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if str(STRUCTURAL_DIR) not in sys.path:
    sys.path.insert(0, str(STRUCTURAL_DIR))

from core.engine_V2 import (  # noqa: E402
    build_surfaces as core_build_surfaces,
    compute_trap_statistics as core_compute_trap_statistics,
    extract_section_stack as core_extract_section_stack,
    generate_velocity_depth_stacks as core_generate_velocity_depth_stacks,
    get_trap_and_spill as core_get_trap_and_spill,
    line_through_center as core_line_through_center,
)
from core.pdf_report_tables_V2 import build_pdf_tables_pages as core_build_pdf_tables_pages  # noqa: E402
from core.qc_analysis_V2 import (  # noqa: E402
    build_geostat_qc_bundle as core_build_geostat_qc_bundle,
    build_pdf_geostat_qc_pages as core_build_pdf_geostat_qc_pages,
)
# Fixed import: Directly import from the folder without hardcoding the folder name
from petrel_surface_parser_V2 import parse_petrel_surface_file  # noqa: E402

try:
    from shared.ui.omv_theme import (
        BLUE_OMV_COLOR,
        DARK_BLUE_OMV_COLOR,
        DARK_MAGENTA_OMV_COLOR,
        MAGENTA_OMV_COLOR,
        NEON_OMV_COLOR,
        docs_button_html,
        get_content_text_color,
        get_dark_select_stylesheets,
        get_extension_raw_css,
        get_main_outer_background,
        get_neon_button_stylesheets,
        get_radio_button_stylesheets,
        get_slider_stylesheets,
        is_dark_mode_from_state,
    )
except ImportError:
    BLUE_OMV_COLOR = "#005A9B"
    DARK_BLUE_OMV_COLOR = "#003056"
    DARK_MAGENTA_OMV_COLOR = "#4536a2"
    MAGENTA_OMV_COLOR = "#8848ea"
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

    def get_radio_button_stylesheets() -> list[str]:
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

ARGS_LOWER = [str(arg).lower() for arg in sys.argv]
TEST_MODE = "--test" in ARGS_LOWER
DEBUG_TRACE_MODE = TEST_MODE or "--debug" in ARGS_LOWER

ASSETS_DIR = ROOT_DIR / "images" / "OMV_brandLogoPack" / "OMV_brandLogoPack"
LOGO_PATH = ASSETS_DIR / "OMV_logo_Neon_Small.png"
FAVICON_PATH = ASSETS_DIR / "OMV_logo_Blue_Small.png"
REPORT_TABLE_LOGO_PATH = ASSETS_DIR / "OMV_logo_RGB_Deep-Blue.png"

LOG_DIR = APP_DIR / "logs"

is_dark_mode = is_dark_mode_from_state()


# ==============================================================================
#  APP LOGGER (KEEP LAST 10 SESSIONS)
# ==============================================================================
def setup_structural_logger(log_dir: Path, logger_name: str = "structural_uncertainty_app") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"structural_uncertainty_{time.strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG if DEBUG_TRACE_MODE else logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s"
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG if DEBUG_TRACE_MODE else logging.INFO)
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
    logger.info("Runtime modes | test_mode=%s debug_trace=%s", TEST_MODE, DEBUG_TRACE_MODE)
    return logger


logger = setup_structural_logger(LOG_DIR)
av_depth_tabs = None
input_surface_tabs = None


def _install_global_exception_hooks() -> None:
    original_excepthook = sys.excepthook

    def _hook(exc_type, exc_value, exc_traceback):
        try:
            logger.critical("UNCAUGHT_EXCEPTION", exc_info=(exc_type, exc_value, exc_traceback))
        except Exception:
            pass
        try:
            original_excepthook(exc_type, exc_value, exc_traceback)
        except Exception:
            pass

    sys.excepthook = _hook

    if hasattr(threading, "excepthook"):
        original_thread_hook = threading.excepthook

        def _thread_hook(args):
            try:
                logger.critical(
                    "UNCAUGHT_THREAD_EXCEPTION | thread=%s",
                    getattr(args.thread, "name", "unknown"),
                    exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
                )
            except Exception:
                pass
            try:
                original_thread_hook(args)
            except Exception:
                pass

        threading.excepthook = _thread_hook


_install_global_exception_hooks()


def load_snapshot_payload() -> dict:
    data_file = os.environ.get("PWR_DATA_FILE")
    if not data_file or not os.path.exists(data_file):
        return {}
    try:
        with open(data_file, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
            return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


SNAPSHOT_PAYLOAD = load_snapshot_payload()
PROJECT_NAME = str(SNAPSHOT_PAYLOAD.get("project", "Unknown"))
IMPORTED_SURFACE_GUID = SNAPSHOT_PAYLOAD.get("selected_surface_guid")
IMPORTED_SURFACE_NAME = str(SNAPSHOT_PAYLOAD.get("selected_surface_name", "Imported Surface"))
IMPORTED_SURFACE_DATA = SNAPSHOT_PAYLOAD.get("imported_surface") if isinstance(SNAPSHOT_PAYLOAD.get("imported_surface"), dict) else None
IMPORTED_SURFACE_TIME_DATA = SNAPSHOT_PAYLOAD.get("imported_surface_time") if isinstance(SNAPSHOT_PAYLOAD.get("imported_surface_time"), dict) else None
IMPORTED_SURFACE_DEPTH_DATA = SNAPSHOT_PAYLOAD.get("imported_surface_depth") if isinstance(SNAPSHOT_PAYLOAD.get("imported_surface_depth"), dict) else None
TEST_SURFACE_TIME_FILE = ROOT_DIR / "referenceDocumentation" / "structuralUncertaintyEvaluation" / "testData" / "surfaceTime_earhVision"
TEST_SURFACE_DEPTH_FILE = ROOT_DIR / "referenceDocumentation" / "structuralUncertaintyEvaluation" / "testData" / "surfaceDepth_earhVision"


# ==============================================================================
#  PANEL EXTENSION & CUSTOM CSS
# ==============================================================================
custom_css = f"""
.dashboard-grid {{
    display: grid !important;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) minmax(0, 1fr) !important;
    grid-template-rows: minmax(0, 1fr) minmax(0, 1fr) !important;
    grid-template-areas:
        "map section hists"
        "realization iso hists";
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
    gap: 12px;
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
radio_group_stylesheets = get_radio_button_stylesheets()


# ==============================================================================
#  INPUT SURFACES: IMPORTED TIME + DEPTH
# ==============================================================================
x_values = np.linspace(0, 10000, 100)
y_values = np.linspace(0, 10000, 100)
_map_diag_m = float(np.hypot(float(np.max(x_values) - np.min(x_values)), float(np.max(y_values) - np.min(y_values))))
DEFAULT_RANGE_M = float(np.clip(0.25 * _map_diag_m, 500.0, 5000.0))
DEFAULT_SILL = 1.0
DEFAULT_NUGGET = 0.1


def _coerce_surface_dict(surface_dict: dict | None) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not isinstance(surface_dict, dict):
        return None
    try:
        x_arr = np.asarray(surface_dict.get("x", []), dtype=float)
        y_arr = np.asarray(surface_dict.get("y", []), dtype=float)
        z_arr = np.asarray(surface_dict.get("z", []), dtype=float)
        if x_arr.ndim == 1 and y_arr.ndim == 1 and z_arr.ndim == 2 and z_arr.shape == (len(y_arr), len(x_arr)) and len(x_arr) > 1 and len(y_arr) > 1:
            return x_arr.astype(np.float32), y_arr.astype(np.float32), z_arr.astype(np.float32)
    except Exception:
        return None
    return None


def _parse_imported_surfaces() -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray] | None, tuple[np.ndarray, np.ndarray, np.ndarray] | None]:
    time_surface = None
    depth_surface = None

    if TEST_MODE:
        if TEST_SURFACE_TIME_FILE.exists():
            try:
                parsed_t = parse_petrel_surface_file(TEST_SURFACE_TIME_FILE)
                time_surface = (
                    np.asarray(parsed_t["x"], dtype=np.float32),
                    np.asarray(parsed_t["y"], dtype=np.float32),
                    np.asarray(parsed_t["z"], dtype=np.float32),
                )
                logger.info("Loaded test time surface: %s", TEST_SURFACE_TIME_FILE)
            except Exception as exc:
                logger.warning("Test-mode time surface load failed: %s", exc)
        if TEST_SURFACE_DEPTH_FILE.exists():
            try:
                parsed_d = parse_petrel_surface_file(TEST_SURFACE_DEPTH_FILE)
                depth_surface = (
                    np.asarray(parsed_d["x"], dtype=np.float32),
                    np.asarray(parsed_d["y"], dtype=np.float32),
                    np.asarray(parsed_d["z"], dtype=np.float32),
                )
                logger.info("Loaded test depth surface: %s", TEST_SURFACE_DEPTH_FILE)
            except Exception as exc:
                logger.warning("Test-mode depth surface load failed: %s", exc)

    if time_surface is None:
        time_surface = _coerce_surface_dict(IMPORTED_SURFACE_TIME_DATA)
    if depth_surface is None:
        depth_surface = _coerce_surface_dict(IMPORTED_SURFACE_DEPTH_DATA)

    if time_surface is None and depth_surface is None:
        fallback = _coerce_surface_dict(IMPORTED_SURFACE_DATA)
        if fallback is not None:
            time_surface = fallback

    return time_surface, depth_surface


def _resample_depth_to_time_grid(
    depth_surface: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    x_time: np.ndarray,
    y_time: np.ndarray,
) -> np.ndarray | None:
    if depth_surface is None:
        return None
    try:
        x_d, y_d, z_d = depth_surface
        da_d = xr.DataArray(np.asarray(z_d, dtype=float), coords=[np.asarray(y_d, dtype=float), np.asarray(x_d, dtype=float)], dims=["y", "x"])
        da_interp = da_d.interp(x=np.asarray(x_time, dtype=float), y=np.asarray(y_time, dtype=float), method="linear")
        arr = np.asarray(da_interp.values, dtype=float)
        if not np.isfinite(arr).all():
            fill_val = float(np.nanmean(arr)) if np.isfinite(arr).any() else 0.0
            arr = np.where(np.isfinite(arr), arr, fill_val)
        return arr.astype(np.float32)
    except Exception:
        return None


IMPORTED_TIME_SURFACE_PARSED, IMPORTED_DEPTH_SURFACE_PARSED = _parse_imported_surfaces()

surface_state: dict[str, object] = {
    "mode": "Synthetic TWT Input",
    "token": 0,
    "section_angle_applied": 0.0,
}
refresh_token = pn.widgets.IntInput(name="_refresh_token", value=0, visible=False)
wells_refresh_token = pn.widgets.IntInput(name="_wells_refresh_token", value=0, visible=False)

stack_cache: dict[str, object] = {"key": None, "data": None}
trap_cache: dict[str, object] = {"key": None, "data": None}
hist_panel_cache: dict[str, object] = {"key": None, "panel": None}
section_line_cache: dict[str, object] = {"key": None, "value": (np.array([], dtype=float), np.array([], dtype=float))}
wells_applied_state: dict[str, object] = {"df": pd.DataFrame(columns=["Well Name", "X", "Y", "Velocity"])}
MAX_VELOCITY_STD_FPS = 250.0


def _normalize_imported_depth_to_m(depth_values, twt_ms_values) -> np.ndarray:
    depth_arr = np.asarray(depth_values, dtype=float)
    twt_arr = np.asarray(twt_ms_values, dtype=float)
    if depth_arr.size == 0:
        return depth_arr.astype(np.float32)

    finite_depth = depth_arr[np.isfinite(depth_arr)]
    if finite_depth.size == 0:
        return np.nan_to_num(depth_arr, nan=0.0).astype(np.float32)

    depth_abs_med = float(np.nanmedian(np.abs(finite_depth)))
    finite_twt = np.abs(twt_arr[np.isfinite(twt_arr)])
    twt_abs_med = float(np.nanmedian(finite_twt)) if finite_twt.size > 0 else 1.0
    twt_abs_med = max(twt_abs_med, 1.0)
    implied_vel_raw = depth_abs_med * 2000.0 / twt_abs_med

    if implied_vel_raw > 8000.0:
        logger.info(
            "Imported depth interpreted as feet (median |z|=%.1f). Converted to meters.",
            depth_abs_med,
        )
        return np.asarray(depth_arr * 0.3048, dtype=np.float32)

    return np.asarray(depth_arr, dtype=np.float32)


def _initialize_surface_state() -> None:
    time_surface = IMPORTED_TIME_SURFACE_PARSED
    depth_surface = IMPORTED_DEPTH_SURFACE_PARSED

    if time_surface is not None:
        x_ref, y_ref, twt_ms = time_surface
    else:
        x_ref, y_ref = x_values, y_values
        twt_ms = core_build_surfaces("Synthetic TWT Input", x_values, y_values)[0]

    if depth_surface is not None:
        x_d, y_d, depth_in = depth_surface
        if (len(x_d) == len(x_ref)) and (len(y_d) == len(y_ref)):
            depth_m = _normalize_imported_depth_to_m(depth_in, twt_ms)
        else:
            depth_resampled = _resample_depth_to_time_grid(depth_surface, x_ref, y_ref)
            depth_m = _normalize_imported_depth_to_m(depth_resampled, twt_ms) if depth_resampled is not None else (np.asarray(twt_ms, dtype=float) * 3050.0) / 2000.0
    else:
        depth_m = (np.asarray(twt_ms, dtype=float) * 3050.0) / 2000.0

    safe_twt = np.where(np.abs(np.asarray(twt_ms, dtype=float)) > 1e-6, np.asarray(twt_ms, dtype=float), np.nan)
    vel = np.where(np.isfinite(safe_twt), (np.asarray(depth_m, dtype=float) / safe_twt) * 2000.0, np.nan)
    vel = np.nan_to_num(vel, nan=3000.0)
    vel = np.maximum(vel, 500.0).astype(np.float32)

    surface_state["mode"] = "Imported Time + Depth"
    surface_state["twt"] = xr.DataArray(np.asarray(twt_ms, dtype=np.float32), coords=[y_ref, x_ref], dims=["y", "x"], name="TWT")
    surface_state["depth_input"] = xr.DataArray(np.asarray(depth_m, dtype=np.float32), coords=[y_ref, x_ref], dims=["y", "x"], name="DepthInput")
    surface_state["velocity"] = xr.DataArray(vel, coords=[y_ref, x_ref], dims=["y", "x"], name="Velocity")
    surface_state["x_coords"] = x_ref
    surface_state["y_coords"] = y_ref
    surface_state["dx"] = float(np.mean(np.diff(x_ref)))
    surface_state["dy"] = float(np.mean(np.diff(y_ref)))
    surface_state["cell_area"] = float(surface_state["dx"] * surface_state["dy"])
    surface_state["imported_available"] = (time_surface is not None)
    surface_state["imported_name"] = IMPORTED_SURFACE_NAME


_initialize_surface_state()


def _set_progress(value: int) -> None:
    try:
        progress_monte_carlo.value = max(0, min(100, int(value)))
    except Exception:
        pass


def _notify_info(message: str, duration: int = 3000) -> None:
    try:
        notifications = getattr(pn.state, "notifications", None)
        if notifications is not None:
            notifications.info(message, duration=duration)
    except Exception:
        pass


def _notify_success(message: str, duration: int = 3000) -> None:
    try:
        notifications = getattr(pn.state, "notifications", None)
        if notifications is not None:
            notifications.success(message, duration=duration)
    except Exception:
        pass


def _notify_warning(message: str, duration: int = 3000) -> None:
    try:
        notifications = getattr(pn.state, "notifications", None)
        if notifications is not None:
            notifications.warning(message, duration=duration)
    except Exception:
        pass


def _notify_error(message: str, duration: int = 3000) -> None:
    try:
        notifications = getattr(pn.state, "notifications", None)
        if notifications is not None:
            notifications.error(message, duration=duration)
    except Exception:
        pass


def _invalidate_caches() -> None:
    stack_cache["key"] = None
    stack_cache["data"] = None
    trap_cache["key"] = None
    trap_cache["data"] = None
    hist_panel_cache["key"] = None
    hist_panel_cache["panel"] = None
    section_line_cache["key"] = None
    section_line_cache["value"] = (np.array([], dtype=float), np.array([], dtype=float))


def _sync_section_y_range() -> None:
    twt = np.asarray(surface_state["twt"].values, dtype=float)
    vel = np.asarray(surface_state["velocity"].values, dtype=float)
    depth = (twt * vel) / 2000.0
    depth_display = _depth_from_m(depth)
    d_min = float(np.nanmin(depth_display))
    d_max = float(np.nanmax(depth_display))
    span = max(1.0, abs(d_max - d_min))
    margin = 0.1 * span
    sld_y_range.start = int(np.floor(min(d_min, d_max) - margin))
    sld_y_range.end = int(np.ceil(max(d_min, d_max) + margin))
    sld_y_range.value = (int(np.floor(d_min - 0.05 * span)), int(np.ceil(d_max + 0.05 * span)))


def _refresh_unit_labels() -> None:
    sld_contours.name = f"Contours Range Number ({'milliseconds' if rg_time_units.value == 'milliseconds' else 'seconds'})"
    sld_input_depth_contours.name = f"Depth Contour Range Number ({_depth_unit_label()})"
    sld_c_inc.name = f"Contour Search Step ({_depth_unit_label()})"
    sld_depth_contours.name = f"Contours Range Number ({_depth_unit_label()})"
    sld_y_range.name = f"Depth Y-Axis Range ({_depth_unit_label()})"
    sld_elong_major.name = f"Elongated Major Axis Sigma ({_xy_unit_label()})"
    sld_elong_minor.name = f"Elongated Minor Axis Sigma ({_xy_unit_label()})"
    sld_twt_base.name = f"TWT Base ({_time_unit_label()})"
    sld_twt_amp.name = f"TWT Closure Amplitude ({_time_unit_label()})"
    sld_vel_base.name = f"Velocity Base ({rg_velocity_units.value})"
    sld_vel_amp.name = f"Velocity Dome Amplitude ({rg_velocity_units.value})"
    if "sld_thick_mean" in globals():
        sld_thick_mean.name = f"Thickness Mean ({_depth_unit_label()})"
    if "sld_thick_std" in globals():
        sld_thick_std.name = f"Thickness Std. Dev. ({_depth_unit_label()})"
    _apply_velocity_std_recommendation()


def _trigger_refresh() -> None:
    refresh_token.value = int(refresh_token.value) + 1


def _mark_general_settings_pending() -> None:
    _notify_info("General settings changed. Press 'Apply General Settings' to update plots.", duration=2800)


def _apply_general_settings(_event=None) -> None:
    _invalidate_caches()
    _bump_wells_refresh()
    _trigger_refresh()


def _on_velocity_units_change(event) -> None:
    old_to_mps = 1.0 if event.old == "m/s" else 0.3048
    new_to_mps = 1.0 if event.new == "m/s" else 0.3048
    current_mps = float(sld_std_dev.value) * old_to_mps
    sld_std_dev.value = current_mps / new_to_mps
    sld_vel_base.value = float(sld_vel_base.value) * old_to_mps / new_to_mps
    sld_vel_amp.value = float(sld_vel_amp.value) * old_to_mps / new_to_mps
    if "tbl_wells" in globals() and isinstance(tbl_wells.value, pd.DataFrame) and not tbl_wells.value.empty:
        df = tbl_wells.value.copy()
        if "Velocity" in df.columns:
            df["Velocity"] = pd.to_numeric(df["Velocity"], errors="coerce") * old_to_mps / new_to_mps
            tbl_wells.value = df
    applied_df = wells_applied_state.get("df")
    if isinstance(applied_df, pd.DataFrame) and not applied_df.empty and "Velocity" in applied_df.columns:
        df_applied = applied_df.copy()
        df_applied["Velocity"] = pd.to_numeric(df_applied["Velocity"], errors="coerce") * old_to_mps / new_to_mps
        wells_applied_state["df"] = _sanitize_wells_dataframe(df_applied)
    _refresh_unit_labels()
    _mark_general_settings_pending()


def _on_xy_units_change(event) -> None:
    old_factor = 1.0 if event.old == "meters" else 3.280839895013123
    new_factor = 1.0 if event.new == "meters" else 3.280839895013123
    sld_elong_major.value = float(sld_elong_major.value) * old_factor / new_factor
    sld_elong_minor.value = float(sld_elong_minor.value) * old_factor / new_factor
    if "tbl_wells" in globals() and isinstance(tbl_wells.value, pd.DataFrame) and not tbl_wells.value.empty:
        df = tbl_wells.value.copy()
        if "X" in df.columns:
            df["X"] = pd.to_numeric(df["X"], errors="coerce") * old_factor / new_factor
        if "Y" in df.columns:
            df["Y"] = pd.to_numeric(df["Y"], errors="coerce") * old_factor / new_factor
        tbl_wells.value = df
    applied_df = wells_applied_state.get("df")
    if isinstance(applied_df, pd.DataFrame) and not applied_df.empty:
        df_applied = applied_df.copy()
        if "X" in df_applied.columns:
            df_applied["X"] = pd.to_numeric(df_applied["X"], errors="coerce") * old_factor / new_factor
        if "Y" in df_applied.columns:
            df_applied["Y"] = pd.to_numeric(df_applied["Y"], errors="coerce") * old_factor / new_factor
        wells_applied_state["df"] = _sanitize_wells_dataframe(df_applied)
    _refresh_unit_labels()
    _mark_general_settings_pending()


def _on_depth_units_change(event) -> None:
    old_to_m = 1.0 if event.old == "meters" else 0.3048
    new_to_m = 1.0 if event.new == "meters" else 0.3048

    sld_c_inc.value = float(sld_c_inc.value) * old_to_m / new_to_m
    sld_input_depth_contours.value = float(sld_input_depth_contours.value) * old_to_m / new_to_m
    sld_thick_mean.value = float(sld_thick_mean.value) * old_to_m / new_to_m
    sld_thick_std.value = float(sld_thick_std.value) * old_to_m / new_to_m
    sld_depth_contours.value = float(sld_depth_contours.value) * old_to_m / new_to_m

    _sync_section_y_range()
    _refresh_unit_labels()
    _mark_general_settings_pending()


def _on_time_units_change(event) -> None:
    old_factor = 1.0 if event.old == "milliseconds" else 1.0 / 1000.0
    new_factor = 1.0 if event.new == "milliseconds" else 1.0 / 1000.0
    sld_twt_base.value = float(sld_twt_base.value) * old_factor / new_factor
    sld_twt_amp.value = float(sld_twt_amp.value) * old_factor / new_factor
    _refresh_unit_labels()
    _mark_general_settings_pending()


def _on_generic_units_change(_event) -> None:
    _refresh_unit_labels()
    _mark_general_settings_pending()


def _on_depth_contour_toggle(event) -> None:
    sld_depth_contours.visible = bool(event.new)


def _disable_bokeh_grid(plot, _element) -> None:
    fig = getattr(plot, "state", None)
    if fig is None:
        return
    for grid in getattr(fig, "xgrid", []):
        grid.visible = False
    for grid in getattr(fig, "ygrid", []):
        grid.visible = False


def _surface_xy_bounds() -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
    x_coords = np.asarray(surface_state.get("x_coords", surface_state["twt"].x.values), dtype=float)
    y_coords = np.asarray(surface_state.get("y_coords", surface_state["twt"].y.values), dtype=float)
    x_min = float(np.nanmin(x_coords))
    x_max = float(np.nanmax(x_coords))
    y_min = float(np.nanmin(y_coords))
    y_max = float(np.nanmax(y_coords))
    return x_min, x_max, y_min, y_max, x_coords, y_coords


def _applied_section_angle() -> float:
    return float(surface_state.get("section_angle_applied", 0.0))


def _section_line_display_coords(angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    cache_key = (int(surface_state.get("token", 0)), round(float(angle_deg), 4), str(rg_xy_units.value))
    if section_line_cache.get("key") == cache_key:
        cached_x, cached_y = section_line_cache.get("value", (np.array([], dtype=float), np.array([], dtype=float)))
        return np.asarray(cached_x, dtype=float), np.asarray(cached_y, dtype=float)

    x_coords_m = np.asarray(surface_state.get("x_coords", surface_state["twt"].x.values), dtype=float)
    y_coords_m = np.asarray(surface_state.get("y_coords", surface_state["twt"].y.values), dtype=float)
    xs_line_m, ys_line_m = core_line_through_center(x_coords_m, y_coords_m, float(angle_deg))
    xs_line = _xy_from_m(xs_line_m)
    ys_line = _xy_from_m(ys_line_m)
    section_line_cache["key"] = cache_key
    section_line_cache["value"] = (xs_line, ys_line)
    return xs_line, ys_line


def _project_wells_to_section_distance_m(
    well_x_m: np.ndarray,
    well_y_m: np.ndarray,
    x_values_m: np.ndarray,
    y_values_m: np.ndarray,
    angle_deg: float,
    section_distance_m: np.ndarray,
) -> np.ndarray:
    if well_x_m.size == 0 or section_distance_m.size == 0:
        return np.array([], dtype=float)

    center_x = float(np.mean(x_values_m))
    center_y = float(np.mean(y_values_m))
    x_span = float(np.max(x_values_m) - np.min(x_values_m))
    y_span = float(np.max(y_values_m) - np.min(y_values_m))
    half_len = 0.5 * np.sqrt(x_span**2 + y_span**2)
    phi = np.deg2rad((angle_deg + 90.0) % 360.0)
    if half_len <= 0.0:
        return np.full(well_x_m.size, float(section_distance_m[0]), dtype=float)

    t_vals = (well_x_m - center_x) * np.cos(phi) + (well_y_m - center_y) * np.sin(phi)
    t_samples = np.linspace(-half_len, half_len, int(section_distance_m.size))
    return np.interp(t_vals, t_samples, np.asarray(section_distance_m, dtype=float))


def _velocity_unit_to_mps() -> float:
    return 1.0 if rg_velocity_units.value == "m/s" else 0.3048


def _velocity_from_mps(values) -> np.ndarray:
    return np.asarray(values, dtype=float) / _velocity_unit_to_mps()


def _xy_from_m(values) -> np.ndarray:
    factor = 1.0 if rg_xy_units.value == "meters" else 3.280839895013123
    return np.asarray(values, dtype=float) * factor


def _depth_unit_to_m() -> float:
    return 1.0 if rg_depth_units.value == "meters" else 0.3048


def _depth_from_m(values) -> np.ndarray:
    return np.asarray(values, dtype=float) / _depth_unit_to_m()


def _time_from_ms(values) -> np.ndarray:
    return np.asarray(values, dtype=float) if rg_time_units.value == "milliseconds" else np.asarray(values, dtype=float) / 1000.0


def _area_from_m2(values) -> np.ndarray:
    factor = 1.0 if rg_area_units.value == "m2" else 10.763910416709722
    return np.asarray(values, dtype=float) * factor


def _volume_from_m3(values) -> np.ndarray:
    factor = 1.0 if rg_volume_units.value == "m3" else 35.31466672148859
    return np.asarray(values, dtype=float) * factor


def _xy_unit_label() -> str:
    return "m" if rg_xy_units.value == "meters" else "ft"


def _depth_unit_label() -> str:
    return "m" if rg_depth_units.value == "meters" else "ft"


def _time_unit_label() -> str:
    return "ms" if rg_time_units.value == "milliseconds" else "s"


def _sanitize_wells_dataframe(df_input: pd.DataFrame | None) -> pd.DataFrame:
    if df_input is None or not isinstance(df_input, pd.DataFrame) or df_input.empty:
        return pd.DataFrame(columns=["Well Name", "X", "Y", "Velocity"])

    df = df_input.copy()
    if "Well Name" not in df.columns:
        df.insert(0, "Well Name", [f"Well_{i+1}" for i in range(len(df))])

    for col in ["X", "Y", "Velocity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Well Name"] = df["Well Name"].astype(str).str.strip()
    df = df.dropna(subset=["X", "Y", "Velocity"], how="any").reset_index(drop=True)
    if len(df) > 0:
        generated = [f"Well_{i+1}" for i in range(len(df))]
        df["Well Name"] = [name if name else generated[i] for i, name in enumerate(df["Well Name"].tolist())]

    return df[["Well Name", "X", "Y", "Velocity"]]


def _pending_wells_dataframe_clean() -> pd.DataFrame:
    if "tbl_wells" not in globals() or not isinstance(tbl_wells.value, pd.DataFrame):
        return pd.DataFrame(columns=["Well Name", "X", "Y", "Velocity"])
    return _sanitize_wells_dataframe(tbl_wells.value)


def _wells_dataframe_clean() -> pd.DataFrame:
    return _sanitize_wells_dataframe(wells_applied_state.get("df"))


def _well_arrays_internal() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = _wells_dataframe_clean()
    if df.empty:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)

    x_vals = df["X"].to_numpy(dtype=float)
    y_vals = df["Y"].to_numpy(dtype=float)
    vel_vals = df["Velocity"].to_numpy(dtype=float)

    x_factor = 1.0 if rg_xy_units.value == "meters" else 3.280839895013123
    vel_factor = _velocity_unit_to_mps()
    return x_vals / x_factor, y_vals / x_factor, vel_vals * vel_factor


def _well_section_anchor_xy_m() -> tuple[np.ndarray, np.ndarray]:
    well_x_m, well_y_m, _ = _well_arrays_internal()
    if well_x_m.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    if not bool(chk_use_wells.value):
        return well_x_m, well_y_m

    x_coords = np.asarray(surface_state.get("x_coords", surface_state["twt"].x.values), dtype=float)
    y_coords = np.asarray(surface_state.get("y_coords", surface_state["twt"].y.values), dtype=float)

    if x_coords.size < 2 or y_coords.size < 2:
        nearest_idx_x = np.array([int(np.argmin(np.abs(x_coords - wx))) for wx in well_x_m], dtype=int)
        nearest_idx_y = np.array([int(np.argmin(np.abs(y_coords - wy))) for wy in well_y_m], dtype=int)
    else:
        x_inc = bool(x_coords[-1] >= x_coords[0])
        y_inc = bool(y_coords[-1] >= y_coords[0])
        x_search = x_coords if x_inc else x_coords[::-1]
        y_search = y_coords if y_inc else y_coords[::-1]

        x_pos = np.searchsorted(x_search, well_x_m)
        y_pos = np.searchsorted(y_search, well_y_m)
        x_pos = np.clip(x_pos, 1, len(x_search) - 1)
        y_pos = np.clip(y_pos, 1, len(y_search) - 1)

        x_left = x_search[x_pos - 1]
        x_right = x_search[x_pos]
        y_left = y_search[y_pos - 1]
        y_right = y_search[y_pos]

        nearest_idx_x = np.where(np.abs(well_x_m - x_left) <= np.abs(well_x_m - x_right), x_pos - 1, x_pos)
        nearest_idx_y = np.where(np.abs(well_y_m - y_left) <= np.abs(well_y_m - y_right), y_pos - 1, y_pos)

        if not x_inc:
            nearest_idx_x = (len(x_coords) - 1) - nearest_idx_x
        if not y_inc:
            nearest_idx_y = (len(y_coords) - 1) - nearest_idx_y

        nearest_idx_x = nearest_idx_x.astype(int)
        nearest_idx_y = nearest_idx_y.astype(int)

    return x_coords[nearest_idx_x], y_coords[nearest_idx_y]


def _well_names_internal() -> np.ndarray:
    df = _wells_dataframe_clean()
    if df.empty:
        return np.array([], dtype=object)
    return df["Well Name"].to_numpy(dtype=object)


def _well_arrays_display() -> tuple[np.ndarray, np.ndarray]:
    df = _wells_dataframe_clean()
    if df.empty:
        return np.array([], dtype=float), np.array([], dtype=float)
    return df["X"].to_numpy(dtype=float), df["Y"].to_numpy(dtype=float)


def _well_overlay_dataframe() -> pd.DataFrame:
    df = _wells_dataframe_clean().copy()
    if df.empty:
        return pd.DataFrame(columns=["x", "y", "well_name", "velocity"])
    df = df.rename(columns={"Well Name": "well_name", "X": "x", "Y": "y", "Velocity": "velocity"})
    return df[["x", "y", "well_name", "velocity"]]


def _covariance_values(h: np.ndarray, model: str, range_val: float, sill: float, nugget: float) -> np.ndarray:
    safe_range = max(float(range_val), 1e-6)
    safe_sill = max(float(sill), 1e-9)
    total_var = safe_sill + max(float(nugget), 0.0)
    if model == "Gaussian":
        cov = safe_sill * np.exp(-3.0 * (h / safe_range) ** 2)
    elif model == "Exponential":
        cov = safe_sill * np.exp(-3.0 * h / safe_range)
    else:
        ratio = h / safe_range
        cov = np.where(h <= safe_range, safe_sill * (1.0 - (1.5 * ratio - 0.5 * ratio**3)), 0.0)
    cov = np.where(h < 1e-6, total_var, cov)
    return cov


def _experimental_variogram_from_grid(
    grid_values: np.ndarray,
    x_coords_m: np.ndarray,
    y_coords_m: np.ndarray,
    max_pairs: int = 4000,
    n_bins: int = 18,
) -> pd.DataFrame:
    values = np.asarray(grid_values, dtype=float)
    xv, yv = np.meshgrid(np.asarray(x_coords_m, dtype=float), np.asarray(y_coords_m, dtype=float))
    pts = np.column_stack([xv.ravel(), yv.ravel(), values.ravel()])
    pts = pts[np.isfinite(pts[:, 2])]
    if len(pts) < 20:
        return pd.DataFrame(columns=["lag", "gamma", "pairs"])

    rng = np.random.default_rng(42)
    sample_n = min(len(pts), 900)
    idx = rng.choice(len(pts), size=sample_n, replace=False)
    s = pts[idx]

    i = rng.integers(0, sample_n, size=max_pairs)
    j = rng.integers(0, sample_n, size=max_pairs)
    mask = i != j
    i = i[mask]
    j = j[mask]
    dx = s[i, 0] - s[j, 0]
    dy = s[i, 1] - s[j, 1]
    h = np.sqrt(dx * dx + dy * dy)
    gamma = 0.5 * (s[i, 2] - s[j, 2]) ** 2

    if len(h) == 0 or not np.isfinite(h).any():
        return pd.DataFrame(columns=["lag", "gamma", "pairs"])

    h_max = float(np.nanpercentile(h, 90))
    h_max = max(h_max, 1.0)
    bins = np.linspace(0.0, h_max, n_bins + 1)
    out_rows: list[dict[str, float]] = []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        m = (h >= b0) & (h < b1)
        if int(np.sum(m)) < 8:
            continue
        out_rows.append({"lag": float(0.5 * (b0 + b1)), "gamma": float(np.nanmean(gamma[m])), "pairs": int(np.sum(m))})
    return pd.DataFrame(out_rows)


def _wells_covariance_dataframe() -> pd.DataFrame:
    x_w, y_w, _ = _well_arrays_internal()
    if x_w.size == 0:
        return pd.DataFrame(columns=["Well"])
    dist = np.sqrt((x_w[:, None] - x_w[None, :]) ** 2 + (y_w[:, None] - y_w[None, :]) ** 2)
    cov = _covariance_values(dist, sld_slope.value, float(sld_range.value), float(sld_sill.value), float(sld_nugget.value))
    names = [str(n) for n in _well_names_internal().tolist()]
    labels: list[str] = []
    counts: dict[str, int] = {}
    for idx in range(x_w.size):
        base = names[idx] if idx < len(names) and names[idx] else f"Well_{idx+1}"
        counts[base] = counts.get(base, 0) + 1
        labels.append(base if counts[base] == 1 else f"{base}_{counts[base]}")
    return pd.DataFrame(cov, index=labels, columns=labels)


def _depth_step_m() -> float:
    return float(sld_c_inc.value) * _depth_unit_to_m()


def _fallback_velocity_base_mps() -> float:
    twt_vals = np.asarray(surface_state["twt"].values, dtype=float)
    depth_vals = np.asarray(surface_state.get("depth_input", surface_state["twt"]).values, dtype=float)
    safe_twt = np.where(np.abs(twt_vals) > 1e-6, twt_vals, np.nan)
    vel_est = (depth_vals / safe_twt) * 2000.0
    if np.isfinite(vel_est).any():
        return float(np.clip(np.nanmedian(vel_est), 500.0, 8000.0))
    return 3000.0


def _volumetric_series(trap_stats: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arr_thick = _depth_from_m(np.asarray(trap_stats["thickness_total"]))
    arr_area = _area_from_m2(np.asarray(trap_stats["areas"]))
    arr_grv = _volume_from_m3(np.asarray(trap_stats["grv"]))
    arr_stooip = (
        np.asarray(trap_stats["grv"])
        * sld_ntg.value
        * sld_poro.value
        * (1.0 - sld_sw.value)
        / sld_fvf.value
        * 6.2898
        / 1e6
    )
    return arr_thick, arr_area, arr_grv, arr_stooip


def _build_geostat_qc_bundle_payload(
    velocity_stack: np.ndarray,
    depth_stack: np.ndarray,
    avg_velocity_map: np.ndarray,
    final_depth_map: np.ndarray,
    trap_stats: dict[str, np.ndarray],
) -> dict[str, pd.DataFrame]:
    twt_da = surface_state["twt"]
    vel_da = surface_state["velocity"]
    x_coords = np.asarray(surface_state.get("x_coords", twt_da.x.values), dtype=float)
    y_coords = np.asarray(surface_state.get("y_coords", twt_da.y.values), dtype=float)

    return core_build_geostat_qc_bundle(
        twt_ms=np.asarray(twt_da.values, dtype=float),
        base_velocity=np.asarray(vel_da.values, dtype=float),
        velocity_stack=np.asarray(velocity_stack, dtype=float),
        depth_stack=np.asarray(depth_stack, dtype=float),
        final_depth_map=np.asarray(final_depth_map, dtype=float),
        trap_masks=np.asarray(trap_stats["trap_masks"], dtype=bool),
        x_coords=x_coords,
        y_coords=y_coords,
        wells_cov_df=_wells_covariance_dataframe(),
        model=str(sld_slope.value),
        range_val=float(sld_range.value),
        sill=float(sld_sill.value),
        nugget=float(sld_nugget.value),
    )


def _recommend_velocity_std_from_surface() -> dict[str, float]:
    twt = np.asarray(surface_state["twt"].values, dtype=float)
    x_min, x_max, y_min, y_max, _, _ = _surface_xy_bounds()

    x_span = max(1.0, x_max - x_min)
    y_span = max(1.0, y_max - y_min)
    map_diag = float(np.hypot(x_span, y_span))

    twt_min = float(np.nanmin(twt))
    twt_max = float(np.nanmax(twt))
    twt_range = max(1.0, twt_max - twt_min)
    abs_twt_median = float(np.nanmedian(np.abs(twt)))
    abs_twt_median = max(abs_twt_median, 100.0)

    vel_ref = float(np.nanmean(np.asarray(surface_state["velocity"].values, dtype=float)))
    depth_relief = (twt_range * vel_ref) / 2000.0

    target_depth_std = float(np.clip(0.20 * depth_relief, 5.0, 120.0))
    suggested_vel_std = float(np.clip(target_depth_std * 2000.0 / abs_twt_median, 5.0, 250.0))
    upper_reasonable_vel_std = MAX_VELOCITY_STD_FPS * 0.3048

    return {
        "x_span": x_span,
        "y_span": y_span,
        "map_diag": map_diag,
        "twt_min": twt_min,
        "twt_max": twt_max,
        "twt_range": twt_range,
        "depth_relief": depth_relief,
        "suggested_vel_std": suggested_vel_std,
        "upper_reasonable_vel_std": upper_reasonable_vel_std,
    }


def _apply_velocity_std_recommendation() -> None:
    rec = _recommend_velocity_std_from_surface()
    if "sld_std_dev" not in globals():
        return
    std_max_display = (MAX_VELOCITY_STD_FPS * 0.3048) / _velocity_unit_to_mps()
    sld_std_dev.end = round(std_max_display, 3)
    sld_std_dev.name = f"Velocity Std. Dev. ({rg_velocity_units.value})"

    suggested = rec["suggested_vel_std"]
    upper_reasonable = rec["upper_reasonable_vel_std"]
    sill_value = float(sld_sill.value) if "sld_sill" in globals() else 1.0
    model_factor = float(np.clip(np.sqrt(max(sill_value, 1e-6)), 0.35, 1.6))
    auto_std_mps = float(np.clip(suggested * model_factor, 0.15, upper_reasonable))
    sld_std_dev.value = min(auto_std_mps / _velocity_unit_to_mps(), sld_std_dev.end)
    current_std_mps = float(sld_std_dev.value) * _velocity_unit_to_mps()

    logger.info(
        "Velocity std analysis | map_span=(%.1f x %.1f)m diag=%.1fm twt=[%.2f, %.2f] range=%.2fms depth_relief=%.2fm suggested_std=%.2f(m/s) max_reasonable=%.2f(m/s) current_std=%.2f(m/s)",
        rec["x_span"],
        rec["y_span"],
        rec["map_diag"],
        rec["twt_min"],
        rec["twt_max"],
        rec["twt_range"],
        rec["depth_relief"],
        suggested,
        upper_reasonable,
        current_std_mps,
    )


def _run_test_mode_diagnostics(trigger: str) -> None:
    if not TEST_MODE:
        return

    try:
        _, depth_stack, avg_velocity_map, final_depth_map = get_velocity_and_depth_stacks()
        depth_arr = np.asarray(depth_stack, dtype=float)
        final_arr = np.asarray(final_depth_map, dtype=float)
        av_arr = np.asarray(avg_velocity_map, dtype=float)

        finite_depth_pct = float(np.isfinite(depth_arr).mean() * 100.0)
        finite_final_pct = float(np.isfinite(final_arr).mean() * 100.0)
        final_range = float(np.nanmax(final_arr) - np.nanmin(final_arr)) if np.isfinite(final_arr).any() else 0.0

        logger.info(
            "TEST_DIAG[%s] n_maps=%d std=%.2f smooth=%.2f finite_depth=%.2f%% finite_final=%.2f%% final_range=%.3f av=[%.2f, %.2f]",
            trigger,
            int(depth_arr.shape[0]),
            float(sld_std_dev.value),
            float(sld_smooth.value),
            finite_depth_pct,
            finite_final_pct,
            final_range,
            float(np.nanmin(av_arr)),
            float(np.nanmax(av_arr)),
        )

        if finite_final_pct < 99.0 or final_range <= 0.0:
            _notify_warning(
                "Test-mode diagnostic: Final depth map has low finite coverage or zero range. Check velocity std/smoothing settings.",
                duration=5000,
            )
    except Exception:
        logger.exception("TEST_DIAG[%s] failed", trigger)


def _stack_key() -> tuple:
    return (
        int(surface_state["token"]),
        int(wells_refresh_token.value),
        sld_n_maps.value,
        sld_slope.value,
        round(float(sld_range.value), 3),
        round(float(sld_sill.value), 6),
        round(float(sld_nugget.value), 6),
        round(float(sld_smooth.value), 3),
        round(float(sld_std_dev.value), 4),
        rg_velocity_units.value,
        rg_depth_units.value,
        rg_time_units.value,
        rg_xy_units.value,
        bool(chk_use_wells.value),
        bool(chk_torch_accel.value),
    )


def _trap_key(stack_key: tuple) -> tuple:
    return (
        stack_key,
        round(float(sld_c_inc.value), 3),
        round(float(sld_thick_mean.value), 3),
        round(float(sld_thick_std.value), 3),
        rg_depth_units.value,
        rg_area_units.value,
        rg_volume_units.value,
    )


def _apply_surface_generation(_event=None) -> None:
    time_surface, depth_surface = _parse_imported_surfaces()
    if time_surface is None:
        _notify_warning("Time surface not found; update aborted.", duration=3500)
        return

    x_coords, y_coords, twt_ms = time_surface
    if depth_surface is not None and len(depth_surface[0]) == len(x_coords) and len(depth_surface[1]) == len(y_coords):
        depth_map_m = _normalize_imported_depth_to_m(depth_surface[2], twt_ms)
    else:
        depth_resampled = _resample_depth_to_time_grid(depth_surface, x_coords, y_coords)
        depth_map_m = _normalize_imported_depth_to_m(depth_resampled, twt_ms) if depth_resampled is not None else (np.asarray(twt_ms, dtype=float) * _fallback_velocity_base_mps()) / 2000.0

    safe_twt = np.where(np.abs(np.asarray(twt_ms, dtype=float)) > 1e-6, np.asarray(twt_ms, dtype=float), np.nan)
    vel = np.where(np.isfinite(safe_twt), (np.asarray(depth_map_m, dtype=float) / safe_twt) * 2000.0, np.nan)
    vel = np.nan_to_num(vel, nan=_fallback_velocity_base_mps())
    vel = np.maximum(vel, 500.0).astype(np.float32)

    # Calculate dx, dy, and cell_area for imported or synthetic grid
    dx = float(np.mean(np.diff(x_coords))) if len(x_coords) > 1 else 1.0
    dy = float(np.mean(np.diff(y_coords))) if len(y_coords) > 1 else 1.0
    cell_area = dx * dy

    surface_state["mode"] = "Imported Time + Depth"
    surface_state["token"] = int(surface_state["token"]) + 1
    surface_state["twt"] = xr.DataArray(np.asarray(twt_ms, dtype=np.float32), coords=[y_coords, x_coords], dims=["y", "x"], name="TWT")
    surface_state["depth_input"] = xr.DataArray(np.asarray(depth_map_m, dtype=np.float32), coords=[y_coords, x_coords], dims=["y", "x"], name="DepthInput")
    surface_state["velocity"] = xr.DataArray(vel, coords=[y_coords, x_coords], dims=["y", "x"], name="Velocity")
    surface_state["x_coords"] = x_coords
    surface_state["y_coords"] = y_coords
    surface_state["cell_area"] = cell_area
    surface_state["dx"] = dx
    surface_state["dy"] = dy
    _apply_velocity_std_recommendation()

    _invalidate_caches()
    _sync_section_y_range()
    _set_progress(0)
    _trigger_refresh()
    x_min, x_max, y_min, y_max, _, _ = _surface_xy_bounds()
    logger.info(
        "Surface regenerated | mode=%s token=%s cell_area=%.2f x=[%.3f, %.3f] y=[%.3f, %.3f]",
        surface_state["mode"],
        surface_state["token"],
        cell_area,
        x_min,
        x_max,
        y_min,
        y_max,
    )

    _run_test_mode_diagnostics("surface_generate")


def _on_surface_mode_change(event) -> None:
    return


def _on_depth_update_click(_event) -> None:
    progress_monte_carlo.visible = True
    _set_progress(2)
    surface_state["section_angle_applied"] = float(sld_section_angle.value)
    _invalidate_caches()
    try:
        _, depth_stack, _, _ = get_velocity_and_depth_stacks()
        _ = get_trap_stats(depth_stack)
    except Exception:
        logger.exception("Velocity/depth update failed")
        _notify_error("Velocity/depth update failed. Check logs.", duration=5000)
    finally:
        _set_progress(100)
        time.sleep(0.2)
        progress_monte_carlo.visible = False
        _set_progress(0)
    _trigger_refresh()
    _run_test_mode_diagnostics("depth_update")


def get_velocity_and_depth_stacks() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    key = _stack_key()
    if stack_cache["key"] == key and stack_cache["data"] is not None:
        return stack_cache["data"]

    _set_progress(1)
    base_twt_da = surface_state["twt"]
    base_vel_da = surface_state["velocity"]
    x_coords = np.asarray(surface_state.get("x_coords", base_twt_da.x.values), dtype=float)
    y_coords = np.asarray(surface_state.get("y_coords", base_twt_da.y.values), dtype=float)
    dx = float(surface_state.get("dx", 100.0))
    dy = float(surface_state.get("dy", 100.0))

    def _sim_progress(done: int, total: int) -> None:
        _set_progress(5 + int(65 * done / max(total, 1)))

    well_x, well_y, well_velocity = _well_arrays_internal()
    if not bool(chk_use_wells.value):
        well_x = np.array([], dtype=float)
        well_y = np.array([], dtype=float)
        well_velocity = np.array([], dtype=float)

    result = core_generate_velocity_depth_stacks(
        twt_ms=base_twt_da.values,
        base_velocity=base_vel_da.values,
        x_coords=x_coords,
        y_coords=y_coords,
        n_maps=sld_n_maps.value,
        model=sld_slope.value,
        range_val=float(sld_range.value),
        sill=float(sld_sill.value),
        nugget=float(sld_nugget.value),
        smooth_sigma=float(sld_smooth.value),
        velocity_std=float(sld_std_dev.value) * _velocity_unit_to_mps(),
        well_x=well_x.tolist(),
        well_y=well_y.tolist(),
        well_velocity=well_velocity.tolist(),
        dx=dx,
        dy=dy,
        seed=42,
        use_torch=bool(chk_torch_accel.value),
        progress_cb=_sim_progress,
    )

    stack_cache["key"] = key
    stack_cache["data"] = result
    _set_progress(70)
    return result


def get_trap_stats(depth_stack: np.ndarray) -> dict[str, np.ndarray]:
    stack_key = _stack_key()
    key = _trap_key(stack_key)
    if trap_cache["key"] == key and trap_cache["data"] is not None:
        return trap_cache["data"]

    def _trap_progress(done: int, total: int) -> None:
        _set_progress(72 + int(28 * done / max(total, 1)))

    cell_area = float(surface_state.get("cell_area", 10000.0))
    trap_data = core_compute_trap_statistics(
        depth_stack,
        contour_step=_depth_step_m(),
        thickness_mean=float(sld_thick_mean.value) * _depth_unit_to_m(),
        thickness_std=float(sld_thick_std.value) * _depth_unit_to_m(),
        cell_area=cell_area,
        progress_cb=_trap_progress,
    )

    trap_cache["key"] = key
    trap_cache["data"] = trap_data
    _set_progress(100)
    return trap_data


def _build_report_payload() -> dict[str, np.ndarray | pd.DataFrame | float]:
    _, depth_stack, avg_velocity_map, final_depth_map = get_velocity_and_depth_stacks()
    selected_idx = min(max(int(sld_map_show.value) - 1, 0), depth_stack.shape[0] - 1)
    selected_depth_map = np.asarray(depth_stack[selected_idx], dtype=float)
    trap_stats = get_trap_stats(depth_stack)

    arr_thick, arr_area, arr_grv, arr_stooip = _volumetric_series(trap_stats)

    probability = trap_stats["trap_masks"].mean(axis=0) * 100.0

    summary_df = pd.DataFrame(
        {
            "Metric": [f"Thickness ({_depth_unit_label()})", f"Area ({rg_area_units.value})", f"GRV ({rg_volume_units.value})", "STOOIP (MMbbls)"],
            "P90": [np.percentile(arr_thick, 10), np.percentile(arr_area, 10), np.percentile(arr_grv, 10), np.percentile(arr_stooip, 10)],
            "P50": [np.percentile(arr_thick, 50), np.percentile(arr_area, 50), np.percentile(arr_grv, 50), np.percentile(arr_stooip, 50)],
            "P10": [np.percentile(arr_thick, 90), np.percentile(arr_area, 90), np.percentile(arr_grv, 90), np.percentile(arr_stooip, 90)],
            "Mean": [np.mean(arr_thick), np.mean(arr_area), np.mean(arr_grv), np.mean(arr_stooip)],
        }
    )

    realization_df = pd.DataFrame(
        {
            "Realization": np.arange(1, depth_stack.shape[0] + 1),
            f"SpillDepth({_depth_unit_label()})": _depth_from_m(trap_stats["spill_depths"]),
            f"CrestDepth({_depth_unit_label()})": _depth_from_m(trap_stats["crest_depths"]),
            f"Thickness({_depth_unit_label()})": arr_thick,
            f"Area({rg_area_units.value})": arr_area,
            f"GRV({rg_volume_units.value})": arr_grv,
            "STOOIP(MMbbls)": arr_stooip,
        }
    )

    return {
        "depth_stack": depth_stack,
        "avg_velocity_map": avg_velocity_map,
        "final_depth_map": final_depth_map,
        "selected_idx": selected_idx,
        "selected_depth_map": selected_depth_map,
        "trap_stats": trap_stats,
        "probability": probability,
        "arr_thick": arr_thick,
        "arr_area": arr_area,
        "arr_grv": arr_grv,
        "arr_stooip": arr_stooip,
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
    selected_idx = int(payload.get("selected_idx", 0))
    selected_depth_map = np.asarray(payload.get("selected_depth_map", final_depth_map), dtype=float)
    probability = payload["probability"]
    arr_thick = payload["arr_thick"]
    arr_area = payload["arr_area"]
    arr_grv = payload["arr_grv"]
    arr_stooip = payload["arr_stooip"]

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(report_title, fontsize=15, fontweight="bold")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.0], height_ratios=[1.0, 1.0], wspace=0.62, hspace=0.34)

    base_twt = surface_state["twt"]
    x_coords = surface_state.get("x_coords", base_twt.x.values)
    y_coords = surface_state.get("y_coords", base_twt.y.values)
    well_x_m, well_y_m, _ = _well_arrays_internal()
    well_names = [str(name) for name in _well_names_internal().tolist()]
    extent = [x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()]

    ax_twt = fig.add_subplot(gs[0, 0])
    twt_img = ax_twt.imshow(base_twt.values, origin="lower", extent=extent, cmap=sel_cmap.value, aspect="auto")
    ax_twt.contour(x_coords, y_coords, base_twt.values, levels=sld_contours.value, colors="black", alpha=0.35, linewidths=0.7)
    xs_line, ys_line = core_line_through_center(x_coords, y_coords, sld_section_angle.value)
    ax_twt.plot(xs_line, ys_line, color="red", linestyle="--", linewidth=1.4)
    if well_x_m.size > 0:
        ax_twt.scatter(well_x_m, well_y_m, c="yellow", edgecolors="black", s=36, zorder=8)
        for idx, (wx, wy) in enumerate(zip(well_x_m, well_y_m)):
            label = well_names[idx] if idx < len(well_names) else f"Well_{idx+1}"
            ax_twt.annotate(label, (wx, wy), xytext=(10, 8), textcoords="offset points", color="yellow", fontsize=9, fontweight="bold")
    ax_twt.set_xlim(float(extent[0]), float(extent[1]))
    ax_twt.set_ylim(float(extent[2]), float(extent[3]))
    ax_twt.margins(x=0, y=0)
    ax_twt.set_aspect("auto")
    ax_twt.set_title("Input Deterministic TWT Map (ms)", fontsize=10, fontweight="bold")
    ax_twt.set_xlabel("x")
    ax_twt.set_ylabel("y")
    cb_twt = fig.colorbar(twt_img, ax=ax_twt, fraction=0.046, pad=0.02)
    cb_twt.set_label(f"Time [{_time_unit_label()}]")

    ax_section = fig.add_subplot(gs[0, 1])
    section_x, final_section, section_stack = core_extract_section_stack(
        depth_stack,
        final_depth_map,
        x_coords,
        y_coords,
        sld_section_angle.value,
    )
    for row in section_stack:
        ax_section.plot(section_x, row, color="red", alpha=0.2, linewidth=0.7)
    ax_section.plot(section_x, final_section, color="black", linewidth=2.2, label=f"Final Depth [{_depth_unit_label()}] from Mean AV")
    section_std = np.nanstd(section_stack, axis=0)
    ax_section.plot(section_x, final_section + section_std, color="black", linestyle="--", linewidth=1.4, alpha=0.9, label="Std +1sigma (dashed black)")
    ax_section.plot(section_x, final_section - section_std, color="black", linestyle="--", linewidth=1.4, alpha=0.9, label="Std -1sigma (dashed black)")
    deterministic_spill, _, _ = core_get_trap_and_spill(final_depth_map, _depth_step_m())
    ax_section.axhline(
        deterministic_spill,
        color="blue",
        linestyle="--",
        linewidth=1.6,
        label=f"Spill depth (dashed blue): {float(_depth_from_m([deterministic_spill])[0]):.2f} {_depth_unit_label()}",
    )
    if well_x_m.size > 0:
        well_anchor_x_m, well_anchor_y_m = _well_section_anchor_xy_m()
        well_dist = _project_wells_to_section_distance_m(
            well_x_m=well_anchor_x_m,
            well_y_m=well_anchor_y_m,
            x_values_m=np.asarray(x_coords, dtype=float),
            y_values_m=np.asarray(y_coords, dtype=float),
            angle_deg=_applied_section_angle(),
            section_distance_m=np.asarray(section_x, dtype=float),
        )
        well_depth = np.interp(well_dist, section_x, final_section)
        ax_section.scatter(well_dist, well_depth, c=MAGENTA_OMV_COLOR, edgecolors="black", s=40, zorder=9)
        for wx in well_dist:
            ax_section.axvline(float(wx), color=MAGENTA_OMV_COLOR, linestyle="--", linewidth=1.5, alpha=0.9)
        for idx, (wd, wz) in enumerate(zip(well_dist, well_depth)):
            label = well_names[idx] if idx < len(well_names) else f"Well_{idx+1}"
            ax_section.annotate(label, (wd, wz), xytext=(6, 4), textcoords="offset points", color=DARK_MAGENTA_OMV_COLOR, fontsize=9, fontweight="bold", ha="left")
    # Dynamically set Y-axis range for section based on section data
    section_min = float(np.nanmin(section_stack))
    section_max = float(np.nanmax(section_stack))
    margin = 0.05 * abs(section_max - section_min)
    is_negative_domain = bool(np.nanmax(section_stack) <= 0.0)
    if is_negative_domain:
        ax_section.set_ylim(section_min - margin, section_max + margin)
    else:
        ax_section.set_ylim(section_max + margin, section_min - margin)
    ax_section.set_title(f"Structural Section (Perpendicular, Angle={sld_section_angle.value}degree)", fontsize=10, fontweight="bold")
    ax_section.set_xlabel("Section Samples")
    ax_section.set_ylabel("Depth (m)")
    ax_section.yaxis.labelpad = 2
    if well_x_m.size > 0:
        ax_section.plot([], [], color=MAGENTA_OMV_COLOR, linestyle="--", linewidth=1.5, label="Well location (dashed magenta)")
    ax_section.legend(loc="upper right", fontsize=8)
    ax_section.grid(alpha=0.25)

    hist_gs = gs[:, 2].subgridspec(4, 1, hspace=0.45)
    _plot_hist_with_lines(fig.add_subplot(hist_gs[0, 0]), arr_thick, "Crest-to-Spill Thick.", f"Thickness ({_depth_unit_label()})", "#8888ff")
    _plot_hist_with_lines(fig.add_subplot(hist_gs[1, 0]), arr_area, "Closure Area", f"Area ({rg_area_units.value})", "#ff8888")
    _plot_hist_with_lines(fig.add_subplot(hist_gs[2, 0]), arr_grv, "GRV Dist.", f"GRV ({rg_volume_units.value})", "#88ff88")
    _plot_hist_with_lines(fig.add_subplot(hist_gs[3, 0]), arr_stooip, "STOOIP Dist.", "STOOIP (MMbbls)", "#ffaa00")

    ax_depth = fig.add_subplot(gs[1, 0])
    depth_img = ax_depth.imshow(selected_depth_map, origin="lower", extent=extent, cmap=sel_cmap.value, aspect="auto")
    ax_depth.plot(xs_line, ys_line, color="red", linestyle="--", linewidth=1.4)
    if well_x_m.size > 0:
        ax_depth.scatter(well_x_m, well_y_m, c="yellow", edgecolors="black", s=36, zorder=8)
        for idx, (wx, wy) in enumerate(zip(well_x_m, well_y_m)):
            label = well_names[idx] if idx < len(well_names) else f"Well_{idx+1}"
            ax_depth.annotate(label, (wx, wy), xytext=(10, 8), textcoords="offset points", color="yellow", fontsize=9, fontweight="bold")
    if chk_depth_contour.value:
        contour_interval_m = max(float(sld_depth_contours.value) * _depth_unit_to_m(), 0.1)
        depth_min = float(np.nanmin(selected_depth_map))
        depth_max = float(np.nanmax(selected_depth_map))
        if depth_max > depth_min:
            levels = np.arange(depth_min, depth_max, contour_interval_m)
            if len(levels) > 0:
                ax_depth.contour(x_coords, y_coords, selected_depth_map, levels=levels, colors="white", linewidths=0.6, alpha=0.75)

    try:
        ax_depth.contour(x_coords, y_coords, selected_depth_map, levels=[deterministic_spill], colors="red", linestyles="--", linewidths=1.8)
    except Exception:
        logger.warning("PDF depth spill contour skipped due to contour level/data mismatch")

    ax_depth.set_xlim(float(extent[0]), float(extent[1]))
    ax_depth.set_ylim(float(extent[2]), float(extent[3]))
    ax_depth.margins(x=0, y=0)
    ax_depth.set_aspect("auto")
    ax_depth.set_title(f"Selected Depth Realization #{selected_idx + 1}", fontsize=10, fontweight="bold")
    ax_depth.set_xlabel("x")
    ax_depth.set_ylabel("y")
    cb_depth = fig.colorbar(depth_img, ax=ax_depth, fraction=0.046, pad=0.02)
    cb_depth.set_label(f"Depth [{_depth_unit_label()}]")

    ax_prob = fig.add_subplot(gs[1, 1])
    prob_img = ax_prob.imshow(probability, origin="lower", extent=extent, cmap="rainbow", vmin=0, vmax=100, aspect="auto")
    ax_prob.contour(x_coords, y_coords, probability, levels=10, colors="white", alpha=0.65, linewidths=0.6)
    if well_x_m.size > 0:
        ax_prob.scatter(well_x_m, well_y_m, c="yellow", edgecolors="black", s=36, zorder=8)
        for idx, (wx, wy) in enumerate(zip(well_x_m, well_y_m)):
            label = well_names[idx] if idx < len(well_names) else f"Well_{idx+1}"
            ax_prob.annotate(label, (wx, wy), xytext=(10, 8), textcoords="offset points", color="yellow", fontsize=9, fontweight="bold")
    ax_prob.set_xlim(float(extent[0]), float(extent[1]))
    ax_prob.set_ylim(float(extent[2]), float(extent[3]))
    ax_prob.margins(x=0, y=0)
    ax_prob.set_aspect("auto")
    ax_prob.set_title(f"Isoprobability Map (Isolated Trap, N={depth_stack.shape[0]})", fontsize=10, fontweight="bold")
    ax_prob.set_xlabel("x")
    ax_prob.set_ylabel("y")
    cb_prob = fig.colorbar(prob_img, ax=ax_prob, fraction=0.046, pad=0.02)
    cb_prob.set_label("Probability [%]")

    return fig


def _build_pdf_parameters_page(report_title: str) -> plt.Figure:
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_subplot(111)
    ax.axis("off")

    well_x_m, well_y_m, well_v_mps = _well_arrays_internal()
    well_names = [str(name) for name in _well_names_internal().tolist()]
    cov_df = _wells_covariance_dataframe()
    cov_condition = float(np.linalg.cond(cov_df.to_numpy(dtype=float))) if not cov_df.empty else float("nan")
    cov_offdiag_mean = float("nan")
    if not cov_df.empty and cov_df.shape[0] > 1:
        cov_np = cov_df.to_numpy(dtype=float)
        tri = cov_np[np.triu_indices(cov_np.shape[0], k=1)]
        if tri.size > 0:
            cov_offdiag_mean = float(np.nanmean(tri))

    lines = [
        f"{report_title} - Parameters and Configuration",
        "",
        "Input & Realizations",
        f"- Surface: {sel_surface.value}",
        f"- Number of realizations: {sld_n_maps.value}",
        f"- Velocity Std. Dev. ({rg_velocity_units.value}): {sld_std_dev.value}",
        f"- Smoothing sigma: {sld_smooth.value}",
        f"- Section/Cut angle (degree): {sld_section_angle.value}",
        f"- Torch acceleration enabled: {chk_torch_accel.value}",
        f"- Contour step (spill search, {_depth_unit_label()}): {sld_c_inc.value}",
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
        "Wells & Conditioning",
        f"- Wells count: {len(well_x_m)}",
        f"- Wells used as anchors: {chk_use_wells.value}",
        f"- Show wells on maps: {chk_show_wells.value}",
    ]
    if len(well_x_m) > 0:
        lines.extend([
            f"- First well ({well_names[0] if well_names else 'Well_1'}) (x,y,v m/s): ({well_x_m[0]:.1f}, {well_y_m[0]:.1f}, {well_v_mps[0]:.2f})",
            f"- Well names: {', '.join(well_names[:8])}{' ...' if len(well_names) > 8 else ''}",
            f"- Covariance matrix condition #: {cov_condition:.3e}",
            f"- Covariance off-diagonal mean: {cov_offdiag_mean:.4f}",
        ])

    lines.extend([
        "",
        "QC Additions",
        "- Covariance matrix among wells exported to Excel",
        "- Well metadata and nearest-grid diagnostics exported to Excel",
    ])
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=11)

    return fig


# ==============================================================================
#  UI WIDGETS
# ==============================================================================
_default_surface_mode = "Imported Time + Depth"
_surface_mode_hint = pn.pane.Markdown(
    "",
    visible=False,
)

txt_input_time_surface = pn.pane.Markdown(
    f"**Time Surface:** {TEST_SURFACE_TIME_FILE if TEST_MODE else 'Imported from payload'}",
    sizing_mode="stretch_width",
)
txt_input_depth_surface = pn.pane.Markdown(
    f"**Depth Surface:** {TEST_SURFACE_DEPTH_FILE if TEST_MODE else 'Imported from payload'}",
    sizing_mode="stretch_width",
)

sel_surface = pn.widgets.Select(name="Select Surface", options=["Imported Time + Depth"], value=_default_surface_mode, stylesheets=select_stylesheets, visible=False)
sld_elong_major = pn.widgets.FloatSlider(
    name="Elongated Major Axis Sigma (m)",
    start=500,
    end=16000,
    value=9186,
    step=100,
    visible=False,
    stylesheets=slider_stylesheets,
)
sld_elong_minor = pn.widgets.FloatSlider(
    name="Elongated Minor Axis Sigma (m)",
    start=300,
    end=9000,
    value=3937,
    step=50,
    visible=False,
    stylesheets=slider_stylesheets,
)
sld_elong_rotation = pn.widgets.FloatSlider(
    name="Elongated Azimuth (degree)",
    start=0,
    end=180,
    value=25,
    step=1,
    visible=False,
    stylesheets=slider_stylesheets,
)
sld_twt_base = pn.widgets.FloatSlider(
    name="TWT Base (ms)",
    start=-3500,
    end=-800,
    value=-1850,
    step=25,
    visible=False,
    stylesheets=slider_stylesheets,
)
sld_twt_amp = pn.widgets.FloatSlider(
    name="TWT Closure Amplitude (ms)",
    start=20,
    end=900,
    value=260,
    step=10,
    visible=False,
    stylesheets=slider_stylesheets,
)
sld_vel_base = pn.widgets.FloatSlider(
    name="Velocity Base (ft/s)",
    start=7000,
    end=14000,
    value=9843,
    step=25,
    visible=False,
    stylesheets=slider_stylesheets,
)
sld_vel_amp = pn.widgets.FloatSlider(
    name="Velocity Dome Amplitude (ft/s)",
    start=0,
    end=1640,
    value=394,
    step=5,
    visible=False,
    stylesheets=slider_stylesheets,
)
sld_contours = pn.widgets.IntSlider(
    name="Contours Range Number (milliseconds)",
    start=5,
    end=50,
    value=20,
    stylesheets=slider_stylesheets,
)
sld_input_depth_contours = pn.widgets.FloatSlider(
    name="Depth Contour Range Number (ft)",
    start=10,
    end=1000,
    value=100,
    step=10,
    stylesheets=slider_stylesheets,
)
sel_cmap = pn.widgets.Select(
    name="Colormap",
    options=["viridis", "plasma", "terrain", "rainbow"],
    value="terrain",
    stylesheets=select_stylesheets,
)
btn_generate_surf = pn.widgets.Button(
    name="Update TWT/Depth Inputs",
    css_classes=["omv-run-btn"],
    stylesheets=get_neon_button_stylesheets(),
    sizing_mode="stretch_width",
)

btn_add_well = pn.widgets.Button(
    name="+ Add Well",
    css_classes=["omv-run-btn"],
    stylesheets=get_neon_button_stylesheets(),
    sizing_mode="stretch_width",
)
btn_update_well_location = pn.widgets.Button(
    name="Update Well Location",
    css_classes=["omv-run-btn"],
    stylesheets=get_neon_button_stylesheets(),
    sizing_mode="stretch_width",
)
chk_show_wells = pn.widgets.Checkbox(name="Show wells on maps", value=True)
chk_use_wells = pn.widgets.Checkbox(name="Use wells as velocity anchors", value=True)
btn_remove_well = pn.widgets.Button(
    name="Remove selected row",
    button_type="danger",
    sizing_mode="stretch_width",
)
tbl_wells = pn.widgets.Tabulator(
    pd.DataFrame(columns=["Well Name", "X", "Y", "Velocity"]),
    show_index=False,
    selectable=1,
    height=170,
    sizing_mode="stretch_width",
)


def _bump_wells_refresh() -> None:
    wells_refresh_token.value = int(wells_refresh_token.value) + 1


def _default_well_values() -> tuple[str, float, float, float]:
    x_min, x_max, y_min, y_max, _, _ = _surface_xy_bounds()
    x_mid = float(_xy_from_m([(x_min + x_max) / 2.0])[0])
    y_mid = float(_xy_from_m([(y_min + y_max) / 2.0])[0])
    v_mid = float(_velocity_from_mps([np.nanmean(np.asarray(surface_state["velocity"].values, dtype=float))])[0])
    v_mid = round(v_mid, 1)
    pending_df = _pending_wells_dataframe_clean()
    next_name = f"Well_{len(pending_df) + 1}"
    return next_name, x_mid, y_mid, v_mid


def _add_well_row(_event=None) -> None:
    well_name, x_mid, y_mid, v_mid = _default_well_values()
    df = tbl_wells.value.copy()
    df.loc[len(df)] = {"Well Name": well_name, "X": x_mid, "Y": y_mid, "Velocity": v_mid}
    tbl_wells.value = df


def _remove_selected_well_row(_event=None) -> None:
    df = tbl_wells.value.copy()
    if df.empty:
        return
    if getattr(tbl_wells, "selection", None):
        idx = int(tbl_wells.selection[0])
        if 0 <= idx < len(df):
            df = df.drop(df.index[idx]).reset_index(drop=True)
    else:
        df = df.iloc[:-1].reset_index(drop=True)
    tbl_wells.value = df


def _on_update_well_location(_event) -> None:
    wells_applied_state["df"] = _pending_wells_dataframe_clean().copy()
    _invalidate_caches()
    _bump_wells_refresh()
    _set_progress(0)
    _trigger_refresh()


btn_add_well.on_click(_add_well_row)
btn_remove_well.on_click(_remove_selected_well_row)
btn_update_well_location.on_click(_on_update_well_location)

sld_c_inc = pn.widgets.FloatSlider(
    name="Contour Search Step (ft)",
    start=1,
    end=20,
    value=20,
    stylesheets=slider_stylesheets,
)
chk_eliminate = pn.widgets.Checkbox(name="Eliminate outside closure", value=False)
chk_close_poly = pn.widgets.Checkbox(name="Show closure contour (red)", value=True)
chk_depth_contour = pn.widgets.Checkbox(name="Contour Final Depth Map", value=True)
sld_depth_contours = pn.widgets.FloatSlider(
    name="Contours Range Number (ft)",
    start=0,
    end=200,
    value=100,
    step=1,
    stylesheets=slider_stylesheets,
)
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
    name="Velocity Std. Dev. (ft/s)",
    start=0.0,
    end=50.0,
    value=3.937,
    visible=False,
    stylesheets=slider_stylesheets,
)
sld_smooth = pn.widgets.FloatSlider(
    name="Smoothing (Sigma)",
    start=0.0,
    end=25.0,
    value=10.0,
    step=0.5,
    stylesheets=slider_stylesheets,
)
sld_y_range = pn.widgets.RangeSlider(
    name="Depth Y-Axis Range (ft)",
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
sld_section_angle = pn.widgets.FloatSlider(
    name="Section/Cut Angle (degree)",
    start=0,
    end=360,
    value=0,
    step=1,
    stylesheets=slider_stylesheets,
)
chk_torch_accel = pn.widgets.Checkbox(name="Use Torch acceleration (if available)", value=True)
btn_update_depth_maps = pn.widgets.Button(
    name="Update Velocity & Depth Realizations",
    css_classes=["omv-run-btn"],
    stylesheets=get_neon_button_stylesheets(),
    sizing_mode="stretch_width",
)
progress_monte_carlo = pn.widgets.Progress(
    name="Monte Carlo Progress",
    value=0,
    max=100,
    bar_color="primary",
    sizing_mode="stretch_width",
    visible=False,
)

rg_velocity_units = pn.widgets.RadioButtonGroup(
    name="",
    options=["ft/s", "m/s"],
    value="ft/s",
    button_type="default",
    stylesheets=radio_group_stylesheets,
    width=160,
)
rg_xy_units = pn.widgets.RadioButtonGroup(
    name="",
    options=["meters", "feet"],
    value="feet",
    button_type="default",
    stylesheets=radio_group_stylesheets,
    width=160,
)
rg_depth_units = pn.widgets.RadioButtonGroup(
    name="",
    options=["feet", "meters"],
    value="feet",
    button_type="default",
    stylesheets=radio_group_stylesheets,
    width=160,
)
rg_time_units = pn.widgets.RadioButtonGroup(
    name="",
    options=["seconds", "milliseconds"],
    value="milliseconds",
    button_type="default",
    stylesheets=radio_group_stylesheets,
    width=160,
)
rg_area_units = pn.widgets.RadioButtonGroup(
    name="",
    options=["m2", "ft2"],
    value="m2",
    button_type="default",
    stylesheets=radio_group_stylesheets,
    width=160,
)
rg_volume_units = pn.widgets.RadioButtonGroup(
    name="",
    options=["m3", "ft3"],
    value="m3",
    button_type="default",
    stylesheets=radio_group_stylesheets,
    width=160,
)
btn_apply_general_settings = pn.widgets.Button(
    name="Apply General Settings",
    css_classes=["omv-run-btn"],
    stylesheets=get_neon_button_stylesheets(),
    sizing_mode="stretch_width",
)


def _general_setting_row(label: str, widget) -> pn.Row:
    return pn.Row(
        pn.pane.HTML(f"<div style='margin:0; line-height:32px; padding-left:10px;'>{label}</div>", width=95, margin=(0, 0, 0, 0)),
        widget,
        align="center",
        sizing_mode="stretch_width",
        margin=(0, 0, 6, 0),
    )

_sync_section_y_range()
_refresh_unit_labels()

for _w in [sld_elong_major, sld_elong_minor, sld_elong_rotation, sld_twt_base, sld_twt_amp, sld_vel_amp]:
    _w.visible = bool(sel_surface.value == "Elongated / Ellipsoidal")


@pn.depends(sld_n_maps, watch=True)
def _update_map_slider(n_maps_val: int) -> None:
    sld_map_show.end = n_maps_val
    if sld_map_show.value > n_maps_val:
        sld_map_show.value = n_maps_val


sld_range = pn.widgets.FloatSlider(
    name="Range (m)",
    start=500,
    end=10000,
    value=DEFAULT_RANGE_M,
    stylesheets=slider_stylesheets,
)
sld_sill = pn.widgets.FloatSlider(
    name="Sill (Variance)",
    start=0.1,
    end=2.0,
    value=DEFAULT_SILL,
    stylesheets=slider_stylesheets,
)
sld_nugget = pn.widgets.FloatSlider(
    name="Nugget",
    start=0.0,
    end=0.5,
    value=DEFAULT_NUGGET,
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
    name="Porosity (phi)",
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

rg_velocity_units.param.watch(_on_velocity_units_change, "value")
rg_depth_units.param.watch(_on_depth_units_change, "value")
rg_xy_units.param.watch(_on_xy_units_change, "value")
rg_time_units.param.watch(_on_time_units_change, "value")
rg_area_units.param.watch(_on_generic_units_change, "value")
rg_volume_units.param.watch(_on_generic_units_change, "value")
chk_depth_contour.param.watch(_on_depth_contour_toggle, "value")
btn_generate_surf.on_click(_apply_surface_generation)
btn_update_depth_maps.on_click(_on_depth_update_click)
btn_update_vario.on_click(_on_depth_update_click)
btn_update_culm.on_click(_on_depth_update_click)
btn_update_volumetrics.on_click(_on_depth_update_click)
btn_apply_general_settings.on_click(_apply_general_settings)
chk_show_wells.param.watch(
    lambda _event: _notify_info("Press 'Update Well Location' to apply well visibility changes.", duration=2800),
    "value",
)
chk_use_wells.param.watch(
    lambda _event: _notify_info("Press 'Update Well Location' to apply well-anchor conditioning changes.", duration=2800),
    "value",
)

txt_report_header = pn.widgets.TextInput(name="Plot Header", value="Structural Uncertainty Evaluation")
txt_report_filename = pn.widgets.TextInput(name="File Name", value="structural_uncertainty_report")
txt_report_location = pn.widgets.TextInput(name="Location", value=str(APP_DIR / "reports"))
sel_report_type = pn.widgets.Select(name="Type", options=[".png", ".jpeg", ".pdf", ".xlsx"], value=".pdf", stylesheets=select_stylesheets)
btn_export_report = pn.widgets.Button(
    name="Generate Report",
    css_classes=["omv-run-btn"],
    stylesheets=get_neon_button_stylesheets(),
    sizing_mode="stretch_width",
)
report_status = pn.pane.Markdown("", visible=False)
qc_export_status = pn.pane.Markdown("", visible=False)
report_export_progress = pn.widgets.Progress(
    name="Export Progress",
    value=0,
    max=100,
    sizing_mode="stretch_width",
    visible=False,
)

txt_export_petrel_surface_name = pn.widgets.TextInput(
    name="Export Name Prefix",
    value="StructuralUncertainty",
    sizing_mode="stretch_width",
)
chk_export_average_mean = pn.widgets.Checkbox(name="Average Velocity Map", value=True)
chk_export_final_depth = pn.widgets.Checkbox(name="Final Depth", value=True)
chk_export_final_depth_contour = pn.widgets.Checkbox(name="Final Depth Contour Map", value=True)
chk_export_spill_contour_line = pn.widgets.Checkbox(name="Spill Contour Line (red dashed)", value=True)
chk_export_isoprobability = pn.widgets.Checkbox(name="Isoprobability", value=True)
chk_export_all_realizations = pn.widgets.Checkbox(name="All realization maps (N)", value=False)
btn_export_petrel_surface = pn.widgets.Button(
    name="Export Selected Surfaces to Petrel",
    css_classes=["omv-run-btn"],
    stylesheets=get_neon_button_stylesheets(),
    sizing_mode="stretch_width",
)
petrel_export_status = pn.pane.Markdown("", visible=False)


def _set_petrel_export_status(message: str) -> None:
    petrel_export_status.object = message
    petrel_export_status.visible = True


# ==============================================================================
#  PLOTTING FUNCTIONS
# ==============================================================================
@pn.depends(
    btn_generate_surf,
    btn_update_depth_maps,
    refresh_token,
    wells_refresh_token,
)
def plot_top_left_input_twt_map(*_args):
    logger.debug("Rendering input TWT map")
    global input_surface_tabs

    da = surface_state["twt"].copy()
    depth_input_da = surface_state.get("depth_input")
    if depth_input_da is None:
        depth_input_da = xr.DataArray((np.asarray(da.values, dtype=float) * np.asarray(surface_state["velocity"].values, dtype=float)) / 2000.0, coords=da.coords, dims=da.dims)
    x_min, x_max, y_min, y_max, x_coords_m, y_coords_m = _surface_xy_bounds()
    x_coords = _xy_from_m(x_coords_m)
    y_coords = _xy_from_m(y_coords_m)
    twt_display = _time_from_ms(da.values)
    twt_hover = HoverTool(tooltips=[("X", "$x{0,0.00}"), ("Y", "$y{0,0.00}"), (f"TWT ({_time_unit_label()})", "@{TWT}{0,0.00}")])
    base_plot = hv.QuadMesh((x_coords, y_coords, twt_display), kdims=["x", "y"], vdims=["TWT"]).opts(
        cmap=sel_cmap.value,
        title=f"Input Deterministic TWT Map ({_time_unit_label()})",
        colorbar=True,
        colorbar_opts={"title": f"Time [{_time_unit_label()}]"},
        tools=[twt_hover],
    )
    da_display = xr.DataArray(twt_display, coords=[y_coords, x_coords], dims=["y", "x"], name="TWT")
    contours = da_display.hvplot.contour(levels=sld_contours.value, color="black", alpha=0.5)
    angle_deg = _applied_section_angle()
    xs_line, ys_line = _section_line_display_coords(angle_deg)
    cross_section = hv.Curve((xs_line, ys_line)).opts(color="red", line_width=2, line_dash="dashed")
    overlay_time = base_plot * contours * cross_section
    wells_df = _well_overlay_dataframe() if chk_show_wells.value else pd.DataFrame()
    if chk_show_wells.value:
        if not wells_df.empty:
            well_hover = HoverTool(tooltips=[("Well", "@well_name"), ("X", "@x{0,0.00}"), ("Y", "@y{0,0.00}")])
            well_pts = hv.Points(wells_df, kdims=["x", "y"], vdims=["well_name", "velocity"]).opts(
                color="yellow",
                size=9,
                marker="circle",
                line_color="black",
                tools=[well_hover],
            )
            well_labels = hv.Labels(wells_df, kdims=["x", "y"], vdims=["well_name"]).opts(
                text_font_size="9pt",
                text_color="yellow",
                text_align="left",
                xoffset=10,
                yoffset=8,
            )
            overlay_time = overlay_time * well_pts * well_labels

    depth_display = _depth_from_m(np.asarray(depth_input_da.values, dtype=float))
    depth_hover = HoverTool(tooltips=[("X", "$x{0,0.00}"), ("Y", "$y{0,0.00}"), (f"Depth ({_depth_unit_label()})", "@{Depth}{0,0.00}")])
    depth_plot = hv.QuadMesh((x_coords, y_coords, depth_display), kdims=["x", "y"], vdims=["Depth"]).opts(
        cmap=sel_cmap.value,
        title=f"Input Deterministic Depth Map ({_depth_unit_label()})",
        colorbar=True,
        colorbar_opts={"title": f"Depth [{_depth_unit_label()}]"},
        tools=[depth_hover],
    )
    depth_da_display = xr.DataArray(depth_display, coords=[y_coords, x_coords], dims=["y", "x"], name="DepthInput")
    depth_step = max(float(sld_input_depth_contours.value), 0.5)
    dmin = float(np.nanmin(depth_display))
    dmax = float(np.nanmax(depth_display))
    depth_levels = np.arange(dmin, dmax + depth_step, depth_step) if dmax > dmin else np.array([])
    depth_contours = depth_da_display.hvplot.contour(levels=list(depth_levels), color="white", alpha=0.7) if len(depth_levels) > 0 else hv.Curve([])
    overlay_depth = depth_plot * depth_contours * cross_section

    if chk_show_wells.value:
        if not wells_df.empty:
            well_hover = HoverTool(tooltips=[("Well", "@well_name"), ("X", "@x{0,0.00}"), ("Y", "@y{0,0.00}")])
            well_pts = hv.Points(wells_df, kdims=["x", "y"], vdims=["well_name", "velocity"]).opts(
                color="yellow",
                size=9,
                marker="circle",
                line_color="black",
                tools=[well_hover],
            )
            well_labels = hv.Labels(wells_df, kdims=["x", "y"], vdims=["well_name"]).opts(
                text_font_size="9pt",
                text_color="yellow",
                text_align="left",
                xoffset=10,
                yoffset=8,
            )
            overlay_depth = overlay_depth * well_pts * well_labels

    time_panel = pn.panel(
        overlay_time.opts(
            toolbar="above",
            aspect=None,
            data_aspect=None,
            xlim=(float(np.min(x_coords)), float(np.max(x_coords))),
            ylim=(float(np.min(y_coords)), float(np.max(y_coords))),
            xlabel=f"x ({_xy_unit_label()})",
            ylabel=f"y ({_xy_unit_label()})",
            shared_axes=False,
            framewise=True,
        ),
        sizing_mode="stretch_both",
    )
    depth_panel = pn.panel(
        overlay_depth.opts(
            toolbar="above",
            aspect=None,
            data_aspect=None,
            xlim=(float(np.min(x_coords)), float(np.max(x_coords))),
            ylim=(float(np.min(y_coords)), float(np.max(y_coords))),
            xlabel=f"x ({_xy_unit_label()})",
            ylabel=f"y ({_xy_unit_label()})",
            shared_axes=False,
            framewise=True,
        ),
        sizing_mode="stretch_both",
    )

    if input_surface_tabs is None:
        input_surface_tabs = pn.Tabs(("Input Time Surface", time_panel), ("Input Depth Surface", depth_panel), dynamic=False, tabs_location="above", sizing_mode="stretch_both")
    else:
        active = input_surface_tabs.active
        input_surface_tabs[:] = [("Input Time Surface", time_panel), ("Input Depth Surface", depth_panel)]
        input_surface_tabs.active = min(active, 1)

    return input_surface_tabs


@pn.depends(
    btn_update_depth_maps,
    btn_update_culm,
    btn_update_vario,
    refresh_token,
    wells_refresh_token,
)
def plot_top_right_section(*_args):
    logger.debug("Rendering structural section with stochastic depth realizations")

    angle_deg = _applied_section_angle()
    _, depth_stack, _, final_depth_map = get_velocity_and_depth_stacks()
    base_twt = surface_state["twt"]
    x_coords_m, final_section_m, section_stack_m = core_extract_section_stack(
        depth_stack,
        final_depth_map,
        base_twt.x.values,
        base_twt.y.values,
        angle_deg,
    )

    x_coords = _xy_from_m(x_coords_m)
    final_section = _depth_from_m(final_section_m)
    section_stack = _depth_from_m(section_stack_m)

    base_curve = hv.Curve(
        (x_coords, final_section),
        kdims=["Distance_x"],
        vdims=["Depth_Z"],
        label=f"Final Depth [{_depth_unit_label()}] from Mean AV",
    ).opts(color="black", line_width=3)
    section_std = np.nanstd(section_stack, axis=0)
    upper_std_curve = hv.Curve(
        (x_coords, final_section + section_std),
        kdims=["Distance_x"],
        vdims=["Depth_Z"],
        label="Std +1sigma (dashed black)",
    ).opts(color="black", line_dash="dashed", line_width=1.5, alpha=0.9)
    lower_std_curve = hv.Curve(
        (x_coords, final_section - section_std),
        kdims=["Distance_x"],
        vdims=["Depth_Z"],
        label="Std -1sigma (dashed black)",
    ).opts(color="black", line_dash="dashed", line_width=1.5, alpha=0.9)

    selected_idx = min(max(sld_map_show.value - 1, 0), section_stack.shape[0] - 1)
    n_maps = int(section_stack.shape[0])
    max_display_curves = 120
    stride = max(1, int(np.ceil(n_maps / max_display_curves)))
    display_indices = list(range(0, n_maps, stride))
    if selected_idx not in display_indices:
        display_indices.append(selected_idx)
    display_indices = sorted(set(display_indices))
    realization_curves = [
        hv.Curve((x_coords, section_stack[i]), label="Stochastic Depth Realizations").opts(
            color="red",
            line_width=0.6,
            alpha=0.25,
            show_legend=False,
        )
        for i in display_indices
        if i != selected_idx
    ]
    selected_curve = hv.Curve(
        (x_coords, section_stack[selected_idx]),
        kdims=["Distance_x"],
        vdims=["Depth_Z"],
        label=f"Selected Realization #{selected_idx + 1}",
    ).opts(color="red", line_width=2.3, alpha=1.0, show_legend=False)

    deterministic_spill_m, _, _ = core_get_trap_and_spill(final_depth_map, _depth_step_m())
    deterministic_spill = float(_depth_from_m([deterministic_spill_m])[0])
    spill_label = f"Spill depth: {deterministic_spill:.2f} {_depth_unit_label()}"
    spill_line = hv.HLine(
        deterministic_spill,
        label=f"Spill depth (dashed blue): {deterministic_spill:.2f} {_depth_unit_label()}",
    ).opts(color="blue", line_dash="dashed", line_width=2)

    is_negative_domain = bool(np.nanmax(section_stack) <= 0.0)
    crest_idx = int(np.nanargmax(final_section)) if is_negative_domain else int(np.nanargmin(final_section))
    crest_x = float(x_coords[crest_idx])
    crest_y = float(final_section[crest_idx])
    crest_point = hv.Scatter(([crest_x], [crest_y]), kdims=["Distance_x"], vdims=["Depth_Z"]).opts(
        color="black",
        size=8,
        marker="circle",
        line_color="black",
    )

    top_y, bottom_y = sld_y_range.value
    min_y = min(top_y, bottom_y)
    max_y = max(top_y, bottom_y)
    ylim_value = (min_y, max_y) if is_negative_domain else (max_y, min_y)
    x_text = float(np.max(x_coords) - 0.02 * (np.max(x_coords) - np.min(x_coords)))
    y_span = max(1e-6, max_y - min_y)
    y_text = float(deterministic_spill + 0.03 * y_span)
    spill_text = hv.Text(x_text, y_text, spill_label, halign="right", valign="bottom").opts(color="blue", text_font_size="9pt", text_font_style="bold")
    crest_point_text = hv.Text(
        float(crest_x + 0.03 * (np.max(x_coords) - np.min(x_coords))),
        crest_y,
        f"Crest: {crest_y:.2f} {_depth_unit_label()}",
        halign="left",
        valign="bottom",
    ).opts(color="black", text_font_size="9pt", text_font_style="bold")

    section_overlay = hv.Overlay(realization_curves + [selected_curve]) * base_curve * spill_line * crest_point * spill_text * crest_point_text

    if chk_show_wells.value:
        wx_m, wy_m = _well_section_anchor_xy_m()
        well_names = [str(name) for name in _well_names_internal().tolist()]
        if wx_m.size > 0:
            distance_m = _project_wells_to_section_distance_m(
                well_x_m=np.asarray(wx_m, dtype=float),
                well_y_m=np.asarray(wy_m, dtype=float),
                x_values_m=np.asarray(base_twt.x.values, dtype=float),
                y_values_m=np.asarray(base_twt.y.values, dtype=float),
                angle_deg=angle_deg,
                section_distance_m=np.asarray(x_coords_m, dtype=float),
            )
            distance_vals = _xy_from_m(distance_m)
            section_depth_vals = np.interp(distance_vals, x_coords, final_section)
            section_wells = pd.DataFrame(
                {
                    "Distance": distance_vals,
                    "Depth": section_depth_vals,
                    "well_name": well_names,
                }
            )
            section_hover = HoverTool(tooltips=[("Well", "@well_name"), ("Distance", "@Distance{0,0.00}"), ("Depth", "@Depth{0,0.00}")])
            section_overlay = section_overlay * hv.Points(section_wells, kdims=["Distance", "Depth"], vdims=["well_name"]).opts(
                color=MAGENTA_OMV_COLOR,
                size=8,
                marker="circle",
                line_color="black",
                tools=[section_hover],
            )
            section_overlay = section_overlay * hv.Labels(section_wells, kdims=["Distance", "Depth"], vdims=["well_name"]).opts(
                text_font_size="9pt",
                text_color=DARK_MAGENTA_OMV_COLOR,
                text_align="left",
                xoffset=6,
                yoffset=4,
            )

            for idx, well_x in enumerate(distance_vals):
                line_label = "Well location (dashed magenta)" if idx == 0 else ""
                section_overlay = section_overlay * hv.VLine(float(well_x), label=line_label).opts(
                    color=MAGENTA_OMV_COLOR,
                    line_dash="dashed",
                    line_width=1.6,
                    alpha=0.9,
                )

    section_overlay = section_overlay * upper_std_curve * lower_std_curve
    return section_overlay.opts(
        toolbar="above",
        title=f"Structural Section (Perpendicular, Angle={angle_deg:.0f}degree)",
        ylim=ylim_value,
        xlabel=f"Distance along section ({_xy_unit_label()})",
        ylabel=f"Depth ({_depth_unit_label()})",
        shared_axes=False,
        show_legend=True,
        legend_position="top_right",
    )


@pn.depends(
    btn_update_depth_maps,
    btn_update_culm,
    btn_update_vario,
    refresh_token,
    wells_refresh_token,
)
def plot_bottom_left_final_depth_from_av(*_args):
    global av_depth_tabs
    logger.debug("Rendering average velocity map and selected realization map")

    angle_deg = _applied_section_angle()
    velocity_stack, depth_stack, avg_velocity_map, final_depth_map = get_velocity_and_depth_stacks()
    selected_idx = min(max(sld_map_show.value - 1, 0), velocity_stack.shape[0] - 1)

    base_twt = surface_state["twt"]
    x_min, x_max, y_min, y_max, x_coords_m, y_coords_m = _surface_xy_bounds()
    x_coords = _xy_from_m(x_coords_m)
    y_coords = _xy_from_m(y_coords_m)
    av_display = _velocity_from_mps(avg_velocity_map)
    av_da = xr.DataArray(av_display, coords=base_twt.coords, dims=base_twt.dims, name="AV")
    depth_arr = np.asarray(final_depth_map, dtype=float)
    if not np.isfinite(depth_arr).any():
        fallback_depth = (np.asarray(base_twt.values, dtype=float) * np.asarray(avg_velocity_map, dtype=float)) / 2000.0
        depth_arr = fallback_depth
        _notify_warning("Final depth map had no finite values; fallback depth map was used.", duration=4000)
    depth_display = _depth_from_m(depth_arr)
    depth_da = xr.DataArray(depth_display, coords=base_twt.coords, dims=base_twt.dims, name="Depth")
    depth_contour_da = xr.DataArray(depth_display, coords=[y_coords, x_coords], dims=["y", "x"], name="DepthDisp")

    av_hover = HoverTool(tooltips=[("X", "$x{0,0.00}"), ("Y", "$y{0,0.00}"), (f"AV ({rg_velocity_units.value})", "@{AV}{0,0.00}")])
    real_hover = HoverTool(tooltips=[("X", "$x{0,0.00}"), ("Y", "$y{0,0.00}"), (f"Depth ({_depth_unit_label()})", "@{DepthReal}{0,0.00}")])
    av_plot = hv.QuadMesh((x_coords, y_coords, av_da.values), kdims=["x", "y"], vdims=["AV"]).opts(
        cmap="viridis",
        line_color=None,
        line_alpha=0.0,
        show_grid=False,
        hooks=[_disable_bokeh_grid],
        colorbar=True,
        colorbar_opts={"title": f"Velocity [{rg_velocity_units.value}]"},
        title=f"Average Velocity Map (AV, {rg_velocity_units.value})",
        aspect=None,
        data_aspect=None,
        toolbar="above",
        xlim=(float(np.min(x_coords)), float(np.max(x_coords))),
        ylim=(float(np.min(y_coords)), float(np.max(y_coords))),
        xlabel=f"x ({_xy_unit_label()})",
        ylabel=f"y ({_xy_unit_label()})",
        tools=[av_hover],
        shared_axes=False,
        framewise=True,
    )
    selected_depth_arr = np.asarray(depth_stack[selected_idx], dtype=float)
    selected_depth_display = _depth_from_m(selected_depth_arr)
    selected_depth_da_display = xr.DataArray(selected_depth_display, coords=[y_coords, x_coords], dims=["y", "x"], name="DepthReal")
    realization_plot = hv.QuadMesh((x_coords, y_coords, selected_depth_display), kdims=["x", "y"], vdims=["DepthReal"]).opts(
        cmap=sel_cmap.value,
        line_color=None,
        line_alpha=0.0,
        show_grid=False,
        hooks=[_disable_bokeh_grid],
        colorbar=True,
        colorbar_opts={"title": f"Depth [{_depth_unit_label()}]"},
        title=f"Depth Realization #{selected_idx + 1} ({_depth_unit_label()})",
        aspect=None,
        data_aspect=None,
        toolbar="above",
        xlim=(float(np.min(x_coords)), float(np.max(x_coords))),
        ylim=(float(np.min(y_coords)), float(np.max(y_coords))),
        xlabel=f"x ({_xy_unit_label()})",
        ylabel=f"y ({_xy_unit_label()})",
        tools=[real_hover],
        shared_axes=False,
        framewise=True,
    )
    xs_line, ys_line = _section_line_display_coords(angle_deg)
    av_overlay = av_plot * hv.Curve((xs_line, ys_line)).opts(color="red", line_width=2, line_dash="dashed")
    overlay = realization_plot * hv.Curve((xs_line, ys_line)).opts(color="red", line_width=2, line_dash="dashed")

    if chk_show_wells.value:
        wells_df = _well_overlay_dataframe()
        if not wells_df.empty:
            well_hover = HoverTool(tooltips=[("Well", "@well_name"), ("X", "@x{0,0.00}"), ("Y", "@y{0,0.00}")])
            well_pts = hv.Points(wells_df, kdims=["x", "y"], vdims=["well_name", "velocity"]).opts(
                color="yellow",
                size=9,
                marker="circle",
                line_color="black",
                tools=[well_hover],
            )
            well_labels = hv.Labels(wells_df, kdims=["x", "y"], vdims=["well_name"]).opts(
                text_font_size="9pt",
                text_color="yellow",
                text_align="left",
                xoffset=10,
                yoffset=8,
            )
            av_overlay = av_overlay * well_pts * well_labels
            overlay = overlay * well_pts * well_labels

    deterministic_spill_m, deterministic_mask, _ = core_get_trap_and_spill(depth_arr, _depth_step_m())
    deterministic_spill = float(_depth_from_m([deterministic_spill_m])[0])

    if chk_eliminate.value:
        if np.any(deterministic_mask):
            pass
        else:
            _notify_warning("No closure mask found for current settings; showing unmasked final depth map.", duration=3500)

    if chk_close_poly.value:
        try:
            red_contour = depth_contour_da.hvplot.contour(levels=[deterministic_spill], cmap=["red"]).opts(line_width=3, line_dash="dashed")
            av_overlay = av_overlay * red_contour
        except Exception:
            logger.warning("Final depth contour rendering skipped due to contour level/data mismatch")

    if chk_depth_contour.value:
        try:
            step_val = max(float(sld_depth_contours.value), 0.5)
            vmin = float(np.nanmin(depth_da.values))
            vmax = float(np.nanmax(depth_da.values))
            if vmax > vmin:
                levels = np.arange(vmin, vmax + step_val, step_val)
                if len(levels) > 0:
                    white_contours = depth_contour_da.hvplot.contour(levels=list(levels), color="white", line_width=1, alpha=0.8)
                    av_overlay = av_overlay * white_contours
            rvmin = float(np.nanmin(selected_depth_da_display.values))
            rvmax = float(np.nanmax(selected_depth_da_display.values))
            if rvmax > rvmin:
                rlevels = np.arange(rvmin, rvmax + step_val, step_val)
                if len(rlevels) > 0:
                    r_contours = selected_depth_da_display.hvplot.contour(levels=list(rlevels), color="white", line_width=1, alpha=0.8)
                    overlay = overlay * r_contours
        except Exception:
            logger.warning("Final depth white contour rendering skipped")

    average_panel = pn.panel(
        av_overlay.opts(
            aspect=None,
            data_aspect=None,
            xlim=(float(np.min(x_coords)), float(np.max(x_coords))),
            ylim=(float(np.min(y_coords)), float(np.max(y_coords))),
            shared_axes=False,
            framewise=True,
        ),
        sizing_mode="stretch_both",
    )
    final_panel = pn.panel(
        overlay.opts(
            aspect=None,
            data_aspect=None,
            xlim=(float(np.min(x_coords)), float(np.max(x_coords))),
            ylim=(float(np.min(y_coords)), float(np.max(y_coords))),
            shared_axes=False,
            framewise=True,
        ),
        sizing_mode="stretch_both",
    )

    if av_depth_tabs is None:
        av_depth_tabs = pn.Tabs(
            ("Average Velocity Map", average_panel),
            ("Selected Depth Realization", final_panel),
            dynamic=False,
            tabs_location="above",
            sizing_mode="stretch_both",
        )
    else:
        current_active = av_depth_tabs.active
        av_depth_tabs[:] = [("Average Velocity Map", average_panel), ("Selected Depth Realization", final_panel)]
        av_depth_tabs.active = min(current_active, 1)

    return av_depth_tabs


@pn.depends(btn_update_vario, refresh_token, wells_refresh_token)
def plot_bottom_left_variogram(*_args):
    velocity_stack, _, _, _ = get_velocity_and_depth_stacks()
    selected_idx = min(max(sld_map_show.value - 1, 0), velocity_stack.shape[0] - 1)
    selected_map = np.asarray(velocity_stack[selected_idx], dtype=float)
    selected_centered = selected_map - float(np.nanmean(selected_map))
    raw_residual_std = float(np.nanstd(selected_centered))
    raw_residual_var = float(np.nanvar(selected_centered))
    if raw_residual_std > 0:
        selected_map_for_vario = selected_centered / raw_residual_std
    else:
        selected_map_for_vario = selected_centered
    x_coords_m = np.asarray(surface_state.get("x_coords", surface_state["twt"].x.values), dtype=float)
    y_coords_m = np.asarray(surface_state.get("y_coords", surface_state["twt"].y.values), dtype=float)

    empirical_df = _experimental_variogram_from_grid(selected_map_for_vario, x_coords_m, y_coords_m)
    h = np.linspace(0, 10000, 250)
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
    curve = df.hvplot.line(x="Distance", y="Semivariance", color="teal", line_width=2, label="Fitted model")
    empirical = hv.Scatter([])
    if not empirical_df.empty:
        empirical = hv.Scatter(empirical_df, kdims=["lag"], vdims=["gamma", "pairs"], label="Experimental").opts(color="orange", size=6, alpha=0.9)

    norm_var_val = float(np.nanvar(selected_map_for_vario)) if np.isfinite(selected_map_for_vario).any() else float("nan")
    norm_std_val = float(np.nanstd(selected_map_for_vario)) if np.isfinite(selected_map_for_vario).any() else float("nan")
    txt = hv.Text(
        float(0.98 * np.max(h)),
        float(np.nanmax(gamma) * 0.92),
        f"Raw Residual Var: {raw_residual_var:.2f}\nRaw Residual Std: {raw_residual_std:.2f}\nNorm Var: {norm_var_val:.2f} | Norm Std: {norm_std_val:.2f}",
        halign="right",
        valign="top",
    ).opts(
        color="black",
        text_font_size="9pt",
        text_font_style="bold",
    )

    sill_line = hv.HLine(sld_nugget.value + sld_sill.value).opts(color="red", line_width=1)
    range_line = hv.VLine(sld_range.value).opts(color="red", line_width=1)

    return (curve * empirical * sill_line * range_line * txt).opts(
        toolbar="above",
        xlim=(0, None),
        ylim=(0, max(1.5, float(np.nanmax(gamma) * 1.25))),
        title="Variogram (Experimental + Fitted, Normalized)",
        show_legend=True,
        legend_position="bottom_right",
        legend_opts={"background_fill_alpha": 0.0, "border_line_alpha": 0.0},
    )


@pn.depends(
    btn_update_depth_maps,
    btn_update_culm,
    btn_update_vario,
    btn_update_volumetrics,
    refresh_token,
    wells_refresh_token,
)
def plot_bottom_right_isoprobability(*_args):
    logger.debug("Rendering isoprobability map")

    _, depth_stack, _, final_depth_map = get_velocity_and_depth_stacks()
    n_maps = depth_stack.shape[0]

    trap_stats = get_trap_stats(depth_stack)
    count_array = trap_stats["trap_masks"].sum(axis=0).astype(float)
    prob_array = (count_array / n_maps) * 100 if n_maps > 0 else np.zeros_like(count_array)
    if float(np.max(prob_array)) <= 0.0:
        spill_fallback, det_mask, _ = core_get_trap_and_spill(np.asarray(final_depth_map, dtype=float), _depth_step_m())
        if np.any(det_mask):
            count_array = det_mask.astype(float) * float(max(n_maps, 1))
            prob_array = det_mask.astype(float) * 100.0
            _notify_info("No isolated stochastic closure detected; displaying deterministic closure probability proxy.", duration=4500)
        else:
            _notify_warning("No isolated closures detected with current settings. Try increasing variogram range/sill or adjusting contour step.", duration=4000)

    _, _, _, _, x_coords_m, y_coords_m = _surface_xy_bounds()
    x_coords = _xy_from_m(x_coords_m)
    y_coords = _xy_from_m(y_coords_m)
    ds_prob = xr.Dataset(
        {
            "Probability": (["y", "x"], prob_array),
            "ClosureCount": (["y", "x"], count_array),
            "TotalMaps": (["y", "x"], np.full_like(count_array, n_maps)),
        },
        coords={"x": x_coords, "y": y_coords},
    )

    custom_hover = HoverTool(
        tooltips=[
            ("X, Y", "$x{0,0}, $y{0,0}"),
            ("Probability", "@{Probability}{0.0}%"),
            ("Closure Detail", "In @ClosureCount maps of @TotalMaps this node is within isolated trap"),
        ]
    )

    hv_ds = hv.Dataset(ds_prob)
    img = hv_ds.to(hv.QuadMesh, kdims=["x", "y"], vdims=["Probability", "ClosureCount", "TotalMaps"]).opts(
        cmap="rainbow",
        clim=(0, 100),
        line_color=None,
        line_alpha=0.0,
        show_grid=False,
        hooks=[_disable_bokeh_grid],
        colorbar=True,
        colorbar_opts={"title": "Probability [%]"},
        tools=[custom_hover],
    )
    contour_levels = list(np.linspace(10, 90, 9))
    contours = hv.operation.contours(img, levels=contour_levels).opts(color="black", line_width=1.0, alpha=0.85, show_legend=False)
    angle_deg = _applied_section_angle()
    xs_line, ys_line = _section_line_display_coords(angle_deg)
    section_line = hv.Curve((xs_line, ys_line)).opts(color="red", line_width=2, line_dash="dashed", alpha=0.95)

    overlay = img * contours * section_line
    if chk_show_wells.value:
        wells_df = _well_overlay_dataframe()
        if not wells_df.empty:
            well_hover = HoverTool(tooltips=[("Well", "@well_name"), ("X", "@x{0,0.00}"), ("Y", "@y{0,0.00}")])
            well_pts = hv.Points(wells_df, kdims=["x", "y"], vdims=["well_name", "velocity"]).opts(
                color="yellow",
                size=9,
                marker="circle",
                line_color="black",
                tools=[well_hover],
            )
            well_labels = hv.Labels(wells_df, kdims=["x", "y"], vdims=["well_name"]).opts(
                text_font_size="9pt",
                text_color="yellow",
                text_align="left",
                xoffset=10,
                yoffset=8,
            )
            overlay = overlay * well_pts * well_labels

    return overlay.opts(
        toolbar="above",
        aspect=None,
        data_aspect=None,
        xlim=(float(np.nanmin(x_coords)), float(np.nanmax(x_coords))),
        ylim=(float(np.nanmin(y_coords)), float(np.nanmax(y_coords))),
        xlabel=f"x ({_xy_unit_label()})",
        ylabel=f"y ({_xy_unit_label()})",
        shared_axes=False,
        framewise=True,
        show_legend=False,
        title=f"Isoprobability Map (Isolated Trap, N={n_maps})",
    )


def create_hist_with_stats(data: np.ndarray, name: str, color: str, title: str):
    if len(data) == 0 or np.all(data == 0):
        return hv.Curve([]).opts(title=title)

    arr = np.asarray(data, dtype=float)
    mean_val = float(np.mean(arr))
    p90_val = float(np.percentile(arr, 10))
    p50_val = float(np.percentile(arr, 50))
    p10_val = float(np.percentile(arr, 90))

    counts, edges = np.histogram(arr, bins=20)
    max_y = float(counts.max())
    y_top = max(max_y * 1.22, 1.0)
    x_text = float(arr.min() + (arr.max() - arr.min()) * 0.03)

    hist = hv.Histogram((edges, counts), kdims=[name], vdims=["Freq"]).opts(
        color=color,
        alpha=0.72,
        line_color="white",
        title=title,
        ylabel="Freq",
    )

    l_mean = hv.VLine(mean_val, label="Mean").opts(color="black", line_dash="dashed", line_width=1.5)
    l_p90 = hv.VLine(p90_val, label="P90").opts(color="red", line_dash="dashed", line_width=1.5)
    l_p50 = hv.VLine(p50_val, label="P50").opts(color="green", line_dash="dashed", line_width=1.5)
    l_p10 = hv.VLine(p10_val, label="P10").opts(color="blue", line_dash="dashed", line_width=1.5)

    t_mean = hv.Text(x_text, y_top * 0.965, f"Mean: {mean_val:.2f}", halign="left", valign="top").opts(
        color="black",
        text_font_size="8pt",
        text_font_style="bold",
    )
    t_p90 = hv.Text(x_text, y_top * 0.865, f"P90: {p90_val:.2f}", halign="left", valign="top").opts(
        color="red",
        text_font_size="8pt",
        text_font_style="bold",
    )
    t_p50 = hv.Text(x_text, y_top * 0.765, f"P50: {p50_val:.2f}", halign="left", valign="top").opts(
        color="green",
        text_font_size="8pt",
        text_font_style="bold",
    )
    t_p10 = hv.Text(x_text, y_top * 0.665, f"P10: {p10_val:.2f}", halign="left", valign="top").opts(
        color="blue",
        text_font_size="8pt",
        text_font_style="bold",
    )

    return (hist * l_p90 * l_p50 * l_p10 * l_mean * t_p90 * t_p50 * t_p10 * t_mean).opts(
        show_legend=True,
        legend_position="top_right",
        shared_axes=False,
        ylim=(0, y_top),
        fontsize={"title": 10, "labels": 9, "xticks": 8, "yticks": 8},
    )


def _export_report(_event) -> None:
    header = txt_report_header.value.strip() or "Structural Uncertainty Evaluation"
    base_name = txt_report_filename.value.strip() or "structural_uncertainty_report"
    extension = sel_report_type.value
    out_dir = Path(txt_report_location.value.strip() or str(APP_DIR / "reports"))

    try:
        report_export_progress.visible = True
        report_export_progress.value = 5
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{base_name}{extension}"
        _notify_info(f"Exporting to {output_path} ...", duration=4000)

        report_export_progress.value = 20
        payload = _build_report_payload()
        velocity_stack, depth_stack_qc, avg_velocity_map_qc, final_depth_map_qc = get_velocity_and_depth_stacks()
        trap_stats_qc = get_trap_stats(depth_stack_qc)
        qc_bundle = _build_geostat_qc_bundle_payload(
            velocity_stack=velocity_stack,
            depth_stack=depth_stack_qc,
            avg_velocity_map=avg_velocity_map_qc,
            final_depth_map=final_depth_map_qc,
            trap_stats=trap_stats_qc,
        )
        report_export_progress.value = 40

        if extension == ".xlsx":
            _export_qc_excel_workbook(output_path, payload, qc_bundle=qc_bundle)
            report_export_progress.value = 100
        elif extension in {".png", ".jpeg"}:
            fig = _build_dashboard_figure(header, payload)
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            report_export_progress.value = 100
        else:
            with PdfPages(output_path) as pdf:
                fig_dashboard = _build_dashboard_figure(header, payload)
                pdf.savefig(fig_dashboard, bbox_inches="tight")
                plt.close(fig_dashboard)
                report_export_progress.value = 60

                for fig_tables in core_build_pdf_tables_pages(
                    report_title=header,
                    summary_df=payload["summary_df"],
                    realization_df=payload["realization_df"],
                    logo_path=REPORT_TABLE_LOGO_PATH,
                    header_color=DARK_BLUE_OMV_COLOR,
                    rows_per_page=40,
                ):
                    pdf.savefig(fig_tables, bbox_inches="tight")
                    plt.close(fig_tables)
                report_export_progress.value = 85

                fig_params = _build_pdf_parameters_page(header)
                pdf.savefig(fig_params, bbox_inches="tight")
                plt.close(fig_params)
                for fig_qc in core_build_pdf_geostat_qc_pages(
                    report_title=header,
                    logo_path=REPORT_TABLE_LOGO_PATH,
                    wells_cov_df=_wells_covariance_dataframe(),
                    qc_bundle=qc_bundle,
                ):
                    pdf.savefig(fig_qc, bbox_inches="tight")
                    plt.close(fig_qc)
                report_export_progress.value = 95

        report_export_progress.value = 100
        _notify_success(f"Export completed: {output_path}", duration=4200)
        if extension == ".xlsx":
            qc_export_status.object = f"Excel exported: {output_path}"
        else:
            qc_export_status.object = ""
        qc_export_status.visible = True
        logger.info("Report exported: %s", output_path)
    except Exception as exc:
        _notify_error(f"Export failed: {exc}", duration=6000)
        logger.exception("Report export failed")
    finally:
        time.sleep(0.2)
        report_export_progress.visible = False
        report_export_progress.value = 0


btn_export_report.on_click(_export_report)


def _export_qc_excel_workbook(
    output_path: Path,
    payload: dict[str, np.ndarray | pd.DataFrame | float],
    qc_bundle: dict[str, pd.DataFrame] | None = None,
) -> None:
    velocity_stack, depth_stack, avg_velocity_map, final_depth_map = get_velocity_and_depth_stacks()
    trap_stats = get_trap_stats(depth_stack)

    twt_da = surface_state["twt"]
    vel_da = surface_state["velocity"]
    x_coords = np.asarray(twt_da.x.values, dtype=float)
    y_coords = np.asarray(twt_da.y.values, dtype=float)
    xv, yv = np.meshgrid(x_coords, y_coords)

    twt_flat = np.asarray(twt_da.values, dtype=float).ravel()
    base_vel_flat = np.asarray(vel_da.values, dtype=float).ravel()
    x_flat = xv.ravel()
    y_flat = yv.ravel()

    n_maps = int(depth_stack.shape[0])
    n_cells = int(twt_flat.size)

    probability = np.asarray(payload["probability"], dtype=float).ravel()
    trap_count = trap_stats["trap_masks"].sum(axis=0).astype(int).ravel()
    av_flat = np.asarray(avg_velocity_map, dtype=float).ravel()
    final_depth_flat = np.asarray(final_depth_map, dtype=float).ravel()
    depth_from_av_flat = (twt_flat * av_flat) / 2000.0

    arr_thick, arr_area, arr_grv, arr_stooip = _volumetric_series(trap_stats)
    selected_idx = min(max(int(sld_map_show.value) - 1, 0), velocity_stack.shape[0] - 1)
    selected_velocity = np.asarray(velocity_stack[selected_idx], dtype=float)
    selected_residual = selected_velocity - float(np.nanmean(selected_velocity))
    selected_residual_std = float(np.nanstd(selected_residual)) if np.isfinite(selected_residual).any() else float("nan")
    selected_residual_var = float(np.nanvar(selected_residual)) if np.isfinite(selected_residual).any() else float("nan")
    wells_x_m, wells_y_m, wells_v_mps = _well_arrays_internal()
    wells_names = [str(name) for name in _well_names_internal().tolist()]
    wells_cov_df = _wells_covariance_dataframe()
    cov_condition = float(np.linalg.cond(wells_cov_df.to_numpy(dtype=float))) if not wells_cov_df.empty else float("nan")

    if qc_bundle is None:
        qc_bundle = _build_geostat_qc_bundle_payload(
            velocity_stack=velocity_stack,
            depth_stack=depth_stack,
            avg_velocity_map=avg_velocity_map,
            final_depth_map=final_depth_map,
            trap_stats=trap_stats,
        )

    readme_df = pd.DataFrame(
        {
            "Section": [
                "Inputs",
                "Formulas",
                "Formulas",
                "Formulas",
                "Units",
                "QC guidance",
            ],
            "Description": [
                "Surface and realization values exported for audit/QC.",
                "Depth per cell = TWT_ms * Velocity_mps / 2000.",
                "IsoProbability_% = (TrapCount / N_Realizations) * 100.",
                "STOOIP_MMbbls = GRV_m3 * N/G * Phi * (1-Sw) / FVF * 6.2898 / 1e6.",
                "Internal calculations are exported in base units (m, m/s, ms, m2, m3).",
                "Use Formula columns to compare Excel recalculation vs exported value.",
            ],
        }
    )

    params_df = pd.DataFrame(
        {
            "Parameter": [
                "Surface mode",
                "N realizations",
                "Variogram model",
                "Range",
                "Sill",
                "Nugget",
                "Velocity std dev",
                "Smoothing sigma",
                "Contour increment",
                "Thickness mean",
                "Thickness std dev",
                "N/G",
                "Porosity",
                "Water saturation",
                "FVF",
                "Section angle",
                "Selected realization (for plots)",
                "Selected residual std (m/s)",
                "Selected residual var ((m/s)^2)",
                "Wells count",
                "Use wells as anchors",
                "Covariance matrix condition #",
            ],
            "Value": [
                sel_surface.value,
                int(sld_n_maps.value),
                sld_slope.value,
                float(sld_range.value),
                float(sld_sill.value),
                float(sld_nugget.value),
                float(sld_std_dev.value),
                float(sld_smooth.value),
                float(sld_c_inc.value),
                float(sld_thick_mean.value),
                float(sld_thick_std.value),
                float(sld_ntg.value),
                float(sld_poro.value),
                float(sld_sw.value),
                float(sld_fvf.value),
                float(sld_section_angle.value),
                int(selected_idx + 1),
                selected_residual_std,
                selected_residual_var,
                int(len(wells_x_m)),
                bool(chk_use_wells.value),
                cov_condition,
            ],
            "Unit/Notes": [
                "UI selection",
                "count",
                "Gaussian / Exponential / Spherical",
                "m",
                "-",
                "-",
                rg_velocity_units.value,
                "-",
                rg_depth_units.value,
                rg_depth_units.value,
                rg_depth_units.value,
                "fraction",
                "fraction",
                "fraction",
                "-",
                "degrees",
                "index",
                "computed from selected realization",
                "computed from selected realization",
                "count",
                "boolean",
                "conditioning matrix numerical stability",
            ],
        }
    )

    if len(wells_x_m) > 0:
        nearest_idx_x = np.array([int(np.argmin(np.abs(x_coords - wx))) for wx in wells_x_m], dtype=int)
        nearest_idx_y = np.array([int(np.argmin(np.abs(y_coords - wy))) for wy in wells_y_m], dtype=int)
        base_vel_at_well = np.array([float(vel_da.values[iy, ix]) for iy, ix in zip(nearest_idx_y, nearest_idx_x)], dtype=float)
        wells_df = pd.DataFrame(
            {
                "Well": [wells_names[i] if i < len(wells_names) else f"Well_{i+1}" for i in range(len(wells_x_m))],
                "X_m": wells_x_m,
                "Y_m": wells_y_m,
                "Velocity_mps": wells_v_mps,
                "NearestGrid_Ix": nearest_idx_x,
                "NearestGrid_Iy": nearest_idx_y,
                "BaseVelocityAtGrid_mps": base_vel_at_well,
                "VelocityDeltaToBase_mps": wells_v_mps - base_vel_at_well,
            }
        )
    else:
        wells_df = pd.DataFrame(columns=["Well", "X_m", "Y_m", "Velocity_mps", "NearestGrid_Ix", "NearestGrid_Iy", "BaseVelocityAtGrid_mps", "VelocityDeltaToBase_mps"])

    if not wells_cov_df.empty:
        cov_np = wells_cov_df.to_numpy(dtype=float)
        if cov_np.shape[0] > 1:
            tri = cov_np[np.triu_indices(cov_np.shape[0], k=1)]
            cov_diag_df = pd.DataFrame(
                {
                    "Metric": ["CovarianceConditionNumber", "CovarianceOffDiagonalMean", "CovarianceOffDiagonalMin", "CovarianceOffDiagonalMax"],
                    "Value": [
                        float(cov_condition),
                        float(np.nanmean(tri)) if tri.size else float("nan"),
                        float(np.nanmin(tri)) if tri.size else float("nan"),
                        float(np.nanmax(tri)) if tri.size else float("nan"),
                    ],
                }
            )
        else:
            cov_diag_df = pd.DataFrame(
                {
                    "Metric": ["CovarianceConditionNumber", "CovarianceOffDiagonalMean", "CovarianceOffDiagonalMin", "CovarianceOffDiagonalMax"],
                    "Value": [float(cov_condition), float("nan"), float("nan"), float("nan")],
                }
            )
    else:
        cov_diag_df = pd.DataFrame(
            {
                "Metric": ["CovarianceConditionNumber", "CovarianceOffDiagonalMean", "CovarianceOffDiagonalMin", "CovarianceOffDiagonalMax"],
                "Value": [float("nan"), float("nan"), float("nan"), float("nan")],
            }
        )

    input_surface_df = pd.DataFrame(
        {
            "X_m": x_flat,
            "Y_m": y_flat,
            "TWT_ms": twt_flat,
            "BaseVelocity_mps": base_vel_flat,
        }
    )

    map_qc_df = pd.DataFrame(
        {
            "X_m": x_flat,
            "Y_m": y_flat,
            "TWT_ms": twt_flat,
            "AV_Mean_mps": av_flat,
            "DepthFromAV_m": depth_from_av_flat,
            "FinalDepth_m": final_depth_flat,
            "DepthDelta_m": final_depth_flat - depth_from_av_flat,
            "TrapCount": trap_count,
            "N_Realizations": np.full_like(trap_count, n_maps),
            "IsoProbability_pct": probability,
        }
    )

    realization_summary_df = pd.DataFrame(payload["realization_df"]).copy()
    volumetrics_qc_df = pd.DataFrame(
        {
            "Realization": np.arange(1, n_maps + 1),
            "GRV_m3": np.asarray(trap_stats["grv"], dtype=float),
            "N_G": np.full(n_maps, float(sld_ntg.value)),
            "Phi": np.full(n_maps, float(sld_poro.value)),
            "OneMinusSw": np.full(n_maps, 1.0 - float(sld_sw.value)),
            "FVF": np.full(n_maps, float(sld_fvf.value)),
            "STOOIP_MMbbls_Exported": arr_stooip,
            "ThicknessTotal_Exported": arr_thick,
            "Area_Exported": arr_area,
            "GRV_Exported": arr_grv,
        }
    )

    max_excel_rows = 1_048_000
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        readme_df.to_excel(writer, sheet_name="QC_Readme", index=False)
        params_df.to_excel(writer, sheet_name="Parameters", index=False)
        input_surface_df.to_excel(writer, sheet_name="Input_Surface", index=False)
        map_qc_df.to_excel(writer, sheet_name="Map_QC", index=False)
        realization_summary_df.to_excel(writer, sheet_name="Realization_Summary", index=False)
        volumetrics_qc_df.to_excel(writer, sheet_name="Volumetrics_QC", index=False)
        wells_df.to_excel(writer, sheet_name="Wells_Input", index=False)
        if not wells_cov_df.empty:
            wells_cov_df.to_excel(writer, sheet_name="Wells_Covariance", index=True)
        else:
            pd.DataFrame({"Info": ["No wells defined"]}).to_excel(writer, sheet_name="Wells_Covariance", index=False)
        cov_diag_df.to_excel(writer, sheet_name="Wells_Covariance_QC", index=False)
        qc_bundle.get("empirical_variogram_df", pd.DataFrame()).to_excel(writer, sheet_name="Empirical_Variogram", index=False)
        qc_bundle.get("theoretical_variogram_df", pd.DataFrame()).to_excel(writer, sheet_name="Theoretical_Variogram", index=False)
        qc_bundle.get("recommended_parameters_df", pd.DataFrame()).to_excel(writer, sheet_name="Geostat_Recommendations", index=False)
        qc_bundle.get("coherence_checks_df", pd.DataFrame()).to_excel(writer, sheet_name="Engineering_Coherence", index=False)

        ws_map = writer.sheets["Map_QC"]
        ws_map["K1"] = "DepthFromAV_ExcelFormula"
        ws_map["L1"] = "DepthDelta_ExcelFormula"
        ws_map["M1"] = "IsoProbability_ExcelFormula"
        for row_idx in range(2, len(map_qc_df) + 2):
            ws_map[f"K{row_idx}"] = f"=C{row_idx}*D{row_idx}/2000"
            ws_map[f"L{row_idx}"] = f"=F{row_idx}-K{row_idx}"
            ws_map[f"M{row_idx}"] = f"=IF(I{row_idx}=0,0,H{row_idx}/I{row_idx}*100)"

        ws_vol = writer.sheets["Volumetrics_QC"]
        ws_vol["K1"] = "STOOIP_MMbbls_ExcelFormula"
        ws_vol["L1"] = "STOOIP_Delta"
        for row_idx in range(2, len(volumetrics_qc_df) + 2):
            ws_vol[f"K{row_idx}"] = f"=B{row_idx}*C{row_idx}*D{row_idx}*E{row_idx}/F{row_idx}*6.2898/1000000"
            ws_vol[f"L{row_idx}"] = f"=G{row_idx}-K{row_idx}"

        realization_ids = np.repeat(np.arange(1, n_maps + 1), n_cells)
        x_rep = np.tile(x_flat, n_maps)
        y_rep = np.tile(y_flat, n_maps)
        twt_rep = np.tile(twt_flat, n_maps)
        vel_rep = np.asarray(velocity_stack, dtype=float).reshape(n_maps, -1).ravel()
        depth_rep = np.asarray(depth_stack, dtype=float).reshape(n_maps, -1).ravel()
        trap_mask_rep = np.asarray(trap_stats["trap_masks"], dtype=bool).reshape(n_maps, -1).astype(int).ravel()

        realization_cells_df = pd.DataFrame(
            {
                "Realization": realization_ids,
                "X_m": x_rep,
                "Y_m": y_rep,
                "TWT_ms": twt_rep,
                "Velocity_mps": vel_rep,
                "Depth_m": depth_rep,
                "TrapMask": trap_mask_rep,
            }
        )

        if len(realization_cells_df) == 0:
            realization_cells_df.to_excel(writer, sheet_name="Realization_Cells_1", index=False)
        else:
            n_chunks = int(np.ceil(len(realization_cells_df) / max_excel_rows))
            for chunk_idx in range(n_chunks):
                start = chunk_idx * max_excel_rows
                end = min((chunk_idx + 1) * max_excel_rows, len(realization_cells_df))
                chunk_df = realization_cells_df.iloc[start:end].copy()
                sheet_name = f"Realization_Cells_{chunk_idx + 1}"
                chunk_df.to_excel(writer, sheet_name=sheet_name, index=False)
                ws_cells = writer.sheets[sheet_name]
                ws_cells["H1"] = "Depth_ExcelFormula"
                ws_cells["I1"] = "Depth_Delta"
                for row_idx in range(2, len(chunk_df) + 2):
                    ws_cells[f"H{row_idx}"] = f"=D{row_idx}*E{row_idx}/2000"
                    ws_cells[f"I{row_idx}"] = f"=F{row_idx}-H{row_idx}"


def _export_surface_to_petrel(_event) -> None:
    try:
        from cegalprizm.pythontool import PetrelConnection
    except Exception as exc:
        _set_petrel_export_status(f"Export failed: cegalprizm unavailable ({exc})")
        return

    try:
        _, depth_stack, avg_velocity_map, final_depth_map = get_velocity_and_depth_stacks()
        twt_da = surface_state["twt"]
        x_coords = np.asarray(twt_da.x.values, dtype=float)
        y_coords = np.asarray(twt_da.y.values, dtype=float)

        if len(x_coords) < 2 or len(y_coords) < 2:
            _set_petrel_export_status("Export failed: invalid surface grid axes")
            return

        n_maps = int(depth_stack.shape[0])
        trap_stats = get_trap_stats(depth_stack)
        count_array = trap_stats["trap_masks"].sum(axis=0).astype(float)
        prob_array = (count_array / n_maps) * 100.0 if n_maps > 0 else np.zeros_like(count_array)

        prefix = txt_export_petrel_surface_name.value.strip() or "StructuralUncertainty"
        selected_surfaces: list[tuple[str, np.ndarray]] = []

        if chk_export_average_mean.value:
            selected_surfaces.append((f"{prefix}_AverageVelocityMap", np.asarray(avg_velocity_map, dtype=float).T))
        if chk_export_final_depth.value:
            selected_surfaces.append((f"{prefix}_FinalDepth", np.asarray(final_depth_map, dtype=float).T))
        if chk_export_final_depth_contour.value:
            contour_step_m = max(float(sld_depth_contours.value) * _depth_unit_to_m(), 0.1)
            contour_map = np.asarray(final_depth_map, dtype=float)
            contour_map = np.where(np.isfinite(contour_map), np.round(contour_map / contour_step_m) * contour_step_m, np.nan)
            selected_surfaces.append((f"{prefix}_FinalDepthContour", contour_map.T))
        if chk_export_spill_contour_line.value:
            deterministic_spill, _, _ = core_get_trap_and_spill(final_depth_map, _depth_step_m())
            line_half_width = max(float(sld_depth_contours.value) * _depth_unit_to_m() * 0.5, 0.1)
            contour_line_surface = np.where(
                np.abs(np.asarray(final_depth_map, dtype=float) - float(deterministic_spill)) <= line_half_width,
                float(deterministic_spill),
                np.nan,
            )
            selected_surfaces.append((f"{prefix}_SpillContourLine", contour_line_surface.T))
        if chk_export_isoprobability.value:
            selected_surfaces.append((f"{prefix}_Isoprobability", np.asarray(prob_array, dtype=float).T))
        if chk_export_all_realizations.value:
            for idx in range(n_maps):
                selected_surfaces.append((f"{prefix}_Realization_{idx + 1:03d}", np.asarray(depth_stack[idx], dtype=float).T))

        if not selected_surfaces:
            _set_petrel_export_status("Select at least one export map.")
            return

        ptp = PetrelConnection(allow_experimental=True)
        folder = next(iter(ptp.interpretation_folders.values()), None)
        if folder is None:
            _set_petrel_export_status("Export failed: no interpretation folder available in Petrel")
            return

        existing_names = {getattr(s, "petrel_name", "") for s in ptp.surfaces}

        def _unique_name(base_name: str) -> str:
            if base_name not in existing_names:
                existing_names.add(base_name)
                return base_name
            suffix = 1
            candidate = f"{base_name}_{suffix:03d}"
            while candidate in existing_names:
                suffix += 1
                candidate = f"{base_name}_{suffix:03d}"
            existing_names.add(candidate)
            return candidate

        p0 = (float(np.min(x_coords)), float(np.min(y_coords)))
        p1 = (float(np.max(x_coords)), float(np.min(y_coords)))
        p2 = (float(np.min(x_coords)), float(np.max(y_coords)))

        created_names: list[str] = []
        failed_names: list[str] = []
        for base_name, surface_array in selected_surfaces:
            final_name = _unique_name(base_name)
            try:
                created = ptp.create_surface(
                    name=final_name,
                    domain="Elevation depth",
                    folder=folder,
                    origin_corner=p0,
                    i_corner=p1,
                    j_corner=p2,
                    array=surface_array,
                )
                created_names.append(getattr(created, "petrel_name", final_name))
            except Exception:
                failed_names.append(final_name)

        if failed_names and created_names:
            _set_petrel_export_status(
                f"Partial export: {len(created_names)} created, {len(failed_names)} failed. "
                f"Created: {', '.join(created_names[:5])}{' ...' if len(created_names) > 5 else ''}"
            )
            return
        if failed_names and not created_names:
            _set_petrel_export_status(
                f"Export failed for all selected surfaces ({len(failed_names)} attempted)."
            )
            return

        _set_petrel_export_status(
            f"Exported {len(created_names)} surface(s): "
            f"{', '.join(created_names[:5])}{' ...' if len(created_names) > 5 else ''}"
        )
    except Exception as exc:
        _set_petrel_export_status(f"Export failed: {exc}")


btn_export_petrel_surface.on_click(_export_surface_to_petrel)


@pn.depends(btn_update_volumetrics, btn_update_depth_maps, btn_update_culm, btn_update_vario, refresh_token)
def plot_third_column_volumetrics(*_args):
    logger.debug("Computing volumetrics distribution")

    _, depth_stack, _, _ = get_velocity_and_depth_stacks()

    trap_stats = get_trap_stats(depth_stack)
    hist_key = (
        trap_cache.get("key"),
        _depth_unit_label(),
        rg_area_units.value,
        rg_volume_units.value,
    )
    if hist_panel_cache["key"] == hist_key and hist_panel_cache["panel"] is not None:
        return hist_panel_cache["panel"]

    arr_thick, arr_area, arr_grv, arr_stooip = _volumetric_series(trap_stats)

    p1 = create_hist_with_stats(arr_thick, f"Thickness ({_depth_unit_label()})", "#8888ff", "Crest-to-Spill Thick.").opts(xlabel="")
    p2 = create_hist_with_stats(arr_area, f"Area ({rg_area_units.value})", "#ff8888", "Closure Area").opts(xlabel="")
    p3 = create_hist_with_stats(arr_grv, f"GRV ({rg_volume_units.value})", "#88ff88", "GRV Dist.").opts(xlabel="")
    p4 = create_hist_with_stats(arr_stooip, "STOOIP (MMbbls)", "#ffaa00", "STOOIP Dist.")

    hist_grid = pn.GridSpec(nrows=4, ncols=1, sizing_mode="stretch_both", margin=0)
    hist_grid[0, 0] = pn.panel(p1, sizing_mode="stretch_both")
    hist_grid[1, 0] = pn.panel(p2, sizing_mode="stretch_both")
    hist_grid[2, 0] = pn.panel(p3, sizing_mode="stretch_both")
    hist_grid[3, 0] = pn.panel(p4, sizing_mode="stretch_both")
    hist_panel = pn.Column(hist_grid, css_classes=["hist-container"], sizing_mode="stretch_both", margin=0)
    hist_panel_cache["key"] = hist_key
    hist_panel_cache["panel"] = hist_panel
    return hist_panel


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
    wrap_plot("Input Time and Depth Surfaces", plot_top_left_input_twt_map, "map"),
    wrap_plot("Structural Section (Depth Realizations)", plot_top_right_section, "section"),
    wrap_plot("Average Velocity and Selected Realization", plot_bottom_left_final_depth_from_av, "realization"),
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


card_1 = create_sidebar_card(
    "Input Selection",
    txt_input_time_surface,
    txt_input_depth_surface,
    sld_contours,
    sld_input_depth_contours,
    sel_cmap,
    btn_generate_surf,
    collapsed=True,
)
card_wells = create_sidebar_card(
    "Wells",
    tbl_wells,
    btn_add_well,
    btn_remove_well,
    btn_update_well_location,
    chk_show_wells,
    chk_use_wells,
    collapsed=True,
)
card_2 = create_sidebar_card("Trap Detection Rules", sld_c_inc, sld_depth_contours, chk_eliminate, chk_close_poly, chk_depth_contour, btn_update_culm)
card_3 = create_sidebar_card(
    "Velocity Realizations",
    sld_n_maps,
    sld_smooth,
    sld_y_range,
    sld_map_show,
    sld_section_angle,
    chk_torch_accel,
    progress_monte_carlo,
    btn_update_depth_maps,
)

vario_pane = pn.panel(plot_bottom_left_variogram, sizing_mode="stretch_width", min_height=250)
card_4 = create_sidebar_card("Variograms", sld_slope, sld_range, sld_sill, sld_nugget, btn_update_vario, vario_pane)

card_5 = create_sidebar_card("GRV and STOOIP", sld_thick_mean, sld_thick_std, sld_ntg, sld_poro, sld_sw, sld_fvf, btn_update_volumetrics)
card_6 = create_sidebar_card(
    "Report",
    txt_report_header,
    txt_report_filename,
    txt_report_location,
    sel_report_type,
    report_export_progress,
    btn_export_report,
    qc_export_status,
)
card_7 = create_sidebar_card(
    "Export to Petrel",
    txt_export_petrel_surface_name,
    chk_export_average_mean,
    chk_export_final_depth,
    chk_export_final_depth_contour,
    chk_export_spill_contour_line,
    chk_export_isoprobability,
    chk_export_all_realizations,
    btn_export_petrel_surface,
    petrel_export_status,
)
card_8 = create_sidebar_card(
    "General Setting",
    _general_setting_row("Velocity", rg_velocity_units),
    _general_setting_row("X / Y", rg_xy_units),
    _general_setting_row("Depth", rg_depth_units),
    _general_setting_row("Time", rg_time_units),
    _general_setting_row("Area", rg_area_units),
    _general_setting_row("Volume", rg_volume_units),
    btn_apply_general_settings,
)

sidebar_items = [card_1, card_wells, card_4, card_3, card_2, card_5, card_6, card_7, card_8]


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
logger.info("Log directory: %s", LOG_DIR)
if TEST_MODE:
    logger.info(
        "Test mode active: running without launcher/constants payload is supported; fallback test surface files or imported payload surfaces will be used when available."
    )
pn.template.FastListTemplate(**template_kwargs).servable()

# Working code!!!!! Feb 25th, 14:13
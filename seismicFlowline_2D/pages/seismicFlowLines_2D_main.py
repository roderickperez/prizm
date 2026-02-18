import os
import sys
import json
import ctypes
import time
import logging
import platform
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import panel as pn
import holoviews as hv
import torch
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from holoviews import streams
from bokeh.models import ColorBar
from scipy import signal, ndimage
from sklearn.cluster import KMeans

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from seismicAttributes.core.segy_loader import SegyDataStore
from seismicFlowline_2D.core.flowline_algorithms import (
    compute_gst_gpu,
    kmeans_torch,
    rk4_trace_vector_field,
)
from shared.ui.omv_theme import (
    BLUE_OMV_COLOR,
    DARK_BLUE_OMV_COLOR,
    GREEN_OMV_COLOR,
    MAGENTA_OMV_COLOR,
    NEON_OMV_COLOR,
    NEON_MAGENTA_OMV_COLOR,
    docs_button_html,
    get_content_text_color,
    get_dark_colorbar_opts,
    get_dark_select_stylesheets,
    get_dark_text_input_stylesheets,
    get_extension_raw_css,
    get_main_outer_background,
    get_neon_button_stylesheets,
    get_plot_surface_background,
    get_section_card_colors,
    get_slider_stylesheets,
    is_dark_mode_from_state,
)


# --- Legacy Petrel import retained for export behavior ---
try:
    pythontool_module = __import__("cegalprizm.pythontool", fromlist=["PetrelConnection"])
    PetrelConnection = getattr(pythontool_module, "PetrelConnection", None)
except Exception:
    PetrelConnection = None


def apply_dll_fix() -> None:
    if platform.system() != "Windows":
        return
    try:
        current_venv = sys.prefix
        paths_to_check = [
            os.path.join(current_venv, "Lib", "site-packages", "torch", "lib"),
            os.path.expanduser(r"~\py_pkgs\torch\lib"),
        ]
        for lib_path in paths_to_check:
            if not os.path.exists(lib_path):
                continue
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(lib_path)
                except Exception:
                    pass
            for dll in ("libiomp5md.dll", "c10.dll", "torch_python.dll"):
                dll_file = os.path.join(lib_path, dll)
                if os.path.exists(dll_file):
                    try:
                        ctypes.CDLL(dll_file)
                    except Exception:
                        pass
    except Exception:
        pass


apply_dll_fix()


APP_TITLE = "FlowLine 2D"
DOCUMENTATION_URL = "https://example.com/docs"
SEGY_FILE_PATH = ROOT_DIR / "testData" / "1_Original_Seismics.sgy"

ASSETS_DIR = ROOT_DIR / "images" / "OMV_brandLogoPack" / "OMV_brandLogoPack"
LOGO_PATH = ASSETS_DIR / "OMV_logo_Neon_Small.png"
FAVICON_PATH = ASSETS_DIR / "OMV_logo_Blue_Small.png"

valid_logo = str(LOGO_PATH) if LOGO_PATH.exists() else None
valid_favicon = str(FAVICON_PATH) if FAVICON_PATH.exists() else None

APP_DIR = Path(__file__).resolve().parent
LOG_DIR = APP_DIR / "logs"


is_dark_mode = is_dark_mode_from_state()
pn.extension("tabulator", raw_css=get_extension_raw_css(is_dark_mode))
hv.extension("bokeh")


def setup_flow_logger(log_dir: Path, logger_name: str = "seismic_flowlines_2d_app") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"flowlines_2d_{time.strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    existing_logs = sorted(log_dir.glob("flowlines_2d_*.log"), key=lambda p: p.stat().st_mtime)
    while len(existing_logs) > 10:
        oldest = existing_logs.pop(0)
        try:
            oldest.unlink()
        except OSError:
            pass

    logger.info("FlowLine logger initialized")
    return logger


def get_radio_group_stylesheets() -> list[str]:
    return [
        f"""
        .bk-btn-group > .bk-btn,
        .bk-btn-group > button.bk-btn {{
            background: {NEON_MAGENTA_OMV_COLOR} !important;
            background-color: {NEON_MAGENTA_OMV_COLOR} !important;
            color: {DARK_BLUE_OMV_COLOR} !important;
            border-color: {NEON_MAGENTA_OMV_COLOR} !important;
            font-weight: 600 !important;
        }}

        .bk-btn-group > .bk-btn.bk-active,
        .bk-btn-group > button.bk-btn.bk-active,
        .bk-btn-group > .bk-btn[aria-pressed="true"],
        .bk-btn-group > button.bk-btn[aria-pressed="true"] {{
            background: {MAGENTA_OMV_COLOR} !important;
            background-color: {MAGENTA_OMV_COLOR} !important;
            color: white !important;
            border-color: {MAGENTA_OMV_COLOR} !important;
        }}
        """
    ]


def get_cluster_numeric_input_stylesheets() -> list[str]:
    return [
        f"""
        :host {{
            --input-background: {GREEN_OMV_COLOR};
            --input-color: white;
            --input-border-color: {GREEN_OMV_COLOR};
        }}

        input,
        input[type="number"],
        .bk-input,
        .bk-input[type="number"],
        .bk-input-group input,
        .bk-input-group .bk-input {{
            background: {GREEN_OMV_COLOR} !important;
            background-color: {GREEN_OMV_COLOR} !important;
            color: white !important;
            border-color: {GREEN_OMV_COLOR} !important;
        }}

        .bk-Spinner button,
        .bk-Spinner .bk-btn,
        .bk-input-group .bk-Spinner button,
        .bk-input-group .bk-Spinner .bk-btn {{
            background: {GREEN_OMV_COLOR} !important;
            background-color: {GREEN_OMV_COLOR} !important;
            color: white !important;
            border-color: {GREEN_OMV_COLOR} !important;
        }}
        """
    ]


class SeismicFlowApp:
    def __init__(self):
        self.logger = setup_flow_logger(LOG_DIR)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_gpu = self.device.type == "cuda"
        self.logger.info(f"App start | device={self.device.type} | segy_path={SEGY_FILE_PATH}")

        self.session_data = self.load_session_data()
        self.cube_guid = self.session_data.get("selected_cube_guid", None)
        self.cube_name = self.session_data.get("selected_cube_name", SEGY_FILE_PATH.stem)

        self.data_store = SegyDataStore(SEGY_FILE_PATH, logger=None, cache_size=16)
        self.has_seismic_data = False
        self.inline_values = np.array([], dtype=int)
        self.xline_values = np.array([], dtype=int)
        self.inline_index_map: dict[int, int] = {}
        self.xline_index_map: dict[int, int] = {}
        self.dims = (0, 0, 0)
        self.amp_limit = 1000.0

        self.locked_slice: np.ndarray | None = None
        self.locked_meta: dict[str, str | int] = {}

        self.res_mag: np.ndarray | None = None
        self.res_ori: np.ndarray | None = None
        self.res_vec: np.ndarray | None = None
        self.flowlines: list[np.ndarray] = []
        self.flowline_ids = np.array([], dtype=int)
        self.seismic_cluster_map: np.ndarray | None = None
        self.unconformity_surfaces: list[np.ndarray] = []
        self.cluster_labels: np.ndarray | None = None

        self.select_stylesheets = get_dark_select_stylesheets(is_dark_mode)
        self.slider_stylesheets = get_slider_stylesheets()
        self.button_stylesheets = get_neon_button_stylesheets()
        self.text_input_stylesheets = get_dark_text_input_stylesheets(is_dark_mode)
        self.cluster_numeric_input_stylesheets = get_cluster_numeric_input_stylesheets()
        self.radio_group_stylesheets = get_radio_group_stylesheets()
        self.marker_opts = dict(color="red", size=8, line_color="white", line_width=1)
        self.applied_fl_color = "#FF0000"
        self.applied_fl_width = 1
        self._left_plot_figure = None
        self._right_plot_figure = None

        card_colors = get_section_card_colors(is_dark_mode)
        section_header_background = card_colors["header_background"]
        section_header_text = card_colors["header_text"]
        section_body_background = card_colors["body_background"]
        section_body_text = card_colors["body_text"]

        self.show_right_base = pn.widgets.Checkbox(name="Show Seismic (Right)", value=True)
        self.show_right_seis_cluster = pn.widgets.Checkbox(name="Show Clustered Seismic", value=True)
        self.show_right_attr = pn.widgets.Checkbox(name="Show Attribute Overlay", value=True)
        self.show_right_flowlines = pn.widgets.Checkbox(name="Show Flowlines", value=True)
        self.show_right_cluster_flowlines = pn.widgets.Checkbox(name="Show Clustered Flowlines", value=True)
        self.show_right_unconf = pn.widgets.Checkbox(name="Show Unconformities", value=True)

        self.load_segy_volume()

        self.radio_group = pn.widgets.RadioButtonGroup(
            name="Slice Type",
            options=["Inline", "Crossline"],
            value="Inline",
            button_type="default",
            sizing_mode="stretch_width",
            stylesheets=self.radio_group_stylesheets,
        )

        self.slice_slider = pn.widgets.IntSlider(
            name="Slice Index",
            start=0,
            end=max(0, self.dims[0] - 1),
            value=max(0, self.dims[0] // 2),
            sizing_mode="stretch_width",
            stylesheets=self.slider_stylesheets,
        )

        self.seismic_cmap = pn.widgets.Select(
            name="Seismic Colormap",
            options=["gray", "RdBu", "bwr", "PuOr", "viridis"],
            value="gray",
            sizing_mode="stretch_width",
            stylesheets=self.select_stylesheets,
        )

        self.slice_update_btn = pn.widgets.Button(
            name="Update Preview",
            button_type="default",
            sizing_mode="stretch_width",
            stylesheets=self.button_stylesheets,
        )

        self.extract_line_btn = pn.widgets.Button(
            name="Extract Current Section",
            button_type="default",
            sizing_mode="stretch_width",
            stylesheets=self.button_stylesheets,
        )
        self.extract_progress = pn.widgets.Progress(name="Extraction Progress", value=0, visible=False, sizing_mode="stretch_width")
        self.extract_status = pn.pane.Markdown(
            "No extraction performed.",
            styles={"font-size": "0.9em", "color": section_body_text if is_dark_mode else "#666"},
        )

        self.control_card = pn.Card(
            pn.Column(
                self.radio_group,
                self.slice_slider,
                self.seismic_cmap,
                self.slice_update_btn,
                self.show_right_base,
                pn.layout.Divider(),
                self.extract_line_btn,
                self.extract_progress,
                self.extract_status,
                sizing_mode="stretch_width",
            ),
            title="Slice Controls",
            collapsed=True,
            header_background=section_header_background,
            active_header_background=section_header_background,
            header_color=section_header_text,
            styles={"background": section_body_background, "color": section_body_text},
            sizing_mode="stretch_width",
        )

        self.gst_sigma = pn.widgets.FloatSlider(
            name="Smoothing (Sigma)",
            start=0.1,
            end=10.0,
            value=1.0,
            step=0.1,
            sizing_mode="stretch_width",
            stylesheets=self.slider_stylesheets,
        )
        self.gst_rho = pn.widgets.FloatSlider(
            name="Tensor Scale (Rho)",
            start=0.5,
            end=10.0,
            value=3.0,
            step=0.1,
            sizing_mode="stretch_width",
            stylesheets=self.slider_stylesheets,
        )
        self.gst_calc_btn = pn.widgets.Button(
            name="Calculate Gradient Structure Tensor",
            button_type="default",
            sizing_mode="stretch_width",
            stylesheets=self.button_stylesheets,
            disabled=True,
        )
        self.gst_progress = pn.widgets.Progress(name="GST Progress", value=0, visible=False, sizing_mode="stretch_width")
        self.gst_status = pn.pane.Markdown("", styles={"font-size": "0.8em"})

        self.gst_view_select = pn.widgets.Select(
            name="Attribute to Show",
            options=["Magnitude", "Orientation"],
            value="Magnitude",
            sizing_mode="stretch_width",
            stylesheets=self.select_stylesheets,
        )
        self.gst_cmap = pn.widgets.Select(
            name="Attribute Colormap",
            options=["viridis", "twilight", "magma", "inferno", "gray"],
            value="viridis",
            sizing_mode="stretch_width",
            stylesheets=self.select_stylesheets,
        )
        self.gst_opacity = pn.widgets.FloatSlider(
            name="Attribute Opacity",
            start=0.0,
            end=1.0,
            value=0.6,
            step=0.05,
            sizing_mode="stretch_width",
            stylesheets=self.slider_stylesheets,
        )
        self.gst_update_btn = pn.widgets.Button(
            name="Update GST Overlay",
            button_type="default",
            sizing_mode="stretch_width",
            stylesheets=self.button_stylesheets,
        )

        self.gst_card = pn.Card(
            pn.Column(
                self.gst_sigma,
                self.gst_rho,
                self.gst_calc_btn,
                self.gst_progress,
                self.gst_status,
                pn.layout.Divider(),
                self.gst_view_select,
                self.gst_cmap,
                self.gst_opacity,
                self.gst_update_btn,
                self.show_right_attr,
                sizing_mode="stretch_width",
            ),
            title="Gradient Structure Tensor",
            collapsed=True,
            header_background=section_header_background,
            active_header_background=section_header_background,
            header_color=section_header_text,
            styles={"background": section_body_background, "color": section_body_text},
            sizing_mode="stretch_width",
        )

        self.fl_step = pn.widgets.FloatInput(name="Integration Step", value=0.5, step=0.1, sizing_mode="stretch_width")
        self.fl_iters = pn.widgets.IntInput(name="Max Iterations", value=250, step=10, start=10, sizing_mode="stretch_width")
        self.fl_param_help = pn.pane.HTML(
            """
            <div style='font-size:0.86em; opacity:0.9;'>
                <b>ℹ️ Flowline parameter help</b><br>
                <span title='Numerical step size used while tracing each flowline. Smaller values produce smoother/more precise trajectories but take longer.'>
                    <b>Integration Step</b>
                </span>
                — tracing step size.<br>
                <span title='Maximum tracing iterations for each forward/backward direction. Higher values allow longer flowlines but increase runtime.'>
                    <b>Max Iterations</b>
                </span>
                — max tracing length.<br>
                <span title='Number of pixels between neighboring seed columns. Lower values generate denser flowlines and increase compute cost.'>
                    <b>Seed Density (Pixels)</b>
                </span>
                — spacing of starting seeds.
            </div>
            """,
            sizing_mode="stretch_width",
        )
        self.fl_density = pn.widgets.IntSlider(
            name="Seed Density (Pixels)",
            start=5,
            end=100,
            value=20,
            sizing_mode="stretch_width",
            stylesheets=self.slider_stylesheets,
        )
        self.fl_max_lines = pn.widgets.IntInput(
            name="Max Flowlines (0 = no limit)",
            value=0,
            step=100,
            start=0,
            sizing_mode="stretch_width",
        )
        self.fl_color_mode = pn.widgets.RadioButtonGroup(
            name="Flowline Coloring",
            options=["Solid Color", "Colorbar"],
            value="Solid Color",
            button_type="default",
            sizing_mode="stretch_width",
            stylesheets=self.radio_group_stylesheets,
        )
        self.fl_color = pn.widgets.ColorPicker(name="Line Color", value="#FF0000", sizing_mode="stretch_width")
        self.fl_cmap = pn.widgets.Select(
            name="Flowline Colormap",
            options=["jet", "viridis", "plasma", "inferno", "magma", "cividis", "turbo"],
            value="jet",
            sizing_mode="stretch_width",
            stylesheets=self.select_stylesheets,
        )
        self.fl_width = pn.widgets.IntSlider(
            name="Line Thickness",
            start=1,
            end=5,
            value=1,
            step=1,
            sizing_mode="stretch_width",
            stylesheets=self.slider_stylesheets,
        )
        self.fl_apply_btn = pn.widgets.Button(
            name="Update Flowlines",
            button_type="default",
            sizing_mode="stretch_width",
            stylesheets=self.button_stylesheets,
            disabled=True,
        )
        self.fl_calc_btn = pn.widgets.Button(
            name="Extract Flowlines",
            button_type="default",
            sizing_mode="stretch_width",
            stylesheets=self.button_stylesheets,
            disabled=True,
        )
        self.fl_progress = pn.widgets.Progress(name="Flowline Progress", value=0, visible=False, sizing_mode="stretch_width")
        self.fl_count_label = pn.pane.Markdown("Flowlines Extracted: 0", styles={"font-size": "0.9em", "font-weight": "bold"})
        self.flowline_id_range = pn.widgets.IntRangeSlider(
            name="Flowline ID Range",
            start=0,
            end=1,
            value=(0, 0),
            step=1,
            disabled=True,
            sizing_mode="stretch_width",
            stylesheets=self.slider_stylesheets,
        )
        self.flowline_sel_label = pn.pane.Markdown(
            "Selected Flowlines: 0",
            styles={"font-size": "0.9em", "font-weight": "bold"},
        )

        for widget in (self.fl_step, self.fl_iters, self.fl_max_lines):
            widget.stylesheets = self.text_input_stylesheets

        self.flow_card = pn.Card(
            pn.Column(
                self.fl_param_help,
                self.fl_step,
                self.fl_iters,
                self.fl_density,
                self.fl_max_lines,
                self.fl_calc_btn,
                self.fl_progress,
                self.fl_count_label,
                self.flowline_id_range,
                self.flowline_sel_label,
                pn.layout.Divider(),
                self.fl_color_mode,
                self.fl_color,
                self.fl_cmap,
                self.fl_width,
                self.fl_apply_btn,
                self.show_right_flowlines,
                sizing_mode="stretch_width",
            ),
            title="Extract Flowlines",
            collapsed=True,
            header_background=section_header_background,
            active_header_background=section_header_background,
            header_color=section_header_text,
            styles={"background": section_body_background, "color": section_body_text},
            sizing_mode="stretch_width",
        )

        self.cl_k = pn.widgets.IntInput(name="Number of Clusters (K)", value=10, start=2, step=1, sizing_mode="stretch_width")
        self.cl_cmap = pn.widgets.Select(
            name="Cluster Colormap",
            options=["tab10", "Set1", "Paired", "Accent"],
            value="tab10",
            sizing_mode="stretch_width",
            stylesheets=self.select_stylesheets,
        )
        self.cl_width = pn.widgets.IntSlider(
            name="Cluster Line Thickness",
            start=1,
            end=5,
            value=1,
            step=1,
            sizing_mode="stretch_width",
            stylesheets=self.slider_stylesheets,
        )
        self.cl_calc_btn = pn.widgets.Button(
            name="Cluster Flowlines",
            button_type="default",
            sizing_mode="stretch_width",
            stylesheets=self.button_stylesheets,
            disabled=True,
        )
        self.cl_progress = pn.widgets.Progress(name="Clustering Progress", value=0, visible=False, sizing_mode="stretch_width")

        self.cluster_card = pn.Card(
            pn.Column(
                self.cl_k,
                self.cl_cmap,
                self.cl_width,
                self.cl_calc_btn,
                self.cl_progress,
                self.show_right_cluster_flowlines,
                sizing_mode="stretch_width",
            ),
            title="Flowline Clustering",
            collapsed=True,
            header_background=section_header_background,
            active_header_background=section_header_background,
            header_color=section_header_text,
            styles={"background": section_body_background, "color": section_body_text},
            sizing_mode="stretch_width",
        )

        self.sc_k = pn.widgets.IntInput(name="Number of Clusters (K)", value=8, start=2, step=1, sizing_mode="stretch_width")
        self.sc_cmap = pn.widgets.Select(
            name="Seismic Cluster Colormap",
            options=["jet", "viridis", "tab10", "turbo", "nipy_spectral"],
            value="jet",
            sizing_mode="stretch_width",
            stylesheets=self.select_stylesheets,
        )
        self.sc_alpha = pn.widgets.FloatSlider(
            name="Cluster Overlay Opacity",
            start=0.1,
            end=1.0,
            value=0.6,
            step=0.05,
            sizing_mode="stretch_width",
            stylesheets=self.slider_stylesheets,
        )
        self.sc_calc_btn = pn.widgets.Button(
            name="Cluster Seismic",
            button_type="default",
            sizing_mode="stretch_width",
            stylesheets=self.button_stylesheets,
            disabled=True,
        )
        self.sc_progress = pn.widgets.Progress(name="Seismic Cluster Progress", value=0, visible=False, sizing_mode="stretch_width")

        for widget in (self.cl_k, self.sc_k):
            widget.stylesheets = self.cluster_numeric_input_stylesheets

        self.seismic_cluster_card = pn.Card(
            pn.Column(
                self.sc_k,
                self.sc_cmap,
                self.sc_alpha,
                self.sc_calc_btn,
                self.sc_progress,
                self.show_right_seis_cluster,
                sizing_mode="stretch_width",
            ),
            title="Cluster Seismic",
            collapsed=True,
            header_background=section_header_background,
            active_header_background=section_header_background,
            header_color=section_header_text,
            styles={"background": section_body_background, "color": section_body_text},
            sizing_mode="stretch_width",
        )

        self.uc_thresh = pn.widgets.FloatSlider(
            name="Convergence Margin (px)",
            start=2.0,
            end=50.0,
            value=10.0,
            step=0.5,
            sizing_mode="stretch_width",
            stylesheets=self.slider_stylesheets,
        )
        self.uc_count = pn.widgets.IntSlider(
            name="Major Surfaces",
            start=1,
            end=100,
            value=20,
            step=1,
            sizing_mode="stretch_width",
            stylesheets=self.slider_stylesheets,
        )
        self.uc_color = pn.widgets.ColorPicker(name="Unconformity Color", value="#32CD32", sizing_mode="stretch_width")
        self.uc_width = pn.widgets.IntSlider(
            name="Thickness",
            start=1,
            end=5,
            value=2,
            step=1,
            sizing_mode="stretch_width",
            stylesheets=self.slider_stylesheets,
        )
        self.uc_calc_btn = pn.widgets.Button(
            name="Extract Unconformities",
            button_type="default",
            sizing_mode="stretch_width",
            stylesheets=self.button_stylesheets,
            disabled=True,
        )
        self.uc_apply_btn = pn.widgets.Button(
            name="Update Unconformities",
            button_type="default",
            sizing_mode="stretch_width",
            stylesheets=self.button_stylesheets,
            disabled=True,
        )
        self.uc_progress = pn.widgets.Progress(name="Unconformity Progress", value=0, visible=False, sizing_mode="stretch_width")

        self.unconf_card = pn.Card(
            pn.Column(
                self.uc_thresh,
                self.uc_count,
                self.uc_calc_btn,
                self.uc_progress,
                pn.layout.Divider(),
                self.uc_color,
                self.uc_width,
                self.uc_apply_btn,
                self.show_right_unconf,
                sizing_mode="stretch_width",
            ),
            title="Unconformities",
            collapsed=True,
            header_background=section_header_background,
            active_header_background=section_header_background,
            header_color=section_header_text,
            styles={"background": section_body_background, "color": section_body_text},
            sizing_mode="stretch_width",
        )

        self.exp_gst_mag = pn.widgets.Checkbox(name="GST Magnitude", value=True)
        self.exp_gst_ori = pn.widgets.Checkbox(name="GST Orientation", value=False)
        self.exp_orig = pn.widgets.Checkbox(name="Original 2D (Extraction)", value=False)
        self.exp_fl = pn.widgets.Checkbox(name="2D Flowlines (Single)", value=False)
        self.exp_cl = pn.widgets.Checkbox(name="2D Grouped Flowlines", value=False)
        self.exp_uc = pn.widgets.Checkbox(name="2D Unconformities", value=False)
        self.exp_name = pn.widgets.TextInput(name="Export Suffix", value="_Result", sizing_mode="stretch_width")
        self.exp_btn = pn.widgets.Button(
            name="Export to Petrel",
            button_type="default",
            sizing_mode="stretch_width",
            stylesheets=self.button_stylesheets,
            disabled=True,
        )
        self.exp_progress = pn.widgets.Progress(name="Export Progress", value=0, visible=False, sizing_mode="stretch_width")
        self.exp_status = pn.pane.Markdown("Ready", styles={"font-size": "0.9em"})

        self.exp_name.stylesheets = self.text_input_stylesheets

        self.export_card = pn.Card(
            pn.Column(
                self.exp_gst_mag,
                self.exp_gst_ori,
                self.exp_orig,
                self.exp_fl,
                self.exp_cl,
                self.exp_uc,
                self.exp_name,
                self.exp_btn,
                self.exp_progress,
                self.exp_status,
                sizing_mode="stretch_width",
            ),
            title="Export",
            collapsed=True,
            header_background=section_header_background,
            active_header_background=section_header_background,
            header_color=section_header_text,
            styles={"background": section_body_background, "color": section_body_text},
            sizing_mode="stretch_width",
        )

        self.tap_stream = streams.Tap(x=None, y=None)
        self.marker_dmap = hv.DynamicMap(self.get_marker, streams=[self.tap_stream])

        pane_opts = dict(sizing_mode="stretch_both", min_height=0)
        self.pane_left = pn.pane.HoloViews(object=None, **pane_opts)
        self.pane_right = pn.pane.HoloViews(object=None, **pane_opts)

        pane_styles = {
            "height": "100%",
            "min-height": "0",
            "overflow": "hidden",
            "display": "flex",
            "flex-direction": "column",
            "flex": "1 1 auto",
        }
        self.left_plot_pane = pn.Column(self.pane_left, sizing_mode="stretch_both", styles=pane_styles)
        self.right_plot_pane = pn.Column(self.pane_right, sizing_mode="stretch_both", styles=pane_styles)

        self.radio_group.param.watch(self.update_slider_limits, "value")
        self.slice_update_btn.on_click(self.run_preview)
        self.extract_line_btn.on_click(self.run_extraction)

        self.gst_calc_btn.on_click(self.run_gst_engine)
        self.gst_update_btn.on_click(self.update_view)
        self.gst_view_select.param.watch(self.auto_cmap_gst, "value")

        self.fl_calc_btn.on_click(self.run_flowline_engine)
        self.fl_apply_btn.on_click(self.apply_flowline_style)
        self.fl_color_mode.param.watch(self._update_flowline_style_visibility, "value")
        self.flowline_id_range.param.watch(self._on_flowline_range_change, "value")
        self.show_right_base.param.watch(self.update_view, "value")
        self.show_right_seis_cluster.param.watch(self.update_view, "value")
        self.show_right_attr.param.watch(self.update_view, "value")
        self.show_right_flowlines.param.watch(self.update_view, "value")
        self.show_right_cluster_flowlines.param.watch(self.update_view, "value")
        self.show_right_unconf.param.watch(self.update_view, "value")
        self.cl_calc_btn.on_click(self.run_clustering_engine)
        self.sc_calc_btn.on_click(self.run_seismic_clustering_engine)
        self.uc_calc_btn.on_click(self.run_unconf_engine)
        self.uc_apply_btn.on_click(self.apply_unconf_style)

        self.exp_btn.on_click(self.run_export)

        self.update_slider_limits(None)
        self._update_flowline_style_visibility(None)
        self.applied_uc_color = self.uc_color.value
        self.applied_uc_width = int(self.uc_width.value)
        restored = self._restore_state_from_cache()
        if restored and self.locked_slice is not None:
            self.update_view()
        else:
            self.run_preview(None)

    def load_session_data(self) -> dict:
        data_file = os.environ.get("PWR_DATA_FILE")
        if data_file and os.path.exists(data_file):
            try:
                with open(data_file, "r", encoding="utf-8") as file_handle:
                    return json.load(file_handle)
            except Exception:
                return {}
        return {}

    def _state_cache_key(self) -> str:
        cube_part = str(self.cube_guid or self.cube_name or "default")
        return f"seismic_flowlines_2d_state::{cube_part}"

    def _save_state_to_cache(self) -> None:
        try:
            pn.state.cache[self._state_cache_key()] = {
                "locked_slice": None if self.locked_slice is None else np.array(self.locked_slice, copy=True),
                "locked_meta": dict(self.locked_meta),
                "res_mag": None if self.res_mag is None else np.array(self.res_mag, copy=True),
                "res_ori": None if self.res_ori is None else np.array(self.res_ori, copy=True),
                "res_vec": None if self.res_vec is None else np.array(self.res_vec, copy=True),
                "flowlines": [np.array(line, copy=True) for line in self.flowlines],
                "flowline_ids": np.array(self.flowline_ids, copy=True),
                "seismic_cluster_map": None if self.seismic_cluster_map is None else np.array(self.seismic_cluster_map, copy=True),
                "unconformity_surfaces": [np.array(line, copy=True) for line in self.unconformity_surfaces],
                "cluster_labels": None if self.cluster_labels is None else np.array(self.cluster_labels, copy=True),
                "applied_fl_color": self.applied_fl_color,
                "applied_fl_width": int(self.applied_fl_width),
                "applied_uc_color": self.applied_uc_color,
                "applied_uc_width": int(self.applied_uc_width),
                "ui": {
                    "radio_group": self.radio_group.value,
                    "slice_slider": int(self.slice_slider.value),
                    "seismic_cmap": self.seismic_cmap.value,
                    "gst_sigma": float(self.gst_sigma.value),
                    "gst_rho": float(self.gst_rho.value),
                    "gst_view_select": self.gst_view_select.value,
                    "gst_cmap": self.gst_cmap.value,
                    "gst_opacity": float(self.gst_opacity.value),
                    "fl_step": float(self.fl_step.value),
                    "fl_iters": int(self.fl_iters.value),
                    "fl_density": int(self.fl_density.value),
                    "fl_max_lines": int(self.fl_max_lines.value),
                    "fl_color_mode": self.fl_color_mode.value,
                    "fl_color": self.fl_color.value,
                    "fl_cmap": self.fl_cmap.value,
                    "fl_width": int(self.fl_width.value),
                    "flowline_id_range": tuple(self.flowline_id_range.value),
                    "cl_k": int(self.cl_k.value),
                    "cl_cmap": self.cl_cmap.value,
                    "cl_width": int(self.cl_width.value),
                    "sc_k": int(self.sc_k.value),
                    "sc_cmap": self.sc_cmap.value,
                    "sc_alpha": float(self.sc_alpha.value),
                    "uc_thresh": float(self.uc_thresh.value),
                    "uc_count": int(self.uc_count.value),
                    "uc_color": self.uc_color.value,
                    "uc_width": int(self.uc_width.value),
                    "show_right_base": bool(self.show_right_base.value),
                    "show_right_seis_cluster": bool(self.show_right_seis_cluster.value),
                    "show_right_attr": bool(self.show_right_attr.value),
                    "show_right_flowlines": bool(self.show_right_flowlines.value),
                    "show_right_cluster_flowlines": bool(self.show_right_cluster_flowlines.value),
                    "show_right_unconf": bool(self.show_right_unconf.value),
                },
            }
        except Exception as exc:
            self.logger.warning(f"[State] Save failed: {exc}")

    def _restore_state_from_cache(self) -> bool:
        cached = pn.state.cache.get(self._state_cache_key())
        if not isinstance(cached, dict):
            return False

        try:
            ui = cached.get("ui", {})
            if ui:
                self.radio_group.value = ui.get("radio_group", self.radio_group.value)
                self.update_slider_limits(None)

                self.slice_slider.value = int(np.clip(
                    ui.get("slice_slider", self.slice_slider.value),
                    self.slice_slider.start,
                    self.slice_slider.end,
                ))

                self.seismic_cmap.value = ui.get("seismic_cmap", self.seismic_cmap.value)
                self.gst_sigma.value = ui.get("gst_sigma", self.gst_sigma.value)
                self.gst_rho.value = ui.get("gst_rho", self.gst_rho.value)
                self.gst_view_select.value = ui.get("gst_view_select", self.gst_view_select.value)
                self.gst_cmap.value = ui.get("gst_cmap", self.gst_cmap.value)
                self.gst_opacity.value = ui.get("gst_opacity", self.gst_opacity.value)
                self.fl_step.value = ui.get("fl_step", self.fl_step.value)
                self.fl_iters.value = ui.get("fl_iters", self.fl_iters.value)
                self.fl_density.value = ui.get("fl_density", self.fl_density.value)
                self.fl_max_lines.value = ui.get("fl_max_lines", self.fl_max_lines.value)
                self.fl_color_mode.value = ui.get("fl_color_mode", self.fl_color_mode.value)
                self.fl_color.value = ui.get("fl_color", self.fl_color.value)
                self.fl_cmap.value = ui.get("fl_cmap", self.fl_cmap.value)
                self.fl_width.value = ui.get("fl_width", self.fl_width.value)
                self.cl_k.value = ui.get("cl_k", self.cl_k.value)
                self.cl_cmap.value = ui.get("cl_cmap", self.cl_cmap.value)
                self.cl_width.value = ui.get("cl_width", self.cl_width.value)
                self.sc_k.value = ui.get("sc_k", self.sc_k.value)
                self.sc_cmap.value = ui.get("sc_cmap", self.sc_cmap.value)
                self.sc_alpha.value = ui.get("sc_alpha", self.sc_alpha.value)
                self.uc_thresh.value = ui.get("uc_thresh", self.uc_thresh.value)
                self.uc_count.value = ui.get("uc_count", self.uc_count.value)
                self.uc_color.value = ui.get("uc_color", self.uc_color.value)
                self.uc_width.value = ui.get("uc_width", self.uc_width.value)
                self.show_right_base.value = ui.get("show_right_base", self.show_right_base.value)
                self.show_right_seis_cluster.value = ui.get("show_right_seis_cluster", self.show_right_seis_cluster.value)
                self.show_right_attr.value = ui.get("show_right_attr", self.show_right_attr.value)
                self.show_right_flowlines.value = ui.get("show_right_flowlines", self.show_right_flowlines.value)
                self.show_right_cluster_flowlines.value = ui.get("show_right_cluster_flowlines", self.show_right_cluster_flowlines.value)
                self.show_right_unconf.value = ui.get("show_right_unconf", self.show_right_unconf.value)

            self.locked_slice = cached.get("locked_slice")
            self.locked_meta = dict(cached.get("locked_meta", {}))
            self.res_mag = cached.get("res_mag")
            self.res_ori = cached.get("res_ori")
            self.res_vec = cached.get("res_vec")
            self.flowlines = list(cached.get("flowlines", []))
            self.flowline_ids = np.array(cached.get("flowline_ids", np.array([], dtype=int)), copy=True)
            self.seismic_cluster_map = cached.get("seismic_cluster_map")
            self.unconformity_surfaces = list(cached.get("unconformity_surfaces", []))
            self.cluster_labels = cached.get("cluster_labels")

            self.applied_fl_color = cached.get("applied_fl_color", self.fl_color.value)
            self.applied_fl_width = int(cached.get("applied_fl_width", int(self.fl_width.value)))
            self.applied_uc_color = cached.get("applied_uc_color", self.uc_color.value)
            self.applied_uc_width = int(cached.get("applied_uc_width", int(self.uc_width.value)))

            self.fl_count_label.object = f"Flowlines Extracted: {len(self.flowlines)}"
            self._update_flowline_selector_limits()
            saved_range = ui.get("flowline_id_range") if isinstance(ui, dict) else None
            if saved_range and not self.flowline_id_range.disabled:
                low, high = int(saved_range[0]), int(saved_range[1])
                low = max(self.flowline_id_range.start, min(low, self.flowline_id_range.end))
                high = max(self.flowline_id_range.start, min(high, self.flowline_id_range.end))
                if high < low:
                    low, high = high, low
                self.flowline_id_range.value = (low, high)

            self.gst_calc_btn.disabled = self.locked_slice is None
            has_gst = self.res_ori is not None and self.res_mag is not None
            has_flowlines = len(self.flowlines) > 0
            self.fl_calc_btn.disabled = not has_gst
            self.fl_apply_btn.disabled = not has_flowlines
            self.cl_calc_btn.disabled = not has_flowlines
            self.sc_calc_btn.disabled = not has_flowlines
            self.uc_calc_btn.disabled = not has_flowlines
            self.uc_apply_btn.disabled = len(self.unconformity_surfaces) == 0
            self.exp_btn.disabled = not has_gst

            if self.locked_slice is not None and self.locked_meta:
                mode = self.locked_meta.get("mode", "Inline")
                idx = self.locked_meta.get("idx", self.slice_slider.value)
                self.extract_status.object = f"Locked {mode} {idx}. Restored from theme switch/session."

            self._update_flowline_style_visibility(None)
            return True
        except Exception as exc:
            self.logger.warning(f"[State] Restore failed: {exc}")
            return False

    # NOTE: Current seismic loading from Petrel has been intentionally replaced by SEG-Y loading.
    # Legacy approach (kept as comment for reference):
    # - read cube guid from session
    # - PetrelConnection(...).get_petrelobjects_by_guids(...)
    # - cube.chunk(...) per selected slice

    def load_segy_volume(self) -> None:
        self.logger.info("[SEGY] Loading SEG-Y volume")
        self.has_seismic_data = self.data_store.load()
        if not self.has_seismic_data:
            self.logger.warning("[SEGY] Volume load failed")
            return
        self.inline_values = self.data_store.inline_values
        self.xline_values = self.data_store.xline_values
        self.inline_index_map = self.data_store.inline_index_map
        self.xline_index_map = self.data_store.xline_index_map
        self.dims = self.data_store.dims
        self.amp_limit = self.data_store.amp_limit
        self.logger.info(f"[SEGY] Loaded dims={self.dims} amp_limit={self.amp_limit:.3f}")

    def _report_progress(self, stage: str, value: int) -> None:
        value_int = int(max(0, min(100, value)))
        print(f"[{stage}] {value_int}%")
        self.logger.info(f"[{stage}] {value_int}%")

    def update_slider_limits(self, event) -> None:
        mode = self.radio_group.value if event is None else event.new
        values = self.inline_values if mode == "Inline" else self.xline_values
        if values.size == 0:
            self.slice_slider.start = 0
            self.slice_slider.end = 0
            self.slice_slider.value = 0
            return
        self.slice_slider.start = int(values.min())
        self.slice_slider.end = int(values.max())
        self.slice_slider.value = int(values[len(values) // 2])

    def auto_cmap_gst(self, event) -> None:
        self.gst_cmap.value = "viridis" if event.new == "Magnitude" else "twilight"

    def _update_flowline_style_visibility(self, _event=None) -> None:
        is_colorbar = self.fl_color_mode.value == "Colorbar"
        self.fl_color.visible = not is_colorbar
        self.fl_cmap.visible = is_colorbar

    def _update_flowline_selector_limits(self) -> None:
        n_lines = len(self.flowlines)
        if n_lines == 0 or self.flowline_ids.size == 0:
            self.flowline_id_range.start = 0
            self.flowline_id_range.end = 1
            self.flowline_id_range.value = (0, 0)
            self.flowline_id_range.disabled = True
            self.flowline_sel_label.object = "Selected Flowlines: 0"
            return

        min_id = int(np.min(self.flowline_ids))
        max_id = int(np.max(self.flowline_ids))
        self.flowline_id_range.start = min_id
        self.flowline_id_range.end = max_id if max_id > min_id else (min_id + 1)
        default_count = min(200, n_lines)
        self.flowline_id_range.value = (min_id, min(max_id, min_id + default_count - 1))
        self.flowline_id_range.disabled = False
        low_id, high_id = self.flowline_id_range.value
        self.flowline_sel_label.object = f"Selected Flowlines: {high_id - low_id + 1}"

    def _on_flowline_range_change(self, _event=None) -> None:
        if not self.flowlines:
            self.flowline_sel_label.object = "Selected Flowlines: 0"
            return
        selected = self._get_selected_flowline_indices()
        self.flowline_sel_label.object = f"Selected Flowlines: {len(selected)}"
        self.update_view()

    def _get_selected_flowline_indices(self) -> np.ndarray:
        if not self.flowlines or self.flowline_ids.size == 0:
            return np.array([], dtype=int)
        low, high = self.flowline_id_range.value
        low = max(int(np.min(self.flowline_ids)), int(low))
        high = min(int(np.max(self.flowline_ids)), int(high))
        if high < low:
            low, high = high, low
        selected_mask = (self.flowline_ids >= low) & (self.flowline_ids <= high)
        return np.where(selected_mask)[0].astype(int)

    def _get_selected_flowlines(self) -> list[np.ndarray]:
        selected_indices = self._get_selected_flowline_indices()
        return [self.flowlines[int(idx)] for idx in selected_indices]

    def _line_shape_feature(self, line: np.ndarray, n_samples: int = 64) -> np.ndarray:
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

    def _extend_line_to_full_width(self, line: np.ndarray, width: int, height: int) -> np.ndarray:
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

    def get_marker(self, x, y):
        if x is None or y is None:
            return hv.Points([]).opts(**self.marker_opts)
        return hv.Points([(x, y)]).opts(**self.marker_opts)

    def _apply_dark_plot_theme(self, plot, _element) -> None:
        if not is_dark_mode:
            return
        plot_bg = get_plot_surface_background(is_dark_mode)
        fig = plot.state
        fig.background_fill_color = plot_bg
        fig.border_fill_color = plot_bg
        fig.frame_fill_color = plot_bg
        fig.outline_line_color = plot_bg
        fig.title.text_color = "white"

        for axis in fig.xaxis + fig.yaxis:
            axis.axis_label_text_color = "white"
            axis.major_label_text_color = "white"
            axis.major_tick_line_color = "white"
            axis.minor_tick_line_color = "white"
            axis.axis_line_color = "white"

        for grid in fig.xgrid + fig.ygrid:
            grid.grid_line_color = "#3b4d7a"

        for panel in list(fig.right) + list(fig.left) + list(fig.above) + list(fig.below):
            if hasattr(panel, "background_fill_color"):
                panel.background_fill_color = plot_bg
            if hasattr(panel, "background_fill_alpha"):
                panel.background_fill_alpha = 1.0
            if hasattr(panel, "border_line_color"):
                panel.border_line_color = plot_bg
            if isinstance(panel, ColorBar):
                panel.major_label_text_color = "white"
                panel.title_text_color = "white"
                panel.major_tick_line_color = "white"
                panel.minor_tick_line_color = "white"
                panel.bar_line_color = "white"

    def _prepare_visual_slice(self, mode: str, idx: int) -> tuple[np.ndarray, str, str]:
        raw_data, display_data, labels = self.data_store.get_slice(mode, int(idx))
        x_dim, y_dim = labels
        vis = np.swapaxes(display_data, 0, 1)
        vis = vis[::-1, :]
        return vis.astype(np.float32), x_dim, y_dim

    def _link_plot_ranges(self) -> None:
        if self._left_plot_figure is None or self._right_plot_figure is None:
            return
        left_fig = self._left_plot_figure
        right_fig = self._right_plot_figure
        if right_fig.x_range is not left_fig.x_range:
            right_fig.x_range = left_fig.x_range
        if right_fig.y_range is not left_fig.y_range:
            right_fig.y_range = left_fig.y_range

    def _capture_left_plot(self, plot, _element) -> None:
        self._left_plot_figure = plot.state
        self._link_plot_ranges()

    def _capture_right_plot(self, plot, _element) -> None:
        self._right_plot_figure = plot.state
        self._link_plot_ranges()

    def _show_empty_state(self, message: str) -> None:
        self._left_plot_figure = None
        self._right_plot_figure = None
        plot_bg = get_plot_surface_background(is_dark_mode)
        txt_color = get_content_text_color(is_dark_mode)
        empty_plot = hv.Text(0.5, 0.5, message).opts(
            xaxis=None,
            yaxis=None,
            toolbar=None,
            width=700,
            height=500,
            text_color=txt_color,
            bgcolor=plot_bg,
            fontsize=14,
            hooks=[self._apply_dark_plot_theme],
        )
        self.pane_left.object = empty_plot
        self.pane_right.object = empty_plot

    def run_preview(self, event) -> None:
        if not self.has_seismic_data:
            self._show_empty_state("Seismic volume not loaded. Verify segyio and testData path.")
            return
        try:
            mode = self.radio_group.value
            idx = int(self.slice_slider.value)
            data, x_dim, y_dim = self._prepare_visual_slice(mode, idx)
            h, w = data.shape
            bounds = (0, 0, w, h)

            plot_bg = get_plot_surface_background(is_dark_mode)
            colorbar_opts = get_dark_colorbar_opts(is_dark_mode)

            img = hv.Image(data, bounds=bounds, kdims=[x_dim, y_dim]).opts(
                cmap=self.seismic_cmap.value,
                clim=(-self.amp_limit, self.amp_limit),
                invert_yaxis=True,
                responsive=True,
                toolbar="above",
                colorbar=True,
                colorbar_opts=colorbar_opts,
                bgcolor=plot_bg,
                title=f"Preview {mode} {idx}",
                hooks=[self._apply_dark_plot_theme, self._capture_left_plot],
            )
            self.tap_stream.source = img
            self.pane_left.object = (img * self.marker_dmap).opts(responsive=True)
        except Exception as exc:
            self.logger.exception(f"[Preview] Error: {exc}")
            self._show_empty_state(f"Preview error: {exc}")

    def run_extraction(self, event) -> None:
        if not self.has_seismic_data:
            self.extract_status.object = "SEGY volume unavailable."
            return

        self.extract_line_btn.disabled = True
        self.extract_progress.visible = True
        self.extract_progress.value = 10
        self._report_progress("Extraction", 10)
        self.extract_status.object = "Extracting selected section..."
        try:
            mode = self.radio_group.value
            idx = int(self.slice_slider.value)
            self.extract_progress.value = 40
            self._report_progress("Extraction", 40)
            data, x_dim, y_dim = self._prepare_visual_slice(mode, idx)

            self.locked_slice = data
            self.locked_meta = {"mode": mode, "idx": idx, "xl": x_dim, "yl": y_dim}
            self.extract_progress.value = 70
            self._report_progress("Extraction", 70)

            self.res_mag = None
            self.res_ori = None
            self.res_vec = None
            self.flowlines = []
            self.flowline_ids = np.array([], dtype=int)
            self.seismic_cluster_map = None
            self.unconformity_surfaces = []
            self.cluster_labels = None
            self._update_flowline_selector_limits()

            self.gst_calc_btn.disabled = False
            self.fl_calc_btn.disabled = True
            self.fl_apply_btn.disabled = True
            self.cl_calc_btn.disabled = True
            self.sc_calc_btn.disabled = True
            self.uc_calc_btn.disabled = True
            self.uc_apply_btn.disabled = True
            self.exp_btn.disabled = True
            self.fl_count_label.object = "Flowlines Extracted: 0"

            self.extract_status.object = f"Locked {mode} {idx}. Ready for GST."
            self.logger.info(f"[Extraction] Locked slice mode={mode} idx={idx} shape={data.shape}")
            self.run_preview(None)
            self.update_view()
            self._save_state_to_cache()
            self.extract_progress.value = 100
            self._report_progress("Extraction", 100)
        except Exception as exc:
            self.logger.exception(f"[Extraction] Error: {exc}")
            self.extract_status.object = f"Error: {exc}"
        finally:
            self.extract_line_btn.disabled = False
            self.extract_progress.visible = False

    def run_gst_engine(self, event) -> None:
        if self.locked_slice is None:
            return

        self.gst_calc_btn.disabled = True
        self.gst_progress.visible = True
        self.gst_progress.value = 10
        self.gst_status.object = "Computing GST..."
        self._report_progress("GST", 10)
        self.logger.info(f"[GST] Starting | device={self.device.type} | shape={self.locked_slice.shape}")

        doc = pn.state.curdoc

        def ui_call(fn):
            if doc is not None:
                doc.add_next_tick_callback(fn)

        def worker() -> None:
            try:
                def gst_progress(value: int) -> None:
                    ui_call(lambda: setattr(self.gst_progress, "value", value))
                    self._report_progress("GST", value)

                mag, ori = compute_gst_gpu(
                    self.locked_slice,
                    rho=float(self.gst_rho.value),
                    progress_callback=gst_progress,
                    log_callback=self.logger.info,
                )

                vec_x = -np.sin(ori)
                vec_y = np.cos(ori)
                flip_mask = vec_x < 0
                vec_x = np.where(flip_mask, -vec_x, vec_x)
                vec_y = np.where(flip_mask, -vec_y, vec_y)
                vec = np.stack([vec_y, vec_x]).astype(np.float32)

                def on_done():
                    self.res_mag = mag
                    self.res_ori = ori
                    self.res_vec = vec
                    self.gst_progress.value = 100
                    self.gst_status.object = "GST complete."
                    self.fl_calc_btn.disabled = False
                    self.exp_btn.disabled = False
                    self.update_view()
                    self._save_state_to_cache()
                    self._report_progress("GST", 100)
                    self.logger.info("[GST] Completed")

                ui_call(on_done)
            except Exception as exc:
                self.logger.exception(f"[GST] Error: {exc}")
                ui_call(lambda: setattr(self.gst_status, "object", f"Error: {exc}"))
            finally:
                ui_call(lambda: setattr(self.gst_progress, "visible", False))
                ui_call(lambda: setattr(self.gst_calc_btn, "disabled", False))

        threading.Thread(target=worker, daemon=True).start()

    def run_flowline_engine(self, event) -> None:
        if self.locked_slice is None or self.res_ori is None:
            return

        self.fl_calc_btn.disabled = True
        self.fl_progress.visible = True
        self.fl_progress.value = 10
        self._report_progress("Flowlines", 10)
        self.logger.info(f"[Flowlines] Starting | device={self.device.type}")

        doc = pn.state.curdoc

        def ui_call(fn):
            if doc is not None:
                doc.add_next_tick_callback(fn)

        def worker() -> None:
            try:
                h, w = self.res_ori.shape
                density = max(2, int(self.fl_density.value))
                max_lines = max(0, int(self.fl_max_lines.value))
                mode = "both"

                vector_array = self.res_vec
                if vector_array is None:
                    vec_x = -np.sin(self.res_ori)
                    vec_y = np.cos(self.res_ori)
                    flip_mask = vec_x < 0
                    vec_x = np.where(flip_mask, -vec_x, vec_x)
                    vec_y = np.where(flip_mask, -vec_y, vec_y)
                    vector_array = np.stack([vec_y, vec_x]).astype(np.float32)

                seed_columns = np.arange(0, max(1, w - 1), density, dtype=int)
                total_columns = max(1, len(seed_columns))
                lines: list[np.ndarray] = []
                neg_vector_array = vector_array * -1.0
                progress_stride = max(1, total_columns // 50)

                ui_call(lambda: setattr(self.fl_progress, "value", 25))
                self._report_progress("Flowlines", 25)

                max_iters = max(10, int(self.fl_iters.value))
                rk_step = max(0.25, float(self.fl_step.value))
                edge_step_scale = 1.0 / rk_step

                for col_index, x_seed in enumerate(seed_columns, start=1):
                    if max_lines > 0 and len(lines) >= max_lines:
                        break

                    trace = self.locked_slice[:, int(x_seed)]
                    if mode == "peak":
                        peaks, _ = signal.find_peaks(trace)
                    elif mode == "trough":
                        peaks, _ = signal.find_peaks(-trace)
                    else:
                        peaks, _ = signal.find_peaks(np.abs(trace))

                    for peak in peaks:
                        if max_lines > 0 and len(lines) >= max_lines:
                            break

                        y0 = int(peak)
                        x0 = int(x_seed)

                        edge_forward_steps = int(np.ceil(max(1, (w - 1) - x0) * edge_step_scale)) + 2
                        edge_backward_steps = int(np.ceil(max(1, x0) * edge_step_scale)) + 2
                        forward_steps = max(max_iters, edge_forward_steps)
                        backward_steps = max(max_iters, edge_backward_steps)

                        px, py = rk4_trace_vector_field(x0, y0, rk_step, forward_steps, vector_array)
                        px_b, py_b = rk4_trace_vector_field(x0, y0, rk_step, backward_steps, neg_vector_array)

                        mask_f = (px >= 0) & (px < w) & (py >= 0) & (py < h)
                        mask_b = (px_b >= 0) & (px_b < w) & (py_b >= 0) & (py_b < h)
                        px, py = px[mask_f], py[mask_f]
                        px_b, py_b = px_b[mask_b], py_b[mask_b]

                        if len(px) == 0 or len(px_b) == 0:
                            continue

                        merged = np.concatenate(
                            [
                                np.column_stack([px_b[::-1], py_b[::-1]]),
                                np.column_stack([px, py]),
                            ],
                            axis=0,
                        )
                        if merged.shape[0] > 5:
                            merged = merged.astype(np.float32)
                            merged[:, 1] = (h - 1) - merged[:, 1]
                            lines.append(merged)

                    if col_index % progress_stride == 0 or col_index == total_columns:
                        progress_value = 25 + int(75 * col_index / total_columns)
                        ui_call(lambda val=progress_value: setattr(self.fl_progress, "value", val))
                        self._report_progress("Flowlines", progress_value)

                def on_done():
                    self.flowlines = lines
                    self.flowline_ids = np.arange(1, len(lines) + 1, dtype=int)
                    self.cluster_labels = None
                    self.seismic_cluster_map = None
                    self.unconformity_surfaces = []
                    if max_lines > 0 and len(lines) >= max_lines:
                        self.fl_count_label.object = f"Flowlines Extracted: {len(lines)} (capped)"
                    else:
                        self.fl_count_label.object = f"Flowlines Extracted: {len(lines)}"
                    self.fl_progress.value = 100
                    self.fl_apply_btn.disabled = len(lines) == 0
                    self.cl_calc_btn.disabled = len(lines) == 0
                    self.sc_calc_btn.disabled = len(lines) == 0
                    self.uc_calc_btn.disabled = len(lines) == 0
                    self.uc_apply_btn.disabled = True
                    self._update_flowline_selector_limits()
                    self.update_view()
                    self._save_state_to_cache()
                    self.logger.info(f"[Flowlines] Completed | n_lines={len(lines)}")

                ui_call(on_done)
            except Exception as exc:
                self.logger.exception(f"[Flowlines] Error: {exc}")
                ui_call(lambda: setattr(self.fl_count_label, "object", "Flowline extraction failed."))
            finally:
                ui_call(lambda: setattr(self.fl_progress, "visible", False))
                ui_call(lambda: setattr(self.fl_calc_btn, "disabled", False))

        threading.Thread(target=worker, daemon=True).start()

    def run_clustering_engine(self, event) -> None:
        if not self.flowlines:
            return

        self.cl_calc_btn.disabled = True
        self.cl_progress.visible = True
        self.cl_progress.value = 10
        self._report_progress("Clustering", 10)
        self.logger.info(f"[Clustering] Starting | device={self.device.type}")
        try:
            selected_indices = self._get_selected_flowline_indices()
            if selected_indices.size == 0:
                self.cluster_labels = None
                self.logger.info("[Clustering] No selected flowlines")
                return

            selected_lines = [self.flowlines[int(idx)] for idx in selected_indices]
            features = [self._line_shape_feature(line) for line in selected_lines]
            self.cl_progress.value = 30
            self._report_progress("Clustering", 30)

            n_clusters = max(2, min(int(self.cl_k.value), len(features)))
            feature_arr = np.asarray(features, dtype=np.float32)

            if self.use_gpu:
                def cl_progress(value: int) -> None:
                    mapped = max(int(self.cl_progress.value), min(100, 30 + int(value * 0.7)))
                    self.cl_progress.value = mapped
                    self._report_progress("Clustering", mapped)

                selected_labels = kmeans_torch(
                    feature_arr,
                    n_clusters=n_clusters,
                    max_iter=25,
                    progress_callback=cl_progress,
                )
            else:
                model = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
                selected_labels = model.fit_predict(feature_arr)
                self.cl_progress.value = 100
                self._report_progress("Clustering", 100)

            full_labels = np.full((len(self.flowlines),), -1, dtype=int)
            full_labels[selected_indices.astype(int)] = selected_labels
            self.cluster_labels = full_labels

            self.update_view()
            self._save_state_to_cache()
            self.logger.info(f"[Clustering] Completed | clusters={n_clusters}")
        except Exception as exc:
            self.logger.exception(f"[Clustering] Error: {exc}")
            self.cluster_labels = None
        finally:
            self.cl_calc_btn.disabled = False
            self.cl_progress.visible = False

    def run_unconf_engine(self, event) -> None:
        if not self.flowlines or self.locked_slice is None:
            return

        self.uc_calc_btn.disabled = True
        self.uc_progress.visible = True
        self.uc_progress.value = 15
        self._report_progress("Unconformities", 15)
        self.logger.info(f"[Unconformities] Starting | device={self.device.type}")
        try:
            h, w = self.locked_slice.shape
            n_surfaces = max(1, int(self.uc_count.value))
            self.uc_progress.value = 40
            self._report_progress("Unconformities", 40)

            candidate_lines: list[np.ndarray] = []
            for line in self._get_selected_flowlines():
                if line.shape[0] < 12:
                    continue
                x_span = float(np.max(line[:, 0]) - np.min(line[:, 0]))
                if x_span < 0.75 * max(1, (w - 1)):
                    continue
                candidate_lines.append(line)

            if candidate_lines:
                heatmap = np.zeros((h, w), dtype=np.float32)
                for line in candidate_lines:
                    x_idx = np.clip(np.round(line[:, 0]).astype(int), 0, w - 1)
                    y_idx = np.clip(np.round(line[:, 1]).astype(int), 0, h - 1)
                    np.add.at(heatmap, (y_idx, x_idx), 1.0)

                scored_lines: list[tuple[float, np.ndarray]] = []
                for line in candidate_lines:
                    x_idx = np.clip(np.round(line[:, 0]).astype(int), 0, w - 1)
                    y_idx = np.clip(np.round(line[:, 1]).astype(int), 0, h - 1)
                    score = float(np.mean(heatmap[y_idx, x_idx]))
                    scored_lines.append((score, line.astype(np.float32)))

                scored_lines.sort(key=lambda item: item[0], reverse=True)
                selected_ranked = [line for _, line in scored_lines[: min(n_surfaces, len(scored_lines))]]
                selected = [self._extend_line_to_full_width(line, w, h) for line in selected_ranked]
            else:
                selected = []

            self.uc_progress.value = 80
            self._report_progress("Unconformities", 80)

            self.unconformity_surfaces = selected
            self.uc_apply_btn.disabled = len(self.unconformity_surfaces) == 0
            self.uc_progress.value = 100
            self._report_progress("Unconformities", 100)
            self.update_view()
            self._save_state_to_cache()
            self.logger.info(f"[Unconformities] Completed | n_surfaces={len(self.unconformity_surfaces)}")
        except Exception as exc:
            self.logger.exception(f"[Unconformities] Error: {exc}")
        finally:
            self.uc_progress.visible = False
            self.uc_calc_btn.disabled = False

    def run_seismic_clustering_engine(self, event) -> None:
        if not self.flowlines or self.locked_slice is None:
            return

        self.sc_calc_btn.disabled = True
        self.sc_progress.visible = True
        self.sc_progress.value = 10
        self._report_progress("SeisClustering", 10)
        self.logger.info(f"[SeisClustering] Starting | device={self.device.type}")
        try:
            selected_indices = self._get_selected_flowline_indices()
            if selected_indices.size == 0:
                self.seismic_cluster_map = None
                self.update_view()
                self.logger.info("[SeisClustering] No selected flowlines")
                return

            selected_lines = [self.flowlines[int(idx)] for idx in selected_indices]
            features = [self._line_shape_feature(line) for line in selected_lines]
            self.sc_progress.value = 35
            self._report_progress("SeisClustering", 35)

            n_clusters = max(2, min(int(self.sc_k.value), len(features)))
            feature_arr = np.asarray(features, dtype=np.float32)

            if self.use_gpu:
                def sc_progress(value: int) -> None:
                    mapped = max(int(self.sc_progress.value), min(85, 35 + int(value * 0.5)))
                    self.sc_progress.value = mapped
                    self._report_progress("SeisClustering", mapped)

                selected_labels = kmeans_torch(
                    feature_arr,
                    n_clusters=n_clusters,
                    max_iter=25,
                    progress_callback=sc_progress,
                )
            else:
                model = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
                selected_labels = model.fit_predict(feature_arr)

            h, w = self.locked_slice.shape
            sparse_map = np.full((h, w), -1, dtype=np.int32)
            for line, label in zip(selected_lines, selected_labels):
                x_idx = np.clip(np.round(line[:, 0]).astype(int), 0, w - 1)
                y_idx = np.clip(np.round(line[:, 1]).astype(int), 0, h - 1)
                y_idx = np.clip((h - 1) - y_idx, 0, h - 1)
                sparse_map[y_idx, x_idx] = int(label)

            known_mask = sparse_map >= 0
            if not np.any(known_mask):
                self.seismic_cluster_map = None
                self.logger.info("[SeisClustering] No labeled pixels after rasterization")
            else:
                _, nearest_indices = ndimage.distance_transform_edt(~known_mask, return_indices=True)
                filled_map = sparse_map[tuple(nearest_indices)]
                self.seismic_cluster_map = filled_map.astype(np.int32)

            self.sc_progress.value = 100
            self._report_progress("SeisClustering", 100)
            self.update_view()
            self._save_state_to_cache()
            self.logger.info(f"[SeisClustering] Completed | clusters={n_clusters}")
        except Exception as exc:
            self.logger.exception(f"[SeisClustering] Error: {exc}")
            self.seismic_cluster_map = None
        finally:
            self.sc_progress.visible = False
            self.sc_calc_btn.disabled = False

    def _cluster_color(self, idx: int, total: int) -> str:
        if total <= 1:
            return "#ff8800"
        ratio = idx / max(1, total - 1)
        cmap_name = self.cl_cmap.value
        try:
            rgba = cm.get_cmap(cmap_name)(ratio)
            return mcolors.to_hex(rgba)
        except Exception:
            return "#ff8800"

    def apply_flowline_style(self, event=None) -> None:
        self.applied_fl_color = self.fl_color.value
        self.applied_fl_width = int(self.fl_width.value)
        self.update_view()
        self._save_state_to_cache()

    def apply_unconf_style(self, event=None) -> None:
        self.applied_uc_color = self.uc_color.value
        self.applied_uc_width = int(self.uc_width.value)
        self.update_view()
        self._save_state_to_cache()

    def update_view(self, event=None) -> None:
        if self.locked_slice is None:
            return

        h, w = self.locked_slice.shape
        bounds = (0, 0, w, h)
        x_dim = str(self.locked_meta.get("xl", "X"))
        y_dim = str(self.locked_meta.get("yl", "Y"))

        plot_bg = get_plot_surface_background(is_dark_mode)
        colorbar_opts = get_dark_colorbar_opts(is_dark_mode)

        hv_opts = dict(
            invert_yaxis=True,
            responsive=True,
            xlim=(0, w),
            ylim=(0, h),
            bgcolor=plot_bg,
        )

        img_left = hv.Image(self.locked_slice, bounds=bounds, kdims=[x_dim, y_dim], label="Seismic").opts(
            cmap=self.seismic_cmap.value,
            clim=(-self.amp_limit, self.amp_limit),
            colorbar=True,
            colorbar_opts=colorbar_opts,
            title=f"{self.locked_meta.get('mode')} {self.locked_meta.get('idx')} (Seismic)",
            toolbar="above",
            hooks=[self._apply_dark_plot_theme, self._capture_left_plot],
            **hv_opts,
        )
        self.tap_stream.source = img_left
        self.pane_left.object = (img_left * self.marker_dmap).opts(responsive=True)

        gst_ready = self.res_mag is not None and self.res_ori is not None
        right_title = f"{self.locked_meta.get('mode')} {self.locked_meta.get('idx')} (Analysis)"
        show_attr_overlay = gst_ready and self.show_right_attr.value
        show_seis_cluster = self.seismic_cluster_map is not None and self.show_right_seis_cluster.value
        show_seismic_colorbar = False
        show_attr_colorbar = False
        show_flowline_colorbar = False
        show_cluster_flowline_colorbar = False

        if show_seis_cluster:
            pass
        elif self.cluster_labels is not None and self.show_right_cluster_flowlines.value:
            show_cluster_flowline_colorbar = True
        elif self.fl_color_mode.value == "Colorbar" and self.show_right_flowlines.value:
            show_flowline_colorbar = True
        elif show_attr_overlay:
            show_attr_colorbar = True
        elif self.show_right_base.value:
            show_seismic_colorbar = True

        if self.show_right_base.value:
            img_right_base = hv.Image(self.locked_slice, bounds=bounds, kdims=[x_dim, y_dim], label="Seismic").opts(
                cmap="gray" if show_attr_overlay else self.seismic_cmap.value,
                clim=(-self.amp_limit, self.amp_limit),
                colorbar=show_seismic_colorbar,
                colorbar_opts=colorbar_opts,
                toolbar=None,
                title=right_title,
                hooks=[self._apply_dark_plot_theme, self._capture_right_plot],
                **hv_opts,
            )
        else:
            img_right_base = hv.Image(np.zeros_like(self.locked_slice), bounds=bounds, kdims=[x_dim, y_dim], label="Seismic").opts(
                alpha=0.0,
                colorbar=False,
                toolbar=None,
                title=right_title,
                hooks=[self._apply_dark_plot_theme, self._capture_right_plot],
                **hv_opts,
            )

        attr_layer = hv.Image(np.zeros_like(self.locked_slice), bounds=bounds, kdims=[x_dim, y_dim]).opts(
            alpha=0.0,
            colorbar=False,
            colorbar_opts=colorbar_opts,
            toolbar=None,
            **hv_opts,
        )

        seismic_cluster_layer = hv.Image(np.zeros_like(self.locked_slice), bounds=bounds, kdims=[x_dim, y_dim]).opts(
            alpha=0.0,
            colorbar=False,
            colorbar_opts=colorbar_opts,
            toolbar=None,
            **hv_opts,
        )

        if show_seis_cluster:
            max_label = int(np.max(self.seismic_cluster_map)) if self.seismic_cluster_map.size > 0 else 0
            seismic_cluster_layer = hv.Image(self.seismic_cluster_map.astype(np.float32), bounds=bounds, kdims=[x_dim, y_dim]).opts(
                cmap=self.sc_cmap.value,
                alpha=float(self.sc_alpha.value),
                clim=(0.0, float(max(1, max_label))),
                colorbar=True,
                colorbar_opts=colorbar_opts,
                toolbar=None,
                **hv_opts,
            )

        if show_attr_overlay:
            attr_data = self.res_mag if self.gst_view_select.value == "Magnitude" else self.res_ori
            attr_layer = hv.Image(attr_data, bounds=bounds, kdims=[x_dim, y_dim]).opts(
                cmap=self.gst_cmap.value,
                alpha=float(self.gst_opacity.value),
                clim=(float(np.nanmin(attr_data)), float(np.nanmax(attr_data))),
                colorbar=show_attr_colorbar,
                colorbar_opts=colorbar_opts,
                toolbar=None,
                **hv_opts,
            )

        line_overlay = hv.Path([])
        clustered_line_overlay = hv.Path([])
        flowline_colorbar_layer = None
        cluster_flowline_colorbar_layer = None

        selected_indices = self._get_selected_flowline_indices()
        selected_lines = [self.flowlines[int(idx)] for idx in selected_indices] if selected_indices.size > 0 else []

        if selected_lines and self.show_right_flowlines.value:
            if self.fl_color_mode.value == "Colorbar":
                flowline_means = np.asarray([float(np.mean(line[:, 1])) for line in selected_lines], dtype=np.float32)
                min_val = float(np.min(flowline_means))
                max_val = float(np.max(flowline_means))
                if max_val <= min_val:
                    max_val = min_val + 1.0

                seg_x0, seg_y0, seg_x1, seg_y1, seg_v = [], [], [], [], []
                for line, mean_val in zip(selected_lines, flowline_means):
                    if line.shape[0] < 2:
                        continue
                    seg_x0.append(line[:-1, 0])
                    seg_y0.append(line[:-1, 1])
                    seg_x1.append(line[1:, 0])
                    seg_y1.append(line[1:, 1])
                    seg_v.append(np.full(line.shape[0] - 1, float(mean_val), dtype=np.float32))

                if seg_x0:
                    x0_arr = np.concatenate(seg_x0)
                    y0_arr = np.concatenate(seg_y0)
                    x1_arr = np.concatenate(seg_x1)
                    y1_arr = np.concatenate(seg_y1)
                    v_arr = np.concatenate(seg_v)
                    flowline_colorbar_layer = hv.Segments(
                        (x0_arr, y0_arr, x1_arr, y1_arr, v_arr),
                        kdims=["x0", "y0", "x1", "y1"],
                        vdims=["value"],
                    ).opts(
                        color="value",
                        cmap=self.fl_cmap.value,
                        clim=(min_val, max_val),
                        colorbar=show_flowline_colorbar,
                        colorbar_opts=colorbar_opts,
                        line_width=self.applied_fl_width,
                        **hv_opts,
                    )
                    line_overlay = flowline_colorbar_layer
            else:
                line_overlay = hv.Path(selected_lines).opts(color=self.applied_fl_color, line_width=self.applied_fl_width)

        if selected_lines and self.cluster_labels is not None and self.show_right_cluster_flowlines.value:
            valid_pairs = [
                (self.flowlines[int(idx)], int(self.cluster_labels[int(idx)]))
                for idx in selected_indices
                if int(self.cluster_labels[int(idx)]) >= 0
            ]
            if valid_pairs:
                cluster_labels = [pair[1] for pair in valid_pairs]
                n_clusters = int(max(cluster_labels)) + 1
                seg_x0, seg_y0, seg_x1, seg_y1, seg_v = [], [], [], [], []
                for line, label in valid_pairs:
                    if line.shape[0] < 2:
                        continue
                    seg_x0.append(line[:-1, 0])
                    seg_y0.append(line[:-1, 1])
                    seg_x1.append(line[1:, 0])
                    seg_y1.append(line[1:, 1])
                    seg_v.append(np.full(line.shape[0] - 1, float(label), dtype=np.float32))

                if seg_x0:
                    x0_arr = np.concatenate(seg_x0)
                    y0_arr = np.concatenate(seg_y0)
                    x1_arr = np.concatenate(seg_x1)
                    y1_arr = np.concatenate(seg_y1)
                    v_arr = np.concatenate(seg_v)
                    cluster_flowline_colorbar_layer = hv.Segments(
                        (x0_arr, y0_arr, x1_arr, y1_arr, v_arr),
                        kdims=["x0", "y0", "x1", "y1"],
                        vdims=["cluster"],
                    ).opts(
                        color="cluster",
                        cmap=self.cl_cmap.value,
                        clim=(0.0, float(max(1, n_clusters - 1))),
                        colorbar=show_cluster_flowline_colorbar,
                        colorbar_opts=colorbar_opts,
                        line_width=int(self.cl_width.value),
                        **hv_opts,
                    )
                    clustered_line_overlay = cluster_flowline_colorbar_layer

        unconf_overlay = hv.Path([])
        if self.unconformity_surfaces and self.show_right_unconf.value:
            unconf_overlay = hv.Path(self.unconformity_surfaces).opts(
                color=self.applied_uc_color,
                line_width=self.applied_uc_width,
            )

        self.pane_right.object = (
            img_right_base
            * seismic_cluster_layer
            * attr_layer
            * line_overlay
            * clustered_line_overlay
            * unconf_overlay
            * self.marker_dmap
        ).opts(responsive=True)

    def run_export(self, event) -> None:
        self.exp_btn.disabled = True
        self.exp_progress.visible = True
        self.exp_progress.value = 10
        self._report_progress("Export", 10)
        self.exp_status.object = "Exporting to Petrel..."
        self.logger.info("[Export] Started")

        try:
            if PetrelConnection is None:
                self.exp_status.object = "Petrel export unavailable in this environment"
                self.logger.warning("[Export] Petrel unavailable")
                return

            ptp = PetrelConnection(allow_experimental=True)
            suffix = self.exp_name.value
            folder = ptp.create_folder(f"FlowLine Results {suffix}")
            self.exp_progress.value = 40
            self._report_progress("Export", 40)

            if self.exp_fl.value and self.flowlines:
                pset = ptp.create_polylineset(f"Flowlines{suffix}", folder=folder)
                pset.readonly = False
                dataframes = []
                for line_index, line in enumerate(self.flowlines):
                    subsampled = line[::5]
                    if subsampled.shape[0] == 0:
                        continue
                    frame = pd.DataFrame(subsampled, columns=["Inline", "Time"])
                    frame["X"] = subsampled[:, 0]
                    frame["Z"] = subsampled[:, 1]
                    frame["Y"] = 0
                    frame["Poly"] = line_index
                    dataframes.append(frame)

                if dataframes:
                    pset.set_values(pd.concat(dataframes, ignore_index=True))
            self.exp_progress.value = 90
            self._report_progress("Export", 90)

            self.exp_status.object = "Success: Data exported to results folder."
            pn.state.notifications.success("Export Complete")
            self.exp_progress.value = 100
            self._report_progress("Export", 100)
            self.logger.info("[Export] Completed")
        except Exception as exc:
            self.logger.exception(f"[Export] Error: {exc}")
            self.exp_status.object = f"Error: {exc}"
        finally:
            self.exp_btn.disabled = False
            self.exp_progress.visible = False

    def get_template(self):
        plot_title_color = get_content_text_color(is_dark_mode)
        plot_card_background = get_plot_surface_background(is_dark_mode)

        style = {
            "background": plot_card_background,
            "color": plot_title_color,
            "padding": "10px",
            "border-radius": "8px",
            "box-shadow": "0 2px 5px rgba(0,0,0,0.1)",
            "flex": "1 1 0",
            "width": "100%",
            "height": "100%",
            "min-height": "0",
            "overflow": "hidden",
            "display": "flex",
            "flex-direction": "column",
        }

        main = pn.FlexBox(
            pn.Column(
                pn.pane.HTML(f"<h3 style='margin:0; color:{plot_title_color};'>Seismic Preview</h3>"),
                self.left_plot_pane,
                styles=style,
                sizing_mode="stretch_both",
            ),
            pn.Column(
                pn.pane.HTML(f"<h3 style='margin:0; color:{plot_title_color};'>Flowline Analysis</h3>"),
                self.right_plot_pane,
                styles=style,
                sizing_mode="stretch_both",
            ),
            justify_content="space-between",
            align_items="stretch",
            gap="10px",
            flex_wrap="nowrap",
            sizing_mode="stretch_both",
        )

        main.styles = {
            "background": get_main_outer_background(is_dark_mode),
            "padding": "6px",
            "height": "calc(100vh - 96px)",
            "width": "100%",
            "overflow": "hidden",
        }

        return pn.template.FastListTemplate(
            title=APP_TITLE,
            logo=valid_logo,
            favicon=valid_favicon,
            accent_base_color=BLUE_OMV_COLOR,
            header_background=DARK_BLUE_OMV_COLOR,
            main_layout=None,
            main_max_width="",
            sidebar=[
                self.control_card,
                pn.layout.Divider(),
                self.gst_card,
                pn.layout.Divider(),
                self.flow_card,
                pn.layout.Divider(),
                self.cluster_card,
                pn.layout.Divider(),
                self.seismic_cluster_card,
                pn.layout.Divider(),
                self.unconf_card,
                pn.layout.Divider(),
                self.export_card,
            ],
            header=[
                pn.Row(
                    pn.Spacer(sizing_mode="stretch_width"),
                    pn.pane.HTML(docs_button_html(DOCUMENTATION_URL)),
                    sizing_mode="stretch_width",
                    margin=0,
                )
            ],
            main=[main],
        )


app = SeismicFlowApp()
template = app.get_template()
template.servable()

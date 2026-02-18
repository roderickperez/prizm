import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import panel as pn
import holoviews as hv
from holoviews import streams
from bokeh.models import ColorBar

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Try to import psutil for RAM monitoring
try:
    import psutil
except ImportError:
    psutil = None

from seismicAttributes.core.attribute_engine import compute_attribute
from seismicAttributes.core.logging_utils import setup_app_logger
from seismicAttributes.core.runtime_utils import apply_dll_fix
from seismicAttributes.core.segy_loader import SegyDataStore

apply_dll_fix()

# --- Prizm Imports ---
#from cegalprizm.pythontool import PetrelConnection
PetrelConnection = None

from shared.ui.omv_theme import (
    BLUE_OMV_COLOR,
    DARK_BLUE_OMV_COLOR,
    DARK_GREEN_OMV_COLOR,
    MAGENTA_OMV_COLOR,
    NEON_MAGENTA_OMV_COLOR,
    NEON_OMV_COLOR,
    docs_button_html,
    get_content_text_color,
    get_dark_colorbar_opts,
    get_dark_select_stylesheets,
    get_extension_raw_css,
    get_main_outer_background,
    get_neon_button_stylesheets,
    get_plot_surface_background,
    get_slider_stylesheets,
    is_dark_mode_from_state,
)

# --- App UI Constants ---
APP_TITLE = "Seismic Attributes +"
DOCUMENTATION_URL = "https://example.com/docs"
SEGY_FILE_PATH = ROOT_DIR / "testData" / "1_Original_Seismics.sgy"
APP_DIR = Path(__file__).resolve().parent
LOG_DIR = APP_DIR / "logs"

# --- Initialize Panel & HoloViews ---
is_dark_mode = is_dark_mode_from_state()
pn.extension('tabulator', raw_css=get_extension_raw_css(is_dark_mode))
hv.extension('bokeh')

# --- Image Paths ---
ASSETS_DIR = ROOT_DIR / "images" / "OMV_brandLogoPack" / "OMV_brandLogoPack"
LOGO_PATH = ASSETS_DIR / "OMV_logo_Neon_Small.png"
FAVICON_PATH = ASSETS_DIR / "OMV_logo_Blue_Small.png"

valid_logo = str(LOGO_PATH) if LOGO_PATH.exists() else None
valid_favicon = str(FAVICON_PATH) if FAVICON_PATH.exists() else None


APP_LOGGER = setup_app_logger(LOG_DIR)


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


def get_text_input_stylesheets() -> list[str]:
    if not is_dark_mode:
        return []
    return [
        f"""
        :host {{
            --input-background: {DARK_GREEN_OMV_COLOR};
            --input-color: white;
            --input-border-color: {DARK_GREEN_OMV_COLOR};
        }}

        input,
        textarea,
        .bk-input,
        .bk-input-group input,
        .bk-input-group .bk-input {{
            background: {DARK_GREEN_OMV_COLOR} !important;
            background-color: {DARK_GREEN_OMV_COLOR} !important;
            color: white !important;
            border-color: {DARK_GREEN_OMV_COLOR} !important;
        }}
        """
    ]

# ==============================================================================
#  SEISMIC APP CLASS
# ==============================================================================
class SeismicApp:
    def __init__(self):
        self.logger = APP_LOGGER
        self.logger.info("Initializing SeismicApp")

        # 1. Load Session & Metadata
        self.session_data = self.load_session_data()
        self.project_name = self.session_data.get("project", "Unknown")
        self.cube_name = self.session_data.get("selected_cube_name", SEGY_FILE_PATH.stem)
        self.cube_guid = self.session_data.get("selected_cube_guid", None)

        self.data_store = SegyDataStore(SEGY_FILE_PATH, logger=self.logger, cache_size=12)
        self.has_seismic_data = False
        self.inline_values = np.array([], dtype=int)
        self.xline_values = np.array([], dtype=int)
        self.timeslice_values = np.array([], dtype=int)
        self.inline_index_map = {}
        self.xline_index_map = {}
        self.timeslice_index_map = {}

        self.load_segy_volume()
        
        self.dims = (0, 0, 0) 
        self.amp_limit = 1000.0 
        self.fetch_metadata_and_stats() 

        self.select_stylesheets = get_dark_select_stylesheets(is_dark_mode)
        self.slider_stylesheets = get_slider_stylesheets()
        self.action_button_stylesheets = get_neon_button_stylesheets()
        self.radio_group_stylesheets = get_radio_group_stylesheets()
        self.text_input_stylesheets = get_text_input_stylesheets()
        self.marker_opts = dict(color='red', size=8, line_color='white', line_width=1)
        self._left_plot_figure = None
        self._right_plot_figure = None

        section_header_background = NEON_OMV_COLOR
        section_header_text = DARK_BLUE_OMV_COLOR
        section_body_background = DARK_BLUE_OMV_COLOR if is_dark_mode else "white"
        section_body_text = "white" if is_dark_mode else "inherit"

        # 2. Widgets
        
        # -- Slice Controls --
        self.radio_group = pn.widgets.RadioButtonGroup(
            name='Slice Type', options=['Inline', 'Crossline', 'Timeslice'],
            button_type='default',
            value='Inline',
            sizing_mode='stretch_width',
            stylesheets=self.radio_group_stylesheets,
        )

        i_max, j_max, k_max = self.dims
        self.slice_slider = pn.widgets.IntSlider(
            name='Slice Index',
            start=0,
            end=max(0, i_max - 1),
            value=i_max//2,
            sizing_mode='stretch_width',
            stylesheets=self.slider_stylesheets,
        )

        self.seismic_cmap = pn.widgets.Select(
            name='Seismic Colormap',
            options=['RdBu', 'gray', 'bwr', 'PuOr', 'viridis'],
            value='gray',
            sizing_mode='stretch_width',
            stylesheets=self.select_stylesheets,
        )
        
        self.slice_update_btn = pn.widgets.Button(
            name='Update Preview',
            button_type='default',
            sizing_mode='stretch_width',
            stylesheets=self.action_button_stylesheets,
        )

        self.control_card = pn.Card(
            pn.Column(
                self.radio_group,
                self.slice_slider,
                self.seismic_cmap,
                self.slice_update_btn,
                sizing_mode='stretch_width'
            ),
            title="Slice Controls",
            collapsed=True,
            header_background=section_header_background,
            active_header_background=section_header_background,
            header_color=section_header_text,
            styles={'background': section_body_background, 'color': section_body_text},
            sizing_mode='stretch_width'
        )

        # -- Attribute Controls --
        self.attr_selector = pn.widgets.Select(
            name='Attribute', 
            options=[
                'Envelope', 'Instantaneous Phase', 'Instantaneous Frequency', 
                'Similarity (GST)', 'RMS Amplitude', 'Energy', 'Cosine of Phase',
                'Gradient Structure Tensor (GPU)'
            ],
            value='Envelope',
            sizing_mode='stretch_width',
            stylesheets=self.select_stylesheets,
        )

        self.attr_cmap_selector = pn.widgets.Select(
            name='Attribute Colormap', 
            options=['Viridis', 'Inferno', 'Magma', 'Plasma', 'Jet', 'Rainbow', 'Fire', 'Gray'],
            value='Viridis',
            sizing_mode='stretch_width',
            stylesheets=self.select_stylesheets,
        )
        
        self.opacity_slider = pn.widgets.FloatSlider(
            name='Attribute Opacity',
            start=0.0,
            end=1.0,
            step=0.05,
            value=0.6,
            sizing_mode='stretch_width',
            stylesheets=self.slider_stylesheets,
        )

        # Param Inputs
        self.dt_input = pn.widgets.FloatInput(name="Sample Interval (dt) [s]", value=0.004, step=0.001, sizing_mode='stretch_width')
        self.duration_input = pn.widgets.FloatInput(name="Window Duration [s]", value=0.036, step=0.004, sizing_mode='stretch_width')
        self.window_size_input = pn.widgets.IntInput(name="RMS Window Size (samples)", value=9, step=2, start=1, sizing_mode='stretch_width')
        
        self.sigma_input = pn.widgets.FloatInput(name="GST Sigma (Inner Scale)", value=1.0, step=0.1, start=0.1, sizing_mode='stretch_width')
        self.rho_input = pn.widgets.FloatInput(name="GST Rho (Outer Scale)", value=1.0, step=0.1, start=0.1, sizing_mode='stretch_width')

        for widget in [
            self.dt_input,
            self.duration_input,
            self.window_size_input,
            self.sigma_input,
            self.rho_input,
        ]:
            widget.stylesheets = self.text_input_stylesheets
        
        self.param_container = pn.Column(
            self.dt_input, self.duration_input, self.window_size_input, 
            self.sigma_input, self.rho_input,
            sizing_mode='stretch_width'
        )
        self.update_param_visibility() 

        self.attr_selector.param.watch(self.toggle_attribute_params, 'value')
        
        self.attr_update_btn = pn.widgets.Button(
            name='Attribute Preview',
            button_type='default',
            sizing_mode='stretch_width',
            stylesheets=self.action_button_stylesheets,
        )
        
        # [NEW] Preview Progress Bar
        self.preview_progress = pn.widgets.Progress(name='Preview Progress', value=0, min_width=200, sizing_mode='stretch_width', visible=False)

        self.attribute_card = pn.Card(
            pn.Column(
                self.attr_selector, 
                self.param_container, 
                self.attr_cmap_selector, 
                self.opacity_slider,
                self.attr_update_btn,
                self.preview_progress, # Added here
                sizing_mode='stretch_width'
            ),
            title="Attribute Settings",
            collapsed=True,
            header_background=section_header_background,
            active_header_background=section_header_background,
            header_color=section_header_text,
            styles={'background': section_body_background, 'color': section_body_text},
            sizing_mode='stretch_width'
        )

        self.export_name_input = pn.widgets.TextInput(
            name='Output Name',
            value=f"{self.cube_name}_Attr",
            sizing_mode='stretch_width',
            stylesheets=self.text_input_stylesheets,
        )
        self.export_btn = pn.widgets.Button(
            name='Export to Petrel',
            button_type='default',
            sizing_mode='stretch_width',
            stylesheets=self.action_button_stylesheets,
        )
        self.export_spinner = pn.indicators.LoadingSpinner(value=False, width=25, height=25, align='center')
        
        self.export_progress = pn.widgets.Progress(name='Export Progress', value=0, min_width=200, sizing_mode='stretch_width', visible=False)
        self.export_status = pn.pane.Markdown(
            "Ready",
            styles={'font-size': '0.9em', 'color': 'white' if is_dark_mode else '#666'},
        )

        self.export_card = pn.Card(
            pn.Column(
                self.export_name_input, 
                self.export_btn, 
                self.export_progress,
                pn.Row(self.export_spinner, self.export_status), 
                sizing_mode='stretch_width'
            ),
            title="Export",
            collapsed=True,
            header_background=section_header_background,
            active_header_background=section_header_background,
            header_color=section_header_text,
            styles={'background': section_body_background, 'color': section_body_text},
            sizing_mode='stretch_width'
        )

        # --- LAYOUT SETUP ---
        self.tap_stream = streams.Tap(x=None, y=None)
        self.marker_dmap = hv.DynamicMap(self.get_marker, streams=[self.tap_stream])

        pane_opts = dict(sizing_mode='stretch_both', min_height=0)
        
        self.pane_left_1 = pn.pane.HoloViews(object=None, **pane_opts)
        self.pane_right_base = pn.pane.HoloViews(object=None, **pane_opts)
        self.pane_right_overlay = pn.pane.HoloViews(object=None, **pane_opts)

        pane_col_styles = {
            'height': '100%',
            'min-height': '0',
            'overflow': 'hidden',
            'display': 'flex',
            'flex-direction': 'column',
            'flex': '1 1 auto',
        }
        self.col_left_1 = pn.Column(self.pane_left_1, sizing_mode='stretch_both', styles=pane_col_styles)
        self.col_right_overlay = pn.Column(self.pane_right_overlay, sizing_mode='stretch_both', styles=pane_col_styles)

        plot_pane_styles = {
            'height': '100%',
            'min-height': '0',
            'overflow': 'hidden',
            'display': 'flex',
            'flex-direction': 'column',
            'flex': '1 1 auto',
        }
        self.left_plot_pane = pn.Column(self.col_left_1, sizing_mode='stretch_both', styles=plot_pane_styles)
        self.right_plot_pane = pn.Column(self.col_right_overlay, sizing_mode='stretch_both', styles=plot_pane_styles)

        # Interactions
        self.radio_group.param.watch(self.update_slider_limits, 'value')
        self.radio_group.param.watch(lambda e: self.logger.info(f"Slice type changed: {e.new}"), 'value')
        self.slice_slider.param.watch(lambda e: self.logger.info(f"Slice index changed: {e.new}"), 'value_throttled')
        self.seismic_cmap.param.watch(lambda e: self.logger.info(f"Seismic cmap changed: {e.new}"), 'value')
        self.attr_selector.param.watch(lambda e: self.logger.info(f"Attribute changed: {e.new}"), 'value')
        self.attr_cmap_selector.param.watch(lambda e: self.logger.info(f"Attribute cmap changed: {e.new}"), 'value')
        self.opacity_slider.param.watch(lambda e: self.logger.info(f"Opacity changed: {e.new}"), 'value_throttled')
        
        self.slice_update_btn.on_click(self.run_update)
        self.attr_update_btn.on_click(self.run_update)
        
        self.export_btn.on_click(self.run_export)
        self.update_slider_limits(None)
        self.run_update(None)

    def get_marker(self, x, y):
        if x is None or y is None: return hv.Points([]).opts(**self.marker_opts)
        return hv.Points([(x, y)]).opts(**self.marker_opts)

    def load_session_data(self):
        data_file = os.environ.get("PWR_DATA_FILE")
        if data_file and os.path.exists(data_file):
            try:
                with open(data_file, "r") as f: return json.load(f)
            except Exception: pass
        return {}

    def toggle_attribute_params(self, event):
        self.update_param_visibility()

    def update_param_visibility(self):
        attr = self.attr_selector.value
        self.dt_input.visible = attr in ['Instantaneous Frequency', 'Similarity (GST)', 'Energy']
        self.duration_input.visible = attr in ['Similarity (GST)', 'Energy']
        self.window_size_input.visible = attr == 'RMS Amplitude'
        is_gst = attr == 'Gradient Structure Tensor (GPU)'
        self.sigma_input.visible = is_gst
        self.rho_input.visible = is_gst

    def load_segy_volume(self):
        loaded = self.data_store.load()
        self.has_seismic_data = loaded
        if not loaded:
            return

        self.inline_values = self.data_store.inline_values
        self.xline_values = self.data_store.xline_values
        self.timeslice_values = self.data_store.timeslice_values
        self.inline_index_map = self.data_store.inline_index_map
        self.xline_index_map = self.data_store.xline_index_map
        self.timeslice_index_map = self.data_store.timeslice_index_map
        self.dims = self.data_store.dims
        self.amp_limit = self.data_store.amp_limit
        self.cube_name = self.data_store.segy_path.stem

    def fetch_metadata_and_stats(self):
        if self.has_seismic_data:
            self.dims = self.data_store.dims
            self.amp_limit = self.data_store.amp_limit
            return

        # Legacy Petrel loading section retained as requested (commented, do not remove):
        # if not self.cube_guid:
        #     return
        # try:
        #     ptp = PetrelConnection(allow_experimental=True)
        #     objs = ptp.get_petrelobjects_by_guids([self.cube_guid])
        #     if objs:
        #         cube = objs[0]
        #         self.dims = cube.extent
        #         i_mid, j_mid, k_mid = self.dims.i // 2, self.dims.j // 2, self.dims.k // 2
        #         chunk_i = cube.chunk((i_mid, i_mid), (0, self.dims.j-1), (0, self.dims.k-1)).as_array()
        #         chunk_j = cube.chunk((0, self.dims.i-1), (j_mid, j_mid), (0, self.dims.k-1)).as_array()
        #         chunk_k = cube.chunk((0, self.dims.i-1), (0, self.dims.j-1), (k_mid, k_mid)).as_array()
        #         sample_data = np.concatenate([chunk_i.flatten(), chunk_j.flatten(), chunk_k.flatten()])
        #         sample_data = sample_data[~np.isnan(sample_data)]
        #         self.amp_limit = np.percentile(np.abs(sample_data), 99) if len(sample_data) > 0 else 1000.0
        # except Exception as e:
        #     print(f"Meta fetch error: {e}")

    def update_slider_limits(self, event):
        mode = self.radio_group.value if event is None else event.new

        if mode == 'Inline':
            values = self.inline_values
        elif mode == 'Crossline':
            values = self.xline_values
        else:
            values = self.timeslice_values

        if values.size == 0:
            self.slice_slider.start = 0
            self.slice_slider.end = 0
            self.slice_slider.value = 0
            return

        start = int(values.min())
        end = int(values.max())
        mid = int(values[len(values) // 2])
        self.slice_slider.start = start
        self.slice_slider.end = end
        self.slice_slider.value = mid

    def _show_empty_state(self, message: str):
        self._left_plot_figure = None
        self._right_plot_figure = None
        empty_plot_bg = get_plot_surface_background(is_dark_mode)
        empty_plot_text = get_content_text_color(is_dark_mode)
        empty_plot = hv.Text(0.5, 0.5, message).opts(
            xaxis=None,
            yaxis=None,
            toolbar=None,
            width=700,
            height=500,
            text_color=empty_plot_text,
            bgcolor=empty_plot_bg,
            fontsize=14,
        )
        self.pane_left_1.object = empty_plot
        self.pane_right_overlay.object = empty_plot

    def _link_plot_ranges(self):
        if self._left_plot_figure is None or self._right_plot_figure is None:
            return
        left_fig = self._left_plot_figure
        right_fig = self._right_plot_figure
        if right_fig.x_range is not left_fig.x_range:
            right_fig.x_range = left_fig.x_range
        if right_fig.y_range is not left_fig.y_range:
            right_fig.y_range = left_fig.y_range

    def _capture_left_plot(self, plot, _element):
        self._left_plot_figure = plot.state
        self._link_plot_ranges()

    def _capture_right_plot(self, plot, _element):
        self._right_plot_figure = plot.state
        self._link_plot_ranges()

    def _apply_dark_plot_theme(self, plot, _element):
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

        side_panels = list(fig.right) + list(fig.left) + list(fig.above) + list(fig.below)
        for panel in side_panels:
            if hasattr(panel, "background_fill_color"):
                panel.background_fill_color = plot_bg
            if hasattr(panel, "background_fill_alpha"):
                panel.background_fill_alpha = 1.0
            if hasattr(panel, "border_line_color"):
                panel.border_line_color = plot_bg

        for panel in fig.right:
            if isinstance(panel, ColorBar):
                panel.background_fill_color = plot_bg
                panel.border_line_color = plot_bg
                panel.major_label_text_color = "white"
                panel.title_text_color = "white"
                panel.major_tick_line_color = "white"
                panel.minor_tick_line_color = "white"
                panel.bar_line_color = "white"
                if hasattr(panel, "color_mapper") and panel.color_mapper is not None:
                    panel.color_mapper.nan_color = plot_bg

    def run_update(self, event):
        if not self.has_seismic_data:
            self._show_empty_state("Seismic volume not loaded. Verify segyio and SEGY file path.")
            self.logger.warning("Update requested but seismic data not loaded")
            return

        mode = self.radio_group.value
        idx = self.slice_slider.value
        seis_cmap = self.seismic_cmap.value
        attr_name = self.attr_selector.value
        attr_cmap = self.attr_cmap_selector.value
        opacity = self.opacity_slider.value
        self.logger.info(
            f"Update preview start | mode={mode} idx={idx} seis_cmap={seis_cmap} attr={attr_name} attr_cmap={attr_cmap} opacity={opacity}"
        )
        
        # Update Text & Visibility
        if hasattr(self, 'slice_update_btn'): self.slice_update_btn.name = "Loading..."
        if hasattr(self, 'attr_update_btn'): self.attr_update_btn.name = "Calculating..."
        
        # [NEW] Reset & Show Preview Progress
        self.preview_progress.visible = True
        self.preview_progress.value = 10
        
        try:
            # 1. Fetch Data
            self.preview_progress.value = 30
            raw_data, display_data, labels = self.fetch_and_prepare_data(mode, idx)
            x_dim, y_dim = labels 
            
            self.logger.info(f"Processing attribute: {attr_name}")
            self.logger.info(f"Slice: {mode} #{idx}")
            
            # 2. Compute Attribute
            self.preview_progress.value = 50
            attr_data = self.compute_attribute(raw_data, attr_name, export_mode=False)
            
            # 3. Prepare Visuals
            self.preview_progress.value = 80
            vis_seismic = np.swapaxes(display_data, 0, 1)
            vis_attr = np.swapaxes(attr_data, 0, 1)
            invert_y = (mode != "Timeslice")
            if invert_y:
                vis_seismic = vis_seismic[::-1, :]
                vis_attr = vis_attr[::-1, :]

            h, w = vis_seismic.shape
            bounds = (0, 0, w, h)
            amp_peak = max(abs(float(np.nanmax(display_data))), abs(float(np.nanmin(display_data))))
            seismic_clim_limit = max(float(self.amp_limit) * 1.5, amp_peak * 1.05)
            plot_bg_color = get_plot_surface_background(is_dark_mode)
            colorbar_opts = get_dark_colorbar_opts(is_dark_mode)

            hv_opts = dict(
                responsive=True,
                tools=['hover', 'crosshair', 'tap'], 
                active_tools=['box_zoom', 'tap'],
                invert_yaxis=invert_y,
                bgcolor=plot_bg_color,
                xlim=(0, w), ylim=(0, h),
            )

            base_seismic_left = hv.Image(
                vis_seismic, bounds=bounds, kdims=[x_dim, y_dim], label="Seismic"
            ).opts(
                **hv_opts, cmap=seis_cmap, clim=(-seismic_clim_limit, seismic_clim_limit),
                hooks=[self._apply_dark_plot_theme, self._capture_left_plot],
                colorbar=True, colorbar_opts=colorbar_opts, toolbar='above',
            )

            base_seismic_right = hv.Image(
                vis_seismic, bounds=bounds, kdims=[x_dim, y_dim], label="Seismic"
            ).opts(
                **hv_opts, cmap=seis_cmap, clim=(-seismic_clim_limit, seismic_clim_limit),
                hooks=[self._apply_dark_plot_theme, self._capture_right_plot],
                colorbar=False, toolbar=None,
            )

            self.tap_stream.source = base_seismic_left

            av_min = np.nanpercentile(attr_data, 2)
            av_max = np.nanpercentile(attr_data, 98)
            
            if attr_name == "Instantaneous Phase": av_min, av_max = -np.pi, np.pi
            elif attr_name == "Cosine of Phase": av_min, av_max = -1.0, 1.0
            elif attr_name == "Similarity (GST)": av_min, av_max = 0.0, 1.0
            elif attr_name == "Gradient Structure Tensor (GPU)": av_min, av_max = 0.0, 1.0

            img_attribute = hv.Image(
                vis_attr, bounds=bounds, kdims=[x_dim, y_dim], label=attr_name
            ).opts(
                **hv_opts, cmap=attr_cmap, clim=(av_min, av_max), alpha=opacity,
                hooks=[self._apply_dark_plot_theme],
                colorbar=True, colorbar_opts=colorbar_opts, toolbar=None
            )

            view_left = (base_seismic_left * self.marker_dmap).opts(
                title=f"{mode} {idx} (Seismic)",
            )
            
            view_right_overlay = (base_seismic_right * img_attribute * self.marker_dmap).opts(
                title=f"{mode} {idx} ({attr_name})",
            )

            self.pane_left_1.object = view_left
            self.pane_right_overlay.object = view_right_overlay
            
            # [NEW] Complete
            self.preview_progress.value = 100
            self.logger.info("Update preview completed")

        except Exception as e:
            self.logger.exception(f"Update error: {e}")
            self._show_empty_state(f"Update error: {e}")
        finally:
             if hasattr(self, 'slice_update_btn'): self.slice_update_btn.name = "Update Preview"
             if hasattr(self, 'attr_update_btn'): self.attr_update_btn.name = "Attribute Preview"
             self.preview_progress.visible = False

    def get_ram_usage(self):
        """Helper to get RAM usage string safely."""
        try:
            if psutil:
                process = psutil.Process(os.getpid())
                mem = process.memory_info().rss / (1024 ** 3) # GB
                return f"{mem:.2f} GB"
            return "N/A"
        except:
            return "N/A"

    def run_export(self, event):
        if not self.cube_guid:
            return
        if PetrelConnection is None:
            self.export_status.object = "Petrel export unavailable in this environment"
            self.logger.warning("Export skipped: PetrelConnection is not available")
            return
        self.logger.info("Export started")
        self.export_btn.disabled = True
        self.export_spinner.value = True
        self.export_progress.visible = True
        self.export_progress.value = 0
        self.export_status.object = "Exporting..."
        
        try:
            ptp = PetrelConnection(allow_experimental=True)
            src_cube = ptp.get_petrelobjects_by_guids([self.cube_guid])[0]
            target_cube = src_cube.clone(self.export_name_input.value, copy_values=False)
            i_max, j_max, k_max = src_cube.extent
            chunk_size = 32
            
            self.logger.info(f"Starting export for {self.attr_selector.value}")
            
            # [NEW] Timer Start
            t_start = time.time()
            
            for i in range(0, i_max, chunk_size):
                # Update Progress
                prog = int((i / i_max) * 100)
                self.export_progress.value = prog
                self.export_status.object = f"Processing Inlines {i}..."
                
                # [NEW] Calculate ETA & RAM
                elapsed = time.time() - t_start
                if i > 0:
                    rate = i / elapsed # inlines per second
                    remaining = i_max - i
                    eta_sec = remaining / rate
                    eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_sec))
                else:
                    eta_str = "Calculating..."
                
                ram_usage = self.get_ram_usage()
                
                self.logger.info(f"Export progress {prog}% | Inlines {i}/{i_max} | ETA {eta_str} | RAM {ram_usage}")
                
                src = src_cube.chunk((i, min(i+chunk_size, i_max)-1), (0, j_max-1), (0, k_max-1)).as_array()
                res = self.compute_attribute(src, self.attr_selector.value, export_mode=True)
                target_cube.chunk((i, min(i+chunk_size, i_max)-1), (0, j_max-1), (0, k_max-1)).set(res)
            
            self.export_progress.value = 100
            self.export_status.object = "Done!"
            
            total_time = time.time() - t_start
            self.logger.info(f"Export finished in {total_time:.1f}s")
            
        except Exception as e: 
            self.export_status.object = f"Error: {e}"
            self.logger.exception(f"Export error: {e}")
        finally:
            self.export_btn.disabled = False
            self.export_spinner.value = False
            self.logger.info("Export ended")

    def fetch_and_prepare_data(self, mode, idx):
        return self.data_store.get_slice(mode, int(idx))

    def compute_attribute(self, data, attr_name, export_mode=False):
        return compute_attribute(
            data=data,
            attr_name=attr_name,
            dt=self.dt_input.value,
            duration=self.duration_input.value,
            window_size=self.window_size_input.value,
            sigma=self.sigma_input.value,
            rho=self.rho_input.value,
            export_mode=export_mode,
        )

    def get_template(self):
        plot_title_color = get_content_text_color(is_dark_mode)
        plot_card_background = get_plot_surface_background(is_dark_mode)
        style = {
            'background': plot_card_background,
            'color': plot_title_color,
            'padding': '10px',
            'border-radius': '8px', 
            'box-shadow': '0 2px 5px rgba(0,0,0,0.1)',
            'flex': '1 1 0',
            'width': '100%',
            'height': '100%',
            'min-height': '0',
            'overflow': 'hidden',
            'display': 'flex',
            'flex-direction': 'column',
        }
        
        main = pn.FlexBox(
            pn.Column(
                pn.pane.HTML(f"<h3 style='margin:0; color:{plot_title_color};'>Seismic Preview</h3>"),
                self.left_plot_pane,
                styles=style,
                sizing_mode='stretch_both'
            ),
            pn.Column(
                pn.pane.HTML(f"<h3 style='margin:0; color:{plot_title_color};'>Seismic Attribute</h3>"),
                self.right_plot_pane,
                styles=style,
                sizing_mode='stretch_both'
            ),
            justify_content='space-between',
            align_items='stretch',
            gap='10px',
            flex_wrap='nowrap',
            sizing_mode='stretch_both',
        )
        main.styles = {
            'background': get_main_outer_background(is_dark_mode),
            'padding': '6px',
            'height': 'calc(100vh - 96px)',
            'width': '100%',
            'overflow': 'hidden',
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
                pn.pane.Markdown("### Dashboard"), 
                self.control_card, 
                pn.layout.Divider(), 
                self.attribute_card, 
                pn.layout.Divider(), 
                self.export_card
            ],
            header=[
                pn.Row(
                    pn.Spacer(sizing_mode='stretch_width'),
                    pn.pane.HTML(docs_button_html(DOCUMENTATION_URL)),
                    sizing_mode='stretch_width',
                    margin=0,
                )
            ],
            main=[main]
        )

app = SeismicApp()
template = app.get_template()
template.servable()

### Working code | Feb, 10th, 2026 | 15:55
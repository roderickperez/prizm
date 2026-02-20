import json
import logging
import os
import sys
import time
from pathlib import Path

import holoviews as hv
import numpy as np
import panel as pn
from holoviews import streams

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from shared.ui.omv_theme import (
    BLUE_OMV_COLOR,
    DARK_BLUE_OMV_COLOR,
    DARK_GREEN_OMV_COLOR,
    MAGENTA_OMV_COLOR,
    NEON_MAGENTA_OMV_COLOR,
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

try:
    from . import seismic_compute
except Exception:
    import seismic_compute

try:
    from cegalprizm.pythontool import PetrelConnection  # type: ignore[import-not-found]
except Exception:
    PetrelConnection = None


APP_TITLE = "Seismic Footprint Suppression (EMD)"
DOCUMENTATION_URL = "https://example.com/docs"
APP_DIR = Path(__file__).resolve().parent
LOG_DIR = ROOT_DIR / "seismicFootprint" / "logs"
SEGY_FILE_PATH = ROOT_DIR / "testData" / "1_Original_Seismics.sgy"
TEST_MODE = ("--test" in sys.argv) or (os.environ.get("PRIZM_FOOTPRINT_TEST_MODE", "0") in {"1", "true", "True"})

is_dark_mode = is_dark_mode_from_state()
pn.extension("tabulator", raw_css=get_extension_raw_css(is_dark_mode))
hv.extension("bokeh")

ASSETS_DIR = ROOT_DIR / "images" / "OMV_brandLogoPack" / "OMV_brandLogoPack"
LOGO_PATH = ASSETS_DIR / "OMV_logo_Neon_Small.png"
FAVICON_PATH = ASSETS_DIR / "OMV_logo_Blue_Small.png"

valid_logo = str(LOGO_PATH) if LOGO_PATH.exists() else None
valid_favicon = str(FAVICON_PATH) if FAVICON_PATH.exists() else None


def setup_app_logger(log_dir: Path, logger_name: str = "seismic_footprint_app") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"session_{time.strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(file_handler)

    existing_logs = sorted(log_dir.glob("session_*.log"), key=lambda path: path.stat().st_mtime)
    while len(existing_logs) > 10:
        oldest = existing_logs.pop(0)
        try:
            oldest.unlink()
        except OSError:
            pass

    logger.info("Logger initialized | test_mode=%s", TEST_MODE)
    return logger


APP_LOGGER = setup_app_logger(LOG_DIR)


def get_radio_group_stylesheets() -> list[str]:
    return [
        f"""
        .bk-btn-group > .bk-btn,
        .bk-btn-group > button.bk-btn {{
            background: {NEON_MAGENTA_OMV_COLOR} !important;
            color: {DARK_BLUE_OMV_COLOR} !important;
            border-color: {NEON_MAGENTA_OMV_COLOR} !important;
            font-weight: 600 !important;
        }}
        .bk-btn-group > .bk-btn.bk-active,
        .bk-btn-group > button.bk-btn.bk-active,
        .bk-btn-group > .bk-btn[aria-pressed='true'],
        .bk-btn-group > button.bk-btn[aria-pressed='true'] {{
            background: {MAGENTA_OMV_COLOR} !important;
            color: white !important;
            border-color: {MAGENTA_OMV_COLOR} !important;
        }}
        """
    ]


class SeismicFootprintApp:
    def __init__(self):
        self.logger = APP_LOGGER
        self.session_data = self.load_session_data()

        self.project_name = self.session_data.get("project", "Unknown")
        self.cube_name = self.session_data.get("selected_cube_name", SEGY_FILE_PATH.stem if TEST_MODE else "None")
        self.cube_guid = self.session_data.get("selected_cube_guid", None)

        self.compute_engine = seismic_compute.FootprintComputeEngine(logger=self.logger)
        self.dims = (1, 1, 1)
        self.amp_limit = 1000.0
        self.last_raw_slice = None
        self.last_filtered_slice = None

        self.select_stylesheets = get_dark_select_stylesheets(is_dark_mode)
        self.slider_stylesheets = get_slider_stylesheets()
        self.button_stylesheets = get_neon_button_stylesheets()
        self.radio_stylesheets = get_radio_group_stylesheets()
        self.section_header_background = NEON_OMV_COLOR
        self.section_header_text = DARK_BLUE_OMV_COLOR
        self.section_body_background = DARK_BLUE_OMV_COLOR if is_dark_mode else "white"
        self.section_body_text = get_content_text_color(is_dark_mode)

        self.info_card = self._create_sidebar_card(
            pn.Column(
                pn.pane.Markdown(f"**Project:** {self.project_name}"),
                pn.pane.Markdown(f"**Selected Volume:** {self.cube_name}"),
                pn.pane.Markdown(f"**Mode:** {'Test SEG-Y' if TEST_MODE else 'Petrel'}"),
            ),
            title="Session Info",
            collapsed=False,
        )

        self.radio_group = pn.widgets.RadioButtonGroup(
            name="Slice Type",
            options=["Inline", "Crossline", "Timeslice"],
            value="Timeslice",
            sizing_mode="stretch_width",
            stylesheets=self.radio_stylesheets,
        )
        self.slice_slider = pn.widgets.IntSlider(
            name="Slice Index",
            start=0,
            end=1,
            value=0,
            disabled=True,
            sizing_mode="stretch_width",
            stylesheets=self.slider_stylesheets,
        )
        self.seismic_cmap = pn.widgets.Select(
            name="Seismic Colormap",
            options=["RdBu", "gray", "bwr", "PuOr", "viridis"],
            value="RdBu",
            sizing_mode="stretch_width",
            stylesheets=self.select_stylesheets,
        )
        self.preview_btn = pn.widgets.Button(
            name="Update Preview",
            button_type="default",
            sizing_mode="stretch_width",
            stylesheets=self.button_stylesheets,
        )

        self.control_card = self._create_sidebar_card(
            pn.Column(self.radio_group, self.slice_slider, self.seismic_cmap, self.preview_btn),
            title="Slice Controls",
            collapsed=False,
        )

        self.direction_selector = pn.widgets.Select(
            name="Filter Direction",
            options=["Horizontal (Trace-wise)", "Vertical (Column-wise)"],
            value="Horizontal (Trace-wise)",
            sizing_mode="stretch_width",
            stylesheets=self.select_stylesheets,
        )
        self.imf_remove_input = pn.widgets.IntSlider(
            name="Remove First N IMFs",
            start=1,
            end=6,
            value=2,
            step=1,
            sizing_mode="stretch_width",
            stylesheets=self.slider_stylesheets,
        )
        self.eemd_trials_input = pn.widgets.IntInput(
            name="EEMD Trials",
            value=12,
            step=1,
            start=1,
            sizing_mode="stretch_width",
        )
        self.noise_width_input = pn.widgets.FloatInput(
            name="EEMD Noise Width",
            value=0.05,
            step=0.01,
            start=0.0,
            sizing_mode="stretch_width",
        )
        self.noise_amp_scale = pn.widgets.FloatSlider(
            name="Difference Scale Factor",
            start=0.2,
            end=2.0,
            step=0.1,
            value=1.0,
            sizing_mode="stretch_width",
            stylesheets=self.slider_stylesheets,
        )

        self.progress_bar = pn.indicators.Progress(name="Progress", value=0, max=100, visible=False, sizing_mode="stretch_width")
        self.progress_text = pn.pane.Markdown("", styles={"font-size": "0.85em"})
        self.update_btn = pn.widgets.Button(
            name="Run EEMD Filter",
            button_type="default",
            sizing_mode="stretch_width",
            stylesheets=self.button_stylesheets,
        )

        self.emd_card = self._create_sidebar_card(
            pn.Column(
                self.direction_selector,
                self.imf_remove_input,
                self.eemd_trials_input,
                self.noise_width_input,
                self.noise_amp_scale,
                pn.layout.Divider(),
                self.update_btn,
                self.progress_bar,
                self.progress_text,
            ),
            title="Footprint Suppression (EMD)",
            collapsed=False,
        )

        self.export_name_input = pn.widgets.TextInput(name="Output Name", value=f"{self.cube_name}_EMD_Filtered", sizing_mode="stretch_width")
        self.export_btn = pn.widgets.Button(
            name="Export to Petrel",
            button_type="default",
            sizing_mode="stretch_width",
            stylesheets=self.button_stylesheets,
            disabled=TEST_MODE,
        )
        self.export_status = pn.pane.Markdown("Ready", styles={"font-size": "0.85em"})
        self.export_card = self._create_sidebar_card(
            pn.Column(self.export_name_input, self.export_btn, self.export_status),
            title="Export",
            collapsed=False,
        )

        self.plot_orig_pane = pn.pane.HoloViews(sizing_mode="stretch_both")
        self.plot_filt_pane = pn.pane.HoloViews(sizing_mode="stretch_both")
        self.plot_diff_pane = pn.pane.HoloViews(sizing_mode="stretch_both")

        self.tap_stream = streams.Tap(x=0, y=0)
        self.plot_side_pane = pn.pane.HoloViews(hv.DynamicMap(self.plot_trace, streams=[self.tap_stream]), sizing_mode="stretch_both")

        self.radio_group.param.watch(self.update_slider_limits, "value")
        self.preview_btn.on_click(self.refresh_original_view)
        self.slice_slider.param.watch(self.refresh_original_view, "value")
        self.update_btn.on_click(self.run_emd_calculation)
        self.export_btn.on_click(self.run_export)

    def _create_sidebar_card(self, *widgets, title: str, collapsed: bool = True) -> pn.Card:
        return pn.Card(
            *widgets,
            title=title,
            collapsed=collapsed,
            hide_header=False,
            sizing_mode="stretch_width",
            header_background=self.section_header_background,
            active_header_background=self.section_header_background,
            header_color=self.section_header_text,
            styles={"background": self.section_body_background, "color": self.section_body_text},
            margin=(0, 0, 12, 0),
        )

    def load_session_data(self):
        data_file = os.environ.get("PWR_DATA_FILE")
        if data_file and os.path.exists(data_file):
            try:
                with open(data_file, "r", encoding="utf-8") as handle:
                    return json.load(handle)
            except Exception as exc:
                self.logger.warning("Failed reading PWR_DATA_FILE: %s", exc)
        return {}

    def startup_sequence(self):
        self.update_btn.disabled = True
        try:
            self.fetch_metadata_and_stats()
            self.refresh_original_view(None)
        finally:
            self.update_btn.disabled = False

    def fetch_metadata_and_stats(self):
        if TEST_MODE:
            loaded = self.compute_engine.load_test_volume(SEGY_FILE_PATH)
            if loaded:
                self.dims = self.compute_engine.dims
                self.amp_limit = self.compute_engine.amp_limit
                self.slice_slider.disabled = False
                self.update_slider_limits(None)
                self.slice_slider.value = self.slice_slider.end // 2
                self.logger.info("Loaded test SEG-Y volume: dims=%s amp_limit=%.2f", self.dims, self.amp_limit)
            return

        if not self.cube_guid or PetrelConnection is None:
            self.logger.warning("No Petrel cube selected")
            return

        try:
            ptp = PetrelConnection(allow_experimental=True)
            cube = ptp.get_petrelobjects_by_guids([self.cube_guid])[0]
            extent = cube.extent
            i_max = int(getattr(extent, "i", extent[0]))
            j_max = int(getattr(extent, "j", extent[1]))
            k_max = int(getattr(extent, "k", extent[2]))
            self.dims = (i_max, j_max, k_max)
            self.amp_limit = 2000.0
            self.slice_slider.disabled = False
            self.update_slider_limits(None)
            self.slice_slider.value = self.slice_slider.end // 2
            self.logger.info("Loaded Petrel cube metadata: dims=%s", self.dims)
        except Exception as exc:
            self.logger.exception("Metadata fetch failed: %s", exc)

    def update_slider_limits(self, _event):
        mode = self.radio_group.value
        if mode == "Inline":
            upper = max(0, self.dims[0] - 1)
        elif mode == "Crossline":
            upper = max(0, self.dims[1] - 1)
        else:
            upper = max(0, self.dims[2] - 1)
        self.slice_slider.end = upper
        if self.slice_slider.value > upper:
            self.slice_slider.value = upper

    def get_marker(self, x, y):
        if x is None or y is None:
            return hv.Points([]).opts(color="red")
        return hv.Points([(x, y)]).opts(color="red", size=10, fill_alpha=0, line_width=2, marker="circle")

    def fetch_slice_data(self, mode, idx):
        if TEST_MODE:
            data, labels = self.compute_engine.get_test_slice(mode, idx)
            return data, labels

        if PetrelConnection is None:
            raise RuntimeError("PetrelConnection is not available")
        if not self.cube_guid:
            raise RuntimeError("No cube GUID provided")

        ptp = PetrelConnection(allow_experimental=True)
        cube = ptp.get_petrelobjects_by_guids([self.cube_guid])[0]
        extent = cube.extent
        i_max = int(getattr(extent, "i", extent[0]))
        j_max = int(getattr(extent, "j", extent[1]))
        k_max = int(getattr(extent, "k", extent[2]))

        if mode == "Inline":
            chunk = cube.chunk((idx, idx), (0, j_max - 1), (0, k_max - 1)).as_array()[0, :, :]
            return chunk, ("Crossline", "Time/Depth")
        if mode == "Crossline":
            chunk = cube.chunk((0, i_max - 1), (idx, idx), (0, k_max - 1)).as_array()[:, 0, :]
            return chunk, ("Inline", "Time/Depth")

        chunk = cube.chunk((0, i_max - 1), (0, j_max - 1), (idx, idx)).as_array()[:, :, 0]
        return chunk, ("Inline", "Crossline")

    def build_image(self, data_2d, labels, cmap, title, clim):
        vis = np.asarray(data_2d).T
        h, w = vis.shape
        bounds = (0, 0, w, h)
        tools_list = ["hover", "crosshair", "box_zoom", "pan", "reset", "tap"]
        return hv.Image(vis, bounds=bounds, kdims=[labels[0], labels[1]], label=title).opts(
            cmap=cmap,
            clim=clim,
            colorbar=True,
            toolbar="above",
            aspect="equal",
            responsive=True,
            min_height=500,
            tools=tools_list,
            title=title,
        )

    def refresh_original_view(self, _event):
        if self.slice_slider.disabled:
            return
        try:
            mode = self.radio_group.value
            idx = self.slice_slider.value
            cmap = self.seismic_cmap.value
            raw_data, labels = self.fetch_slice_data(mode, idx)
            self.last_raw_slice = raw_data
            base_clim = (-self.amp_limit, self.amp_limit)
            img_orig = self.build_image(raw_data, labels, cmap, f"Original ({mode} {idx})", base_clim)
            self.tap_stream.source = img_orig
            marker_dmap = hv.DynamicMap(self.get_marker, streams=[self.tap_stream])
            self.plot_orig_pane.object = img_orig * marker_dmap

            placeholder = hv.Text(0.5, 0.5, "Run EEMD Filter\nto view results").opts(xaxis=None, yaxis=None, text_align="center")
            self.plot_filt_pane.object = placeholder
            self.plot_diff_pane.object = placeholder
        except Exception as exc:
            self.logger.exception("Preview refresh failed: %s", exc)
            self.plot_orig_pane.object = hv.Text(0.5, 0.5, f"Error: {exc}")

    def _on_parallel_progress(self, completed, total):
        pct = int((completed / max(1, total)) * 100)
        self.progress_bar.value = pct
        self.progress_text.object = f"Processing... {completed}/{total} ({pct}%)"

    def run_emd_calculation(self, _event):
        if self.slice_slider.disabled:
            return

        mode = self.radio_group.value
        idx = self.slice_slider.value
        cmap = self.seismic_cmap.value
        emd_axis = 1 if "Horizontal" in self.direction_selector.value else 0

        self.update_btn.disabled = True
        self.update_btn.name = "Running EEMD..."
        self.progress_bar.visible = True
        self.progress_bar.value = 0
        self.progress_text.object = "Initializing..."

        try:
            raw_data, labels = self.fetch_slice_data(mode, idx)
            denoised_data, noise_data = self.compute_engine.apply_eemd_parallel(
                data_2d=raw_data,
                axis=emd_axis,
                imfs_to_remove=self.imf_remove_input.value,
                trials=self.eemd_trials_input.value,
                noise_width=self.noise_width_input.value,
                progress_callback=self._on_parallel_progress,
            )
            self.last_raw_slice = raw_data
            self.last_filtered_slice = denoised_data

            base_clim = (-self.amp_limit, self.amp_limit)
            diff_scale = float(max(0.2, self.noise_amp_scale.value))
            diff_clim = (-self.amp_limit / diff_scale, self.amp_limit / diff_scale)

            marker_dmap = hv.DynamicMap(self.get_marker, streams=[self.tap_stream])
            self.plot_filt_pane.object = self.build_image(denoised_data, labels, cmap, "EMD Filtered", base_clim) * marker_dmap
            self.plot_diff_pane.object = self.build_image(noise_data, labels, cmap, "Difference (Footprint)", diff_clim) * marker_dmap

            self.progress_bar.value = 100
            self.progress_text.object = "✅ Done"
            self.logger.info("EEMD filtering complete | mode=%s slice=%s axis=%s", mode, idx, emd_axis)
        except Exception as exc:
            self.progress_text.object = f"❌ Error: {exc}"
            self.logger.exception("EEMD run failed: %s", exc)
        finally:
            self.update_btn.disabled = False
            self.update_btn.name = "Run EEMD Filter"

    def _fetch_trace_data(self, i_idx, j_idx):
        if TEST_MODE and self.compute_engine.has_test_volume and self.compute_engine.trace_index_grid is not None:
            if not (0 <= i_idx < self.compute_engine.dims[0] and 0 <= j_idx < self.compute_engine.dims[1]):
                return None
            trace_idx = int(self.compute_engine.trace_index_grid[i_idx, j_idx])
            if trace_idx < 0:
                return None
            trace = np.asarray(self.compute_engine._segy_file.trace[trace_idx], dtype=np.float32)
            return trace

        if not self.cube_guid or PetrelConnection is None:
            return None
        try:
            ptp = PetrelConnection(allow_experimental=True)
            cube = ptp.get_petrelobjects_by_guids([self.cube_guid])[0]
            trace_data = cube.chunk((i_idx, i_idx), (j_idx, j_idx), (0, self.dims[2] - 1)).as_array().flatten()
            return np.asarray(trace_data, dtype=np.float32)
        except Exception:
            return None

    def plot_trace(self, x, y):
        if x is None or y is None:
            i_idx = self.dims[0] // 2
            j_idx = self.dims[1] // 2
        else:
            i_idx = int(round(x))
            j_idx = int(round(y))

        trace_data = self._fetch_trace_data(i_idx, j_idx)
        if trace_data is None:
            return (hv.Curve([]) * hv.HLine(0)).opts(title="Trace unavailable")

        time_axis = np.arange(trace_data.size)
        curve_orig = hv.Curve((trace_data, time_axis), kdims=["Amplitude"], vdims=["Time/Depth"], label="Original").opts(
            color="black", line_width=1, invert_yaxis=True
        )

        overlays = [curve_orig]
        if self.last_filtered_slice is not None:
            try:
                filt_val = np.asarray(self.last_filtered_slice).T
                fx = int(np.clip(i_idx, 0, filt_val.shape[1] - 1))
                fy = int(np.clip(j_idx, 0, filt_val.shape[0] - 1))
                filtered_trace = np.repeat(filt_val[fy, fx], trace_data.size)
                overlays.append(hv.Curve((filtered_trace, time_axis), kdims=["Amplitude"], vdims=["Time/Depth"], label="Filtered Amp").opts(color="red", line_width=1))
            except Exception:
                pass

        return hv.Overlay(overlays).opts(
            title=f"Trace View (IL:{i_idx}, XL:{j_idx})",
            invert_yaxis=True,
            legend_position="top_right",
            toolbar="above",
            responsive=True,
            min_height=600,
        )

    def run_export(self, _event):
        if TEST_MODE:
            self.export_status.object = "Test mode active: export to Petrel is disabled"
            return

        if not self.cube_guid or PetrelConnection is None:
            self.export_status.object = "No Petrel cube available"
            return

        if self.last_filtered_slice is None:
            self.export_status.object = "Run filtering before export"
            return

        self.export_btn.disabled = True
        try:
            ptp = PetrelConnection(allow_experimental=True)
            src_cube = ptp.get_petrelobjects_by_guids([self.cube_guid])[0]
            out_name = self.export_name_input.value.strip()
            if not out_name:
                self.export_status.object = "Enter output name"
                return

            target_cube = src_cube.clone(out_name, copy_values=False)
            extent = src_cube.extent
            i_max = int(getattr(extent, "i", extent[0]))
            j_max = int(getattr(extent, "j", extent[1]))
            k_max = int(getattr(extent, "k", extent[2]))

            emd_axis = 1 if "Horizontal" in self.direction_selector.value else 0
            imfs_remove = int(self.imf_remove_input.value)
            trials = int(self.eemd_trials_input.value)
            noise_width = float(self.noise_width_input.value)

            for k_idx in range(k_max):
                self.export_status.object = f"Processing timeslice {k_idx + 1}/{k_max}"
                src_chunk = src_cube.chunk((0, i_max - 1), (0, j_max - 1), (k_idx, k_idx))
                slice_data = np.asarray(src_chunk.as_array()[:, :, 0], dtype=np.float32)
                denoised, _ = self.compute_engine.apply_eemd_parallel(
                    data_2d=slice_data,
                    axis=emd_axis,
                    imfs_to_remove=imfs_remove,
                    trials=trials,
                    noise_width=noise_width,
                    progress_callback=None,
                )

                dst_chunk = target_cube.chunk((0, i_max - 1), (0, j_max - 1), (k_idx, k_idx))
                dst_chunk.set(denoised[:, :, np.newaxis])

            self.export_status.object = f"✅ Export complete: {target_cube.petrel_name}"
            self.logger.info("Export completed to cube: %s", out_name)
        except Exception as exc:
            self.export_status.object = f"Export error: {exc}"
            self.logger.exception("Export failed: %s", exc)
        finally:
            self.export_btn.disabled = False

    def get_template(self):
        box_style = {
            "background": "#ffffff" if not is_dark_mode else DARK_BLUE_OMV_COLOR,
            "border-radius": "5px",
            "box-shadow": "0px 0px 5px rgba(0,0,0,0.1)",
            "padding": "10px",
            "border": "1px solid #e0e0e0",
            "overflow": "hidden",
            "color": self.section_body_text,
        }

        col_orig = pn.Column(pn.pane.Markdown("### 1. Original Seismic"), self.plot_orig_pane, styles=box_style, sizing_mode="stretch_both")
        col_filt = pn.Column(pn.pane.Markdown("### 2. EMD Filtered"), self.plot_filt_pane, styles=box_style, sizing_mode="stretch_both")
        col_diff = pn.Column(pn.pane.Markdown("### 3. Difference (Footprint)"), self.plot_diff_pane, styles=box_style, sizing_mode="stretch_both")
        col_trace = pn.Column(pn.pane.Markdown("### 4. Trace View"), self.plot_side_pane, styles=box_style, sizing_mode="stretch_both")

        main_layout = pn.Row(col_orig, col_filt, col_diff, col_trace, sizing_mode="stretch_both")
        main_content = pn.Column(
            main_layout,
            sizing_mode="stretch_both",
            margin=0,
            styles={
                "height": "100%",
                "overflow": "hidden",
                "background": get_main_outer_background(is_dark_mode),
                "color": self.section_body_text,
            },
        )

        sidebar_content = [
            self.info_card,
            pn.Spacer(height=8),
            self.control_card,
            pn.Spacer(height=8),
            self.emd_card,
            pn.Spacer(height=8),
            self.export_card,
        ]

        return pn.template.FastListTemplate(
            title=APP_TITLE,
            logo=valid_logo,
            favicon=valid_favicon,
            accent_base_color=BLUE_OMV_COLOR,
            header_background=DARK_BLUE_OMV_COLOR,
            header=[pn.Row(pn.Spacer(sizing_mode="stretch_width"), pn.pane.HTML(docs_button_html(DOCUMENTATION_URL)), sizing_mode="stretch_width", margin=0)],
            sidebar=sidebar_content,
            main=[main_content],
            main_layout=None,
            main_max_width="",
        )


if __name__.startswith("bokeh"):
    app = SeismicFootprintApp()
    template = app.get_template()
    pn.state.onload(lambda: app.startup_sequence())
    template.servable()

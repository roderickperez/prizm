import sys
import os
import time
import psutil
from pathlib import Path
from datetime import datetime # <--- Added for timestamps

# --- CRITICAL FIX FOR PARALLEL PROCESSING ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
if current_script_dir not in sys.path:
    sys.path.append(current_script_dir)

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import panel as pn
import pandas as pd
import json
import numpy as np
import holoviews as hv
from holoviews import opts, streams
from scipy.ndimage import generic_filter
import concurrent.futures 

from shared.ui.omv_theme import DARK_BLUE_OMV_COLOR, MAGENTA_OMV_COLOR, NEON_MAGENTA_OMV_COLOR

# --- IMPORT LOCAL WORKER MODULE ---
try:
    import seismic_compute
except ImportError:
    print("WARNING: Could not import seismic_compute.py. Parallel processing will fail.")

# --- EMD Import ---
try:
    from PyEMD import EEMD
    HAS_EMD = True
    print("SUCCESS: PyEMD library loaded.")
except ImportError:
    HAS_EMD = False
    print("WARNING: PyEMD library not found.")

from cegalprizm.pythontool import PetrelConnection

# --- App UI Constants ---
APP_TITLE = "Seismic Footprint Suppression (EMD)"
ACCENT_COLOR = "#052759"

pn.extension('tabulator')
hv.extension('bokeh')

# --- Image Paths ---
LOGO_PATH = r"N:\_USER_GLOBAL\PETREL\Prizm\wf\Vienna\images\OMV_brandLogoPack\OMV_brandLogoPack\OMV_logo_Neon_Small.png"
FAVICON_PATH = r"N:\_USER_GLOBAL\PETREL\Prizm\wf\Vienna\images\OMV_brandLogoPack\OMV_brandLogoPack\OMV_logo_Blue_Small.png"

valid_logo = LOGO_PATH if os.path.exists(LOGO_PATH) else None
valid_favicon = FAVICON_PATH if os.path.exists(FAVICON_PATH) else None


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

# ==============================================================================
#  SEISMIC FOOTPRINT APP CLASS
# ==============================================================================
class SeismicFootprintApp:
    def __init__(self):
        self.session_data = self.load_session_data()
        self.project_name = self.session_data.get("project", "Unknown")
        self.cube_name = self.session_data.get("selected_cube_name", "None")
        self.cube_guid = self.session_data.get("selected_cube_guid", None)
        
        self.dims = (1, 1, 1) 
        self.amp_limit = 1000.0 
        self.radio_group_stylesheets = get_radio_group_stylesheets()
        
        self.info_card = pn.Card(
            pn.Column(
                pn.pane.Markdown(f"**Project:** {self.project_name}"),
                pn.pane.Markdown(f"**Selected Volume:** {self.cube_name}"),
            ),
            title="Session Info", collapsed=True, styles={'background': 'white'}
        )

        self.radio_group = pn.widgets.RadioButtonGroup(
            name='Slice Type',
            options=['Timeslice'],
            button_type='success',
            value='Timeslice',
            stylesheets=self.radio_group_stylesheets,
        )
        self.slice_slider = pn.widgets.IntSlider(name='Time Slice Selection', start=0, end=1, value=0, disabled=True)
        self.seismic_cmap = pn.widgets.Select(name='Seismic Colormap', options=['RdBu', 'gray', 'bwr', 'PuOr', 'viridis'], value='RdBu')

        self.control_card = pn.Card(
            pn.Column(
                pn.pane.Markdown("**Orientation:**"), self.radio_group, self.slice_slider,
                pn.pane.Markdown("**Visualization:**"), self.seismic_cmap,
            ),
            title="Slice Controls", collapsed=True, styles={'background': 'white'}
        )

        self.direction_selector = pn.widgets.Select(name='Filter Direction', options=['Horizontal (Trace-wise)', 'Vertical (Column-wise)'], value='Horizontal (Trace-wise)')
        self.imf_remove_input = pn.widgets.IntSlider(name="Remove First N IMFs", start=1, end=5, value=1, step=1)
        self.noise_amp_scale = pn.widgets.FloatSlider(name='Difference Scale Factor', start=0.1, end=2.0, step=0.1, value=1.0)
        
        self.progress_bar = pn.indicators.Progress(name='Progress', value=0, max=100, bar_color='success', height=20, visible=False)
        self.progress_text = pn.pane.Markdown("", styles={'font-size': '0.8em', 'color': '#666'})

        self.update_btn = pn.widgets.Button(name='Run EMD Filter (Parallel)', button_type='primary')

        self.emd_card = pn.Card(
            pn.Column(
                pn.pane.Markdown("**EMD Settings:**"), self.direction_selector, self.imf_remove_input,
                pn.layout.Divider(), pn.pane.Markdown("**Visuals:**"), self.noise_amp_scale,
                pn.layout.Divider(), self.update_btn, self.progress_bar, self.progress_text
            ),
            title="Footprint Suppression (EMD)", collapsed=True, styles={'background': 'white'}
        )

        self.export_name_input = pn.widgets.TextInput(name='Output Name', value=f"{self.cube_name}_EMD_Filtered")
        self.export_btn = pn.widgets.Button(name='Export to Petrel', button_type='warning')
        self.export_spinner = pn.indicators.LoadingSpinner(value=False, width=25, height=25, align='center')
        self.export_status = pn.pane.Markdown("Ready", styles={'font-size': '0.9em', 'color': '#666'})

        self.export_card = pn.Card(
            pn.Column(self.export_name_input, self.export_btn, pn.Row(self.export_spinner, self.export_status)),
            title="Export", collapsed=True, styles={'background': 'white'}
        )

        self.plot_orig_pane = pn.pane.HoloViews(sizing_mode='stretch_both')
        self.plot_filt_pane = pn.pane.HoloViews(sizing_mode='stretch_both')
        self.plot_diff_pane = pn.pane.HoloViews(sizing_mode='stretch_both')
        
        self.tap_stream = streams.Tap(x=0, y=0)
        self.plot_side_pane = pn.pane.HoloViews(hv.DynamicMap(self.plot_trace, streams=[self.tap_stream]), sizing_mode='stretch_both')

        self.radio_group.param.watch(self.update_slider_limits, 'value')
        self.update_btn.on_click(self.run_emd_calculation)
        self.slice_slider.param.watch(self.refresh_original_view, 'value')
        self.export_btn.on_click(self.run_export)

    def load_session_data(self):
        data_file = os.environ.get("PWR_DATA_FILE")
        if data_file and os.path.exists(data_file):
            try:
                with open(data_file, "r") as f: return json.load(f)
            except Exception: pass
        return {}

    def startup_sequence(self):
        self.update_btn.disabled = True
        try:
            print("[App]: Starting delayed initialization...")
            self.fetch_metadata_and_stats()
            self.refresh_original_view(None)
        finally:
            self.update_btn.disabled = False
            print("[App]: Initialization complete.")

    def fetch_metadata_and_stats(self):
        if not self.cube_guid: return
        print("[App]: Fetching metadata from Petrel...")
        try:
            ptp = PetrelConnection(allow_experimental=True)
            objs = ptp.get_petrelobjects_by_guids([self.cube_guid])
            if objs:
                cube = objs[0]
                self.dims = cube.extent 
                print(f"[App]: Cube dimensions: {self.dims}")
                k_max = self.dims[2]
                self.slice_slider.end = max(0, k_max - 1)
                self.slice_slider.value = k_max // 2
                self.slice_slider.disabled = False
                i_mid = self.dims.i // 2
                chunk = cube.chunk((i_mid, i_mid), (0, self.dims.j-1), (0, self.dims.k-1)).as_array()
                sample_data = chunk.flatten()
                sample_data = sample_data[~np.isnan(sample_data)]
                if len(sample_data) > 0: self.amp_limit = np.percentile(np.abs(sample_data), 98)
                else: self.amp_limit = 2000.0
                print(f"[App]: Calculated Amp Limit: {self.amp_limit}")
        except Exception as e: print(f"[App]: Meta fetch error: {e}")

    def update_slider_limits(self, event):
        k_max = self.dims[2]
        self.slice_slider.end = max(0, k_max - 1)

    def get_marker(self, x, y):
        if x is None or y is None: return hv.Points([]).opts(color='red')
        return hv.Points([(x, y)]).opts(color='red', size=10, fill_alpha=0, line_width=2, marker='circle')

    # --- ROBUST PARALLEL IMPLEMENTATION (Safety Timeouts + Time Logging) ---
    def apply_emd_filter_parallel(self, data_2d, axis, imfs_to_remove):
        if not HAS_EMD: return data_2d, np.zeros_like(data_2d)

        rows, cols = data_2d.shape
        denoised = np.zeros_like(data_2d)
        noise = np.zeros_like(data_2d)
        
        if axis == 1: 
            tasks = [data_2d[r, :] for r in range(rows)]
            total_tasks = rows
        else: 
            tasks = [data_2d[:, c] for c in range(cols)]
            total_tasks = cols
            
        results = []
        print(f"[App] Starting Parallel EMD on {total_tasks} traces...") 

        start_time_total = time.time()
        batch_start_time = time.time()

        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_to_idx = {
                executor.submit(seismic_compute.emd_worker, signal, imfs_to_remove): i 
                for i, signal in enumerate(tasks)
            }
            
            completed_count = 0
            update_interval = max(5, total_tasks // 20) 

            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    # --- SAFETY TIMEOUT: 60 Seconds per trace ---
                    # If a single trace hangs, we skip it to save the app.
                    res_signal, res_noise = future.result(timeout=60)
                    results.append((idx, res_signal, res_noise))
                except concurrent.futures.TimeoutError:
                    print(f"[WARN] Trace {idx} timed out (bad data?). Skipping.")
                    # Fill with zeros or original data so we don't crash
                    results.append((idx, np.zeros_like(tasks[idx]), np.zeros_like(tasks[idx])))
                except Exception as e:
                    print(f"[ERROR] Trace {idx} failed: {e}")
                
                completed_count += 1
                
                # Update UI & Terminal
                if completed_count % update_interval == 0 or completed_count == total_tasks:
                    current_time = time.time()
                    batch_duration = current_time - batch_start_time
                    timestamp_str = datetime.now().strftime("%H:%M:%S")
                    
                    mem = psutil.virtual_memory()
                    
                    prog_val = int((completed_count / total_tasks) * 100)
                    self.progress_bar.value = prog_val
                    
                    # LOGGING FORMAT REQUESTED
                    msg = f"Batch: {completed_count}/{total_tasks} | Start: {timestamp_str} | Time: {batch_duration:.2f}s | RAM: {mem.percent}%"
                    self.progress_text.object = f"Processing... {completed_count}/{total_tasks} ({prog_val}%)"
                    print(f"[App] {msg}")
                    
                    # Reset batch timer
                    batch_start_time = time.time()

        print(f"[App] Total Processing Time: {time.time() - start_time_total:.2f}s")

        for idx, clean, dirty in results:
            if axis == 1:
                denoised[idx, :] = clean
                noise[idx, :] = dirty
            else:
                denoised[:, idx] = clean
                noise[:, idx] = dirty
                
        return denoised, noise

    def plot_trace(self, x, y):
        def safe_empty_overlay(title_text): return (hv.Curve([]) * hv.Curve([]) * hv.HLine(0)).opts(title=title_text)
        if x is None or y is None: target_i, target_j = self.dims[0]//2, self.dims[1]//2
        else: target_i, target_j = int(round(x)), int(round(y))

        if not self.cube_guid or self.dims == (1,1,1): return safe_empty_overlay("Waiting for Data...")

        try:
            ptp = PetrelConnection(allow_experimental=True)
            cube = ptp.get_petrelobjects_by_guids([self.cube_guid])[0]
            current_time_idx = self.slice_slider.value
            if not (0 <= target_i < self.dims[0] and 0 <= target_j < self.dims[1]): return safe_empty_overlay("Selection out of bounds")

            chunk = cube.chunk((target_i, target_i), (target_j, target_j), (0, self.dims[2]-1))
            trace_data = chunk.as_array().flatten() 
            time_axis = np.arange(len(trace_data))
            
            trace_opts = dict(line_width=1, invert_yaxis=True, responsive=True, aspect=None, framewise=True, xticks=3, show_grid=False)
            curve_orig = hv.Curve((trace_data, time_axis), kdims=['Amplitude'], vdims=['Time'], label='Original').opts(color='black', title=f"Trace (IL:{target_i}, XL:{target_j})", **trace_opts)
            curve_emd = hv.Curve([], kdims=['Amplitude'], vdims=['Time'], label='EMD Filtered').opts(**trace_opts)
            hline = hv.HLine(current_time_idx).opts(color='red', line_width=2)
            return (curve_orig * curve_emd * hline).opts(invert_yaxis=True, legend_position='top_right', toolbar='above', responsive=True, aspect=None, min_height=600)
        except Exception as e:
            print(f"Trace fetch error: {e}")
            return safe_empty_overlay("Error fetching trace")

    def refresh_original_view(self, event):
        if not self.cube_guid: return
        mode = 'Timeslice' 
        idx = self.slice_slider.value
        cmap = self.seismic_cmap.value
        try:
            raw_data, labels = self.fetch_slice_data(mode, idx)
            x_dim, y_dim = labels 
            vis_orig = raw_data.T
            h, w = vis_orig.shape
            bounds = (0, 0, w, h)
            base_clim = (-self.amp_limit, self.amp_limit)
            
            tools_list = ['hover', 'crosshair', 'box_zoom', 'pan', 'reset', 'tap']
            common_opts = dict(cmap=cmap, clim=base_clim, colorbar=True, toolbar='above', aspect='equal', min_height=500, responsive=True, tools=tools_list)

            img_orig = hv.Image(vis_orig, bounds=bounds, kdims=[x_dim, y_dim], label="Original").opts(**common_opts, title=f"Original (Timeslice {idx})")
            self.tap_stream.source = img_orig
            marker_dmap = hv.DynamicMap(self.get_marker, streams=[self.tap_stream])

            self.plot_orig_pane.object = img_orig * marker_dmap
            msg_plot = hv.Text(w/2, h/2, "Click 'Run EMD Filter'\nto see results").opts(xaxis=None, yaxis=None, text_align='center', text_color='black', text_font_size='12pt')
            self.plot_filt_pane.object = msg_plot
            self.plot_diff_pane.object = msg_plot
        except Exception as e: self.plot_orig_pane.object = hv.Text(0.5, 0.5, f"Error: {str(e)}")

    def run_emd_calculation(self, event):
        if not self.cube_guid: return
        mode = 'Timeslice' 
        idx = self.slice_slider.value
        cmap = self.seismic_cmap.value
        direction_str = self.direction_selector.value
        imfs_remove = self.imf_remove_input.value
        diff_scale = self.noise_amp_scale.value
        emd_axis = 1 if 'Horizontal' in direction_str else 0

        self.update_btn.name = "Calculating EMD (Parallel)..."
        self.update_btn.disabled = True
        self.progress_bar.visible = True
        self.progress_bar.value = 0
        self.progress_text.object = "Initializing Workers..."
        
        try:
            raw_data, labels = self.fetch_slice_data(mode, idx)
            x_dim, y_dim = labels 
            denoised_data, noise_data = self.apply_emd_filter_parallel(raw_data, emd_axis, imfs_remove)
            
            vis_denoised = denoised_data.T
            vis_diff = noise_data.T
            h, w = vis_denoised.shape
            bounds = (0, 0, w, h)
            base_clim = (-self.amp_limit, self.amp_limit)
            diff_clim = (-self.amp_limit / diff_scale, self.amp_limit / diff_scale) if diff_scale != 1.0 else base_clim

            tools_list = ['hover', 'crosshair', 'box_zoom', 'pan', 'reset', 'tap']
            common_opts = dict(cmap=cmap, clim=base_clim, colorbar=True, toolbar='above', aspect='equal', min_height=500, responsive=True, tools=tools_list)

            img_filt = hv.Image(vis_denoised, bounds=bounds, kdims=[x_dim, y_dim], label="EMD Filtered").opts(**common_opts, title=f"EMD Output")
            img_diff = hv.Image(vis_diff, bounds=bounds, kdims=[x_dim, y_dim], label="Difference").opts(cmap=cmap, clim=diff_clim, colorbar=True, toolbar='above', aspect='equal', min_height=500, responsive=True, title=f"Difference", tools=tools_list)
            
            marker_dmap = hv.DynamicMap(self.get_marker, streams=[self.tap_stream])
            self.plot_filt_pane.object = img_filt * marker_dmap
            self.plot_diff_pane.object = img_diff * marker_dmap
            self.progress_text.object = "✅ Done!"
            print("[App] Calculation Finished!")

        except Exception as e:
            self.plot_filt_pane.object = hv.Text(0.5, 0.5, f"Error: {str(e)}")
            self.progress_text.object = f"❌ Error: {str(e)}"
            print(f"[Error] Update Failed: {e}")
        finally:
             self.update_btn.name = "Run EMD Filter (Parallel)"
             self.update_btn.disabled = False

    def run_export(self, event):
        if not self.cube_guid: return
        new_name = self.export_name_input.value
        if not new_name:
            self.export_status.object = "⚠️ Enter a name first!"
            return
        self.export_btn.disabled = True
        self.export_spinner.value = True
        self.export_status.object = "Initializing Export..."
        try:
            ptp = PetrelConnection(allow_experimental=True)
            src_cube = ptp.get_petrelobjects_by_guids([self.cube_guid])[0]
            self.export_status.object = f"Creating cube '{new_name}'..."
            target_cube = src_cube.clone(new_name, copy_values=False)
            i_max, j_max, k_max = src_cube.extent
            
            direction_str = self.direction_selector.value
            imfs_remove = self.imf_remove_input.value
            emd_axis = 1 if 'Horizontal' in direction_str else 0
            
            chunk_size_z = 1 
            range_k = range(0, k_max, chunk_size_z)
            total_chunks = len(range_k)
            
            for count, k_start in enumerate(range_k):
                k_end = min(k_start + chunk_size_z, k_max)
                self.export_status.object = f"Processing TimeSlice {k_start}-{k_end} ({count+1}/{total_chunks})"
                
                src_chunk = src_cube.chunk((0, i_max-1), (0, j_max-1), (k_start, k_end-1))
                data_block = src_chunk.as_array()
                result_block = np.zeros_like(data_block)
                for z_local in range(data_block.shape[2]):
                    slice_2d = data_block[:, :, z_local]
                    denoised, _ = self.apply_emd_filter_parallel(slice_2d, emd_axis, imfs_remove)
                    result_block[:, :, z_local] = denoised
                
                dst_chunk = target_cube.chunk((0, i_max-1), (0, j_max-1), (k_start, k_end-1))
                dst_chunk.set(result_block)
            
            self.export_status.object = f"✅ Export Complete: {new_name}"
            self.export_name_input.value = f"{new_name}_v2" 
        except Exception as e:
            self.export_status.object = f"❌ Error: {str(e)}"
            print(f"Export Error: {e}")
        finally:
            self.export_btn.disabled = False
            self.export_spinner.value = False

    def fetch_slice_data(self, mode, idx):
        ptp = PetrelConnection(allow_experimental=True)
        cube = ptp.get_petrelobjects_by_guids([self.cube_guid])[0]
        i_ext, j_ext, k_ext = cube.extent
        chunk = cube.chunk((0, i_ext - 1), (0, j_ext - 1), (idx, idx))
        data = chunk.as_array()[:, :, 0] 
        return data, ("Inline", "Crossline")

    def get_template(self):
        main_panel_style = {'background': '#ffffff', 'padding': '5px', 'border-radius': '8px', 'box-shadow': '0px 2px 5px rgba(0,0,0,0.1)', 'width': 'calc((100% - 30px) * 0.3125)', 'height': 'calc(100vh - 220px)', 'overflow': 'hidden'}
        side_panel_style = {'background': '#ffffff', 'padding': '5px', 'border-radius': '8px', 'box-shadow': '0px 2px 5px rgba(0,0,0,0.1)', 'width': 'calc((100% - 30px) * 0.0625)', 'height': 'calc(100vh - 220px)', 'overflow': 'hidden'}

        col_orig = pn.Column(pn.pane.Markdown("### 1. Original Seismic"), self.plot_orig_pane, styles=main_panel_style, sizing_mode='stretch_both')
        col_filt = pn.Column(pn.pane.Markdown("### 2. EMD Filtered"), self.plot_filt_pane, styles=main_panel_style, sizing_mode='stretch_both')
        col_diff = pn.Column(pn.pane.Markdown("### 3. Difference (Noise)"), self.plot_diff_pane, styles=main_panel_style, sizing_mode='stretch_both')
        col_side = pn.Column(pn.pane.Markdown("### 4. Trace View"), self.plot_side_pane, styles=side_panel_style, sizing_mode='stretch_both')

        main_flex = pn.FlexBox(col_orig, col_filt, col_diff, col_side, flex_direction='row', flex_wrap='nowrap', justify_content='space-between', align_items='stretch', gap='10px', sizing_mode='stretch_both')
        header_items = [pn.Spacer(sizing_mode='stretch_width'), pn.pane.HTML("""<a href="#" style="color: white; font-weight: bold; margin-right: 20px;">Documentation</a>""", align='center')]

        template = pn.template.FastListTemplate(title=APP_TITLE, logo=valid_logo, favicon=valid_favicon, accent_base_color=ACCENT_COLOR, header_background=ACCENT_COLOR, header=header_items, sidebar=[pn.pane.Markdown("### Dashboard"), self.info_card, pn.layout.Divider(), self.control_card, pn.layout.Divider(), self.emd_card, pn.layout.Divider(), self.export_card], main=[pn.pane.Markdown(f"# {APP_TITLE}"), main_flex])
        return template

if __name__.startswith("bokeh"):
    app = SeismicFootprintApp()
    template = app.get_template()
    pn.state.onload(lambda: app.startup_sequence())
    template.servable()
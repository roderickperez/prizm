import sys
import os
import time
import psutil
from datetime import datetime
import json
import numpy as np
import panel as pn
import holoviews as hv
from holoviews import opts, streams
from scipy.ndimage import gaussian_filter
import matplotlib.colors as mcolors

# --- Prizm Imports ---
from cegalprizm.pythontool import PetrelConnection

# --- App UI Constants ---
APP_TITLE = "Seismic Curvature Analysis"
ACCENT_COLOR = "#052759"

# --- Initialize Panel & HoloViews ---
pn.extension('tabulator')
hv.extension('bokeh')

# --- Image Paths ---
LOGO_PATH = r"N:\_USER_GLOBAL\PETREL\Prizm\wf\Vienna\images\OMV_brandLogoPack\OMV_brandLogoPack\OMV_logo_Neon_Small.png"
FAVICON_PATH = r"N:\_USER_GLOBAL\PETREL\Prizm\wf\Vienna\images\OMV_brandLogoPack\OMV_brandLogoPack\OMV_logo_Blue_Small.png"

valid_logo = LOGO_PATH if os.path.exists(LOGO_PATH) else None
valid_favicon = FAVICON_PATH if os.path.exists(FAVICON_PATH) else None

# ==============================================================================
#  MATH & ATTRIBUTE FUNCTIONS (GST & Curvature)
# ==============================================================================

def compute_gst_dip(data_3d, sigma=1.0, dt=1.0, dx=1.0, dy=1.0):
    g_il, g_xl, g_t = np.gradient(data_3d) 
    g_il /= dx; g_xl /= dy; g_t /= dt

    j_xx = gaussian_filter(g_il * g_il, sigma)
    j_xy = gaussian_filter(g_il * g_xl, sigma)
    j_xz = gaussian_filter(g_il * g_t, sigma)
    j_yy = gaussian_filter(g_xl * g_xl, sigma)
    j_yz = gaussian_filter(g_xl * g_t, sigma)
    j_zz = gaussian_filter(g_t * g_t, sigma)
    
    denom = j_zz.copy()
    denom[np.abs(denom) < 1e-6] = 1e-6
    
    p = -j_xz / denom 
    q = -j_yz / denom 
    return p, q

def compute_curvature(p, q, dx=1.0, dy=1.0):
    grad_p = np.gradient(p)
    a = grad_p[0] / dx
    b = grad_p[1] / dy 
    
    grad_q = np.gradient(q)
    c = grad_q[1] / dy 
    
    d = p
    e = q
    
    denom_base = 1 + d**2 + e**2
    denom_1_5 = np.power(denom_base, 1.5)
    denom_2_0 = np.power(denom_base, 2.0)
    
    K_mean = (a*(1+e**2) + c*(1+d**2) - 2*b*d*e) / (2 * denom_1_5)
    K_gauss = (a*c - b**2) / denom_2_0
    
    discriminant = K_mean**2 - K_gauss
    discriminant[discriminant < 0] = 0 
    root_disc = np.sqrt(discriminant)
    
    k1 = K_mean + root_disc 
    k2 = K_mean - root_disc 
    
    curvedness = np.sqrt(k1**2 + k2**2)
    
    numerator = k2 + k1
    denominator = k2 - k1
    
    with np.errstate(divide='ignore', invalid='ignore'):
        shape_index = -(2.0/np.pi) * np.arctan(numerator / denominator)
    
    mask_planar = curvedness < 1e-6
    shape_index[mask_planar] = 0.0 
    shape_index = np.clip(shape_index, -1.0, 1.0)
    
    return k1, k2, curvedness, shape_index

def generate_2d_colormap(shape_index, curvedness, c_max=1):
    H = 0.33 * (1.0 - shape_index)
    H = np.clip(H, 0.0, 0.66) 
    S = curvedness / (c_max + 1e-9)
    S = np.clip(S, 0.0, 1.0)
    V = np.ones_like(H)
    hsv = np.stack([H, S, V], axis=-1)
    return mcolors.hsv_to_rgb(hsv)

# ==============================================================================
#  SEISMIC APP CLASS
# ==============================================================================
class SeismicCurvatureApp:
    def __init__(self):
        # 1. Session Data
        self.session_data = self.load_session_data()
        self.project_name = self.session_data.get("project", "Unknown")
        self.cube_name = self.session_data.get("selected_cube_name", "None")
        self.cube_guid = self.session_data.get("selected_cube_guid", None)
        
        self.dims = (0, 0, 0) 
        self.amp_limit = 1000.0 
        self.fetch_metadata_and_stats() 
        
        # Caches
        self.cache_dip_p = None
        self.cache_dip_q = None
        self.cache_k1 = None
        self.cache_k2 = None
        self.cache_curvedness = None
        self.cache_shape_index = None

        # Sync Tap Stream for Point Selection
        self.tap_stream = streams.Tap(x=None, y=None)

        # --- SIDEBAR WIDGETS ---
        
        # 1. Info
        self.info_card = pn.Card(
            pn.Column(
                pn.pane.Markdown(f"**Project:** {self.project_name}"),
                pn.pane.Markdown(f"**Selected Volume:** {self.cube_name}"),
                sizing_mode='stretch_width'
            ),
            title="Session Info", collapsed=False, styles={'background': 'white'}, sizing_mode='stretch_width'
        )

        # 2. Controls
        self.orientation_selector = pn.widgets.RadioButtonGroup(
            name='Orientation', 
            options=['Inline', 'Crossline', 'Timeslice'], 
            button_type='success', 
            value='Timeslice'
        )
        self.slice_slider = pn.widgets.IntSlider(name='Slice Index', start=0, end=100, value=50)
        
        self.opacity_slider = pn.widgets.FloatSlider(name='Overlay Opacity', start=0.0, end=1.0, step=0.05, value=0.6)

        self.control_card = pn.Card(
            pn.Column(
                pn.pane.Markdown("**Orientation:**"),
                self.orientation_selector,
                pn.pane.Markdown("**Position:**"),
                self.slice_slider,
                pn.pane.Markdown("**Visualization:**"),
                self.opacity_slider,
                sizing_mode='stretch_width'
            ),
            title="Slice Controls", collapsed=False, styles={'background': 'white'}, sizing_mode='stretch_width'
        )

        # 3. Calculation & Attributes
        self.sigma_input = pn.widgets.FloatSlider(name='GST Window (Sigma)', start=0.5, end=3.0, step=0.5, value=1.0)
        self.calc_dip_btn = pn.widgets.Button(name='1. Generate GST (Dip)', button_type='primary')
        self.attr_selector = pn.widgets.Select(
            name='View Attribute', 
            options=['Dip Inline', 'Dip Crossline', 'K1 (Most Pos)', 'K2 (Most Neg)', 'Curvedness', 'Shape Index'],
            value='Dip Inline'
        )
        self.calc_curv_btn = pn.widgets.Button(name='2. Calculate Curvature', button_type='success', disabled=True)
        self.export_attr_btn = pn.widgets.Button(name='Export Selected Attribute', button_type='warning')
        
        self.gst_card = pn.Card(
            pn.Column(
                self.sigma_input,
                self.calc_dip_btn,
                pn.layout.Divider(),
                self.calc_curv_btn,
                pn.layout.Divider(),
                self.attr_selector,
                self.export_attr_btn,
                sizing_mode='stretch_width'
            ),
            title="Attribute Calculation", collapsed=False, styles={'background': 'white'}, sizing_mode='stretch_width'
        )

        # 4. Shape Classification
        self.shape_toggle = pn.widgets.Toggle(name='Show Shape Classification (2D Map)', button_type='primary')
        self.c_threshold_slider = pn.widgets.FloatSlider(name='Curvedness Scale (Max)', start=0.0001, end=0.01, step=0.0001, value=0.001)
        self.shape_card = pn.Card(
            pn.Column(
                self.shape_toggle,
                pn.pane.Markdown("Adjust Color Intensity:"),
                self.c_threshold_slider,
                sizing_mode='stretch_width'
            ),
            title="Shape Classification", collapsed=False, styles={'background': 'white'}, sizing_mode='stretch_width'
        )

        # 5. Export
        self.export_name_input = pn.widgets.TextInput(name='Export Name', value=f"{self.cube_name}_Attr")
        self.progress_bar = pn.indicators.Progress(name='Progress', value=0, max=100, bar_color='success', height=20, visible=False)
        self.progress_text = pn.pane.Markdown("", styles={'font-size': '0.8em', 'color': '#666'})
        self.export_card = pn.Card(
            pn.Column(self.export_name_input, self.progress_bar, self.progress_text, sizing_mode='stretch_width'),
            title="Export Status", collapsed=False, styles={'background': 'white'}, sizing_mode='stretch_width'
        )

        # --- PLOT PANES ---
        self.plot_seismic = pn.pane.HoloViews(sizing_mode='stretch_both')
        self.plot_attribute = pn.pane.HoloViews(sizing_mode='stretch_both')
        self.plot_shape_class = pn.pane.HoloViews(sizing_mode='stretch_both')

        # --- INTERACTIONS ---
        self.orientation_selector.param.watch(self.update_slider_limits, 'value')
        self.slice_slider.param.watch(self.update_plots, 'value_throttled')
        self.opacity_slider.param.watch(self.update_overlays, 'value_throttled')
        
        self.calc_dip_btn.on_click(self.run_gst_preview)
        self.calc_curv_btn.on_click(self.run_curvature_preview)
        self.export_attr_btn.on_click(self.run_export_process)
        
        self.attr_selector.param.watch(self.update_mid_panel, 'value')
        self.shape_toggle.param.watch(self.update_right_panel, 'value')
        self.c_threshold_slider.param.watch(self.update_right_panel, 'value_throttled')

        # --- INIT ---
        self.update_slider_limits(None)
        self.update_plots(None)

    def load_session_data(self):
        data_file = os.environ.get("PWR_DATA_FILE")
        if data_file and os.path.exists(data_file):
            try:
                with open(data_file, "r") as f: return json.load(f)
            except Exception: pass
        return {}

    def fetch_metadata_and_stats(self):
        if not self.cube_guid: return
        try:
            ptp = PetrelConnection(allow_experimental=True)
            objs = ptp.get_petrelobjects_by_guids([self.cube_guid])
            if objs:
                cube = objs[0]
                self.dims = cube.extent 
                self.amp_limit = 2000.0
        except Exception as e: print(f"Meta fetch error: {e}")

    def update_slider_limits(self, event):
        mode = self.orientation_selector.value
        i_max, j_max, k_max = self.dims
        if mode == 'Inline':
            self.slice_slider.end = max(0, i_max - 1)
            self.slice_slider.value = i_max // 2
        elif mode == 'Crossline':
            self.slice_slider.end = max(0, j_max - 1)
            self.slice_slider.value = j_max // 2
        else:
            self.slice_slider.end = max(0, k_max - 1)
            self.slice_slider.value = k_max // 2
        self.update_plots(None)

    def fetch_data_slab(self):
        if not self.cube_guid: return None, None, None
        mode = self.orientation_selector.value
        idx = self.slice_slider.value
        pad = 2 
        ptp = PetrelConnection(allow_experimental=True)
        cube = ptp.get_petrelobjects_by_guids([self.cube_guid])[0]
        i_ext, j_ext, k_ext = cube.extent
        
        if mode == 'Inline':
            start = max(0, idx - pad)
            end = min(i_ext - 1, idx + pad)
            chunk = cube.chunk((start, end), (0, j_ext-1), (0, k_ext-1))
            return chunk.as_array(), idx - start, 0
        elif mode == 'Crossline':
            start = max(0, idx - pad)
            end = min(j_ext - 1, idx + pad)
            chunk = cube.chunk((0, i_ext-1), (start, end), (0, k_ext-1))
            return chunk.as_array(), idx - start, 1
        else:
            start = max(0, idx - pad)
            end = min(k_ext - 1, idx + pad)
            chunk = cube.chunk((0, i_ext-1), (0, j_ext-1), (start, end))
            return chunk.as_array(), idx - start, 2

    # --- HELPER: POINT SYNC ---
    def get_marker(self, x, y):
        if x is None or y is None: return hv.Points([]).opts(color='red')
        return hv.Points([(x, y)]).opts(color='red', size=8, line_color='white', line_width=1)

    # --- VISUAL PREPARATION ---
    def get_visual_slices(self):
        data_slab, rel_idx, axis = self.fetch_data_slab()
        if data_slab is None: return None, None, False
        
        mode = self.orientation_selector.value
        invert_y = (mode != 'Timeslice')

        if axis == 0: # Inline (Time on Y)
            slice_data = data_slab[rel_idx, :, :] # (XL, Time)
            xlabel, ylabel = "Crossline", "Time"
        elif axis == 1: # Crossline (Time on Y)
            slice_data = data_slab[:, rel_idx, :] # (IL, Time)
            xlabel, ylabel = "Inline", "Time"
        else: # Timeslice
            slice_data = data_slab[:, :, rel_idx] # (IL, XL)
            xlabel, ylabel = "Inline", "Crossline"
            
        vis_data = slice_data.T
        
        # Explicitly Flip Data vertically if it's a vertical section
        # This fixes the visual orientation issue where deep data appears at top
        if invert_y:
            vis_data = np.flipud(vis_data)
            
        return vis_data, (xlabel, ylabel), invert_y

    # --- PREVIEW CALCULATIONS ---
    def run_gst_preview(self, event):
        self.calc_dip_btn.name = "Calculating..."
        try:
            data_slab, rel_idx, axis = self.fetch_data_slab()
            p, q = compute_gst_dip(data_slab, sigma=self.sigma_input.value)
            
            if axis == 0:
                self.cache_dip_p = p[rel_idx, :, :]
                self.cache_dip_q = q[rel_idx, :, :]
            elif axis == 1:
                self.cache_dip_p = p[:, rel_idx, :]
                self.cache_dip_q = q[:, rel_idx, :]
            else:
                self.cache_dip_p = p[:, :, rel_idx]
                self.cache_dip_q = q[:, :, rel_idx]
            
            self.calc_curv_btn.disabled = False
            self.attr_selector.value = 'Dip Inline'
            self.update_mid_panel(None)
        except Exception as e:
            self.progress_text.object = f"Error: {e}"
        finally:
            self.calc_dip_btn.name = "1. Generate GST (Dip)"

    def run_curvature_preview(self, event):
        if self.cache_dip_p is None: return
        self.calc_curv_btn.name = "Calculating..."
        try:
            p, q = self.cache_dip_p, self.cache_dip_q
            k1, k2, C, s = compute_curvature(p, q)
            
            self.cache_k1 = k1
            self.cache_k2 = k2
            self.cache_curvedness = C
            self.cache_shape_index = s
            
            self.attr_selector.value = 'Shape Index'
            self.update_mid_panel(None)
            self.shape_toggle.value = True
        except Exception as e:
            self.progress_text.object = f"Error: {e}"
        finally:
            self.calc_curv_btn.name = "2. Calculate Curvature"

    # --- PLOTTING UPDATES ---
    def update_plots(self, event):
        vis_data, labels, invert_y = self.get_visual_slices()
        if vis_data is None: return
        xlabel, ylabel = labels
        h, w = vis_data.shape
        bounds = (0, 0, w, h)
        
        # 1. Base Seismic Plot
        img = hv.Image(vis_data, bounds=bounds, kdims=[xlabel, ylabel], label="Amplitude").opts(
            cmap='gray', clim=(-self.amp_limit, self.amp_limit),
            invert_yaxis=invert_y, title=f"Seismic Input",
            responsive=True, aspect=None, tools=['hover', 'crosshair'], active_tools=['box_zoom']
        )
        
        # Link Tap Stream
        self.tap_stream.source = img
        marker_dmap = hv.DynamicMap(self.get_marker, streams=[self.tap_stream])
        
        self.plot_seismic.object = img * marker_dmap
        
        # Trigger updates for overlays
        self.update_mid_panel(None)
        self.update_right_panel(None)

    def update_overlays(self, event):
        # Optimized update for opacity slider drag
        self.update_mid_panel(None)
        self.update_right_panel(None)

    def update_mid_panel(self, event):
        vis_seis, labels, invert_y = self.get_visual_slices()
        if vis_seis is None: return
        h, w = vis_seis.shape
        bounds = (0, 0, w, h)
        xlabel, ylabel = labels
        
        # Base Layer
        base_img = hv.Image(vis_seis, bounds=bounds, kdims=[xlabel, ylabel]).opts(
            cmap='gray', clim=(-self.amp_limit, self.amp_limit), invert_yaxis=invert_y, 
            responsive=True, aspect=None
        )

        # Attribute Layer
        attr = self.attr_selector.value
        data = None
        cmap = 'viridis'; clim = (None, None)
        
        if attr == 'Dip Inline' and self.cache_dip_p is not None:
            data = self.cache_dip_p; cmap = 'RdBu'; clim = (-0.001, 0.001)
        elif attr == 'Dip Crossline' and self.cache_dip_q is not None:
            data = self.cache_dip_q; cmap = 'RdBu'; clim = (-0.001, 0.001)
        elif attr == 'K1 (Most Pos)' and self.cache_k1 is not None:
            data = self.cache_k1; cmap = 'seismic'
        elif attr == 'K2 (Most Neg)' and self.cache_k2 is not None:
            data = self.cache_k2; cmap = 'seismic'
        elif attr == 'Curvedness' and self.cache_curvedness is not None:
            data = self.cache_curvedness; cmap = 'inferno'
        elif attr == 'Shape Index' and self.cache_shape_index is not None:
            data = self.cache_shape_index; cmap = 'coolwarm'; clim = (-1, 1)

        overlay = base_img
        
        if data is not None:
            vis_attr = data.T
            if invert_y: vis_attr = np.flipud(vis_attr)
            
            attr_img = hv.Image(vis_attr, bounds=bounds, kdims=[xlabel, ylabel], label=attr).opts(
                cmap=cmap, clim=clim, alpha=self.opacity_slider.value,
                invert_yaxis=invert_y, colorbar=True, tools=['hover'],
                responsive=True, aspect=None
            )
            overlay = base_img * attr_img
        
        marker_dmap = hv.DynamicMap(self.get_marker, streams=[self.tap_stream])
        self.plot_attribute.object = (overlay * marker_dmap).opts(title=f"Attribute: {attr}")

    def update_right_panel(self, event):
        vis_seis, labels, invert_y = self.get_visual_slices()
        if vis_seis is None: return
        h, w = vis_seis.shape
        bounds = (0, 0, w, h)
        xlabel, ylabel = labels
        
        base_img = hv.Image(vis_seis, bounds=bounds, kdims=[xlabel, ylabel]).opts(
            cmap='gray', clim=(-self.amp_limit, self.amp_limit), invert_yaxis=invert_y, 
            responsive=True, aspect=None
        )
        
        overlay = base_img
        
        if self.shape_toggle.value and self.cache_shape_index is not None:
            s = self.cache_shape_index
            c = self.cache_curvedness
            c_max = self.c_threshold_slider.value
            
            rgb_data = generate_2d_colormap(s, c, c_max=c_max)
            
            # Flip RGB if necessary
            if invert_y:
                rgb_data = np.flipud(rgb_data) # Flip rows (Y axis)
                
            # Transpose (Y, X, 3) -> (X, Y, 3) for HoloViews
            rgb_data = np.transpose(rgb_data, (1, 0, 2))
            
            rgb_img = hv.RGB(rgb_data, bounds=bounds, kdims=[xlabel, ylabel]).opts(
                alpha=self.opacity_slider.value, invert_yaxis=invert_y,
                responsive=True, aspect=None, tools=['hover']
            )
            overlay = base_img * rgb_img
        
        marker_dmap = hv.DynamicMap(self.get_marker, streams=[self.tap_stream])
        self.plot_shape_class.object = (overlay * marker_dmap).opts(title="Classification")

    # --- EXPORT LOGIC ---
    def run_export_process(self, event):
        if not self.cube_guid: return
        new_name = self.export_name_input.value
        attr_mode = self.attr_selector.value
        self.export_attr_btn.disabled = True
        self.progress_bar.visible = True
        self.progress_bar.value = 0
        
        try:
            ptp = PetrelConnection(allow_experimental=True)
            src_cube = ptp.get_petrelobjects_by_guids([self.cube_guid])[0]
            target_cube = src_cube.clone(new_name, copy_values=False)
            i_max, j_max, k_max = src_cube.extent
            sigma = self.sigma_input.value
            
            chunk_size_z = 20
            range_k = range(0, k_max, chunk_size_z)
            total = len(range_k)
            
            for count, k_start in enumerate(range_k):
                k_end = min(k_start + chunk_size_z, k_max)
                pad = 2
                read_k_start = max(0, k_start - pad)
                read_k_end = min(k_max, k_end + pad)
                
                src_chunk = src_cube.chunk((0, i_max-1), (0, j_max-1), (read_k_start, read_k_end-1))
                data_block = src_chunk.as_array()
                
                p, q = compute_gst_dip(data_block, sigma=sigma)
                
                if attr_mode == 'Dip Inline': out = p
                elif attr_mode == 'Dip Crossline': out = q
                else:
                    k1, k2, C, s = compute_curvature(p, q)
                    if attr_mode == 'K1 (Most Pos)': out = k1
                    elif attr_mode == 'K2 (Most Neg)': out = k2
                    elif attr_mode == 'Curvedness': out = C
                    elif attr_mode == 'Shape Index': out = s
                    else: out = np.zeros_like(p)
                
                z_offset_start = k_start - read_k_start
                final = out[:, :, z_offset_start : z_offset_start + (k_end-k_start)]
                
                dst_chunk = target_cube.chunk((0, i_max-1), (0, j_max-1), (k_start, k_end-1))
                dst_chunk.set(final)
                
                self.progress_bar.value = int(((count+1)/total)*100)
                mem = psutil.virtual_memory().percent
                self.progress_text.object = f"Batch {count+1}/{total} | RAM: {mem}%"
            
            self.progress_text.object = f"✅ Export Complete: {new_name}"
        except Exception as e:
            self.progress_text.object = f"❌ Error: {e}"
        finally:
            self.export_attr_btn.disabled = False

    def get_template(self):
        panel_width = 'calc((100% - 20px) / 3)'
        style = {
            'background': '#ffffff', 'padding': '5px', 'border-radius': '5px',
            'box-shadow': '0px 1px 3px rgba(0,0,0,0.1)',
            'width': panel_width, 'height': '800px', 'overflow': 'hidden'
        }
        
        col1 = pn.Column(pn.pane.Markdown("### 1. Seismic Input"), self.plot_seismic, styles=style, sizing_mode='stretch_both')
        col2 = pn.Column(pn.pane.Markdown("### 2. Attribute Overlay"), self.plot_attribute, styles=style, sizing_mode='stretch_both')
        col3 = pn.Column(pn.pane.Markdown("### 3. Classification Overlay"), self.plot_shape_class, styles=style, sizing_mode='stretch_both')

        main_flex = pn.FlexBox(col1, col2, col3, flex_direction='row', flex_wrap='nowrap', justify_content='space-between', gap='10px')
        
        template = pn.template.FastListTemplate(
            title=APP_TITLE, logo=valid_logo, favicon=valid_favicon, accent_base_color=ACCENT_COLOR,
            header_background=ACCENT_COLOR,
            sidebar=[
                pn.pane.Markdown("### Dashboard"), self.info_card, pn.layout.Divider(), 
                self.control_card, pn.layout.Divider(), self.gst_card, pn.layout.Divider(), 
                self.shape_card, pn.layout.Divider(), self.export_card
            ],
            main=[pn.pane.Markdown(f"# {APP_TITLE}"), main_flex]
        )
        return template

app = SeismicCurvatureApp()
app.get_template().servable()
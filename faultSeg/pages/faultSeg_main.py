import os
import sys
import ctypes
import json
import time
import psutil
import math
import threading
import gc
import shutil
import traceback
from pathlib import Path
from collections import OrderedDict
from functools import partial

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# ==============================================================================
#  FEATURE B: ROBUST LOGGING SYSTEM
# ==============================================================================
TMP_DIR = Path(os.environ.get("TEMP", ".")) / "faultseg_tmp"
TMP_DIR.mkdir(exist_ok=True)
LOG_FILE = TMP_DIR / "faultseg_debug.log"

class DualLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')
    
    def write(self, message):
        try:
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        except: pass

    def flush(self):
        try:
            self.terminal.flush()
            self.log.flush()
        except: pass

logger = DualLogger(LOG_FILE)
sys.stdout = logger
sys.stderr = logger

print(f"--- LOG START: {time.ctime()} ---")
print(f"Log file location: {LOG_FILE}")

# ==============================================================================
#  CRITICAL DLL FIX
# ==============================================================================
def apply_dll_fix():
    try:
        current_venv = sys.prefix
        torch_lib_path = os.path.join(current_venv, 'Lib', 'site-packages', 'torch', 'lib')
        user_lib_path = os.path.expanduser(r"~\py_pkgs\torch\lib")
        paths_to_check = [torch_lib_path, user_lib_path]
        for lib_path in paths_to_check:
            if os.path.exists(lib_path):
                if hasattr(os, 'add_dll_directory'):
                    try: os.add_dll_directory(lib_path)
                    except: pass
                dlls = ['libiomp5md.dll', 'c10.dll', 'torch_python.dll']
                for dll in dlls:
                    dll_file = os.path.join(lib_path, dll)
                    if os.path.exists(dll_file):
                        try: ctypes.CDLL(dll_file)
                        except: pass
    except: pass

apply_dll_fix()

# ==============================================================================
#  IMPORTS
# ==============================================================================
import torch
import torch.nn as nn
import numpy as np
import panel as pn
import holoviews as hv
from holoviews import opts, streams
import matplotlib.colors as mcolors

from shared.ui.omv_theme import DARK_BLUE_OMV_COLOR, MAGENTA_OMV_COLOR, NEON_MAGENTA_OMV_COLOR

from cegalprizm.pythontool import PetrelConnection

# --- App UI Constants ---
APP_TITLE = "Fault Segmentation"
ACCENT_COLOR = "#052759"
BASE_MODEL_DIR = Path(r"N:\_USER_GLOBAL\PETREL\Prizm\wf\Vienna\faultSeg_Panel\models")
MODEL_DIR_WU = BASE_MODEL_DIR / "Wu_et_al_2019"
MODEL_DIR_DOU = BASE_MODEL_DIR / "FaultNet_Dou_2022"

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
#  PYTORCH U-NET ARCHITECTURE (Wu et al.)
# ==============================================================================
class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        def double_conv(in_c: int, out_c: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=3, padding=1, bias=True), nn.ReLU(inplace=True),
                nn.Conv3d(out_c, out_c, kernel_size=3, padding=1, bias=True), nn.ReLU(inplace=True)
            )
        self.conv1 = double_conv(in_channels, 16); self.pool1 = nn.MaxPool3d(2, 2)
        self.conv2 = double_conv(16, 32);          self.pool2 = nn.MaxPool3d(2, 2)
        self.conv3 = double_conv(32, 64);          self.pool3 = nn.MaxPool3d(2, 2)
        self.conv4 = double_conv(64, 128)
        self.up5 = nn.Upsample(scale_factor=2, mode='nearest'); self.conv5 = double_conv(128 + 64, 64)
        self.up6 = nn.Upsample(scale_factor=2, mode='nearest'); self.conv6 = double_conv(64 + 32, 32)
        self.up7 = nn.Upsample(scale_factor=2, mode='nearest'); self.conv7 = double_conv(32 + 16, 16)
        self.conv8 = nn.Conv3d(16, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        c1 = self.conv1(x); p1 = self.pool1(c1)
        c2 = self.conv2(p1); p2 = self.pool2(c2)
        c3 = self.conv3(p2); p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        u5 = self.up5(c4); c5 = self.conv5(torch.cat([u5, c3], dim=1))
        u6 = self.up6(c5); c6 = self.conv6(torch.cat([u6, c2], dim=1))
        u7 = self.up7(c6); c7 = self.conv7(torch.cat([u7, c1], dim=1))
        out = torch.sigmoid(self.conv8(c7))
        return out

# ==============================================================================
#  HELPER FUNCTIONS
# ==============================================================================
def get_ram_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def load_faultseg_wu(model_path, device):
    print(f"Loading Wu model from: {model_path}")
    path_str = str(model_path)
    if not os.path.exists(path_str):
        raise FileNotFoundError(f"Model file not found: {path_str}")

    model = UNet().to(device)
    try:
        state = torch.load(path_str, map_location=device)
    except Exception as e:
        print(f"Standard load failed ({e}), trying weights_only=False...")
        state = torch.load(path_str, map_location=device, weights_only=False)

    if "state_dict" in state: state = state["state_dict"]
    new_state = OrderedDict()
    for k, v in state.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        new_state[name] = v
    model.load_state_dict(new_state, strict=False)
    model.eval()
    return model

def load_faultnet_dou(model_path, device):
    print(f"Loading Dou model from: {model_path}")
    path_str = str(model_path)
    if not os.path.exists(path_str):
        raise FileNotFoundError(f"Model file not found: {path_str}")
    model = torch.jit.load(path_str).to(device)
    model.eval()
    return model

def gaussian_mask_3d(patch_size, overlap):
    ov = int(max(0, min(overlap, patch_size // 2)))
    w = np.ones(patch_size, dtype=np.float32)
    if ov > 0:
        sig = ov / 4.0
        inv2s2 = 0.5 / (sig * sig + 1e-8)
        ramp = np.arange(ov, dtype=np.float32)
        edge = np.exp(-((ov - ramp) ** 2) * inv2s2)
        w[:ov] *= edge
        w[-ov:] *= edge[::-1]
    return (w[:, None, None] * w[None, :, None] * w[None, None, :]).astype(np.float32)

def get_starts(dim_size, patch_size, stride):
    if dim_size <= patch_size: return [0]
    starts = list(range(0, dim_size - patch_size + 1, stride))
    if starts[-1] != (dim_size - patch_size):
        starts.append(dim_size - patch_size)
    return starts

# ==============================================================================
#  MAIN APP CLASS
# ==============================================================================
class FaultSegApp:
    def __init__(self):
        self.cleanup_temp_files(force=True, verbose=False)

        self.session_data = self.load_session_data()
        self.project_name = self.session_data.get("project", "Unknown")
        self.cube_name = self.session_data.get("seismic_name", "None")
        self.cube_guid = self.session_data.get("selected_cube_guid", None)
        
        self.dims = (0, 0, 0) 
        self.amp_limit = 1000.0 
        self.res_path = None
        self.masked_res_path = None 
        self.res_shape = None
        self.radio_group_stylesheets = get_radio_group_stylesheets()

        self.fetch_metadata_and_stats() 

        # --- A. SLICE CONTROLS ---
        self.radio_group = pn.widgets.RadioButtonGroup(
            name='Slice Type', options=['Inline', 'Crossline', 'Timeslice'],
            button_type='success', value='Inline', sizing_mode='stretch_width',
            stylesheets=self.radio_group_stylesheets,
        )
        init_end = max(0, self.dims[0]-1)
        self.slice_slider = pn.widgets.IntSlider(name='Index', start=0, end=init_end, value=init_end//2, sizing_mode='stretch_width')
        self.seismic_cmap = pn.widgets.Select(name='Seismic Colormap', options=['RdBu', 'gray', 'bwr', 'PuOr', 'viridis'], value='gray', sizing_mode='stretch_width')

        self.control_card = pn.Card(
            pn.Column(self.radio_group, self.slice_slider, self.seismic_cmap),
            title="Slice Controls", collapsed=True, sizing_mode='stretch_width', styles={'background': 'white'}
        )

        # --- B. FAULT SEG MODEL ---
        self.model_arch = pn.widgets.Select(name="Architecture", options=["FaultSeg (Wu et al., 2019)", "FaultNet (Dou et al., 2022)"], value="FaultSeg (Wu et al., 2019)", sizing_mode='stretch_width')
        self.model_weights = pn.widgets.Select(name="Weights", options=[], sizing_mode='stretch_width')
        self.patch_size = pn.widgets.IntInput(name="Patch Size", value=128, disabled=True, sizing_mode='stretch_width')
        self.overlap = pn.widgets.IntSlider(name="Overlap", start=0, end=32, value=12, step=4, sizing_mode='stretch_width')
        self.apply_sigmoid = pn.widgets.Checkbox(name="Apply Sigmoid (Logits -> Prob)", value=False, sizing_mode='stretch_width')
        self.swap_dims = pn.widgets.Checkbox(name="Swap Inline/Crossline Input", value=False, sizing_mode='stretch_width')
        self.gain = pn.widgets.FloatSlider(name="Signal Gain", start=0.1, end=5.0, step=0.1, value=1.0, sizing_mode='stretch_width')

        self.update_model_list(None) 

        self.run_btn = pn.widgets.Button(name='Run Segmentation', button_type='primary', sizing_mode='stretch_width')
        self.clear_btn = pn.widgets.Button(name='Clear Disk Space', button_type='danger', sizing_mode='stretch_width')
        self.progress_bar = pn.indicators.Progress(name='Progress', value=0, width=200, visible=False, sizing_mode='stretch_width')
        self.status_txt = pn.pane.Markdown("", styles={'font-size': '0.9em', 'color': '#333', 'font-weight': 'bold'})

        self.model_card = pn.Card(
            pn.Column(
                self.model_arch, self.model_weights, 
                pn.Row(self.patch_size, self.overlap),
                pn.layout.Divider(),
                pn.pane.Markdown("**Model Tweaks:**", styles={'font-weight': 'bold'}),
                self.apply_sigmoid, self.swap_dims, self.gain,
                pn.layout.Divider(),
                self.run_btn,
                self.clear_btn, 
                self.progress_bar,
                self.status_txt
            ),
            title="Fault Segmentation Model", collapsed=True, sizing_mode='stretch_width', styles={'background': 'white'}
        )

        # --- C. MASK SEG MODEL ---
        self.mask_slider = pn.widgets.FloatSlider(name='Mask Threshold', start=0.0, end=1.0, step=0.01, value=0.95, sizing_mode='stretch_width')
        self.mask_color = pn.widgets.ColorPicker(name='Mask Color', value='#32CD32', width=80, sizing_mode='fixed')
        self.mask_transparency = pn.widgets.Checkbox(name='Transparent Background', value=True, sizing_mode='stretch_width')
        
        self.mask_btn = pn.widgets.Button(name='Apply Mask to Segm Model', button_type='primary', disabled=True, sizing_mode='stretch_width')
        self.mask_progress = pn.indicators.Progress(name='Mask Progress', value=0, width=200, visible=False, sizing_mode='stretch_width')
        self.mask_status = pn.pane.Markdown("", styles={'font-size': '0.9em'})

        self.mask_card = pn.Card(
            pn.Column(
                self.mask_slider,
                self.mask_color,
                self.mask_transparency,
                self.mask_btn,
                self.mask_progress,
                self.mask_status
            ),
            title="Mask Segmentation Model", collapsed=True, sizing_mode='stretch_width', styles={'background': 'white'}
        )

        # --- D. VISUALIZATION ---
        self.view_mode = pn.widgets.RadioButtonGroup(
            name='View Mode', options=['Fault Segmentation', 'Masked'], 
            value='Fault Segmentation', button_type='default', sizing_mode='stretch_width',
            stylesheets=self.radio_group_stylesheets,
        )
        self.fault_cmap = pn.widgets.Select(name='Fault Colormap', options=['jet', 'viridis', 'inferno', 'magma', 'plasma', 'hot', 'gray'], value='jet', sizing_mode='stretch_width')
        self.opacity_slider = pn.widgets.FloatSlider(name='Fault Opacity', start=0.0, end=1.0, step=0.05, value=0.6, sizing_mode='stretch_width')
        
        # Update Button
        self.update_vis_btn = pn.widgets.Button(name='Update Visualization', button_type='primary', sizing_mode='stretch_width')

        self.vis_card = pn.Card(
            pn.Column(self.view_mode, self.fault_cmap, self.opacity_slider, self.update_vis_btn),
            title="Visualization", collapsed=True, sizing_mode='stretch_width', styles={'background': 'white'}
        )

        # --- E. EXPORT ---
        self.export_name = pn.widgets.TextInput(name='Output Name Base', value=f"FaultSeg", sizing_mode='stretch_width')
        self.export_orig_chk = pn.widgets.Checkbox(name="Export Original Model", value=True)
        self.export_mask_chk = pn.widgets.Checkbox(name="Export Masked Model", value=False)
        self.export_btn = pn.widgets.Button(name='Export to Petrel', button_type='warning', sizing_mode='stretch_width')
        self.export_progress = pn.indicators.Progress(name='Export Progress', value=0, width=200, visible=False, sizing_mode='stretch_width')
        self.export_status = pn.pane.Markdown("Ready", styles={'font-size': '0.9em', 'color': '#666'})

        self.export_card = pn.Card(
            pn.Column(
                self.export_name, 
                self.export_orig_chk, 
                self.export_mask_chk, 
                self.export_btn, 
                self.export_progress,
                self.export_status
            ),
            title="Export", collapsed=True, sizing_mode='stretch_width', styles={'background': 'white'}
        )

        # --- LAYOUT SETUP ---
        self.tap_stream = streams.Tap(x=None, y=None)
        self.marker_dmap = hv.DynamicMap(self.get_marker, streams=[self.tap_stream])
        
        min_h = 1100
        pane_opts = dict(sizing_mode='stretch_both', min_height=min_h)
        
        self.pane_left_1 = pn.pane.HoloViews(object=None, **pane_opts)
        self.pane_right_base = pn.pane.HoloViews(object=None, **pane_opts)
        self.pane_right_overlay = pn.pane.HoloViews(object=None, **pane_opts)
        
        self.col_left = pn.Column(self.pane_left_1, sizing_mode='stretch_both', min_height=min_h)
        self.col_right_base = pn.Column(self.pane_right_base, sizing_mode='stretch_both', min_height=min_h)
        self.col_right_overlay = pn.Column(self.pane_right_overlay, sizing_mode='stretch_both', min_height=min_h)
        
        self.swipe_right = pn.Swipe(
            self.col_right_base, self.col_right_overlay, 
            slider_width=5, slider_color='black', sizing_mode='stretch_both'
        )

        self.left_plot_pane = self.col_left
        self.right_plot_pane = pn.Column(self.swipe_right, sizing_mode='stretch_both', min_height=min_h)

        # Interactions
        self.model_arch.param.watch(self.update_model_list, 'value')
        self.radio_group.param.watch(self.update_slider_limits, 'value')
        self.slice_slider.param.watch(self.update_plots, 'value')
        self.seismic_cmap.param.watch(self.update_plots, 'value')
        
        self.view_mode.param.watch(self.update_plots, 'value')
        
        # We REMOVED watchers for colormap/opacity/mask_params to force button usage
        
        self.run_btn.on_click(self.run_inference)
        self.clear_btn.on_click(lambda e: self.cleanup_temp_files(force=True, verbose=True))
        self.mask_btn.on_click(self.run_masking)
        self.update_vis_btn.on_click(self.update_plots) 
        self.export_btn.on_click(self.run_export)

        self.update_plots(None)

    # --- METHODS ---
    def cleanup_temp_files(self, force=False, verbose=False):
        self.res_path = None
        self.masked_res_path = None
        gc.collect() 
        count = 0
        deleted_size = 0
        try:
            for f in TMP_DIR.glob("fault_*.dat"):
                try:
                    size = f.stat().st_size
                    os.remove(f)
                    count += 1
                    deleted_size += size
                except Exception as e: pass
            
            if verbose or force:
                if count > 0:
                    msg = f"Cleaned {count} files ({deleted_size / (1024**3):.2f} GB)"
                else:
                    msg = "Disk is already clean."
                print(msg)
                self.status_txt.object = msg
        except: pass

    def update_model_list(self, event):
        arch = self.model_arch.value
        if "FaultSeg" in arch:
            d = MODEL_DIR_WU
            ext = "*.pth"
            self.apply_sigmoid.value = False 
            self.swap_dims.value = False
        else:
            d = MODEL_DIR_DOU
            ext = "*.pt"
            self.apply_sigmoid.value = True 
            self.swap_dims.value = True     
        files = sorted([f.name for f in d.glob(ext)]) if d.exists() else ["No models found"]
        self.model_weights.options = files
        self.model_weights.value = files[0] if files else None

    def load_session_data(self):
        data_file = os.environ.get("PWR_DATA_FILE")
        if data_file and os.path.exists(data_file):
            try:
                with open(data_file, "r") as f:
                    return json.load(f)
            except:
                pass
        return {}

    def fetch_metadata_and_stats(self):
        if not self.cube_guid: return
        try:
            ptp = PetrelConnection(allow_experimental=True)
            objs = ptp.get_petrelobjects_by_guids([self.cube_guid])
            if objs:
                cube = objs[0]
                self.dims = cube.extent 
                idx = self.dims.i // 2
                chunk = cube.chunk((idx, idx), (0, self.dims.j-1), (0, self.dims.k-1)).as_array()
                self.amp_limit = float(np.percentile(np.abs(chunk[~np.isnan(chunk)]), 98))
        except Exception as e: 
            print(f"Meta fetch error: {e}")
            self.dims = type('obj', (object,), {'i':10, 'j':10, 'k':10})
            self.amp_limit = 1000

    def update_slider_limits(self, event):
        mode = event.new
        try:
            idx = 0 if mode == 'Inline' else (1 if mode == 'Crossline' else 2)
            if hasattr(self.dims, 'extent'):
                dim_max = self.dims.extent[idx]
            elif hasattr(self.dims, 'i'): 
                dim_max = [self.dims.i, self.dims.j, self.dims.k][idx]
            else:
                dim_max = 10
            self.slice_slider.end = max(0, dim_max - 1)
            self.slice_slider.value = dim_max // 2
        except: pass
        self.update_plots(None)

    def enable_mask_btn(self, event):
        if self.res_path and os.path.exists(self.res_path):
            self.mask_btn.disabled = False

    def get_marker(self, x, y):
        opts_pts = dict(color='red', size=8, line_color='white', line_width=1)
        if x is None or y is None: return hv.Points([]).opts(**opts_pts)
        return hv.Points([(x, y)]).opts(**opts_pts)

    def get_slice_data(self, mode, idx):
        try:
            ptp = PetrelConnection(allow_experimental=True)
            cube = ptp.get_petrelobjects_by_guids([self.cube_guid])[0]
            iR, jR, kR = cube.extent
            if mode == 'Inline':
                data = cube.chunk((idx, idx), (0, jR-1), (0, kR-1)).as_array()[0,:,:].T
                xlabel, ylabel = "Crossline", "Time/Depth"
            elif mode == 'Crossline':
                data = cube.chunk((0, iR-1), (idx, idx), (0, kR-1)).as_array()[:,0,:].T
                xlabel, ylabel = "Inline", "Time/Depth"
            else: # Timeslice
                data = cube.chunk((0, iR-1), (0, jR-1), (idx, idx)).as_array()[:,:,0].T
                xlabel, ylabel = "Inline", "Crossline"
            return data, xlabel, ylabel
        except Exception as e:
            print(f"Petrel Data Fetch Error: {e}")
            return np.zeros((10, 10)), "X", "Y"

    # --- LOGIC: INFERENCE ---
    def run_inference(self, event):
        if not self.cube_guid: return
        self.status_txt.object = "Cleaning disk space..."
        self.cleanup_temp_files(force=False)
        self.run_btn.disabled = True
        self.mask_btn.disabled = True
        self.progress_bar.visible = True
        self.progress_bar.value = 0
        self.status_txt.object = "Initializing Background Worker..."
        t = threading.Thread(target=self._inference_worker)
        t.start()

    def _inference_worker(self):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dev_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else "CPU"
            
            print(f"\n### RUNNING ON: {dev_name} ###\n", flush=True)
            self.status_txt.object = f"Running on: {dev_name}"

            arch = self.model_arch.value
            weight_file = self.model_weights.value
            use_sigmoid = self.apply_sigmoid.value
            do_swap = self.swap_dims.value
            gain_val = self.gain.value
            
            if not weight_file: raise ValueError("No model weights selected/found!")

            # ---------------------------------------------------------
            # FAULTNET Integration Logic
            # ---------------------------------------------------------
            is_faultnet = "FaultNet" in arch
            
            # 1. Add FaultNet folder to path to allow imports from utils.py if needed
            if is_faultnet:
                if str(MODEL_DIR_DOU) not in sys.path:
                    sys.path.append(str(MODEL_DIR_DOU))
                
                # Attempt to import normalization from utils.py
                try:
                    from utils import normalization
                    print("Successfully imported normalization from utils.py")
                except ImportError:
                    print("Could not import utils.py, using local min-max fallback.")
                    def normalization(data):
                        _min, _max = data.min(), data.max()
                        if _max > _min: return (data - _min) / (_max - _min)
                        return data * 0

            # 2. Load Model
            if not is_faultnet:
                model_path = MODEL_DIR_WU / weight_file
                model = load_faultseg_wu(model_path, device)
            else:
                model_path = MODEL_DIR_DOU / weight_file
                model = load_faultnet_dou(model_path, device)
            
            ptp = PetrelConnection(allow_experimental=True)
            cube = ptp.get_petrelobjects_by_guids([self.cube_guid])[0]
            iR, jR, kR = cube.extent
            
            run_id = int(time.time())
            res_path = TMP_DIR / f"fault_result_{run_id}.dat"
            wgt_path = TMP_DIR / f"fault_weight_{run_id}.dat"
            
            result_map = np.memmap(res_path, dtype='float32', mode='w+', shape=(iR, jR, kR))
            weight_map = np.memmap(wgt_path, dtype='float32', mode='w+', shape=(iR, jR, kR))
            result_map[:] = 0; weight_map[:] = 0
            
            patch = self.patch_size.value
            overlap = self.overlap.value
            stride = patch - overlap
            
            starts_i = get_starts(iR, patch, stride)
            starts_j = get_starts(jR, patch, stride)
            starts_k = get_starts(kR, patch, stride)
            total_chunks = len(starts_i) * len(starts_j) * len(starts_k)
            g_mask = gaussian_mask_3d(patch, overlap) 
            
            count = 0
            print(f"--- Starting Inference: {total_chunks} Chunks ---", flush=True)
            
            for i in starts_i:
                for j in starts_j:
                    for k in starts_k:
                        t0_chunk = time.time()
                        i_end, j_end, k_end = min(i+patch, iR), min(j+patch, jR), min(k+patch, kR)
                        ri, rj, rk = i_end-i, j_end-j, k_end-k
                        chunk_data = cube.chunk((i, i_end-1), (j, j_end-1), (k, k_end-1)).as_array()
                        inp = np.zeros((patch, patch, patch), dtype=np.float32)
                        inp[:ri, :rj, :rk] = chunk_data
                        
                        # --- NORMALIZATION SWITCH ---
                        if is_faultnet:
                            # FaultNet uses Min-Max (0-1) as per utils.py
                            inp = normalization(inp)
                        else:
                            # FaultSeg uses Z-Score
                            mu, std = np.mean(inp), np.std(inp)
                            if std > 0: inp = (inp - mu) / std
                        
                        if do_swap: inp_tensor = torch.from_numpy(np.transpose(inp, (2, 0, 1))).unsqueeze(0).unsqueeze(0).to(device)
                        else: inp_tensor = torch.from_numpy(np.transpose(inp, (2, 1, 0))).unsqueeze(0).unsqueeze(0).to(device)
                        
                        with torch.no_grad(): out_tensor = model(inp_tensor)
                        if use_sigmoid: out_tensor = torch.sigmoid(out_tensor)

                        out_np = out_tensor.squeeze().cpu().numpy()
                        if do_swap: out_np = np.transpose(out_np, (1, 2, 0))
                        else: out_np = np.transpose(out_np, (2, 1, 0))
                        
                        result_map[i:i_end, j:j_end, k:k_end] += out_np[:ri, :rj, :rk] * g_mask[:ri, :rj, :rk]
                        weight_map[i:i_end, j:j_end, k:k_end] += g_mask[:ri, :rj, :rk]
                        
                        count += 1
                        t1_chunk = time.time()
                        chunk_time = t1_chunk - t0_chunk
                        print(f"Chunk {count}/{total_chunks} | Size: {patch}^3 | Time: {chunk_time:.2f}s", flush=True)
                        self.progress_bar.value = int((count / total_chunks) * 100)
                        self.status_txt.object = f"Processing Chunk {count}/{total_chunks} ({chunk_time:.2f}s)"

            valid = weight_map > 0
            result_map[valid] /= weight_map[valid]
            if gain_val != 1.0:
                result_map *= gain_val
                np.clip(result_map, 0, 1, out=result_map)

            result_map.flush()
            weight_map.flush()
            
            self.res_path = str(res_path)
            self.res_shape = (iR, jR, kR)
            self.masked_res_path = None
            
            self.status_txt.object = "Processing Complete!"
            print("\n### Inference Job Finished ###", flush=True)
            
            def ui_update():
                self.mask_btn.disabled = False
                self.view_mode.value = 'Fault Segmentation'
                self.update_plots(None)
            if pn.state.curdoc: pn.state.curdoc.add_next_tick_callback(ui_update)
            else: ui_update()
            
        except Exception as e:
            self.status_txt.object = f"Error: {str(e)}"
            print(traceback.format_exc(), flush=True)
        finally:
            self.run_btn.disabled = False
            self.progress_bar.visible = False

    # --- LOGIC: MASKING ---
    def run_masking(self, event):
        if not self.res_path: return
        self.mask_btn.disabled = True
        self.mask_progress.visible = True
        self.mask_progress.value = 10
        self.mask_status.object = "Calculating mask..."
        try:
            threshold = self.mask_slider.value
            run_id = int(time.time())
            masked_path = TMP_DIR / f"fault_masked_{run_id}.dat"
            
            data = np.memmap(self.res_path, dtype='float32', mode='r', shape=self.res_shape)
            self.mask_progress.value = 40
            masked_data = np.memmap(masked_path, dtype='float32', mode='w+', shape=self.res_shape)
            self.mask_progress.value = 60
            masked_data[:] = np.where(data > threshold, 1.0, 0.0)
            
            masked_data.flush()
            del data; del masked_data
            
            self.masked_res_path = str(masked_path)
            self.mask_progress.value = 100
            self.mask_status.object = f"Mask applied (Thresh: {threshold})"
            self.view_mode.value = 'Masked'
            self.update_plots(None)
        except Exception as e:
            self.mask_status.object = f"Error: {e}"
        finally:
            self.mask_btn.disabled = False
            self.mask_progress.visible = False

    # --- LOGIC: PLOTS (CORRECTED & ALIGNED) ---
    def update_plots(self, event):
        if not self.cube_guid: return
        mode = self.radio_group.value
        idx = self.slice_slider.value
        seis_cmap = self.seismic_cmap.value
        fault_cmap_name = self.fault_cmap.value
        view_mode = self.view_mode.value
        
        raw_data, xl, yl = self.get_slice_data(mode, idx)
        fault_data = None
        
        if view_mode == 'Masked':
            path_to_load = self.masked_res_path if self.masked_res_path else self.res_path
        else:
            path_to_load = self.res_path

        if path_to_load is not None:
            try:
                mem = np.memmap(path_to_load, dtype='float32', mode='r', shape=self.res_shape)
                if mode == 'Inline': raw = mem[idx, :, :].T
                elif mode == 'Crossline': raw = mem[:, idx, :].T
                else: raw = mem[:, :, idx].T
                fault_data = raw
                del mem
            except: pass

        invert_y = (mode != 'Timeslice')
        vis_seismic = raw_data
        vis_fault = fault_data
        if invert_y:
            vis_seismic = np.flipud(vis_seismic)
            if vis_fault is not None: vis_fault = np.flipud(vis_fault)
        
        h, w = vis_seismic.shape
        bounds = (0, 0, w, h)
        # REMOVED sizing_mode from options.
        common_opts = dict(cmap=seis_cmap, clabel='Amplitude', toolbar='above', 
                           active_tools=['box_zoom'], invert_yaxis=invert_y,
                           aspect='equal')
        
        img_seis_left = hv.Image(vis_seismic, bounds=bounds, kdims=[xl, yl], label="Seismic").opts(title=f"{mode} {idx} (Seismic)", clim=(-self.amp_limit, self.amp_limit), **common_opts)
        img_seis_right = hv.Image(vis_seismic, bounds=bounds, kdims=[xl, yl], label="Seismic").opts(clim=(-self.amp_limit, self.amp_limit), colorbar=False, **common_opts)
        
        if vis_fault is not None:
            alpha = self.opacity_slider.value
            
            # --- VIEW LOGIC ---
            if view_mode == 'Masked':
                # Convert 0s to NaNs if transparency is requested to GUARANTEE transparency
                if self.mask_transparency.value:
                    vis_fault = np.where(vis_fault < 0.5, np.nan, vis_fault)
                    
                hex_color = self.mask_color.value
                try:
                    c_rgb = mcolors.to_rgba(hex_color)
                    # If using NaNs, we only need the color for "1"
                    # But if not using NaNs (unchecked), we need black for "0"
                    bg_color = (0, 0, 0, 1) # Black for 0 if transparency OFF
                    
                    if self.mask_transparency.value:
                        # If transparent, NaN handles the background. We just need a colormap for the foreground.
                        # We create a map where the low end doesn't matter (it's NaN) and high is color
                        cmap_cols = [c_rgb, c_rgb] 
                    else:
                        cmap_cols = [bg_color, c_rgb]
                        
                    final_cmap = mcolors.ListedColormap(cmap_cols)
                    clim_vals = (0, 1)
                except: final_cmap = 'jet'; clim_vals = (0, 1)
            else:
                # ORIGINAL VIEW (Continuous)
                final_cmap = fault_cmap_name
                dmin, dmax = np.nanmin(vis_fault), np.nanmax(vis_fault)
                if dmax > dmin:
                    clim_vals = (dmin, dmax)
                else:
                    clim_vals = (0, 1)

            # Ensure colorbar=False so the plot size matches the seismic plot exactly
            img_fault = hv.Image(vis_fault, bounds=bounds, kdims=[xl, yl]).opts(
                cmap=final_cmap, alpha=alpha, clim=clim_vals, 
                colorbar=False, 
                invert_yaxis=invert_y, aspect='equal'
            )
            img_overlay = (img_seis_right * img_fault)
            title_right = f"{mode} {idx} ({view_mode})"
        else:
            img_overlay = img_seis_right
            title_right = "Run Model to see Overlay"

        self.tap_stream.source = img_seis_left
        view_left = (img_seis_left * self.marker_dmap).opts(responsive=True)
        view_right_base = (img_seis_right * self.marker_dmap).opts(title=title_right, responsive=True)
        view_right_overlay = (img_overlay * self.marker_dmap).opts(title=title_right, responsive=True)

        self.pane_left_1.object = view_left
        self.pane_right_base.object = view_right_base
        self.pane_right_overlay.object = view_right_overlay

    # --- LOGIC: EXPORT ---
    def run_export(self, event):
        if not self.res_path:
            self.export_status.object = "No result to export!"
            return
        
        # FIX: Capture values in main thread
        export_orig_val = self.export_orig_chk.value
        export_mask_val = self.export_mask_chk.value
        
        if not export_orig_val and not export_mask_val:
            self.export_status.object = "Please check at least one Export option."
            return

        self.export_btn.disabled = True
        self.export_progress.visible = True
        self.export_progress.value = 0
        self.export_status.object = "Initializing Export..."
        
        t = threading.Thread(target=self._export_worker, args=(export_orig_val, export_mask_val))
        t.start()

    def _export_worker(self, export_orig, export_mask):
        try:
            ptp = PetrelConnection(allow_experimental=True)
            src_cube = ptp.get_petrelobjects_by_guids([self.cube_guid])[0]
            model_full = self.model_weights.value
            model_clean = os.path.splitext(model_full)[0] 
            base_input_name = self.export_name.value
            
            tasks = []
            if export_orig: tasks.append((f"{base_input_name}_{model_clean}", self.res_path))
            if export_mask:
                if self.masked_res_path: tasks.append((f"masked_{base_input_name}_{model_clean}", self.masked_res_path))
                else: print("Warning: Export Masked selected but no mask generated. Skipping.")

            total_ops = len(tasks)
            current_op = 0
            
            for out_name, file_path in tasks:
                current_op += 1
                self.export_status.object = f"Exporting {current_op}/{total_ops}: {out_name}"
                target = src_cube.clone(out_name, copy_values=False)
                mem = np.memmap(file_path, dtype='float32', mode='r', shape=self.res_shape)
                iR, jR, kR = self.res_shape
                chunk_size = 10
                
                print(f"\n--- Exporting {out_name} ---", flush=True)
                for i in range(0, iR, chunk_size):
                    end = min(i + chunk_size, iR)
                    block = mem[i:end, :, :]
                    with target.chunk((i, end-1), (0, jR-1), (0, kR-1)).values() as v: v[:] = block
                    
                    progress_pct = int(((i + chunk_size) / iR) * 100)
                    self.export_progress.value = min(progress_pct, 100)
                    print(f"Exported Inlines {i} to {end}", flush=True)
                
                del mem
            
            self.export_status.object = f"Successfully exported {len(tasks)} volume(s)."
            self.export_progress.value = 100
            
        except Exception as e:
            self.export_status.object = f"Export Error: {e}"
            print(traceback.format_exc(), flush=True)
        finally:
            self.export_btn.disabled = False

    def get_template(self):
        style = {'background': '#ffffff', 'padding': '10px', 'border-radius': '8px', 'box-shadow': '0 2px 5px rgba(0,0,0,0.1)', 'width': '49.5%', 'height': 'calc(100vh - 220px)', 'overflow': 'hidden'}
        main = pn.FlexBox(
            pn.Column(pn.pane.Markdown("### Seismic Preview"), self.left_plot_pane, styles=style, sizing_mode='stretch_both'),
            pn.Column(pn.pane.Markdown("### Fault Model Overlay (Swipe)"), self.right_plot_pane, styles=style, sizing_mode='stretch_both'),
            justify_content='space-between', align_items='start', gap='5px', sizing_mode='stretch_width'
        )
        return pn.template.FastListTemplate(
            title=APP_TITLE, logo=valid_logo, favicon=valid_favicon, 
            accent_base_color=ACCENT_COLOR, header_background=ACCENT_COLOR,
            sidebar=[self.control_card, pn.layout.Divider(), self.model_card, pn.layout.Divider(), self.mask_card, pn.layout.Divider(), self.vis_card, pn.layout.Divider(), self.export_card],
            main=[main]
        )

# --- Launch ---
app = FaultSegApp()
template = app.get_template()
template.servable()

## Working version (FaultSeg, and FaultNet) | Feb, 11th, 2026 | 16:20
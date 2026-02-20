import panel as pn
import pandas as pd
import hvplot.pandas 
import holoviews as hv
import os
import sys
import duckdb
import json
import numpy as np
import io
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import re 
import warnings
import time
from datetime import datetime
from pathlib import Path

# Suppress specific deprecation warnings
warnings.filterwarnings("ignore")

from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image as RLImage, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch, mm
from reportlab.lib.utils import ImageReader

# --- App UI Constants ---
APP_TITLE = "CheckShot QC"
SELECTED_COLOR = "#1cff5a" # Bright Green
DOCUMENTATION_URL = "https://example.com/docs"

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from shared.ui.omv_theme import (  # noqa: E402
    BLUE_OMV_COLOR,
    DARK_BLUE_OMV_COLOR,
    NEON_OMV_COLOR,
    docs_button_html,
    get_content_text_color,
    get_extension_raw_css,
    get_main_outer_background,
    is_dark_mode_from_state,
)

is_dark_mode = is_dark_mode_from_state()
ACCENT_COLOR = BLUE_OMV_COLOR
DEFAULT_EXPORT_PATH = str(Path.home() / "Downloads")
section_header_background = NEON_OMV_COLOR
section_body_background = DARK_BLUE_OMV_COLOR if is_dark_mode else "white"
section_text_color = get_content_text_color(is_dark_mode)

# --- CSS Styling ---
css = f"""
.tabulator-header, .tabulator-header-contents, .tabulator-col, .tabulator-col-content {{
    background-color: {ACCENT_COLOR} !important;
    color: white !important;
    font-weight: bold;
}}
.tabulator-footer {{
    background-color: {ACCENT_COLOR} !important;
    color: white !important;
}}
.tabulator-page {{
    color: #333 !important;
}}
.tabulator-page.active {{
    font-weight: bold;
    background-color: #a6c9ff !important;
}}
.tabulator-row.tabulator-selected {{
    background-color: #a6c9ff !important;
}}

/* Custom Button Styling */
.bk-btn {{
    font-size: 11px !important;
    background-color: {ACCENT_COLOR} !important;
    border-color: {ACCENT_COLOR} !important;
    color: white !important;
}}
/* Active State: Bright Green with Black Text */
.bk-btn.bk-active {{
    background-color: {SELECTED_COLOR} !important;
    border-color: {SELECTED_COLOR} !important;
    color: black !important;
    font-weight: bold !important;
    box-shadow: inset 0 3px 5px rgba(0,0,0,0.125) !important;
}}
"""
pn.config.raw_css.append(css)

# --- Image Paths ---
ASSETS_DIR = ROOT_DIR / "images" / "OMV_brandLogoPack" / "OMV_brandLogoPack"
LOGO_PATH = ASSETS_DIR / "OMV_logo_Neon_Small.png"
FAVICON_PATH = ASSETS_DIR / "OMV_logo_Blue_Small.png"

if not LOGO_PATH.exists():
    network_logo = Path(r"N:\_USER_GLOBAL\PETREL\Prizm\wf\Vienna\images\OMV_brandLogoPack\OMV_brandLogoPack\OMV_logo_Neon_Small.png")
    LOGO_PATH = network_logo if network_logo.exists() else LOGO_PATH

if not FAVICON_PATH.exists():
    network_favicon = Path(r"N:\_USER_GLOBAL\PETREL\Prizm\wf\Vienna\images\OMV_brandLogoPack\OMV_brandLogoPack\OMV_logo_Blue_Small.png")
    FAVICON_PATH = network_favicon if network_favicon.exists() else FAVICON_PATH

valid_logo = str(LOGO_PATH) if LOGO_PATH.exists() else None
valid_favicon = str(FAVICON_PATH) if FAVICON_PATH.exists() else None

# --- Initialize Panel ---
pn.extension('tabulator', notifications=True, raw_css=get_extension_raw_css(is_dark_mode))

# ==============================================================================
#  CHECKSHOT APP CLASS
# ==============================================================================
class CheckShotApp:
    def _create_sidebar_card(self, *widgets, title: str, collapsed: bool = True) -> pn.Card:
        return pn.Card(
            *widgets,
            title=title,
            collapsed=collapsed,
            hide_header=False,
            sizing_mode='stretch_width',
            header_background=section_header_background,
            active_header_background=section_header_background,
            header_color=section_text_color,
            styles={'background': section_body_background, 'color': section_text_color},
            margin=(0, 0, 12, 0),
        )

    def __init__(self):
        # 1. Load Data
        self.df_headers, self.df_checkshots, self.df_surveys, self.df_dt, self.df_tops, self.project_name = self.load_data()
        
        # 1b. Enrich Headers
        self.enrich_headers()
        
        # 1c. Assign Colors
        self.well_colors = self.assign_well_colors()
        if not self.df_headers.empty:
            self.df_headers['Plot'] = True 
        
        # 2. Widgets: Selection
        available_wells = self.df_headers['WellName'].unique().tolist() if not self.df_headers.empty else []
        available_wells.sort()
        
        self.well_selector = pn.widgets.Select(
            name='Select Well', 
            options=available_wells,
            value=available_wells[0] if available_wells else None,
            visible=False 
        )
        
        # --- 3. IMPORT SECTION ---
        self.import_well_selector = pn.widgets.Select(
            options=available_wells,
            value=available_wells[0] if available_wells else None,
            sizing_mode='stretch_width'
        )
        self.import_skip_header = pn.widgets.Checkbox(name='Skip header', value=False)
        self.import_negate_depth = pn.widgets.Checkbox(name='Increase depth negative', value=False)
        
        self.import_depth_fmt = pn.widgets.RadioButtonGroup(
            name='Input Depth Format', 
            options=['MD from DF (ft)', 'MD from KB (ft)', 'MD from GL (ft)'],
            value='MD from KB (ft)', 
            sizing_mode='stretch_width' 
        )
        self.import_time_fmt = pn.widgets.RadioButtonGroup(
            name='Input Time Format', 
            options=['OWT (ms)', 'TWT (ms)'],
            value='OWT (ms)', 
            sizing_mode='stretch_width' 
        )
        
        self.file_input = pn.widgets.FileInput(accept='.csv,.txt,.dat', name='Select File', sizing_mode='stretch_width')
        self.import_btn = pn.widgets.Button(name='IMPORT', button_type='success', sizing_mode='stretch_width')
        
        # --- 4. EXPORT TO PETREL ---
        self.export_petrel_btn = pn.widgets.Button(name='EXPORT TO PETREL', button_type='danger', sizing_mode='stretch_width')
        self.export_petrel_status = pn.pane.Markdown("", styles={'font-size': '0.8em', 'color': '#666'})

        # --- 5. EXPORT CHECKSHOT FILES ---
        self.export_path_input = pn.widgets.TextInput(name='Folder Path', value=DEFAULT_EXPORT_PATH, placeholder='C:/Temp/', sizing_mode='stretch_width')
        self.export_filename_input = pn.widgets.TextInput(name='File Name', placeholder='Exported_Checkshot', sizing_mode='stretch_width')
        
        self.export_content_selector = pn.widgets.RadioButtonGroup(
            name='Export Content',
            options=['Short (Petrel Template)', 'Full Report (All Data)'],
            value='Short (Petrel Template)',
            button_type='primary',
            sizing_mode='stretch_width'
        )
        self.export_fmt_selector = pn.widgets.RadioButtonGroup(
            name='File Format', options=['.csv', '.txt'], value='.csv', button_type='primary', sizing_mode='stretch_width'
        )
        self.export_file_depth_ref = pn.widgets.RadioButtonGroup(
            name='Depth Reference', 
            options=['MD from DF', 'MD from KB', 'MD from GL'], 
            value='MD from DF', 
            sizing_mode='stretch_width' 
        )
        self.export_file_time_ref = pn.widgets.RadioButtonGroup(
            name='Time Reference', 
            options=['OWT', 'TWT'], 
            value='OWT', 
            sizing_mode='stretch_width' 
        )
        self.export_file_btn = pn.widgets.Button(name='Export Files', button_type='success', sizing_mode='stretch_width')
        self.export_content_selector.param.watch(self.toggle_export_settings, 'value')

        # --- 6. EXPORT SUMMARY ---
        self.export_sum_path_input = pn.widgets.TextInput(name='Folder Path', value=DEFAULT_EXPORT_PATH, placeholder='C:/Temp/', sizing_mode='stretch_width')
        self.export_sum_filename_input = pn.widgets.TextInput(name='File Name', placeholder='Summary_Wells', sizing_mode='stretch_width')
        self.export_sum_btn = pn.widgets.Button(name='Export Summary Wells', button_type='primary', sizing_mode='stretch_width')

        # --- 7. EXPORT PDF REPORT ---
        self.export_pdf_path_input = pn.widgets.TextInput(name='Folder Path', value=DEFAULT_EXPORT_PATH, placeholder='C:/Temp/', sizing_mode='stretch_width')
        self.export_pdf_filename_input = pn.widgets.TextInput(name='File Name', placeholder='Report_PDF', sizing_mode='stretch_width')
        
        self.export_pdf_x_selector = pn.widgets.Select(
            name='Plots to Show (X-Axis)', 
            options=['All (4-Panel)', 'OWT', 'TWT', 'Int. Vel.', 'Avg. Vel.'],
            value='All (4-Panel)', sizing_mode='stretch_width'
        )
        self.export_pdf_depth_ref = pn.widgets.RadioButtonGroup(
            name='Depth Reference (Y-Axis)', 
            options=['Depth from DF', 'Depth from KB', 'Depth from GL'], 
            value='Depth from DF', 
            sizing_mode='stretch_width' 
        )
        self.export_pdf_btn = pn.widgets.Button(name='Export PDF Report', button_type='primary', sizing_mode='stretch_width')

        # --- 8. EXPORT PLOT IMAGES ---
        self.export_imgs_path_input = pn.widgets.TextInput(name='Folder Path', value=DEFAULT_EXPORT_PATH, placeholder='C:/Temp/', sizing_mode='stretch_width')
        self.export_imgs_filename_input = pn.widgets.TextInput(name='File Name', placeholder='Plot_Image', sizing_mode='stretch_width')
        
        self.export_imgs_type_selector = pn.widgets.RadioButtonGroup(
            name='Image Plot Type', 
            options=['OWT', 'TWT', 'Int. Vel.', 'Avg. Vel.'],
            value='OWT',
            button_type='primary',
            sizing_mode='stretch_width'
        )
        self.export_imgs_depth_ref = pn.widgets.RadioButtonGroup(
            name='Depth Reference (Y-Axis)', 
            options=['Depth from DF', 'Depth from KB', 'Depth from GL'], 
            value='Depth from DF', 
            sizing_mode='stretch_width' 
        )
        self.export_imgs_dpi_selector = pn.widgets.RadioButtonGroup(
            name='DPI', options=[300, 600, 900], value=300, button_type='primary', sizing_mode='stretch_width'
        )
        self.export_imgs_fmt_selector = pn.widgets.Select(
            name='Format', options=['.png', '.jpg'], value='.png', sizing_mode='stretch_width'
        )
        self.export_imgs_btn = pn.widgets.Button(name='Export CheckShot Plot Images', button_type='primary', sizing_mode='stretch_width')

        # --- 9. WELL PARAMETERS ---
        self.inp_srd = pn.widgets.FloatInput(name="Seismic Reference Datum (SRD) [ft]", value=0.0, step=1.0, sizing_mode='stretch_width')
        self.inp_kb = pn.widgets.FloatInput(name="Kelly Bushing (KB) [ft]", value=0.0, step=1.0, sizing_mode='stretch_width')
        self.inp_df = pn.widgets.FloatInput(name="Drill Floor (DF) [ft]", value=0.0, step=1.0, sizing_mode='stretch_width')
        self.inp_gl = pn.widgets.FloatInput(name="Ground Level (GL) [ft]", value=0.0, step=1.0, sizing_mode='stretch_width')
        self.inp_se = pn.widgets.FloatInput(name="Source Elevation (SE) [ft]", value=0.0, step=1.0, sizing_mode='stretch_width')
        self.inp_so = pn.widgets.FloatInput(name="Source Offset (SO) [ft]", value=0.0, step=1.0, sizing_mode='stretch_width')
        self.inp_vc = pn.widgets.FloatInput(name="Replacement Velocity (Vc) [ft/s]", value=5000.0, step=100.0, sizing_mode='stretch_width')
        
        self.param_update_btn = pn.widgets.Button(name='Update', button_type='primary', sizing_mode='stretch_width')
        self.param_save_btn = pn.widgets.Button(name='Save (DB + Petrel)', button_type='success', sizing_mode='stretch_width')

        # --- 10. SIDEBAR CARDS ---
        self.import_card = self._create_sidebar_card(
            pn.Column(self.file_input, self.import_well_selector, pn.Row(self.import_skip_header, self.import_negate_depth), self.import_depth_fmt, self.import_time_fmt, pn.Spacer(height=5), self.import_btn, sizing_mode='stretch_width'),
            title="Import", collapsed=True
        )
        self.export_petrel_card = self._create_sidebar_card(
            pn.Column(self.export_petrel_btn, self.export_petrel_status, sizing_mode='stretch_width'),
            title="Export to Petrel", collapsed=True
        )
        self.export_file_card = self._create_sidebar_card(
            pn.Column(self.export_path_input, self.export_filename_input, pn.pane.Markdown("**Content:**"), self.export_content_selector, pn.pane.Markdown("**Format:**"), self.export_fmt_selector, pn.pane.Markdown("**Data Selection (Short Only):**"), self.export_file_depth_ref, self.export_file_time_ref, pn.Spacer(height=5), self.export_file_btn, sizing_mode='stretch_width'),
            title="Export CheckShot Files", collapsed=True
        )
        self.export_summary_card = self._create_sidebar_card(
            pn.Column(self.export_sum_path_input, self.export_sum_filename_input, pn.Spacer(height=5), self.export_sum_btn, sizing_mode='stretch_width'),
            title="Export Summary Wells", collapsed=True
        )
        self.export_pdf_card = self._create_sidebar_card(
            pn.Column(self.export_pdf_path_input, self.export_pdf_filename_input, pn.pane.Markdown("**Settings:**"), self.export_pdf_x_selector, self.export_pdf_depth_ref, pn.Spacer(height=5), self.export_pdf_btn, sizing_mode='stretch_width'),
            title="Export PDF Report", collapsed=True
        )
        self.export_imgs_card = self._create_sidebar_card(
            pn.Column(self.export_imgs_path_input, self.export_imgs_filename_input, pn.pane.Markdown("**Data:**"), self.export_imgs_type_selector, self.export_imgs_depth_ref, pn.pane.Markdown("**Settings:**"), self.export_imgs_dpi_selector, self.export_imgs_fmt_selector, pn.Spacer(height=10), self.export_imgs_btn, sizing_mode='stretch_width'),
            title="Export CheckShot Plot Images", collapsed=True
        )

        # 11. Tables & Widgets Logic
        self.all_wells_table = self.get_all_wells_table_widget()
        self.all_wells_table.param.watch(self.update_from_table_selection, 'selection')

        # 12. Plot Controls
        self.plot_mode_selector = pn.widgets.RadioButtonGroup(
            name='Plot Mode', options=['All Wells', 'Single Well'], value='All Wells',
            button_type='primary', sizing_mode='stretch_width'
        )
        
        self.plot_x_selector = pn.widgets.RadioButtonGroup(
            name='X-Axis', options=['OWT', 'TWT', 'Int. Vel.', 'Avg. Vel.', 'TVD', 'MD'],
            value='OWT', button_type='primary', sizing_mode='stretch_width'
        )
        
        self.plot_y_selector = pn.widgets.RadioButtonGroup(
            name='Y-Axis', options=['Depth from DF', 'Depth from KB', 'Depth from GL', 'TWT'],
            value='Depth from DF', button_type='primary', sizing_mode='stretch_width'
        )
        
        # Single Well Specific Widgets
        self.show_tops_check = pn.widgets.Checkbox(name='Show Well Tops', value=True, visible=False)
        self.color_inc_check = pn.widgets.Checkbox(name='Color by Inclination', value=False, visible=False)
        self.show_dots_check = pn.widgets.Checkbox(name='Show Dots', value=True, visible=False)
        self.invert_z_check = pn.widgets.Checkbox(name='Positive Z/MD (Invert)', value=False, visible=False)

        # Watcher to toggle visibility
        self.plot_mode_selector.param.watch(self.on_plot_mode_change, 'value')

        # Bindings
        self.import_btn.on_click(self.run_import)
        self.export_petrel_btn.on_click(self.run_export_to_petrel)
        self.export_file_btn.on_click(self.run_export_files)
        self.export_sum_btn.on_click(self.run_export_summary) 
        self.export_pdf_btn.on_click(self.run_export_pdf) 
        self.export_imgs_btn.on_click(self.run_export_imgs) 
        self.param_save_btn.on_click(self.run_save)
        
        self.well_selector.param.watch(self.update_inputs_from_selection, 'value')

        self.cs_left_pane = pn.bind(self.get_cs_table_left, self.well_selector)
        self.cs_right_pane = pn.bind(self.get_cs_table_right, self.well_selector, self.param_update_btn)
        
        # Plot Binding
        self.plot_pane = pn.bind(
            self.get_plot, 
            self.well_selector, 
            self.plot_mode_selector,
            self.plot_x_selector, 
            self.plot_y_selector, 
            self.all_wells_table.param.value,
            self.show_tops_check,
            self.color_inc_check,
            self.invert_z_check,
            self.show_dots_check
        )
        
        self.dynamic_title = pn.bind(self.get_cs_title, self.well_selector)

        # Init Selection
        if not self.df_headers.empty:
            self.all_wells_table.selection = [0]
            self.update_inputs_from_selection(None)

    def load_data(self):
        data_file = os.environ.get("PWR_DATA_FILE")
        headers = pd.DataFrame(columns=['WellName', 'UWI', 'GUID', 'X', 'Y', 'Lat', 'Long', 'KB', 'GL', 'SRD', 'DF', 'SE', 'SO', 'Vc'])
        checkshots = pd.DataFrame(columns=['WellName', 'MD', 'TWT', 'TVD'])
        surveys = pd.DataFrame(columns=['WellName', 'MD', 'Inclination', 'Azimuth', 'TVD', 'X', 'Y', 'Z'])
        logs = pd.DataFrame(columns=['WellName', 'MD', 'DT'])
        tops = pd.DataFrame(columns=['WellName', 'Surface', 'MD', 'TVD'])
        proj = "Unknown"
        
        if data_file and os.path.exists(data_file):
            try:
                con = duckdb.connect(data_file, read_only=True)
                tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
                if 'headers' in tables: headers = con.execute("SELECT * FROM headers").df()
                if 'checkshots' in tables: checkshots = con.execute("SELECT * FROM checkshots").df()
                if 'surveys' in tables: surveys = con.execute("SELECT * FROM surveys").df()
                if 'dt_logs' in tables: logs = con.execute("SELECT * FROM dt_logs").df()
                if 'tops' in tables: tops = con.execute("SELECT * FROM tops").df()
                if 'meta' in tables: 
                    meta_df = con.execute("SELECT * FROM meta").df()
                    if not meta_df.empty and 'ProjectName' in meta_df.columns:
                        proj = str(meta_df.iloc[0]['ProjectName'])
                con.close()
            except Exception as e: print(f"Data load error: {e}")
        
        # Ensure parameter columns exist in memory df even if not in DB initially
        required_params = ['KB', 'GL', 'SRD', 'DF', 'SE', 'SO', 'Vc']
        for col in required_params:
            if col not in headers.columns:
                headers[col] = 0.0
                if col == 'Vc': headers[col] = 5000.0 # Default Velocity

        return headers, checkshots, surveys, logs, tops, proj

    def enrich_headers(self):
        if self.df_headers.empty: return
        def check_exists(well, df):
            if df.empty: return False
            return well in df['WellName'].values

        self.df_headers['Survey'] = self.df_headers['WellName'].apply(lambda w: check_exists(w, self.df_surveys))
        self.df_headers['DT log'] = self.df_headers['WellName'].apply(lambda w: check_exists(w, self.df_dt))
        self.df_headers['Markers'] = self.df_headers['WellName'].apply(lambda w: check_exists(w, self.df_tops))

    def assign_well_colors(self):
        if self.df_headers.empty: return {}
        wells = self.df_headers['WellName'].unique()
        cmap = plt.cm.tab20
        colors_dict = {}
        for i, w in enumerate(wells):
            rgb = cmap(i % 20)
            colors_dict[w] = mcolors.to_hex(rgb)
        return colors_dict

    def on_plot_mode_change(self, event):
        is_single = (event.new == 'Single Well')
        self.show_tops_check.visible = is_single
        self.color_inc_check.visible = is_single
        self.invert_z_check.visible = is_single
        self.show_dots_check.visible = is_single

    def update_from_table_selection(self, event):
        indices = event.new
        if indices and not self.df_headers.empty:
            idx = indices[0]
            selected_well_name = self.df_headers.iloc[idx]['WellName']
            self.well_selector.value = selected_well_name

    def update_inputs_from_selection(self, event):
        well = self.well_selector.value
        if well and not self.df_headers.empty:
            row = self.df_headers[self.df_headers['WellName'] == well]
            if not row.empty:
                # Update ALL inputs from the in-memory dataframe (df_headers)
                try: self.inp_kb.value = float(row.iloc[0].get('KB', 0.0))
                except: self.inp_kb.value = 0.0
                
                try: self.inp_gl.value = float(row.iloc[0].get('GL', 0.0))
                except: self.inp_gl.value = 0.0

                try: self.inp_srd.value = float(row.iloc[0].get('SRD', 0.0))
                except: self.inp_srd.value = 0.0

                try: self.inp_df.value = float(row.iloc[0].get('DF', 0.0))
                except: self.inp_df.value = 0.0

                try: self.inp_se.value = float(row.iloc[0].get('SE', 0.0))
                except: self.inp_se.value = 0.0
                
                try: self.inp_so.value = float(row.iloc[0].get('SO', 0.0))
                except: self.inp_so.value = 0.0

                try: self.inp_vc.value = float(row.iloc[0].get('Vc', 5000.0))
                except: self.inp_vc.value = 5000.0

    def get_cs_title(self, well_name):
        return pn.pane.Markdown(f"### Checkshot Data for Well: {well_name or 'None'}")

    def toggle_export_settings(self, event):
        if event.new == 'Full Report (All Data)':
            self.export_file_depth_ref.disabled = True
            self.export_file_time_ref.disabled = True
        else:
            self.export_file_depth_ref.disabled = False
            self.export_file_time_ref.disabled = False

    def compute_geophysics(self, well_name, params):
        dff = self.df_checkshots[self.df_checkshots['WellName'] == well_name].copy()
        if dff.empty: return pd.DataFrame()

        kb = params.get('kb', 0)
        gl = params.get('gl', 0)
        srd = params.get('srd', 0)
        se = params.get('se', 0)
        so = params.get('so', 0)
        vc = params.get('vc', 1800)
        df_val = params.get('df', 0)

        dff['MD'] = pd.to_numeric(dff['MD'], errors='coerce')
        dff['TWT'] = pd.to_numeric(dff['TWT'], errors='coerce')
        dff = dff.dropna(subset=['MD', 'TWT']).sort_values(by='MD')
        
        # Average duplicate MD points to fix zig-zags
        if not dff.empty:
            # FIX: Only aggregate numeric columns to prevent TypeError with strings (WellName)
            dff = dff.groupby('MD', as_index=False).mean(numeric_only=True)

        if dff.empty: return pd.DataFrame()

        md_kb = dff['MD'] 
        depth_gl = md_kb - (kb - gl)
        depth_srd = md_kb - (kb - srd)
        depth_kb = md_kb
        dff['Depth_DF'] = md_kb - (kb - df_val)
        
        if 'Z' not in dff.columns: dff['Z'] = dff['MD']

        dz = dff['MD'] - (kb - se)
        hypot = np.sqrt(dz**2 + so**2)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            cos_theta = np.where(hypot == 0, 1, dz / hypot)
            theta_rad = np.arccos(np.clip(cos_theta, -1, 1))

        t_vert_source = dff['TWT'] * cos_theta
        static_source_gl = ((gl - se) / vc) * 1000.0 if vc != 0 else 0
        t_vert_gl = t_vert_source + static_source_gl

        datum_static = ((srd - gl) / vc) * 1000.0 if vc != 0 else 0
        t_vert_srd = t_vert_gl + datum_static

        with np.errstate(divide='ignore', invalid='ignore'):
            avg_vel = depth_srd / (t_vert_srd / 1000.0)
        
        delta_z = depth_srd.diff().fillna(depth_srd.iloc[0])
        delta_t = (t_vert_srd.diff().fillna(t_vert_srd.iloc[0])) / 1000.0
        with np.errstate(divide='ignore', invalid='ignore'):
            int_vel = delta_z / delta_t
        
        res = dff.copy()
        res['Depth_GL'] = depth_gl
        res['Depth_SRD'] = depth_srd
        res['Depth_KB'] = depth_kb
        res['T_Vert_GL'] = t_vert_gl
        res['TWT_Vert_GL'] = t_vert_gl * 2
        res['T_Vert_SRD'] = t_vert_srd
        res['V_avg'] = avg_vel.replace([np.inf, -np.inf], 0).fillna(0)
        res['V_int'] = int_vel.replace([np.inf, -np.inf], 0).fillna(0)
        res['Theta'] = theta_rad
        res['Datum_Static'] = datum_static
        res['SO'] = so
        return res

    # --- WIDGETS LOGIC ---
    def get_all_wells_table_widget(self):
        if self.df_headers.empty: 
            return pn.widgets.Tabulator(pd.DataFrame(), disabled=True)
        cols = ['Plot', 'WellName', 'UWI', 'X', 'Y', 'KB', 'Survey', 'DT log', 'Markers']
        valid_cols = [c for c in cols if c in self.df_headers.columns]
        return pn.widgets.Tabulator(self.df_headers[valid_cols], titles={'WellName': 'Well Name', 'Survey': 'Svy', 'DT log': 'DT', 'Markers': 'Tops'}, formatters={'Plot': 'tickCross', 'Survey': 'tickCross', 'DT log': 'tickCross', 'Markers': 'tickCross'}, text_align={'Plot': 'center', 'Survey': 'center', 'DT log': 'center', 'Markers': 'center'}, editors={'Plot': {'type': 'tickCross', 'tristate': False}, 'WellName': None, 'UWI': None, 'X': None, 'Y': None, 'KB': None, 'Survey': None, 'DT log': None, 'Markers': None}, show_index=False, theme='site', sizing_mode='stretch_both', pagination='remote', page_size=10, selectable=1, widths={'Plot': 50, 'Survey': 50, 'DT log': 50, 'Markers': 50})

    def get_cs_table_left(self, well_name):
        empty_df = pd.DataFrame(columns=["MD (from KB) [ft]", "Observed Travel Time (ms)"])
        if not well_name or self.df_checkshots.empty: return pn.widgets.Tabulator(empty_df, show_index=False, theme='site', sizing_mode='stretch_both')
        dff = self.df_checkshots[self.df_checkshots['WellName'] == well_name].copy()
        if dff.empty: return pn.widgets.Tabulator(empty_df, show_index=False, theme='site', sizing_mode='stretch_both')
        export_df = pd.DataFrame()
        export_df["MD (from KB) [ft]"] = dff['MD']
        export_df["Observed Travel Time (ms)"] = dff['TWT'] 
        return pn.widgets.Tabulator(export_df, show_index=False, theme='site', sizing_mode='stretch_both', pagination='remote', page_size=13)

    def get_cs_table_right(self, well_name, update_trigger=None):
        params = {'kb': self.inp_kb.value, 'gl': self.inp_gl.value, 'srd': self.inp_srd.value, 'se': self.inp_se.value, 'so': self.inp_so.value, 'vc': self.inp_vc.value, 'df': self.inp_df.value}
        calc_df = self.compute_geophysics(well_name, params)
        if calc_df.empty: return pn.widgets.Tabulator(pd.DataFrame(), show_index=False, theme='site', sizing_mode='stretch_both')
        
        export_df = pd.DataFrame()
        export_df["Depth Corr. To GL (ft)"] = calc_df['Depth_GL']
        export_df["Depth Corr to Reference Datum DP (ft)"] = calc_df['Depth_SRD']
        export_df["SRC-REC DISTANCE PALN VIEW (ft)"] = calc_df['SO']
        export_df["Theta (Radians)"] = calc_df['Theta']
        export_df["Vertical Time CORR to GL (ms)"] = calc_df['T_Vert_GL']
        export_df["Vertical Time CORR to GL (TWT) (ms)"] = calc_df['TWT_Vert_GL']
        export_df["DATUM (ms)"] = calc_df['Datum_Static']
        export_df["Vertical Time corr to reference datum DP (ms)"] = calc_df['T_Vert_SRD']
        export_df["Vertical time corr to DP (TWT) (ms)"] = calc_df['T_Vert_SRD'] * 2
        export_df["Average Velocity (ft/s)"] = calc_df['V_avg']
        export_df["Interval Velocity (ft/s)"] = calc_df['V_int']
        
        return pn.widgets.Tabulator(export_df.round(2), show_index=False, disabled=True, theme='site', sizing_mode='stretch_both', pagination='remote', page_size=13)

    # --- PLOT LOGIC ---
    def get_plot(self, well_name, plot_mode, x_sel, y_sel, table_df, show_tops, color_inc, invert_z, show_dots):
        if self.df_checkshots.empty: return pn.pane.Markdown("No Checkshot Data Found")
        
        x_col, x_label = 'TWT', 'OWT (ms)'
        factor = 1.0
        if x_sel == 'TWT': x_col, x_label, factor = 'TWT_Vert_GL', 'TWT (ms)', 2.0
        elif x_sel == 'Int. Vel.': x_col, x_label = 'V_int', 'Interval Velocity (ft/s)'
        elif x_sel == 'Avg. Vel.': x_col, x_label = 'V_avg', 'Average Velocity (ft/s)'
        elif x_sel == 'TVD': x_col, x_label = 'Z', 'TVD (ft)'
        elif x_sel == 'MD': x_col, x_label = 'MD', 'MD (ft)'

        y_col, y_label = 'Depth_DF', 'Depth (DF)'
        if y_sel == 'Depth from KB': y_col, y_label = 'Depth_KB', 'Depth (KB)'
        elif y_sel == 'Depth from GL': y_col, y_label = 'Depth_GL', 'Depth (GL)'
        elif y_sel == 'TWT': y_col, y_label = 'TWT_Vert_GL', 'TWT (ms)'

        invert_y = True 
        if y_sel == 'TWT': invert_y = True 

        if plot_mode == 'Single Well':
            wells_to_plot = [well_name] if well_name else []
        else:
            if not table_df.empty and 'Plot' in table_df.columns:
                wells_to_plot = table_df[table_df['Plot']]['WellName'].tolist()
            else:
                wells_to_plot = self.df_headers['WellName'].unique().tolist()

        g_params = {'gl': self.inp_gl.value, 'srd': self.inp_srd.value, 'se': self.inp_se.value, 'so': self.inp_so.value, 'vc': self.inp_vc.value, 'df': self.inp_df.value}
        plot_list = []
        
        for w in wells_to_plot:
            w_params = g_params.copy()
            if w != well_name:
                row = self.df_headers[self.df_headers['WellName'] == w]
                kb_val = float(row.iloc[0]['KB']) if not row.empty else 0.0
                w_params['kb'] = kb_val
            else:
                w_params['kb'] = self.inp_kb.value

            calc_df = self.compute_geophysics(w, w_params)
            if calc_df.empty: continue
            
            if y_col in calc_df.columns:
                calc_df = calc_df.sort_values(by=y_col)

            plot_df = calc_df.copy()
            if factor != 1.0 and x_col in plot_df.columns:
                plot_df[x_col] = plot_df[x_col] * factor

            if x_col not in plot_df.columns or y_col not in plot_df.columns: continue
            
            if invert_z and x_sel in ['TVD', 'MD']:
                plot_df[x_col] = plot_df[x_col] * -1.0
            
            plot_df = plot_df[pd.to_numeric(plot_df['MD'], errors='coerce').notna()]
            if plot_df.empty: continue

            if plot_mode == 'Single Well':
                line = None
                if color_inc and not self.df_surveys.empty:
                    srv = self.df_surveys[self.df_surveys['WellName'] == w].copy()
                    if not srv.empty:
                        srv = srv.dropna(subset=['MD']).sort_values('MD')
                        plot_df = plot_df.sort_values('MD')
                        try:
                            merged = pd.merge_asof(plot_df, srv[['MD', 'Inclination']], on='MD', direction='nearest')
                            merged['x1'] = merged[x_col].shift(-1)
                            merged['y1'] = merged[y_col].shift(-1)
                            merged = merged.dropna(subset=['x1', 'y1', 'Inclination'])
                            if not merged.empty:
                                line = hv.Segments(merged, [x_col, y_col, 'x1', 'y1'], vdims=['Inclination']).opts(
                                    color='Inclination', cmap='magma', line_width=4, colorbar=True, clim=(0, 90), tools=['hover'])
                        except: pass
                
                if line is None:
                    color = self.well_colors.get(w, 'red')
                    line = hv.Curve(plot_df, x_col, y_col, label=w).opts(color=color, line_width=3)
                
                if show_dots:
                    line = line * hv.Scatter(plot_df, x_col, y_col).opts(color='black', size=6, tools=['hover'])

                if show_tops and not self.df_tops.empty:
                    tops = self.df_tops[self.df_tops['WellName'] == w]
                    if not tops.empty:
                        xp = plot_df['MD'].values
                        yp = plot_df[y_col].values
                        if len(xp) > 1:
                            tops_y = np.interp(tops['MD'], xp, yp)
                            for idx, t_row in tops.iterrows():
                                if idx < len(tops_y):
                                    depth_val = tops_y[idx]
                                    lbl = t_row['Surface']
                                    hline = hv.HLine(depth_val).opts(color='green', line_width=1, line_dash='dashed')
                                    text = hv.Text(plot_df[x_col].min(), depth_val, lbl, halign='left', valign='bottom').opts(text_font_size='8pt', text_color='green')
                                    line = line * hline * text
                plot_list.append(line)
            else:
                color = self.well_colors.get(w, '#808080')
                lw = 4 if w == well_name else 1
                alpha = 1.0 if w == well_name else 0.7
                line = hv.Curve(plot_df, x_col, y_col, label=w).opts(color=color, line_width=lw, alpha=alpha)
                if w == well_name:
                    scatter = hv.Scatter(plot_df, x_col, y_col).opts(color=color, size=8).opts(tools=['hover'])
                    plot_list.append(line * scatter)
                else:
                    plot_list.append(line)

        if not plot_list: return pn.pane.Markdown("No Data to Plot")
        return pn.pane.HoloViews(hv.Overlay(plot_list).opts(responsive=True, min_height=600, show_grid=True, invert_yaxis=invert_y, legend_position='bottom_left', xlabel=x_label, ylabel=y_label, title=f"Checkshots: {x_label} vs {y_label}", toolbar='right'), sizing_mode='stretch_both')

    # --- ACTIONS ---
    def run_import(self, event):
        if self.file_input.value is None:
            pn.state.notifications.error("No file selected", duration=4000)
            return
        
        well_target = self.import_well_selector.value
        depth_fmt = self.import_depth_fmt.value
        time_fmt = self.import_time_fmt.value
        
        try:
            string_io = io.StringIO(self.file_input.value.decode("utf-8"))
            first_line = string_io.readline()
            string_io.seek(0)
            sep = ',' if ',' in first_line else '\t'
            if len(first_line.split()) > 1 and sep == '\t': sep = r'\s+'
            
            header_row = 0 if self.import_skip_header.value else None
            df_imp = pd.read_csv(string_io, sep=sep, header=header_row)
            
            if df_imp.shape[1] < 2:
                pn.state.notifications.error("File must have at least 2 columns", duration=4000)
                return
            
            raw_depth = pd.to_numeric(df_imp.iloc[:, 0], errors='coerce')
            raw_time = pd.to_numeric(df_imp.iloc[:, 1], errors='coerce')
            df_imp = pd.DataFrame({'RawDepth': raw_depth, 'RawTime': raw_time}).dropna()

            row_head = self.df_headers[self.df_headers['WellName'] == well_target]
            if row_head.empty:
                pn.state.notifications.error(f"Well {well_target} not found", duration=4000)
                return
            
            kb = float(row_head.iloc[0]['KB'])
            gl = self.inp_gl.value
            df = self.inp_df.value 
            
            if self.import_negate_depth.value:
                df_imp['RawDepth'] = -1.0 * df_imp['RawDepth']

            if time_fmt == 'TWT (ms)': df_imp['TWT'] = df_imp['RawTime'] / 2.0
            else: df_imp['TWT'] = df_imp['RawTime']

            if depth_fmt == 'MD from KB (ft)': df_imp['MD'] = df_imp['RawDepth']
            elif depth_fmt == 'MD from DF (ft)': df_imp['MD'] = df_imp['RawDepth'] + (kb - df)
            elif depth_fmt == 'MD from GL (ft)': df_imp['MD'] = df_imp['RawDepth'] + (kb - gl)
            else: df_imp['MD'] = df_imp['RawDepth']

            df_imp['TVD'] = df_imp['MD']

            self.df_checkshots = self.df_checkshots[self.df_checkshots['WellName'] != well_target]
            new_rows = df_imp[['MD', 'TWT', 'TVD']].copy()
            new_rows['WellName'] = well_target
            self.df_checkshots = pd.concat([self.df_checkshots, new_rows], ignore_index=True)
            
            pn.state.notifications.success(f"Import successful for {well_target}!", duration=4000)
            self.well_selector.param.trigger('value')
            
        except Exception as e:
            pn.state.notifications.error(f"Import Error: {e}", duration=4000)

    def run_save(self, event):
        well_name = self.well_selector.value
        if not well_name: return

        # Update Memory Dataframe First (so switching wells keeps values)
        if not self.df_headers.empty:
            idx = self.df_headers[self.df_headers['WellName'] == well_name].index
            if not idx.empty:
                i = idx[0]
                self.df_headers.at[i, 'KB'] = self.inp_kb.value
                self.df_headers.at[i, 'GL'] = self.inp_gl.value
                self.df_headers.at[i, 'SRD'] = self.inp_srd.value
                self.df_headers.at[i, 'DF'] = self.inp_df.value
                self.df_headers.at[i, 'SE'] = self.inp_se.value
                self.df_headers.at[i, 'SO'] = self.inp_so.value
                self.df_headers.at[i, 'Vc'] = self.inp_vc.value

        # 1. Update Database
        try:
            db_path = os.environ.get("PWR_DATA_FILE")
            if not db_path:
                pn.state.notifications.error("DB Save Error: PWR_DATA_FILE not set", duration=5000)
                return

            con = duckdb.connect(db_path)
            
            # Check/Add missing columns
            existing_cols = [c[1] for c in con.execute("PRAGMA table_info('headers')").fetchall()]
            required_cols = ['GL', 'SRD', 'DF', 'SE', 'SO', 'Vc']
            for col in required_cols:
                if col not in existing_cols:
                    con.execute(f"ALTER TABLE headers ADD COLUMN {col} DOUBLE")

            if 'WellName' not in existing_cols:
                raise RuntimeError("headers table missing WellName column")

            exists = con.execute("SELECT COUNT(*) FROM headers WHERE WellName=?", (well_name,)).fetchone()[0]
            if exists == 0:
                con.execute(
                    """
                    INSERT INTO headers (WellName, KB, GL, SRD, DF, SE, SO, Vc)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        well_name,
                        self.inp_kb.value,
                        self.inp_gl.value,
                        self.inp_srd.value,
                        self.inp_df.value,
                        self.inp_se.value,
                        self.inp_so.value,
                        self.inp_vc.value,
                    ),
                )
            else:
                con.execute(
                    """
                    UPDATE headers
                    SET KB=?, GL=?, SRD=?, DF=?, SE=?, SO=?, Vc=?
                    WHERE WellName=?
                    """,
                    (
                        self.inp_kb.value,
                        self.inp_gl.value,
                        self.inp_srd.value,
                        self.inp_df.value,
                        self.inp_se.value,
                        self.inp_so.value,
                        self.inp_vc.value,
                        well_name,
                    ),
                )
            
            con.close()
            pn.state.notifications.success(f"Saved Params to DB", duration=3000)
        except Exception as e:
            pn.state.notifications.error(f"DB Save Error: {e}", duration=5000)

        # 2. Export to Petrel (Called by Save button)
        self.run_export_to_petrel(event)

    def run_export_to_petrel(self, event):
        well_name = self.well_selector.value
        if not well_name: return

        try:
            from cegalprizm.pythontool import PetrelConnection
            ptp = PetrelConnection(allow_experimental=True)
            
            # A. Update Well Datum
            well = ptp.wells.get_by_name(well_name)
            if not well:
                pn.state.notifications.error(f"Petrel Error: Well {well_name} not found", duration=5000)
                return

            # CRITICAL: Unlock well
            if well.readonly: well.readonly = False
            
            # Formatted Date Description
            desc = f"Edited by CheckShot Prizm app on {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            well.well_datum = ("KB", self.inp_kb.value, desc)

            # B. Write Other Params as Attributes
            # Map App Params to Attribute Names
            attr_map = {
                "Ground Level": self.inp_gl.value,
                "Seismic Reference Datum": self.inp_srd.value,
                "Drill Floor": self.inp_df.value,
                "Source Elevation": self.inp_se.value,
                "Source Offset": self.inp_so.value,
                "Replacement Velocity": self.inp_vc.value
            }

            for attr_name, val in attr_map.items():
                try:
                    # Robust check for existing attributes to avoid 'list' error
                    exists = False
                    # FIX: Use get_by_name to safely check existence (returns None if missing, object or list if present)
                    res = ptp.well_attributes.get_by_name(attr_name)
                    
                    if res:
                        exists = True
                    else:
                        exists = False

                    if not exists:
                        ptp.create_well_attribute(attr_name, "Continuous")
                        time.sleep(0.5) # Short pause for Petrel to register
                    
                    # Access instance on well
                    # Using try-except block here because accessing attributes dictionary sometimes fails if sync is slow
                    try:
                         if attr_name in well.attributes:
                            attr_inst = well.attributes[attr_name]
                            if attr_inst:
                                if attr_inst.readonly: attr_inst.readonly = False
                                attr_inst.value = float(val)
                    except:
                        pass # Attribute might not be attached to well yet
                        
                except Exception as e:
                    # Log but continue
                    print(f"Warning: Could not set attribute {attr_name}: {e}")

            # C. Export Checkshot Log
            params = {'kb': self.inp_kb.value, 'gl': self.inp_gl.value, 'srd': self.inp_srd.value, 
                      'se': self.inp_se.value, 'so': self.inp_so.value, 'vc': self.inp_vc.value, 'df': self.inp_df.value}
            
            calc_df = self.compute_geophysics(well_name, params)
            
            if not calc_df.empty:
                calc_df = calc_df.sort_values(by='MD')
                md_values = calc_df['MD'].values
                twt_values = calc_df['T_Vert_SRD'].values * 2.0

                gwl_name = "Checkshot_TWT_SRD"
                # Check/Create GWL
                gwl_list = [l for l in ptp.global_well_logs if l.petrel_name == gwl_name]
                if gwl_list:
                    gwl = gwl_list[0]
                else:
                    templ = ptp.templates.get_by_name("Two-way time")
                    if not templ: templ = ptp.templates.get_by_name("One-way time")
                    if templ:
                        gwl = ptp.create_global_well_log(gwl_name, template=templ)
                    else:
                        gwl = ptp.create_global_well_log(gwl_name)

                existing_logs = [l for l in well.logs if l.global_well_log.petrel_name == gwl_name]
                if existing_logs:
                    log = existing_logs[0]
                else:
                    log = gwl.create_well_log(well)

                if log.readonly: log.readonly = False
                log.set_values(md_values, twt_values)
                
                self.export_petrel_status.object = f"Exported: {gwl_name} to {well_name}"
                pn.state.notifications.success(f"Full Export Complete: {well_name}", duration=3000)
            else:
                self.export_petrel_status.object = "No calc data to export."

        except Exception as e:
            self.export_petrel_status.object = f"Error: {e}"
            pn.state.notifications.error(f"Petrel Export Error: {e}", duration=5000)

    
    def run_export_files(self, event): 
        folder = self.export_path_input.value
        filename = self.export_filename_input.value
        file_ext = self.export_fmt_selector.value
        content_mode = self.export_content_selector.value
        
        if not folder or not filename:
            pn.state.notifications.error("Export Failed: Check path/filename", duration=4000)
            return
        if self.df_headers.empty: return

        pn.state.notifications.info(f"Starting Export ({content_mode})...", duration=2000)
        srd_val, vc_val = self.inp_srd.value, self.inp_vc.value
        
        try:
            for w in self.df_headers['WellName'].unique():
                row = self.df_headers[self.df_headers['WellName'] == w]
                kb_val = float(row.iloc[0]['KB']) if not row.empty else 0.0
                params = {'kb': kb_val, 'gl': self.inp_gl.value, 'srd': srd_val, 'se': self.inp_se.value, 'so': self.inp_so.value, 'vc': vc_val, 'df': self.inp_df.value}
                
                calc_df = self.compute_geophysics(w, params)
                if calc_df.empty: continue
                
                exp_df = pd.DataFrame()
                
                if content_mode == 'Full Report (All Data)':
                    exp_df['MD (DF)'] = calc_df['Depth_DF']
                    exp_df['MD (KB)'] = calc_df['Depth_KB']
                    exp_df['MD (GL)'] = calc_df['Depth_GL'] 
                    exp_df['OWT (Observed)'] = calc_df['TWT']
                    exp_df['Vertical OWT to GL'] = calc_df['T_Vert_GL']
                    exp_df['Vertical TWT to GL'] = calc_df['TWT_Vert_GL']
                    exp_df['Vertical OWT to SRD'] = calc_df['T_Vert_SRD']
                    exp_df['Vertical TWT to SRD'] = calc_df['T_Vert_SRD'] * 2
                    exp_df['Theta'] = calc_df['Theta']
                    exp_df['Avg. Vel. (ft/s)'] = calc_df['V_avg']
                    exp_df['Int. Vel. (ft/s)'] = calc_df['V_int']
                    
                    safe_name = re.sub(r'[\\/*?:"<>|]', '_', w)
                    full_path = os.path.join(folder, f"{filename}_FULL_well_{safe_name}{file_ext}")
                    sep = ',' if file_ext == '.csv' else '\t'
                    exp_df.round(2).to_csv(full_path, index=False, sep=sep)
                    
                else:
                    depth_ref = self.export_file_depth_ref.value
                    time_ref = self.export_file_time_ref.value
                    
                    if depth_ref == 'MD from KB': exp_df['Depth'] = calc_df['Depth_KB']
                    elif depth_ref == 'MD from GL': exp_df['Depth'] = calc_df['Depth_GL']
                    else: exp_df['Depth'] = calc_df['Depth_DF']
                    
                    if time_ref == 'TWT': exp_df['Time'] = calc_df['T_Vert_SRD'] * 2
                    else: exp_df['Time'] = calc_df['T_Vert_SRD'] 

                    exp_df['Well name'] = w

                    if file_ext == '.csv':
                        header_template = f"""# Petrel checkshots format,,
# Time SRD {srd_val} ft AMSL,,
# Replacement Velocity {vc_val} ft/s,,
#,,
# {depth_ref} (ft),,
# {time_ref} (ms),,
#,,
VERSION 1,,
BEGIN HEADER,,
MD,,
TWT,,
Well name,,
END HEADER,,
"""
                    else:
                        header_template = f"""# Petrel checkshots format
# Time SRD {srd_val} ft AMSL
# Replacement Velocity {vc_val} ft/s
#
# {depth_ref} (ft)
# {time_ref} (ms)
#
VERSION 1
BEGIN HEADER
MD
TWT
Well name
END HEADER
"""
                    safe_name = re.sub(r'[\\/*?:"<>|]', '_', w)
                    full_path = os.path.join(folder, f"{filename}_well_{safe_name}{file_ext}")
                    
                    with open(full_path, 'w', newline='') as f:
                        f.write(header_template)
                        sep = ',' if file_ext == '.csv' else '\t'
                        exp_df[['Depth', 'Time', 'Well name']].round(2).to_csv(f, index=False, header=False, sep=sep)
            
            pn.state.notifications.success(f"Exported files to {folder}", duration=4000)
        except Exception as e:
            pn.state.notifications.error(f"Error: {e}", duration=4000)

    def run_export_summary(self, event):
        if self.df_headers.empty: return
        folder = self.export_sum_path_input.value
        filename = self.export_sum_filename_input.value
        if not folder or not filename: return
        full_path = os.path.join(folder, f"{filename}.csv")
        df_exp = self.df_headers.copy()
        df_exp = df_exp.rename(columns={'WellName': 'Well Name'})
        required = ['Well Name', 'UWI', 'X', 'Y', 'Lat', 'Long', 'SRD', 'KB', 'DF', 'GL', 'SE', 'SO', 'Vc']
        for c in required: 
            if c not in df_exp.columns: df_exp[c] = 0.0
        try:
            df_exp[required].to_csv(full_path, index=False)
            pn.state.notifications.success(f"Summary saved", duration=4000)
        except: pass

    def run_export_pdf(self, event): 
        folder = self.export_pdf_path_input.value
        filename = self.export_pdf_filename_input.value
        plot_select = self.export_pdf_x_selector.value
        depth_ref = self.export_pdf_depth_ref.value
        
        if not folder or not filename: return
        full_path = os.path.join(folder, f"{filename}.pdf")
        pn.state.notifications.info(f"Generating PDF...", duration=2000)

        try:
            c = canvas.Canvas(full_path, pagesize=landscape(A4))
            width, height = landscape(A4)
            g_params = {'gl': self.inp_gl.value, 'srd': self.inp_srd.value, 'se': self.inp_se.value, 'so': self.inp_so.value, 'vc': self.inp_vc.value, 'df': self.inp_df.value}
            
            def draw_logo(c):
                if valid_logo:
                    logo_w, logo_h = 1.5*inch, 0.5*inch
                    c.drawImage(valid_logo, width - logo_w - 0.5*inch, height - logo_h - 0.3*inch, width=logo_w, height=logo_h, preserveAspectRatio=True, mask='auto')

            def create_plot_strip(target_well=None, all_wells_data=None):
                is_single_plot = plot_select != 'All (4-Panel)'
                if is_single_plot:
                    fig, ax = plt.subplots(figsize=(6, 5))
                    axes = [ax]
                    titles = [plot_select]
                else:
                    fig, axes = plt.subplots(1, 4, figsize=(10, 5))
                    titles = ["OWT (ms)", "TWT (ms)", "Interval Vel (ft/s)", "Average Vel (ft/s)"]
                
                y_key = 'Depth_DF'
                if depth_ref == 'Depth from KB': y_key = 'Depth_KB'
                elif depth_ref == 'Depth from GL': y_key = 'Depth_GL'

                for i, ax_i in enumerate(axes):
                    ax_i.set_title(titles[i], fontsize=10)
                    ax_i.set_ylabel(f"{depth_ref} (ft)", fontsize=8)
                    ax_i.invert_yaxis()
                    ax_i.grid(True, linestyle=':', alpha=0.6)

                colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

                for i, (w_name, w_df) in enumerate(all_wells_data):
                    if w_df.empty: continue
                    if target_well is None:
                        color, lw, alpha, label = colors_list[i % len(colors_list)], 0.8, 0.8, w_name
                        is_target_plot = True
                    else:
                        if w_name != target_well: continue
                        color, lw, alpha, label = 'red', 1.0, 1.0, w_name
                        is_target_plot = True

                    if is_target_plot:
                        def get_col(t):
                            if 'OWT' in t: return 'TWT'
                            if 'TWT' in t: return 'TWT_Vert_GL'
                            if 'Interval' in t: return 'V_int'
                            if 'Average' in t: return 'V_avg'
                            return 'TWT'

                        for j, ax_j in enumerate(axes):
                            col = get_col(titles[j])
                            if col in w_df.columns and y_key in w_df.columns:
                                # FIX: Sort data for PDF report plots
                                w_df_sorted = w_df.sort_values(by=y_key)
                                ax_j.plot(w_df_sorted[col], w_df_sorted[y_key], color=color, linewidth=lw, alpha=alpha, label=label)

                if target_well is None:
                    handles, labels = axes[0].get_legend_handles_labels()
                    if handles: fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=6, fontsize='x-small')
                    plt.tight_layout(rect=[0, 0.1, 1, 1])
                else: plt.tight_layout()
                
                img_data = io.BytesIO()
                plt.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
                img_data.seek(0)
                plt.close(fig)
                return img_data

            all_wells_list = self.df_headers['WellName'].unique()
            precomputed_data = []
            for w in all_wells_list:
                row = self.df_headers[self.df_headers['WellName'] == w]
                kb_val = float(row.iloc[0]['KB']) if not row.empty else 0.0
                w_params = g_params.copy()
                w_params['kb'] = kb_val
                precomputed_data.append((w, self.compute_geophysics(w, w_params)))

            draw_logo(c)
            c.setFont("Helvetica-Bold", 16)
            c.drawString(0.5*inch, height - 0.8*inch, f"Project Summary: {self.project_name}")
            
            table_data = [['Well Name', 'UWI', 'X', 'Y', 'Lat', 'Long', 'SRD', 'KB', 'DF', 'GL', 'SE', 'SO', 'Vc']]
            for i, row in self.df_headers.iterrows():
                r_data = [row.get('WellName', ''), row.get('UWI', ''), f"{row.get('X', 0):.1f}", f"{row.get('Y', 0):.1f}", f"{row.get('Lat', 0):.4f}", f"{row.get('Long', 0):.4f}", g_params['srd'], row.get('KB', 0), self.inp_df.value, g_params['gl'], g_params['se'], g_params['so'], g_params['vc']]
                table_data.append([str(x) for x in r_data])
            t = Table(table_data, colWidths=[1.2*inch, 1.2*inch] + [0.7*inch]*11)
            t.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.Color(5/255, 39/255, 89/255)), ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke), ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), ('FONTSIZE', (0,0), (-1,0), 8), ('BOTTOMPADDING', (0,0), (-1,0), 6), ('BACKGROUND', (0,1), (-1,-1), colors.white), ('GRID', (0,0), (-1,-1), 1, colors.black), ('FONTSIZE', (0,1), (-1,-1), 7)]))
            w_tab, h_tab = t.wrapOn(c, width, height)
            t.drawOn(c, 0.5*inch, height - 1.0*inch - h_tab)

            plot_img = create_plot_strip(target_well=None, all_wells_data=precomputed_data)
            c.drawImage(ImageReader(plot_img), 0.5*inch, 0.5*inch, width=width-1*inch, height=4.5*inch)
            c.showPage()

            for w_name, w_df in precomputed_data:
                draw_logo(c)
                c.setFont("Helvetica-Bold", 18)
                c.drawString(0.5*inch, height - 1.0*inch, f"Well: {w_name}")
                if not w_df.empty:
                    p_buf = create_plot_strip(target_well=w_name, all_wells_data=precomputed_data)
                    c.drawImage(ImageReader(p_buf), 0.5*inch, height - 7.0*inch, width=width-1*inch, height=4.5*inch)
                c.showPage()

            c.save()
            pn.state.notifications.success("PDF Generated", duration=4000)
        except Exception as e:
            pn.state.notifications.error(f"PDF Error: {e}", duration=4000)

    def run_export_imgs(self, event): 
        folder = self.export_imgs_path_input.value
        filename = self.export_imgs_filename_input.value
        plot_type = self.export_imgs_type_selector.value
        depth_ref = self.export_imgs_depth_ref.value 
        dpi = int(self.export_imgs_dpi_selector.value)
        fmt = self.export_imgs_fmt_selector.value
        
        if not folder or not filename: return
        pn.state.notifications.info(f"Exporting Images...", duration=2000)

        all_wells_list = self.df_headers['WellName'].unique()
        g_params = {'gl': self.inp_gl.value, 'srd': self.inp_srd.value, 'se': self.inp_se.value, 'so': self.inp_so.value, 'vc': self.inp_vc.value, 'df': self.inp_df.value}
        precomputed_data = []
        for w in all_wells_list:
            row = self.df_headers[self.df_headers['WellName'] == w]
            w_params = g_params.copy()
            w_params['kb'] = float(row.iloc[0]['KB']) if not row.empty else 0.0
            precomputed_data.append((w, self.compute_geophysics(w, w_params)))

        if plot_type == 'OWT': x_col, x_label = 'TWT', 'OWT (ms)'
        elif plot_type == 'TWT': x_col, x_label = 'TWT_Vert_GL', 'TWT (ms)'
        elif plot_type == 'Int. Vel.': x_col, x_label = 'V_int', 'Interval Velocity (ft/s)'
        else: x_col, x_label = 'V_avg', 'Average Velocity (ft/s)'

        if depth_ref == 'Depth from KB': y_key, y_label = 'Depth_KB', 'Depth from KB (ft)'
        elif depth_ref == 'Depth from GL': y_key, y_label = 'Depth_GL', 'Depth from GL (ft)'
        else: y_key, y_label = 'Depth_DF', 'Depth from DF (ft)'

        def generate_plot(target_well=None):
            fig, ax = plt.subplots(figsize=(5, 11))
            ax.set_title(f"Checkshots: {plot_type}", fontsize=12)
            ax.set_xlabel(x_label, fontsize=10)
            ax.set_ylabel(y_label, fontsize=10)
            ax.invert_yaxis()
            ax.grid(True, linestyle=':', alpha=0.6)
            colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

            for i, (w_name, w_df) in enumerate(precomputed_data):
                if w_df.empty: continue
                
                # FIX: Sort Data
                w_df = w_df.sort_values(by=y_key)

                if target_well is None:
                    color = colors_list[i % len(colors_list)]
                    ax.plot(w_df[x_col], w_df[y_key], color=color, linewidth=1.0, alpha=0.8, label=w_name)
                else:
                    if w_name != target_well: continue
                    ax.plot(w_df[x_col], w_df[y_key], color='red', linewidth=2.0, label=w_name)
            
            if target_well is None: ax.legend(loc='upper right', fontsize='x-small')
            plt.tight_layout()
            return fig

        try:
            fig_all = generate_plot(None)
            fig_all.savefig(os.path.join(folder, f"{filename}_all{fmt}"), dpi=dpi)
            plt.close(fig_all)
            for w, _ in precomputed_data:
                safe = re.sub(r'[\\/*?:"<>|]', '_', w)
                fig = generate_plot(w)
                fig.savefig(os.path.join(folder, f"{filename}_well_{safe}{fmt}"), dpi=dpi)
                plt.close(fig)
            pn.state.notifications.success("Images Exported", duration=4000)
        except Exception as e:
            pn.state.notifications.error(f"Error: {e}", duration=4000)

    # --- TEMPLATE GENERATION ---
    def get_template(self):
        box_style = {
            'background': '#ffffff' if not is_dark_mode else DARK_BLUE_OMV_COLOR,
            'border-radius': '5px',
            'box-shadow': '0px 0px 5px rgba(0,0,0,0.1)',
            'padding': '10px',
            'border': '1px solid #e0e0e0',
            'overflow': 'auto',
            'color': section_text_color,
        }
        
        all_wells_card = pn.Column(pn.pane.Markdown("**All Wells List**"), pn.layout.Divider(), self.all_wells_table, styles=box_style, sizing_mode='stretch_both', margin=(0, 5, 0, 0))
        params_col1 = pn.Column(self.inp_srd, self.inp_kb, self.inp_df, self.inp_gl, sizing_mode='stretch_width')
        params_col2 = pn.Column(self.inp_se, self.inp_so, self.inp_vc, self.param_update_btn, self.param_save_btn, sizing_mode='stretch_width')
        params_card = pn.Column(pn.pane.Markdown("**Well Parameters**"), pn.layout.Divider(), pn.Row(params_col1, params_col2, sizing_mode='stretch_width'), styles=box_style, sizing_mode='stretch_both', margin=(0, 0, 0, 5))
        top_row = pn.GridSpec(sizing_mode='stretch_width', ncols=2, nrows=1, height=400)
        top_row[0, 0] = all_wells_card
        top_row[0, 1] = params_card

        input_data_card = pn.Column(pn.pane.Markdown("**Input Data**"), pn.layout.Divider(), self.cs_left_pane, styles=box_style, width=500, sizing_mode='stretch_height', margin=(0, 5, 0, 0))
        computed_data_card = pn.Column(pn.pane.Markdown("**Computed Data**"), pn.layout.Divider(), self.cs_right_pane, styles=box_style, sizing_mode='stretch_both', margin=(0, 0, 0, 5))
        bottom_row = pn.Row(input_data_card, computed_data_card, sizing_mode='stretch_both')

        left_container = pn.Column(top_row, pn.Spacer(height=15), self.dynamic_title, bottom_row, sizing_mode='stretch_both', margin=(0, 10, 0, 0))
        
        # Plot Box: Fixed width, stretch height
        plot_box = pn.Column(
            pn.pane.Markdown("### Plot Data"), 
            self.plot_mode_selector,
            self.plot_x_selector, 
            self.plot_y_selector, 
            pn.Row(self.show_tops_check, self.color_inc_check, self.invert_z_check, self.show_dots_check),
            pn.layout.Divider(), 
            self.plot_pane, 
            styles=box_style, 
            width=600, 
            sizing_mode='stretch_height'
        )

        # MAIN LAYOUT: Row instead of GridSpec
        main_layout = pn.Row(left_container, plot_box, sizing_mode='stretch_both')

        main_content = pn.Column(
            main_layout,
            sizing_mode='stretch_both',
            margin=0,
            styles={
                'height': '100%',
                'overflow': 'hidden',
                'background': get_main_outer_background(is_dark_mode),
                'color': section_text_color if is_dark_mode else 'inherit',
            },
        )

        sidebar_content = [self.import_card, pn.Spacer(height=10), self.export_petrel_card, pn.Spacer(height=10), self.export_file_card, pn.Spacer(height=10), self.export_summary_card, pn.Spacer(height=10), self.export_pdf_card, pn.Spacer(height=10), self.export_imgs_card]

        template = pn.template.FastListTemplate(
            title=APP_TITLE,
            logo=valid_logo,
            favicon=valid_favicon,
            accent_base_color=BLUE_OMV_COLOR,
            header_background=DARK_BLUE_OMV_COLOR,
            header=[
                pn.Row(
                    pn.Spacer(sizing_mode='stretch_width'),
                    pn.pane.HTML(docs_button_html(DOCUMENTATION_URL)),
                    sizing_mode='stretch_width',
                    margin=0,
                )
            ],
            sidebar=sidebar_content,
            main=[main_content],
            main_layout=None,
            main_max_width="",
        )
        return template

# Run App
app = CheckShotApp()
template = app.get_template()
template.servable()

# Working code Feb, 12th, 2026 | 17:10
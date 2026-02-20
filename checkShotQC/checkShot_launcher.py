# Start: PWR Description
import sys
import os
import subprocess
import time
import webbrowser
import warnings
import psutil 
import socket
import threading
import pandas as pd
import duckdb
import logging
from datetime import datetime
from pathlib import Path

# --- 1. SETUP PATH TO LOCAL CONSTANTS ---
current_dir = str(Path(__file__).resolve().parent)

if current_dir not in sys.path:
    sys.path.append(current_dir)

# --- 2. IMPORT CONSTANTS ---
try:
    import checkShot_constants as constants
    from importlib import reload
    reload(constants)
except ImportError as e:
    raise ImportError(f"PWR Block failed to load constants from {current_dir}. Error: {e}")

from cegalprizm.pycoderunner import WorkflowDescription
from cegalprizm.pythontool import DomainObjectsEnum

# --- 3. DEFINE WORKFLOW ---
pwr_description = WorkflowDescription(
    name=constants.WF_NAME,
    category=constants.WF_CATEGORY,
    description=constants.WF_DESCRIPTION,
    authors=constants.WF_AUTHORS,
    version=constants.WF_VERSION
)

# A. Well Selection
pwr_description.add_object_ref_parameter(
    name="selected_wells",
    label="Select Wells",
    description="Select the wells to extract data from.",
    object_type=DomainObjectsEnum.Well,
    select_multiple=True,
    parameter_group="1. Wells"
)

# B. Optional Data
pwr_description.add_object_ref_parameter(
    name="sel_dt_global",
    label="Select DT Log (Global)",
    description="Select the Global Well Log representing DT (Sonic) to extract.",
    object_type=DomainObjectsEnum.GlobalWellLog,
    select_multiple=False,
    parameter_group="2. Optional Data"
)

pwr_description.add_object_ref_parameter(
    name="sel_marker_coll",
    label="Select Marker Collection",
    description="Select the Well Tops folder to extract.",
    object_type=DomainObjectsEnum.MarkerCollection,
    select_multiple=False,
    parameter_group="2. Optional Data"
)
# End: PWR Description

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ==============================================================================
#  LOGGING SETUP
# ==============================================================================
def setup_logger():
    if not constants.LOGS_DIR.exists():
        constants.LOGS_DIR.mkdir(parents=True)
    
    # Cleanup old logs
    log_files = sorted(constants.LOGS_DIR.glob("session_*.log"), key=os.path.getmtime)
    while len(log_files) >= 5:
        os.remove(log_files.pop(0))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = constants.LOGS_DIR / f"session_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"--- New Session Started: {timestamp} ---")
    return logging.getLogger()

logger = setup_logger()


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _get_well_attribute_value(well, *attr_names, default=0.0):
    try:
        attrs = getattr(well, "attributes", {}) or {}
    except Exception:
        attrs = {}

    for name in attr_names:
        try:
            if name in attrs and attrs[name] is not None:
                attr_obj = attrs[name]
                value = getattr(attr_obj, "value", attr_obj)
                return _safe_float(value, default=default)
        except Exception:
            continue

    return _safe_float(default)

# ==============================================================================
#  HELPER: KILL PROCESS
# ==============================================================================
def kill_process_on_port(port):
    try:
        for con in psutil.net_connections(kind='inet'):
            if not con.laddr or con.laddr.port != port or con.pid is None:
                continue
            proc = psutil.Process(con.pid)
            logger.info(f"Killing process {proc.name()} (PID: {proc.pid})")
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except psutil.TimeoutExpired:
                proc.kill()
            return
    except Exception:
        return

try:
    kill_process_on_port(constants.APP_PORT)
    time.sleep(1)
except: pass

# ==============================================================================
#  DATA EXTRACTION
# ==============================================================================
logger.info("--- 1. Connecting to Petrel ---")

try:
    if 'parameters' not in locals() and 'parameters' not in globals():
        # Dummy data for testing outside Petrel
        input_guids = []
        dt_guid = None
        tops_guid = None
    else:
        input_guids = parameters.get('selected_wells', [])
        dt_guid = parameters.get('sel_dt_global', None)
        tops_guid = parameters.get('sel_marker_coll', None)
except NameError:
    input_guids = []

if not input_guids:
    logger.error("CRITICAL: No wells selected. Aborting.")
    sys.exit(0)

try:
    from cegalprizm.pythontool import PetrelConnection
    ptp = PetrelConnection(allow_experimental=True)
    project_name = ptp.get_current_project_name()
    logger.info(f"Connected to: {project_name}")
except Exception as e:
    logger.critical(f"Failed to connect to Petrel. {e}")
    sys.exit(1)

logger.info("--- 2. Extracting Data ---")

# Resolve Optional Objects
target_dt_name = None
if dt_guid:
    try:
        dt_obj = ptp.get_petrelobjects_by_guids([dt_guid])[0]
        target_dt_name = dt_obj.petrel_name
        logger.info(f"Target DT Log: {target_dt_name}")
    except: pass

target_tops_coll = None
if tops_guid:
    try:
        target_tops_coll = ptp.get_petrelobjects_by_guids([tops_guid])[0]
        logger.info(f"Target Tops: {target_tops_coll.petrel_name}")
    except: pass

# Containers
well_headers = []
checkshot_data = []
survey_data = []
log_data = []
tops_data = []

selected_wells = ptp.get_petrelobjects_by_guids(input_guids)
all_checkshots = [cs for cs in ptp.checkshots]

wells_with_cs = 0

for well in selected_wells:
    logger.info(f"Processing well: {well.petrel_name}")
    
    # 1. Header
    try:
        stats = well.retrieve_stats()
        def get_stat_float(key_part):
            for k, v in stats.items():
                if key_part in k:
                    try: return float(v)
                    except: pass
            return 0.0

        wh_coords = well.wellhead_coordinates if well.wellhead_coordinates else (0.0, 0.0)
        kb_val = 0.0
        if well.well_datum:
            try: kb_val = well.well_datum[1] 
            except: pass

        gl_val = _get_well_attribute_value(well, "Ground Level", "GL", default=0.0)
        srd_val = _get_well_attribute_value(well, "Seismic Reference Datum", "SRD", default=0.0)
        df_val = _get_well_attribute_value(well, "Drill Floor", "DF", default=0.0)
        se_val = _get_well_attribute_value(well, "Source Elevation", "SE", default=0.0)
        so_val = _get_well_attribute_value(well, "Source Offset", "SO", default=0.0)
        vc_val = _get_well_attribute_value(well, "Replacement Velocity", "Vc", default=5000.0)

        well_headers.append({
            "WellName": str(well.petrel_name),
            "UWI": str(getattr(well, 'uwi', '')),
            "GUID": str(well.droid),
            "X": float(wh_coords[0]),
            "Y": float(wh_coords[1]),
            "Lat": get_stat_float('Lat Min'),
            "Long": get_stat_float('Long Min'),
            "KB": float(kb_val),
            "GL": float(gl_val),
            "SRD": float(srd_val),
            "DF": float(df_val),
            "SE": float(se_val),
            "SO": float(so_val),
            "Vc": float(vc_val),
        })
    except: pass

    # 2. Checkshots
    has_cs = False
    for cs in all_checkshots:
        try:
            linked_wells = cs.get_wells()
            if any(w.droid == well.droid for w in linked_wells):
                df_cs = cs.as_dataframe(include_unconnected_checkshots=False, wells_filter=[well])
                if not df_cs.empty:
                    df_cs = df_cs.fillna(0)
                    cols = [c for c in df_cs.columns if c in ['MD', 'TWT', 'TVD', 'Average Velocity', 'Interval Velocity', 'Z']]
                    df_s = df_cs[cols].copy()
                    if 'MD' in df_s.columns:
                        df_s['MD'] = pd.to_numeric(df_s['MD'], errors='coerce')
                    if 'TWT' in df_s.columns:
                        df_s['TWT'] = pd.to_numeric(df_s['TWT'], errors='coerce')
                    df_s['WellName'] = str(well.petrel_name)
                    df_s['CheckshotName'] = str(cs.petrel_name)
                    df_s = df_s.dropna(subset=[c for c in ['MD', 'TWT'] if c in df_s.columns])
                    checkshot_data.append(df_s)
                    has_cs = True
        except: continue
    
    if has_cs: wells_with_cs += 1

    # 3. Surveys (MD, Inc, Azim)
    try:
        # Try to get definitive, or MD Incl Azim, or first available
        survey = None
        for s in well.surveys:
            if getattr(s, 'is_definitive', False):
                survey = s
                break
        
        if not survey and well.surveys:
            survey = list(well.surveys)[0]
            
        if survey:
            df_srv = survey.as_dataframe()
            if not df_srv.empty:
                # Standardize columns
                # Petrel usually gives 'MD', 'Inclination', 'Azimuth GN' (or similar)
                cols_map = {c: c for c in df_srv.columns}
                for c in df_srv.columns:
                    if 'Azimuth' in c: cols_map[c] = 'Azimuth'
                
                df_srv = df_srv.rename(columns=cols_map)
                keep = [c for c in df_srv.columns if c in ['MD', 'Inclination', 'Azimuth', 'TVD', 'X', 'Y', 'Z']]
                df_srv = df_srv[keep].copy()
                df_srv['WellName'] = str(well.petrel_name)
                survey_data.append(df_srv)
    except Exception as e:
        logger.warning(f"Survey error {well.petrel_name}: {e}")

    # 4. DT Log
    if target_dt_name:
        try:
            # Find log in well matching global name
            target_logs = [l for l in well.logs if l.petrel_name == target_dt_name]
            if target_logs:
                l_obj = target_logs[0]
                df_l = well.logs_dataframe([l_obj])
                # Ensure MD column
                if 'MD' in df_l.columns:
                    # Rename the value column to generic 'DT'
                    df_l = df_l.rename(columns={target_dt_name: 'DT'})
                    df_l = df_l[['MD', 'DT']].dropna()
                    df_l['WellName'] = str(well.petrel_name)
                    log_data.append(df_l)
        except Exception as e:
            logger.warning(f"Log error {well.petrel_name}: {e}")

    # 5. Tops (Markers)
    if target_tops_coll:
        try:
            df_t = target_tops_coll.as_dataframe(wells_filter=[well], include_petrel_index=False)
            if not df_t.empty:
                # Rename standard columns
                renames = {}
                for c in df_t.columns:
                    if 'Surface' in c: renames[c] = 'Surface'
                    if 'MD' in c: renames[c] = 'MD'
                    if 'TVD' in c: renames[c] = 'TVD' # Might be 'Z'
                
                df_t = df_t.rename(columns=renames)
                req = ['Surface', 'MD']
                if all(r in df_t.columns for r in req):
                    keep_cols = ['Surface', 'MD'] + (['TVD'] if 'TVD' in df_t.columns else [])
                    df_t = df_t[keep_cols].copy()
                    df_t['MD'] = pd.to_numeric(df_t['MD'], errors='coerce')
                    if 'TVD' in df_t.columns:
                        df_t['TVD'] = pd.to_numeric(df_t['TVD'], errors='coerce')
                    df_t = df_t.dropna(subset=['MD'])
                    df_t['WellName'] = str(well.petrel_name)
                    tops_data.append(df_t)
        except Exception as e:
            logger.warning(f"Tops error {well.petrel_name}: {e}")

if wells_with_cs == 0:
    logger.error("CRITICAL: No Checkshot data found for selected wells.")
    sys.exit(0)

# --- 3. SAVE TO DUCKDB ---
logger.info("--- 3. Saving Data to DuckDB ---")

try:
    if constants.DATA_FILE.exists():
        try: os.remove(constants.DATA_FILE)
        except: pass
    
    con = duckdb.connect(str(constants.DATA_FILE))
    
    # Headers
    df_h = pd.DataFrame(well_headers)
    con.execute("CREATE TABLE headers AS SELECT * FROM df_h")
    
    # Checkshots
    if checkshot_data:
        df_cs_all = pd.concat(checkshot_data, ignore_index=True)
        con.execute("CREATE TABLE checkshots AS SELECT * FROM df_cs_all")
    else:
        con.execute("CREATE TABLE checkshots (WellName VARCHAR, CheckshotName VARCHAR, MD DOUBLE, TWT DOUBLE, TVD DOUBLE, \"Average Velocity\" DOUBLE, \"Interval Velocity\" DOUBLE, Z DOUBLE)")

    # Surveys
    if survey_data:
        df_srv_all = pd.concat(survey_data, ignore_index=True)
        con.execute("CREATE TABLE surveys AS SELECT * FROM df_srv_all")
    else:
        con.execute("CREATE TABLE surveys (WellName VARCHAR, MD DOUBLE, Inclination DOUBLE)")

    # Logs (DT)
    if log_data:
        df_log_all = pd.concat(log_data, ignore_index=True)
        con.execute("CREATE TABLE dt_logs AS SELECT * FROM df_log_all")
    else:
        con.execute("CREATE TABLE dt_logs (WellName VARCHAR, MD DOUBLE, DT DOUBLE)")

    # Tops
    if tops_data:
        df_tops_all = pd.concat(tops_data, ignore_index=True)
        con.execute("CREATE TABLE tops AS SELECT * FROM df_tops_all")
    else:
        con.execute("CREATE TABLE tops (WellName VARCHAR, Surface VARCHAR, MD DOUBLE, TVD DOUBLE)")
    
    # Meta
    con.execute(f"CREATE TABLE meta AS SELECT '{project_name}' as ProjectName, current_timestamp as ExtractedAt")
    
    con.close()
    logger.info(f"Data saved successfully.")

except Exception as e:
    logger.critical(f"Error saving data: {e}")
    sys.exit(1)

# ==============================================================================
#  LAUNCH PANEL
# ==============================================================================
app_root = constants.APP_ROOT_PATH
panel_script_name = "checkShot_main.py"
panel_script_path = str(app_root / panel_script_name)

logger.info(f"--- 4. Launching: {panel_script_path} ---")

env_vars = os.environ.copy()
env_vars["PWR_DATA_FILE"] = str(constants.DATA_FILE)
env_vars["PYTHONPATH"] = f"{str(app_root)}{os.pathsep}{env_vars.get('PYTHONPATH', '')}"

# Clean env
for key in ["IMPERSONATION_ID", "CEGAL_AUTH_TOKEN"]:
    if key in env_vars: env_vars.pop(key, None)

process = subprocess.Popen(
    [
        sys.executable, "-u", "-m", "panel", "serve", 
        panel_script_name, 
        "--allow-websocket-origin=*", 
        "--port", str(constants.APP_PORT), "--address", "127.0.0.1"
    ],
    stdout=subprocess.PIPE, 
    stderr=subprocess.STDOUT, 
    text=True,
    env=env_vars,
    cwd=str(app_root),
    bufsize=1 
)

def log_reader(proc):
    try:
        for line in proc.stdout:
            if line.strip():
                logger.info(f"[APP]: {line.strip()}")
    except: pass

reader_thread = threading.Thread(target=log_reader, args=(process,), daemon=True)
reader_thread.start()

logger.info("--- Waiting for Port 5006... ---")

url = f"http://127.0.0.1:{constants.APP_PORT}/checkShot_main"
server_ready = False
start_time = time.time()

while time.time() - start_time < 60: 
    if process.poll() is not None:
        logger.error("\n!!! ERROR: Server process died unexpectedly !!!")
        break

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex(('127.0.0.1', constants.APP_PORT)) == 0:
                server_ready = True
                break
    except: pass
    time.sleep(1) 

if server_ready:
    logger.info(f"\nSUCCESS: App is active!")
    webbrowser.open(url)
    while process.poll() is None:
        time.sleep(1)
else:
    logger.error("\n!!! TIMEOUT: Port 5006 never opened. !!!")

if process.poll() is None:
    process.terminate()

# Working code Feb, 12th, 2026 | 11:22
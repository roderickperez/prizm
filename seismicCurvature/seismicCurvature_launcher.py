# Start: PWR Description
import sys
import os
import json
import tempfile
import subprocess
import time
import webbrowser
import warnings
import psutil
from pathlib import Path

# --- 1. SETUP PATH TO LOCAL CONSTANTS ---
current_dir = r"N:\_USER_GLOBAL\PETREL\Prizm\wf\Vienna\seismicCurvature_Panel_app"

if current_dir not in sys.path:
    sys.path.append(current_dir)

# --- 2. IMPORT CONSTANTS ---
try:
    import seismicCurvature_constants as constants
    from importlib import reload
    reload(constants)
except ImportError as e:
    raise ImportError(f"PWR Block failed to load constants. Error: {e}")

from cegalprizm.pycoderunner import WorkflowDescription
from cegalprizm.pythontool import DomainObjectsEnum, PetrelConnection

# --- 3. DEFINE WORKFLOW ---
pwr_description = WorkflowDescription(
    name=constants.WF_NAME,
    category=constants.WF_CATEGORY,
    description=constants.WF_DESCRIPTION,
    authors=constants.WF_AUTHORS,
    version=constants.WF_VERSION
)

# --- ADD SEISMIC SELECTOR ---
pwr_description.add_object_ref_parameter(
    name="seismic_input",
    label="Seismic Volume",
    description="Select ONE seismic cube to analyze",
    object_type=DomainObjectsEnum.SeismicCube,
    select_multiple=False
)
# End: PWR Description

# --- 4. Suppress Warnings ---
warnings.filterwarnings("ignore", category=UserWarning, module='cegalprizm')
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ==============================================================================
#  HELPER: KILL OLD SERVER
# ==============================================================================
def kill_process_on_port(port):
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for con in proc.connections(kind='inet'):
                if con.laddr.port == port:
                    print(f"--- Killing old process on port {port} (PID: {proc.pid}) ---")
                    proc.terminate()
                    return
        except: continue

try:
    kill_process_on_port(5007)
    time.sleep(1)
except: pass

# ==============================================================================
#  DATA EXTRACTION (Runs INSIDE Petrel)
# ==============================================================================
print("--- 1. Connecting to Petrel ---")

selected_cube_name = "None"
selected_cube_guid = None  
project_name = "Unknown"

try:
    ptp = PetrelConnection(allow_experimental=True)
    project_name = ptp.get_current_project_name()
    
    # Retrieve user selection from PWR 'parameters' dict
    if 'parameters' in locals() and 'seismic_input' in parameters:
        selected_cube_guid = parameters['seismic_input'] 
        
        # Fetch actual object to get the name
        objs = ptp.get_petrelobjects_by_guids([selected_cube_guid])
        if objs:
            cube = objs[0]
            selected_cube_name = cube.petrel_name
            print(f"Selected Cube: {selected_cube_name} (GUID: {selected_cube_guid})")
            
except Exception as e:
    print(f"Error extracting seismic data: {e}")

# --- 3. Save Data Snapshot ---
snapshot_payload = {
    "project": project_name,
    "selected_cube_name": selected_cube_name,
    "selected_cube_guid": selected_cube_guid
}

try:
    with open(constants.DATA_FILE, "w") as f:
        json.dump(snapshot_payload, f)
    print(f"Data saved to: {constants.DATA_FILE}")
except Exception as e:
    print(f"Error saving JSON: {e}")

# ==============================================================================
#  LAUNCH PANEL
# ==============================================================================
app_root = constants.APP_ROOT_PATH
panel_script = str(app_root / "seismicCurvature_main.py")

print(f"--- 4. Attempting to launch: {panel_script} ---")

env_vars = os.environ.copy()
env_vars["PWR_DATA_FILE"] = str(constants.DATA_FILE)

for key in ["IMPERSONATION_ID", "CEGAL_AUTH_TOKEN"]:
    if key in env_vars: env_vars.pop(key, None)

process = subprocess.Popen(
    [
        sys.executable, "-m", "panel", "serve", 
        panel_script, "--dev", "--allow-websocket-origin=*", "--port", "5007"
    ],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    env=env_vars, creationflags=subprocess.CREATE_NO_WINDOW
)

print("Waiting for server to initialize...")
time.sleep(3)

if process.poll() is not None:
    print("!!! ERROR: Server crashed !!!")
    stdout, stderr = process.communicate()
    print(stderr)
else:
    print(f"SUCCESS: {constants.WF_NAME} is running.")
    url = "http://localhost:5007/seismicCurvature_main"
    print(f"Opening: {url}")
    webbrowser.open(url)

    # --- KEEP ALIVE LOOP ---
    print("\n--- APP LOGS (Close 'Stop' in Petrel to exit) ---")
    try:
        for line in process.stdout: print(line, end='')
    except: pass
    finally: process.terminate()
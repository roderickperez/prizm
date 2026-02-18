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
current_dir = r"N:\_USER_GLOBAL\PETREL\Prizm\wf\Vienna\faultSeg_Panel"

if current_dir not in sys.path:
    sys.path.append(current_dir)

# --- 2. IMPORT CONSTANTS ---
try:
    import faultSeg_constants
    from importlib import reload
    reload(faultSeg_constants)
except ImportError as e:
    raise ImportError(f"PWR Block failed to load faultSeg_constants.py from {current_dir}. Error: {e}")

from cegalprizm.pycoderunner import WorkflowDescription
from cegalprizm.pythontool import DomainObjectsEnum, PetrelConnection

# --- 3. DEFINE WORKFLOW ---
pwr_description = WorkflowDescription(
    name=faultSeg_constants.WF_NAME,
    category=faultSeg_constants.WF_CATEGORY,
    description=faultSeg_constants.WF_DESCRIPTION,
    authors=faultSeg_constants.WF_AUTHORS,
    version=faultSeg_constants.WF_VERSION
)

# --- ADD SEISMIC SELECTOR ---
pwr_description.add_object_ref_parameter(
    name="seismic_input",
    label="Seismic Volume",
    description="Select the seismic cube to export for segmentation",
    object_type=DomainObjectsEnum.SeismicCube,
    select_multiple=False
)
# End: PWR Description

# --- 4. Suppress Warnings ---
warnings.filterwarnings("ignore", category=UserWarning, module='cegalprizm')
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ==============================================================================
#  HELPER: KILL OLD SERVER (Fixes NameError)
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
    kill_process_on_port(5006)
    time.sleep(1)
except: pass

# ==============================================================================
#  DATA EXTRACTION LOGIC
# ==============================================================================
print("--- 1. Connecting to Petrel ---")

seismic_name = "No Volume Selected"
selected_cube_guid = None
project_name = "Unknown Project"

try:
    # 1. Connect to Petrel
    ptp = PetrelConnection(allow_experimental=True)
    project_name = ptp.get_current_project_name()
    
    # 2. Retrieve User Selection
    if 'parameters' in locals() and 'seismic_input' in parameters:
        # 'seismic_input' is the GUID string from PWR
        guid_val = parameters['seismic_input']
        
        if guid_val:
            selected_cube_guid = guid_val
            # Get the object using the GUID provided by the UI
            selected_objects = ptp.get_petrelobjects_by_guids([selected_cube_guid])
            if selected_objects:
                cube = selected_objects[0]
                seismic_name = cube.petrel_name
                print(f"Selected Seismic Cube: {seismic_name}")
            
except Exception as e:
    print(f"Warning during data extraction: {e}")

# ==============================================================================
#  FEATURE A: VALIDATION CHECK
# ==============================================================================
if not selected_cube_guid:
    print("\n" + "!"*60)
    print("CRITICAL ERROR: NO SEISMIC VOLUME SELECTED")
    print("!"*60)
    print("You must select a seismic volume in the workflow dialog before running.")
    print("The application will NOT launch.")
    print("!"*60 + "\n")
    # Exit with error code to stop execution
    sys.exit(1)

# 3. Save Data to JSON for Panel App
data_payload = {
    "project": project_name,
    "seismic_name": seismic_name,
    "selected_cube_guid": selected_cube_guid
}

try:
    with open(faultSeg_constants.DATA_FILE, "w") as f:
        json.dump(data_payload, f)
    print(f"Data saved to: {faultSeg_constants.DATA_FILE}")
except Exception as e:
    print(f"Error saving JSON: {e}")

# ==============================================================================
#  LAUNCH PANEL
# ==============================================================================
app_root = faultSeg_constants.APP_ROOT_PATH
panel_script = str(app_root / "faultSeg_main.py")

print(f"--- 4. Attempting to launch: {panel_script} ---")

# Setup Env Vars
env_vars = os.environ.copy()
# Pass the JSON path to the Panel app via Environment Variable
env_vars["PWR_DATA_FILE"] = str(faultSeg_constants.DATA_FILE)

for key in ["IMPERSONATION_ID", "CEGAL_AUTH_TOKEN"]:
    if key in env_vars: env_vars.pop(key, None)

process = subprocess.Popen(
    [
        sys.executable, "-m", "panel", "serve", 
        panel_script, "--dev", "--allow-websocket-origin=*", "--port", "5006"
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
    print(f"SUCCESS: {faultSeg_constants.WF_NAME} is running.")
    url = "http://localhost:5006/faultSeg_main"
    print(f"Opening: {url}")
    webbrowser.open(url)

    # --- KEEP ALIVE LOOP ---
    print("\n--- APP LOGS (Close 'Stop' in Petrel to exit) ---")
    try:
        for line in process.stdout: print(line, end='')
    except: pass
    finally: process.terminate()
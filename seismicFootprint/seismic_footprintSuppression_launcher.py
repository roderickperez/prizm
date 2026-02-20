# Start: PWR Description
import sys
import os
import json
import subprocess
import time
import webbrowser
import warnings
import psutil 
import socket
import threading

# --- 1. SETUP PATH TO LOCAL CONSTANTS ---
current_dir = r"N:\_USER_GLOBAL\PETREL\Prizm\wf\Vienna\seismic_footprintSuppression_Panel_app"

if current_dir not in sys.path:
    sys.path.append(current_dir)

# --- 2. IMPORT CONSTANTS ---
try:
    import seismic_footprintSuppression_constants as constants
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

pwr_description.add_object_ref_parameter(
    name="seismic_input",
    label="Seismic Volume",
    description="Select ONE seismic cube to analyze",
    object_type=DomainObjectsEnum.SeismicCube,
    select_multiple=False
)
# End: PWR Description

warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
#  HELPER: KILL OLD SERVER
# ==============================================================================
def kill_process_on_port(port):
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for con in proc.connections(kind='inet'):
                if con.laddr.port == port:
                    print(f"--- Killing zombie process on port {port} (PID: {proc.pid}) ---")
                    proc.terminate()
                    return
        except: continue

try:
    kill_process_on_port(5006)
    time.sleep(2)
except: pass 

# ==============================================================================
#  DATA EXTRACTION
# ==============================================================================
print("--- 1. Connecting to Petrel ---")
from cegalprizm.pythontool import PetrelConnection

selected_cube_name = "None"
selected_cube_guid = None
project_name = "Unknown"

try:
    ptp = PetrelConnection(allow_experimental=True)
    project_name = ptp.get_current_project_name()
    print(f"Connected to: {project_name}")

    if 'parameters' in locals() and 'seismic_input' in parameters:
        selected_cube_guid = parameters['seismic_input']
        objs = ptp.get_petrelobjects_by_guids([selected_cube_guid])
        if objs:
            cube = objs[0]
            selected_cube_name = cube.petrel_name
            print(f"Selected Cube: {selected_cube_name}")

except Exception as e:
    print(f"CRITICAL: Error extracting seismic data. {e}")

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
    print(f"Error saving data: {e}")

# ==============================================================================
#  LAUNCH PANEL (THREADED MODE)
# ==============================================================================
app_root = constants.APP_ROOT_PATH
panel_script = str(app_root / "seismic_footprintSuppression_main.py")

print(f"--- 4. Attempting to launch: {panel_script} ---")

env_vars = os.environ.copy()
env_vars["PWR_DATA_FILE"] = str(constants.DATA_FILE)

# --- CRITICAL FIX: ADD 'PAGES' TO PYTHONPATH ---
# This ensures background workers can find 'seismic_compute.py'
pages_dir = str(app_root)
current_pypath = env_vars.get("PYTHONPATH", "")
env_vars["PYTHONPATH"] = f"{pages_dir}{os.pathsep}{current_pypath}"
# -----------------------------------------------

for key in ["IMPERSONATION_ID", "CEGAL_AUTH_TOKEN"]:
    if key in env_vars: env_vars.pop(key, None)

# Fix: Force IPv4 binding
process = subprocess.Popen(
    [
        sys.executable, "-u", "-m", "panel", "serve", 
        panel_script, "--dev", "--allow-websocket-origin=*", 
        "--port", "5006", "--address", "127.0.0.1"
    ],
    stdout=subprocess.PIPE, 
    stderr=subprocess.STDOUT, 
    text=True,
    env=env_vars,
    bufsize=1 
)

# --- THREADED LOG READER ---
def log_reader(proc):
    try:
        for line in proc.stdout:
            print(f"[Panel]: {line.strip()}")
    except: pass

reader_thread = threading.Thread(target=log_reader, args=(process,), daemon=True)
reader_thread.start()

# --- MAIN THREAD: PORT CHECKER ---
print("--- Waiting for Port 5006 (Checking 127.0.0.1)... ---")

url = "http://127.0.0.1:5006/seismic_footprintSuppression_main"
server_ready = False
start_time = time.time()

while time.time() - start_time < 60: # 60s timeout
    if process.poll() is not None:
        print("\n!!! ERROR: Server process died unexpectedly !!!")
        break

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex(('127.0.0.1', 5006)) == 0:
                server_ready = True
                break
    except: pass
    
    time.sleep(1) 

if server_ready:
    print(f"\nSUCCESS: Port 5006 is active!")
    print(f"Opening: {url}")
    webbrowser.open(url)
    
    print("\n--- APP LOGS STREAM ---")
    while process.poll() is None:
        time.sleep(1)
else:
    print("\n!!! TIMEOUT: Port 5006 never opened. Check logs above. !!!")

if process.poll() is None:
    process.terminate()
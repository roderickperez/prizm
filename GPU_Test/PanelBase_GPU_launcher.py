# Start: PWR Description
import sys
import os
import subprocess
import time
import webbrowser
import psutil 
import socket
import threading
import warnings

# --- 1. SETUP PATH TO LOCAL CONSTANTS ---
current_dir = r"N:\_USER_GLOBAL\PETREL\Prizm\wf\Vienna\PanelBase_GPU_app"

if current_dir not in sys.path:
    sys.path.append(current_dir)

# --- 2. IMPORT CONSTANTS ---
try:
    import PanelBase_GPU_constants as constants
    from importlib import reload
    reload(constants)
except ImportError as e:
    # Fallback if constants file is missing
    class constants:
        APP_ROOT_PATH = os.path.join(current_dir, 'pages')
        WF_NAME = "GPU Diagnostic"
        WF_CATEGORY = "Panel"
        WF_DESCRIPTION = "GPU Check"
        WF_AUTHORS = "User"
        WF_VERSION = "v1.0"

# --- 3. DEFINE WORKFLOW ---
# This import and definition MUST stay inside the PWR Description block
try:
    from cegalprizm.pycoderunner import WorkflowDescription
    
    pwr_description = WorkflowDescription(
        name=constants.WF_NAME,
        category=constants.WF_CATEGORY,
        description=constants.WF_DESCRIPTION,
        authors=constants.WF_AUTHORS,
        version=constants.WF_VERSION
    )
except ImportError:
    # Fallback for testing outside Petrel
    pwr_description = None 
    print("WARNING: cegalprizm module not found. Running in standalone mode.")

# End: PWR Description

# --- 4. Suppress Warnings ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ==============================================================================
#  HELPER: KILL OLD SERVER
# ==============================================================================
def kill_process_on_port(port):
    """Finds and kills any process locking the dashboard port."""
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for con in proc.connections(kind='inet'):
                if con.laddr.port == port:
                    print(f"--- Killing zombie process on port {port} (PID: {proc.pid}) ---")
                    proc.terminate()
                    return
        except: continue

# USE PORT 5007 to avoid conflict with your other apps on 5006
PORT = 5007

try:
    kill_process_on_port(PORT)
    time.sleep(1) 
except: pass 

# ==============================================================================
#  STEP 0: GPU PRE-FLIGHT CHECK (Terminal Verification)
# ==============================================================================
print("\n" + "="*60)
print("Running GPU Pre-flight Check...")
print("="*60)
try:
    # This checks the environment immediately in the terminal
    check_cmd = [
        sys.executable, "-c", 
        "import torch; print('  Torch:', torch.__version__); "
        "print('  CUDA Available:', torch.cuda.is_available()); "
        "print('  CUDA Version:', torch.version.cuda); "
        "print('  GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
    ]
    subprocess.run(check_cmd, check=True)
except Exception as e:
    print(f"!!! WARNING: GPU Check Failed: {e}")
print("="*60 + "\n")

# ==============================================================================
#  LAUNCH PANEL (ROBUST THREADED MODE)
# ==============================================================================
app_root = constants.APP_ROOT_PATH
panel_script_name = "PanelBase_GPU_main.py"
panel_script_path = os.path.join(app_root, panel_script_name)

if not os.path.exists(panel_script_path):
    print(f"CRITICAL ERROR: Could not find main app at: {panel_script_path}")
    # We do not exit here to avoid crashing the workflow runner immediately
else:
    print(f"--- Launching App: {panel_script_path} ---")

    env_vars = os.environ.copy()
    # Add pages to pythonpath to ensure relative imports work if needed
    env_vars["PYTHONPATH"] = f"{str(app_root)}{os.pathsep}{env_vars.get('PYTHONPATH', '')}"

    # Clean Cegal Auth tokens from env to prevent conflicts in subprocess
    for key in ["IMPERSONATION_ID", "CEGAL_AUTH_TOKEN"]:
        if key in env_vars: env_vars.pop(key, None)

    process = subprocess.Popen(
        [
            sys.executable, "-u", "-m", "panel", "serve", 
            panel_script_name, 
            "--dev", "--allow-websocket-origin=*", 
            "--port", str(PORT), "--address", "127.0.0.1"
        ],
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True,
        env=env_vars,
        cwd=str(app_root),
        bufsize=1,
        creationflags=subprocess.CREATE_NO_WINDOW
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
    print(f"--- Waiting for Port {PORT} (Checking 127.0.0.1)... ---")

    url = f"http://127.0.0.1:{PORT}/PanelBase_GPU_main"
    server_ready = False
    start_time = time.time()

    while time.time() - start_time < 60: 
        if process.poll() is not None:
            print("\n!!! ERROR: Server process died unexpectedly !!!")
            break

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                if s.connect_ex(('127.0.0.1', PORT)) == 0:
                    server_ready = True
                    break
        except: pass
        time.sleep(1) 

    if server_ready:
        print(f"\nSUCCESS: Port {PORT} is active!")
        print(f"Opening: {url}")
        webbrowser.open(url)
        
        print("\n--- APP LOGS STREAM (Close 'Stop' in Petrel to exit) ---")
        try:
            while process.poll() is None:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    else:
        print(f"\n!!! TIMEOUT: Port {PORT} never opened. Check logs above. !!!")

    if process.poll() is None:
        process.terminate()
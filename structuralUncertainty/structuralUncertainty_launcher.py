# Start: PWR Description
import json
import os
import subprocess
import sys
import time
import warnings
import webbrowser

import psutil

# --- 1. SETUP PATH TO LOCAL CONSTANTS ---
current_dir = r"N:\_USER_GLOBAL\PETREL\Prizm\wf\Vienna\structuralUncertainty_Panel_app"

if current_dir not in sys.path:
	sys.path.append(current_dir)

# --- 2. IMPORT CONSTANTS ---
try:
	import structuralUncertainty_constants as constants
	from importlib import reload

	reload(constants)
except ImportError as e:
	raise ImportError(f"PWR Block failed to load constants from {current_dir}. Error: {e}")

from cegalprizm.pycoderunner import WorkflowDescription

# --- 3. DEFINE WORKFLOW ---
pwr_description = WorkflowDescription(
	name=constants.WF_NAME,
	category=constants.WF_CATEGORY,
	description=constants.WF_DESCRIPTION,
	authors=constants.WF_AUTHORS,
	version=constants.WF_VERSION,
)
# End: PWR Description

# --- 4. Suppress Warnings ---
warnings.filterwarnings("ignore", category=UserWarning, module="cegalprizm")
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ==============================================================================
#  HELPER: KILL OLD SERVER
# ==============================================================================
def kill_process_on_port(port):
	for proc in psutil.process_iter(["pid", "name"]):
		try:
			for con in proc.connections(kind="inet"):
				if con.laddr.port == port:
					print(f"--- Killing old process on port {port} (PID: {proc.pid}) ---")
					proc.terminate()
					return
		except Exception:
			continue


try:
	kill_process_on_port(5006)
	time.sleep(1)
except Exception:
	pass


# ==============================================================================
#  DATA SNAPSHOT
# ==============================================================================
project_name = "Unknown"
snapshot_payload = {
	"project": project_name,
	"module": "structuralUncertainty",
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
panel_script = str(app_root / "structuralUncertainty.py")

print(f"--- Attempting to launch: {panel_script} ---")

env_vars = os.environ.copy()
env_vars["PWR_DATA_FILE"] = str(constants.DATA_FILE)

for key in ["IMPERSONATION_ID", "CEGAL_AUTH_TOKEN"]:
	if key in env_vars:
		env_vars.pop(key, None)

creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

process = subprocess.Popen(
	[
		sys.executable,
		"-m",
		"panel",
		"serve",
		panel_script,
		"--dev",
		"--allow-websocket-origin=*",
		"--port",
		"5006",
	],
	stdout=subprocess.PIPE,
	stderr=subprocess.PIPE,
	text=True,
	env=env_vars,
	creationflags=creationflags,
)

print("Waiting for server to initialize...")
time.sleep(3)

if process.poll() is not None:
	print("!!! ERROR: Server crashed !!!")
	stdout, stderr = process.communicate()
	print(stderr)
else:
	print(f"SUCCESS: {constants.WF_NAME} is running.")
	url = "http://localhost:5006/structuralUncertainty"
	print(f"Opening: {url}")
	webbrowser.open(url)

	print("\n--- APP LOGS (Close 'Stop' in Petrel to exit) ---")
	try:
		for line in process.stdout:
			print(line, end="")
	except Exception:
		pass
	finally:
		process.terminate()

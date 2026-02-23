# Start: PWR Description
import json
import os
import subprocess
import sys
import time
import warnings
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

# --- 1. SETUP PATH TO LOCAL CONSTANTS ---
current_dir = str(Path(__file__).resolve().parent)

if current_dir not in sys.path:
	sys.path.append(current_dir)

# --- 2. IMPORT CONSTANTS ---
try:
	import structuralUncertainty_constants_V2 as constants
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
	version=constants.WF_VERSION,
)

pwr_description.add_object_ref_parameter(
	name="input_surface",
	label="Input Surface",
	description="Select one Petrel surface to import into the Structural Uncertainty panel",
	object_type=DomainObjectsEnum.Surface,
	select_multiple=False,
)
# End: PWR Description

# --- 4. Suppress Warnings ---
warnings.filterwarnings("ignore", category=UserWarning, module="cegalprizm")
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ==============================================================================
#  HELPER: KILL OLD SERVER
# ==============================================================================
def kill_process_on_port(port):
	try:
		for con in psutil.net_connections(kind="inet"):
			if not con.laddr or con.laddr.port != port or con.pid is None:
				continue
			proc = psutil.Process(con.pid)
			print(f"--- Killing old process on port {port} (PID: {proc.pid}) ---")
			proc.terminate()
			try:
				proc.wait(timeout=3)
			except psutil.TimeoutExpired:
				proc.kill()
			return
	except Exception:
		pass


try:
	kill_process_on_port(constants.APP_PORT)
	time.sleep(1)
except Exception:
	pass


# ==============================================================================
#  DATA SNAPSHOT
# ==============================================================================
project_name = "Unknown"
surface_payload = None
selected_surface_guid = None
selected_surface_name = "None"


def _surface_df_to_payload(df: pd.DataFrame, max_nodes: int = 90000) -> dict | None:
	if df is None or df.empty:
		return None

	for col in ["X", "Y", "Z"]:
		if col not in df.columns:
			return None

	df = df[["X", "Y", "Z"]].copy()
	df["X"] = pd.to_numeric(df["X"], errors="coerce")
	df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
	df["Z"] = pd.to_numeric(df["Z"], errors="coerce")
	df = df.dropna(subset=["X", "Y", "Z"])
	if df.empty:
		return None

	x_unique = np.sort(df["X"].unique())
	y_unique = np.sort(df["Y"].unique())
	if len(x_unique) < 2 or len(y_unique) < 2:
		return None

	pivot = df.pivot_table(index="Y", columns="X", values="Z", aggfunc="mean")
	pivot = pivot.reindex(index=y_unique, columns=x_unique)

	if pivot.isna().all().all():
		return None

	grid = pivot.interpolate(axis=0, limit_direction="both").interpolate(axis=1, limit_direction="both")
	grid = grid.fillna(float(grid.stack().mean()) if not grid.stack().empty else 0.0)

	ny, nx = grid.shape
	total_nodes = nx * ny
	if total_nodes > max_nodes:
		scale = (max_nodes / float(total_nodes)) ** 0.5
		target_nx = max(20, int(nx * scale))
		target_ny = max(20, int(ny * scale))
		x_idx = np.linspace(0, nx - 1, target_nx).astype(int)
		y_idx = np.linspace(0, ny - 1, target_ny).astype(int)
		grid = grid.iloc[y_idx, x_idx]

	x_vals = grid.columns.to_numpy(dtype=float)
	y_vals = grid.index.to_numpy(dtype=float)
	z_vals = grid.to_numpy(dtype=float)

	return {
		"x": x_vals.tolist(),
		"y": y_vals.tolist(),
		"z": z_vals.tolist(),
	}


try:
	from cegalprizm.pythontool import PetrelConnection

	ptp = PetrelConnection(allow_experimental=True)
	project_name = ptp.get_current_project_name()

	if "parameters" in locals() and parameters.get("input_surface"):
		selected_surface_guid = parameters.get("input_surface")
		objs = ptp.get_petrelobjects_by_guids([selected_surface_guid])
		if objs:
			surf_obj = objs[0]
			selected_surface_name = str(getattr(surf_obj, "petrel_name", "SelectedSurface"))
			df_surface = None
			for kwargs in ({"dropna": True}, {}):
				try:
					df_surface = surf_obj.as_dataframe(**kwargs)
					if isinstance(df_surface, pd.DataFrame) and not df_surface.empty:
						break
				except Exception:
					continue
			surface_payload = _surface_df_to_payload(df_surface)
except Exception:
	pass

snapshot_payload = {
	"project": project_name,
	"module": "structuralUncertainty_V2",
	"selected_surface_guid": selected_surface_guid,
	"selected_surface_name": selected_surface_name,
	"imported_surface": surface_payload,
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
panel_script = str(app_root / "structuralUncertainty_V2.py")

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
		"--allow-websocket-origin=*",
		"--port",
		str(constants.APP_PORT),
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
	url = f"http://localhost:{constants.APP_PORT}/structuralUncertainty_V2"
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

# Start: PWR Description
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import warnings
import webbrowser
from pathlib import Path

import psutil


current_dir = str(Path(__file__).resolve().parent)
if current_dir not in sys.path:
	sys.path.append(current_dir)

import CWDB2Studio_constants as constants

try:
	from cegalprizm.pycoderunner import WorkflowDescription

	pwr_description = WorkflowDescription(
		name=constants.WF_NAME,
		category=constants.WF_CATEGORY,
		description=constants.WF_DESCRIPTION,
		authors=constants.WF_AUTHORS,
		version=constants.WF_VERSION,
	)
except Exception:
	pwr_description = None
# End: PWR Description


warnings.filterwarnings("ignore", category=UserWarning, module="cegalprizm")
warnings.filterwarnings("ignore", category=DeprecationWarning)


def kill_process_on_port(port: int) -> None:
	try:
		for con in psutil.net_connections(kind="inet"):
			if not con.laddr or con.laddr.port != port or con.pid is None:
				continue
			proc = psutil.Process(con.pid)
			proc.terminate()
			try:
				proc.wait(timeout=2)
			except psutil.TimeoutExpired:
				proc.kill()
			return
	except Exception:
		return


kill_process_on_port(constants.APP_PORT)
time.sleep(1)

project_name = "Unknown"
petrel_available = False
try:
	from cegalprizm.pythontool import PetrelConnection

	ptp = PetrelConnection(allow_experimental=True)
	project_name = ptp.get_current_project_name()
	petrel_available = True
except Exception:
	pass

snapshot_payload = {
	"project": project_name,
	"petrel_available": petrel_available,
	"test_csv_path": str(constants.TEST_CSV_PATH),
	"temp_db_path": str(constants.TEMP_DB_FILE),
}
constants.SNAPSHOT_FILE.write_text(json.dumps(snapshot_payload, indent=2), encoding="utf-8")

env_vars = os.environ.copy()
env_vars["PWR_DATA_FILE"] = str(constants.SNAPSHOT_FILE)
env_vars["CWDB2_TEMP_DB_FILE"] = str(constants.TEMP_DB_FILE)

creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

process = subprocess.Popen(
	[
		sys.executable,
		"-m",
		"panel",
		"serve",
		str(constants.PANEL_ENTRYPOINT),
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

time.sleep(3)

if process.poll() is None:
	url = f"http://localhost:{constants.APP_PORT}/CWDB2Studio"
	webbrowser.open(url)
	try:
		for line in process.stdout:
			print(line, end="")
	except Exception:
		pass
	finally:
		process.terminate()
else:
	_, stderr = process.communicate()
	print(stderr)


# Start: PWR Description
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

import ProjectEDA_constants as constants

from cegalprizm.pycoderunner import WorkflowDescription
from cegalprizm.pythontool import DomainObjectsEnum, PetrelConnection

pwr_description = WorkflowDescription(
    name=constants.WF_NAME,
    category=constants.WF_CATEGORY,
    description=constants.WF_DESCRIPTION,
    authors=constants.WF_AUTHORS,
    version=constants.WF_VERSION,
)

pwr_description.add_enum_parameter(
    name="scope_mode",
    label="Scope",
    description="Extract data for selected wells only, or for all wells in project.",
    options={1: "Selected wells", 2: "All wells in project"},
    default_value=2,
)

pwr_description.add_object_ref_parameter(
    name="well_guids",
    label="Wells",
    description="Optional well selection used when Scope=Selected wells.",
    object_type=DomainObjectsEnum.Well,
    select_multiple=True,
)
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
try:
    ptp = PetrelConnection(allow_experimental=True)
    project_name = ptp.get_current_project_name()
except Exception:
    pass

pwr_parameters = globals().get("parameters", {}) or {}
scope_mode = int(pwr_parameters.get("scope_mode", 2))
well_guids = pwr_parameters.get("well_guids", [])

global_log_names = []
selections_payload = {
    "project_name": project_name,
    "scope_mode": scope_mode,
    "well_guids": list(well_guids or []),
    "global_log_names": global_log_names,
}

constants.SELECTIONS_FILE.write_text(json.dumps(selections_payload, indent=2), encoding="utf-8")

panel_script = str(constants.PANEL_ENTRYPOINT)

env_vars = os.environ.copy()
env_vars["PROJECT_EDA_SELECTIONS_FILE"] = str(constants.SELECTIONS_FILE)

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

time.sleep(3)

if process.poll() is None:
    url = f"http://localhost:{constants.APP_PORT}/ProjectEDA_main"
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

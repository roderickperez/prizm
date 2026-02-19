# N:\_USER_GLOBAL\PETREL\Prizm\wf\Vienna\seismicFlowLines_2_5D_Panel_app\seismicFlowLines_2_5D_constants.py

from pathlib import Path
import tempfile

# --- Petrel Workflow Metadata ---
WF_NAME = "Seismic FlowLines 2.5D"
WF_CATEGORY = "Seismic"
WF_DESCRIPTION = "Panel App Launcher for Seismic FlowLines 2.5D"
WF_AUTHORS = "roderick.perezaltamar@omv.com"
WF_VERSION = "v1.0"

# --- Path to the App ---
# Points to the 'pages' folder containing the main script
# Updated path to the new app folder
APP_ROOT_PATH = Path(r"N:\_USER_GLOBAL\PETREL\Prizm\wf\Vienna\seismicFlowLines_2_5D_Panel_app\pages")

# --- Temp Data File ---
# Shared path for data exchange between Petrel and Panel
# Updated filename to avoid conflict with Attributes app
DATA_FILE = Path(tempfile.gettempdir()) / "seismic_flowlines_2_5d_data.json"

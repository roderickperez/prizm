from pathlib import Path
import tempfile

# --- Petrel Workflow Metadata ---
WF_NAME = "Seismic Fault Seg"
WF_CATEGORY = "Seismic"
WF_DESCRIPTION = "Panel App Launcher"
WF_AUTHORS = "roderick.perezaltamar@omv.com"
WF_VERSION = "v1"

# --- Path to the App ---
APP_ROOT_PATH = Path(r"N:\_USER_GLOBAL\PETREL\Prizm\wf\Vienna\faultSeg_Panel\pages")

# --- Temp Data File ---
# Shared path for data exchange between Petrel and Panel
DATA_FILE = Path(tempfile.gettempdir()) / "faultseg_data.json"
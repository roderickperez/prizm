from pathlib import Path
import tempfile

# --- Petrel Workflow Metadata ---
WF_NAME = "Seismic Footprint Suppression"
WF_CATEGORY = "Panel"
WF_DESCRIPTION = "Panel App for Seismic Footprint Suppression"
WF_AUTHORS = "roderick.perezaltamar@omv.com"
WF_VERSION = "v1.0"

# --- Path to the App ---
# Points to the 'pages' folder containing the main script
APP_ROOT_PATH = Path(r"N:\_USER_GLOBAL\PETREL\Prizm\wf\Vienna\seismic_footprintSuppression_Panel_app\pages")

# --- Temp Data File ---
# Shared path for data exchange between Petrel and Panel
DATA_FILE = Path(tempfile.gettempdir()) / "seismic_footprint_data.json"
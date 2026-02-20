from pathlib import Path
import tempfile

# --- Petrel Workflow Metadata ---
WF_NAME = "Seismic Curvature"
WF_CATEGORY = "Seismic"
WF_DESCRIPTION = "Gradient Structure Tensor & Curvature Analysis with Shape Index"
WF_AUTHORS = "roderick.perezaltamar@omv.com"
WF_VERSION = "v1.2"

# --- Path to the App ---
# Points to the 'pages' folder containing the main script
APP_ROOT_PATH = Path(r"N:\_USER_GLOBAL\PETREL\Prizm\wf\Vienna\seismicCurvature_Panel_app\pages")

# --- Temp Data File ---
# Shared path for data exchange between Petrel and Panel
DATA_FILE = Path(tempfile.gettempdir()) / "seismic_curvature_data.json"
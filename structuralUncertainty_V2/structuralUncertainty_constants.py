from pathlib import Path
import os
import tempfile

# --- Petrel Workflow Metadata ---
WF_NAME = "Structural Uncertainty"
WF_CATEGORY = "Seismic"
WF_DESCRIPTION = "Panel App Launcher for Structural Uncertainty Evaluation"
WF_AUTHORS = "roderick.perezaltamar@omv.com"
WF_VERSION = "v1.0"

# --- Path to the App ---
LOCAL_ROOT = Path(__file__).resolve().parent
ROOT_PATH = Path(os.environ.get("PRIZM_STRUCTURAL_ROOT", str(LOCAL_ROOT)))
APP_ROOT_PATH = ROOT_PATH / "pages"

# --- Temp Data File ---
DATA_FILE = Path(tempfile.gettempdir()) / "structural_uncertainty_data.json"

# --- App Port ---
APP_PORT = int(os.environ.get("PRIZM_STRUCTURAL_PORT", "5006"))

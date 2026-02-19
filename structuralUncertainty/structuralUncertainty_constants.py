from pathlib import Path
import tempfile

# --- Petrel Workflow Metadata ---
WF_NAME = "Structural Uncertainty"
WF_CATEGORY = "Seismic"
WF_DESCRIPTION = "Panel App Launcher for Structural Uncertainty Evaluation"
WF_AUTHORS = "roderick.perezaltamar@omv.com"
WF_VERSION = "v1.0"

# --- Path to the App ---
APP_ROOT_PATH = Path(r"N:\_USER_GLOBAL\PETREL\Prizm\wf\Vienna\structuralUncertainty_Panel_app\pages")

# --- Temp Data File ---
DATA_FILE = Path(tempfile.gettempdir()) / "structural_uncertainty_data.json"

# -*- coding: utf-8 -*-
from pathlib import Path
import os
import tempfile

# --- Petrel Workflow Metadata ---
WF_NAME = "Structural Uncertainty V2"
WF_CATEGORY = "Seismic"
WF_DESCRIPTION = "Panel App Launcher for Structural Uncertainty Evaluation V2"
WF_AUTHORS = "roderick.perezaltamar@omv.com"
WF_VERSION = "v2.0"

# --- Path to the App ---
try:
    LOCAL_ROOT = Path(__file__).resolve().parent
except NameError:
    # Fallback for Cegal Prizm runtime where __file__ is undefined
    LOCAL_ROOT = Path(r"N:\_USER_GLOBAL\PETREL\Prizm\wf\Vienna\structuralUncertainty")

ROOT_PATH = Path(os.environ.get("PRIZM_STRUCTURAL_ROOT", str(LOCAL_ROOT)))
APP_ROOT_PATH = ROOT_PATH / "pages"

# --- Temp Data File ---
DATA_FILE = Path(tempfile.gettempdir()) / "structural_uncertainty_v2_data.json"

# --- App Port ---
APP_PORT = int(os.environ.get("PRIZM_STRUCTURAL_PORT", "5006"))
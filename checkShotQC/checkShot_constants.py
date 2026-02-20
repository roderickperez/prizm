from pathlib import Path
import os
import tempfile

# --- Petrel Workflow Metadata ---
WF_NAME = "CheckShot QC"
WF_CATEGORY = "Seismic"
WF_DESCRIPTION = "Extracts checkshot data for QC"
WF_AUTHORS = "roderick.perezaltamar@omv.com"
WF_VERSION = "v2.0"

# --- Paths ---
LOCAL_ROOT = Path(__file__).resolve().parent
ROOT_PATH = Path(os.environ.get("PRIZM_CHECKSHOT_ROOT", str(LOCAL_ROOT)))
APP_ROOT_PATH = ROOT_PATH / "pages"

# --- New: Logs Directory ---
LOGS_DIR = ROOT_PATH / "logs"

# --- Data File (DuckDB) ---
DATA_FILE = Path(tempfile.gettempdir()) / "checkshot_data.duckdb"

# --- App Config ---
APP_PORT = int(os.environ.get("PRIZM_CHECKSHOT_PORT", "5006"))

# Working code Feb, 12th, 2026 | 10:22
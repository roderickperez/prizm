from pathlib import Path
import tempfile

WF_NAME = "Project EDA"
WF_CATEGORY = "Panel"
WF_DESCRIPTION = "Project-level EDA for wells, logs and tops (Panel + DuckDB)."
WF_AUTHORS = "roderick.perezaltamar@omv.com"
WF_VERSION = "v1.0"

PROJECT_ROOT = Path(__file__).resolve().parent
PANEL_ROOT = PROJECT_ROOT / "panel"
PANEL_ENTRYPOINT = PANEL_ROOT / "ProjectEDA_main.py"

ASSETS_DIR = PROJECT_ROOT.parent / "images" / "OMV_brandLogoPack" / "OMV_brandLogoPack"
LOGO_PATH = ASSETS_DIR / "OMV_logo_Neon_Small.png"
FAVICON_PATH = ASSETS_DIR / "OMV_logo_Blue_Small.png"

APP_PORT = 5017
SELECTIONS_FILE = Path(tempfile.gettempdir()) / "project_eda_selections.json"
SNAPSHOT_FILE = Path(tempfile.gettempdir()) / "project_eda_snapshot.json"
DUCKDB_FILE = Path(tempfile.gettempdir()) / "project_eda.duckdb"

MAX_POINTS_PER_LOG = 15000

from pathlib import Path
import tempfile


WF_NAME = "CWDB2Studio"
WF_CATEGORY = "Panel"
WF_DESCRIPTION = "CWDB CSV loader, QC map/statistics and Petrel well export."
WF_AUTHORS = "roderick.perezaltamar@omv.com"
WF_VERSION = "v1.0"

PROJECT_ROOT = Path(__file__).resolve().parent
PAGES_ROOT = PROJECT_ROOT / "pages"
PANEL_ENTRYPOINT = PAGES_ROOT / "CWDB2Studio.py"

APP_PORT = 5018

ASSETS_DIR = PROJECT_ROOT.parent / "images" / "OMV_brandLogoPack" / "OMV_brandLogoPack"
LOGO_PATH = ASSETS_DIR / "OMV_logo_Neon_Small.png"
FAVICON_PATH = ASSETS_DIR / "OMV_logo_Blue_Small.png"

SNAPSHOT_FILE = Path(tempfile.gettempdir()) / "cwdb2studio_snapshot.json"
TEMP_DB_FILE = Path(tempfile.gettempdir()) / "cwdb2studio_temp.sqlite"

TEST_CSV_PATH = PROJECT_ROOT.parent / "referenceDocumentation" / "CWDB" / "HR_Data_Prizm_WS.csv"


# N:\_USER_GLOBAL\PETREL\Prizm\wf\1_Wells_EDA\constants.py

from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent

# --- App metadata ---
APP_VERSION = "0.4"

# --- Assets ---
# Put a logo at:  N:\_USER_GLOBAL\PETREL\Prizm\wf\1_Wells_EDA\assets\logo.png
_LOGO_CANDIDATES = [
    APP_ROOT / "assets" / "logo.png",
    APP_ROOT / "logo.png",
]

LOGO_PATH = ""

for p in _LOGO_CANDIDATES:
    if p.exists():
        LOGO_PATH = str(p)
        break

# --- Data files (used by Gantt, etc.) ---
TIMELINE_FILE = str(APP_ROOT / "timeline.json")

# --- Explicit Overrides ---
LOGO_PATH = r"N:\_USER_GLOBAL\PETREL\Prizm\wf\images\OMV_brandLogoPack\OMV_brandLogoPack\OMV_logo_RGB_Deep-Blue.png"
APP_VERSION = "v0.1.0"
TIMELINE_FILE = "timeline_data.json"
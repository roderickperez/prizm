import panel as pn
import pandas as pd
import os
import json

# --- App UI Constants ---
APP_TITLE = "Seismic RGT"
ACCENT_COLOR = "#052759"

# --- Image Paths ---
LOGO_PATH = r"N:\_USER_GLOBAL\PETREL\Prizm\wf\Vienna\images\OMV_brandLogoPack\OMV_brandLogoPack\OMV_logo_Neon_Small.png"
FAVICON_PATH = r"N:\_USER_GLOBAL\PETREL\Prizm\wf\Vienna\images\OMV_brandLogoPack\OMV_brandLogoPack\OMV_logo_Blue_Small.png"

# --- Load Data (Backend Logic) ---
# We load the data here so it is available, even if we don't display it yet.
df_wells = pd.DataFrame()
project_name = "Unknown"

data_file = os.environ.get("PWR_DATA_FILE")
if data_file and os.path.exists(data_file):
    try:
        with open(data_file, "r") as f:
            data = json.load(f)
            project_name = data.get("project", "Unknown")
            df_wells = pd.DataFrame(data.get("wells", []))
    except Exception as e:
        print(f"Error loading data: {e}")

# --- Verify Images ---
valid_logo = LOGO_PATH if os.path.exists(LOGO_PATH) else None
valid_favicon = FAVICON_PATH if os.path.exists(FAVICON_PATH) else None

# --- Initialize Panel ---
pn.extension('tabulator')

# --- Create Template ---
# Empty content as requested, just showing the title and extracted project stats
main_content = pn.Column(
    pn.pane.Markdown(f"# {APP_TITLE}"),
    pn.pane.Markdown(f"**Connected Project:** {project_name}"),
    pn.pane.Markdown(f"**Wells Extracted:** {len(df_wells)}")
)

template = pn.template.FastListTemplate(
    title=APP_TITLE,
    logo=valid_logo,
    favicon=valid_favicon,
    accent_base_color=ACCENT_COLOR,
    header_background=ACCENT_COLOR,
    sidebar=[
        pn.pane.Markdown("### Navigation"), 
        pn.pane.Markdown("Selection"), 
        pn.pane.Markdown("Settings")
    ],
    main=[main_content],
)

template.servable()
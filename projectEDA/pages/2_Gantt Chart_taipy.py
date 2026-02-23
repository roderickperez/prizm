from taipy.gui import Gui
import plotly.express as px
import pandas as pd
import os
import json
import sys
import base64
from pathlib import Path

# --- FIX 1: Add parent directory to path to find utils.py ---
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

import utils

# --- 2. SETUP & DATA LOADING ---

# Define Assets
logo_path = r"N:\_USER_GLOBAL\PETREL\Prizm\wf\1_Wells_EDA\assets\logo.png"

# Helper: Convert Image to Base64 for Browser Display
def load_image_as_b64(path):
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
                return f"data:image/png;base64,{data}"
        except Exception as e:
            print(f"Error loading image: {e}")
    return ""

# Load the logo immediately
logo_data = load_image_as_b64(logo_path)

# Connect to Petrel
petrel_project = utils.get_petrel_connection()

if petrel_project:
    # 1. Get Wells (Scoped)
    wells = utils.get_all_wells_flat_scoped(petrel_project)
    
    # 2. Get Tops
    tops = utils.load_tops_dataframe(petrel_project, "taipy_init")

    # 3. Calculate Metrics
    num_wells = len(wells)
    
    unique_log_names = set()
    for well in wells:
        for log in utils.iter_selected_logs(well):  
            unique_log_names.add(getattr(log, "petrel_name", ""))
    num_logs = len(unique_log_names)

    num_unique_tops = int(tops["Surface"].nunique()) if not tops.empty else 0
    
    # 4. Generate Map Data
    geo_df = utils.get_well_min_lat_long(wells, "taipy_init")
    valid_geo = geo_df.dropna(subset=["latitude", "longitude"])
    
    # 5. Create Plotly Figure
    if valid_geo.empty:
        fig = px.scatter_mapbox(lat=[], lon=[])
        fig.update_layout(title="No coordinates available")
    else:
        fig = px.scatter_mapbox(
            valid_geo,
            lat="latitude",
            lon="longitude",
            hover_name="Well Name",
            zoom=5,
            height=600,
            mapbox_style="open-street-map",
        )
        fig.update_traces(marker=dict(size=12, color="#00872A"))
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
else:
    # Fallback
    num_wells = 0
    num_logs = 0
    num_unique_tops = 0
    fig = px.scatter_mapbox(lat=[], lon=[])
    fig.update_layout(title="Petrel Connection Failed")


# --- 3. GUI PAGES ---

# A. Root Page (The Sidebar Layout)
# The <|content|> tag is where the "Summary" page will appear.
root_md = """
<|layout|columns=300px 1|
    <|part|class_name=sidebar|
        <|{logo_data}|image|width=200px|class_name=logo_style|>
        <|br|>
        <|menu|selector=True|> 
    |>

    <|part|class_name=main_content|
        <|content|>
    |>
|>
"""

# B. Summary Page (The Dashboard Content)
summary_md = """
<|container|
# Project **Summary** {: .header_style}

<|hr|>

<|layout|columns=1 1 1|class_name=kpi_row|
    <|container kpi_card|
### **Wells**
## <|{num_wells}|>
    |>

    <|container kpi_card|
### **Logs**
## <|{num_logs}|>
    |>

    <|container kpi_card|
### **Tops**
## <|{num_unique_tops}|>
    |>
|>

<|part|render={num_logs==0}|class_name=info_box|
**Info:** No logs selected in the launcher (Scope Mode 1).
|>

<|part|render={num_unique_tops==0}|class_name=info_box|
**Info:** No marker collections selected in the launcher.
|>

### Well **Locations**
<|chart|figure={fig}|height=600px|>

|>
"""

# Register pages
# "/" is the root (Sidebar), "Project_Summary" is the home page content
pages = {
    "/": root_md,
    "Project_Summary": summary_md
}

# --- 4. STYLING ---

stylekit = {
    "color_primary": "#00872A",
    "color_secondary": "#D5D6D6",
    "color_paper_dark": "#04235B",
    "font_family": "Arial",
}

# CSS for Sidebar and Cards
css = """
/* Sidebar Styling */
.sidebar {
    background-color: #f0f2f6;
    height: 100vh;
    padding: 20px;
    border-right: 1px solid #d5d6d6;
}

.logo_style {
    display: block;
    margin-bottom: 20px;
}

.main_content {
    padding: 20px;
}

/* Header & KPIs */
.header_style h1 { 
    color: #04235B; 
    margin-top: 0px;
}
.kpi_row {
    gap: 20px;
    margin-bottom: 30px;
}
.kpi_card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 8px;
    border: 1px solid #e0e0e0;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
.kpi_card h3 {
    color: #666;
    margin: 0;
    font-size: 1.2em;
}
.kpi_card h2 {
    color: #00872A;
    margin: 10px 0 0 0;
    font-size: 2.5em;
    font-weight: bold;
}
.info_box {
    background-color: #e3f2fd;
    color: #0d47a1;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 10px;
}
"""

# --- 5. EXECUTION ---
if __name__ == "__main__":
    # Note: We pass 'pages' instead of 'page' now
    gui = Gui(pages=pages, css_file=None, re) 
    
    # Save CSS to temp file
    css_path = parent_dir / "taipy_style.css"
    with open(css_path, "w") as f:
        f.write(css)

    gui.run(
        stylekit=stylekit,
        css_file=str(css_path),
        title="Well EDA",
        port=8088,
        dark_mode=False,
        use_reloader=True 
    )
import panel as pn
import os
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from shared.ui.omv_theme import (
    BLUE_OMV_COLOR,
    DARK_BLUE_OMV_COLOR,
    get_content_text_color,
    NEON_OMV_COLOR,
    docs_button_html,
    get_dark_select_stylesheets,
    get_extension_raw_css,
    get_main_outer_background,
    get_slider_stylesheets,
    is_dark_mode_from_state,
)

APP_TITLE = "Panel Base"
DOCUMENTATION_URL = "https://example.com/docs"

ASSETS_DIR = ROOT_DIR / "images" / "OMV_brandLogoPack" / "OMV_brandLogoPack"
LOGO_PATH = ASSETS_DIR / "OMV_logo_Neon_Small.png"
FAVICON_PATH = ASSETS_DIR / "OMV_logo_Blue_Small.png"

wells_count = 0
project_name = "Unknown"

data_file = os.environ.get("PWR_DATA_FILE")
if data_file and os.path.exists(data_file):
    try:
        with open(data_file, "r") as f:
            data = json.load(f)
            project_name = data.get("project", "Unknown")
            wells_count = len(data.get("wells", []))
    except Exception as e:
        print(f"Error loading data: {e}")

is_dark_mode = is_dark_mode_from_state()

select_stylesheets = get_dark_select_stylesheets(is_dark_mode)
slider_stylesheets = get_slider_stylesheets()

pn.extension("tabulator", raw_css=get_extension_raw_css(is_dark_mode))



def make_placeholder_select() -> pn.widgets.Select:
    return pn.widgets.Select(
        name="Placeholder Dropdown",
        options=["Option A", "Option B", "Option C"],
        value="Option A",
        sizing_mode="stretch_width",
        stylesheets=select_stylesheets,
    )



def make_placeholder_slider() -> pn.widgets.FloatSlider:
    return pn.widgets.FloatSlider(
        name="Placeholder Slider",
        start=0,
        end=100,
        value=50,
        sizing_mode="stretch_width",
        stylesheets=slider_stylesheets,
    )


section_header_background = NEON_OMV_COLOR
section_body_background = DARK_BLUE_OMV_COLOR if is_dark_mode else "white"
section_text_color = get_content_text_color(is_dark_mode)

main_content = pn.Column(
    pn.pane.Markdown(f"# {APP_TITLE}"),
    pn.pane.Markdown(f"**Connected Project:** {project_name}"),
    pn.pane.Markdown(f"**Wells Extracted:** {wells_count}"),
    pn.Spacer(sizing_mode="stretch_both"),
    sizing_mode="stretch_both",
    margin=0,
    styles={
        "height": "100%",
        "overflow": "hidden",
        "background": get_main_outer_background(is_dark_mode),
        "color": section_text_color if is_dark_mode else "inherit",
    },
)

template_kwargs = dict(
    title=APP_TITLE,
    accent_base_color=BLUE_OMV_COLOR,
    header_background=DARK_BLUE_OMV_COLOR,
    main_layout=None,
    main_max_width="",
    main=[main_content],
)

section_1_card = pn.Card(
    make_placeholder_slider(),
    make_placeholder_select(),
    title="Section 1",
    collapsed=True,
    hide_header=False,
    sizing_mode="stretch_width",
    header_background=section_header_background,
    active_header_background=section_header_background,
    header_color=section_text_color,
    styles={"background": section_body_background, "color": section_text_color},
    margin=(0, 0, 12, 0),
)

section_2_card = pn.Card(
    make_placeholder_slider(),
    make_placeholder_select(),
    title="Section 2",
    collapsed=True,
    hide_header=False,
    sizing_mode="stretch_width",
    header_background=section_header_background,
    active_header_background=section_header_background,
    header_color=section_text_color,
    styles={"background": section_body_background, "color": section_text_color},
)

template_kwargs["sidebar"] = [section_1_card, section_2_card]

template_kwargs["header"] = [
    pn.Row(
        pn.Spacer(sizing_mode="stretch_width"),
        pn.pane.HTML(docs_button_html(DOCUMENTATION_URL)),
        sizing_mode="stretch_width",
        margin=0,
    )
]

valid_logo = str(LOGO_PATH) if LOGO_PATH.exists() else None
valid_favicon = str(FAVICON_PATH) if FAVICON_PATH.exists() else None

if valid_logo:
    template_kwargs["logo"] = valid_logo
if valid_favicon:
    template_kwargs["favicon"] = valid_favicon

template = pn.template.FastListTemplate(**template_kwargs)
template.servable()

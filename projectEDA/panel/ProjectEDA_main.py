from __future__ import annotations

import sys
from pathlib import Path

import panel as pn

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from projectEDA import ProjectEDA_constants as constants
from projectEDA.panel.pages import (
    gantt_chart,
    global_logs,
    log_standardization,
    multi_well_plot,
    project_summary,
    single_log,
    well_tops,
    wells,
)
from projectEDA.panel.services.data_service import ProjectEDADataService
from shared.ui.omv_theme import (
    BLUE_OMV_COLOR,
    DARK_BLUE_OMV_COLOR,
    NEON_OMV_COLOR,
    docs_button_html,
    get_extension_raw_css,
    is_dark_mode_from_state,
)


is_dark_mode = is_dark_mode_from_state()

_PROJECT_EDA_SIDEBAR_CSS = """
#sidebar .projecteda-nav-card {
    margin-bottom: 12px !important;
}

#sidebar .projecteda-nav-card .bk-card-header,
#sidebar .projecteda-nav-card .bk-panel-models-layout-Card-header {
    padding-top: 10px !important;
    padding-bottom: 10px !important;
    font-weight: 600 !important;
}

#sidebar .projecteda-nav-card .bk-card-title,
#sidebar .projecteda-nav-card .bk-panel-models-layout-Card-title {
    letter-spacing: 0.2px;
}

#sidebar .projecteda-nav-card .bk-card-body,
#sidebar .projecteda-nav-card .bk-panel-models-layout-Card-body {
    min-height: 8px;
}
"""

pn.extension(
    "tabulator",
    "plotly",
    raw_css=[*get_extension_raw_css(is_dark_mode), _PROJECT_EDA_SIDEBAR_CSS],
)


def _is_test_mode() -> bool:
    args = [a.lower() for a in getattr(sys, "argv", [])]
    return "--test" in args or "test" in args


TEST_MODE = _is_test_mode()

service = ProjectEDADataService(constants.DUCKDB_FILE, test_mode=TEST_MODE)
service.refresh_from_petrel(force=False)

main_content = pn.Column(sizing_mode="stretch_both")


PAGES = {
    "Project Summary": project_summary.build_page,
    "Log Standardization": log_standardization.build_page,
    "Gantt Chart": gantt_chart.build_page,
    "Wells": wells.build_page,
    "Global Logs": global_logs.build_page,
    "Single Log": single_log.build_page,
    "Multi Well Plot": multi_well_plot.build_page,
    "Well Tops": well_tops.build_page,
}


def set_page(page_name: str) -> None:
    builder = PAGES[page_name]
    main_content[:] = [builder(service)]


status = pn.pane.Markdown("", sizing_mode="stretch_width")
if TEST_MODE:
    status.object = "🧪 Test mode enabled: running without Petrel data."


def refresh_data(_=None):
    if TEST_MODE:
        service.refresh_from_petrel(force=True)
        status.object = "🧪 Test mode enabled: data refresh uses empty test dataset."
        set_page("Project Summary")
        return

    service.refresh_from_petrel(force=True)
    status.object = "✅ Data refreshed from Petrel and saved to DuckDB."
    set_page("Project Summary")


refresh_btn = pn.widgets.Button(
    name="Refresh Petrel → DuckDB" if not TEST_MODE else "Refresh Test Dataset",
    button_type="primary",
)
refresh_btn.on_click(refresh_data)

section_body_background = DARK_BLUE_OMV_COLOR if is_dark_mode else "white"
section_body_text = "white" if is_dark_mode else DARK_BLUE_OMV_COLOR

_nav_update_lock = {"locked": False}


def _make_nav_card(page_name: str) -> pn.Card:
    card = pn.Card(
        pn.pane.Markdown(" "),
        title=page_name,
        collapsed=True,
        hide_header=False,
        sizing_mode="stretch_width",
        header_background=NEON_OMV_COLOR,
        active_header_background=NEON_OMV_COLOR,
        header_color=DARK_BLUE_OMV_COLOR,
        styles={"background": section_body_background, "color": section_body_text},
        margin=(0, 0, 12, 0),
        css_classes=["projecteda-nav-card"],
    )

    def _on_toggle(event):
        if _nav_update_lock["locked"]:
            return
        if event.new is False:
            set_page(page_name)
            _nav_update_lock["locked"] = True
            card.collapsed = True
            _nav_update_lock["locked"] = False

    card.param.watch(_on_toggle, "collapsed")
    return card


nav_cards = [_make_nav_card(name) for name in PAGES]


sidebar_items = [
    pn.pane.Markdown("## Sidebar"),
    refresh_btn,
    status,
    *nav_cards,
]

set_page("Project Summary")


template_kwargs = dict(
    title="Project EDA",
    accent_base_color=BLUE_OMV_COLOR,
    header_background=DARK_BLUE_OMV_COLOR,
    main_layout=None,
    main_max_width="",
    main=[main_content],
    sidebar=sidebar_items,
    header=[pn.pane.HTML(docs_button_html("https://example.com/docs"))],
)

if constants.LOGO_PATH.exists():
    template_kwargs["logo"] = str(constants.LOGO_PATH)
if constants.FAVICON_PATH.exists():
    template_kwargs["favicon"] = str(constants.FAVICON_PATH)


template = pn.template.FastListTemplate(**template_kwargs)
template.servable()

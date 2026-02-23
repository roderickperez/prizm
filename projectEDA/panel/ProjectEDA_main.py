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
pn.extension("tabulator", "plotly", raw_css=get_extension_raw_css(is_dark_mode))

service = ProjectEDADataService(constants.DUCKDB_FILE)
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


def refresh_data(_=None):
    service.refresh_from_petrel(force=True)
    status.object = "✅ Data refreshed from Petrel and saved to DuckDB."
    set_page("Project Summary")


refresh_btn = pn.widgets.Button(name="Refresh Petrel → DuckDB", button_type="primary")
refresh_btn.on_click(refresh_data)


accordion = pn.Accordion(
    *[(name, pn.pane.Markdown(" ")) for name in PAGES],
    active=[],
    sizing_mode="stretch_width",
)


def _on_sidebar_section(event):
    active = event.new or []
    if not active:
        return
    idx = active[0]
    names = list(PAGES.keys())
    if idx < len(names):
        set_page(names[idx])


accordion.param.watch(_on_sidebar_section, "active")


sidebar_items = [
    pn.pane.Markdown("## Sidebar"),
    refresh_btn,
    status,
    accordion,
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

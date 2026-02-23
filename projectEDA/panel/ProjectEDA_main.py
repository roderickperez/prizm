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

#sidebar .projecteda-nav-link {
    display: block;
    width: 100%;
    box-sizing: border-box;
    text-decoration: none;
    background: #39e75f;
    color: #003056;
    border: 1px solid #39e75f;
    font-weight: 600;
    text-align: left;
    border-radius: 4px;
    padding: 10px 12px;
    margin-bottom: 10px;
}

#sidebar .projecteda-nav-link:hover {
    filter: brightness(0.95);
}

#sidebar .projecteda-nav-link.active {
    outline: 2px solid #003056;
    outline-offset: 1px;
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

PAGE_SLUGS = {
    "Project Summary": "project-summary",
    "Log Standardization": "log-standardization",
    "Gantt Chart": "gantt-chart",
    "Wells": "wells",
    "Global Logs": "global-logs",
    "Single Log": "single-log",
    "Multi Well Plot": "multi-well-plot",
    "Well Tops": "well-tops",
}
SLUG_TO_PAGE = {slug: page for page, slug in PAGE_SLUGS.items()}


def _initial_page_from_session() -> str:
    try:
        args = getattr(pn.state, "session_args", {}) or {}
        raw = args.get("page", [b""])
        if not raw:
            return "Project Summary"
        token = raw[0]
        slug = token.decode("utf-8") if isinstance(token, (bytes, bytearray)) else str(token)
        return SLUG_TO_PAGE.get(slug, "Project Summary")
    except Exception:
        return "Project Summary"


active_page_name = _initial_page_from_session()
active_page = pn.widgets.StaticText(name="Active Page", value=active_page_name)


def set_page(page_name: str) -> None:
    if page_name not in PAGES:
        status.object = f"❌ Unknown page: {page_name}"
        return

    try:
        builder = PAGES[page_name]
        main_content[:] = [builder(service)]
        active_page.value = page_name
        if TEST_MODE:
            status.object = f"🧪 Test mode enabled: viewing '{page_name}'."
        else:
            status.object = f"✅ Showing page: {page_name}"
    except Exception as exc:
        status.object = f"❌ Failed to open '{page_name}': {exc}"


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

def _make_nav_link(page_name: str) -> pn.pane.HTML:
    slug = PAGE_SLUGS.get(page_name, "")
    active_cls = " active" if page_name == active_page_name else ""
    return pn.pane.HTML(
        f"<a class='projecteda-nav-link{active_cls}' href='?page={slug}'>{page_name}</a>",
        sizing_mode="stretch_width",
        margin=0,
    )


nav_links = [_make_nav_link(name) for name in PAGES]


sidebar_items = [
    pn.pane.Markdown("## Sidebar"),
    refresh_btn,
    status,
    active_page,
    *nav_links,
]

set_page(active_page_name)


template_kwargs = dict(
    title="Project EDA",
    accent_base_color=BLUE_OMV_COLOR,
    header_background=DARK_BLUE_OMV_COLOR,
    main_layout=None,
    main_max_width="",
    main=[main_content],
    sidebar=sidebar_items,
    header=[
        pn.Row(
            pn.Spacer(sizing_mode="stretch_width"),
            pn.pane.HTML(docs_button_html("https://example.com/docs")),
            sizing_mode="stretch_width",
            margin=0,
        )
    ],
)

if constants.LOGO_PATH.exists():
    template_kwargs["logo"] = str(constants.LOGO_PATH)
if constants.FAVICON_PATH.exists():
    template_kwargs["favicon"] = str(constants.FAVICON_PATH)


template = pn.template.FastListTemplate(**template_kwargs)
template.servable()

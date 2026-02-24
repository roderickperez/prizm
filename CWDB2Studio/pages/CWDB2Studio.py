from __future__ import annotations

import io
import json
import os
import sqlite3
import sys
from datetime import date, datetime
from pathlib import Path

import duckdb
import hvplot.pandas  # noqa: F401
import numpy as np
import pandas as pd
import panel as pn
import plotly.express as px

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

CWDB_DIR = Path(__file__).resolve().parents[1]
if str(CWDB_DIR) not in sys.path:
    sys.path.insert(0, str(CWDB_DIR))

import CWDB2Studio_constants as constants

from shared.ui.omv_theme import (
    BLUE_OMV_COLOR,
    DARK_BLUE_OMV_COLOR,
    NEON_OMV_COLOR,
    docs_button_html,
    get_content_text_color,
    get_dark_select_stylesheets,
    get_extension_raw_css,
    get_main_outer_background,
    get_neon_button_stylesheets,
    get_slider_stylesheets,
    is_dark_mode_from_state,
)

APP_TITLE = "CWDB2Studio"
DOCUMENTATION_URL = "https://example.com/docs"

is_dark_mode = is_dark_mode_from_state()
select_stylesheets = get_dark_select_stylesheets(is_dark_mode)
slider_stylesheets = get_slider_stylesheets()

pn.extension("tabulator", "plotly", "bokeh", raw_css=get_extension_raw_css(is_dark_mode))

REQUIRED_COLUMNS = [
    "WELL_NAME",
    "POINT_X",
    "POINT_Y",
    "D_REF_TYPE",
    "ELEV_REF_M",
    "UWI",
    "CONTENT",
]

_EXCEL_EPOCH = pd.Timestamp("1899-12-30")


def _is_test_mode() -> bool:
    return "--test" in [str(arg).lower() for arg in getattr(sys, "argv", [])]


TEST_MODE = _is_test_mode()


def _snapshot_payload() -> dict:
    data_file = os.environ.get("PWR_DATA_FILE")
    if not data_file or not os.path.exists(data_file):
        return {}
    try:
        with open(data_file, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
            return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


SNAPSHOT = _snapshot_payload()
PROJECT_NAME = str(SNAPSHOT.get("project", "Unknown"))
TEMP_DB_PATH = Path(os.environ.get("CWDB2_TEMP_DB_FILE", str(constants.TEMP_DB_FILE)))


def _safe_read_csv_from_bytes(raw: bytes) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame()

    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python")

    df.columns = [str(c).strip() for c in df.columns]
    df.replace("nan", np.nan, inplace=True)

    if "WELL_NAME" in df.columns:
        df = df[(df["WELL_NAME"].notna()) | (~df.isna().all(axis=1))].copy()
    else:
        df = df[~df.isna().all(axis=1)].copy()

    return df


def _safe_read_csv_from_path(path: Path) -> pd.DataFrame:
    try:
        raw = path.read_bytes()
    except Exception:
        return pd.DataFrame()
    return _safe_read_csv_from_bytes(raw)


def _has_required_columns(df: pd.DataFrame) -> tuple[bool, list[str]]:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return len(missing) == 0, missing


def _parse_one_date(v):
    if pd.isna(v):
        return pd.NaT
    if isinstance(v, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(v)

    s = str(v).strip()
    if not s:
        return pd.NaT

    try:
        fv = float(s)
        if fv > 59:
            return _EXCEL_EPOCH + pd.to_timedelta(int(round(fv)), unit="D")
    except Exception:
        pass

    ts = pd.to_datetime(s, errors="coerce", utc=False)
    if pd.isna(ts):
        ts = pd.to_datetime(s, errors="coerce", utc=False, dayfirst=True)
    return ts


def _parse_completion_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "LTCOMP_DT" not in out.columns:
        out["LTCOMP_DT_parsed"] = pd.NaT
        return out
    out["LTCOMP_DT_parsed"] = out["LTCOMP_DT"].map(_parse_one_date)
    return out


def _persist_temp_database(df: pd.DataFrame) -> tuple[Path, int]:
    TEMP_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(TEMP_DB_PATH) as con:
        df.to_sql("cwdb_wells_raw", con, if_exists="replace", index=False)
        row_count = int(con.execute("SELECT COUNT(*) FROM cwdb_wells_raw").fetchone()[0])
    return TEMP_DB_PATH, row_count


def _get_petrel_connection():
    if TEST_MODE:
        return None, "Test mode: Petrel connection disabled"
    try:
        from cegalprizm.pythontool import PetrelConnection

        ptp = PetrelConnection(allow_experimental=True)
        return ptp, "Connected"
    except Exception as exc:
        return None, f"Petrel unavailable: {exc}"


def _export_selected_wells_to_petrel(df: pd.DataFrame, selected_wells: list[str], target_folder_name: str):
    petrel, status = _get_petrel_connection()
    if petrel is None:
        preview = (
            df[df["WELL_NAME"].astype(str).isin(selected_wells)]
            .copy()
            .drop_duplicates(subset=["WELL_NAME"], keep="first")
        )
        return {
            "mode": "test",
            "status": status,
            "success": int(len(preview)),
            "failures": [],
            "preview": preview,
        }

    all_folders = list(getattr(petrel, "well_folders", []))
    target_folder = next((wf for wf in all_folders if wf.petrel_name == target_folder_name), None)
    if target_folder is None:
        return {
            "mode": "petrel",
            "status": "Target folder not found",
            "success": 0,
            "failures": [("(all)", "Target folder not found")],
            "preview": pd.DataFrame(),
        }

    try:
        sym_list = petrel.available_well_symbols()
        sym_by_id = {getattr(s, "id", None): s for s in sym_list}
        sym_by_desc = {getattr(s, "description", str(s)): s for s in sym_list}
    except Exception:
        sym_by_id, sym_by_desc = {}, {}

    work_df = (
        df[df["WELL_NAME"].astype(str).isin(selected_wells)]
        .copy()
        .drop_duplicates(subset=["WELL_NAME"], keep="first")
    )

    successes = 0
    failures: list[tuple[str, str]] = []

    for _, row in work_df.iterrows():
        well_name = str(row.get("WELL_NAME", "") or "")
        try:
            new_well = petrel.create_well(well_name, target_folder)

            x = pd.to_numeric(pd.Series([row.get("POINT_X")]), errors="coerce").iloc[0]
            y = pd.to_numeric(pd.Series([row.get("POINT_Y")]), errors="coerce").iloc[0]
            if pd.notna(x) and pd.notna(y):
                new_well.wellhead_coordinates = (float(x), float(y))

            datum_type = str(row.get("D_REF_TYPE", "") or "")
            elev = pd.to_numeric(pd.Series([row.get("ELEV_REF_M")]), errors="coerce").iloc[0]
            if datum_type or pd.notna(elev):
                new_well.well_datum = (datum_type, float(elev) if pd.notna(elev) else 0.0)

            uwi = str(row.get("UWI", "") or "")
            if uwi:
                new_well.uwi = uwi

            sym_id = row.get("Petrel Well Symbol")
            try:
                sym_id = int(float(sym_id)) if pd.notna(sym_id) and f"{sym_id}".strip() else None
            except Exception:
                sym_id = None
            content = str(row.get("CONTENT", "") or "")

            sym_obj = sym_by_id.get(sym_id) if sym_id is not None else None
            if sym_obj is None and content:
                sym_obj = sym_by_desc.get(content)
            if sym_obj is not None:
                new_well.well_symbol = sym_obj

            successes += 1
        except Exception as exc:
            failures.append((well_name or "(empty name)", str(exc)))

    return {
        "mode": "petrel",
        "status": "Done",
        "success": successes,
        "failures": failures,
        "preview": work_df,
    }


section_header_background = NEON_OMV_COLOR
section_body_background = DARK_BLUE_OMV_COLOR if is_dark_mode else "white"
section_text_color = get_content_text_color(is_dark_mode)

load_mode = pn.widgets.Select(
    name="Load Source",
    options=["Reference CSV (test)", "Upload CSV"],
    value="Reference CSV (test)" if TEST_MODE else "Upload CSV",
    stylesheets=select_stylesheets,
)
file_input = pn.widgets.FileInput(name="Upload CSV", accept=".csv")
btn_load = pn.widgets.Button(
    name="Load Data",
    css_classes=["omv-run-btn"],
    stylesheets=get_neon_button_stylesheets(),
    sizing_mode="stretch_width",
)

status_md = pn.pane.Markdown("", sizing_mode="stretch_width")
db_md = pn.pane.Markdown("", sizing_mode="stretch_width")

map_color_by = pn.widgets.Select(name="Color wells by", options=["(none)"], value="(none)", stylesheets=select_stylesheets)
map_zoom = pn.widgets.IntSlider(name="Map Zoom", start=1, end=12, value=5, stylesheets=slider_stylesheets)

export_folder = pn.widgets.Select(name="Target Well Folder", options=["CWDB2Studio_Test"], value="CWDB2Studio_Test", stylesheets=select_stylesheets)
export_wells = pn.widgets.MultiSelect(name="Select wells to export", options=[], value=[], size=10)
btn_export = pn.widgets.Button(
    name="Export selected wells to Petrel",
    css_classes=["omv-run-btn"],
    stylesheets=get_neon_button_stylesheets(),
    sizing_mode="stretch_width",
)
export_status = pn.pane.Markdown("", sizing_mode="stretch_width")

NAV_PAGES = [
    "Load & Database",
    "Table Visualizer",
    "Map Viewer",
    "Statistics",
    "Gantt Chart Viewer",
    "Export to Petrel",
]


data_state: dict[str, object] = {
    "df": pd.DataFrame(),
}
ui_state = {"loaded": False}


def _build_load_view() -> pn.viewable.Viewable:
    return pn.Column(
        pn.pane.Markdown(f"**Project:** {PROJECT_NAME}"),
        pn.pane.Markdown(f"**Version:** {constants.WF_VERSION}"),
        pn.pane.Markdown(f"**Test Mode:** {'Yes' if TEST_MODE else 'No'}"),
        status_md,
        db_md,
        sizing_mode="stretch_both",
    )


def _build_table_view() -> pn.viewable.Viewable:
    df = data_state["df"]
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pn.pane.Markdown("Upload/load a CSV to preview data.")
    return pn.widgets.Tabulator(df, show_index=False, pagination="remote", page_size=20, sizing_mode="stretch_both")


def _build_map_view() -> pn.viewable.Viewable:
    df = data_state["df"]
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pn.pane.Markdown("Upload/load a CSV to view the map.")

    if "W84_LAT_DG" not in df.columns or "W84_LON_DG" not in df.columns:
        return pn.pane.Markdown("Map requires `W84_LAT_DG` and `W84_LON_DG` columns.")

    map_df = df.copy()
    map_df["W84_LAT_DG"] = pd.to_numeric(map_df["W84_LAT_DG"], errors="coerce")
    map_df["W84_LON_DG"] = pd.to_numeric(map_df["W84_LON_DG"], errors="coerce")
    map_df = map_df.dropna(subset=["W84_LAT_DG", "W84_LON_DG"])

    if map_df.empty:
        return pn.pane.Markdown("No rows with valid W84 coordinates.")

    color_by = map_color_by.value
    if color_by == "(none)":
        fig = px.scatter_mapbox(
            map_df,
            lat="W84_LAT_DG",
            lon="W84_LON_DG",
            hover_name="WELL_NAME" if "WELL_NAME" in map_df.columns else None,
            mapbox_style="open-street-map",
            zoom=map_zoom.value,
            height=520,
        )
        fig.update_traces(marker={"size": 11, "color": "#007733"})
    else:
        fig = px.scatter_mapbox(
            map_df,
            lat="W84_LAT_DG",
            lon="W84_LON_DG",
            color=color_by,
            hover_name="WELL_NAME" if "WELL_NAME" in map_df.columns else None,
            mapbox_style="open-street-map",
            zoom=map_zoom.value,
            height=520,
        )

    fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})
    return pn.pane.Plotly(
        fig,
        sizing_mode="stretch_both",
        config={"scrollZoom": True, "displayModeBar": True, "responsive": True},
    )


def _build_statistics_view() -> pn.viewable.Viewable:
    df = data_state["df"]
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pn.pane.Markdown("Upload/load a CSV to view statistics.")

    n_wells = int(df["WELL_NAME"].dropna().astype(str).nunique()) if "WELL_NAME" in df.columns else int(len(df))
    n_rows = int(len(df))
    has_wgs = "Yes" if {"W84_LAT_DG", "W84_LON_DG"}.issubset(df.columns) else "No"

    metrics = pn.Row(
        pn.indicators.Number(name="Number of wells", value=n_wells, format="{value}"),
        pn.indicators.Number(name="Rows in CSV", value=n_rows, format="{value}"),
        pn.pane.Markdown(f"### Has WGS84 coords?\n**{has_wgs}**"),
        sizing_mode="stretch_width",
    )

    numeric_cols = [c for c in ["ELEV_REF_M", "TD_M", "POINT_X", "POINT_Y", "W84_LAT_DG", "W84_LON_DG"] if c in df.columns]
    if not numeric_cols:
        return pn.Column(metrics, pn.pane.Markdown("No numeric columns available for statistics."), sizing_mode="stretch_both")

    desc = df[numeric_cols].apply(pd.to_numeric, errors="coerce").describe(percentiles=[0.25, 0.5, 0.75]).T.reset_index().rename(columns={"index": "Column"})
    stats_table = pn.widgets.Tabulator(desc.round(4), show_index=False, sizing_mode="stretch_both", pagination="remote", page_size=12)

    plot_views: list[pn.viewable.Viewable] = []

    gdf = _parse_completion_dates(df)
    if "LTCOMP_DT_parsed" in gdf.columns and gdf["LTCOMP_DT_parsed"].notna().any():
        comp = gdf[["LTCOMP_DT_parsed"]].dropna().copy()
        comp["year"] = comp["LTCOMP_DT_parsed"].dt.year.astype("Int64")
        if not comp.empty:
            con = duckdb.connect(database=":memory:")
            con.register("comp", comp)
            per_year = con.execute(
                """
                SELECT year::INTEGER AS year, COUNT(*)::INTEGER AS wells
                FROM comp
                WHERE year IS NOT NULL
                GROUP BY year
                ORDER BY year
                """
            ).fetchdf()
            con.close()
            if not per_year.empty:
                plot_views.append(
                    per_year.hvplot.bar(
                        x="year",
                        y="wells",
                        title="Completions per year",
                        height=280,
                        responsive=True,
                        color=BLUE_OMV_COLOR,
                    )
                )

    if "CONTENT" in df.columns:
        content_counts = (
            df["CONTENT"].dropna().astype(str).value_counts().head(15).rename_axis("CONTENT").reset_index(name="count")
        )
        if not content_counts.empty:
            plot_views.append(
                content_counts.hvplot.bar(
                    x="CONTENT",
                    y="count",
                    rot=45,
                    title="Top well content classes",
                    height=280,
                    responsive=True,
                    color=NEON_OMV_COLOR,
                )
            )

    if "ELEV_REF_M" in df.columns:
        elev = pd.to_numeric(df["ELEV_REF_M"], errors="coerce").dropna()
        if not elev.empty:
            plot_views.append(
                elev.to_frame(name="ELEV_REF_M").hvplot.hist(
                    y="ELEV_REF_M",
                    bins=30,
                    title="Elevation reference distribution (m)",
                    height=280,
                    responsive=True,
                    color=BLUE_OMV_COLOR,
                )
            )

    if "TD_M" in df.columns:
        td = pd.to_numeric(df["TD_M"], errors="coerce").dropna()
        if not td.empty:
            plot_views.append(
                td.to_frame(name="TD_M").hvplot.hist(
                    y="TD_M",
                    bins=30,
                    title="Total depth distribution (m)",
                    height=280,
                    responsive=True,
                    color=NEON_OMV_COLOR,
                )
            )

    if {"POINT_X", "POINT_Y"}.issubset(df.columns):
        xy = df[["POINT_X", "POINT_Y"]].copy()
        xy["POINT_X"] = pd.to_numeric(xy["POINT_X"], errors="coerce")
        xy["POINT_Y"] = pd.to_numeric(xy["POINT_Y"], errors="coerce")
        xy = xy.dropna()
        if not xy.empty:
            plot_views.append(
                xy.hvplot.scatter(
                    x="POINT_X",
                    y="POINT_Y",
                    alpha=0.7,
                    size=6,
                    title="Wellhead XY distribution",
                    height=300,
                    responsive=True,
                    color=BLUE_OMV_COLOR,
                )
            )

    plots_column = (
        pn.Column(*[pn.pane.HoloViews(p, sizing_mode="stretch_width") for p in plot_views], sizing_mode="stretch_both")
        if plot_views
        else pn.pane.Markdown("No chartable columns available for additional statistics plots.")
    )

    return pn.Column(
        metrics,
        pn.pane.Markdown("### Data statistics"),
        stats_table,
        pn.pane.Markdown("### Statistical plots"),
        plots_column,
        sizing_mode="stretch_both",
    )


def _build_gantt_view() -> pn.viewable.Viewable:
    df = data_state["df"]
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pn.pane.Markdown("Upload/load a CSV to visualize completion dates.")
    if "WELL_NAME" not in df.columns or "LTCOMP_DT" not in df.columns:
        return pn.pane.Markdown("Columns `WELL_NAME` and `LTCOMP_DT` are required.")

    gdf = _parse_completion_dates(df)
    plot_df = gdf[["WELL_NAME", "LTCOMP_DT_parsed"]].dropna().copy()
    if plot_df.empty:
        return pn.pane.Markdown("No parsable completion dates found in `LTCOMP_DT`.")

    plot_df["LTCOMP_DT_parsed"] = pd.to_datetime(plot_df["LTCOMP_DT_parsed"]).dt.tz_localize(None)
    plot_df = plot_df.sort_values("LTCOMP_DT_parsed")
    gantt_plot = plot_df.hvplot.scatter(
        x="LTCOMP_DT_parsed",
        y="WELL_NAME",
        title="Completion timeline",
        size=7,
        alpha=0.8,
        height=560,
        responsive=True,
        color=BLUE_OMV_COLOR,
    )
    return pn.pane.HoloViews(gantt_plot, sizing_mode="stretch_both")


def _build_export_view() -> pn.viewable.Viewable:
    df = data_state["df"]
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pn.Column(pn.pane.Markdown("Upload/load a CSV to export wells."), export_status, sizing_mode="stretch_both")

    ok, missing = _has_required_columns(df)
    if not ok:
        return pn.Column(pn.pane.Markdown(f"Cannot export. Missing required columns: {', '.join(missing)}"), export_status, sizing_mode="stretch_both")

    return pn.Column(export_folder, export_wells, btn_export, export_status, sizing_mode="stretch_both")


table_panel = pn.Column(sizing_mode="stretch_both")
map_panel = pn.Column(sizing_mode="stretch_both")
stats_panel = pn.Column(sizing_mode="stretch_both")
gantt_panel = pn.Column(sizing_mode="stretch_both")
export_panel = pn.Column(sizing_mode="stretch_both")
load_panel = pn.Column(sizing_mode="stretch_both")

active_page = {"name": "Load & Database"}
page_title = pn.pane.Markdown("# Load & Database")
page_container = pn.Column(sizing_mode="stretch_both")
section_cards: dict[str, pn.Card] = {}
section_card_wrappers: dict[str, pn.Column] = {}
_syncing_cards = {"active": False}


def _refresh_views() -> None:
    load_panel[:] = [_build_load_view()]
    table_panel[:] = [_build_table_view()]
    map_panel[:] = [_build_map_view()]
    stats_panel[:] = [_build_statistics_view()]
    gantt_panel[:] = [_build_gantt_view()]
    export_panel[:] = [_build_export_view()]
    _update_section_cards_enabled_state()
    _show_active_page(active_page["name"])


def _update_section_cards_enabled_state() -> None:
    for name, card in section_cards.items():
        enabled = name == "Load & Database" or ui_state["loaded"]
        wrapper = section_card_wrappers.get(name)
        card.header_color = section_text_color
        if enabled:
            card.header_background = NEON_OMV_COLOR
            card.active_header_background = NEON_OMV_COLOR
            if wrapper is not None:
                wrapper.styles = {"pointer-events": "auto"}
        else:
            card.collapsed = True
            card.header_background = "lightgray"
            card.active_header_background = "lightgray"
            if wrapper is not None:
                wrapper.styles = {"pointer-events": "none"}


def _show_active_page(page_name: str) -> None:
    view_map = {
        "Load & Database": load_panel,
        "Table Visualizer": table_panel,
        "Map Viewer": map_panel,
        "Statistics": stats_panel,
        "Gantt Chart Viewer": gantt_panel,
        "Export to Petrel": export_panel,
    }

    if page_name != "Load & Database" and not ui_state["loaded"]:
        page_name = "Load & Database"

    active_page["name"] = page_name if page_name in view_map else "Load & Database"
    page_title.object = f"# {active_page['name']}"
    page_container[:] = [view_map[active_page["name"]]]


def _sync_section_cards(selected_page: str) -> None:
    if _syncing_cards["active"]:
        return
    _syncing_cards["active"] = True
    try:
        for name, card in section_cards.items():
            card.collapsed = name != selected_page
    finally:
        _syncing_cards["active"] = False


def _on_section_card_toggle(page_name: str):
    def _handler(event) -> None:
        if _syncing_cards["active"]:
            return
        if page_name != "Load & Database" and not ui_state["loaded"]:
            _show_active_page("Load & Database")
            _sync_section_cards("Load & Database")
            return
        if event.new is False:
            _show_active_page(page_name)
            _sync_section_cards(page_name)
        elif active_page["name"] == page_name:
            _sync_section_cards(page_name)

    return _handler


def _update_map_controls(df: pd.DataFrame) -> None:
    candidates = [c for c in ["GEN_STATUS", "CONTENT", "CLASS", "OPERATOR", "COUNTRY"] if c in df.columns]
    map_color_by.options = ["(none)"] + candidates
    map_color_by.value = "(none)"


def _update_export_controls(df: pd.DataFrame) -> None:
    well_names = df["WELL_NAME"].dropna().astype(str).drop_duplicates().sort_values().tolist() if "WELL_NAME" in df.columns else []
    export_wells.options = well_names
    export_wells.value = well_names

    if TEST_MODE:
        export_folder.options = ["CWDB2Studio_Test"]
        export_folder.value = "CWDB2Studio_Test"
        return

    petrel, _ = _get_petrel_connection()
    if petrel is None:
        export_folder.options = ["CWDB2Studio_Test"]
        export_folder.value = "CWDB2Studio_Test"
        return

    try:
        folder_names = [wf.petrel_name for wf in list(getattr(petrel, "well_folders", []))]
        export_folder.options = folder_names or ["CWDB2Studio_Test"]
        if "Input/Wells" in export_folder.options:
            export_folder.value = "Input/Wells"
        elif "CWDB2Studio" in export_folder.options:
            export_folder.value = "CWDB2Studio"
        else:
            export_folder.value = export_folder.options[0]
    except Exception:
        export_folder.options = ["CWDB2Studio_Test"]
        export_folder.value = "CWDB2Studio_Test"


def _load_data(_event=None) -> None:
    if load_mode.value == "Reference CSV (test)":
        csv_path = constants.TEST_CSV_PATH
        df = _safe_read_csv_from_path(csv_path)
        src = str(csv_path)
    else:
        df = _safe_read_csv_from_bytes(file_input.value)
        src = "uploaded file"

    if df.empty:
        status_md.object = "❌ No data loaded. Check source file and format."
        db_md.object = ""
        ui_state["loaded"] = False
        data_state["df"] = pd.DataFrame()
        _refresh_views()
        _sync_section_cards("Load & Database")
        return

    data_state["df"] = df
    ui_state["loaded"] = True
    db_path, rows = _persist_temp_database(df)
    ok, missing = _has_required_columns(df)

    status_md.object = (
        f"✅ Loaded {len(df)} rows from **{src}**.\n"
        f"{'All required export columns detected.' if ok else 'Missing required columns: ' + ', '.join(missing)}"
    )
    db_md.object = f"💾 Temporary DB updated: **{db_path}** (`cwdb_wells_raw`, rows={rows})"

    _update_map_controls(df)
    _update_export_controls(df)
    _refresh_views()


def _on_export(_event=None) -> None:
    df = data_state["df"]
    if not isinstance(df, pd.DataFrame) or df.empty:
        export_status.object = "⚠️ Load data before exporting."
        return

    selected = list(export_wells.value or [])
    if not selected:
        export_status.object = "⚠️ Select at least one well to export."
        return

    result = _export_selected_wells_to_petrel(df, selected, export_folder.value)
    if result["success"] > 0:
        export_status.object = f"✅ Export processed ({result['mode']}): {result['success']} well(s)."
    else:
        export_status.object = f"⚠️ Export finished with no created wells ({result['mode']})."

    if result["mode"] == "test" and isinstance(result["preview"], pd.DataFrame) and not result["preview"].empty:
        export_panel.append(pn.pane.Markdown("### Test-mode export preview"))
        export_panel.append(
            pn.widgets.Tabulator(
                result["preview"][[c for c in ["WELL_NAME", "POINT_X", "POINT_Y", "UWI", "CONTENT"] if c in result["preview"].columns]],
                show_index=False,
                sizing_mode="stretch_both",
                pagination="remote",
                page_size=10,
            )
        )

    if result["failures"]:
        fail_df = pd.DataFrame(result["failures"], columns=["Well", "Error"])
        export_panel.append(pn.pane.Markdown("### Export errors"))
        export_panel.append(pn.widgets.Tabulator(fail_df, show_index=False, sizing_mode="stretch_both"))


btn_load.on_click(_load_data)
btn_export.on_click(_on_export)
map_color_by.param.watch(lambda _event: _refresh_views(), "value")
map_zoom.param.watch(lambda _event: _refresh_views(), "value")

load_card = pn.Card(
    load_mode,
    file_input,
    btn_load,
    title="Load & Database",
    collapsed=True,
    hide_header=False,
    sizing_mode="stretch_width",
    header_background=NEON_OMV_COLOR,
    active_header_background=NEON_OMV_COLOR,
    header_color=section_text_color,
    styles={"background": section_body_background, "color": section_text_color},
    margin=(0, 0, 12, 0),
)

section_cards["Load & Database"] = load_card

for section_name in NAV_PAGES:
    if section_name == "Load & Database":
        continue
    card = pn.Card(
        pn.pane.Markdown(f"Open {section_name}"),
        title=section_name,
        collapsed=True,
        hide_header=False,
        sizing_mode="stretch_width",
        header_background=NEON_OMV_COLOR,
        active_header_background=NEON_OMV_COLOR,
        header_color=section_text_color,
        styles={"background": section_body_background, "color": section_text_color},
        margin=(0, 0, 12, 0),
    )
    section_cards[section_name] = card

for section_name in NAV_PAGES:
    section_cards[section_name].param.watch(_on_section_card_toggle(section_name), "collapsed")
    section_card_wrappers[section_name] = pn.Column(section_cards[section_name], sizing_mode="stretch_width")

main_content = pn.Column(
    page_title,
    page_container,
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
    sidebar=[section_card_wrappers[name] for name in NAV_PAGES],
    main=[main_content],
    header=[
        pn.Row(
            pn.Spacer(sizing_mode="stretch_width"),
            pn.pane.HTML(docs_button_html(DOCUMENTATION_URL)),
            sizing_mode="stretch_width",
            margin=0,
        )
    ],
)

if constants.LOGO_PATH.exists():
    template_kwargs["logo"] = str(constants.LOGO_PATH)
if constants.FAVICON_PATH.exists():
    template_kwargs["favicon"] = str(constants.FAVICON_PATH)

_refresh_views()
_show_active_page("Load & Database")
_sync_section_cards("Load & Database")
pn.template.FastListTemplate(**template_kwargs).servable()

import streamlit as st
import numpy as np
import pandas as pd
import utils
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# === Standardization: unified block with fallback ===
import json, re
from pathlib import Path
from collections import defaultdict

import utils
utils.render_grouped_sidebar_nav()

##################################################

def _slugify(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\(.*?\)", " ", s)
    s = s.replace("\\", " ").replace("/", " ").replace("_", " ").replace("-", " ")
    s = re.sub(r"[^A-Za-z0-9\s\.]+", " ", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "project"

proj_raw  = utils.get_petrel_connection().get_current_project_name()
proj_slug = _slugify(proj_raw)
STD_DIR   = Path.home() / "Cegal" / "Prizm" / "geoPython" / proj_slug / "logStandarization"

def _load_latest_name_map() -> dict:
    cands = sorted(STD_DIR.glob(f"log_name_map__{proj_slug}__*__*.json"))
    return json.loads(cands[-1].read_text(encoding="utf-8")) if cands else {}

NAME_MAP = _load_latest_name_map()          # {original -> standardized}
STD_TO_ORIG = defaultdict(list)
for orig, std in NAME_MAP.items():
    STD_TO_ORIG[std].append(orig)

# Fallback when there is no standardization map in this project
NAME_MAP_EMPTY = (not bool(NAME_MAP))

if NAME_MAP_EMPTY:
    # (1) eligible wells = all wells (raw mode)
    def eligible_well_names(wells):
        return [w.petrel_name for w in wells]

    # (2) to_standardized_options passes original names through (unique/sorted)
    def to_standardized_options(original_names: list[str]) -> list[str]:
        return sorted({str(n) for n in original_names})

else:
    # Normal behavior: only names present in NAME_MAP are “used in project”
    def to_standardized_options(original_names: list[str]) -> list[str]:
        std_set = set()
        for n in original_names:
            if n in NAME_MAP:
                std_set.add(NAME_MAP[n])
        return sorted(std_set)

    def eligible_well_names(wells) -> list[str]:
        allowed = set(NAME_MAP.keys())
        return [
            w.petrel_name for w in wells
            if any(getattr(lg, "petrel_name", "") in allowed for lg in getattr(w, "logs", []))
        ]

# Optional: colors via mnemonics + mapping_records
MNEM_PATH = Path(r"N:\_USER_GLOBAL\PETREL\Prizm\wf\referenceDocuments\mnemonics_master.json")
try:
    MNEMS = json.loads(MNEM_PATH.read_text(encoding="utf-8")) if MNEM_PATH.exists() else {}
except Exception:
    MNEMS = {}

STD_TO_GROUP = {}
try:
    rec_cands = sorted(STD_DIR.glob(f"log_mapping_records__{proj_slug}__*__*.json"))
    if rec_cands:
        records = json.loads(rec_cands[-1].read_text(encoding="utf-8"))
        for r in records:
            std = str(r.get("Standardized", "")).strip()
            grp = str(r.get("Group", "")).strip()
            if std and grp and grp != "(unmatched)" and std not in STD_TO_GROUP:
                STD_TO_GROUP[std] = grp
except Exception:
    pass

# Pretty label for standardized codes in UI
def _fmt_std(std: str) -> str:
    grp = STD_TO_GROUP.get(std, "")
    return f"{std} — {grp}" if grp else std


def color_for_std_label(std_label: str) -> str | None:
    grp = STD_TO_GROUP.get(std_label, std_label)
    meta = MNEMS.get(grp) or MNEMS.get(std_label) or {}
    style = meta.get("style") if isinstance(meta, dict) else {}
    return (style or {}).get("color")


# STREAMLIT CONFIG  ==========================
st.set_page_config(page_title='GeoPython',
                   #page_icon=':bar_chart:', 
                   layout='wide',
                   initial_sidebar_state='expanded')
                   
# ============== Petrel connection ===============
petrel_project = utils.get_petrel_connection()

# MENU 
#########################################

utils.sidebar_footer(petrel_project, utils.LOGO_PATH, utils.APP_VERSION)

st.title("Well Logs")

# Get wells and other well data from utils
wells = utils.get_all_wells_flat(petrel_project)  # Not cached
wells_summary_df, wells_full_df, stat_keys = utils.get_all_well_data(petrel_project)

# Upload well tops for range selection
tops_df = utils.load_tops_dataframe(petrel_project)

# Use wells safely now
well_names = [w.petrel_name for w in wells]
well_dict = {w.petrel_name: w for w in wells}
well_logs = {
    w.petrel_name: [getattr(lg, "petrel_name", "") for lg in getattr(w, "logs", [])]
    for w in wells
}

# --- Mode selection ---
# mode = st.radio("Select Log Mode", ["All Logs", "Single Log"], horizontal=True)

st.header("Global Wells Logs Analysis")

col1, col2 = st.columns([1, 4])

with col1:
    # Select wells (multi-select)
    # keep only wells that have any mapped logs
    elig_names = eligible_well_names(wells)
    selected_wells = st.multiselect("Select wells to include", options=elig_names, default=elig_names)

    # Build the log options for the selected wells
    logs_for_selected_wells_orig = utils.get_logs_for_selected_wells(well_logs, selected_wells)
    logs_for_selected_wells_std  = to_standardized_options(logs_for_selected_wells_orig)
    
    # --- NEW: Filter by families (groups) coming from Standardization ---
    groups_available = sorted({STD_TO_GROUP.get(s, "(unlabeled)") for s in logs_for_selected_wells_std})
    if groups_available and not NAME_MAP_EMPTY:
        sel_groups = st.multiselect(
            "Filter by groups (families)",
            options=groups_available,
            default=groups_available,
            help="Only logs belonging to these families will be shown below.",
            key="gl_groups"
        )
        logs_for_selected_wells_std = [
            s for s in logs_for_selected_wells_std
            if STD_TO_GROUP.get(s, "(unlabeled)") in set(sel_groups)
        ]

    # After computing logs_for_selected_wells_std
    # Prefer a few common standardized codes (fall back to whatever is present)
    preferred_default_std = ["RHO_STD", "GR_STD", "DT_STD"]
    default_std = [n for n in preferred_default_std if n in logs_for_selected_wells_std] or logs_for_selected_wells_std[:3]

    if not logs_for_selected_wells_std:
        st.warning("No logs match the selected families in the current well selection.")
        selected_logs_std = []
    else:
        selected_logs_std = st.multiselect(
            "Select logs to include",
            options=logs_for_selected_wells_std,
            default=default_std,
            format_func=_fmt_std,  # shows: e.g. 'GR_STD — Gamma Ray'
            key="gl_logs"
        )

    # Expand standardized selections back to originals for data retrieval
    selected_logs_orig = []
    for std in selected_logs_std:
        selected_logs_orig.extend(STD_TO_ORIG.get(std, [std]))

    md_min_default, md_max_default = utils.get_global_md_range(selected_wells, selected_logs_orig, well_dict)

    depth_selection_mode = st.radio("How would you like to select the depth range?", ["Slider", "Tops"], horizontal=True)

    if depth_selection_mode == "Slider":
        # Initialize session state variables once
        if "depth_range_temp" not in st.session_state:
            st.session_state.depth_range_temp = (md_min_default, md_max_default)
        if "depth_range_applied" not in st.session_state:
            st.session_state.depth_range_applied = (md_min_default, md_max_default)

        # Get the current temporary slider value from session state
        temp_range = st.session_state.depth_range_temp

        # Show the slider and store new value in local variable
        new_temp_range = st.slider(
            "Select depth range (MD)",
            min_value=md_min_default,
            max_value=md_max_default,
            value=st.session_state.depth_range_temp,
            step=1
        )

        # Update session state only if slider moved (to avoid constant update on rerun)
        if new_temp_range != temp_range:
            st.session_state.depth_range_temp = new_temp_range

        # Button to apply the chosen depth range
        if st.button("Apply Depth Range"):
            st.session_state.depth_range_applied = st.session_state.depth_range_temp

        depth_min, depth_max = st.session_state.depth_range_applied

    else:
        # Filter tops to only selected wells
        filtered_tops = tops_df[tops_df['Well identifier (Well name)'].isin(selected_wells)].copy()

        # Show top/base selectors
        top_names = sorted(filtered_tops["Surface"].unique())

        top_col1, top_col2 = st.columns(2)
        with top_col1:
            top_marker = st.selectbox("Select **top** marker", top_names)

        with top_col2:
            base_marker = st.selectbox("Select **base** marker", top_names, index=min(len(top_names)-1, 1))

        # Calculate MDs for selected markers (use mean if multiple wells)
        depth_min, depth_max = utils.get_md_from_tops(tops_df, selected_wells, top_marker, base_marker)

        if depth_min is not None and depth_max is not None:
            st.write(f"Selected Depth Range from Tops: {top_marker} ({depth_min:.2f} m) to {base_marker} ({depth_max:.2f} m)")
        else:
            st.warning("Selected tops not found in the selected wells.")
            depth_min, depth_max = md_min_default, md_max_default  # fallback

    # Now load and filter logs regardless of depth_selection_mode
    merged_logs_df = utils.get_logs_data_for_wells_logs(
        selected_wells, selected_logs_orig, wells, depth_min, depth_max
    )

    # Standardize 'Log' values if there is data
    if merged_logs_df is not None and not merged_logs_df.empty:
        df_long_plot = merged_logs_df.copy()
        if "Log" in df_long_plot.columns:
            df_long_plot["Log"] = df_long_plot["Log"].astype(str).map(lambda x: NAME_MAP.get(x, x))
    else:
        df_long_plot = pd.DataFrame()
        st.warning("No logs data available.")


#### -------------------------- TABS SECTION -----------------------------#####################
with col2:
    # Tabs for Data Table, Completeness, and Statistics
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        " Data Table", "Logs Completeness", " Statistics", "Outlier Visualization",
        "Missing-Value Plot", "Quality Checks", "Pairplots", "Heatmap", "Histogram View"
    ])

    with tab1:
        st.markdown("### Well-Log Table")

        if df_long_plot is None or df_long_plot.empty:
            st.info("No data to filter yet.")
        else:
            # Map your expected column names (tweak if different)
            well_col = "Well" if "Well" in df_long_plot.columns else "well"
            log_col  = "Log"  if "Log"  in df_long_plot.columns else "log"
            md_col   = "MD"   if "MD"   in df_long_plot.columns else ("Depth" if "Depth" in df_long_plot.columns else "md")
            val_col  = "Value" if "Value" in df_long_plot.columns else ("value" if "value" in df_long_plot.columns else None)

            missing_cols = [c for c in [well_col, log_col, md_col, val_col] if c not in df_long_plot.columns]
            if missing_cols:
                st.error(f"Missing expected columns in data: {missing_cols}")
            else:
                # Options limited by earlier selections
                well_opts = [w for w in selected_wells if w in df_long_plot[well_col].unique()]
                log_opts  = [l for l in logs_for_selected_wells_std if l in df_long_plot[log_col].unique()]
                if not well_opts or not log_opts:
                    st.warning("No matching wells/logs in the current dataset.")
                else:
                    fcol1, fcol2 = st.columns(2)
                    with fcol1:
                        well_pick = st.selectbox("Well", options=well_opts, index=0)
                    with fcol2:
                        log_pick = st.selectbox("Log", options=log_opts, index=0)

                    # Filter and show only MD + Value
                    mask = (df_long_plot[well_col] == well_pick) & (df_long_plot[log_col] == log_pick)
                    table_df = df_long_plot.loc[mask, [md_col, val_col]].reset_index(drop=True)

                    # Remove index visually
                    st.dataframe(table_df, hide_index=True)

    with tab2:
        st.markdown("### Well-Log Completeness")

        left, right = st.columns([1, 2])

        # ===================== LEFT: Presence table =====================
        with left:
            st.markdown("#### Presence Matrix")
            presence_df = utils.get_log_presence_matrix(
                selected_wells, selected_logs_orig, _well_dict=well_dict
            )
            # Show standardized column names to the user
            presence_df = presence_df.rename(columns=lambda c: NAME_MAP.get(str(c), str(c)))

            colorize = st.checkbox(
                "Colorize presence matrix ( = green,  = red)",
                value=True,
                key="colorize_tab2"
            )

            if colorize:
                st.dataframe(presence_df.style.applymap(utils.highlight_presence),
                            hide_index=True, use_container_width=True)
            else:
                st.dataframe(presence_df, hide_index=True, use_container_width=True)

        # ===================== RIGHT: Map + filter ======================
        with right:
            st.markdown("#### Map")

            if not selected_logs_std:
                st.info("Select at least one log to display on the map.")
            else:
                # Pick which log to check "presence" for on the map
                selected_log_for_map = st.selectbox(
                    "Log to display",
                    options=selected_logs_std,
                    key="tab2_map_log"
                )

                # Get coordinates
                # Get coordinates (already returns Well Name, latitude, longitude)
                geo_df = utils.get_well_min_lat_long(wells).copy()

                # Ensure expected columns are present
                required_cols = {"Well Name", "latitude", "longitude"}
                if not required_cols.issubset(geo_df.columns):
                    st.error("Latitude/Longitude data not found in expected format (`Well Name`, `latitude`, `longitude`).")
                else:
                    # Keep only the needed columns
                    geo_df = geo_df[["Well Name", "latitude", "longitude"]].copy()
                    # Filter to selected wells and drop missing coordinates
                    geo_df = geo_df[geo_df["Well Name"].isin(selected_wells)].dropna(subset=["latitude", "longitude"])

                    if geo_df.empty:
                        st.warning("No coordinates available for the selected wells.")
                    else:
                        # Compute presence for the chosen log in the selected depth range
                        presence_status = []
                        for _, row in geo_df.iterrows():
                            well_name = row["Well Name"]
                            present = False

                            # Find the well & log
                            well_obj = next((w for w in wells if w.petrel_name == well_name), None)
                            if well_obj is not None:
                                candidate_origs = STD_TO_ORIG.get(selected_log_for_map, [selected_log_for_map])
                                log_obj = next((lg for lg in well_obj.logs if lg.petrel_name in candidate_origs), None)
                                if log_obj is not None:
                                    df_log = log_obj.as_dataframe()
                                    if {"MD", "Value"}.issubset(df_log.columns):
                                        df_filtered = df_log[(df_log["MD"] >= depth_min) & (df_log["MD"] <= depth_max)]
                                        if not df_filtered.empty and not df_filtered["Value"].isna().all():
                                            present = True

                            presence_status.append("Present" if present else "Absent")

                        map_df = geo_df.copy()
                        map_df["Presence"] = presence_status

                        color_map = {"Present": "green", "Absent": "red"}
                        fig = px.scatter_mapbox(
                            map_df,
                            lat="latitude",
                            lon="longitude",
                            color="Presence",
                            color_discrete_map=color_map,
                            hover_name="Well Name",
                            zoom=6,
                            height=500,
                            mapbox_style="open-street-map",
                        )
                        fig.update_traces(marker=dict(size=12))
                        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### Well-Log Statistics Matrix")

        left, right = st.columns([1, 2])

        # -------------------- LEFT: Controls --------------------
        with left:
            stat_option = st.selectbox(
                "Statistic",
                ["mean", "median", "min", "max", "25%", "50%", "75%"],
                index=0,
                key="stat_select",
            )

            round_digits = st.number_input(
                "Select number of decimal places",
                min_value=0, max_value=10,
                value=2, step=1,
                format="%d",
                key="logs_round_digits"
            )

        # Compute after controls are chosen
        stat_func = utils.get_stat_func_from_option(stat_option)

        # -------------------- RIGHT: Table ----------------------
        with right:
            stats_df = utils.get_log_statistics(
                selected_wells, selected_logs_orig, well_dict, depth_min, depth_max, stat_func
            )

            if stats_df is None or stats_df.empty:
                st.info("No statistics available for the current selection.")
            else:
                # Standardize any column labels that are logs
                stats_df = stats_df.rename(columns=lambda c: NAME_MAP.get(str(c), str(c)))
                st.dataframe(stats_df.round(round_digits).fillna("-"), use_container_width=True)


    with tab4:
        st.markdown("### Outlier Visualization")

        required_cols = {"Well", "Log", "Value"}
        if df_long_plot is None or df_long_plot.empty or not required_cols.issubset(df_long_plot.columns):
            st.warning("Log data must contain 'Well', 'Log', and 'Value' columns.")
        else:
            # Respect earlier selections (depth already applied upstream)
            df_plot = df_long_plot.copy()
            df_plot = df_long_plot[df_long_plot["Well"].isin(selected_wells) & df_long_plot["Log"].isin(selected_logs_std)]
            df_plot = df_plot[df_plot["Well"] != "Unknown"]

            left, right = st.columns([1, 2])

            with left:
                plot_kind = st.radio("Plot type", ["Box", "Violin"], horizontal=True, key="plot_kind_radio")

            with right:
                if df_plot.empty:
                    st.info("No data after filtering by selected wells/logs.")
                else:
                    fig = go.Figure()
                    logs = sorted(df_plot["Log"].unique())

                    def _hex_or_palette(idx: int, std_label: str):
                        c = color_for_std_label(std_label)
                        return c if c else px.colors.qualitative.Plotly[idx % len(px.colors.qualitative.Plotly)]

                    if plot_kind == "Box":
                        for i, log_name in enumerate(logs):
                            dfl = df_plot[df_plot["Log"] == log_name]
                            if dfl.empty:
                                continue
                            fig.add_trace(go.Box(
                                x=dfl["Well"],
                                y=dfl["Value"],
                                name=log_name,
                                marker_color=_hex_or_palette(i, log_name),
                                boxpoints=False
                            ))
                        fig.update_layout(
                            title="Box Plot of Logs Grouped by Well",
                            xaxis_title="Well",
                            yaxis_title="Log Value",
                            boxmode="group",
                            height=600,
                            legend_title_text="Log"
                        )
                    else:  # Violin
                        for i, log_name in enumerate(logs):
                            dfl = df_plot[df_plot["Log"] == log_name]
                            if dfl.empty:
                                continue
                            fig.add_trace(go.Violin(
                                x=dfl["Well"],
                                y=dfl["Value"],
                                name=log_name,
                                line_color=_hex_or_palette(i, log_name),
                                box_visible=True,
                                meanline_visible=True,
                                points=False
                            ))
                        fig.update_layout(
                            title="Violin Plot of Logs Grouped by Well",
                            xaxis_title="Well",
                            yaxis_title="Log Value",
                            violinmode="group",
                            height=600,
                            legend_title_text="Log"
                        )


                    # Keep x order consistent with current well selection (optional but nice)
                    fig.update_xaxes(categoryorder="array", categoryarray=selected_wells)

                    st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.markdown("### Vertical Well Missing-Value Plot")

        try:
            fig = utils.plot_missing_matrix(stats_df)
            st.pyplot(fig)
        except Exception:
            st.info("No statistics available for the missing-value plot yet.")

    with tab6:
        st.markdown("### Quality Checks")

        # == Left: run + table ================================================
        col2a, col2b = st.columns(2)

        # Define available tests + default params (edit as needed)
        available_tests = {
            "all_positive": {},
            "all_above": {"threshold": 50},
            "mean_below": {"threshold": 100},
            "no_nans": {},
            "range": {"min": 0, "max": 200},
            "no_flat": {},
            "no_monotonic": {},
        }

        with col2a:
            selected_test_name = st.selectbox(
                "Select a quality check", options=list(available_tests.keys()), index=0
            )

            # Parameter controls (only for tests that need them)
            params = {}
            if selected_test_name == "all_above":
                params["threshold"] = st.slider(
                    "Threshold (all_above)", min_value=-1000, max_value=1000,
                    value=available_tests["all_above"]["threshold"]
                )
            elif selected_test_name == "mean_below":
                params["threshold"] = st.slider(
                    "Threshold (mean_below)", min_value=-1000, max_value=1000,
                    value=available_tests["mean_below"]["threshold"]
                )
            elif selected_test_name == "range":
                rmin, rmax = st.slider(
                    "Range (min, max)", min_value=-1000, max_value=1000,
                    value=(available_tests["range"]["min"], available_tests["range"]["max"])
                )
                params["min"], params["max"] = rmin, rmax

            # Run the selected test
            if st.button("Run Quality Checks", type="primary"):
                st.session_state["quality_df"] = utils.run_quality_checks(
                    [well_dict[w] for w in selected_wells],
                    depth_min, depth_max,
                    {selected_test_name: params},
                    selected_logs_orig  # originals for computation
                )

            #  Results table (this was missing)
            if "quality_df" in st.session_state and st.session_state["quality_df"] is not None and not st.session_state["quality_df"].empty:
                qc_df = st.session_state["quality_df"].copy()

                # If it's the long format, map Log column to standardized labels for UI
                if "Log" in qc_df.columns:
                    qc_df["Log"] = qc_df["Log"].astype(str).map(lambda x: NAME_MAP.get(x, x))

                st.markdown("####  Results")
                try:
                    st.dataframe(qc_df.style.applymap(utils.highlight_pass_fail), use_container_width=True)
                except Exception:
                    st.dataframe(qc_df, use_container_width=True)
            else:
                st.info("Run a quality check to see results.")



        # == Right: map (green/red dots) =======================================
        with col2b:
            st.markdown("#### Map")

            if "quality_df" not in st.session_state or st.session_state["quality_df"] is None or st.session_state["quality_df"].empty:
                st.info("Run a quality check to see the map.")
            else:
                qc_df = st.session_state["quality_df"].copy()
                # Standardize Log labels for the map UI if present
                if "Log" in qc_df.columns:
                    qc_df["Log"] = qc_df["Log"].astype(str).map(lambda x: NAME_MAP.get(x, x))

                # Find the well column (fallback to index)
                well_col = next((c for c in qc_df.columns if str(c).lower().startswith(("well", "bore", "hole"))), None)
                if well_col is None:
                    qc_df = qc_df.reset_index().rename(columns={"index": "Well"})
                    well_col = "Well"

                # LONG vs WIDE
                is_long = {"Log", "Result"}.issubset(qc_df.columns)

                if is_long:
                    # Prefer logs already selected in the sidebar; fallback to all in QC
                    log_opts = [l for l in selected_logs_std if l in qc_df["Log"].unique().tolist()] or \
                            sorted(qc_df["Log"].dropna().astype(str).unique())
                    selected_log_for_map = st.selectbox("Select log", log_opts, key="qc_map_log_select")
                    sub = qc_df[qc_df["Log"].astype(str) == selected_log_for_map][[well_col, "Result"]].copy()
                else:
                    # WIDE: pick a QC column (exclude well col)
                    qc_cols = [c for c in qc_df.columns if c != well_col]
                    selected_qc_col = st.selectbox("Select QC column", qc_cols, key="qc_map_col_select")
                    sub = qc_df[[well_col, selected_qc_col]].rename(columns={selected_qc_col: "Result"}).copy()

                # Map PASS/FAIL --> colors (simple + forgiving)
                def to_color(x):
                    s = str(x).strip().lower()
                    if s in ("pass", "", "true", "1", "yes", "ok", "success"):
                        return "green"
                    if s in ("fail", "", "false", "0", "no", "error"):
                        return "red"
                    # numeric fallback: nonzero-->green, zero-->red
                    try:
                        return "green" if float(s) != 0.0 else "red"
                    except Exception:
                        return "gray"

                sub["color"] = sub["Result"].apply(to_color)

                # Coordinates for wells (use Petrel names to join)
                coords_df = utils.get_well_min_lat_long(wells).copy()
                coords_df.columns = ["latitude", "longitude"]
                coords_df[well_col] = [getattr(w, "petrel_name", str(w)) for w in wells]

                # If you want to show only currently selected wells:
                if selected_wells:
                    coords_df = coords_df[coords_df[well_col].isin(selected_wells)]

                plot_df = coords_df.merge(sub[[well_col, "color"]], on=well_col, how="left")

                # Plot just colored dots (no symbols)
                fig = px.scatter_mapbox(
                    plot_df,
                    lat="latitude", lon="longitude",
                    hover_name=well_col,
                    color="color",
                    color_discrete_map="identity",
                    zoom=6, height=500,
                    mapbox_style="open-street-map",
                )
                fig.update_traces(marker=dict(size=12), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

                                # ---------- Failing rows drilldown (below the map) ----------
                # Only appear if at least one well failed (red)
                failing_mask = plot_df["color"].eq("red")
                failed_wells = sorted(plot_df.loc[failing_mask, well_col].dropna().unique().tolist())

                if failed_wells:
                    st.markdown("####  Show failing rows")
                    chosen_well = st.selectbox("Well", failed_wells, key="qc_fail_well_pick")

                    # Determine which log was selected on the map
                    if is_long:
                        std_label = selected_log_for_map  # standardized label from the dropdown
                    else:
                        # Parse "<Log> - <test>" -> get the log part before " - "
                        std_label = str(selected_qc_col).split(" - ", 1)[0]
                    
                    candidate_origs = STD_TO_ORIG.get(std_label, [std_label])

                    # Helper: per-row violation mask (simple tests only)
                    def _violation_mask(values: pd.Series, test_name: str, p: dict) -> pd.Series:
                        v = pd.to_numeric(values, errors="coerce")
                        if test_name == "all_positive":
                            return v <= 0
                        elif test_name == "all_above":
                            thr = float(p.get("threshold", 0))
                            return v < thr
                        elif test_name == "range":
                            vmin = float(p.get("min", -np.inf))
                            vmax = float(p.get("max",  np.inf))
                            return (v < vmin) | (v > vmax)
                        elif test_name == "no_nans":
                            return v.isna()
                        # Aggregate-only tests don't have row-level violations
                        elif test_name in {"mean_below", "no_flat", "no_monotonic"}:
                            return pd.Series(False, index=values.index)
                        else:
                            return pd.Series(False, index=values.index)

                    # Get that well's log data and show failing rows (MD, Value)
                    w = well_dict.get(chosen_well)
                    if w is None:
                        st.warning(f"Well object not found for '{chosen_well}'.")
                    else:
                        log_obj = next((lg for lg in w.logs if lg.petrel_name in candidate_origs), None)
                        if log_obj is None:
                            st.warning(f"No Petrel log found in '{chosen_well}' for {std_label} (tried: {', '.join(candidate_origs)})")
                        else:
                            df = log_obj.as_dataframe()
                            if {"MD", "Value"}.issubset(df.columns):
                                # depth filtering consistent with the rest of the app
                                df = df[(df["MD"] >= depth_min) & (df["MD"] <= depth_max)]
                                if df.empty:
                                    st.info("No data in the selected depth range.")
                                else:
                                    # If the selected QC is aggregate-only, inform and skip row listing
                                    if selected_test_name in {"mean_below", "no_flat", "no_monotonic"}:
                                        st.info(f"'{selected_test_name}' is an aggregate test - row-level violations are not defined.")
                                    else:
                                        mask = _violation_mask(df["Value"], selected_test_name, params)
                                        bad = df.loc[mask, ["MD", "Value"]].copy()
                                        if bad.empty:
                                            st.success("No per-row violations for this well.")
                                        else:
                                            st.dataframe(bad.reset_index(drop=True), hide_index=True, use_container_width=True)
                            else:
                                st.warning("Expected columns 'MD' and 'Value' not found in this log data.")

    with tab7:
        st.markdown("### Pairplots")

        if df_long_plot is None or df_long_plot.empty:
            st.warning("No log data available for the selected wells/logs/depth range.")
        else:
            # Filter to current selection (depth already applied upstream)
            df_pp = df_long_plot[
                df_long_plot["Well"].isin(selected_wells) &
                df_long_plot["Log"].isin(selected_logs_std)
            ].copy()

            if df_pp.empty:
                st.warning("No data after filtering by selected wells/logs.")
            else:
                # Pivot to wide: rows=(Well, MD), columns=logs, values=Value
                wide = (
                    df_pp.pivot_table(index=["Well", "MD"], columns="Log", values="Value", aggfunc="mean")
                    .reset_index()
                )

                # Only use logs that actually exist after pivot
                plot_cols = [c for c in selected_logs_std if c in wide.columns]
                if len(plot_cols) < 2:
                    st.warning("Select at least two logs with overlapping data to build a pairplot.")
                else:
                    # Light subsampling for performance on very large sets
                    max_rows = 20000
                    if len(wide) > max_rows:
                        wide = wide.sample(max_rows, random_state=42)

                    # Build grid specs: histogram on diagonal, scatter on lower triangle; blank upper triangle
                    d = len(plot_cols)
                    specs = [[({"type": "histogram"} if r == c else ({"type": "xy"} if r > c else None))
                            for c in range(d)] for r in range(d)]

                    # Adaptive height so it fits the container (clamped)
                    auto_height = int(min(900, max(750, 130 * d)))

                    fig = make_subplots(
                        rows=d, cols=d,
                        specs=specs,
                        shared_xaxes=False, shared_yaxes=False,
                        horizontal_spacing=0.02, vertical_spacing=0.02
                    )

                    # Color handling
                    well_counts = wide["Well"].nunique()
                    use_color_by_well = well_counts <= 10  # set True to force per-well colors always
                    wells_present = sorted(wide["Well"].dropna().unique().tolist())
                    palette = px.colors.qualitative.Plotly + px.colors.qualitative.D3
                    color_map = {w: palette[i % len(palette)] for i, w in enumerate(wells_present)}
                    legend_shown = set()

                    # -------- Diagonal = histogram(s) --------
                    # If <=10 wells, overlay one histogram per well (color‑coded)
                    # Otherwise, show one aggregated gray histogram
                    for i, col in enumerate(plot_cols):
                        all_vals = pd.to_numeric(wide[col], errors="coerce").dropna()
                        if all_vals.empty:
                            continue
                        xmin = float(all_vals.min())
                        xmax = float(all_vals.max())
                        if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
                            xmax = xmin + 1e-9
                        bin_size = (xmax - xmin) / 30.0

                        if use_color_by_well:
                            # per-well overlayed histograms (aligned bins)
                            for wname in wells_present:
                                sub = pd.to_numeric(
                                    wide.loc[wide["Well"] == wname, col],
                                    errors="coerce"
                                ).dropna()
                                if sub.empty:
                                    continue
                                show_leg = (wname not in legend_shown)
                                fig.add_histogram(
                                    x=sub,
                                    xbins=dict(start=xmin, end=xmax, size=bin_size),
                                    histnorm="probability density",
                                    opacity=0.45,
                                    marker_color=color_map[wname],
                                    name=wname,
                                    showlegend=show_leg,
                                    hovertemplate=(
                                        f"Well: {wname}<br>{col}: %{{x:.3g}}"
                                        "<br>Density: %{y:.3g}<extra></extra>"
                                    ),
                                    row=i+1, col=i+1
                                )
                                if show_leg:
                                    legend_shown.add(wname)
                        else:
                            # aggregated histogram (fast/clean when many wells)
                            fig.add_histogram(
                                x=all_vals,
                                xbins=dict(start=xmin, end=xmax, size=bin_size),
                                histnorm="probability density",
                                marker_color="rgba(120,120,120,0.85)",
                                showlegend=False,
                                hovertemplate=f"{col}: %{{x:.3g}}<br>Density: %{{y:.3g}}<extra></extra>",
                                row=i+1, col=i+1
                            )

                    # -------- Lower triangle = scatter --------
                    for r in range(1, d):
                        for c in range(0, r):
                            xname, yname = plot_cols[c], plot_cols[r]

                            if use_color_by_well:
                                # One trace per well (legend shown once per well)
                                for wname in wells_present:
                                    sub = wide[wide["Well"] == wname]
                                    x = pd.to_numeric(sub[xname], errors="coerce")
                                    y = pd.to_numeric(sub[yname], errors="coerce")
                                    if x.notna().any() and y.notna().any():
                                        show_leg = (wname not in legend_shown)
                                        fig.add_scatter(
                                            x=x, y=y, mode="markers",
                                            marker=dict(size=3, opacity=0.35, color=color_map[wname]),
                                            name=wname,
                                            showlegend=show_leg,
                                            hovertemplate=(
                                                f"Well: {wname}<br>{xname}: %{{x:.3g}}"
                                                f"<br>{yname}: %{{y:.3g}}<extra></extra>"
                                            ),
                                            row=r+1, col=c+1
                                        )
                                        if show_leg:
                                            legend_shown.add(wname)
                            else:
                                # Single fast scattergl for all points (show well in hover text)
                                x = pd.to_numeric(wide[xname], errors="coerce")
                                y = pd.to_numeric(wide[yname], errors="coerce")
                                txt = wide["Well"].astype(str)
                                fig.add_scattergl(
                                    x=x, y=y, mode="markers",
                                    marker=dict(size=3, opacity=0.35),
                                    name="points",
                                    showlegend=False,
                                    text=txt,
                                    hovertemplate=(
                                        "Well: %{text}<br>"
                                        f"{xname}: %{{x:.3g}}<br>{yname}: %{{y:.3g}}<extra></extra>"
                                    ),
                                    row=r+1, col=c+1
                                )

                    # Axis labels only on bottom row / left col to keep it clean
                    for i, col in enumerate(plot_cols):
                        fig.update_xaxes(title_text=col, row=d, col=i+1)
                        fig.update_yaxes(title_text=col, row=i+1, col=1)

                    # Layout: compact and responsive; overlay for histogram bars
                    fig.update_layout(
                        height=auto_height,
                        margin=dict(l=20, r=20, t=40, b=20),
                        title="Pairplot (hist on diagonal)",
                        legend_title_text="Well" if use_color_by_well else None,
                        hovermode="closest",
                        barmode="overlay"   # critical for per‑well hist overlays
                    )

                    # Hide legend if too many wells
                    if not use_color_by_well:
                        fig.update_layout(showlegend=False)

                    st.plotly_chart(fig, use_container_width=True)

    with tab8:
        st.markdown("### Log Correlation Heatmap")

        if df_long_plot is None or df_long_plot.empty:
            st.warning("No log data available for the selected wells/logs/depth range.")
        else:
            df = df_long_plot.copy()
            df = df[df["Well"].isin(selected_wells) & df["Log"].isin(selected_logs_std)]

            if df.empty:
                st.warning("No data after filtering by selected wells/logs.")
            else:
                # Pivot to wide: rows = (Well, MD), columns = logs, values = Value
                wide = (
                    df.pivot_table(index=["Well", "MD"], columns="Log", values="Value", aggfunc="mean")
                    .reset_index()
                )

                # Only keep selected logs that actually exist in the pivot
                plot_cols = [c for c in selected_logs_std if c in wide.columns]
                if len(plot_cols) < 2:
                    st.warning("Select at least two logs with overlapping data to compute correlations.")
                else:
                    st.markdown("This heatmap shows correlation between the selected logs based on depth samples.")

                    # == Layout: controls (left) & plot (right)
                    left, right = st.columns([1, 3])

                    with left:
                        corr_method = st.selectbox(
                            "Correlation method",
                            options=["pearson", "spearman", "kendall"],
                            index=0,
                            key="corr_method_tab8",
                        )

                        # Size & font (smaller defaults)
                        fig_w = st.slider("Figure width (inches)", 4, 18, 6, key="hm_w_tab8")
                        fig_h = st.slider("Figure height (inches)", 3, 14, 4, key="hm_h_tab8")
                        base_font = st.slider("Base font size", 4, 12, 5, key="hm_basefont_tab8")

                        colormap_options = [
                            "viridis", "plasma", "inferno", "magma", "cividis",
                            "coolwarm", "bwr", "seismic", "Spectral", "YlGnBu", "RdBu"
                        ]
                        cmap_option = st.selectbox(
                            "Colormap", options=colormap_options,
                            index=colormap_options.index("coolwarm"),
                            key="hm_cmap_tab8",
                        )

                        show_font = st.checkbox(
                            "Show annotations (numeric values in cells)",
                            value=True, key="hm_annot_tab8"
                        )
                        if show_font:
                            ann_font = st.slider("Annotation font size", 6, 18, max(8, base_font-1), key="hm_ann_font_tab8")

                        zscore_on = st.checkbox(
                            "Z-score per log (standardize values before correlation)",
                            value=False, key="hm_zscore_tab8"
                        )

                    # Build the matrix to correlate
                    data = wide[plot_cols].copy()
                    if zscore_on:
                        data = (data - data.mean(skipna=True)) / data.std(skipna=True)

                    if data.dropna(how="all").empty:
                        with right:
                            st.warning("No numeric samples available to compute correlation after filtering/standardization.")
                    else:
                        corr_df = data.corr(method=corr_method)

                        with right:
                            fig, ax = plt.subplots(figsize=(fig_w, fig_h))

                            heatmap_kwargs = {
                                "annot": show_font,
                                "fmt": ".2f",
                                "linewidths": 0.5,
                                "cmap": cmap_option,
                                "ax": ax,
                                "vmin": -1, "vmax": 1,
                            }
                            if show_font:
                                heatmap_kwargs["annot_kws"] = {"size": ann_font}

                            sns.heatmap(corr_df, **heatmap_kwargs)

                            # Title & ticks sized to your base font
                            ax.set_title(f"Log Correlation Heatmap ({corr_method.title()})", fontsize=base_font + 2)
                            ax.tick_params(axis='x', labelsize=base_font)
                            ax.tick_params(axis='y', labelsize=base_font)

                            # Keep layout tight for the smaller figure
                            plt.tight_layout()
                            st.pyplot(fig)

    with tab9:
        st.markdown("### Histogram View (per log, all selected wells)")

        # === Two columns: controls (left) and plot (right)
        left, right = st.columns([1, 3])

        # ---- Controls (LEFT) ----
        with left:
            # Build standardized choices for the UI
            log_opts_all_orig = utils.get_logs_for_selected_wells(well_logs, selected_wells)
            log_opts_all = sorted(to_standardized_options(log_opts_all_orig))

            default_one = "Gamma Ray" if "Gamma Ray" in log_opts_all else (log_opts_all[0] if log_opts_all else None)

            chosen_log = st.selectbox(
                "Choose a log",
                options=log_opts_all,
                index=log_opts_all.index(default_one) if (default_one in log_opts_all) else 0,
                key="hv_log_select"
            )

            nbins = st.slider("Bins", 10, 200, 60, 5, key="hv_bins")
            density = st.checkbox("Normalize (density)", value=True, key="hv_density")
            show_outliers = st.checkbox("Show outliers on box", value=False, key="hv_outliers")
            alpha = st.slider("Histogram opacity", 0.1, 1.0, 0.45, 0.05, key="hv_alpha")
            plot_h = st.slider("Plot height (px)", 400, 1200, 720, 20, key="hv_height")


        # ---- Data prep (shared) ----
        dfh = None
        if chosen_log is not None:
            # If the chosen standardized log is already in the long DF, use it; else fetch using original candidates
            if ("Log" in df_long_plot.columns) and (chosen_log in df_long_plot["Log"].unique()):
                dfh = df_long_plot.loc[
                    (df_long_plot["Log"] == chosen_log) & (df_long_plot["Well"].isin(selected_wells)),
                    ["Well", "Value"]
                ].copy()
            else:
                candidate_origs = STD_TO_ORIG.get(chosen_log, [chosen_log])
                dfh = utils.get_logs_data_for_wells_logs(
                    selected_wells, candidate_origs, wells, depth_min, depth_max
                )
                if dfh is not None and not dfh.empty:
                    # Keep only selected wells and the needed columns
                    dfh = dfh.loc[dfh["Well"].isin(selected_wells), ["Well", "Value"]].copy()

        # ---- Plot (RIGHT) ----
        with right:
            if dfh is None or dfh.empty:
                st.info("No data for this log in the selected wells / depth range.")
            else:
                dfh["Value"] = pd.to_numeric(dfh["Value"], errors="coerce")
                dfh = dfh.dropna(subset=["Value"])
                dfh = dfh[dfh["Well"] != "Unknown"]

                wells_present = sorted(dfh["Well"].dropna().unique().tolist())
                if not wells_present:
                    st.info("No valid samples to display.")
                else:
                    palette = px.colors.qualitative.D3
                    color_map = {w: palette[i % len(palette)] for i, w in enumerate(wells_present)}

                    fig = make_subplots(
                        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.28, 0.72],
                        subplot_titles=(f"Box per well - {chosen_log}", f"Histogram - {chosen_log}")
                    )

                    # Row 1: horizontal boxplots (one per well)
                    for wname in wells_present:
                        sub = dfh.loc[dfh["Well"] == wname, "Value"]
                        if sub.empty:
                            continue
                        fig.add_box(
                            x=sub,
                            y=[wname] * len(sub),
                            name=wname,
                            marker_color=color_map[wname],
                            boxpoints="outliers" if show_outliers else False,
                            orientation="h",
                            showlegend=False,
                            row=1, col=1
                        )

                    # Row 2: overlapped histograms
                    histnorm = "probability density" if density else ""
                    for wname in wells_present:
                        sub = dfh.loc[dfh["Well"] == wname, "Value"]
                        if sub.empty:
                            continue
                        fig.add_histogram(
                            x=sub,
                            name=wname,
                            nbinsx=nbins,
                            histnorm=histnorm,
                            marker_color=color_map[wname],
                            opacity=alpha,
                            hovertemplate=f"Well: {wname}<br>{chosen_log}: %{{x}}<br>"+("Density" if density else "Count")+": %{{y}}<extra></extra>",
                            row=2, col=1
                        )

                    fig.update_layout(
                        barmode="overlay",
                        height=plot_h,
                        margin=dict(l=40, r=20, t=60, b=40),
                        legend_title_text="Well"
                    )
                    fig.update_yaxes(title_text="Well", row=1, col=1)
                    fig.update_yaxes(title_text=("Density" if density else "Count"), row=2, col=1)
                    fig.update_xaxes(title_text=f"{chosen_log} (value)", row=2, col=1)

                    st.plotly_chart(fig, use_container_width=True)
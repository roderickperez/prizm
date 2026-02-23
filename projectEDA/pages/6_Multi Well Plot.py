import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import utils
import plotly.express as px
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# === NEW/CHANGED ===
import json, re
from pathlib import Path
from collections import defaultdict

import utils
utils.render_grouped_sidebar_nav()

# ==========================  STREAMLIT CONFIG  ==========================
st.set_page_config(page_title='GeoPython',
                   #page_icon=':bar_chart:',
                   layout='wide',
                   initial_sidebar_state='expanded')

# ============== Petrel connection ===============
petrel_project = utils.get_petrel_connection()

# MENU
#########################################
utils.sidebar_footer(petrel_project, utils.LOGO_PATH, utils.APP_VERSION)

st.title("Multi well Plot")

# === NEW/CHANGED ===
# -------------------- Name-map (orig -> standardized) + helpers --------------------
# -------------------- Name-map (orig -> standardized) + helpers --------------------
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

# newest name-map: {original -> standardized}
nm_cands = sorted(STD_DIR.glob(f"log_name_map__{proj_slug}__*__*.json"))
NAME_MAP = json.loads(nm_cands[-1].read_text(encoding="utf-8")) if nm_cands else {}

# define the flag BEFORE using it
NAME_MAP_EMPTY = (not bool(NAME_MAP))
if NAME_MAP_EMPTY:
    st.info("No standardization map found for this project — showing original log names.")

# reverse: standardized -> [originals]
STD_TO_ORIG = defaultdict(list)
for orig, std in NAME_MAP.items():
    STD_TO_ORIG[std].append(orig)

# Fallbacks / normal behavior (do NOT override later)
if NAME_MAP_EMPTY:
    def eligible_well_names(wells):
        return [w.petrel_name for w in wells]

    def to_standardized_options(original_names: list[str]) -> list[str]:
        # pass-through originals (unique/sorted) in raw mode
        return sorted({str(n) for n in original_names})
else:
    def eligible_well_names(wells) -> list[str]:
        allowed = set(NAME_MAP.keys())
        return [
            w.petrel_name for w in wells
            if any(getattr(lg, "petrel_name", "") in allowed for lg in getattr(w, "logs", []))
        ]

    def to_standardized_options(original_names: list[str]) -> list[str]:
        # only standardized codes that are present in the NAME_MAP
        std_set = set()
        for n in original_names:
            if n in NAME_MAP:
                std_set.add(NAME_MAP[n])
        return sorted(std_set)

# --- Optional: colors via mnemonics + mapping_records (not strictly needed here)
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

def color_for_std_label(std_label: str) -> str | None:
    grp = STD_TO_GROUP.get(std_label, std_label)
    meta = MNEMS.get(grp) or MNEMS.get(std_label) or {}
    style = meta.get("style") if isinstance(meta, dict) else {}
    return (style or {}).get("color")

# Get wells and tops
wells = utils.get_all_wells_flat(petrel_project)
tops_df = utils.load_tops_dataframe(petrel_project)

well_names = [w.petrel_name for w in wells]
well_dict = {w.petrel_name: w for w in wells}
well_logs = {
    w.petrel_name: [getattr(lg, "petrel_name", "") for lg in getattr(w, "logs", [])]
    for w in wells
}

# --- UI Selections ---
col1, col2 = st.columns([1, 4])

elig_names = eligible_well_names(wells)
if not elig_names:
    st.warning("No wells found.")
    st.stop()

with col1:
    selected_well = st.selectbox("Select a Well", elig_names)

    # === NEW/CHANGED === originals for this well
    available_logs_orig = well_logs.get(selected_well, [])

    # depth defaults over all originals
    md_min_default, md_max_default = utils.get_global_md_range([selected_well], available_logs_orig, well_dict)

    multi_key = (selected_well, tuple(available_logs_orig))
    if st.session_state.get("multi_last_key") != multi_key:
        st.session_state["multi_last_key"] = multi_key
        st.session_state["multi_depth_applied"] = (md_min_default, md_max_default)
        for k in ("multi_depth_pending", "multi_top_pending", "multi_base_pending"):
            st.session_state.pop(k, None)

    if "multi_depth_applied" not in st.session_state:
        st.session_state["multi_depth_applied"] = (md_min_default, md_max_default)

    depth_selection_mode = st.radio("Depth selection mode", ["Slider", "Tops"], horizontal=True, key="multi_depth_mode")

    if depth_selection_mode == "Slider":
        with st.form("multi_depth_slider_form"):
            pending = st.session_state.get("multi_depth_pending", st.session_state["multi_depth_applied"])
            temp_range = st.slider(
                "Select depth range (MD)",
                min_value=md_min_default,
                max_value=md_max_default,
                value=tuple(pending),
                step=1,
                key="multi_depth_slider"
            )
            st.session_state["multi_depth_pending"] = temp_range
            if st.form_submit_button("Apply Depth Range"):
                st.session_state["multi_depth_applied"] = temp_range
    else:
        filtered_tops = tops_df[tops_df['Well identifier (Well name)'] == selected_well].copy()
        top_names = sorted(filtered_tops["Surface"].unique())

        with st.form("multi_tops_form"):
            default_top  = st.session_state.get("multi_top_pending",  top_names[0] if top_names else None)
            default_base = st.session_state.get("multi_base_pending", top_names[min(len(top_names)-1, 1)] if top_names else None)

            top_marker = st.selectbox(
                "Top marker",
                top_names,
                index=top_names.index(default_top) if default_top in top_names else 0,
                key="multi_top_marker"
            )
            base_marker = st.selectbox(
                "Base marker",
                top_names,
                index=top_names.index(default_base) if default_base in top_names else min(len(top_names)-1, 1),
                key="multi_base_marker"
            )
            st.session_state["multi_top_pending"]  = top_marker
            st.session_state["multi_base_pending"] = base_marker

            if st.form_submit_button("Apply Tops Range"):
                dm, dx = utils.get_md_from_tops(tops_df, [selected_well], top_marker, base_marker)
                if dm is not None and dx is not None:
                    st.session_state["multi_depth_applied"] = (dm, dx)
                else:
                    st.warning("Selected tops not found in this well. Using full range.")
                    st.session_state["multi_depth_applied"] = (md_min_default, md_max_default)

    depth_min, depth_max = st.session_state["multi_depth_applied"]
    st.caption(f"Applied depth range: {depth_min:.2f}  {depth_max:.2f} m")

    # === NEW/CHANGED ===
    # Load logs for this well into standardized columns (fetching by originals).
    logs_data = {}  # dict: std_label -> DataFrame[MD, std_label]
    well_obj = well_dict[selected_well]

    # build local std -> [orig candidates present in this well]
    STD_TO_LOCAL_ORIG = defaultdict(list)
    for orig in available_logs_orig:
        std = NAME_MAP.get(orig, orig)
        STD_TO_LOCAL_ORIG[std].append(orig)

    # choose the candidate original with the MOST valid samples in the current depth window
    for std_label, orig_list in STD_TO_LOCAL_ORIG.items():
        best_df = None
        best_count = -1
        for orig in orig_list:
            log_obj = next((lg for lg in well_obj.logs if lg.petrel_name == orig), None)
            if log_obj is None:
                continue
            df = log_obj.as_dataframe()
            if {"MD", "Value"}.issubset(df.columns):
                df2 = df[(df["MD"] >= depth_min) & (df["MD"] <= depth_max)][["MD", "Value"]].dropna()
                cnt = int(df2["Value"].notna().sum())
                if cnt > best_count:
                    best_count = cnt
                    best_df = df2
        if best_df is not None and not best_df.empty:
            logs_data[std_label] = best_df.rename(columns={"Value": std_label})

    def merge_on_md(names, strategy: str = "asof"):
        """
        Align on MD across the chosen logs.

        strategy='asof'  -> nearest-neighbor per MD within st.session_state['mw_md_tol'] (meters).
        strategy='inner' -> exact MD intersection only.
        strategy='grid'  -> resample each series to a common MD grid (step = mw_grid_step).
        """
        # --- helper for grid resampling ---
        def _resample_to_grid(grid_md: np.ndarray, df: pd.DataFrame) -> pd.Series:
            d = df.copy()
            d["MD"] = pd.to_numeric(d["MD"], errors="coerce")
            d = d.dropna(subset=["MD"]).sort_values("MD")
            vcols = [c for c in d.columns if c != "MD"]
            if not vcols:
                return pd.Series(index=grid_md, dtype=float)
            v = pd.to_numeric(d[vcols[0]], errors="coerce")
            m = np.isfinite(d["MD"]) & np.isfinite(v)
            if m.sum() < 2:
                # not enough points to interpolate
                return pd.Series(index=grid_md, dtype=float)
            return pd.Series(np.interp(grid_md, d.loc[m, "MD"].to_numpy(), v.loc[m].to_numpy()), index=grid_md)

        # --- collect + sanitize inputs ---
        dfs = []
        for n in names:
            df = logs_data.get(n)
            if df is None or df.empty:
                continue
            d = df.copy()
            d["MD"] = pd.to_numeric(d["MD"], errors="coerce")
            d = d.dropna(subset=["MD"]).sort_values("MD")
            vcols = [c for c in d.columns if c != "MD"]
            if vcols:
                d[vcols[0]] = pd.to_numeric(d[vcols[0]], errors="coerce")
            dfs.append((n, d[["MD"] + vcols]))

        if not dfs:
            return pd.DataFrame()

        # --- grid resampling path ---
        if strategy == "grid":
            lo, hi = st.session_state.get("multi_depth_applied", (None, None))
            if lo is None or hi is None or not np.isfinite([lo, hi]).all():
                return pd.DataFrame()
            step = float(st.session_state.get("mw_grid_step", 0.5))
            step = max(step, 1e-6)
            grid = np.arange(float(lo), float(hi) + step * 0.5, step)

            out = pd.DataFrame({"MD": grid})
            for n, d in dfs:
                out[n] = _resample_to_grid(grid, d).values
            return out

        # --- exact intersection path ---
        if strategy == "inner" or ("mw_md_tol" not in st.session_state):
            out = dfs[0][1]
            for _, d in dfs[1:]:
                out = pd.merge(out, d, on="MD", how="inner")
            return out

        # --- nearest neighbor path (asof) ---
        tol = float(st.session_state.get("mw_md_tol", 2.0))
        ref_idx = int(np.argmax([len(d[1]) for d in dfs]))
        _, ref = dfs[ref_idx]
        out = ref.copy()
        for i, (n, d) in enumerate(dfs):
            if i == ref_idx:
                continue
            out = pd.merge_asof(
                out.sort_values("MD"),
                d.sort_values("MD"),
                on="MD",
                direction="nearest",
                tolerance=tol,
            )
        return out




with col2:
    # Tabs
    tab1, tab2 = st.tabs(["Bi-Variable Plots", "Multi-Variable Plots"])

    with tab1:
        ctl_col, plt_col = st.columns([1, 3])

        with ctl_col:
            plot_type = st.selectbox(
                "Plot Type",
                [
                    "Classic Scatter",
                    "2D Histogram (Heatmap)",
                    "OLS Trend",
                    "Joint KDE",
                    "Hexbin Jointplot",
                    "Joint + Marginal Hist",
                    "Scatter + Hist2D + Contours",
                    "Scatter + Marginal Ticks",
                    "Linear Reg + Marginals",
                    "Smooth KDE + Marginal Hist",
                ],
                index=0,
                key="mw_plot_type",
            )

            # Align samples across logs
            align_choice = st.radio(
                "Align samples by",
                ["Nearest MD (tolerance)", "Exact MD intersection", "Resample to grid"],
                index=0, horizontal=True, key="mw_align"
            )
            if align_choice == "Nearest MD (tolerance)":
                st.slider("MD match tolerance (m)", 0.0, 10.0, 2.0, 0.1, key="mw_md_tol")
                st.caption("Samples within this MD distance will be aligned across logs.")
            elif align_choice == "Resample to grid":
                st.slider("Grid step (m)", 0.05, 5.0, 0.5, 0.05, key="mw_grid_step")
                st.caption("Each log is linearly resampled to a common MD grid.")


            # === NEW/CHANGED === standardized options
            log_options = list(logs_data.keys())

            # --- NEW: family filter for available logs ---
            groups_avail = sorted({STD_TO_GROUP.get(s, "(unlabeled)") for s in log_options})
            if groups_avail:
                sel_groups = st.multiselect(
                    "Filter logs by family", groups_avail, default=groups_avail,
                    key="mw_groups"
                )
                log_options = [s for s in log_options if STD_TO_GROUP.get(s, "(unlabeled)") in set(sel_groups)]

            if not log_options:
                st.warning("No logs available in the selected depth range.")
                st.stop()

            x_axis = st.selectbox("X-axis", log_options, index=0, key="mw_x")
            y_axis = st.selectbox("Y-axis", log_options, index=min(1, len(log_options)-1), key="mw_y")

            with st.expander("Plot options", expanded=False):
                if plot_type == "Classic Scatter":
                    opacity = st.slider("Opacity", 0.1, 1.0, 0.7, 0.05, key="mw_sc_op")
                    size_val = st.slider("Marker size", 2, 16, 6, 1, key="mw_sc_sz")

                    # Color mode
                    color_mode = st.radio(
                        "Color mode",
                        ["Single color", "Color by variable"],
                        index=0,
                        key="mw_sc_color_mode"
                    )

                    if color_mode == "Single color":
                        single_color = st.color_picker("Marker color", value="#1f77b4", key="mw_sc_single_color")
                        color_var = None
                        cmap = None
                        show_cbar = False
                    else:
                        color_var = st.selectbox("Color by", log_options, index=0, key="mw_sc_color_var")
                        cs = sorted(set(px.colors.named_colorscales()))
                        default_idx = cs.index("viridis") if "viridis" in cs else 0
                        cmap = st.selectbox("Colorscale", cs, index=default_idx, key="mw_sc_cmap")
                        show_cbar = st.checkbox("Show colorbar", value=True, key="mw_sc_showcbar")

                    # Trendline (optional)
                    add_trend = st.checkbox("Add trendline", value=False, key="mw_sc_trend_on")
                    if add_trend:
                        model = st.selectbox("Model", ["Linear", "Exponential", "Polynomial"], index=0, key="mw_sc_tr_model")
                        if model == "Polynomial":
                            degree = st.slider("Polynomial degree", 1, 6, 2, 1, key="mw_sc_tr_degree")
                        line_color = st.text_input("Line color", "darkblue", key="mw_sc_tr_col")
                        line_dash = st.selectbox("Line style", ["solid", "dash", "dot", "dashdot"], index=0, key="mw_sc_tr_dash")
                        st.button("Fit trendline", key="mw_sc_fit_button")

                elif plot_type == "2D Histogram (Heatmap)":
                    nbx = st.slider("Bins (X)", 5, 200, 50, 1, key="mw_hm_nbx")
                    nby = st.slider("Bins (Y)", 5, 200, 50, 1, key="mw_hm_nby")
                    z_histfunc = st.selectbox("Aggregation", ["count", "avg", "sum", "min", "max"], index=0, key="mw_hm_func")
                    cs = sorted(set(px.colors.named_colorscales()))
                    default_idx = cs.index("viridis") if "viridis" in cs else 0
                    cmap = st.selectbox("Color scale", cs, index=default_idx, key="mw_hm_cmap")

                elif plot_type == "OLS Trend":
                    opacity = st.slider("Opacity", 0.1, 1.0, 0.65, 0.05, key="mw_ols_op")
                    trend_color = st.text_input("Trendline color", "darkblue", key="mw_ols_col")

                elif plot_type == "Joint KDE":
                    levels = st.slider("KDE levels", 3, 20, 10, 1, key="mw_kde_lv")
                    fill = st.checkbox("Fill contours", value=False, key="mw_kde_fill")

                elif plot_type == "Hexbin Jointplot":
                    hex_color = st.text_input("Hexbin color", "#4CB391", key="mw_hex_col")

                elif plot_type == "Joint + Marginal Hist":
                    bins = st.slider("Bins", 10, 200, 50, 5, key="mw_jh_bins")
                    cmap = st.selectbox("Colormap", ["mako", "rocket", "viridis", "plasma", "magma", "cividis", "inferno", "turbo",
                                                      "cubehelix", "crest", "flare", "icefire", "vlag", "Spectral", "coolwarm",
                                                      "Greys", "Blues", "Greens", "Oranges", "Reds", "Purples",
                                                      "twilight", "twilight_shifted", "rainbow", "jet"],
                                                      index=0, key="mw_jh_cmap")
                    y_log = st.checkbox("Log scale Y", value=False, key="mw_jh_log")
                    show_cbar = st.checkbox("Show colorbar", value=True, key="mw_jh_cbar")

                elif plot_type == "Scatter + Hist2D + Contours":
                    bins = st.slider("Bins", 10, 200, 50, 5, key="mw_shc_bins")
                    levels = st.slider("Contour levels", 3, 20, 5, 1, key="mw_shc_lv")
                    pt_sz = st.slider("Point size", 2, 20, 5, 1, key="mw_shc_sz")
                    cmap = st.selectbox("Colormap", ["mako", "rocket", "viridis", "plasma", "magma", "cividis", "inferno", "turbo",
                                                      "cubehelix", "crest", "flare", "icefire", "vlag", "Spectral", "coolwarm",
                                                      "twilight", "twilight_shifted", "rainbow", "jet"],
                                                      index=0, key="mw_shc_cmap")
                    contour_color = st.color_picker("Contour line color", value="#000000", key="mw_shc_linecolor")
                    contour_lw = st.slider("Contour line width", 0.5, 4.0, 1.0, 0.1, key="mw_shc_lw")

                elif plot_type == "Scatter + Marginal Ticks":
                    size_by = st.selectbox("Size by (optional)", ["None"] + log_options, index=0, key="mw_rug_sizeby")
                    alpha = st.slider("Opacity", 0.1, 1.0, 0.6, 0.05, key="mw_rug_alpha")

                elif plot_type == "Linear Reg + Marginals":
                    color = st.text_input("Line color", "m", key="mw_reg_c")

                elif plot_type == "Smooth KDE + Marginal Hist":
                    levels = st.slider("Levels", 10, 200, 100, 5, key="mw_smooth_lv")
                    bins = st.slider("Marginal bins", 5, 100, 25, 1, key="mw_smooth_bins")
                    fill = st.checkbox("Fill", value=True, key="mw_smooth_fill")

                    # NEW: colormap + colorbar controls for the KDE
                    cmaps_extended = [
                        "mako", "rocket", "viridis", "plasma", "magma", "cividis", "inferno", "turbo",
                        "cubehelix", "crest", "flare", "icefire", "vlag", "Spectral", "coolwarm",
                        "Greys", "Blues", "Greens", "Oranges", "Reds", "Purples",
                        "twilight", "twilight_shifted", "rainbow", "jet"
                    ]

                    kde_cmap = st.selectbox(
                        "KDE colormap",
                        cmaps_extended,
                        index=(cmaps_extended.index("rocket") if "rocket" in cmaps_extended else 0),
                        key="mw_smooth_cmap"
                    )
                    show_kde_cbar = st.checkbox("Show KDE colorbar", value=True, key="mw_smooth_cbar")
                    cbar_label = st.text_input("Colorbar label", "Density", key="mw_smooth_cbar_label")

                    # NEW: separate colors for marginal hist bars (X and Y)
                    x_bar_color = st.color_picker("X-marginal bar color", value="#03051A", key="mw_smooth_xbar")
                    y_bar_color = st.color_picker("Y-marginal bar color", value="#03051A", key="mw_smooth_ybar")

                # Shared - height & width
                plot_h = st.slider("Plot height (px)", 300, 1400, 700, 50, key="mw_plot_h")
                plot_w = st.slider("Plot width (px)",  300, 2000, 900, 50, key="mw_plot_w")
                _DPI = 100
                h_in = plot_h / _DPI
                w_in = plot_w / _DPI

        with plt_col:
            def _lims(series_like):
                if isinstance(series_like, pd.DataFrame):
                    if series_like.shape[1] >= 1:
                        series_like = series_like.iloc[:, 0]
                    else:
                        return None, None
                if isinstance(series_like, pd.Series):
                    s = series_like
                elif isinstance(series_like, (list, tuple, np.ndarray)):
                    s = pd.Series(series_like)
                else:
                    return None, None
                s = pd.to_numeric(s, errors="coerce").dropna()
                if s.empty:
                    return None, None
                a, b = float(s.min()), float(s.max())
                if a == b:
                    pad = 1.0 if a == 0 else abs(a) * 0.01
                    a, b = a - pad, b + pad
                return a, b

            columns_needed = [x_axis, y_axis]
            if (
                st.session_state.get("mw_sc_color_mode") == "Color by variable"
                and st.session_state.get("mw_sc_color_var")
            ):
                color_var = st.session_state["mw_sc_color_var"]
                if color_var not in columns_needed:
                    columns_needed.append(color_var)

            strategy = (
                "asof" if align_choice == "Nearest MD (tolerance)"
                else "grid" if align_choice == "Resample to grid"
                else "inner"
            )

            df_xy = merge_on_md(columns_needed, strategy=strategy)

            dfp_base = df_xy[[x_axis, y_axis]].dropna()
            dfp = dfp_base.copy()

            if plot_type == "Classic Scatter":
                # ensure eq_text defined even if trendline fails
                eq_text = ""  # === NEW/CHANGED ===

                use_color_by = (
                    st.session_state.get("mw_sc_color_mode") == "Color by variable"
                    and st.session_state.get("mw_sc_color_var") in df_xy.columns
                )
                color_var = st.session_state.get("mw_sc_color_var")
                needed_for_plot = [x_axis, y_axis] + ([color_var] if use_color_by else [])
                needed_for_plot = list(dict.fromkeys([c for c in needed_for_plot if c is not None]))

                dfp = df_xy[needed_for_plot].copy()
                dfp = dfp.loc[:, ~dfp.columns.duplicated()]
                dfp = dfp.dropna(subset=needed_for_plot)

                if dfp.empty:
                    st.warning("No overlapping depth samples for the selected logs (including the color-by variable, if selected).")
                else:
                    xmin, xmax = _lims(dfp[x_axis])
                    ymin, ymax = _lims(dfp[y_axis])

                    if st.session_state.get("mw_sc_color_mode") == "Single color":
                        single_color = st.session_state.get("mw_sc_single_color", "#1f77b4")
                        fig = px.scatter(dfp, x=x_axis, y=y_axis, opacity=opacity,
                                         color_discrete_sequence=[single_color])
                        fig.update_traces(marker=dict(size=size_val, showscale=False))
                    else:
                        if use_color_by:
                            cmap = st.session_state.get("mw_sc_cmap", "viridis")
                            show_cbar = st.session_state.get("mw_sc_showcbar", True)
                            fig = px.scatter(dfp, x=x_axis, y=y_axis, opacity=opacity,
                                             color=color_var, color_continuous_scale=cmap)
                            fig.update_traces(marker=dict(size=size_val, showscale=show_cbar))
                        else:
                            st.info("Selected color-by variable isn't available after aligning samples; using single color instead.")
                            fig = px.scatter(dfp, x=x_axis, y=y_axis, opacity=opacity,
                                             color_discrete_sequence=["#1f77b4"])
                            fig.update_traces(marker=dict(size=size_val, showscale=False))

                    if xmin is not None and xmax is not None:
                        fig.update_xaxes(range=[xmin, xmax])
                    if ymin is not None and ymax is not None:
                        fig.update_yaxes(range=[ymin, ymax])

                    # Trendline overlay + equation (LaTeX)
                    if st.session_state.get("mw_sc_trend_on", False):
                        xv = pd.to_numeric(dfp[x_axis], errors="coerce").to_numpy()
                        yv = pd.to_numeric(dfp[y_axis], errors="coerce").to_numpy()
                        mask = np.isfinite(xv) & np.isfinite(yv)
                        xv, yv = xv[mask], yv[mask]

                        if xv.size >= 2:
                            xs = np.linspace(np.nanmin(xv), np.nanmax(xv), 200)

                            mdl       = st.session_state.get("mw_sc_tr_model", "Linear")
                            line_col  = st.session_state.get("mw_sc_tr_col", "darkblue")
                            line_dash = st.session_state.get("mw_sc_tr_dash", "solid")

                            ys = None
                            if mdl == "Linear":
                                m, b = np.polyfit(xv, yv, 1)
                                ys = m * xs + b
                                y_pred = m * xv + b
                                ss_res = np.sum((yv - y_pred)**2)
                                ss_tot = np.sum((yv - np.mean(yv))**2)
                                r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
                                eq_text = rf"$y = {m:.3g}x + {b:.3g}\;\;(R^2={r2:.3f})$"

                            elif mdl == "Polynomial":
                                deg = int(st.session_state.get("mw_sc_tr_degree", 2))
                                coeffs = np.polyfit(xv, yv, deg)
                                ys = np.polyval(coeffs, xs)
                                y_pred = np.polyval(coeffs, xv)
                                ss_res = np.sum((yv - y_pred)**2)
                                ss_tot = np.sum((yv - np.mean(yv))**2)
                                r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan

                                terms = []
                                for i, c in enumerate(coeffs[:-1]):
                                    p = deg - i
                                    if p == 1:
                                        terms.append(rf"{c:.3g}x")
                                    else:
                                        terms.append(rf"{c:.3g}x^{p}")
                                eq_text = rf"$y = {' + '.join(terms)} + {coeffs[-1]:.3g}\;\;(R^2={r2:.3f})$"

                            else:  # Exponential: y = a e^{b x}
                                pos = yv > 0
                                if pos.sum() >= 2:
                                    m, b = np.polyfit(xv[pos], np.log(yv[pos]), 1)
                                    a = np.exp(b)
                                    ys = a * np.exp(m * xs)
                                    y_pred = a * np.exp(m * xv[pos])
                                    ss_res = np.sum((yv[pos] - y_pred)**2)
                                    ss_tot = np.sum((yv[pos] - np.mean(yv[pos]))**2)
                                    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
                                    eq_text = rf"$y = {a:.3g}e^{{{m:.3g}x}}\;\;(R^2={r2:.3f})$"
                                else:
                                    st.warning("Exponential fit requires positive Y values.")

                            if ys is not None:
                                fig.add_scatter(
                                    x=xs, y=ys, mode="lines",
                                    line=dict(color=line_col, dash=line_dash, width=2),
                                    name="Trendline"
                                )
                                fig.add_annotation(
                                    xref="paper", yref="paper", x=0.02, y=0.98,
                                    xanchor="left", yanchor="top",
                                    text=eq_text, showarrow=False,
                                    bgcolor="rgba(255,255,255,0.7)"
                                )

                    fig.update_layout(height=plot_h, width=plot_w, margin=dict(l=40, r=20, t=40, b=40))
                    st.plotly_chart(fig, use_container_width=False)

                    if st.session_state.get("mw_sc_trend_on", False) and eq_text:
                        st.latex(eq_text.strip("$"))

            elif plot_type == "2D Histogram (Heatmap)":
                dfp = dfp_base
                xmin, xmax = _lims(dfp[x_axis])
                fig = px.density_heatmap(
                    dfp, x=x_axis, y=y_axis,
                    nbinsx=nbx, nbinsy=nby,
                    histfunc=None if z_histfunc == "count" else z_histfunc,
                    color_continuous_scale=cmap,
                    range_x=[xmin, xmax] if xmin is not None and xmax is not None else None,
                )
                if xmin is not None and xmax is not None:
                    fig.update_xaxes(range=[xmin, xmax])
                fig.update_layout(height=plot_h, width=plot_w, margin=dict(l=40, r=20, t=40, b=40))
                st.plotly_chart(fig, use_container_width=False)

            elif plot_type == "OLS Trend":
                dfp = dfp_base
                xmin, xmax = _lims(dfp[x_axis])
                ymin, ymax = _lims(dfp[y_axis])

                try:
                    fig = px.scatter(dfp, x=x_axis, y=y_axis,
                                     opacity=opacity,
                                     trendline="ols",
                                     trendline_color_override=trend_color)
                    if xmin is not None and xmax is not None:
                        fig.update_xaxes(range=[xmin, xmax])
                    if ymin is not None and ymax is not None:
                        fig.update_yaxes(range=[ymin, ymax])
                    fig.update_layout(height=plot_h, width=plot_w, margin=dict(l=40, r=20, t=40, b=40))
                    st.plotly_chart(fig, use_container_width=False)
                except Exception:
                    st.info("OLS trendline requires `statsmodels`.")
                    fig = px.scatter(dfp, x=x_axis, y=y_axis, opacity=opacity)
                    if xmin is not None and xmax is not None:
                        fig.update_xaxes(range=[xmin, xmax])
                    fig.update_layout(height=plot_h, width=plot_w, margin=dict(l=40, r=20, t=40, b=40))
                    st.plotly_chart(fig, use_container_width=False)

            elif plot_type == "Joint KDE":
                kind_kwargs = {"fill": True} if fill else {}
                g = sns.jointplot(data=dfp, x=x_axis, y=y_axis, kind="kde", levels=levels, height=h_in, **kind_kwargs)
                st.pyplot(g.fig)

            elif plot_type == "Hexbin Jointplot":
                g = sns.jointplot(data=dfp, x=x_axis, y=y_axis, kind="hex", color=hex_color, height=h_in)
                st.pyplot(g.fig)

            elif plot_type == "Joint + Marginal Hist":
                g = sns.JointGrid(data=dfp, x=x_axis, y=y_axis, marginal_ticks=True)
                g.fig.set_size_inches(w_in, h_in)
                if y_log:
                    g.ax_joint.set(yscale="log")
                cax = None
                if show_cbar:
                    cax = g.figure.add_axes([.15, .55, .02, .2])
                g.plot_joint(sns.histplot, bins=bins, pthresh=.8, cmap=cmap, cbar=show_cbar, cbar_ax=cax)
                g.plot_marginals(sns.histplot, element="step", color="#03012d")
                st.pyplot(g.fig)

            elif plot_type == "Scatter + Hist2D + Contours":
                f, ax = plt.subplots(figsize=(w_in, h_in))
                sns.scatterplot(data=dfp, x=x_axis, y=y_axis, s=pt_sz, color=".15", ax=ax)
                cmap_obj = cm.get_cmap(cmap)
                sns.histplot(
                    data=dfp, x=x_axis, y=y_axis,
                    bins=bins, pthresh=.1, cmap=cmap_obj, ax=ax, cbar=False
                )
                sns.kdeplot(
                    data=dfp, x=x_axis, y=y_axis,
                    levels=levels,
                    color=st.session_state.get("mw_shc_linecolor", "#FFFFFF"),
                    linewidths=st.session_state.get("mw_shc_lw", 1.0),
                    ax=ax
                )
                quadmeshes = [c for c in ax.collections if isinstance(c, mpl.collections.QuadMesh)]
                if quadmeshes:
                    mappable = quadmeshes[-1]
                    mappable.set_cmap(cmap_obj)
                    cbar = f.colorbar(mappable, ax=ax)
                    cbar.set_label("Counts", rotation=270, labelpad=15)
                ax.set_title(f"{x_axis} vs {y_axis}")
                st.pyplot(f)

            elif plot_type == "Scatter + Marginal Ticks":
                g = sns.JointGrid(data=dfp, x=x_axis, y=y_axis, space=0, ratio=17)
                g.fig.set_size_inches(w_in, h_in)
                size_by = st.session_state.get("mw_rug_sizeby", "None")
                alpha = st.session_state.get("mw_rug_alpha", 0.6)
                if size_by != "None" and size_by in df_xy.columns:
                    sdat = pd.to_numeric(df_xy[size_by], errors="coerce").reindex(dfp.index)
                    g.plot_joint(sns.scatterplot, size=sdat, sizes=(30, 120), color="g", alpha=alpha, legend=False)
                else:
                    g.plot_joint(sns.scatterplot, color="g", alpha=alpha, legend=False)
                g.plot_marginals(sns.rugplot, height=1, color="g", alpha=alpha)
                st.pyplot(g.fig)

            elif plot_type == "Linear Reg + Marginals":
                g = sns.jointplot(data=dfp, x=x_axis, y=y_axis, kind="reg", truncate=False, color=st.session_state.get("mw_reg_c", "m"), height=h_in)
                try:
                    xv = dfp[x_axis].to_numpy(); yv = dfp[y_axis].to_numpy()
                    if xv.size >= 2:
                        m, b = np.polyfit(xv, yv, 1)
                        y_pred = m * xv + b
                        ss_res = np.sum((yv - y_pred) ** 2)
                        ss_tot = np.sum((yv - yv.mean()) ** 2)
                        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
                        txt = f"y = {m:.3g}x + {b:.3g}\nR² = {r2:.3f}"
                        g.ax_joint.text(0.05, 0.95, txt, transform=g.ax_joint.transAxes,
                                        ha="left", va="top",
                                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
                except Exception:
                    pass
                st.pyplot(g.fig)

            elif plot_type == "Smooth KDE + Marginal Hist":
                g = sns.JointGrid(data=dfp, x=x_axis, y=y_axis, space=0)
                g.fig.set_size_inches(w_in, h_in)
                kde = sns.kdeplot(
                    data=dfp, x=x_axis, y=y_axis,
                    fill=st.session_state.get("mw_smooth_fill", True),
                    thresh=0, levels=st.session_state.get("mw_smooth_lv", 100),
                    cmap=st.session_state.get("mw_smooth_cmap", "rocket"),
                    ax=g.ax_joint
                )
                if st.session_state.get("mw_smooth_cbar", True):
                    mappables = [c for c in g.ax_joint.collections if hasattr(c, "get_cmap")]
                    if mappables:
                        mappable = mappables[-1]
                        try:
                            mappable.set_cmap(cm.get_cmap(st.session_state.get("mw_smooth_cmap", "rocket")))
                        except Exception:
                            pass
                        from mpl_toolkits.axes_grid1 import make_axes_locatable
                        divider = make_axes_locatable(g.ax_marg_y)
                        cax = divider.append_axes("right", size="20%", pad=0.3)
                        cbar = g.fig.colorbar(mappable, cax=cax)
                        cbar.set_label(st.session_state.get("mw_smooth_cbar_label", "Density"), rotation=270, labelpad=12)
                sns.histplot(
                    x=dfp[x_axis],
                    bins=st.session_state.get("mw_smooth_bins", 25),
                    color=st.session_state.get("mw_smooth_xbar", "#03051A"),
                    alpha=1,
                    ax=g.ax_marg_x,
                )
                sns.histplot(
                    y=dfp[y_axis],
                    bins=st.session_state.get("mw_smooth_bins", 25),
                    color=st.session_state.get("mw_smooth_ybar", "#03051A"),
                    alpha=1,
                    ax=g.ax_marg_y,
                    orientation="horizontal",
                )
                st.pyplot(g.fig)

    with tab2:
        st.subheader("Multi-Variable Plots")

        if not logs_data:
            st.warning("No logs available in the selected depth range.")
        else:
            mv_cols = list(logs_data.keys())  # === NEW/CHANGED === standardized column names

            left, right = st.columns([1, 3])

            def _pick_default(name: str, options: list, fallback_idx: int = 0):
                if name in options:
                    return options.index(name)
                return min(fallback_idx, len(options) - 1) if options else 0

            with left:
                # === NEW/CHANGED === set standardized defaults
                default_x_name = "RHO_STD"
                default_y_name = "DT_STD"
                default_3_name = "GR_STD"

                n_vars = st.selectbox("Number of logs", [3, 4], index=0, key="mv_n")

                x_idx = _pick_default(default_x_name, mv_cols, 0)
                x_log = st.selectbox("X-axis", mv_cols, index=x_idx, key="mv_x")

                y_candidates = [c for c in mv_cols if c != x_log]
                y_idx = _pick_default(default_y_name, y_candidates, 0)
                y_log = st.selectbox("Y-axis", y_candidates, index=y_idx if y_candidates else 0, key="mv_y")

                remaining = [c for c in mv_cols if c not in (x_log, y_log)]
                v3_idx = _pick_default(default_3_name, remaining, 0)
                var3 = st.selectbox("3rd variable", remaining, index=v3_idx if remaining else 0, key="mv_var3")
                role3 = st.selectbox("Map 3rd variable to", ["color", "size"], index=0, key="mv_role3")

                if n_vars == 3 and role3 == "color":
                    with st.expander("Colorbar options", expanded=False):
                        cs = sorted(set(px.colors.named_colorscales()))
                        default_idx = cs.index("viridis") if "viridis" in cs else 0
                        st.selectbox("Colormap", cs, index=default_idx, key="mv3_cmap")
                        st.text_input("Colorbar title", value=var3, key="mv3_cbar_title")
                        st.checkbox("Show colorbar", value=True, key="mv3_show_cbar")

                var4 = None
                role4 = None
                if n_vars == 4:
                    remaining2 = [c for c in remaining if c != var3]
                    var4 = st.selectbox("4th variable", remaining2, key="mv_var4")
                    role4 = st.selectbox("Map 4th variable to", ["color", "size"], index=1, key="mv_role4")

                plot_size_2 = st.slider("Plot size (px)", 300, 1200, 700, 50, key="mv_plot_size")

            with right:
                use_names = [x_log, y_log, var3] + ([var4] if var4 else [])
                strategy = "asof" if align_choice == "Nearest MD (tolerance)" else "inner"
                dfm = merge_on_md(use_names, strategy=strategy)

                if dfm.empty:
                    st.warning("No overlapping depth samples for the selected logs.")
                else:
                    color_kw = {}
                    size_kw = {}

                    if n_vars == 3:
                        if role3 == "color":
                            color_kw["color"] = var3
                        else:
                            size_kw["size"] = var3
                    else:
                        if role3 == "color":
                            color_kw["color"] = var3
                        else:
                            size_kw["size"] = var3
                        if var4:
                            if role4 == "color":
                                color_kw["color"] = var4  # last wins
                            else:
                                size_kw["size"] = var4

                    if n_vars == 3 and role3 == "color":
                        cmap_sel = st.session_state.get("mv3_cmap", "viridis")
                        fig = px.scatter(dfm, x=x_log, y=y_log, color=var3,
                                         color_continuous_scale=cmap_sel)
                        if not st.session_state.get("mv3_show_cbar", True):
                            fig.update_layout(coloraxis_showscale=False)
                        else:
                            fig.update_layout(coloraxis_colorbar=dict(
                                title=st.session_state.get("mv3_cbar_title", var3)
                            ))
                    else:
                        fig = px.scatter(dfm, x=x_log, y=y_log, **color_kw, **size_kw)

                    fig.update_layout(
                        height=plot_size_2,
                        margin=dict(l=40, r=20, t=40, b=40)
                    )
                    st.plotly_chart(fig, use_container_width=True)
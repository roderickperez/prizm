# N:\_USER_GLOBAL\PETREL\Prizm\wf\EDA\pages\Automatic EDA.py

import streamlit as st
import numpy as np
import pandas as pd
import utils
from utils import filter_std_by_groups, format_std_with_group

import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno

from io import BytesIO
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

import base64
from PIL import Image as PILImage
from reportlab.platypus import Image as RLImage
from reportlab.lib.units import inch

import utils
utils.render_grouped_sidebar_nav()

# Optional: detect plotly image support
try:
    import kaleido  # noqa: F401
    _HAS_KALEIDO = True
except Exception:
    _HAS_KALEIDO = False
# ==========================  STREAMLIT CONFIG  ==========================
st.set_page_config(
    page_title="GeoPython | Automatic EDA",
    layout="wide",
    initial_sidebar_state="expanded",
)

# === Standardization (name-map) helpers (robust loader) ===
import json, re
from pathlib import Path
from collections import defaultdict
from utils import try_get_project_name  # ensure slug matches Standardization page

def _slugify(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\(.*?\)", " ", s)
    s = s.replace("\\", " ").replace("/", " ").replace("_", " ").replace("-", " ")
    s = re.sub(r"[^A-Za-z0-9\s\.]+", " ", s)   # keep dots for *.pet projects
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "project"

# Use the same project-name logic as the Standardization page for consistent slugs
_proj = utils.get_petrel_connection()
proj_raw  = try_get_project_name(_proj) or _proj.get_current_project_name()
proj_slug = _slugify(proj_raw)

STD_DIR   = Path.home() / "Cegal" / "Prizm" / "geoPython" / proj_slug / "logStandarization"

def _load_std_artifacts():
    """
    Load:
      - NAME_MAP: {original_name -> standardized_code} (preferred)
      - STD_TO_GROUP: {standardized_code -> family}
      - STD_TO_ORIG: {standardized_code -> [original_names]}
    Strategy:
      1) Try latest log_name_map__*.json
      2) If missing/empty, derive name-map from latest log_mapping_records__*.json
    Returns: (NAME_MAP, STD_TO_GROUP, STD_TO_ORIG, source_summary_str)
    """
    name_map = {}
    std_to_group = {}
    std_to_orig = defaultdict(list)

    nm_files   = sorted(STD_DIR.glob(f"log_name_map__{proj_slug}__*__*.json"))
    rec_files  = sorted(STD_DIR.glob(f"log_mapping_records__{proj_slug}__*__*.json"))
    sources = []

    # 1) Try simple name-map first
    if nm_files:
        try:
            name_map = json.loads(nm_files[-1].read_text(encoding="utf-8"))
            sources.append(f"name_map: {nm_files[-1].name}")
        except Exception:
            name_map = {}
            sources.append(f"name_map: {nm_files[-1].name} (failed to read)")

    # 2) Always parse records for STD_TO_GROUP; also derive name_map if still empty
    if rec_files:
        try:
            records = json.loads(rec_files[-1].read_text(encoding="utf-8"))
            sources.append(f"records: {rec_files[-1].name}")
            for r in records:
                std = str(r.get("Standardized", "")).strip()
                grp = str(r.get("Group", "")).strip()
                if std and grp and grp != "(unmatched)" and std not in std_to_group:
                    std_to_group[std] = grp
            if not name_map:
                # derive name_map only for rows actually marked "Use in project"
                for r in records:
                    if bool(r.get("Use in project", True)):
                        orig = str(r.get("Original log name", "")).strip()
                        std  = str(r.get("Standardized", "")).strip()
                        if orig and std:
                            name_map[orig] = std
                sources.append("derived name_map from records")
        except Exception:
            sources.append(f"records: {rec_files[-1].name} (failed to read)")

    # Build reverse index
    for orig, std in name_map.items():
        std_to_orig[std].append(orig)

    return name_map, std_to_group, std_to_orig, " | ".join(sources) if sources else "no stamped files found"

NAME_MAP, STD_TO_GROUP, STD_TO_ORIG, _STD_SRC = _load_std_artifacts()

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

def _fmt_std(std: str) -> str:
    grp = STD_TO_GROUP.get(std, "")
    return f"{std} — {grp}" if grp else std

def to_standardized_options(original_names: list[str]) -> list[str]:
    """Only show standardized codes for names explicitly marked 'Use in project' (present in NAME_MAP)."""
    std_set = set()
    for n in original_names:
        if n in NAME_MAP:   # only mapped (Use in project = True)
            std_set.add(NAME_MAP[n])
    return sorted(std_set)

def eligible_well_names(wells) -> list[str]:
    """Only wells that have at least one original log present in NAME_MAP keys (i.e., mapped)"""
    allowed = set(NAME_MAP.keys())
    return [
        w.petrel_name for w in wells
        if any(getattr(lg, "petrel_name", "") in allowed for lg in getattr(w, "logs", []))
    ]

# ============== Petrel connection & Footer ===============
petrel_project = utils.get_petrel_connection()
utils.sidebar_footer(petrel_project, utils.LOGO_PATH, utils.APP_VERSION)

with st.expander("Standardization status (loaded files)", expanded=False):
    st.caption(f"Project slug: `{proj_slug}`")
    st.caption(f"STD_DIR: `{STD_DIR}`")
    st.write(f"Loaded: {_STD_SRC or '—'}")
    st.write(f"Name-map entries: {len(NAME_MAP)}")
    st.write(f"Distinct standardized codes: {len(STD_TO_ORIG)}")
    fams = sorted(set(STD_TO_GROUP.values()))
    st.write(f"Families detected: {len(fams)}")
    if fams:
        st.write(", ".join(fams[:30]) + (" ..." if len(fams) > 30 else ""))

# =============================  HELPERS =============================

def _depth_selector_ui(
    key_prefix: str,
    selected_wells: list,
    selected_logs: list,
    well_dict: dict,
    tops_df: pd.DataFrame
):
    """
    Unified depth selection: 'Slider' or 'Tops'.
    Returns: (depth_min, depth_max, label)
    """
    # Defaults from available data
    md_min_default, md_max_default = utils.get_global_md_range(
        selected_wells or list(well_dict.keys()),
        selected_logs or [],
        well_dict
    )

    if "depth_mode_" + key_prefix not in st.session_state:
        st.session_state["depth_mode_" + key_prefix] = "Slider"

    depth_selection_mode = st.radio(
        "Depth selection mode",
        ["Slider", "Tops"],
        horizontal=True,
        key="depth_mode_" + key_prefix
    )

    label = ""
    if depth_selection_mode == "Slider":
        # stick to local state keys to avoid clashing with other pages
        rng_key = f"{key_prefix}_depth_slider"

        # read a default without writing to session_state
        default_range = st.session_state.get(
            rng_key,
            (int(md_min_default), int(md_max_default))
        )

        depth_min, depth_max = st.slider(
            "Select depth range (MD)",
            min_value=int(md_min_default),
            max_value=int(md_max_default),
            value=default_range,
            step=1,
            key=rng_key
        )
        label = f"{depth_min}–{depth_max} m (MD)"


    else:
        # TOPS: average across selected wells
        filt = tops_df[tops_df["Well identifier (Well name)"].isin(selected_wells)]
        top_names = sorted(filt["Surface"].unique().tolist())
        if not top_names:
            st.warning("No tops found for the selected wells. Falling back to full depth range.")
            depth_min, depth_max = md_min_default, md_max_default
            label = f"{depth_min}–{depth_max} m (MD)"
        else:
            c1, c2 = st.columns(2)
            with c1:
                top_marker = st.selectbox("Top marker", top_names, key=f"{key_prefix}_top_sel")
            with c2:
                base_marker = st.selectbox(
                    "Base marker", top_names,
                    index=min(len(top_names)-1, 1),
                    key=f"{key_prefix}_base_sel"
                )
            dm, dx = utils.get_md_from_tops(tops_df, selected_wells, top_marker, base_marker)
            if (dm is None) or (dx is None):
                st.warning("Selected tops not found in this selection. Using full MD range.")
                depth_min, depth_max = md_min_default, md_max_default
                label = f"{depth_min}–{depth_max} m (MD)"
            else:
                depth_min, depth_max = dm, dx
                label = f"{top_marker} → {base_marker}  ({depth_min:.2f}–{depth_max:.2f} m MD)"

    return int(depth_min), int(depth_max), label


def _build_long_logs(wells_sel, logs_sel, wells_all, dmin, dmax) -> pd.DataFrame:
    """
    Creates the long-format logs DF: [Well, Log, MD, Value] within the given depth range.
    """
    return utils.get_logs_data_for_wells_logs(
        wells_list=wells_sel,
        logs_list=logs_sel,
        _wells=wells_all,
        depth_min=dmin,
        depth_max=dmax
    )


def _presence_map(df_long: pd.DataFrame, wells_all, wells_sel, depth_min, depth_max, log_for_map: str):
    """
    Plot a presence/absence map for a chosen log in the depth window.
    """
    # Coordinates
    geo_df = utils.get_well_min_lat_long(wells_all).copy()
    if geo_df.shape[1] >= 2:
        geo_df.columns = ["latitude", "longitude"][:geo_df.shape[1]]
    geo_df["Well Name"] = [w.petrel_name for w in wells_all]
    geo_df = geo_df[geo_df["Well Name"].isin(wells_sel)].dropna(subset=["latitude", "longitude"])
    if geo_df.empty:
        st.warning("No coordinates for selected wells.")
        return

    # Presence per well
    presence_status = []
    for _, row in geo_df.iterrows():
        wn = row["Well Name"]
        sub = df_long[(df_long["Well"] == wn) & (df_long["Log"] == log_for_map)]
        present = not sub.empty and sub["Value"].notna().any()
        presence_status.append("Present" if present else "Absent")
    geo_df["Presence"] = presence_status

    color_map = {"Present": "green", "Absent": "red"}
    fig = px.scatter_mapbox(
        geo_df,
        lat="latitude", lon="longitude",
        color="Presence",
        color_discrete_map=color_map,
        hover_name="Well Name",
        zoom=6, height=500,
        mapbox_style="open-street-map",
    )
    fig.update_traces(marker=dict(size=12))
    st.plotly_chart(fig, use_container_width=True)
    _capture_plotly_png(fig, f"Well Log Completeness — Map ({log_for_map})")


def _statistics_tables(selected_wells, selected_logs, well_dict, dmin, dmax, metrics: list):
    """
    Build & show one table per metric. Uses utils.get_log_statistics for computation.
    """
    metric_map = {
        "mean": np.nanmean,
        "median": np.nanmedian,
        "min": np.nanmin,
        "max": np.nanmax,
        "std": np.nanstd,
        "25%": (lambda arr: np.nanquantile(arr, 0.25)),
        "50%": (lambda arr: np.nanquantile(arr, 0.50)),
        "75%": (lambda arr: np.nanquantile(arr, 0.75)),
    }
    for m in metrics:
        func = metric_map.get(m)
        if func is None:
            st.info(f"Metric '{m}' not supported.")
            continue
        tbl = utils.get_log_statistics(selected_wells, selected_logs, well_dict, dmin, dmax, func)
        st.markdown(f"**Statistics — {m}**")
        if tbl is None or tbl.empty:
            st.info("No data for this selection.")
        else:
            st.dataframe(tbl.round(3).fillna("-"), use_container_width=True)


def _outlier_plot(df_long: pd.DataFrame, kind: str = "Box"):
    """
    Single figure showing distributions of selected logs grouped by Well.
    """
    req = {"Well", "Log", "Value"}
    if df_long.empty or not req.issubset(df_long.columns):
        st.info("No data available for outlier plot.")
        return
    dfp = df_long.copy()
    dfp = dfp[dfp["Well"] != "Unknown"]
    if dfp.empty:
        st.info("No valid samples to display.")
        return

    colors = px.colors.qualitative.Plotly
    logs = sorted(dfp["Log"].unique())

    fig = go.Figure()
    if kind == "Box":
        for i, log in enumerate(logs):
            sub = dfp[dfp["Log"] == log]
            if sub.empty:
                continue
            fig.add_trace(go.Box(
                x=sub["Well"],
                y=sub["Value"],
                name=log,
                marker_color=colors[i % len(colors)],
                boxpoints=False
            ))
        fig.update_layout(
            title="Box Plot of Logs Grouped by Well",
            xaxis_title="Well",
            yaxis_title="Value",
            boxmode="group",
            height=520,
            legend_title_text="Log"
        )
    else:
        for i, log in enumerate(logs):
            sub = dfp[dfp["Log"] == log]
            if sub.empty:
                continue
            fig.add_trace(go.Violin(
                x=sub["Well"],
                y=sub["Value"],
                name=log,
                line_color=colors[i % len(colors)],
                box_visible=True,
                meanline_visible=True,
                points=False
            ))
        fig.update_layout(
            title="Violin Plot of Logs Grouped by Well",
            xaxis_title="Well",
            yaxis_title="Value",
            violinmode="group",
            height=520,
            legend_title_text="Log"
        )

        st.plotly_chart(fig, use_container_width=True)
        _capture_plotly_png(fig, f"Outlier Visualization ({kind})")  # NEW

def _missing_plot(df_long: pd.DataFrame):
    """
    Missing-data matrix built on a wide pivot: rows=(Well, MD), columns=Log.
    """
    if df_long.empty:
        st.info("No data available for missing-value plot.")
        return
    wide = (
        df_long.pivot_table(index=["Well", "MD"], columns="Log", values="Value", aggfunc="mean")
        .reset_index(drop=False)
    )
    if wide.empty:
        st.info("No data after pivot.")
        return
    fig, ax = plt.subplots(figsize=(8, 3.5))
    try:
        msno.matrix(wide.drop(columns=["Well", "MD"]), ax=ax, sparkline=False)
    except Exception:
        # fallback if msno has trouble with dtypes
        msno.matrix(wide.select_dtypes(include=[float, int]), ax=ax, sparkline=False)
    ax.tick_params(axis='x', rotation=90, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    st.pyplot(fig)
    _capture_mpl_png(fig, "Missing-Value Plot") 


def _quality_checks(wells_sel, well_dict, dmin, dmax, logs_sel, qc_name, qc_params, show_table=True, show_map=True):
    """
    Run QC via utils.run_quality_checks and display as table and/or map.
    """
    # Compute QC
    qc_df = utils.run_quality_checks(
        wells=[well_dict[w] for w in wells_sel],
        depth_min=dmin, depth_max=dmax,
        selected_checks={qc_name: qc_params},
        selected_logs=logs_sel if logs_sel else None
    )

    if qc_df is None or qc_df.empty:
        st.info("No QC results for this selection.")
        return

    # --- Table
    if show_table:
        st.markdown("**Quality Check — Results Table**")
        try:
            st.dataframe(qc_df, use_container_width=True)
        except Exception:
            st.write(qc_df)

    # --- Map: PASS/FAIL by well (aggregate). Simple interpretation.
    if show_map:
        # Try to produce a single PASS/FAIL per well by OR-ing failures
        q = qc_df.copy()
        if "Well" not in q.columns:
            q = q.reset_index().rename(columns={"index": "Well"})
        val_cols = [c for c in q.columns if c != "Well"]
        # Assume cell truthy ~ PASS (your utils used "" as both pass/fail in earlier draft; here we coerce)
        def to_fail(series):
            s = series.astype(str).str.strip().str.lower()
            # treat 'fail' or 'false' or '0' as fail
            if (s == "fail").any() or (s == "false").any() or (s == "0").any():
                return True
            # numeric: 0 -> fail
            try:
                if pd.to_numeric(series, errors="coerce").fillna(1).eq(0).any():
                    return True
            except Exception:
                pass
            return False

        by_well = []
        for _, row in q.iterrows():
            wname = row["Well"]
            fail = to_fail(row[val_cols])
            by_well.append((wname, "FAIL" if fail else "PASS"))
        map_df = pd.DataFrame(by_well, columns=["Well", "Result"])

        # coordinates
        geo = utils.get_well_min_lat_long([well_dict[w] for w in wells_sel]).copy()
        if geo.shape[1] >= 2:
            geo.columns = ["latitude", "longitude"][:geo.shape[1]]
        geo["Well"] = wells_sel  # relies on ordering of wells_sel vs coords; safe in your utils

        plot_df = geo.merge(map_df, on="Well", how="left")
        if plot_df.empty or plot_df[["latitude", "longitude"]].dropna().empty:
            st.info("No coordinates for QC map.")
            return

        color_map = {"PASS": "green", "FAIL": "red"}
        fig = px.scatter_mapbox(
            plot_df,
            lat="latitude", lon="longitude",
            color="Result",
            color_discrete_map=color_map,
            hover_name="Well",
            zoom=6, height=480,
            mapbox_style="open-street-map",
        )
        fig.update_traces(marker=dict(size=12))
        st.plotly_chart(fig, use_container_width=True)
        _capture_plotly_png(fig, f"Quality Check — Map ({qc_name})")


def _pairplot(df_long: pd.DataFrame, logs_sel: list, hue_mode: str = "None", diag_kind="hist", offdiag_kind="scatter"):
    """
    Simple seaborn pairplot on a wide pivot (rows=(Well, MD), columns=logs).
    """
    if df_long.empty or not logs_sel:
        st.info("No data for pairplot.")
        return

    wide = (
        df_long.pivot_table(index=["Well", "MD"], columns="Log", values="Value", aggfunc="mean")
        .reset_index()
    )
    plot_cols = [c for c in logs_sel if c in wide.columns]
    if len(plot_cols) < 2:
        st.info("Select at least two logs with overlapping data.")
        return

    dfp = wide[["Well"] + plot_cols].dropna(how="all", subset=plot_cols).copy()

    hue = None
    if hue_mode == "Wells":
        hue = "Well"

    with sns.plotting_context("notebook", font_scale=0.9):
        g = sns.pairplot(
            dfp,
            vars=plot_cols,
            hue=hue,
            diag_kind=diag_kind,
            kind=offdiag_kind
        )
    st.pyplot(g.fig)
    _capture_mpl_png(g.fig, "Pairplot")


def _corr_heatmap(df_long: pd.DataFrame, logs_sel: list, method="pearson", cmap="coolwarm", annot=True):
    """
    Correlation heatmap across selected logs.
    """
    if df_long.empty or not logs_sel:
        st.info("No data for heatmap.")
        return

    wide = (
        df_long.pivot_table(index=["Well", "MD"], columns="Log", values="Value", aggfunc="mean")
        .reset_index()
    )
    plot_cols = [c for c in logs_sel if c in wide.columns]
    if len(plot_cols) < 2:
        st.info("Select at least two logs with overlapping data.")
        return

    corr_df = wide[plot_cols].corr(method=method)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(
        corr_df, ax=ax,
        cmap=cmap, vmin=-1, vmax=1,
        annot=annot, fmt=".2f", linewidths=0.5
    )
    ax.set_title(f"Correlation Heatmap ({method.title()})")
    plt.tight_layout()
    st.pyplot(fig)
    _capture_mpl_png(fig, f"Correlation Heatmap ({method.title()})")


def _histogram_view(df_long: pd.DataFrame, wells_sel: list, chosen_log: str, nbins=60, density=True, alpha=0.45, height=640):
    """
    Overlapped histograms per well for a chosen log.
    """
    if df_long.empty or not chosen_log:
        st.info("No data for histogram view.")
        return

    dfh = df_long[df_long["Log"] == chosen_log].copy()
    if dfh.empty:
        st.info("No samples for the chosen log.")
        return

    dfh = dfh[dfh["Well"].isin(wells_sel)]
    dfh["Value"] = pd.to_numeric(dfh["Value"], errors="coerce")
    dfh = dfh.dropna(subset=["Value"])
    wells_present = sorted(dfh["Well"].dropna().unique().tolist())
    if not wells_present:
        st.info("No valid samples to display.")
        return

    fig = go.Figure()
    palette = px.colors.qualitative.D3
    histnorm = "probability density" if density else ""
    for i, wname in enumerate(wells_present):
        sub = dfh.loc[dfh["Well"] == wname, "Value"]
        fig.add_histogram(
            x=sub,
            name=wname,
            nbinsx=nbins,
            histnorm=histnorm,
            marker_color=palette[i % len(palette)],
            opacity=alpha
        )
    fig.update_layout(
        barmode="overlay",
        height=height,
        margin=dict(l=40, r=20, t=40, b=40),
        legend_title_text="Well"
    )
    fig.update_yaxes(title_text=("Density" if density else "Count"))
    fig.update_xaxes(title_text=f"{chosen_log} (value)")
    st.plotly_chart(fig, use_container_width=True)
    _capture_plotly_png(fig, f"Histogram — {chosen_log}")

# Will collect (title, png_bytes) for export
EXPORT_PNGS = []

def _capture_plotly_png(fig, title: str, width=1000, height=600, scale=1):
    """
    Try to rasterize a Plotly figure to PNG and collect for export.
    Requires 'kaleido'. If not present, it silently skips.
    """
    try:
        png = fig.to_image(format="png", width=width, height=height, scale=scale)
        EXPORT_PNGS.append((title, png))
    except Exception:
        pass  # no kaleido or other issue — just skip

def _capture_mpl_png(fig, title: str, dpi=150):
    """Rasterize a Matplotlib figure to PNG and collect for export."""
    try:
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        EXPORT_PNGS.append((title, buf.getvalue()))
    except Exception:
        pass

# =============================  PAGE UI  =============================

st.title("Auto-EDA")

# ----------- Global filters -----------
wells = utils.get_all_wells_flat(petrel_project)
well_names = [w.petrel_name for w in wells]
well_dict = {w.petrel_name: w for w in wells}
tops_df = utils.load_tops_dataframe(petrel_project)

filters, preview = st.columns([1, 2.2])

with filters:
    st.subheader("Global Filters")

    wells = utils.get_all_wells_flat(petrel_project)
    well_names_all = [w.petrel_name for w in wells]
    well_dict = {w.petrel_name: w for w in wells}
    tops_df = utils.load_tops_dataframe(petrel_project)

    # Only keep wells that have at least one mapped log (Use in project = True)
    elig_well_names = eligible_well_names(wells)
    if not elig_well_names:
        # Likely cause: no stamped name_map yet, or all rows are excluded.
        st.warning(
            "No wells match the current name‑map (Use in project = True). "
            "Showing all wells as a fallback.\n\n"
            "Tip: In **Log Standardization**, click **Save ALL choices** or "
            "**Export simple name‑map JSON (stamped)** to create the mapping."
        )
        elig_well_names = [w.petrel_name for w in wells]


    sel_wells = st.multiselect("Select wells", options=elig_well_names, default=elig_well_names)

    # Build per-well -> original logs, but only mapped originals
    well_logs_map = {
        w.petrel_name: [
            getattr(lg, "petrel_name", "")
            for lg in getattr(w, "logs", [])
            if getattr(lg, "petrel_name", "") in NAME_MAP
        ]
        for w in wells
    }

    def _logs_for_wells(well_logs_map, wells_list):
        o = []
        for w in wells_list:
            o.extend(well_logs_map.get(w, []))
        return sorted(set(o))

    # --- FAMILIES-FIRST SELECTION (drop-in replacement) ----------------------

    # Standardized codes present in the selected wells (Use-in-project only)
    orig_pool = _logs_for_wells(well_logs_map, sel_wells) if sel_wells else []
    log_options_std = to_standardized_options(orig_pool)  # e.g., ["GR_STD", "RHO_STD", ...]

    # Build Group -> [standardized] mapping (restricted to what's present)
    from collections import defaultdict as _dd
    GROUP_TO_STD = _dd(list)
    for std in log_options_std:
        grp = STD_TO_GROUP.get(std, "(unlabeled)")
        GROUP_TO_STD[grp].append(std)

    # Families available (ignore empty labels)
    family_options = sorted([g for g in GROUP_TO_STD.keys() if g])

    # Default families: prefer GR / RHO / DT if present; else first three
    preferred_default_std = ["RHO_STD", "GR_STD", "DT_STD"]
    preferred_default_grps = [
        STD_TO_GROUP[s] for s in preferred_default_std
        if (s in STD_TO_GROUP) and (s in log_options_std)
    ]
    default_families = [g for g in family_options if g in preferred_default_grps] or family_options[:3]

    # UI: choose FAMILIES ONLY (groups)
    sel_groups = st.multiselect(
        "Select families (groups)",
        options=family_options,
        default=default_families,
        key="autoeda_groups"
    )

    # Expand chosen families -> standardized codes (internal use)
    sel_logs_std = sorted({std for g in sel_groups for std in GROUP_TO_STD.get(g, [])})

    # Backward-compat aliases so the rest of the file keeps working
    sel_logs = sel_logs_std          # many later calls use `sel_logs`
    log_options = log_options_std    # one UI block later refers to `log_options`

    # Expand standardized selections to the original names for data retrieval
    sel_logs_orig = []
    for std in sel_logs_std:
        sel_logs_orig.extend(STD_TO_ORIG.get(std, [std]))

    # Depth selection uses original names for safe min/max
    dmin, dmax, depth_label = _depth_selector_ui(
        key_prefix="autoeda",
        selected_wells=sel_wells or eligible_well_names(wells),
        selected_logs=sel_logs_orig or orig_pool,
        well_dict=well_dict,
        tops_df=tops_df
    )
    st.caption(f"**Applied depth window:** {depth_label}")
    # -------------------------------------------------------------------------

with preview:
    st.subheader("Well Locations (Preview)")
    geo_df = utils.get_well_min_lat_long(wells).copy()
    if geo_df.shape[1] >= 2:
        geo_df.columns = ["latitude", "longitude"][:geo_df.shape[1]]
    geo_df["Well Name"] = [w.petrel_name for w in wells]
    geo_df = geo_df[geo_df["Well Name"].isin(sel_wells)]
    if geo_df.empty or geo_df[["latitude", "longitude"]].dropna().empty:
        st.info("No coordinates available for the selected wells.")
    else:
        fig_map = px.scatter_mapbox(
            geo_df, lat="latitude", lon="longitude", hover_name="Well Name",
            zoom=6, height=500, mapbox_style="open-street-map"
        )
        fig_map.update_traces(marker=dict(size=12))
        st.plotly_chart(fig_map, use_container_width=True)

st.markdown("---")

# -------------------- Config Expanders --------------------

st.subheader("Report Contents")

# Group 1: Well Log Analysis
with st.expander("Well Log Analysis", expanded=True):
    c1, c2 = st.columns(2)

    with c1:
        inc_compl_map = st.checkbox("Include: Well Log Completeness — Map", value=False, key="inc_comp_map")
        compl_map_group = None
        if inc_compl_map:
            # choose a Family; we will map it to a representative standardized code on run
            compl_map_group = st.selectbox(
                "Family for completeness map",
                options=sel_groups if sel_groups else (family_options if 'family_options' in globals() else []),
                index=0 if (sel_groups or ('family_options' in globals() and family_options)) else 0,
                key="comp_map_group"
            )

        inc_stats = st.checkbox("Include: Statistics", value=True, key="inc_stats")
        stats_metrics = st.multiselect(
            "Statistics to compute",
            options=["mean", "median", "min", "max", "std", "25%", "50%", "75%"],
            default=["mean", "min", "max"],
            key="stats_metrics_sel"
        ) if inc_stats else []

        inc_outliers = st.checkbox("Include: Outlier Visualization", value=False, key="inc_outliers")
        outlier_kind = st.radio(
            "Outlier plot type", ["Box", "Violin"],
            key="outlier_kind", horizontal=True
        ) if inc_outliers else "Box"

    with c2:
        inc_missing = st.checkbox("Include: Missing-Value Plot", value=False, key="inc_missing")

        inc_qc = st.checkbox("Include: Quality Check", value=False, key="inc_qc")
        if inc_qc:
            qc_name = st.selectbox(
                "QC test",
                ["all_positive", "all_above", "mean_below", "no_nans", "range", "no_flat", "no_monotonic"],
                index=0, key="qc_name_sel"
            )
            qc_params = {}
            if qc_name == "all_above":
                qc_params["threshold"] = st.number_input("Threshold (all_above)", value=50.0, step=1.0, key="qc_thr_above")
            elif qc_name == "mean_below":
                qc_params["threshold"] = st.number_input("Threshold (mean_below)", value=100.0, step=1.0, key="qc_thr_mean")
            elif qc_name == "range":
                rmin, rmax = st.slider("Range (min, max)", -1000, 1000, (0, 200), key="qc_range")
                qc_params["min"], qc_params["max"] = rmin, rmax
            qc_show = st.multiselect("Show QC output", ["Table", "Map"], default=["Table"], key="qc_show_sel")

# Group 2: Pairplot
with st.expander("Pairplot", expanded=False):
    inc_pair = st.checkbox("Include: Pairplot", value=False, key="inc_pair")
    if inc_pair:
        hue_mode = st.radio("Color by", ["None", "Wells"], index=0, key="pp_hue_mode", horizontal=True)
        diag_kind = st.selectbox("Diagonal", ["hist", "kde"], index=0, key="pp_diag_kind")
        offdiag_kind = st.selectbox("Off-diagonal", ["scatter", "reg", "kde", "hist"], index=0, key="pp_offdiag_kind")

# Group 3: Heatmap
with st.expander("Heatmap", expanded=False):
    inc_heat = st.checkbox("Include: Correlation Heatmap", value=False, key="inc_heat")
    if inc_heat:
        heat_method = st.selectbox("Correlation method", ["pearson", "spearman", "kendall"], index=0, key="hm_method")
        heat_cmap = st.selectbox(
            "Colormap",
            ["coolwarm", "viridis", "plasma", "inferno", "magma", "cividis", "Spectral", "YlGnBu", "RdBu", "seismic", "bwr"],
            index=0, key="hm_cmap"
        )
        heat_annot = st.checkbox("Show annotations", value=True, key="hm_annot")

# Group 4: Histogram View
with st.expander("Histogram View", expanded=False):
    inc_hist = st.checkbox("Include: Histogram View", value=False, key="inc_hist")
    if inc_hist:
        chosen_group_for_hist = st.selectbox(
            "Choose a family",
            options=(sel_groups if sel_groups else (family_options if 'family_options' in globals() else [])),
            index=0 if (sel_groups or ('family_options' in globals() and family_options)) else 0,
            key="hv_group"
        )
        # pick the first standardized code under that family for the plot function
        hv_std_for_hist = GROUP_TO_STD.get(chosen_group_for_hist, sel_logs_std)[0] if 'GROUP_TO_STD' in globals() else None
        st.session_state["hv_log"] = hv_std_for_hist  # keep downstream call unchanged

        nbins = st.slider("Bins", 10, 200, 60, 5, key="hv_bins")
        density = st.checkbox("Normalize (density)", value=True, key="hv_density")
        alpha = st.slider("Histogram opacity", 0.1, 1.0, 0.45, 0.05, key="hv_alpha")

# ---- Run button (requested) ----
run_clicked = st.button("Run Automatic EDA", type="primary")

st.markdown("---")

# =============================  EXECUTION  =============================

if run_clicked:
    if not sel_wells:
        st.warning("Select at least one well.")
        st.stop()
    if not sel_groups:
        st.warning("Select at least one family.")
        st.stop()

    # Build the long table once (reused by most sections)
    df_long = _build_long_logs(sel_wells, sel_logs_orig, wells, dmin, dmax)
    # Standardize labels in the long table so downstream visuals use codes
    if df_long is not None and not df_long.empty and "Log" in df_long.columns:
        df_long["Log"] = df_long["Log"].astype(str).map(lambda x: NAME_MAP.get(x, x))

    # Collect fresh images for this run
    EXPORT_PNGS.clear()


    st.success("Automatic EDA executed with current selections.")
    st.markdown("## Results")

    # --- Completeness table ---
    if st.session_state.get("inc_comp_tbl", False):
        st.markdown("### Well Log Completeness — Table")
        presence_df = utils.get_log_presence_matrix(sel_wells, sel_logs, _well_dict=well_dict)
        if presence_df is None or presence_df.empty:
            st.info("No data to show.")
        else:
            # style can be added later; simple table for this test
            st.dataframe(presence_df, use_container_width=True, hide_index=True)

    # --- Completeness map ---
    if st.session_state.get("inc_comp_map", False) and st.session_state.get("comp_map_group"):
        grp = st.session_state["comp_map_group"]
        std_for_map = GROUP_TO_STD.get(grp, [None])[0] if 'GROUP_TO_STD' in globals() else None
        if std_for_map:
            st.markdown("### Well Log Completeness — Map")
            _presence_map(df_long, wells, sel_wells, dmin, dmax, std_for_map)
        else:
            st.info("No standardized log available for the selected family.")

    # --- Statistics ---
    if st.session_state.get("inc_stats", False) and st.session_state.get("stats_metrics_sel"):
        st.markdown("### Statistics")
        _statistics_tables(sel_wells, sel_logs, well_dict, dmin, dmax, st.session_state["stats_metrics_sel"])

    # --- Outliers ---
    if st.session_state.get("inc_outliers", False):
        st.markdown("### Outlier Visualization")
        _outlier_plot(df_long, kind=st.session_state.get("outlier_kind", "Box"))

    # --- Missing values ---
    if st.session_state.get("inc_missing", False):
        st.markdown("### Missing-Value Plot")
        _missing_plot(df_long)

    # --- Quality checks ---
    if st.session_state.get("inc_qc", False):
        st.markdown("### Quality Check")
        qc_name = st.session_state.get("qc_name_sel")
        qc_params = {}
        if qc_name == "all_above":
            qc_params["threshold"] = st.session_state.get("qc_thr_above", 50.0)
        elif qc_name == "mean_below":
            qc_params["threshold"] = st.session_state.get("qc_thr_mean", 100.0)
        elif qc_name == "range":
            rmin, rmax = st.session_state.get("qc_range", (0, 200))
            qc_params["min"], qc_params["max"] = rmin, rmax

        show_tbl = "Table" in st.session_state.get("qc_show_sel", ["Table"])
        show_map = "Map" in st.session_state.get("qc_show_sel", [])
        _quality_checks(sel_wells, well_dict, dmin, dmax, sel_logs, qc_name, qc_params, show_tbl, show_map)

    # --- Pairplot ---
    if st.session_state.get("inc_pair", False):
        st.markdown("### Pairplot")
        _pairplot(
            df_long,
            logs_sel=sel_logs,
            hue_mode=st.session_state.get("pp_hue_mode", "None"),
            diag_kind=st.session_state.get("pp_diag_kind", "hist"),
            offdiag_kind=st.session_state.get("pp_offdiag_kind", "scatter"),
        )

    # --- Heatmap ---
    if st.session_state.get("inc_heat", False):
        st.markdown("### Correlation Heatmap")
        _corr_heatmap(
            df_long,
            logs_sel=sel_logs,
            method=st.session_state.get("hm_method", "pearson"),
            cmap=st.session_state.get("hm_cmap", "coolwarm"),
            annot=st.session_state.get("hm_annot", True)
        )

    # --- Histogram view ---
    if st.session_state.get("inc_hist", False) and st.session_state.get("hv_log"):
        st.markdown("### Histogram View")
        _histogram_view(
            df_long,
            wells_sel=sel_wells,
            chosen_log=st.session_state["hv_log"],
            nbins=st.session_state.get("hv_bins", 60),
            density=st.session_state.get("hv_density", True),
            alpha=st.session_state.get("hv_alpha", 0.45),
            height=640
        )

else:
    st.info("Configure the sections above, then click **Run Automatic EDA**.")


# =============================  EXPORT (HTML / PDF)  =============================
st.markdown("---")
st.subheader("Export Report")

if not _HAS_KALEIDO:
    st.info("Plotly figures will appear in the HTML, but to embed them as images in PDF/HTML, "
            "install kaleido: `pip install -U kaleido`")

# Recompute the key tables (so export works even if user unchecked some visuals)
# 1) Completeness table
export_presence_df = None
if st.session_state.get("inc_comp_tbl", False):
    try:
        export_presence_df = utils.get_log_presence_matrix(sel_wells, sel_logs, _well_dict=well_dict)
    except Exception:
        export_presence_df = None

# 2) Statistics (one table per selected metric)
export_stats = []
if st.session_state.get("inc_stats", False):
    metric_map = {
        "mean": np.nanmean,
        "median": np.nanmedian,
        "min": np.nanmin,
        "max": np.nanmax,
        "std": np.nanstd,
        "25%": (lambda arr: np.nanquantile(arr, 0.25)),
        "50%": (lambda arr: np.nanquantile(arr, 0.50)),
        "75%": (lambda arr: np.nanquantile(arr, 0.75)),
    }
    for m in st.session_state.get("stats_metrics_sel", []):
        func = metric_map.get(m)
        if func is None:
            continue
        try:
            tbl = utils.get_log_statistics(sel_wells, sel_logs, well_dict, dmin, dmax, func)
            if tbl is not None and not tbl.empty:
                export_stats.append((m, tbl.round(3)))
        except Exception:
            pass

# 3) QC table (aggregate) – only if QC was requested
export_qc_df = None
if st.session_state.get("inc_qc", False):
    try:
        qc_name = st.session_state.get("qc_name_sel")
        qc_params = {}
        if qc_name == "all_above":
            qc_params["threshold"] = st.session_state.get("qc_thr_above", 50.0)
        elif qc_name == "mean_below":
            qc_params["threshold"] = st.session_state.get("qc_thr_mean", 100.0)
        elif qc_name == "range":
            rmin, rmax = st.session_state.get("qc_range", (0, 200))
            qc_params["min"], qc_params["max"] = rmin, rmax

        export_qc_df = utils.run_quality_checks(
            wells=[well_dict[w] for w in sel_wells],
            depth_min=dmin, depth_max=dmax,
            selected_checks={qc_name: qc_params},
            selected_logs=sel_logs if sel_logs else None
        )
    except Exception:
        export_qc_df = None

# -------- Build a simple self-contained HTML report --------
def _df_to_html(df: pd.DataFrame) -> str:
    try:
        return df.to_html(index=True, border=0, classes="table", justify="center")
    except Exception:
        return "<p><i>Table could not be rendered.</i></p>"

now = datetime.now().strftime("%Y-%m-%d %H:%M")
wells_txt = ", ".join(sel_wells) if sel_wells else "—"
logs_txt = ", ".join(sel_groups) if sel_groups else "—"

html_parts = [
    "<!DOCTYPE html><html><head><meta charset='utf-8'>",
    "<style>",
    "body{font-family:Arial,Helvetica,sans-serif; margin:24px;}",
    "h1{margin-bottom:4px;} .muted{color:#555;}",
    "h2{margin-top:28px; border-bottom:1px solid #ddd; padding-bottom:4px;}",
    "table{border-collapse:collapse; width:100%; margin:10px 0;}",
    "th,td{border:1px solid #ddd; padding:6px; font-size:12px;}",
    "th{background:#f5f5f5;}",
    "</style></head><body>",
    f"<h1>Automatic EDA Report</h1>",
    f"<div class='muted'>Generated: {now}</div>",
    "<h2>Selections</h2>",
    f"<p><b>Wells:</b> {wells_txt}</p>",
    f"<p><b>Logs:</b> {logs_txt}</p>",
    f"<p><b>Depth window:</b> {depth_label}</p>",
]

# Completeness
if export_presence_df is not None and not export_presence_df.empty:
    html_parts += ["<h2>Well Log Completeness — Table</h2>", _df_to_html(export_presence_df)]

# Statistics
if export_stats:
    html_parts.append("<h2>Statistics</h2>")
    for metric_name, df_metric in export_stats:
        html_parts.append(f"<h3>{metric_name}</h3>")
        html_parts.append(_df_to_html(df_metric))

# QC
if isinstance(export_qc_df, pd.DataFrame) and not export_qc_df.empty:
    html_parts += ["<h2>Quality Check</h2>", _df_to_html(export_qc_df)]

# Figures (as base64 PNG)
if EXPORT_PNGS:
    html_parts.append("<h2>Figures</h2>")
    for title, png in EXPORT_PNGS:
        b64 = base64.b64encode(png).decode("utf-8")
        html_parts.append(f"<h3>{title}</h3>")
        html_parts.append(f"<img src='data:image/png;base64,{b64}' style='max-width:100%;'/>")

if not _HAS_KALEIDO:
    html_parts.append("<p style='color:#a00'><i>Note: To include Plotly figures as images, install <b>kaleido</b>.</i></p>")

html_parts.append("</body></html>")
export_html = "".join(html_parts).encode("utf-8")

# HTML download
st.download_button(
    "Download HTML report",
    data=export_html,
    file_name="automatic_eda_report.html",
    mime="text/html",
    type="primary"
)

    # PDF download (ReportLab; avoids native deps)
try:
    def _df_to_table(df: pd.DataFrame):
        df_reset = df.reset_index(drop=False)
        data = [list(df_reset.columns)] + df_reset.values.tolist()
        tbl = Table(data, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
            ("TOPPADDING", (0, 0), (-1, 0), 6),
        ]))
        return tbl

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
    )
    styles = getSampleStyleSheet()
    story = []

    # Header
    story.append(Paragraph("Automatic EDA Report", styles["Title"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Generated: {now}", styles["Normal"]))
    story.append(Paragraph(f"Wells: {wells_txt}", styles["Normal"]))
    story.append(Paragraph(f"Logs: {logs_txt}", styles["Normal"]))
    story.append(Paragraph(f"Depth window: {depth_label}", styles["Normal"]))
    story.append(Spacer(1, 12))

    # Completeness table
    if export_presence_df is not None and not export_presence_df.empty:
        story.append(Paragraph("Well Log Completeness — Table", styles["Heading2"]))
        story.append(_df_to_table(export_presence_df))
        story.append(Spacer(1, 12))

    # Statistics tables
    if export_stats:
        story.append(Paragraph("Statistics", styles["Heading2"]))
        for metric_name, df_metric in export_stats:
            story.append(Paragraph(metric_name, styles["Heading3"]))
            story.append(_df_to_table(df_metric))
            story.append(Spacer(1, 8))

    # QC table
    if isinstance(export_qc_df, pd.DataFrame) and not export_qc_df.empty:
        story.append(Paragraph("Quality Check", styles["Heading2"]))
        story.append(_df_to_table(export_qc_df))
        story.append(Spacer(1, 12))

        # Figures (rasterized)
    if EXPORT_PNGS:
        story.append(Paragraph("Figures", styles["Heading2"]))
        for title, png in EXPORT_PNGS:
            story.append(Paragraph(title, styles["Heading3"]))
            img_buf = BytesIO(png)
            try:
                pil = PILImage.open(img_buf)
                w, h = pil.size
                max_w = 6.5 * inch  # page width minus margins
                scale = min(1.0, max_w / float(w)) if w else 1.0
                story.append(RLImage(BytesIO(png), width=w*scale, height=h*scale))
            except Exception:
                story.append(Paragraph("(Could not render image)", styles["Italic"]))
            story.append(Spacer(1, 8))


    doc.build(story)
    pdf_bytes = buf.getvalue()

    st.download_button(
        "Download PDF report",
        data=pdf_bytes,
        file_name="automatic_eda_report.pdf",
        mime="application/pdf"
    )
except Exception as e:
    st.info(f"PDF export not available: {e}")
# N:\_USER_GLOBAL\PETREL\Prizm\wf\1_Wells_EDA\pages\1_Automatic Seismic Inversion EDA.py

import streamlit as st
import numpy as np
import pandas as pd
import utils
from utils import try_get_project_name
import re

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno

from io import BytesIO
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.platypus import Image as RLImage
from reportlab.lib.units import inch
from PIL import Image as PILImage
import base64

# Kaleido (optional) to rasterize Plotly in exports
try:
    import kaleido  # noqa: F401
    _HAS_KALEIDO = True
except Exception:
    _HAS_KALEIDO = False

# ---------------------- Streamlit + Nav ----------------------
st.set_page_config(
    page_title="GeoPython | Automatic Seismic Inversion EDA",
    layout="wide",
    initial_sidebar_state="expanded",
)
utils.render_grouped_sidebar_nav()

# Connect & footer (keeps logo + "Connected to project..." at sidebar bottom)
petrel_project = utils.get_petrel_connection()
utils.sidebar_footer(petrel_project, utils.LOGO_PATH, utils.APP_VERSION)

# ---------------------- Standardization ----------------------
import json, re
from pathlib import Path
from collections import defaultdict

def _slugify(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\(.*?\)", " ", s)
    s = s.replace("\\", " ").replace("/", " ").replace("_", " ").replace("-", " ")
    s = re.sub(r"[^A-Za-z0-9\s\.]+", " ", s)   # keep dots for *.pet projects
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "project"

proj_raw  = try_get_project_name(petrel_project) or petrel_project.get_current_project_name()
proj_slug = _slugify(proj_raw)
STD_DIR   = Path.home() / "Cegal" / "Prizm" / "geoPython" / proj_slug / "logStandarization"

def _load_std_artifacts():
    """
    Returns:
      NAME_MAP:     {original_name -> standardized_code}
      STD_TO_GROUP: {standardized_code -> family}
      STD_TO_ORIG:  {standardized_code -> [original_names]}
      src_summary:  short text of loaded files
    """
    name_map = {}
    std_to_group = {}
    std_to_orig = defaultdict(list)
    nm_files  = sorted(STD_DIR.glob(f"log_name_map__{proj_slug}__*__*.json"))
    rec_files = sorted(STD_DIR.glob(f"log_mapping_records__{proj_slug}__*__*.json"))
    srcs = []

    if nm_files:
        try:
            name_map = json.loads(nm_files[-1].read_text(encoding="utf-8"))
            srcs.append(f"name_map: {nm_files[-1].name}")
        except Exception:
            srcs.append(f"name_map: {nm_files[-1].name} (read error)")

    if rec_files:
        try:
            recs = json.loads(rec_files[-1].read_text(encoding="utf-8"))
            srcs.append(f"records: {rec_files[-1].name}")
            for r in recs:
                std = str(r.get("Standardized", "")).strip()
                grp = str(r.get("Group", "")).strip()
                if std and grp and grp != "(unmatched)" and std not in std_to_group:
                    std_to_group[std] = grp
            if not name_map:
                for r in recs:
                    if bool(r.get("Use in project", True)):
                        orig = str(r.get("Original log name", "")).strip()
                        std  = str(r.get("Standardized", "")).strip()
                        if orig and std:
                            name_map[orig] = std
                srcs.append("derived name_map from records")
        except Exception:
            srcs.append(f"records: {rec_files[-1].name} (read error)")

    for orig, std in name_map.items():
        std_to_orig[std].append(orig)

    return name_map, std_to_group, std_to_orig, " | ".join(srcs) if srcs else "no stamped files found"

NAME_MAP, STD_TO_GROUP, STD_TO_ORIG, _STD_SRC = _load_std_artifacts()
NAME_MAP_EMPTY = (len(NAME_MAP) == 0)

# --- Robust standardization helpers (works with or without a name‑map) ---
def _build_orig_to_std(name_map: dict, std_to_orig: dict) -> dict:
    d = dict(name_map) if name_map else {}
    for std, origs in (std_to_orig or {}).items():
        for o in origs:
            d.setdefault(str(o), str(std))
    return d

ORIG_TO_STD = _build_orig_to_std(NAME_MAP, STD_TO_ORIG)

# Regex aliases as a last resort if no stamped files exist
_STD_SYNONYMS = {
    "GR_STD":  [r"\bgr\b", r"gamma\s*ray", r"\bgrc?\b"],
    "RHO_STD": [r"\brhob\b", r"\bdens(ity)?\b", r"bulk\s*den"],
    "DT_STD":  [r"\bdt(co)?\b", r"\bsonic\b", r"compressional", r"p[-\s]*wave"],
}

def _std_alias_from_text(s: str) -> str | None:
    t = str(s or "").strip().lower()
    for std, pats in _STD_SYNONYMS.items():
        for pat in pats:
            if re.search(pat, t):
                return std
    return None

def _to_std(label: str) -> str:
    lab = str(label)
    return ORIG_TO_STD.get(lab) or _std_alias_from_text(lab) or lab

def _ensure_logs_in_df_long(df_long: pd.DataFrame,
                            logs_std: list[str],
                            wells_sel: list[str],
                            wells_all,
                            dmin: int, dmax: int) -> pd.DataFrame:
    """Guarantee df_long contains the requested standardized logs; fetch missing ones and standardize."""
    if df_long is None or df_long.empty:
        present = set()
    else:
        present = set(map(str, df_long.get("Log", pd.Series(dtype=str)).unique()))

    need_std = [s for s in logs_std if s not in present]
    if not need_std:
        return df_long

    need_orig = []
    for s in need_std:
        need_orig.extend(STD_TO_ORIG.get(s, [s]))

    add = utils.get_logs_data_for_wells_logs(
        wells_list=wells_sel,
        logs_list=need_orig,
        _wells=wells_all,
        depth_min=dmin,
        depth_max=dmax
    )
    if add is not None and not add.empty and "Log" in add.columns:
        add = add.copy()
        add["Log"] = add["Log"].map(_to_std).astype(str)
        df_long = pd.concat([df_long, add], ignore_index=True) if df_long is not None else add
    return df_long


def _fmt_std(std: str) -> str:
    grp = STD_TO_GROUP.get(std, "")
    return f"{std} — {grp}" if grp else std

# Fallbacks when there is no name-map
def to_standardized_options(original_names: list[str]) -> list[str]:
    """If NAME_MAP present: return mapped standardized codes used in project; else pass-through unique originals."""
    if NAME_MAP_EMPTY:
        return sorted({str(n) for n in original_names})
    std_set = set()
    for n in original_names:
        if n in NAME_MAP:
            std_set.add(NAME_MAP[n])
    return sorted(std_set)

def eligible_well_names(wells) -> list[str]:
    if NAME_MAP_EMPTY:
        return [w.petrel_name for w in wells]
    allowed = set(NAME_MAP.keys())
    return [
        w.petrel_name for w in wells
        if any(getattr(lg, "petrel_name", "") in allowed for lg in getattr(w, "logs", []))
    ]

# ---------------------- Helpers ----------------------
def _depth_selector_ui(
    key_prefix: str,
    selected_wells: list,
    selected_logs_orig: list,
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
        selected_logs_orig or [],
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
        rng_key = f"{key_prefix}_depth_slider"
        default_range = st.session_state.get(
            rng_key, (int(md_min_default), int(md_max_default))
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

def _build_long_logs(wells_sel, logs_sel_orig, wells_all, dmin, dmax) -> pd.DataFrame:
    """Long-format logs DF: [Well, Log, MD, Value] in depth window."""
    return utils.get_logs_data_for_wells_logs(
        wells_list=wells_sel,
        logs_list=logs_sel_orig,
        _wells=wells_all,
        depth_min=dmin,
        depth_max=dmax
    )

def _presence_map(df_long: pd.DataFrame, wells_all, wells_sel, log_for_map: str, height=500):
    """Presence/absence map for a chosen standardized log within df_long’s window."""
    geo_df = utils.get_well_min_lat_long(wells_all).copy()
    if geo_df.shape[1] >= 2:
        geo_df.columns = ["latitude", "longitude"][:geo_df.shape[1]]
    geo_df["Well Name"] = [w.petrel_name for w in wells_all]
    geo_df = geo_df[geo_df["Well Name"].isin(wells_sel)].dropna(subset=["latitude", "longitude"])
    if geo_df.empty:
        st.warning("No coordinates for selected wells.")
        return

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
        zoom=6, height=height,
        mapbox_style="open-street-map",
    )
    fig.update_traces(marker=dict(size=12))
    st.plotly_chart(fig, use_container_width=True)
    _capture_plotly_png(fig, f"Completeness — Map ({log_for_map})")

def _required_logs_summary(df_long: pd.DataFrame, wells_sel: list, req_logs_std: list, dmin: int, dmax: int):
    """
    Inversion readiness per well for required logs (std or originals):
    For each required standardized log, compute Present (✓/✗), MD start, MD end, Coverage %.
    """
    # Accept both standardized codes and their original aliases
    cand = {std: {std} | set(STD_TO_ORIG.get(std, [])) for std in req_logs_std}
    rows = []
    for w in wells_sel:
        row = {"Well": w}
        for std in req_logs_std:
            subset = df_long[(df_long["Well"] == w) & (df_long["Log"].isin(cand[std]))].copy()
            present = (not subset.empty) and subset["Value"].notna().any()
            if present:
                subset = subset.dropna(subset=["MD", "Value"])
                if not subset.empty:
                    md_start = int(np.nanmin(subset["MD"]))
                    md_end   = int(np.nanmax(subset["MD"]))
                    cov_pct  = max(0.0, min(100.0, 100.0 * (md_end - md_start) / max(1, (dmax - dmin))))
                else:
                    md_start, md_end, cov_pct = None, None, 0.0
            else:
                md_start, md_end, cov_pct = None, None, 0.0

            row[f"{std} — Present"]    = "✓" if present else "✗"
            row[f"{std} — MD start"]   = md_start if md_start is not None else "-"
            row[f"{std} — MD end"]     = md_end   if md_end   is not None else "-"
            row[f"{std} — Coverage %"] = round(cov_pct, 1)
        rows.append(row)
    return pd.DataFrame(rows)

def _statistics_tables(selected_wells, selected_logs_std, logs_sel_orig, well_dict, dmin, dmax, metrics: list):
    """One table per metric (mean/median/etc.) using utils.get_log_statistics."""
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
            continue
        tbl = utils.get_log_statistics(selected_wells, logs_sel_orig, well_dict, dmin, dmax, func)
        st.markdown(f"**Statistics — {m}**")
        if tbl is None or tbl.empty:
            st.info("No data for this selection.")
        else:
            tbl = tbl.rename(columns=lambda c: NAME_MAP.get(str(c), str(c)))
            st.dataframe(tbl.round(3).fillna("-"), use_container_width=True)

def _outlier_plot(df_long: pd.DataFrame, selected_wells: list, selected_logs_std: list, kind: str = "Box"):
    """Box/violin plots per standardized log, grouped by Well."""
    req = {"Well", "Log", "Value"}
    if df_long.empty or not req.issubset(df_long.columns):
        st.info("No data available for outlier plot.")
        return
    dfp = df_long[(df_long["Well"].isin(selected_wells)) & (df_long["Log"].isin(selected_logs_std))].copy()
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
            height=560,
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
            height=560,
            legend_title_text="Log"
        )

    fig.update_xaxes(categoryorder="array", categoryarray=selected_wells)
    st.plotly_chart(fig, use_container_width=True)
    _capture_plotly_png(fig, f"Outlier Visualization ({kind})")

def _missing_plot(df_long: pd.DataFrame):
    """Missing-data matrix on wide pivot."""
    if df_long.empty:
        st.info("No data for missing-value plot.")
        return
    wide = (
        df_long.pivot_table(index=["Well", "MD"], columns="Log", values="Value", aggfunc="mean")
        .reset_index(drop=False)
    )
    if wide.empty:
        st.info("No data after pivot.")
        return
    fig, ax = plt.subplots(figsize=(8, 3.4))
    try:
        msno.matrix(wide.drop(columns=["Well", "MD"]), ax=ax, sparkline=False)
    except Exception:
        msno.matrix(wide.select_dtypes(include=[float, int]), ax=ax, sparkline=False)
    ax.tick_params(axis='x', rotation=90, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    st.pyplot(fig)
    _capture_mpl_png(fig, "Missing-Value Plot")

def _pairplot_grid_plotly(df_long: pd.DataFrame, selected_wells: list, selected_logs_std: list):
    """Lower triangle scatter + diagonal histograms (per-well overlay when wells ≤ 10)."""
    if df_long.empty or not selected_logs_std:
        st.info("No data for pairplot.")
        return

    wide = (
        df_long[df_long["Well"].isin(selected_wells) & df_long["Log"].isin(selected_logs_std)]
        .pivot_table(index=["Well", "MD"], columns="Log", values="Value", aggfunc="mean")
        .reset_index()
    )
    plot_cols = [c for c in selected_logs_std if c in wide.columns]
    if len(plot_cols) < 2:
        st.info("Select at least two logs with overlapping data.")
        return

    # Subsample if huge
    if len(wide) > 20000:
        wide = wide.sample(20000, random_state=42)

    d = len(plot_cols)
    specs = [[({"type": "histogram"} if r == c else ({"type": "xy"} if r > c else None))
             for c in range(d)] for r in range(d)]

    auto_height = int(min(900, max(600, 130 * d)))
    fig = make_subplots(
        rows=d, cols=d,
        specs=specs,
        shared_xaxes=False, shared_yaxes=False,
        horizontal_spacing=0.02, vertical_spacing=0.02
    )

    use_color_by_well = wide["Well"].nunique() <= 10
    wells_present = sorted(wide["Well"].dropna().unique().tolist())
    palette = px.colors.qualitative.Plotly + px.colors.qualitative.D3
    color_map = {w: palette[i % len(palette)] for i, w in enumerate(wells_present)}
    legend_shown = set()

    # Diagonals
    for i, col in enumerate(plot_cols):
        all_vals = pd.to_numeric(wide[col], errors="coerce").dropna()
        if all_vals.empty:
            continue
        xmin = float(all_vals.min()); xmax = float(all_vals.max())
        if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
            xmax = xmin + 1e-9
        bin_size = (xmax - xmin) / 30.0
        if use_color_by_well:
            for wname in wells_present:
                sub = pd.to_numeric(wide.loc[wide["Well"] == wname, col], errors="coerce").dropna()
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
                    hovertemplate=(f"Well: {wname}<br>{col}: %{{x:.3g}}<br>"
                                   "Density: %{y:.3g}<extra></extra>"),
                    row=i+1, col=i+1
                )
                if show_leg:
                    legend_shown.add(wname)
        else:
            fig.add_histogram(
                x=all_vals,
                xbins=dict(start=xmin, end=xmax, size=bin_size),
                histnorm="probability density",
                marker_color="rgba(120,120,120,0.85)",
                showlegend=False,
                hovertemplate=f"{col}: %{{x:.3g}}<br>Density: %{{y:.3g}}<extra></extra>",
                row=i+1, col=i+1
            )

    # Lower triangle
    for r in range(1, d):
        for c in range(0, r):
            xname, yname = plot_cols[c], plot_cols[r]
            if use_color_by_well:
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
                            hovertemplate=(f"Well: {wname}<br>{xname}: %{{x:.3g}}"
                                           f"<br>{yname}: %{{y:.3g}}<extra></extra>"),
                            row=r+1, col=c+1
                        )
                        if show_leg:
                            legend_shown.add(wname)
            else:
                x = pd.to_numeric(wide[xname], errors="coerce")
                y = pd.to_numeric(wide[yname], errors="coerce")
                txt = wide["Well"].astype(str)
                fig.add_scattergl(
                    x=x, y=y, mode="markers",
                    marker=dict(size=3, opacity=0.35),
                    name="points",
                    showlegend=False,
                    text=txt,
                    hovertemplate=("Well: %{text}<br>"
                                   f"{xname}: %{{x:.3g}}<br>{yname}: %{{y:.3g}}<extra></extra>"),
                    row=r+1, col=c+1
                )

    # Labels
    for i, col in enumerate(plot_cols):
        fig.update_xaxes(title_text=col, row=d, col=i+1)
        fig.update_yaxes(title_text=col, row=i+1, col=1)

    fig.update_layout(
        height=auto_height,
        margin=dict(l=20, r=20, t=40, b=20),
        title="Pairplot (hist on diagonal)",
        legend_title_text="Well" if use_color_by_well else None,
        hovermode="closest",
        barmode="overlay"
    )
    if not use_color_by_well:
        fig.update_layout(showlegend=False)

    st.plotly_chart(fig, use_container_width=True)
    try:
        _capture_plotly_png(fig, "Pairplot (histograms on diagonal)")
    except Exception:
        pass

def _corr_heatmap(df_long: pd.DataFrame, selected_wells: list, selected_logs_std: list,
                  method="pearson", cmap="coolwarm", annot=True, fig_w=6.0, fig_h=4.6):
    """Correlation heatmap across selected logs."""
    if df_long.empty or not selected_logs_std:
        st.info("No data for heatmap.")
        return
    wide = (
        df_long[df_long["Well"].isin(selected_wells) & df_long["Log"].isin(selected_logs_std)]
        .pivot_table(index=["Well", "MD"], columns="Log", values="Value", aggfunc="mean")
        .reset_index()
    )
    plot_cols = [c for c in selected_logs_std if c in wide.columns]
    if len(plot_cols) < 2:
        st.info("Select at least two logs with overlapping data.")
        return

    data = wide[plot_cols].copy()
    if data.dropna(how="all").empty:
        st.info("No numeric samples to compute correlation.")
        return

    corr_df = data.corr(method=method)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        corr_df, ax=ax,
        cmap=cmap, vmin=-1, vmax=1,
        annot=annot, fmt=".2f", linewidths=0.5
    )
    ax.set_title(f"Log Correlation Heatmap ({method.title()})")
    plt.tight_layout()
    st.pyplot(fig)
    _capture_mpl_png(fig, f"Correlation Heatmap ({method.title()})")

def _histogram_view(df_long: pd.DataFrame, wells_sel: list, chosen_log: str, nbins=60, density=True, alpha=0.45, height=640):
    """Overlapped histograms per well for a chosen standardized log."""
    if df_long.empty or not chosen_log:
        st.info("No data for histogram view.")
        return
    dfh = df_long[(df_long["Log"] == chosen_log) & (df_long["Well"].isin(wells_sel))].copy()
    if dfh.empty:
        st.info("No samples for the chosen log.")
        return

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

# ---------- Export collectors ----------
EXPORT_PNGS = []
def _capture_plotly_png(fig, title: str, width=1000, height=600, scale=1):
    try:
        png = fig.to_image(format="png", width=width, height=height, scale=scale)
        EXPORT_PNGS.append((title, png))
    except Exception:
        pass

def _capture_mpl_png(fig, title: str, dpi=150):
    try:
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        EXPORT_PNGS.append((title, buf.getvalue()))
    except Exception:
        pass

# =============================  PAGE  =============================
st.title("Automatic Seismic Inversion EDA")

# ----------- Global filters -----------
wells = utils.get_all_wells_flat(petrel_project)
well_dict = {w.petrel_name: w for w in wells}
tops_df = utils.load_tops_dataframe(petrel_project, utils._sel_cache_key())

filters, preview = st.columns([1, 2.2])

with filters:
    st.subheader("Global Filters")

    # Eligible wells (mapped logs if name-map present; all wells otherwise)
    elig_well_names = eligible_well_names(wells)
    if not elig_well_names:
        st.warning(
            "No wells match the current name‑map (Use in project = True). "
            "Showing all wells as a fallback."
        )
        elig_well_names = [w.petrel_name for w in wells]

    sel_wells = st.multiselect("Select wells", options=elig_well_names, default=elig_well_names)

    # Logs present in selected wells (original names, limited to NAME_MAP if present)
    well_logs_map = {
        w.petrel_name: [
            getattr(lg, "petrel_name", "")
            for lg in getattr(w, "logs", [])
            if (NAME_MAP_EMPTY or getattr(lg, "petrel_name", "") in NAME_MAP)
        ]
        for w in wells
    }
    def _logs_for_wells(well_logs_map, wells_list):
        o = []
        for wn in wells_list:
            o.extend(well_logs_map.get(wn, []))
        return sorted(set(o))
    orig_pool = _logs_for_wells(well_logs_map, sel_wells) if sel_wells else []

    # Standardized options available
    log_options_std = to_standardized_options(orig_pool)

    # Groups (families) available within current log_options_std
    from collections import defaultdict as _dd
    GROUP_TO_STD = _dd(list)
    for std in log_options_std:
        grp = STD_TO_GROUP.get(std, "(unlabeled)")
        GROUP_TO_STD[grp].append(std)

    family_options = sorted([g for g in GROUP_TO_STD.keys() if g])
    # prefer GR, RHO, DT defaults if possible
    # preferred = ["GR_STD", "RHO_STD", "DT_STD"]
    # preferred_grps = [STD_TO_GROUP[s] for s in preferred if (s in STD_TO_GROUP and s in log_options_std)]
    # default_families = [g for g in family_options if g in preferred_grps] or family_options[:3]

    # sel_groups = st.multiselect(
    #     "Select families (groups)",
    #     options=family_options,
    #     default=default_families
    # )

    from collections import defaultdict as _dd

    # Groups (families) available within current standardized options
    GROUP_TO_STD = _dd(list)
    for std in log_options_std:
        grp = STD_TO_GROUP.get(std, "(unlabeled)")
        GROUP_TO_STD[grp].append(std)

    family_options = sorted([g for g in GROUP_TO_STD.keys() if g])

    def _pick_families(families: list[str]) -> list[str]:
        wants = ["Gamma Ray", "Density", "Sonic", "DT"]
        chosen = []
        for want in wants:
            for g in families:
                if want.lower() in g.lower():
                    chosen.append(g); break
        return list(dict.fromkeys(chosen))  # unique, keep order

    preferred_families = _pick_families(family_options)
    if preferred_families:
        default_families = preferred_families
    else:
        # fallback: try GR/RHO/DT standardized groups if known
        preferred_std = ["GR_STD", "RHO_STD", "DT_STD"]
        preferred_grps = [STD_TO_GROUP.get(s) for s in preferred_std if STD_TO_GROUP.get(s)]
        default_families = [g for g in family_options if g in preferred_grps] or family_options[:3]

    sel_groups = st.multiselect(
        "Select families (groups)",
        options=family_options,
        default=default_families
    )

    sel_logs_std = sorted({std for g in sel_groups for std in GROUP_TO_STD.get(g, [])})

    # Expand standardized -> originals for data retrieval
    sel_logs_orig = []
    for std in sel_logs_std:
        sel_logs_orig.extend(STD_TO_ORIG.get(std, [std]))

    sel_logs_std = sorted({std for g in sel_groups for std in GROUP_TO_STD.get(g, [])})
    # Expand standardized -> originals for data retrieval
    sel_logs_orig = []
    for std in sel_logs_std:
        sel_logs_orig.extend(STD_TO_ORIG.get(std, [std]))

    # Depth selection (Slider or Tops)
    dmin, dmax, depth_label = _depth_selector_ui(
        key_prefix="inv",
        selected_wells=sel_wells or eligible_well_names(wells),
        selected_logs_orig=sel_logs_orig or orig_pool,
        well_dict=well_dict,
        tops_df=tops_df
    )
    st.caption(f"**Applied depth window:** {depth_label}")

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
            zoom=6, height=480, mapbox_style="open-street-map"
        )
        fig_map.update_traces(marker=dict(size=12))
        st.plotly_chart(fig_map, use_container_width=True)

st.markdown("---")

# -------------------- Report Contents (config) --------------------
st.subheader("Report Contents")
with st.expander("Configure sections (Seismic Inversion QC)", expanded=False):
    # Required logs for inversion (default on)
    st.markdown("**Required logs for inversion QC (checked by default)**")
    colA, colB, colC = st.columns(3)
    with colA:
        use_GR = st.checkbox("Gamma Ray (GR_STD)", value=True, key="req_gr")
    with colB:
        use_RHO = st.checkbox("Density (RHO_STD)", value=True, key="req_rho")
    with colC:
        use_DT = st.checkbox("Sonic (DT_STD)", value=True, key="req_dt")
    req_logs_std = [std for std, on in [("GR_STD", use_GR), ("RHO_STD", use_RHO), ("DT_STD", use_DT)] if on]

    st.markdown("---")
    st.markdown("**Sections to include**")

    # 1) Data Table
    inc_table = st.checkbox("Include: Well‑Log Table (MD vs Value per Well/Log)", value=True, key="inc_tbl")

    # 2) Completeness (matrix + map)
    col1, col2 = st.columns(2)
    with col1:
        inc_compl_tbl = st.checkbox("Include: Well‑Log Completeness — Table", value=True, key="inc_comp_tbl")
    with col2:
        inc_compl_map = st.checkbox("Include: Well‑Log Completeness — Map", value=True, key="inc_comp_map")

    # 3) Statistics
    inc_stats = st.checkbox("Include: Statistics Matrix", value=True, key="inc_stats")
    stats_metrics = st.multiselect(
        "Statistics to compute",
        options=["mean", "median", "min", "max", "std", "25%", "50%", "75%"],
        default=["mean", "min", "max"],
        key="stats_metrics_sel"
    ) if inc_stats else []

    # 4) Outliers
    inc_outliers = st.checkbox("Include: Outlier Visualization", value=False, key="inc_outliers")
    outlier_kind = st.radio("Outlier plot type", ["Box", "Violin"], horizontal=True, key="outlier_kind") if inc_outliers else "Box"

    # 5) Missing values
    inc_missing = st.checkbox("Include: Missing‑Value Plot", value=False, key="inc_missing")

    # 6) QC
    inc_qc = st.checkbox("Include: Quality Checks", value=False, key="inc_qc")
    if inc_qc:
        available_tests = {
            "all_positive": {},
            "all_above": {"threshold": 50},
            "mean_below": {"threshold": 100},
            "no_nans": {},
            "range": {"min": 0, "max": 200},
            "no_flat": {},
            "no_monotonic": {},
        }
        qc_name = st.selectbox("Quality check", options=list(available_tests.keys()), index=0, key="qc_test_sel")
        # parameter controls (unique keys)
        if qc_name == "all_above":
            st.slider("Threshold (all_above)", -1000, 1000, available_tests["all_above"]["threshold"], key="qc_thr_above")
        elif qc_name == "mean_below":
            st.slider("Threshold (mean_below)", -1000, 1000, available_tests["mean_below"]["threshold"], key="qc_thr_mean")
        elif qc_name == "range":
            st.slider("Range (min, max)", -1000, 1000, (available_tests["range"]["min"], available_tests["range"]["max"]), key="qc_range")
        st.multiselect("Show QC output", ["Table", "Map"], default=["Table"], key="qc_show_sel")

    # 7) Pairplot
    inc_pair = st.checkbox("Include: Pairplot (interactive grid)", value=False, key="inc_pair")

    # 8) Heatmap
    inc_heat = st.checkbox("Include: Correlation Heatmap", value=False, key="inc_heat")
    if inc_heat:
        st.selectbox("Correlation method", ["pearson", "spearman", "kendall"], index=0, key="hm_method")
        st.selectbox(
            "Colormap",
            ["coolwarm", "viridis", "plasma", "inferno", "magma", "cividis", "Spectral", "YlGnBu", "RdBu", "seismic", "bwr"],
            index=0, key="hm_cmap"
        )
        st.checkbox("Show annotations", value=True, key="hm_annot")

    # 9) Histogram View
    inc_hist = st.checkbox("Include: Histogram View (per log)", value=False, key="inc_hist")
    hv_std_for_hist = None
    if inc_hist:
        fam_for_hist = st.selectbox(
            "Family for histogram view",
            options=(sel_groups if sel_groups else list(GROUP_TO_STD.keys()) or ["(none)"]),
            index=0,
            key="hv_group"
        )
        cand = GROUP_TO_STD.get(fam_for_hist, [])
        hv_std_for_hist = cand[0] if cand else (sel_logs_std[0] if sel_logs_std else None)
        st.session_state["hv_log_std"] = hv_std_for_hist
        st.slider("Bins", 10, 200, 60, 5, key="hv_bins")
        st.checkbox("Normalize (density)", value=True, key="hv_density")
        st.slider("Histogram opacity", 0.1, 1.0, 0.45, 0.05, key="hv_alpha")

    # 10) Global multi‑well, multi‑log curves
    inc_global_plot = st.checkbox("Include: Global Well Log Visualization", value=False, key="inc_global_plot")
    logs_for_global_plot = []
    if inc_global_plot:
        st.caption("Select which standardized logs to plot (one column per log).")
        defaults = sel_logs_std[:3] if sel_logs_std else []
        logs_for_global_plot = st.multiselect(
            "Logs to plot (columns)",
            options=(sel_logs_std or log_options_std),
            default=defaults,
            key="global_plot_logs"
        )

    # 11) Histogram per log (ALL selected logs)
    inc_hist_multi = st.checkbox("Include: Histogram View (per log, ALL selected)", value=False, key="inc_hist_multi")


# ---- Run button ----
run_clicked = st.button("Run Automatic Seismic Inversion EDA", type="primary")
st.markdown("---")

# =============================  EXECUTION  =============================
if run_clicked:
    # Guards
    if not sel_wells:
        st.warning("Select at least one well.")
        st.stop()
    if not sel_groups:
        st.warning("Select at least one family.")
        st.stop()

    # Build long DF once for the selected depth window
    # df_long = _build_long_logs(sel_wells, sel_logs_orig, wells, dmin, dmax)
    # # ---- FIX: never use "or pd.DataFrame()" with a DataFrame (ambiguous truth value)
    # if df_long is None or not isinstance(df_long, pd.DataFrame) or df_long.empty:
    #     df_long = pd.DataFrame(columns=["Well", "Log", "MD", "Value"])
    # else:
    #     # Standardize labels so downstream visuals use codes (if a map exists)
    #     if "Log" in df_long.columns:
    #         df_long["Log"] = df_long["Log"].astype(str).map(lambda x: NAME_MAP.get(x, x))

    df_long = _build_long_logs(sel_wells, sel_logs_orig, wells, dmin, dmax)
    if df_long is None or not isinstance(df_long, pd.DataFrame) or df_long.empty:
        df_long = pd.DataFrame(columns=["Well", "Log", "MD", "Value"])
    else:
        # Robust: standardize with map + aliases so GR/RHO/DT become GR_STD/RHO_STD/DT_STD
        if "Log" in df_long.columns:
            df_long = df_long.copy()
            df_long["Log"] = df_long["Log"].map(_to_std).astype(str)

    # Ensure the "required logs" (GR/RHO/DT as selected above) are present even if families excluded them
    if req_logs_std:
        df_long = _ensure_logs_in_df_long(df_long, req_logs_std, sel_wells, wells, dmin, dmax)

    # (Optional) quick sanity snapshot while debugging
    # st.caption("Debug — first standardized labels: " + ", ".join(sorted(set(map(str, df_long['Log'].unique())))[:12]))


    EXPORT_PNGS.clear()
    st.success("Executed with current selections.")
    st.markdown("## Results")

    # ---------- 0) Inversion readiness (required GR/RHO/DT) ----------
    # if req_logs_std:
        # with st.expander("Inversion Readiness (Required Logs: GR / RHO / DT)", expanded=True):
        #     inv_tbl = _required_logs_summary(df_long, sel_wells, req_logs_std, dmin, dmax)
        #     if inv_tbl.empty:
        #         st.info("No data found for the required logs in the current selection.")
        #     else:
        #         def _style_presence(v):
        #             return "color:#0a0; font-weight:600;" if v == "✓" else ("color:#a00; font-weight:600;" if v == "✗" else "")
        #         def _style_coverage(v):
        #             try:
        #                 v = float(v)
        #                 return "color:#a60" if v < 60 else ""
        #             except Exception:
        #                 return ""
        #         # Build a Styler with per-column rules
        #         sty = inv_tbl.style
        #         for std in req_logs_std:
        #             present_col = f"{std} — Present"
        #             cov_col = f"{std} — Coverage %"
        #             if present_col in inv_tbl.columns:
        #                 sty = sty.applymap(_style_presence, subset=[present_col])
        #             if cov_col in inv_tbl.columns:
        #                 sty = sty.applymap(_style_coverage, subset=[cov_col])
        #         st.dataframe(sty, use_container_width=True)

        #         # Missing summary
        #         miss = []
        #         for w in sel_wells:
        #             for std in req_logs_std:
        #                 col = f"{std} — Present"
        #                 if col in inv_tbl.columns:
        #                     val = inv_tbl.loc[inv_tbl["Well"] == w, col]
        #                     if not val.empty and val.iloc[0] == "✗":
        #                         miss.append((w, std))
        #         if miss:
        #             st.info("**Missing required logs**: " + ", ".join([f"{w}:{std}" for w,std in miss]))

    # ---------- 1) Well‑Log Table ----------
    if st.session_state.get("inc_tbl", True):
        with st.expander("Well‑Log Table", expanded=False):
            if df_long.empty:
                st.info("No data to filter yet.")
            else:
                well_col, log_col, md_col = "Well", "Log", "MD"
                val_col = "Value" if "Value" in df_long.columns else None
                missing_cols = [c for c in [well_col, log_col, md_col, val_col] if c not in df_long.columns]
                if missing_cols:
                    st.error(f"Missing expected columns in data: {missing_cols}")
                else:
                    well_opts = [w for w in sel_wells if w in df_long[well_col].unique()]
                    log_opts  = [l for l in sel_logs_std if l in df_long[log_col].unique()]
                    if not well_opts or not log_opts:
                        st.warning("No matching wells/logs in the dataset.")
                    else:
                        fcol1, fcol2 = st.columns(2)
                        with fcol1:
                            well_pick = st.selectbox("Well", options=well_opts, index=0)
                        with fcol2:
                            log_pick = st.selectbox("Log", options=log_opts, index=0)
                        mask = (df_long[well_col] == well_pick) & (df_long[log_col] == log_pick)
                        table_df = df_long.loc[mask, [md_col, val_col]].reset_index(drop=True)
                        st.dataframe(table_df, hide_index=True, use_container_width=True)

    # ---------- 2) Completeness ----------
    if st.session_state.get("inc_comp_tbl", True) or st.session_state.get("inc_comp_map", True):
        with st.expander("Well‑Log Completeness", expanded=False):
            left, right = st.columns([1,2], gap="large")

            with left:
                if st.session_state.get("inc_comp_tbl", True):
                    st.markdown("#### Presence Matrix")
                    presence_df = utils.get_log_presence_matrix(sel_wells, sel_logs_orig, _well_dict=well_dict)
                    presence_df = presence_df.rename(columns=lambda c: NAME_MAP.get(str(c), str(c)))
                    if presence_df is None or presence_df.empty:
                        st.info("No presence data to show.")
                    else:
                        colorize = st.checkbox("Colorize matrix (✓=green, ✗=red)", value=True, key="comp_tbl_colorize")
                        if colorize:
                            try:
                                st.dataframe(presence_df.style.applymap(utils.highlight_presence),
                                             hide_index=True, use_container_width=True)
                            except Exception:
                                st.dataframe(presence_df, hide_index=True, use_container_width=True)
                        else:
                            st.dataframe(presence_df, hide_index=True, use_container_width=True)

            with right:
                if st.session_state.get("inc_comp_map", True):
                    st.markdown("#### Map")
                    if not sel_logs_std:
                        st.info("Select at least one log to display on the map.")
                    else:
                        # default to a required log if present in df; else first selected
                        candidates = [s for s in ["GR_STD", "RHO_STD", "DT_STD"] if s in df_long["Log"].unique()]
                        default_log_for_map = (candidates[0] if candidates else sel_logs_std[0])
                        selected_log_for_map = st.selectbox(
                            "Log to display (presence)",
                            options=sel_logs_std,
                            index=sel_logs_std.index(default_log_for_map) if default_log_for_map in sel_logs_std else 0
                        )
                        _presence_map(df_long, wells, sel_wells, selected_log_for_map)

    # ----------  X) Global Well Log Visualization ----------
    if st.session_state.get("inc_global_plot", False):
        with st.expander("Global Well Log Visualization", expanded=False):
            logs_to_plot = st.session_state.get("global_plot_logs", sel_logs_std)
            if not logs_to_plot:
                st.info("No logs selected to plot.")
            else:
                # Guarantee data exist for all requested logs
                df_long_plot = _ensure_logs_in_df_long(df_long, logs_to_plot, sel_wells, wells, dmin, dmax)

                # Controls
                colA, colB, colC, colD = st.columns([1, 1, 1, 1])
                with colA:
                    gv_lw = st.slider("Line width", 0.5, 4.0, 1.2, 0.1, key="gv_lw")
                with colB:
                    gv_height = st.slider("Figure height (px)", 400, 1200, 620, 20, key="gv_h")
                with colC:
                    gv_show_legend = st.checkbox("Show legend", value=True, key="gv_legend")
                with colD:
                    gv_show_window_lines = st.checkbox("Show depth-window lines", value=True, key="gv_win_lines")

                # Build figure
                from plotly.subplots import make_subplots
                n = len(logs_to_plot)
                fig = make_subplots(
                    rows=1, cols=n,
                    shared_yaxes=True,
                    horizontal_spacing=0.03,
                    subplot_titles=[str(s) for s in logs_to_plot]
                )

                wells_present = sorted([w for w in sel_wells if w in set(df_long_plot["Well"].unique())])
                palette = px.colors.qualitative.Plotly + px.colors.qualitative.D3 + px.colors.qualitative.T10
                color_map = {w: palette[i % len(palette)] for i, w in enumerate(wells_present)}

                for j, std_name in enumerate(logs_to_plot, start=1):
                    sub = df_long_plot[df_long_plot["Log"] == std_name]
                    if sub.empty:
                        fig.add_annotation(text="No data", row=1, col=j, showarrow=False, y=(dmin+dmax)/2)
                        fig.update_xaxes(title_text=str(std_name), row=1, col=j)
                        continue

                    # Per well curves
                    for w in wells_present:
                        sw = sub[sub["Well"] == w]
                        if sw.empty:
                            continue
                        x = pd.to_numeric(sw["Value"], errors="coerce")
                        y = pd.to_numeric(sw["MD"], errors="coerce")
                        m = x.notna() & y.notna()
                        if not m.any():
                            continue
                        fig.add_trace(
                            go.Scattergl(
                                x=x[m], y=y[m], mode="lines",
                                name=w, legendgroup=w, showlegend=(j == 1 and gv_show_legend),
                                line=dict(width=gv_lw, color=color_map[w]),
                                hovertemplate=f"Well: {w}<br>MD: %{{y}}<br>{std_name}: %{{x}}<extra></extra>"
                            ),
                            row=1, col=j
                        )

                    # Optional: show depth-window lines
                    if gv_show_window_lines:
                        # Span across current x-range
                        xv = pd.to_numeric(sub["Value"], errors="coerce")
                        x0 = float(np.nanmin(xv)) if np.isfinite(pd.to_numeric([np.nanmin(xv)], errors="coerce")).all() else None
                        x1 = float(np.nanmax(xv)) if np.isfinite(pd.to_numeric([np.nanmax(xv)], errors="coerce")).all() else None
                        if x0 is not None and x1 is not None and x1 != x0:
                            for yline, dash in [(dmin, "dot"), (dmax, "dot")]:
                                fig.add_shape(
                                    type="line", xref=f"x{j}", yref=f"y{j}",
                                    x0=x0, x1=x1, y0=yline, y1=yline,
                                    line=dict(color="rgba(150,150,150,0.7)", width=1, dash=dash)
                                )

                    fig.update_xaxes(title_text=str(std_name), row=1, col=j)

                fig.update_yaxes(autorange="reversed", title_text="Measured Depth (MD)", row=1, col=1)
                fig.update_layout(height=gv_height, margin=dict(l=40, r=20, t=50, b=40))
                st.plotly_chart(fig, use_container_width=True)
                _capture_plotly_png(fig, f"Global Well Log Visualization ({', '.join(map(str, logs_to_plot))})")

    
    # ---------- 3) Statistics ----------
    if st.session_state.get("inc_stats", False) and st.session_state.get("stats_metrics_sel"):
        with st.expander("Well‑Log Statistics Matrix", expanded=False):
            _statistics_tables(sel_wells, sel_logs_std, sel_logs_orig, well_dict, dmin, dmax, st.session_state["stats_metrics_sel"])

    # ---------- 4) Outliers ----------
    if st.session_state.get("inc_outliers", False):
        with st.expander("Outlier Visualization", expanded=False):
            _outlier_plot(df_long, sel_wells, sel_logs_std, kind=st.session_state.get("outlier_kind", "Box"))

    # ---------- 5) Missing values ----------
    if st.session_state.get("inc_missing", False):
        with st.expander("Missing‑Value Plot", expanded=False):
            _missing_plot(df_long)

    # ---------- 6) Quality checks ----------
    if st.session_state.get("inc_qc", False):
        with st.expander("Quality Checks", expanded=False):
            qc_name = st.session_state.get("qc_test_sel")
            qc_params = {}
            if qc_name == "all_above":
                qc_params["threshold"] = st.session_state.get("qc_thr_above", 50)
            elif qc_name == "mean_below":
                qc_params["threshold"] = st.session_state.get("qc_thr_mean", 100)
            elif qc_name == "range":
                rmin, rmax = st.session_state.get("qc_range", (0, 200))
                qc_params["min"], qc_params["max"] = rmin, rmax

            qc_df = utils.run_quality_checks(
                wells=[well_dict[w] for w in sel_wells],
                depth_min=dmin, depth_max=dmax,
                selected_checks={qc_name: qc_params},
                selected_logs=sel_logs_orig if sel_logs_orig else None
            )

            show_tbl = "Table" in st.session_state.get("qc_show_sel", ["Table"])
            show_map = "Map" in st.session_state.get("qc_show_sel", [])

            if qc_df is None or qc_df.empty:
                st.info("No QC results.")
            else:
                if "Log" in qc_df.columns:
                    qc_df["Log"] = qc_df["Log"].astype(str).map(lambda x: NAME_MAP.get(x, x))
                if show_tbl:
                    st.markdown("**QC Results — Table**")
                    try:
                        st.dataframe(qc_df.style.applymap(utils.highlight_pass_fail), use_container_width=True)
                    except Exception:
                        st.dataframe(qc_df, use_container_width=True)

                if show_map:
                    # compress to PASS/FAIL per well
                    q = qc_df.copy()
                    if "Well" not in q.columns:
                        q = q.reset_index().rename(columns={"index": "Well"})
                    val_cols = [c for c in q.columns if c != "Well"]
                    def to_fail(series):
                        s = series.astype(str).str.strip().str.lower()
                        if (s == "fail").any() or (s == "false").any() or (s == "0").any():
                            return True
                        try:
                            if pd.to_numeric(series, errors="coerce").fillna(1).eq(0).any():
                                return True
                        except Exception:
                            pass
                        return False
                    by_well = [(row["Well"], "FAIL" if to_fail(row[val_cols]) else "PASS") for _, row in q.iterrows()]
                    map_df = pd.DataFrame(by_well, columns=["Well", "Result"])
                    geo = utils.get_well_min_lat_long([well_dict[w] for w in sel_wells]).copy()
                    if geo.shape[1] >= 2:
                        geo.columns = ["latitude", "longitude"][:geo.shape[1]]
                    geo["Well"] = sel_wells
                    plot_df = geo.merge(map_df, on="Well", how="left")
                    color_map = {"PASS": "green", "FAIL": "red"}
                    fig = px.scatter_mapbox(
                        plot_df,
                        lat="latitude", lon="longitude",
                        color="Result",
                        color_discrete_map=color_map,
                        hover_name="Well",
                        zoom=6, height=500,
                        mapbox_style="open-street-map",
                    )
                    fig.update_traces(marker=dict(size=12))
                    st.plotly_chart(fig, use_container_width=True)
                    _capture_plotly_png(fig, f"Quality Check — Map ({qc_name})")

    # ---------- 7) Pairplot ----------
    if st.session_state.get("inc_pair", False):
        with st.expander("Pairplot", expanded=False):
            _pairplot_grid_plotly(df_long, sel_wells, sel_logs_std)

    # ---------- 8) Heatmap ----------
    if st.session_state.get("inc_heat", False):
        with st.expander("Correlation Heatmap", expanded=False):
            _corr_heatmap(
                df_long,
                selected_wells=sel_wells,
                selected_logs_std=sel_logs_std,
                method=st.session_state.get("hm_method", "pearson"),
                cmap=st.session_state.get("hm_cmap", "coolwarm"),
                annot=st.session_state.get("hm_annot", True),
                fig_w=6.0, fig_h=4.6
            )

    # ---------- 9) Histogram ----------
    if st.session_state.get("inc_hist", False) and st.session_state.get("hv_log_std"):
        with st.expander("Histogram View", expanded=False):
            _histogram_view(
                df_long,
                wells_sel=sel_wells,
                chosen_log=st.session_state["hv_log_std"],
                nbins=st.session_state.get("hv_bins", 60),
                density=st.session_state.get("hv_density", True),
                alpha=st.session_state.get("hv_alpha", 0.45),
                height=640
            )

    # ----------  Y) Histogram View (per log, ALL selected logs) ----------
    if st.session_state.get("inc_hist_multi", False):
        with st.expander("Histogram View (per log, ALL selected logs)", expanded=False):
            # Controls
            left, right = st.columns([1, 3])
            with left:
                hm_logs = st.multiselect(
                    "Logs to include",
                    options=(sel_logs_std or log_options_std),
                    default=(sel_logs_std if sel_logs_std else []),
                    key="hv_all_logs"
                )
                nbins   = st.slider("Bins", 10, 200, 60, 5, key="hv_bins_all")
                density = st.checkbox("Normalize (density)", value=True, key="hv_density_all")
                show_outliers = st.checkbox("Show outliers on box", value=False, key="hv_outliers_all")
                alpha   = st.slider("Histogram opacity", 0.1, 1.0, 0.45, 0.05, key="hv_alpha_all")
                plot_h  = st.slider("Plot height (px)", 400, 1200, 720, 20, key="hv_height_all")

            with right:
                if not hm_logs:
                    st.info("Pick at least one log.")
                else:
                    # Make sure data are present
                    df_long_plot = _ensure_logs_in_df_long(df_long, hm_logs, sel_wells, wells, dmin, dmax)
                    wells_present = sorted(df_long_plot["Well"].dropna().unique().tolist())
                    palette = px.colors.qualitative.D3 + px.colors.qualitative.Plotly
                    color_map = {w: palette[i % len(palette)] for i, w in enumerate(wells_present)}

                    for chosen_log in hm_logs:
                        # Prepare data for this log
                        dfh = df_long_plot[
                            (df_long_plot["Log"] == chosen_log) & (df_long_plot["Well"].isin(sel_wells))
                        ][["Well", "Value"]].copy()

                        if dfh.empty:
                            st.info(f"No data for {chosen_log} in the selected depth window.")
                            continue

                        dfh["Value"] = pd.to_numeric(dfh["Value"], errors="coerce")
                        dfh = dfh.dropna(subset=["Value"])
                        dfh = dfh[dfh["Well"] != "Unknown"]
                        wells_here = sorted(dfh["Well"].dropna().unique().tolist())
                        if not wells_here:
                            st.info(f"No valid samples for {chosen_log}.")
                            continue

                        from plotly.subplots import make_subplots
                        fig = make_subplots(
                            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                            row_heights=[0.28, 0.72],
                            subplot_titles=(f"Box per well — {chosen_log}", f"Histogram — {chosen_log}")
                        )

                        # Row 1: horizontal boxes (one per well)
                        for w in wells_here:
                            sub = dfh.loc[dfh["Well"] == w, "Value"]
                            if sub.empty:
                                continue
                            fig.add_box(
                                x=sub, y=[w] * len(sub),
                                name=w, marker_color=color_map[w],
                                boxpoints="outliers" if show_outliers else False,
                                orientation="h", showlegend=False,
                                row=1, col=1
                            )

                        # Row 2: overlapped histograms
                        histnorm = "probability density" if density else ""
                        for w in wells_here:
                            sub = dfh.loc[dfh["Well"] == w, "Value"]
                            if sub.empty:
                                continue
                            fig.add_histogram(
                                x=sub, name=w, nbinsx=nbins, histnorm=histnorm,
                                marker_color=color_map[w], opacity=alpha,
                                hovertemplate=f"Well: {w}<br>{chosen_log}: %{{x}}<br>"
                                            + ("Density" if density else "Count")
                                            + ": %{y}<extra></extra>",
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
                        _capture_plotly_png(fig, f"Histogram — {chosen_log}")


else:
    st.info("Configure the sections above, then click **Run Automatic Seismic Inversion EDA**.")

# =============================  EXPORT (HTML / PDF)  =============================
st.markdown("---")
st.subheader("Export Report")

if not _HAS_KALEIDO:
    st.info("Plotly figures will appear in the HTML; to embed them as images, install **kaleido** (`pip install -U kaleido`).")

# Try to reconstruct a few tables for export (lightweight, works even after toggles)
export_presence_df = None
if st.session_state.get("inc_comp_tbl", False) and 'sel_wells' in locals() and 'sel_logs_orig' in locals():
    try:
        export_presence_df = utils.get_log_presence_matrix(sel_wells, sel_logs_orig, _well_dict=well_dict)
        export_presence_df = export_presence_df.rename(columns=lambda c: NAME_MAP.get(str(c), str(c)))
    except Exception:
        export_presence_df = None

# Statistics tables
export_stats = []
if st.session_state.get("inc_stats", False) and st.session_state.get("stats_metrics_sel") and 'sel_wells' in locals():
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
            tbl = utils.get_log_statistics(sel_wells, sel_logs_orig, well_dict, dmin, dmax, func)
            if tbl is not None and not tbl.empty:
                export_stats.append((m, tbl.rename(columns=lambda c: NAME_MAP.get(str(c), str(c))).round(3)))
        except Exception:
            pass

# # Inversion readiness table (if we still have a df_long in scope)
# try:
#     _df_long_for_export = locals().get("df_long", pd.DataFrame())
#     req_for_export = []
#     if st.session_state.get("req_gr", False):  req_for_export.append("GR_STD")
#     if st.session_state.get("req_rho", False): req_for_export.append("RHO_STD")
#     if st.session_state.get("req_dt", False):  req_for_export.append("DT_STD")
#     export_inv_tbl = _required_logs_summary(_df_long_for_export, sel_wells, req_for_export, dmin, dmax)
# except Exception:
#     export_inv_tbl = None

def _df_to_html(df: pd.DataFrame) -> str:
    try:
        return df.to_html(index=True, border=0, classes="table", justify="center")
    except Exception:
        return "<p><i>Table could not be rendered.</i></p>"

now = datetime.now().strftime("%Y-%m-%d %H:%M")
wells_txt = ", ".join(sel_wells) if 'sel_wells' in locals() and sel_wells else "—"
logs_txt = ", ".join(sel_groups) if 'sel_groups' in locals() and sel_groups else "—"
depth_label = locals().get("depth_label", "—")

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
    "<h1>Automatic Seismic Inversion EDA</h1>",
    f"<div class='muted'>Generated: {now}</div>",
    "<h2>Selections</h2>",
    f"<p><b>Wells:</b> {wells_txt}</p>",
    f"<p><b>Families:</b> {logs_txt}</p>",
    f"<p><b>Depth window:</b> {depth_label}</p>",
]

# if export_inv_tbl is not None and not export_inv_tbl.empty:
#     html_parts += ["<h2>Inversion Readiness (Required Logs)</h2>", _df_to_html(export_inv_tbl)]

if export_presence_df is not None and not export_presence_df.empty:
    html_parts += ["<h2>Well Log Completeness — Table</h2>", _df_to_html(export_presence_df)]

if export_stats:
    html_parts.append("<h2>Statistics</h2>")
    for metric_name, df_metric in export_stats:
        html_parts.append(f"<h3>{metric_name}</h3>")
        html_parts.append(_df_to_html(df_metric))

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

st.download_button(
    "Download HTML report",
    data=export_html,
    file_name="automatic_seismic_inversion_eda_report.html",
    mime="text/html",
    type="primary"
)

# PDF
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

    story.append(Paragraph("Automatic Seismic Inversion EDA", styles["Title"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Generated: {now}", styles["Normal"]))
    story.append(Paragraph(f"Wells: {wells_txt}", styles["Normal"]))
    story.append(Paragraph(f"Families: {logs_txt}", styles["Normal"]))
    story.append(Paragraph(f"Depth window: {depth_label}", styles["Normal"]))
    story.append(Spacer(1, 12))

    # if export_inv_tbl is not None and not export_inv_tbl.empty:
    #     story.append(Paragraph("Inversion Readiness (Required Logs)", styles["Heading2"]))
    #     story.append(_df_to_table(export_inv_tbl))
    #     story.append(Spacer(1, 12))

    if export_presence_df is not None and not export_presence_df.empty:
        story.append(Paragraph("Well Log Completeness — Table", styles["Heading2"]))
        story.append(_df_to_table(export_presence_df))
        story.append(Spacer(1, 12))

    if export_stats:
        story.append(Paragraph("Statistics", styles["Heading2"]))
        for metric_name, df_metric in export_stats:
            story.append(Paragraph(metric_name, styles["Heading3"]))
            story.append(_df_to_table(df_metric))
            story.append(Spacer(1, 8))

    if EXPORT_PNGS:
        story.append(Paragraph("Figures", styles["Heading2"]))
        for title, png in EXPORT_PNGS:
            story.append(Paragraph(title, styles["Heading3"]))
            img_buf = BytesIO(png)
            try:
                pil = PILImage.open(img_buf)
                w, h = pil.size
                max_w = 6.5 * inch
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
        file_name="automatic_seismic_inversion_eda_report.pdf",
        mime="application/pdf"
    )
except Exception as e:
    st.info(f"PDF export not available: {e}")

# N:\_USER_GLOBAL\PETREL\Prizm\wf\1_Wells_EDA\pages\2_Automatic_Rock_Physics_EDA.py

import streamlit as st
import numpy as np
import pandas as pd
import utils
from utils import try_get_project_name

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

# Optional (for Plotly PNG export inside HTML/PDF)
try:
    import kaleido  # noqa: F401
    _HAS_KALEIDO = True
except Exception:
    _HAS_KALEIDO = False

# ============================  STREAMLIT + NAV  ============================
st.set_page_config(
    page_title="GeoPython | Automatic Rock Physics EDA",
    layout="wide",
    initial_sidebar_state="expanded",
)
utils.render_grouped_sidebar_nav()

petrel_project = utils.get_petrel_connection()
utils.sidebar_footer(petrel_project, utils.LOGO_PATH, utils.APP_VERSION)

# ============================  STANDARDIZATION  ============================
import json, re
from pathlib import Path
from collections import defaultdict

def _slugify(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\(.*?\)", " ", s)
    s = s.replace("\\", " ").replace("/", " ").replace("_", " ").replace("-", " ")
    s = re.sub(r"[^A-Za-z0-9\s\.]+", " ", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "project"

proj_raw  = try_get_project_name(petrel_project) or petrel_project.get_current_project_name()
proj_slug = _slugify(proj_raw)
STD_DIR   = Path.home() / "Cegal" / "Prizm" / "geoPython" / proj_slug / "logStandarization"

def _load_std_artifacts():
    name_map = {}
    std_to_group = {}
    std_to_orig = defaultdict(list)
    nm_files  = sorted(STD_DIR.glob(f"log_name_map__{proj_slug}__*__*.json"))
    rec_files = sorted(STD_DIR.glob(f"log_mapping_records__{proj_slug}__*__*.json"))
    if nm_files:
        try:
            name_map = json.loads(nm_files[-1].read_text(encoding="utf-8"))
        except Exception:
            pass
    if rec_files:
        try:
            recs = json.loads(rec_files[-1].read_text(encoding="utf-8"))
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
        except Exception:
            pass

    for orig, std in name_map.items():
        std_to_orig[std].append(orig)
    return name_map, std_to_group, std_to_orig

NAME_MAP, STD_TO_GROUP, STD_TO_ORIG = _load_std_artifacts()
NAME_MAP_EMPTY = (len(NAME_MAP) == 0)

# ---- Robust mapping (works without stamped files too) ----
def _build_orig_to_std(name_map: dict, std_to_orig: dict) -> dict:
    d = dict(name_map) if name_map else {}
    for std, origs in (std_to_orig or {}).items():
        for o in origs:
            d.setdefault(str(o), str(std))
    return d

ORIG_TO_STD = _build_orig_to_std(NAME_MAP, STD_TO_ORIG)

# Add rock-physics synonyms (fallbacks)
_STD_SYNONYMS = {
    # petrophysical
    "GR_STD":   [r"\bgr\b", r"gamma\s*ray", r"\bgrc?\b"],
    "RHO_STD":  [r"\brho\b", r"\brhob\b", r"\bdens(ity)?\b", r"bulk\s*den"],
    "DT_STD":   [r"\bdt(co)?\b", r"\bsonic\b", r"compressional", r"p[-\s]*wave", r"dtp"],
    "DTS_STD":  [r"\bdts?\b", r"shear\s*(slowness|sonic)", r"s[-\s]*wave\s*sonic"],
    "VP_STD":   [r"\bvp\b", r"p[-\s]*wave\s*vel", r"compres+sional\s*vel", r"velp\b"],
    "VS_STD":   [r"\bvs\b", r"s[-\s]*wave\s*vel", r"shear\s*vel", r"vels\b"],
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

def to_standardized_options(original_names: list[str]) -> list[str]:
    if NAME_MAP_EMPTY:
        return sorted({str(n) for n in original_names})
    std_set = set()
    for n in original_names:
        if n in NAME_MAP:
            std_set.add(NAME_MAP[n])
    # Include common standardized codes even if not mapped (so users can pick derived families)
    std_set |= {"GR_STD", "RHO_STD", "DT_STD", "DTS_STD", "VP_STD", "VS_STD"}
    return sorted(std_set)

def eligible_well_names(wells) -> list[str]:
    if NAME_MAP_EMPTY:
        return [w.petrel_name for w in wells]
    allowed = set(NAME_MAP.keys())
    return [
        w.petrel_name for w in wells
        if any(getattr(lg, "petrel_name", "") in allowed for lg in getattr(w, "logs", []))
    ]

# ============================  PAGE HELPERS  ============================
def _depth_selector_ui(key_prefix, selected_wells, selected_logs_orig, well_dict, tops_df):
    md_min_default, md_max_default = utils.get_global_md_range(
        selected_wells or list(well_dict.keys()),
        selected_logs_orig or [],
        well_dict
    )
    if "depth_mode_" + key_prefix not in st.session_state:
        st.session_state["depth_mode_" + key_prefix] = "Slider"

    depth_selection_mode = st.radio(
        "Depth selection mode", ["Slider", "Tops"], horizontal=True,
        key="depth_mode_" + key_prefix
    )
    label = ""
    if depth_selection_mode == "Slider":
        rng_key = f"{key_prefix}_depth_slider"
        default_range = st.session_state.get(rng_key, (int(md_min_default), int(md_max_default)))
        depth_min, depth_max = st.slider(
            "Select depth range (MD)",
            min_value=int(md_min_default),
            max_value=int(md_max_default),
            value=default_range, step=1, key=rng_key
        )
        label = f"{depth_min}–{depth_max} m (MD)"
    else:
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
    return utils.get_logs_data_for_wells_logs(
        wells_list=wells_sel,
        logs_list=logs_sel_orig,
        _wells=wells_all,
        depth_min=dmin,
        depth_max=dmax
    )

def _ensure_logs_in_df_long(df_long: pd.DataFrame,
                            logs_std: list[str],
                            wells_sel: list[str],
                            wells_all,
                            dmin: int, dmax: int) -> pd.DataFrame:
    present = set(map(str, df_long.get("Log", pd.Series(dtype=str)).unique()))
    need_std = [s for s in logs_std if s not in present]
    if not need_std:
        return df_long
    need_orig = []
    for s in need_std:
        need_orig.extend(STD_TO_ORIG.get(s, [s]))
    add = utils.get_logs_data_for_wells_logs(
        wells_list=wells_sel, logs_list=need_orig, _wells=wells_all, depth_min=dmin, depth_max=dmax
    )
    if add is not None and not add.empty and "Log" in add.columns:
        add = add.copy()
        add["Log"] = add["Log"].map(_to_std).astype(str)
        df_long = pd.concat([df_long, add], ignore_index=True)
    return df_long

# ---------- Plot exporters ----------
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

# ============================  PAGE BODY  ============================
st.title("Automatic Rock Physics EDA")

# ----------- Global filters -----------
wells = utils.get_all_wells_flat(petrel_project)
well_dict = {w.petrel_name: w for w in wells}
tops_df = utils.load_tops_dataframe(petrel_project, utils._sel_cache_key())

filters, preview = st.columns([1, 2.2])

with filters:
    st.subheader("Global Filters")

    elig_well_names = eligible_well_names(wells) or [w.petrel_name for w in wells]
    sel_wells = st.multiselect("Select wells", options=elig_well_names, default=elig_well_names)

    # Build original-log pool limited to selected wells
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

    # Standardized options visible in project
    log_options_std = to_standardized_options(orig_pool)

    # Families present (from stamped artifacts, if any)
    from collections import defaultdict as _dd
    GROUP_TO_STD = _dd(list)
    for std in log_options_std:
        grp = STD_TO_GROUP.get(std, "(unlabeled)")
        GROUP_TO_STD[grp].append(std)

    family_options = sorted([g for g in GROUP_TO_STD.keys() if g])
    # Prefer Gamma Ray / Density / Sonic defaults
    def _pick_families(families):
        wants = ["Gamma Ray", "Density", "Sonic", "DT"]
        out = []
        for want in wants:
            for g in families:
                if want.lower() in g.lower(): out.append(g); break
        return list(dict.fromkeys(out))
    default_families = _pick_families(family_options) or family_options[:3]

    sel_groups = st.multiselect(
        "Select families (groups)",
        options=family_options,
        default=default_families
    )
    sel_logs_std = sorted({std for g in sel_groups for std in GROUP_TO_STD.get(g, [])})
    sel_logs_orig = []
    for std in sel_logs_std:
        sel_logs_orig.extend(STD_TO_ORIG.get(std, [std]))

    # Depth window
    dmin, dmax, depth_label = _depth_selector_ui(
        key_prefix="rp",
        selected_wells=sel_wells or elig_well_names,
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

# -------------------- Report contents (config) --------------------
st.subheader("Report Contents")
with st.expander("Configure sections (Rock Physics EDA)", expanded=True):
    # Input availability summary
    st.markdown("**Required inputs (for full EDA)**")
    colA, colB, colC, colD = st.columns(4)
    with colA: use_RHO = st.checkbox("Density (RHO_STD)", value=True, key="rp_rho")
    with colB: use_DT  = st.checkbox("Compressional slowness (DT_STD) or VP_STD", value=True, key="rp_dt")
    with colC: use_DTS = st.checkbox("Shear slowness (DTS_STD) or VS_STD (optional)", value=True, key="rp_dts")
    with colD: use_GR  = st.checkbox("Gamma Ray (GR_STD) for Vsh (optional)", value=True, key="rp_gr")

    # Units + derivation options
    st.markdown("**Units & derivations**")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        rho_unit = st.selectbox("Density unit", ["g/cc", "kg/m³"], index=0)
    with col2:
        dt_unit  = st.selectbox("DT unit (compressional)", ["µs/ft", "µs/m"], index=0)
    with col3:
        dts_unit = st.selectbox("DTS unit (shear)", ["µs/ft", "µs/m"], index=0)
    with col4:
        v_unit   = st.selectbox("VP/VS log unit (if present)", ["m/s", "km/s"], index=0)
    with col5:
        estimate_vs = st.checkbox("Estimate VS via Castagna line if DTS/VS missing", value=True)

    st.caption("Castagna (mudrock) overlay: VP = 1.16 VS + 1.36 (km/s); VS ≈ 0.862 VP − 1.172 (km/s).")

    # Vsh & porosity settings
    st.markdown("**Vsh & ρ‑φ settings**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        gr_clean = st.number_input("GR clean (API)", value=30.0, step=1.0)
    with c2:
        gr_shale = st.number_input("GR shale (API)", value=100.0, step=1.0)
    with c3:
        rho_matrix = st.number_input("ρ_matrix (g/cc)", value=2.65, step=0.01)
    with c4:
        rho_fluid  = st.number_input("ρ_fluid (g/cc)",  value=1.00, step=0.01)

    # Sections
    inc_presence    = st.checkbox("Include: Input presence matrix/table", value=True)
    inc_vp_vs       = st.checkbox("Include: Vp–Vs crossplot (Castagna overlay)", value=True)
    inc_ai_vpvs     = st.checkbox("Include: AI vs VP/VS crossplot", value=True)
    inc_lmr         = st.checkbox("Include: λρ–μρ crossplot", value=True)
    inc_pr_ai       = st.checkbox("Include: Poisson’s ratio vs AI", value=False)
    inc_hist_derived= st.checkbox("Include: Histograms (derived logs)", value=False)

    # Global multi‑well curves for derived logs
    inc_global_plot = st.checkbox("Include: Global Well Log Visualization (derived)", value=True)
    logs_for_global = []
    if inc_global_plot:
        defaults = ["VP", "VS", "AI", "VPVS", "LAMBDARHO", "MURHO"]
        logs_for_global = st.multiselect(
            "Derived logs to plot (columns)",
            options=defaults,
            default=defaults[:3]
        )

# ---- Run button ----
run_clicked = st.button("Run Automatic Rock Physics EDA", type="primary")
st.markdown("---")

# ============================  EXECUTION  ============================
if run_clicked:
    if not sel_wells:
        st.warning("Select at least one well.")
        st.stop()

    # Fetch base logs in depth window
    base_sel_std = []
    if use_RHO: base_sel_std.append("RHO_STD")
    if use_DT:  base_sel_std += ["DT_STD", "VP_STD"]  # fetch either/or
    if use_DTS: base_sel_std += ["DTS_STD", "VS_STD"]
    if use_GR:  base_sel_std.append("GR_STD")

    sel_logs_orig = []
    for std in base_sel_std:
        sel_logs_orig.extend(STD_TO_ORIG.get(std, [std]))
    df_long = _build_long_logs(sel_wells, sel_logs_orig, wells, dmin, dmax)
    if df_long is None or not isinstance(df_long, pd.DataFrame) or df_long.empty:
        df_long = pd.DataFrame(columns=["Well", "Log", "MD", "Value"])
    else:
        df_long = df_long.copy()
        df_long["Log"] = df_long["Log"].map(_to_std).astype(str)

    # ---------- Presence summary (inputs) ----------
    if inc_presence:
        with st.expander("Input Presence (RHO / DT or VP / DTS or VS / GR)", expanded=True):
            pres_logs = [std for std in ["RHO_STD", "DT_STD", "VP_STD", "DTS_STD", "VS_STD", "GR_STD"]
                         if std in set(df_long["Log"].unique())]
            if not pres_logs:
                st.info("No input logs found for the current selection.")
            else:
                # wide Boolean presence
                pres = (
                    df_long[df_long["Log"].isin(pres_logs)]
                    .groupby(["Well", "Log"])["Value"]
                    .apply(lambda s: s.notna().any())
                    .unstack("Log")
                    .fillna(False)
                    .astype(bool)
                    .reset_index()
                )
                pres = pres.replace({True:"✓", False:"✗"})
                st.dataframe(pres, use_container_width=True)

    # ---------- Derived properties ----------
    # Build wide pivot to compute per (Well, MD)
    wide = (
        df_long.pivot_table(index=["Well", "MD"], columns="Log", values="Value", aggfunc="mean")
        .sort_index()
    )

    # Unit conversions -> numeric arrays
    def _to_mps_from_dt(dt, unit):
        dt = pd.to_numeric(dt, errors="coerce")
        if unit == "µs/ft":
            # V (m/s) = 0.3048 / (DT_us_per_ft * 1e-6)
            return 0.3048 / (dt * 1e-6)
        else:  # µs/m
            return 1e6 / dt

    def _to_mps(v, unit):
        v = pd.to_numeric(v, errors="coerce")
        return v*1e3 if unit == "km/s" else v

    def _rho_to_kgm3(rho, unit):
        rho = pd.to_numeric(rho, errors="coerce")
        return rho*1000.0 if unit == "g/cc" else rho

    # Derive VP
    VP = None
    if "VP_STD" in wide.columns:
        VP = _to_mps(wide["VP_STD"], v_unit)
    elif "DT_STD" in wide.columns:
        VP = _to_mps_from_dt(wide["DT_STD"], dt_unit)

    # Derive VS
    VS = None
    if "VS_STD" in wide.columns:
        VS = _to_mps(wide["VS_STD"], v_unit)
    elif "DTS_STD" in wide.columns:
        VS = _to_mps_from_dt(wide["DTS_STD"], dts_unit)
    elif estimate_vs and VP is not None:
        # Castagna mudrock line (km/s): VP = 1.16 VS + 1.36
        # -> VS_kmps = (VP_kmps - 1.36)/1.16
        VS = (VP/1000.0 - 1.36)/1.16 * 1000.0  # m/s

    # Density
    RHO = _rho_to_kgm3(wide["RHO_STD"], rho_unit) if "RHO_STD" in wide.columns else None

    # Optional GR, Vsh, Porosity
    GR = pd.to_numeric(wide["GR_STD"], errors="coerce") if "GR_STD" in wide.columns else None
    VSH = None
    if GR is not None and np.isfinite(gr_shale - gr_clean) and (gr_shale > gr_clean):
        VSH = np.clip((GR - gr_clean)/(gr_shale - gr_clean), 0.0, 1.0)

    PHI = None
    if RHO is not None:
        # Porosity from density: phi = (rho_matrix - rho_bulk) / (rho_matrix - rho_fluid)
        # Use g/cc constants, convert to kg/m3 to match RHO if needed
        rho_m = 1000.0*rho_matrix
        rho_f = 1000.0*rho_fluid
        denom = max(1e-6, (rho_m - rho_f))
        PHI = np.clip((rho_m - RHO)/denom, 0.0, 1.0)

    # Elastic properties (SI)
    AI = SI = VPVS = PR = LAMBDARHO = MURHO = K = MU = None
    if (VP is not None) and (RHO is not None):
        AI = RHO * VP
    if (VS is not None) and (RHO is not None):
        SI = RHO * VS
    if (VP is not None) and (VS is not None):
        vpvs = pd.to_numeric(VP/VS, errors="coerce")
        VPVS = vpvs
        r = (vpvs**2)
        PR = (r - 2.0) / (2.0*(r - 1.0))  # Poisson's ratio
    if (VP is not None) and (VS is not None) and (RHO is not None):
        MU = RHO * (VS**2)                              # shear modulus
        K  = RHO * (VP**2 - (4.0/3.0)*(VS**2))          # bulk modulus
        LAMBDARHO = RHO * (VP**2 - 2.0*(VS**2))         # λρ (LMR)
        MURHO     = RHO * (VS**2)                       # μρ (LMR)

    # Assemble derived long DF
    derived_cols = {
        "VP": VP, "VS": VS, "AI": AI, "SI": SI, "VPVS": VPVS, "PR": PR,
        "LAMBDARHO": LAMBDARHO, "MURHO": MURHO, "K": K, "MU": MU,
        "VSH": VSH, "PHI": PHI
    }
    frames = []
    for name, series in derived_cols.items():
        if series is not None:
            s = series.rename("Value").to_frame()
            s["Log"] = name
            frames.append(s)
    df_der = pd.concat(frames, axis=0) if frames else pd.DataFrame(columns=["Value","Log"])
    if not df_der.empty:
        df_der = df_der.reset_index().rename(columns={"level_0":"Well","level_1":"MD"})
        df_der = df_der[["Well","Log","MD","Value"]]

    st.success("Executed with current selections.")

    # ---------------- VISUALS ----------------
    # Helper: color by VSH or PHI if available
    def _color_series(name: str):
        if name == "VSH" and VSH is not None: return VSH
        if name == "PHI" and PHI is not None: return PHI
        return None

    # 1) Vp–Vs crossplot + Castagna overlay
    if inc_vp_vs and (VP is not None) and (VS is not None):
        with st.expander("Vp–Vs crossplot (Castagna overlay)", expanded=True):
            dfp = pd.DataFrame({"VP": VP/1000.0, "VS": VS/1000.0}).reset_index()  # km/s for plotting
            dfp = dfp.rename(columns={"level_0":"Well","level_1":"MD"})
            dfp = dfp.dropna()
            # optional color by VSH/PHI
            color_by = st.selectbox("Color by", ["None","VSH","PHI"], index=1 if VSH is not None else 0, key="vpvs_color")
            cvals = _color_series(color_by)
            if cvals is not None:
                dfp[color_by] = cvals.values

            fig = px.scatter(
                dfp, x="VP", y="VS", color=(color_by if color_by!="None" else None),
                opacity=0.55, render_mode="webgl",
                title="Vp–Vs (km/s)"
            )
            # Castagna mudrock line: VP = 1.16*VS + 1.36 (km/s)
            vs_line = np.linspace(float(dfp["VS"].min()), float(dfp["VS"].max()), 50)
            vp_line = 1.16*vs_line + 1.36
            fig.add_trace(go.Scatter(x=vp_line, y=vs_line, mode="lines",
                                     line=dict(width=2), name="Castagna (mudrock)"))
            fig.update_layout(xaxis_title="Vp (km/s)", yaxis_title="Vs (km/s)")
            st.plotly_chart(fig, use_container_width=True)
            _capture_plotly_png(fig, "Vp–Vs crossplot (Castagna)")

    # 2) AI vs VP/VS
    if inc_ai_vpvs and (AI is not None) and (VPVS is not None):
        with st.expander("AI vs VP/VS crossplot", expanded=False):
            dfp = pd.DataFrame({"AI": AI, "VPVS": VPVS}).reset_index().rename(columns={"level_0":"Well","level_1":"MD"})
            dfp = dfp.dropna()
            color_by = st.selectbox("Color by", ["None","VSH","PHI"], index=1 if VSH is not None else 0, key="ai_color")
            cvals = _color_series(color_by)
            if cvals is not None:
                dfp[color_by] = cvals.values
            fig = px.scatter(
                dfp, x="VPVS", y="AI", color=(color_by if color_by!="None" else None),
                opacity=0.55, render_mode="webgl",
                title="AI vs VP/VS"
            )
            fig.update_layout(xaxis_title="VP/VS", yaxis_title="AI (kg/m³·m/s)")
            st.plotly_chart(fig, use_container_width=True)
            _capture_plotly_png(fig, "AI vs VP/VS")

    # 3) λρ–μρ crossplot
    if inc_lmr and (LAMBDARHO is not None) and (MURHO is not None):
        with st.expander("λρ–μρ crossplot (LMR)", expanded=False):
            dfp = pd.DataFrame({"LAMBDARHO": LAMBDARHO, "MURHO": MURHO}).reset_index().rename(columns={"level_0":"Well","level_1":"MD"})
            dfp = dfp.dropna()
            color_by = st.selectbox("Color by", ["None","VSH","PHI"], index=1 if VSH is not None else 0, key="lmr_color")
            cvals = _color_series(color_by)
            if cvals is not None:
                dfp[color_by] = cvals.values
            fig = px.scatter(
                dfp, x="MURHO", y="LAMBDARHO", color=(color_by if color_by!="None" else None),
                opacity=0.55, render_mode="webgl",
                title="λρ–μρ"
            )
            fig.update_layout(xaxis_title="μρ (Pa)", yaxis_title="λρ (Pa)")
            st.plotly_chart(fig, use_container_width=True)
            _capture_plotly_png(fig, "λρ–μρ crossplot")

    # 4) Poisson’s ratio vs AI
    if inc_pr_ai and (PR is not None) and (AI is not None):
        with st.expander("Poisson’s ratio vs AI", expanded=False):
            dfp = pd.DataFrame({"PR": PR, "AI": AI}).reset_index().rename(columns={"level_0":"Well","level_1":"MD"})
            dfp = dfp.dropna()
            color_by = st.selectbox("Color by", ["None","VSH","PHI"], index=1 if VSH is not None else 0, key="pr_color")
            cvals = _color_series(color_by)
            if cvals is not None:
                dfp[color_by] = cvals.values
            fig = px.scatter(
                dfp, x="AI", y="PR", color=(color_by if color_by!="None" else None),
                opacity=0.55, render_mode="webgl",
                title="Poisson’s ratio vs AI"
            )
            fig.update_layout(xaxis_title="AI (kg/m³·m/s)", yaxis_title="ν")
            st.plotly_chart(fig, use_container_width=True)
            _capture_plotly_png(fig, "Poisson vs AI")

    # 5) Histograms of derived logs
    if inc_hist_derived and not df_der.empty:
        with st.expander("Histograms — Derived logs", expanded=False):
            pick_logs = st.multiselect(
                "Pick derived logs", options=sorted(df_der["Log"].unique().tolist()),
                default=["VP", "VS", "AI", "VPVS", "LAMBDARHO", "MURHO"]
            )
            nbins = st.slider("Bins", 10, 200, 60, 5)
            alpha = st.slider("Histogram opacity", 0.1, 1.0, 0.45, 0.05)
            for lg in pick_logs:
                dfl = df_der[df_der["Log"] == lg].copy()
                dfl["Value"] = pd.to_numeric(dfl["Value"], errors="coerce")
                dfl = dfl.dropna()
                if dfl.empty:
                    st.info(f"No samples for {lg}."); continue
                fig = go.Figure()
                palette = px.colors.qualitative.D3
                wells_present = sorted(dfl["Well"].unique())
                for i, w in enumerate(wells_present):
                    sub = dfl.loc[dfl["Well"] == w, "Value"]
                    fig.add_histogram(
                        x=sub, name=w, nbinsx=nbins,
                        opacity=alpha, marker_color=palette[i % len(palette)],
                        histnorm=""  # counts
                    )
                fig.update_layout(barmode="overlay", title=f"Histogram — {lg}")
                st.plotly_chart(fig, use_container_width=True)
                _capture_plotly_png(fig, f"Histogram — {lg}")

    # 6) Global multi‑well curves (derived)
    if inc_global_plot and not df_der.empty and logs_for_global:
        with st.expander("Global Well Log Visualization (derived)", expanded=True):
            logs_to_plot = [lg for lg in logs_for_global if lg in set(df_der["Log"].unique())]
            if not logs_to_plot:
                st.info("No selected derived logs found."); 
            else:
                lw = st.slider("Line width", 0.5, 3.0, 1.2, 0.1)
                height = st.slider("Figure height (px)", 400, 1200, 680, 20)
                show_legend = st.checkbox("Show legend", value=True)
                fig = make_subplots(rows=1, cols=len(logs_to_plot), shared_yaxes=True,
                                    horizontal_spacing=0.03, subplot_titles=logs_to_plot)
                wells_present = sorted(df_der["Well"].unique())
                palette = px.colors.qualitative.Plotly + px.colors.qualitative.D3
                color_map = {w: palette[i % len(palette)] for i, w in enumerate(wells_present)}
                for j, name in enumerate(logs_to_plot, start=1):
                    sub = df_der[df_der["Log"] == name].copy()
                    for w in wells_present:
                        sw = sub[sub["Well"] == w]
                        if sw.empty: continue
                        x = pd.to_numeric(sw["Value"], errors="coerce")
                        y = pd.to_numeric(sw["MD"], errors="coerce")
                        m = x.notna() & y.notna()
                        if not m.any(): continue
                        fig.add_trace(
                            go.Scattergl(
                                x=x[m], y=y[m], mode="lines",
                                name=w, legendgroup=w, showlegend=(j==1 and show_legend),
                                line=dict(width=lw, color=color_map[w]),
                                hovertemplate=f"Well: {w}<br>MD: %{{y}}<br>{name}: %{{x}}<extra></extra>"
                            ),
                            row=1, col=j
                        )
                    fig.update_xaxes(title_text=name, row=1, col=j)
                fig.update_yaxes(autorange="reversed", title_text="Measured Depth (MD)", row=1, col=1)
                fig.update_layout(height=height, margin=dict(l=40, r=20, t=50, b=40))
                st.plotly_chart(fig, use_container_width=True)
                _capture_plotly_png(fig, f"Global Well Log Visualization (derived: {', '.join(logs_to_plot)})")

else:
    st.info("Configure the sections above, then click **Run Automatic Rock Physics EDA**.")

# ============================  EXPORT  ============================
st.markdown("---")
st.subheader("Export Report")

if not _HAS_KALEIDO:
    st.info("Plotly figures will appear in the HTML; to embed them as images, install **kaleido** (`pip install -U kaleido`).")

# We export figures + (optionally) a small input presence table if it was computed
# (For brevity here we export only figures, like your other page.)
now = datetime.now().strftime("%Y-%m-%d %H:%M")
wells_txt = ", ".join(sel_wells) if 'sel_wells' in locals() and sel_wells else "—"
depth_label = locals().get("depth_label", "—")

def _df_to_html(df: pd.DataFrame) -> str:
    try:
        return df.to_html(index=True, border=0, classes="table", justify="center")
    except Exception:
        return "<p><i>Table could not be rendered.</i></p>"

html_parts = [
    "<!DOCTYPE html><html><head><meta charset='utf-8'>",
    "<style>",
    "body{font-family:Arial,Helvetica,sans-serif; margin:24px;}",
    "h1{margin-bottom:4px;} .muted{color:#555;}",
    "h2{margin-top:28px; border-bottom:1px solid #ddd; padding-bottom:4px;}",
    "img{max-width:100%;}",
    "</style></head><body>",
    "<h1>Automatic Rock Physics EDA</h1>",
    f"<div class='muted'>Generated: {now}</div>",
    "<h2>Selections</h2>",
    f"<p><b>Wells:</b> {wells_txt}</p>",
    f"<p><b>Depth window:</b> {depth_label}</p>",
]

# Figures (as base64 PNG)
if EXPORT_PNGS:
    html_parts.append("<h2>Figures</h2>")
    for title, png in EXPORT_PNGS:
        b64 = base64.b64encode(png).decode("utf-8")
        html_parts.append(f"<h3>{title}</h3>")
        html_parts.append(f"<img src='data:image/png;base64,{b64}'/>")

if not _HAS_KALEIDO:
    html_parts.append("<p style='color:#a00'><i>Note: To include Plotly figures as images, install <b>kaleido</b>.</i></p>")

html_parts.append("</body></html>")
export_html = "".join(html_parts).encode("utf-8")

st.download_button(
    "Download HTML report",
    data=export_html,
    file_name="automatic_rock_physics_eda_report.html",
    mime="text/html",
    type="primary"
)

# PDF export (figures only)
try:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("Automatic Rock Physics EDA", styles["Title"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Generated: {now}", styles["Normal"]))
    story.append(Paragraph(f"Wells: {wells_txt}", styles["Normal"]))
    story.append(Paragraph(f"Depth window: {depth_label}", styles["Normal"]))
    story.append(Spacer(1, 12))
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
        file_name="automatic_rock_physics_eda_report.pdf",
        mime="application/pdf"
    )
except Exception as e:
    st.info(f"PDF export not available: {e}")
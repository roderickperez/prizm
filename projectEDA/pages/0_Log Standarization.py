# N:\_USER_GLOBAL\PETREL\Prizm\wf\EDA\pages\0_Log Standarization.py
# (Log Standardization & Mnemonics Library)

import os
import re
import json
from pathlib import Path
from collections import defaultdict, Counter
from io import BytesIO
from datetime import datetime, date
import getpass

import streamlit as st
import pandas as pd

# PDF/HTML exports
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

import utils  # Petrel connection + footer
from utils import try_get_project_name

import utils
utils.render_grouped_sidebar_nav()
# ==========================  STREAMLIT CONFIG  ==========================
st.set_page_config(
    page_title="GeoPython | Log Standardization",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============== Petrel connection ===============
petrel_project = utils.get_petrel_connection()
utils.sidebar_footer(petrel_project, utils.LOGO_PATH, utils.APP_VERSION)

# ------------------------------- Helpers --------------------------------

# ------------------------------- Paths (UNIFIED) ----------------------------------

def slugify(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\(.*?\)", " ", s)
    s = s.replace("\\", " ").replace("/", " ").replace("_", " ").replace("-", " ")
    s = re.sub(r"[^A-Za-z0-9\s\.]+", " ", s)   # keep dots for *.pet projects
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "project"

PROJECT_NAME_RAW = try_get_project_name(petrel_project) or "project"
PROJECT_SLUG = slugify(PROJECT_NAME_RAW)

USERNAME   = os.environ.get("USERNAME") or Path.home().name
DATE_STAMP = datetime.now().strftime("%Y%m%d")

# Per-user, per-project local base:
USER_ROOT = Path.home() / "Cegal" / "Prizm"
GP_ROOT   = USER_ROOT / "geoPython"
PROJ_ROOT = GP_ROOT / PROJECT_SLUG

STD_DIR   = PROJ_ROOT / "logStandarization"
GANTT_DIR = PROJ_ROOT / "ganttChart"  # for your other page
STD_DIR.mkdir(parents=True, exist_ok=True)
GANTT_DIR.mkdir(parents=True, exist_ok=True)

def stamped_path(prefix: str) -> Path:
    return STD_DIR / f"{prefix}__{PROJECT_SLUG}__{USERNAME}__{DATE_STAMP}.json"

def load_latest_stamped_for(prefix: str) -> dict:
    """
    Load the most-recent stamped JSON for this user+project; {} if none.
    """
    files = sorted(STD_DIR.glob(f"{prefix}__{PROJECT_SLUG}__{USERNAME}__*.json"), reverse=True)
    if not files:
        return {}
    try:
        return json.loads(files[0].read_text(encoding="utf-8"))
    except Exception:
        return {}

def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

# Stamped targets (ONLY)
SUMMARY_JSON_STAMPED       = stamped_path("log_groups_summary")
MAPPING_JSON_STAMPED       = stamped_path("log_mapping")
MAPPING_RECORDS_STAMPED    = stamped_path("log_mapping_records")
NAME_MAP_JSON_STAMPED      = stamped_path("log_name_map")
PACKAGE_JSON_STAMPED       = stamped_path("mnemonics_mapping_package")

# Shared (read-only) master library
NETWORK_MNEM_PATH = Path(r"N:\_USER_GLOBAL\PETREL\Prizm\wf\referenceDocuments\mnemonics_master.json")

# Optional: per-user copy of the library (keep if you want to allow local edits to the library itself)
USER_MNEM_LATEST  = STD_DIR / "mnemonics_user_library.json"
USER_MNEM_STAMPED = stamped_path("mnemonics_user_library")

# ---------------------------- Defaults/Library ---------------------------
def default_mnemonics_library() -> dict:
    """
    Seed groups and aliases per OMV Standard‑Log conventions.
    canonical = Curve Mnemonic / Output Name (e.g., 'GR_STD', 'DT_STD', ...).
    """
    return {
        # ---------------- Primary ----------------
        "Bit Size": {
            "canonical": "BS_STD",
            "aliases": ["bs", "bit size", "bit", "bs_std", "bs cmp", "bs_cmp"],
            "style": {"color": "#000000", "line_style": "dash", "scale": [6, 26]},
            "units": ["in"],
            "description": "STANDARD-LOG BIT SIZE",
        },
        "Caliper": {
            "canonical": "CALI_STD",
            "aliases": ["caliper", "cali", "hole size", "cal", "cali_std", "cali cmp", "cali_cmp"],
            "style": {"color": "#0000FF", "line_style": "solid", "scale": [6, 26]},
            "units": ["in", "mm"],
            "description": "STANDARD-LOG CALIPER",
        },
        "Sonic (DT)": {
            "canonical": "DT_STD",
            "aliases": ["dt", "sonic", "compressional", "ac", "dt_std", "dt cmp", "dt_cmp", "dtc"],
            "style": {"color": "#32CD32", "line_style": "solid", "scale": [240, 40]},  # alt window 800–130 as needed
            "units": ["µs/ft", "µs/m"],
            "description": "STANDARD-LOG ACOUSTIC SLOWNESS COMPRESSIONAL",
        },
        "Gamma Ray": {
            "canonical": "GR_STD",
            "aliases": ["gr", "gamma ray", "gamma", "gammaray", "gr_std", "gr cmp", "gr_cmp"],
            "style": {"color": "#00BE00", "line_style": "solid", "scale": [0, 150]},
            "units": ["gAPI"],
            "description": "STANDARD-LOG GAMMA RAY",
        },
        "Neutron Porosity": {
            "canonical": "NEU_STD",
            "aliases": ["neutron", "neu", "nphi", "phi_n", "neu_std", "neu cmp", "neu_cmp"],
            "style": {"color": "#0000FF", "line_style": "dash", "scale": [0.45, -0.15]},
            "units": ["m3/m3", "v/v", "pu"],
            "description": "STANDARD-LOG NEUTRON POROSITY",
        },
        "PEF": {
            "canonical": "PEF_STD",
            "aliases": ["pef", "photoelectric", "pe", "pef_std", "pef cmp", "pef_cmp"],
            "style": {"color": "#A52A2A", "line_style": "solid", "scale": [0, 10]},
            "units": ["b/e", "unitless"],
            "description": "STANDARD-LOG PHOTOELECTRIC FACTOR",
        },
        "Resistivity Deep": {
            "canonical": "RD_STD",
            "aliases": ["rd", "ild", "rt", "res deep", "deep resistivity", "rd_std", "rd cmp", "rd_cmp"],
            "style": {"color": "#FF0000", "line_style": "solid", "scale": [0.2, 2000]},
            "units": ["ohm.m", "ohmm"],
            "description": "STANDARD-LOG RESISTIVITY DEEP",
        },
        "Density": {
            "canonical": "RHO_STD",
            "aliases": ["density", "rho", "rhob", "rho_std", "rho cmp", "rho_cmp", "rhob_gcc"],
            "style": {"color": "#FF0000", "line_style": "dash", "scale": [1.95, 2.95]},
            "units": ["g/cm3", "g/cc"],
            "description": "STANDARD-LOG FORMATION DENSITY",
        },
        "Resistivity Medium": {
            "canonical": "RM_STD",
            "aliases": ["rm", "ilm", "res medium", "medium resistivity", "rm_std", "rm cmp", "rm_cmp"],
            "style": {"color": "#FF00FF", "line_style": "solid", "scale": [0.2, 2000]},
            "units": ["ohm.m", "ohmm"],
            "description": "STANDARD-LOG RESISTIVITY MEDIUM",
        },
        "Resistivity Shallow": {
            "canonical": "RS_STD",
            "aliases": ["rs", "ils", "sfl", "shallow resistivity", "rs_std", "rs cmp", "rs_cmp"],
            "style": {"color": "#0000FF", "line_style": "solid", "scale": [0.2, 2000]},
            "units": ["ohm.m", "ohmm"],
            "description": "STANDARD-LOG RESISTIVITY SHALLOW",
        },
        "Resistivity Micro": {
            "canonical": "RXO_STD",
            "aliases": ["rxo", "microres", "microlog", "rxo_std", "rxo cmp", "rxo_cmp"],
            "style": {"color": "#000000", "line_style": "solid", "scale": [0.2, 2000]},
            "units": ["ohm.m", "ohmm"],
            "description": "STANDARD-LOG RESISTIVITY MICRO",
        },
        "SP": {
            "canonical": "SP_STD",
            "aliases": ["sp", "spontaneous potential", "sp_std", "sp cmp", "sp_cmp"],
            "style": {"color": "#FF0000", "line_style": "solid", "scale": [-160, 40]},
            "units": ["mV"],
            "description": "STANDARD-LOG SPONTANEOUS POTENTIAL",
        },
        "TVD": {
            "canonical": "TVD_STD",
            "aliases": ["tvd", "true vertical depth", "tvd_std", "tvd cmp", "tvd_cmp"],
            "style": {"color": "#00BE00", "line_style": "solid", "scale": [None, None]},
            "units": ["m"],
            "description": "STANDARD-LOG TRUE VERTICAL DEPTH",
        },
        "TVDSS": {
            "canonical": "TVDSS_STD",
            "aliases": ["tvdss", "true vertical depth subsea", "tvdss_std", "tvdss cmp", "tvdss_cmp"],
            "style": {"color": "#0000FF", "line_style": "solid", "scale": [None, None]},
            "units": ["m"],
            "description": "STANDARD-LOG TRUE VERTICAL DEPTH SUBSEA",
        },

        # ---------------- Secondary ----------------
        "Density Correction": {
            "canonical": "DRHO_STD",
            "aliases": ["drho", "delta rho", "drho_std", "drho cmp", "drho_cmp"],
            "style": {"color": "#000000", "line_style": "dot", "scale": [-0.75, 0.25]},
            "units": ["g/cm3"],
            "description": "STANDARD-LOG FORMATION DENSITY CORRECTION",
        },
        "Sonic Shear (DTS)": {
            "canonical": "DTS_STD",
            "aliases": ["dts", "shear", "dt_s", "dts_std", "dts cmp", "dts_cmp"],
            "style": {"color": "#FF0000", "line_style": "solid", "scale": [460, 60]},  # alt 1500–200 µs/ft if needed
            "units": ["µs/ft", "µs/m"],
            "description": "STANDARD-LOG ACOUSTIC SLOWNESS SHEAR",
        },
        "Sonic Stoneley (DTST)": {
            "canonical": "DTST_STD",
            "aliases": ["dtst", "stoneley", "dtst_std", "dtst cmp", "dtst_cmp"],
            "style": {"color": "#0000FF", "line_style": "solid", "scale": [None, None]},
            "units": ["µs/ft", "µs/m"],
            "description": "STANDARD-LOG ACOUSTIC SLOWNESS STONELEY",
        },
        "Potassium": {
            "canonical": "POTA_STD",
            "aliases": ["pota", "k", "k_pct", "pota_std", "pota cmp", "pota_cmp"],
            "style": {"color": "#0000FF", "line_style": "solid", "scale": [-10, 10]},
            "units": ["%"],
            "description": "STANDARD-LOG POTASSIUM",
        },
        "Thorium": {
            "canonical": "THOR_STD",
            "aliases": ["thor", "th", "thor_std", "thor cmp", "thor_cmp"],
            "style": {"color": "#FF00FF", "line_style": "solid", "scale": [-10, 30]},
            "units": ["ppm"],
            "description": "STANDARD-LOG THORIUM",
        },
        "Uranium": {
            "canonical": "URAN_STD",
            "aliases": ["uran", "u", "uran_std", "uran cmp", "uran_cmp"],
            "style": {"color": "#FF0000", "line_style": "solid", "scale": [-10, 30]},
            "units": ["ppm"],
            "description": "STANDARD-LOG URANIUM",
        },
        "Velocity (Compressional)": {
            "canonical": "VELC_STD",
            "aliases": ["velc", "vp", "velc_std", "velc cmp", "velc_cmp"],
            "style": {"color": "#FF00FF", "line_style": "solid", "scale": [0, 8000]},
            "units": ["m/s"],
            "description": "STANDARD-LOG VELOCITY COMPRESSIONAL",
        },
        "Velocity (Shear)": {
            "canonical": "VELS_STD",
            "aliases": ["vels", "vs", "vels_std", "vels cmp", "vels_cmp"],
            "style": {"color": "#A52A2A", "line_style": "dash", "scale": [0, 5000]},
            "units": ["m/s"],
            "description": "STANDARD-LOG VELOCITY SHEAR",
        },

        # ---------------- Best/Composite ----------------
        "Resistivity (Best)": {
            "canonical": "RES_STD",
            "aliases": ["res", "res_best", "res_std", "res_cmp", "best resistivity"],
            "style": {"color": "#FF0000", "line_style": "solid", "scale": [0.2, 2000]},
            "units": ["ohm.m", "ohmm"],
            "description": "Best (composite) resistivity per OMV",
        },

        # ---------------- Vintage ----------------
        "GR (uR/h or CPS or RADT)": {
            "canonical": "GR_UR/H_STD",  # representative; you may switch to GR_CPS_STD or GR_RADT_STD as needed
            "aliases": ["gr_ur/h", "gr_cps", "gr_radt", "gr_ur/h_std", "gr_cps_std", "gr_radt_std",
                        "gr_ur/h_cmp", "gr_cps_cmp", "gr_radt_cmp"],
            "style": {"color": "#00BE00", "line_style": "dash", "scale": [0, 1000]},
            "units": ["uR/h", "cps", "unitless"],
            "description": "Vintage gamma (micro‑roentgen/h, cps or Ra eq. per ton).",
        },
        "Micro Resistivity (Inverse)": {
            "canonical": "MINV_STD",
            "aliases": ["minv", "micro inv", "minv_std", "minv_cmp"],
            "style": {"color": "#A52A2A", "line_style": "solid", "scale": [0.2, 2000]},
            "units": ["ohm.m", "ohmm"],
            "description": "STANDARD-LOG RESISTIVITY MICRO INVERSE",
        },
        "Micro Resistivity (Normale)": {
            "canonical": "MNOR_STD",
            "aliases": ["mnor", "micro normale", "mnor_std", "mnor_cmp"],
            "style": {"color": "#800080", "line_style": "dash", "scale": [0.2, 2000]},
            "units": ["ohm.m", "ohmm"],
            "description": "STANDARD-LOG RESISTIVITY MICRO NORMALE",
        },
        "Neutron (API/CPS)": {
            "canonical": "NEU_API_STD",  # representative; alternative: NEU_CPS_STD
            "aliases": ["neu_api", "neu_cps", "neutron api", "neutron cps",
                        "neu_api_std", "neu_cps_std", "neu_api_cmp", "neu_cps_cmp"],
            "style": {"color": "#0000FF", "line_style": "long-dash", "scale": [0, 5000]},
            "units": ["API", "cps"],
            "description": "Vintage neutron (API or cps).",
        },
    }

def _safe_json_load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def load_mnemonics() -> tuple[dict, Path, str]:
    """
    Load library with precedence:
      1) User copy (per-user/per-project) if exists
      2) Shared network master (read-only)
      3) Default seed
    Returns (library, path_selected, mode_str)
    """
    if USER_MNEM_LATEST.exists():
        data = _safe_json_load(USER_MNEM_LATEST)
        if data:
            return data, USER_MNEM_LATEST, "Loaded user copy"
    if NETWORK_MNEM_PATH.exists():
        data = _safe_json_load(NETWORK_MNEM_PATH)
        if data:
            return data, NETWORK_MNEM_PATH, "Loaded shared master (read-only)"
    return default_mnemonics_library(), Path("[defaults]"), "Using defaults (not yet saved)"

def save_mnemonics(data: dict) -> list[str]:
    """
    Save ONLY to per-user/per-project location (do not modify shared master).
    Writes both a 'latest' and a stamped copy. Returns messages.
    """
    msgs = []
    try:
        USER_MNEM_LATEST.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        msgs.append(f"Saved user copy: {USER_MNEM_LATEST}")
    except Exception as e:
        msgs.append(f"Could not save user copy: {e}")
    try:
        USER_MNEM_STAMPED.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        msgs.append(f"Saved stamped user copy: {USER_MNEM_STAMPED.name}")
    except Exception as e:
        msgs.append(f"Could not save stamped copy: {e}")
    return msgs

def load_mapping(path: Path) -> dict:
    """
    Holds:
      {
        "targets": { "<group>": "<standardized name>" },
        "selection_mode": { "<group>": "One per group"|"Multiple allowed" },
        "preferred_log": { "<group>": "<one of detected names>" }
      }
    """
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"targets": {}, "selection_mode": {}, "preferred_log": {}}

def save_mapping(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

# -------------------------- Project log discovery ------------------------
def normalize_name(s: str) -> str:
    if s is None:
        return ""
    s = s.lower()
    s = re.sub(r"\(.*?\)", " ", s)
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def alias_matches(alias: str, candidate: str) -> bool:
    """
    Robust but conservative matching:
    - exact match
    - whole-word match if alias length >= 3
    - fallback substring for multi-word aliases
    """
    a = normalize_name(alias)
    c = normalize_name(candidate)
    if not a or not c:
        return False
    if a == c:
        return True
    if len(a) >= 3:
        if re.search(rf"\b{re.escape(a)}\b", c):
            return True
        if " " in a and a in c:
            return True
    return False

@st.cache_data(show_spinner=False)
def get_unique_project_logs(_wells) -> list:
    names = set()
    for w in _wells:
        for lg in getattr(w, "logs", []):
            if hasattr(lg, "petrel_name"):
                names.add(str(lg.petrel_name))
    return sorted(names)

def group_logs_by_family(all_log_names: list, library: dict):
    """Return (family->set(names)), and the set of unmatched names."""
    family_map = {fam: set() for fam in library.keys()}
    unmatched = set(all_log_names)
    norm_aliases = {fam: [normalize_name(a) for a in v.get("aliases", [])] for fam, v in library.items()}
    for name in all_log_names:
        matched_any = False
        for fam, aliases in norm_aliases.items():
            for a in aliases:
                if alias_matches(a, name):
                    family_map[fam].add(name)
                    matched_any = True
                    break
            if matched_any:
                break
        if matched_any and name in unmatched:
            unmatched.remove(name)
    return family_map, unmatched

def wells_containing_any_of(names_set: set, wells) -> int:
    if not names_set:
        return 0
    count = 0
    for w in wells:
        well_names = {str(lg.petrel_name) for lg in getattr(w, "logs", [])}
        if any(n in well_names for n in names_set):
            count += 1
    return count

# ------------------------------- UI -------------------------------------
st.title("Log Standardization & Mnemonics Library")

# ---- SAFE INIT ----
if "mnemonics_lib" not in st.session_state:
    lib, lib_path, lib_mode = load_mnemonics()
    st.session_state.mnemonics_lib = lib
    st.session_state._mnemonic_loaded_from = (lib_path, lib_mode)

if "std_mapping" not in st.session_state:
    st.session_state.std_mapping = load_latest_stamped_for("log_mapping") or {
        "targets": {}, "selection_mode": {}, "preferred_log": {}
    }

mnems = st.session_state.mnemonics_lib
mapping = st.session_state.std_mapping
lib_path, lib_mode = st.session_state.get("_mnemonic_loaded_from", (Path("[defaults]"), "Using defaults"))

with st.toast(f"{lib_mode}", icon="📚"):
    src = f"{lib_path}" if isinstance(lib_path, (str, Path)) else str(lib_path)
    st.caption(f"Library source: {src}")

# Discover logs from Petrel
wells = petrel_project.wells
all_log_names = get_unique_project_logs(wells)
families, unmatched = group_logs_by_family(all_log_names, mnems)

# --------------------------- Overview Row -------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Wells", len(wells))
c2.metric("Unique log names", len(all_log_names))
c3.metric("Matched to families", sum(len(v) for v in families.values()))
c4.metric("Unmatched", len(unmatched))

with st.expander("Show group summary (counts & names)", expanded=False):
    rows = []
    for fam, names in families.items():
        rows.append({
            "Group": fam,
            "Curve Mnemonic / Output Name": mnems[fam].get("canonical", fam),  # <<< CHANGED
            "Unique names (#)": len(names),
            "Names": ", ".join(sorted(names)) if names else ""
        })
    df_summary = pd.DataFrame(rows).sort_values(["Group"]).reset_index(drop=True)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)

    # Persist optional JSON so other pages can read the same table
    subc1, subc2, subc3, subc4 = st.columns([1.1, 1.4, 1.2, 1.2])
    with subc1:
        if st.button("Save summary for other pages"):
            try:
                SUMMARY_JSON_STAMPED.write_text(
                    df_summary.to_json(orient="records", force_ascii=False),
                    encoding="utf-8"
                )
                st.success(f"Saved: {SUMMARY_JSON_STAMPED}")

            except Exception as e:
                st.error(f"Could not save summary: {e}")

    # --- HTML / PDF downloads of the summary table ---
    def _df_to_html_table(df: pd.DataFrame) -> str:
        try:
            return df.to_html(index=False, border=0, classes="table", justify="center")
        except Exception:
            return "<p><i>Table could not be rendered.</i></p>"

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    html_doc = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<style>body{font-family:Arial,Helvetica,sans-serif;margin:24px;}"
        "h1{margin-bottom:8px;} table{border-collapse:collapse;width:100%;}"
        "th,td{border:1px solid #ddd;padding:6px;font-size:12px;} th{background:#f5f5f5;}</style></head><body>"
        f"<h1>Project Log Group Summary</h1><div>Generated: {now_str}</div>"
        + _df_to_html_table(df_summary) +
        "</body></html>"
    ).encode("utf-8")

    with subc2:
        st.download_button(
            "Download Project Summary (HTML)",
            data=html_doc,
            file_name=f"project_log_groups_summary__{PROJECT_SLUG}__{USERNAME}__{DATE_STAMP}.html",
            mime="text/html"
        )

    def _df_to_table_pdf(df: pd.DataFrame):
        df_reset = df.copy()
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

    with subc3:
        try:
            buf = BytesIO()
            doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
            styles = getSampleStyleSheet()
            story = [
                Paragraph("Project Log Group Summary", styles["Title"]),
                Paragraph(f"Generated: {now_str}", styles["Normal"]),
                Spacer(1, 12),
                _df_to_table_pdf(df_summary)
            ]
            doc.build(story)
            pdf_summary_bytes = buf.getvalue()
            st.download_button(
                "Download Project Summary (PDF)",
                data=pdf_summary_bytes,
                file_name=f"project_log_groups_summary__{PROJECT_SLUG}__{USERNAME}__{DATE_STAMP}.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.info(f"PDF export not available: {e}")

    with subc4:
        st.download_button(
            label="Download Project Summary (CSV)",
            data=df_summary.to_csv(index=False).encode("utf-8"),
            file_name=f"project_log_groups_summary__{PROJECT_SLUG}__{USERNAME}__{DATE_STAMP}.csv",
            mime="text/csv"
        )

# ---------------------- Library-level actions (top) ----------------------
# ---------------------- Library-level actions (top) ----------------------
st.subheader("Library actions")

# Build CSV payload once for the buttons (aliases flattened)
lib_rows = []
for fam, meta in mnems.items():
    canonical = meta.get("canonical", fam)
    style = meta.get("style", {})
    units = ", ".join(meta.get("units", []))
    aliases = sorted(set(meta.get("aliases", [])))
    if aliases:
        for a in aliases:
            lib_rows.append({
                "Group": fam,
                "Canonical": canonical,
                "Alias": a,
                "Color (hex)": style.get("color", ""),
                "Line style": style.get("line_style", ""),
                "Scale (min->max)": str(style.get("scale", "")),
                "Units": units,
                "Description": meta.get("description", "")
            })
    else:
        lib_rows.append({
            "Group": fam,
            "Canonical": canonical,
            "Alias": "",
            "Color (hex)": style.get("color", ""),
            "Line style": style.get("line_style", ""),
            "Scale (min->max)": str(style.get("scale", "")),
            "Units": units,
            "Description": meta.get("description", "")
        })
df_lib = pd.DataFrame(lib_rows) if lib_rows else pd.DataFrame(columns=[
    "Group", "Canonical", "Alias", "Color (hex)", "Line style", "Scale (min->max)", "Units", "Description"
])

left_actions, right_actions = st.columns([1.6, 2.4])

# LEFT: only the uploader + caption
with left_actions:
    up = st.file_uploader("Reload from user file (JSON)", type=["json"], accept_multiple_files=False)
    st.caption(
        "Upload a **mnemonics library** JSON (e.g. "
        "`mnemonics_user_library__<project>__<user>__YYYYMMDD.json` or the shared "
        "master `mnemonics_master.json`).\n\n"
        "**Do not upload** mapping files like `log_mapping__...`, `log_name_map__...`, etc."
    )
    if up is not None:
        try:
            data = json.loads(up.read().decode("utf-8"))
            if isinstance(data, dict) and data:
                st.session_state.mnemonics_lib = data
                st.session_state._mnemonic_loaded_from = ("[uploaded]", "Loaded from user file")
                st.success("Library loaded from uploaded file.")
                st.rerun()
            else:
                st.error("Uploaded JSON is empty or invalid.")
        except Exception as e:
            st.error(f"Could not read uploaded JSON: {e}")

# RIGHT: stack the rest vertically
with right_actions:
    colA, colB = st.columns(2)
    with colA:
        if st.button("Reset to (Master) defaults", use_container_width=True):
            if NETWORK_MNEM_PATH.exists():
                data = _safe_json_load(NETWORK_MNEM_PATH) or default_mnemonics_library()
                st.session_state.mnemonics_lib = data
                st.session_state._mnemonic_loaded_from = (NETWORK_MNEM_PATH, "Loaded shared master (read-only)")
            else:
                st.session_state.mnemonics_lib = default_mnemonics_library()
                st.session_state._mnemonic_loaded_from = (Path("[defaults]"), "Using defaults (not yet saved)")
            st.rerun()
    with colB:
        if st.button("Save library to disk", use_container_width=True):
            msgs = save_mnemonics(st.session_state.mnemonics_lib)
            for m in msgs:
                st.info(m)

    st.download_button(
        label="Download Master Library (CSV)",
        data=df_lib.to_csv(index=False).encode("utf-8"),
        file_name=f"mnemonics_master__{PROJECT_SLUG}__{USERNAME}__{DATE_STAMP}.csv",
        mime="text/csv",
        use_container_width=True
    )
    st.download_button(
        label="Download Master Library (JSON)",
        data=json.dumps(mnems, indent=2, ensure_ascii=False).encode("utf-8"),
        file_name=f"mnemonics_master__{PROJECT_SLUG}__{USERNAME}__{DATE_STAMP}.json",
        mime="application/json",
        use_container_width=True
    )
    st.markdown(
        "<div style='font-size:12px;color:#555;line-height:1.3; margin-top:.5rem;'>"
        "<b>What this section does:</b> Upload a custom library, reset to seeded defaults or the shared master "
        "(read-only), save your current library to your local user folder, or download it as CSV/JSON.</div>",
        unsafe_allow_html=True
    )


# --------------- Add or remove families (optional convenience) ----------
# --------------- Add or remove families (optional convenience) ----------
with st.expander("➕ Add a new family (optional)"):
    with st.form("new_family_form", clear_on_submit=True):
        new_fam = st.text_input("New family name", key="new_family_name")
        create_family = st.form_submit_button("Create family")

    if create_family:
        nf = (new_fam or "").strip()
        if nf and nf not in mnems:
            mnems[nf] = {
                "canonical": nf,
                "aliases": [],
                "style": {"color": "#000000", "line_style": "solid", "scale": [None, None]},
                "units": [],
                "description": ""
            }
            # re-group immediately so the mid/right panels reflect the new library
            families, unmatched = group_logs_by_family(all_log_names, mnems)
            st.success(f"Added '{nf}'. Edit its details below and Save library to persist.")
        else:
            st.warning("Empty name or already exists.")

# ----------------------- Per-family configuration ------------------------
st.subheader("Families (edit standards, review matches, choose standardized name)")

for fam in sorted(mnems.keys()):
    meta = mnems[fam]
    fam_key = re.sub(r"\W+", "_", fam.lower())
    detected = sorted(families.get(fam, set()))
    wells_in_group = wells_containing_any_of(families.get(fam, set()), wells)

    with st.expander(f"📁 {fam} — {len(detected)} matched names • {wells_in_group} wells", expanded=False):
        col_left, col_mid, col_right = st.columns([1.25, 1.4, 1.15])

        # -------- Left: Master (edit library) --------
        with col_left:
            st.markdown("**Master (library) — canonical, aliases, style & metadata**")

            canonical_val = st.text_input(
                "Curve mnemonic / Output Name (e.g., GR_STD)",  # <<< CHANGED
                value=meta.get("canonical", fam),
                key=f"canon_{fam_key}"
            )

            aliases_csv = st.text_area(
                "Aliases (comma‑separated)",
                value=", ".join(sorted(set(meta.get("aliases", [])))) if meta.get("aliases") else "",
                height=80,
                key=f"aliases_{fam_key}"
            )

            style = meta.get("style", {})
            color_hex = st.color_picker("Color", value=style.get("color", "#000000"), key=f"color_{fam_key}")
            line_style = st.selectbox(
                "Line style",
                options=["solid", "dash", "dot", "long-dash"],
                index=["solid", "dash", "dot", "long-dash"].index(style.get("line_style", "solid")),
                key=f"linestyle_{fam_key}"
            )
            sc = style.get("scale", [None, None])
            sc_min = st.text_input("Scale min (e.g., 0.2)", value="" if sc[0] is None else str(sc[0]), key=f"scmin_{fam_key}")
            sc_max = st.text_input("Scale max (e.g., 2000)", value="" if sc[1] is None else str(sc[1]), key=f"scmax_{fam_key}")

            units_csv = st.text_input(
                "Units (comma‑separated)",
                value=", ".join(meta.get("units", [])),
                key=f"units_{fam_key}"
            )
            descr = st.text_area(
                "Description",
                value=meta.get("description", ""),
                height=70,
                key=f"desc_{fam_key}"
            )

            if st.button("Save group edits", key=f"save_{fam_key}"):
                new_aliases = [a.strip() for a in aliases_csv.split(",") if a.strip()]

                def _parse(x):
                    try:
                        return float(x)
                    except Exception:
                        return None
                new_sc = [_parse(sc_min), _parse(sc_max)]

                mnems[fam]["canonical"] = canonical_val.strip() or fam
                mnems[fam]["aliases"] = sorted(set(new_aliases))
                mnems[fam]["style"] = {"color": color_hex, "line_style": line_style, "scale": new_sc}
                mnems[fam]["units"] = [u.strip() for u in units_csv.split(",") if u.strip()]
                mnems[fam]["description"] = descr.strip()
                families, unmatched = group_logs_by_family(all_log_names, mnems)
                st.success(f"Updated library for '{fam}'. To persist, click **Save library to disk** above.")

            st.markdown("---")
            st.markdown("**Add unmatched names as aliases**")
            candidates = sorted(unmatched)
            to_add = st.multiselect(
                "Select from currently unmatched:",
                options=candidates,
                key=f"add_unmatched_{fam_key}"
            )
            if st.button("Add selected to aliases", key=f"add_btn_{fam_key}"):
                if to_add:
                    current = set(mnems[fam].get("aliases", []))
                    mnems[fam]["aliases"] = sorted(current | set(to_add))
                    families, unmatched = group_logs_by_family(all_log_names, mnems)
                    st.success(f"Added {len(to_add)} alias(es) to '{fam}'.")
                else:
                    st.info("No items selected.")

        # -------- Middle: Project names detected --------
        with col_mid:
            st.markdown("**Detected in project**")
            if detected:
                per_name_counts = Counter()
                for w in wells:
                    have = {str(lg.petrel_name) for lg in getattr(w, "logs", [])}
                    for n in detected:
                        if n in have:
                            per_name_counts[n] += 1
                df_detected = pd.DataFrame({
                    "Log name": detected,
                    "# of wells containing it": [per_name_counts[n] for n in detected]
                })
                st.dataframe(df_detected, use_container_width=True, hide_index=True, height=220)
            else:
                st.info("No current project log names match this family.")

        # -------- Right: Standardization choice (global per group) --------
        with col_right:
            st.markdown("**Standardization**")
            sel_mode = st.radio(
                "Selection mode",
                options=["One per group", "Multiple allowed"],
                index=0 if mapping["selection_mode"].get(fam, "One per group") == "One per group" else 1,
                key=f"selmode_{fam_key}",
                help="If *One per group*, only the preferred name will be used; others are flagged as excluded."
            )
            mapping["selection_mode"][fam] = sel_mode

            default_target = mapping["targets"].get(fam, meta.get("canonical", fam))
            target = st.text_input(
                "Standardized name to use",
                value=default_target,
                key=f"target_{fam_key}",
                help="Canonical label downstream (e.g., 'Gamma Ray')."
            )
            mapping["targets"][fam] = target.strip() or meta.get("canonical", fam)

            if sel_mode == "One per group":
                if detected:
                    preferred = mapping["preferred_log"].get(fam, detected[0])
                    idx = detected.index(preferred) if preferred in detected else 0
                    mapping["preferred_log"][fam] = st.selectbox(
                        "Preferred project log (if keeping one)",
                        options=detected,
                        index=idx,
                        key=f"pref_{fam_key}"
                    )
                else:
                    st.selectbox("Preferred project log (if keeping one)", options=[], key=f"pref_{fam_key}")
                    mapping["preferred_log"].pop(fam, None)
            else:
                mapping["preferred_log"].pop(fam, None)

            st.markdown("---")
            if st.button("Save choices (this group)", key=f"save_choice_{fam_key}"):
                # Save latest + stamped
                write_json(MAPPING_JSON_STAMPED, mapping)
                st.success(f"Saved: {MAPPING_JSON_STAMPED.name}")

# ------------------------- Unmatched names panel -------------------------
with st.expander(f"🔎 Unmatched log names ({len(unmatched)})", expanded=False):
    if unmatched:
        st.write(", ".join(sorted(unmatched)))
    else:
        st.success("All current log names are covered by your families & aliases.")

# ------------------------- Mapping preview -------------------------------
st.markdown("---")
st.subheader("Project Standardization Preview")

# Build a flat mapping table: each original name -> (group, standardized, use_in_project)
rows = []
fam_by_name = {}
for fam, names in families.items():
    for n in names:
        fam_by_name[n] = fam

for name in all_log_names:
    fam = fam_by_name.get(name)
    if fam is None:
        rows.append({
            "Original log name": name,
            "Group": "(unmatched)",
            "Standardized": name,
            "Use in project": False,                      # <<< CHANGED (was True)
            "Reason": "Unmatched — excluded by default"   # <<< CHANGED
        })
    else:
        target = mapping["targets"].get(fam, mnems[fam]["canonical"])
        selmode = mapping["selection_mode"].get(fam, "One per group")
        preferred = mapping["preferred_log"].get(fam)
        use_flag = True
        reason = "Included"
        if selmode == "One per group" and preferred and name != preferred:
            use_flag = False
            reason = f"Excluded (keeping '{preferred}')"
        rows.append({
            "Original log name": name,
            "Group": fam,
            "Standardized": target,
            "Use in project": use_flag,
            "Reason": reason
        })

df_map = pd.DataFrame(rows).sort_values(["Group", "Original log name"]).reset_index(drop=True)
st.dataframe(df_map, use_container_width=True, hide_index=True)

# Helper: write a simple {original->standardized} JSON (only for rows selected for use)
def _write_flat_name_map(df: pd.DataFrame, path: Path) -> dict:
    name_map = {}
    for _, r in df.iterrows():
        if bool(r.get("Use in project", True)):
            name_map[str(r["Original log name"])] = str(r["Standardized"])
    path.write_text(json.dumps(name_map, indent=2, ensure_ascii=False), encoding="utf-8")
    return name_map

# ---------------------- Save / Export (NEW expander) ---------------------
with st.expander("💾 Save mnemonics & export mapping", expanded=True):
    colA, colB = st.columns([1.4, 2.6])

    with colA:
        st.markdown("**Metadata**")
        project_name = st.text_input("Project name", value=PROJECT_NAME_RAW)
        user_name = st.text_input("User name", value=USERNAME)
        run_date = st.date_input("Date", value=date.today())
        notes = st.text_area("Notes (optional)", value="", height=60)

        st.caption(f"Files will be saved under:  {STD_DIR}")

        def build_name_map_from_df(df: pd.DataFrame) -> dict:
            nm = {}
            for _, r in df.iterrows():
                # keep rows marked "Use in project"
                if bool(r.get("Use in project", True)):
                    nm[str(r["Original log name"])] = str(r["Standardized"])
            return nm

        if st.button("Save ALL choices (stamped per user & project)"):
            try:
                saved_files = []

                # 1) Structured mapping
                write_json(MAPPING_JSON_STAMPED, mapping)
                saved_files.append(MAPPING_JSON_STAMPED)

                # 2) Records snapshot (flat table)
                mapping_records = json.loads(df_map.to_json(orient="records"))
                write_json(MAPPING_RECORDS_STAMPED, mapping_records)
                saved_files.append(MAPPING_RECORDS_STAMPED)

                # 3) Simple name-map
                name_map = build_name_map_from_df(df_map)
                write_json(NAME_MAP_JSON_STAMPED, name_map)
                saved_files.append(NAME_MAP_JSON_STAMPED)

                # 4) Package with metadata
                package = {
                    "meta": {
                        "project_name": PROJECT_NAME_RAW or "",
                        "user_name": USERNAME or "",
                        "date": date.today().isoformat(),
                        "notes": notes,
                        "generated": datetime.now().isoformat(timespec="seconds"),
                    },
                    "library_path": str(st.session_state.get("_mnemonic_loaded_from", ("", ""))[0]),
                    "mapping_structured": mapping,
                    "mapping_records": mapping_records,
                    "name_map": name_map,
                }
                write_json(PACKAGE_JSON_STAMPED, package)
                saved_files.append(PACKAGE_JSON_STAMPED)

                st.success("Saved files:\n" + "\n".join(f"- {p}" for p in saved_files))
            
                with st.expander("What did we just save?", expanded=False):
                    st.markdown(f"**Folder:** `{STD_DIR}`")
                    st.markdown("""
                        - **log_mapping__...json** — Structured selections per family (`targets`, `selection_mode`, `preferred_log`).  
                        *Reopen/continue editing later.*

                        - **log_mapping_records__...json** — Flat snapshot of the mapping preview (one row per original name).  
                        *Audit/QA/sharing.*

                        - **log_name_map__...json** — `{original_name: standardized_label}` for rows **used in project**.  
                        *Other pages load this to show standardized names.*

                        - **mnemonics_mapping_package__...json** — Bundle of metadata + the three files above (+ library path).  
                        *Archival/portability/traceability.*

                        > If you used **Save library to disk**, your **mnemonics_user_library__...json** is your personal library copy.  
                        > Use it in **Reload from user file (JSON)** to restore your custom family/alias/style definitions.
                            """)

            
            except Exception as e:
                st.error(f"Could not save: {e}")


        st.markdown("---")
        st.markdown("**Quick export: simple name-map**")
        if st.button("Export simple name-map JSON (stamped)"):
            try:
                name_map = build_name_map_from_df(df_map)
                write_json(NAME_MAP_JSON_STAMPED, name_map)
                out = NAME_MAP_JSON_STAMPED
                st.success(f"Saved: {out}  ({len(name_map)} names)")
            except Exception as e:
                st.error(f"Could not save name-map: {e}")

    with colB:
        st.markdown("**Downloads (Mapping Preview Table)**")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.download_button(
                label="Download mapping (CSV)",
                data=df_map.to_csv(index=False).encode("utf-8"),
                file_name=f"log_standardization_mapping__{PROJECT_SLUG}__{USERNAME}__{DATE_STAMP}.csv",
                mime="text/csv"
            )
        with c2:
            st.download_button(
                label="Download mapping (JSON)",
                data=df_map.to_json(orient="records", indent=2).encode("utf-8"),
                file_name=f"log_standardization_mapping__{PROJECT_SLUG}__{USERNAME}__{DATE_STAMP}.json",
                mime="application/json"
            )

        # HTML / PDF of df_map
        def _df_to_html(df: pd.DataFrame) -> str:
            try:
                return df.to_html(index=False, border=0, classes="table", justify="center")
            except Exception:
                return "<p><i>Table could not be rendered.</i></p>"

        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        html_parts = [
            "<!DOCTYPE html><html><head><meta charset='utf-8'>",
            "<style>",
            "body{font-family:Arial,Helvetica,sans-serif; margin:24px;}",
            "h1{margin-bottom:4px;} .muted{color:#555;}",
            "table{border-collapse:collapse; width:100%; margin:10px 0;}",
            "th,td{border:1px solid #ddd; padding:6px; font-size:12px;}",
            "th{background:#f5f5f5;}",
            "</style></head><body>",
            f"<h1>Log Standardization Mapping</h1>",
            f"<div class='muted'>Generated: {now}</div>",
            _df_to_html(df_map),
            "</body></html>",
        ]
        export_html = "".join(html_parts).encode("utf-8")

        with c3:
            st.download_button(
                "Download mapping (HTML)",
                data=export_html,
                file_name=f"log_standardization_mapping__{PROJECT_SLUG}__{USERNAME}__{DATE_STAMP}.html",
                mime="text/html"
            )

        def _df_to_table(df: pd.DataFrame):
            df_reset = df.copy()
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

        with c4:
            try:
                buf = BytesIO()
                doc = SimpleDocTemplate(
                    buf, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
                )
                styles = getSampleStyleSheet()
                story = [
                    Paragraph("Log Standardization Mapping", styles["Title"]),
                    Paragraph(f"Generated: {now}", styles["Normal"]),
                    Paragraph(f"Project: {project_name or '-'} | User: {user_name or '-'} | Date: {run_date.isoformat()}", styles["Normal"]),
                    Spacer(1, 12),
                    _df_to_table(df_map)
                ]
                doc.build(story)
                pdf_bytes = buf.getvalue()
                st.download_button(
                    "Download mapping (PDF)",
                    data=pdf_bytes,
                    file_name=f"log_standardization_mapping__{PROJECT_SLUG}__{USERNAME}__{DATE_STAMP}.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.info(f"PDF export not available: {e}")
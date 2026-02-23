import numpy as np
import missingno as msno
import pandas as pd
import streamlit as st
from cegalprizm.pythontool import PetrelConnection
from cegalprizm.pythontool.exceptions import PythonToolException
from types import SimpleNamespace
import re 
import math
import constants
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from types import SimpleNamespace

# ---- Single source of truth for pages ----
from pathlib import Path

APP_ROOT  = Path(__file__).resolve().parent
PAGES_DIR = APP_ROOT / "pages"

def _page_relpath(spec: dict) -> str:
    """Return a RELATIVE path (as Streamlit expects), always inside /pages."""
    return f"pages/{spec['file']}"

# Controls expander order in the custom sidebar
PAGE_GROUP_ORDER = ["QC", "Basic", "Auto EDA", "Charts", "Documentation"]

# One row per page. Remove is_root=True for Overview; it must live in /pages.
PAGE_REGISTRY = [
    {"file": "Project Summary.py",     "title": "Overview",              "group": None,           "default": True},

    # QC
    {"file": "0_Log Standarization.py","title": "Log Standardization",   "group": "QC"},

    # Basic
    {"file": "3_Wells.py",             "title": "Wells",                 "group": "Basic"},
    {"file": "4_Global Logs.py",       "title": "Global Logs",           "group": "Basic"},
    {"file": "7_Well Tops.py",         "title": "Well Tops",             "group": "Basic"},
    # {"file": "10_Checkshot QC.py",   "title": "Check Shots",           "group": "Basic"},

    # Documentation / Auto EDA
    {"file": "1_Automatic Well Logs EDA.py",         "title": "General",            "group": "Auto EDA"},
    {"file": "1_Automatic Seismic Inversion EDA.py", "title": "Seismic Inversion",  "group": "Auto EDA"},
    {"file": "1_Auto Rock Physics EDA.py",           "title": "Rock Physics",       "group": "Auto EDA"},
    {"file": "1_Auto Velocity Model EDA.py",         "title": "Velocity Model",     "group": "Auto EDA"},

    # Charts
    {"file": "2_Gantt Chart.py",       "title": "Gantt Chart",           "group": "Charts"},

    # Documentation
    {"file": "11_Documentation.py",    "title": "PDF Viewer",            "group": "Documentation"},
]


# Read the PWR selections
# --- selection bridge (always read fresh) ---
def _load_pwr_selections() -> dict:
    path = os.environ.get("PRIZM_PWR_SELECTIONS_FILE", "")
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f) or {}
    except Exception:
        return {}

def scope_mode() -> int:
    sels = _load_pwr_selections()
    try:
        return int(sels.get("scope_mode", 1))
    except Exception:
        return 1

def selected_well_guids() -> set[str]:
    return set(_load_pwr_selections().get("well_guids", []))

def selected_log_guids() -> set[str]:
    return set(_load_pwr_selections().get("log_guids", []))

def selected_global_log_guids() -> set[str]:
    return set(_load_pwr_selections().get("global_log_guids", []))

@st.cache_resource
def _petrel_conn(): return PetrelConnection()

@st.cache_data(show_spinner=False)
def selected_global_log_names(_cache_key: str) -> set[str]:
    gids = list(selected_global_log_guids())
    if not gids:
        return set()
    try:
        objs = _petrel_conn().get_petrelobjects_by_guids(gids)
    except Exception:
        objs = []
    return {getattr(o, "petrel_name", "") for o in objs if getattr(o, "petrel_name", "")}

def get_all_wells_flat(petrel):
    """
    Prefer the selected wells (resolved by GUIDs) when scope = Selected only.
    Otherwise fall back to discovering all wells from the project.
    """
    try:
        if scope_mode() == 1:
            guids = list(selected_well_guids())
            if not guids:
                return []  # strict: no selection -> no wells
            try:
                wells = _petrel_conn().get_petrelobjects_by_guids(guids)
                wells = [w for w in wells if hasattr(w, "petrel_name")]
                if wells:
                    return wells
            except Exception:
                return []  # strict: on error with selections, don’t fall back to all

    except Exception:
        pass

    # Default discovery (as before)
    try:
        wf = petrel.well_folders["Input/Wells"]
        return wf.get_wells(recursive=True)
    except Exception:
        pass
    try:
        return flatten_wells_like(petrel.wells)
    except Exception:
        return []

def _selected_log_paths() -> set[str]:
    """
    Return the set of per‑well continuous log paths selected via PWR (if any).
    With your current launcher this will usually be empty, but we keep it
    so the code also works if you later expose per‑well log selection again.
    """
    try:
        objs = _resolve_selected_objects(_sel_cache_key())
        return {
            getattr(lg, "path", "")
            for lg in (objs.get("logs") or [])
            if getattr(lg, "path", "")
        }
    except Exception:
        return set()

def iter_selected_logs(well):
    """
    Return logs for a well, respecting selections.

    STRICT mode for scope == 1 (Selected objects only):
      - If no global log names AND no per-well logs were selected -> return []
      - Otherwise filter by those selections.

    For scope == 2 (Whole project): return all logs.
    """
    logs = list(getattr(well, "logs", [])) or []

    # Whole project -> permissive
    if scope_mode() != 1:
        return logs

    # Selected objects only -> strict
    name_allow = selected_global_log_names(_sel_cache_key())  # from global_log_guids (by name)
    path_allow = _selected_log_paths()                        # from log_guids (by path; unused now)

    # STRICT: nothing selected -> no logs
    if not name_allow and not path_allow:
        return []

    # Filter by selections
    out = []
    for lg in logs:
        nm = getattr(lg, "petrel_name", "")
        p  = getattr(lg, "path", "")
        if (name_allow and nm in name_allow) or (path_allow and p in path_allow):
            out.append(lg)
    return out

def _sel_cache_key() -> str:
    """Changes when the selections file changes, so Streamlit cache invalidates."""
    path = os.environ.get("PRIZM_PWR_SELECTIONS_FILE", "")
    try:
        s = os.stat(path)
        return f"{path}:{s.st_mtime_ns}:{s.st_size}"
    except Exception:
        return path

@st.cache_data(show_spinner=False)
def _resolve_selected_objects(_cache_key: str) -> dict:
    """
    Resolve selected GUIDs into Petrel objects once per *selections file* revision.
    Returns dict with keys: wells, logs, surveys, marker_collections, seismic3d, seismic2d
    """
    sels = _load_pwr_selections()
    ptp = _petrel_conn()
    out = {k: [] for k in ["wells", "logs", "surveys", "marker_collections", "seismic3d", "seismic2d"]}

    def _fetch(key_in, key_out):
        guids = list(map(str, sels.get(key_in, []) or []))
        if not guids:
            return
        try:
            objs = ptp.get_petrelobjects_by_guids(guids)
            out[key_out] = [o for o in objs if o is not None]
        except Exception:
            out[key_out] = []

    _fetch("well_guids", "wells")
    _fetch("log_guids", "logs")
    _fetch("survey_guids", "surveys")
    _fetch("marker_collection_guids", "marker_collections")
    _fetch("seismic3d_guids", "seismic3d")
    _fetch("seismic2d_guids", "seismic2d")
    return out

@st.cache_data(show_spinner=False)
def load_tops_dataframe(_petrel_project, _cache_key: str):
    """
    Strict in scope == 1:
      - If one or more marker collections were selected -> concatenate & return them
      - If none selected -> return an empty tops DataFrame (no fallback)

    Permissive in scope == 2:
      - Fall back to the first marker collection in the project (as before)
    """
    EMPTY_TOPS = pd.DataFrame(columns=["Well identifier (Well name)", "Surface", "MD"])

    # --- STRICT: Selected objects only ---
    if scope_mode() == 1:
        # Resolve selected marker collections from the selections file
        try:
            objs = _resolve_selected_objects(_cache_key)
        except Exception:
            # If selections cannot be resolved, act strict: return empty
            return EMPTY_TOPS

        mcs = objs.get("marker_collections") or []
        if not mcs:
            # Nothing was selected -> return empty (no fallback)
            return EMPTY_TOPS

        frames = []
        for mc in mcs:
            try:
                frames.append(mc.as_dataframe(include_unconnected_markers=False))
            except Exception:
                # Skip any that error
                pass

        return pd.concat(frames, ignore_index=True) if frames else EMPTY_TOPS

    # --- PERMISSIVE: Whole project (ignore selections) ---
    try:
        mc = [i for i in _petrel_project.markercollections][0]
        return mc.as_dataframe(include_unconnected_markers=False)
    except Exception:
        return EMPTY_TOPS


def _page_path(spec: dict) -> Path:
    """Resolve a registry item to its absolute path."""
    return (APP_ROOT / spec["file"]) if spec.get("is_root") else (PAGES_DIR / spec["file"])

def hide_native_sidebar_pages():
    """Hide Streamlit’s built-in pages list (we render our own grouped menu)."""
    st.markdown("""
        <style>
        section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] { display: none; }
        </style>
    """, unsafe_allow_html=True)

def build_streamlit_navigation(position="sidebar", expanded=False):
    """Build the st.navigation() object from PAGE_REGISTRY."""
    pages = []
    for spec in PAGE_REGISTRY:
        pages.append(
            st.Page(
                _page_relpath(spec),       # << relative path (pages/...)
                title=spec["title"],
                default=spec.get("default", False)
            )
        )
    return st.navigation(pages, position=position, expanded=expanded)

LOGO_PATH = constants.LOGO_PATH
APP_VERSION = constants.APP_VERSION
TIMELINE_FILE = constants.TIMELINE_FILE

# utils.py (add this helper near the top)
def try_get_project_name(_petrel_project) -> str:
    """Return project name or a safe fallback if hub auth is unavailable."""
    try:
        return _petrel_project.get_current_project_name()
    except Exception as e:
        # cache the message so pages can show a small notice instead of crashing
        st.session_state["petrel_auth_error"] = f"{e}"
        return "(not connected)"

def sidebar_footer(_petrel_project, logo_path: str, app_version: str):
    project_name = try_get_project_name(_petrel_project)   # <-- use safe getter
    with open(logo_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    st.sidebar.markdown(f"""
        <div class="sidebar-footer">
            <img src="data:image/png;base64,{img_b64}" alt="Logo" />
            <div class="meta"><strong>Connected to project:</strong> {project_name}</div>
            <div class="meta"><strong>Version:</strong> {app_version}</div>
        </div>
    """, unsafe_allow_html=True)

    # Optional small warning (non-blocking) if auth failed
    if "petrel_auth_error" in st.session_state:
        st.sidebar.caption("⚠️ Petrel connection: " + st.session_state["petrel_auth_error"])

    
@st.cache_resource(show_spinner=False)
def get_petrel_connection():
    try:
        return PetrelConnection()
    except PythonToolException as e:
        # Record the message for the footer and return None to avoid crashing pages
        st.session_state["petrel_auth_error"] = str(e)
        return None
    except Exception as e:
        st.session_state["petrel_auth_error"] = f"Unexpected error: {e}"
        return None
    
@st.cache_data(show_spinner=False)
def get_all_well_data(_project):
    """
    Returns: (wells_summary_df, wells_full_df, stat_keys)
    Robust to being called with either a Petrel project or a (possibly nested) list of wells.
    Streamlit won’t hash '_project', so caching is still enabled for the rest of the args/body.
    """

    # --- normalize input to a flat list of Well-like objects ---
    wells = getattr(_project, "wells", _project)
    if wells is None:
        wells = []
    elif not isinstance(wells, (list, tuple)):
        wells = list(wells)

    flat = []
    for obj in wells:
        if isinstance(obj, (list, tuple)):
            flat.extend(obj)
        else:
            flat.append(obj)

    def _is_well_like(x):
        return hasattr(x, "retrieve_stats") and hasattr(x, "petrel_name")

    wells = [w for w in flat if _is_well_like(w)]

    if not wells:
        bad_types = [type(x).__name__ for x in flat][:5]
        raise TypeError(
            "get_all_well_data(): did not find any Well-like objects. "
            f"Got sample types: {bad_types}. "
            "Pass a project (with .wells) or a list of Well objects."
        )

    # --- build rows from per-well stats ---
        # --- build rows from per-well stats ---
    rows = []
    stat_keys = set()

    for w in wells:
        try:
            stats = map_wellstats(w.retrieve_stats())
        except Exception:
            stats = {}

        # Convert SimpleNamespace -> dict (or accept dicts if you ever return them)
        if isinstance(stats, SimpleNamespace):
            stats_dict = vars(stats)
        elif isinstance(stats, dict):
            stats_dict = stats
        else:
            # last-resort defensive cast
            try:
                stats_dict = dict(stats)
            except Exception:
                stats_dict = {}

        stat_keys.update(stats_dict.keys())
        rows.append({"Well": w.petrel_name, "Well Name": w.petrel_name, **stats_dict})

    wells_summary_df = pd.DataFrame(rows)
    wells_full_df = wells_summary_df.copy()
    return wells_summary_df, wells_full_df, sorted(stat_keys)

def global_md_range(_wells_dict):
    md_lo = np.inf
    md_hi = -np.inf
    for obj in _wells_dict.values():
        for w in (obj if isinstance(obj, list) else [obj]):
            for lg in w.logs:
                md = lg.as_dataframe()["MD"]
                mmin, mmax = md.min(), md.max()
                if not np.isnan(mmin):
                    md_lo = min(md_lo, mmin)
                    md_hi = max(md_hi, mmax)
    return (None, None) if np.isinf(md_lo) else (md_lo, md_hi)

def retrieve_well(ptp_object, pattern):
    wells = ptp_object.wells
    matching_wells = [well for well in wells if well.path and re.search(pattern, well.path)]

    if not matching_wells:
        st.error(f"No well found matching pattern: '{pattern}'")
        return None

    return matching_wells[0]

def map_wellstats(stats: dict) -> SimpleNamespace:
    mapped = {}
    for k, v in stats.items():
        attr = ''.join(word.title() for word in k.split())
        try:
            mapped[attr] = int(v)
        except (ValueError, TypeError):
            try:
                mapped[attr] = float(v)
            except (ValueError, TypeError):
                mapped[attr] = v
    return SimpleNamespace(**mapped)

def get_well_xy(_wells, _cache_key: str | None = None) -> pd.DataFrame:
    """
    Return one row per well with:
      - 'Well Name'
      - 'X' (Wellhead X-coord. or X Min)
      - 'Y' (Wellhead Y-coord. or Y Min)
    """
    rows = []

    for w in _wells:
        try:
            raw_stats = w.retrieve_stats() or {}
        except Exception:
            raw_stats = {}

        # Normalize stats -> dict
        if isinstance(raw_stats, dict):
            stats = raw_stats
        elif isinstance(raw_stats, SimpleNamespace):
            stats = vars(raw_stats)
        else:
            try:
                stats = dict(raw_stats)
            except Exception:
                stats = {}

        def _to_float(v):
            try:
                return float(str(v).replace(",", "."))
            except Exception:
                return np.nan

        x_val = np.nan
        y_val = np.nan

        if isinstance(stats, dict) and stats:
            x_raw = stats.get("Wellhead X-coord.", stats.get("X Min", None))
            y_raw = stats.get("Wellhead Y-coord.", stats.get("Y Min", None))

            x_val = _to_float(x_raw)
            y_val = _to_float(y_raw)

        rows.append(
            {
                "Well Name": getattr(w, "petrel_name", ""),
                "X": x_val,
                "Y": y_val,
            }
        )

    return pd.DataFrame(rows, columns=["Well Name", "X", "Y"])


# def get_well_min_lat_long(_wells, _cache_key: str | None = None) -> pd.DataFrame:
#     """
#     Very simple: use Wellhead X/Y (or X/Y Min) as coordinates.

#     Returns one row per well with:
#       - 'Well Name'
#       - 'latitude'  (from Y)
#       - 'longitude' (from X)
#     """
#     rows = []

#     for w in _wells:
#         # Defaults
#         x_val = np.nan
#         y_val = np.nan

#         # Try to get stats
#         try:
#             raw_stats = w.retrieve_stats() or {}
#         except Exception:
#             raw_stats = {}

#         # Normalize stats -> plain dict
#         if isinstance(raw_stats, dict):
#             stats = raw_stats
#         elif isinstance(raw_stats, SimpleNamespace):
#             stats = vars(raw_stats)
#         else:
#             try:
#                 stats = dict(raw_stats)
#             except Exception:
#                 stats = {}

#         def _to_float(v):
#             try:
#                 return float(str(v).replace(",", "."))
#             except Exception:
#                 return np.nan

#         if isinstance(stats, dict) and stats:
#             # X first: Wellhead X-coord. or X Min
#             x_raw = _find_stat_value(
#                 stats,
#                 [
#                     r"^Wellhead X-coord\.?$",
#                     r"^X Min$",
#                 ],
#             )
#             # Y first: Wellhead Y-coord. or Y Min
#             y_raw = _find_stat_value(
#                 stats,
#                 [
#                     r"^Wellhead Y-coord\.?$",
#                     r"^Y Min$",
#                 ],
#             )

#             x_val = _to_float(x_raw)
#             y_val = _to_float(y_raw)

#         rows.append(
#             {
#                 "Well Name": getattr(w, "petrel_name", ""),
#                 "latitude": y_val,
#                 "longitude": x_val,
#             }
#         )

#     return pd.DataFrame(rows, columns=["Well Name", "latitude", "longitude"])

def get_well_min_lat_long(_wells, _cache_key: str | None = None) -> pd.DataFrame:
    """
    Approximate conversion from projected X/Y (meters) to lat/lon (degrees),
    using a simple linearization around a reference point.

    Assumptions:
      - X/Y are in meters in some projected CRS (e.g. UTM).
      - We choose a reference point (CRS_X0, CRS_Y0) and a target lat/lon
        (REF_LAT, REF_LON). You can tweak these constants to better fit your project.

    Returns one row per well with:
      - 'Well Name'
      - 'latitude'
      - 'longitude'
    """

    # ---- TUNABLE CONSTANTS ----
    # Approximate geographic location of your field / project
    REF_LAT = 60.0   # degrees
    REF_LON = 10.0   # degrees

    # Approximate X/Y (in meters) at that reference location
    # Adjust these so wells appear in the right place on the map
    CRS_X0 = 600000.0
    CRS_Y0 = 6700000.0
    # ---------------------------

    rows = []

    for w in _wells:
        try:
            raw_stats = w.retrieve_stats() or {}
        except Exception:
            raw_stats = {}

        # Normalize stats -> dict
        if isinstance(raw_stats, dict):
            stats = raw_stats
        elif isinstance(raw_stats, SimpleNamespace):
            stats = vars(raw_stats)
        else:
            try:
                stats = dict(raw_stats)
            except Exception:
                stats = {}

        def _to_float(v):
            try:
                return float(str(v).replace(",", "."))
            except Exception:
                return None

        # Prefer Wellhead X/Y, fallback to X/Y Min if needed
        x_m = _to_float(stats.get("Wellhead X-coord.", stats.get("X Min", None)))
        y_m = _to_float(stats.get("Wellhead Y-coord.", stats.get("Y Min", None)))

        lat = None
        lon = None

        if x_m is not None and y_m is not None:
            # Offset from reference point in meters
            dx = x_m - CRS_X0
            dy = y_m - CRS_Y0

            # Approximate conversion: 1 degree lat ≈ 111 km
            meters_per_deg_lat = 111_000.0
            # 1 degree lon ≈ 111 km * cos(latitude)
            meters_per_deg_lon = meters_per_deg_lat * math.cos(math.radians(REF_LAT))

            dlat = dy / meters_per_deg_lat
            dlon = dx / meters_per_deg_lon

            lat = REF_LAT + dlat
            lon = REF_LON + dlon

        rows.append(
            {
                "Well Name": getattr(w, "petrel_name", ""),
                "latitude": lat if lat is not None else np.nan,
                "longitude": lon if lon is not None else np.nan,
            }
        )

    return pd.DataFrame(rows, columns=["Well Name", "latitude", "longitude"])


def map_wellstats(stats: dict) -> SimpleNamespace:
    mapped = {}
    for k, v in stats.items():
        attr = ''.join(word.title() for word in k.split())
        try:
            mapped[attr] = int(v)
        except (ValueError, TypeError):
            try:
                mapped[attr] = float(v)
            except (ValueError, TypeError):
                mapped[attr] = v
    return SimpleNamespace(**mapped)

def _parse_lat_lon_value(value, *, is_lon: bool) -> float | None:
    """
    Try to coerce a latitude/longitude value to float.
    Accepts plain floats/ints or strings with N/S/E/W, degrees symbol, etc.
    Returns None if it can't parse or the value is out of valid range.
    """
    if value is None:
        return None

    # Already numeric
    if isinstance(value, (int, float)):
        x = float(value)
    else:
        s = str(value).strip()
        orig = s  # keep original for N/S/E/W detection

        # Remove degree/min/sec symbols and whitespace
        s = re.sub(r"[°'\"\s]", " ", s)
        # Remove N/S/E/W letters for numeric parse
        s = re.sub(r"[NnSsEeWw]", "", s)
        # Normalize comma decimal
        s = s.replace(",", ".").strip()

        try:
            # Remove any remaining spaces like "13 . 5"
            s = re.sub(r"\s+", "", s)
            x = float(s)
        except Exception:
            return None

        # Apply sign based on hemisphere letters in original string
        if is_lon and re.search(r"[Ww]", orig):
            x = -abs(x)
        if not is_lon and re.search(r"[Ss]", orig):
            x = -abs(x)

    # Validate geographic ranges
    if is_lon and not (-180.0 <= x <= 180.0):
        return None
    if not is_lon and not (-90.0 <= x <= 90.0):
        return None
    return x


def _find_stat_value(stats: dict, regexes: list[str]) -> object | None:
    """
    From a dict of stats, find the first value whose key matches any regex (case-insensitive).
    """
    if not isinstance(stats, dict):
        return None
    for rx in regexes:
        pat = re.compile(rx, flags=re.IGNORECASE)
        for k, v in stats.items():
            if isinstance(k, str) and pat.search(k):
                return v
    return None


@st.cache_data
def extract_stat_keys_from_wells(_wells):
    all_stat_keys = set()
    for well in _wells:
        stats = map_wellstats(well.retrieve_stats())
        all_stat_keys.update(vars(stats).keys())
    return sorted(all_stat_keys)

@st.cache_data
def build_filtered_well_dataframe(wells, selected_wells, selected_columns):
    filtered_rows = []
    for well in wells:
        if well.petrel_name not in selected_wells:
            continue
        stats = map_wellstats(well.retrieve_stats())
        row = {key: getattr(stats, key, None) for key in selected_columns if key in vars(stats)}
        if "KB (m)" in selected_columns:
            row["KB (m)"] = well.well_datum[1]
        row["Well Name"] = well.petrel_name
        filtered_rows.append(row)
    df_filtered = pd.DataFrame(filtered_rows)

    # Reorder columns
    cols_order = ["Well Name"]
    if "KB (m)" in selected_columns:
        cols_order.append("KB (m)")
    cols_order += [col for col in selected_columns if col not in cols_order]
    return df_filtered

@st.cache_data(show_spinner=False)
def get_logs_data_for_wells_logs(wells_list, logs_list, _wells, depth_min=None, depth_max=None):
    well_dict = {w.petrel_name: w for w in _wells}
    records = []
    depth_min = -np.inf if depth_min is None else depth_min
    depth_max =  np.inf if depth_max is None else depth_max

    for well_name in wells_list:
        well = well_dict.get(well_name)
        if well is None:
            continue
        for log in iter_selected_logs(well):  # <-- SCOPED
            if log.petrel_name in logs_list:
                df = log.as_dataframe()
                if "MD" in df.columns and "Value" in df.columns:
                    df_filtered = df[(df["MD"] >= depth_min) & (df["MD"] <= depth_max)].copy()
                    df_filtered["Well"] = well_name
                    df_filtered["Log"] = log.petrel_name
                    records.append(df_filtered[["Well", "Log", "MD", "Value"]])

    if records:
        merged_df = pd.concat(records, ignore_index=True)
        return merged_df.sort_values(["Well", "Log", "MD"]).reset_index(drop=True)
    return pd.DataFrame(columns=["Well", "Log", "MD", "Value"])

def get_global_md_range(wells_list, logs_list, well_dict):
    md_min = np.inf; md_max = -np.inf
    for well_name in wells_list:
        well = well_dict.get(well_name)
        if well is None:
            continue
        for log in iter_selected_logs(well):  # <-- SCOPED
            if log.petrel_name in logs_list:
                df = log.as_dataframe()
                if "MD" in df.columns:
                    mmin, mmax = df["MD"].min(), df["MD"].max()
                    if not np.isnan(mmin):
                        md_min = min(md_min, mmin)
                    if not np.isnan(mmax):
                        md_max = max(md_max, mmax)
    if np.isinf(md_min) or np.isinf(md_max):
        return (0, 1000)
    return (int(md_min), int(md_max))

@st.cache_data
def get_logs_for_selected_wells(well_logs, selected_wells):
    return sorted(set(
        log for well in selected_wells for log in well_logs.get(well, [])
    ))

@st.cache_data
def get_md_from_tops(tops_df, selected_wells, top_marker, base_marker):
    filtered = tops_df[tops_df['Well identifier (Well name)'].isin(selected_wells)]
    top_md = filtered[filtered["Surface"] == top_marker]["MD"].mean()
    base_md = filtered[filtered["Surface"] == base_marker]["MD"].mean()
    return sorted([top_md, base_md]) if pd.notna(top_md) and pd.notna(base_md) else (None, None)

@st.cache_data
def get_log_statistics(selected_wells, selected_logs, _well_dict, depth_min, depth_max, stat_func):
    rows = []
    for w in selected_wells:
        row = {"Well": w}
        for lg in selected_logs:
            log_obj = next((log for log in _well_dict[w].logs if log.petrel_name == lg), None)
            if log_obj:
                df = log_obj.as_dataframe()
                if "MD" in df.columns and "Value" in df.columns:
                    vals = df.loc[(df["MD"] >= depth_min) & (df["MD"] <= depth_max), "Value"].to_numpy()
                else:
                    vals = np.array([])
                row[lg] = stat_func(vals) if vals.size and not np.all(np.isnan(vals)) else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        # return an empty, well-formed frame with the expected index
        return pd.DataFrame(columns=["Well"]).set_index("Well")

    if "Well" not in df.columns:
        # defensive: ensure Well column exists (should not happen if rows were built correctly)
        df.insert(0, "Well", [r.get("Well", None) for r in rows])

    return df.set_index("Well")

@st.cache_data
def get_log_presence_matrix(selected_wells, selected_logs, _well_dict):
    """
    Returns a DataFrame showing presence (✓) or absence (✗) of selected logs for each well.
    """
    matrix = []
    for well_name in selected_wells:
        row = {"Well Name": well_name}
        well = _well_dict.get(well_name)
        available_logs = {log.petrel_name for log in well.logs} if well else set()
        for log_name in selected_logs:
            row[log_name] = "✓" if log_name in available_logs else "✗"
        matrix.append(row)
    return pd.DataFrame.from_records(matrix)

def highlight_presence(val):
    if val == "✓":
        return "background-color:#c6f6d5"  # green
    if val == "✗":
        return "background-color:#feb2b2"  # red
    return ""


def get_stat_func_from_option(stat_option: str):
    if stat_option.endswith("%"):
        q = float(stat_option.rstrip("%")) / 100
        return lambda arr: np.nanquantile(arr, q)
    else:
        return getattr(np, stat_option)

@st.cache_data(show_spinner=False)
def plot_outliers(df_long, plot_kind="Box", stat_option="mean"):
    fig, ax = plt.subplots(figsize=(12, 6))
    if plot_kind == "Box":
        sns.boxplot(data=df_long, x="Log", y="Value", ax=ax)
    else:
        sns.violinplot(data=df_long, x="Log", y="Value", ax=ax)
    ax.set_title(f"{plot_kind} Plot of {stat_option} Values per Log")
    ax.tick_params(axis="x", rotation=45)
    return fig

@st.cache_data(show_spinner=False)
def plot_missing_matrix(df):
    fig, ax = plt.subplots(figsize=(10, 4))
    msno.matrix(df, ax=ax, sparkline=False)
    ax.tick_params(axis='x', rotation=90, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    return fig

def filter_by_depth(log, depth_min, depth_max):
    # Assuming log has MD and values as arrays or DataFrame columns:
    df = log.as_dataframe()
    mask = (df['MD'] >= depth_min) & (df['MD'] <= depth_max)
    return df.loc[mask, 'Value'].values

def run_quality_checks(wells, depth_min, depth_max, selected_checks, selected_logs=None):
    """
    Returns a DataFrame with columns "<log> - <check>" marked PASS/FAIL.
    """
    quality_results = []

    for well in wells:
        row = {"Well": well.petrel_name}
        for log in well.logs:
            if selected_logs is not None and log.petrel_name not in selected_logs:
                continue

            vals = filter_by_depth(log, depth_min, depth_max)
            if vals.size == 0 or np.all(np.isnan(vals)):
                for test_name in selected_checks.keys():
                    row[f"{log.petrel_name} - {test_name}"] = "FAIL"
                continue

            for test_name, params in selected_checks.items():
                try:
                    if test_name == "all_positive":
                        passed = np.all(vals > 0)
                    elif test_name == "all_above":
                        threshold = params.get("threshold", 0)
                        passed = np.all(vals > threshold)
                    elif test_name == "mean_below":
                        threshold = params.get("threshold", np.inf)
                        passed = np.nanmean(vals) < threshold
                    elif test_name == "no_nans":
                        passed = not np.any(np.isnan(vals))
                    elif test_name == "range":
                        min_val = params.get("min", -np.inf)
                        max_val = params.get("max", np.inf)
                        passed = np.all((vals >= min_val) & (vals <= max_val))
                    elif test_name == "no_flat":
                        diffs = np.diff(vals)
                        passed = not np.any(diffs == 0)
                    elif test_name == "no_monotonic":
                        diffs = np.diff(vals)
                        passed = not (np.all(diffs >= 0) or np.all(diffs <= 0))
                    else:
                        passed = False
                except Exception:
                    passed = False

                row[f"{log.petrel_name} - {test_name}"] = "PASS" if passed else "FAIL"

        quality_results.append(row)

    return pd.DataFrame(quality_results).set_index("Well")

def highlight_pass_fail(val):
    s = str(val).strip().lower()
    if s in ("pass", "true", "ok", "1", "yes", "success"):
        return "background-color:#c6f6d5"
    if s in ("fail", "false", "0", "no", "error"):
        return "background-color:#feb2b2"
    return ""


# ==========================  FUNCTIONS FOR GANTT CHART  ==========================
def load_timeline_data():
    if os.path.exists(TIMELINE_FILE):
        with open(TIMELINE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_timeline_data(data):
    with open(TIMELINE_FILE, "w") as f:
        json.dump(data, f, indent=4, default=str)

def load_seismic_data():
    # Load from a JSON file, e.g., "seismic_data.json"
    try:
        with open("seismic_data.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_seismic_data(data):
    with open("seismic_data.json", "w") as f:
        json.dump(data, f, indent=2)

def safe_date_input(label, value=None, min_value=None, max_value=None, key=None):
    """
    Wrapper around st.date_input that ensures value is a valid date.
    If invalid or None, sets to min_value or today's date.
    """
    from datetime import date

    # If value is string, convert to date
    if isinstance(value, str):
        try:
            value = pd.to_datetime(value).date()
        except Exception:
            value = None

    # Set fallback date
    if value is None:
        if min_value:
            value = min_value
        else:
            value = date.today()

    return st.date_input(label, value=value, min_value=min_value, max_value=max_value, key=key)

def inject_sidebar_css():
    st.markdown("""
        <style>
            /* Make the sidebar content a full-height flex column */
            [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
                display: flex !important;
                flex-direction: column;
                height: 100%;
            }
            /* The footer will naturally sit at the bottom with margin-top:auto */
            .sidebar-footer {
                margin-top: auto;          /* pushes it to the bottom */
                padding: 12px 10px 16px;
                text-align: center;
                box-sizing: border-box;
                border-top: 1px solid rgba(0,0,0,0.06);
            }
            .sidebar-footer img {
                max-width: 100%;
                height: auto;
                display: block;
                margin: 0 auto 6px auto;
            }
            .sidebar-footer .meta {
                font-size: 0.85rem;
                line-height: 1.2rem;
                color: rgba(49,51,63,0.75);
                margin-top: 2px;
            }
        </style>
    """, unsafe_allow_html=True)

# --- utils.py ---

def flatten_wells_like(old_wells):
    """Accept dict/list/iterator and return a flat list of Well objects."""
    out = []
    if isinstance(old_wells, dict):
        vals = old_wells.values()
    else:
        vals = old_wells
    for v in vals:
        if isinstance(v, (list, tuple)):
            out.extend(v)
        else:
            out.append(v)
    return out

def get_all_wells_flat_scoped(petrel):
    # Already selection-aware; no second filter here
    return get_all_wells_flat(petrel)

def filter_std_by_groups(std_options: list[str], std_to_group: dict, selected_groups: list[str]) -> list[str]:
    """Return only standardized codes whose family is in selected_groups."""
    if not selected_groups:
        return []
    sg = set(selected_groups)
    return [s for s in std_options if std_to_group.get(s, "(unlabeled)") in sg]

def format_std_with_group(std: str, std_to_group: dict) -> str:
    grp = std_to_group.get(std, "")
    return f"{std} — {grp}" if grp else std

def render_grouped_sidebar_nav():
    """Custom grouped sidebar: Overview link on top, expanders below."""
    hide_native_sidebar_pages()

    with st.sidebar:
        # Top Overview link = the registry item with default=True
        summary = next((s for s in PAGE_REGISTRY if s.get("default")), None)
        if summary:
            st.page_link(_page_relpath(summary), label=summary["title"])
        st.markdown("---")

        for group in PAGE_GROUP_ORDER:
            items = [s for s in PAGE_REGISTRY if s.get("group") == group]
            if not items:
                continue
            with st.expander(group, expanded=False):
                for spec in items:
                    st.page_link(_page_relpath(spec), label=spec["title"])
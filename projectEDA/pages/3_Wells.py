import streamlit as st
from cegalprizm.pythontool import PetrelConnection, Well, WellLog, Point
import pandas as pd
import utils
import plotly.express as px
import pandas.api.types as pdt


# === Standardization: latest map + families ===
import json, re
from pathlib import Path
from collections import defaultdict

import utils
utils.render_grouped_sidebar_nav()
########################################
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

NAME_MAP = _load_latest_name_map()  # {original -> standardized code like 'GR_STD'}
NAME_MAP_EMPTY = not bool(NAME_MAP)

# reverse: standardized -> [originals]
STD_TO_ORIG = defaultdict(list)
for orig, std in NAME_MAP.items():
    STD_TO_ORIG[std].append(orig)

# standardized code -> family (group)
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

def _fmt_std(std: str) -> str:
    grp = STD_TO_GROUP.get(std, "")
    return f"{std} — {grp}" if grp else std

# Helper: which standardized codes are available from a list of original names?
def to_standardized_options(original_names: list[str]) -> list[str]:
    if NAME_MAP_EMPTY:
        return sorted({str(n) for n in original_names})  # fallback
    std_set = set()
    for n in original_names:
        if n in NAME_MAP:
            std_set.add(NAME_MAP[n])
    return sorted(std_set)

# Only wells that contain at least one "Use in project" log when map exists
def eligible_well_names(wells) -> list[str]:
    if NAME_MAP_EMPTY:
        return [w.petrel_name for w in wells]
    allowed = set(NAME_MAP.keys())
    return [
        w.petrel_name for w in wells
        if any(getattr(lg, "petrel_name", "") in allowed for lg in getattr(w, "logs", []))
    ]

# ==========================  STREAMLIT CONFIG  ==========================
st.set_page_config(page_title='GeoPython',
                   layout='wide',
                   #page_icon=':bar_chart:',
                   initial_sidebar_state='expanded')

# ============== Petrel connection ===============
petrel_project = utils.get_petrel_connection()

############################################

def to_standardized_options(original_names: list[str]) -> list[str]:
    std_set = set()
    for n in original_names:
        std = NAME_MAP.get(n, n)
        std_set.add(std)
    return sorted(std_set)

# MENU 
#########################################
utils.sidebar_footer(petrel_project, utils.LOGO_PATH, utils.APP_VERSION)

# ===============================   LOGIC   ==============================
wells = utils.get_all_wells_flat(petrel_project)
df_summary, _, _ = utils.get_all_well_data(petrel_project)

# =============================    MAIN PAGE   ============================
st.title("GeoPython")

# ===================== two-column layout =====================
left, right = st.columns([1, 2])

# ========================= LEFT: filters + table =========================
with left:
    st.subheader("Filters")

    # Well picker
    all_well_names = df_summary["Well Name"].tolist()
    selected_wells = st.multiselect(
        "Select Wells to Display",
        options=all_well_names,
        default=all_well_names
    )

    # Available attributes
    all_stat_keys = utils.extract_stat_keys_from_wells(wells)
    default_columns = ["KB (m)", "MdAtFirstPoint", "MdAtLastPoint"]
    additional_columns = sorted(k for k in all_stat_keys if k not in default_columns)
    column_options = default_columns + additional_columns

    selected_columns = st.multiselect(
        "Select Well Attributes to Display",
        options=column_options,
        default=default_columns
    )

    # Build filtered dataframe from full stats
    filtered_rows = []
    for well in wells:
        if well.petrel_name not in selected_wells:
            continue

        stats = utils.map_wellstats(well.retrieve_stats())
        # only keep requested columns that exist in stats
        row = {key: getattr(stats, key, None) for key in selected_columns if key in vars(stats)}

        # KB explicitly from well object if requested
        if "KB (m)" in selected_columns:
            row["KB (m)"] = well.well_datum[1]

        row["Well Name"] = well.petrel_name
        filtered_rows.append(row)

    df_filtered = pd.DataFrame(filtered_rows)

    # Reorder columns to ensure "Well Name" is first
    if not df_filtered.empty:
        cols_order = ["Well Name"] + [c for c in selected_columns if c != "Well Name" and c in df_filtered.columns]
        if "KB (m)" in selected_columns and "KB (m)" not in cols_order and "KB (m)" in df_filtered.columns:
            cols_order.append("KB (m)")
        # Keep only existing columns (defensive)
        cols_order = [c for c in cols_order if c in df_filtered.columns]
        df_filtered = df_filtered[cols_order]

    st.subheader("Selected Wells Table")
    st.dataframe(df_filtered, use_container_width=True, hide_index=True)

# ========================= RIGHT: map (always visible) =========================
with right:
    st.subheader("Map")

    # Basic sanity
    if not selected_wells:
        st.info("Select at least one well to display on the map.")
    else:
        # Get lat/lon
        geo_df = utils.get_well_min_lat_long(wells).copy()

        # Ensure we have latitude/longitude column names
        if "latitude" not in geo_df.columns or "longitude" not in geo_df.columns:
            # Assume first two columns are lat/lon if names are unknown
            if len(geo_df.columns) >= 2:
                geo_df = geo_df.rename(columns={geo_df.columns[0]: "latitude", geo_df.columns[1]: "longitude"})

        # Attach well names in the same order as wells
        geo_df = geo_df.copy()
        geo_df["Well Name"] = [w.petrel_name for w in wells]

        # Keep only selected wells
        geo_df = geo_df[geo_df["Well Name"].isin(selected_wells)].reset_index(drop=True)

        # Merge the selected attributes so we can color by them
        # (left is geo_df so we never drop coordinates)
        df_map = geo_df.merge(df_filtered, on="Well Name", how="left")

        # Build list of viable numeric columns to color by
        exclude_cols = {"Well Name", "latitude", "longitude"}
        color_candidates = [
            c for c in df_map.columns
            if c not in exclude_cols and pdt.is_numeric_dtype(df_map[c])
        ]

        # UI control for color-by (map-specific filter)
        color_attr = None
        if color_candidates:
            # Try to prefer a non-KB default if available
            default_color = next((c for c in color_candidates if c != "KB (m)"), color_candidates[0])
            color_attr = st.selectbox(
                "Color wells by",
                options=color_candidates,
                index=color_candidates.index(default_color)
            )
        else:
            st.info("No numeric attributes available for color coding. Showing single-color markers.")

        # Plot
        if df_map.empty or df_map[["latitude", "longitude"]].dropna().empty:
            st.warning("No coordinates available for the selected wells.")
        else:
            if color_attr:
                fig = px.scatter_mapbox(
                    df_map,
                    lat="latitude",
                    lon="longitude",
                    color=color_attr,
                    hover_name="Well Name",
                    zoom=6,
                    height=600,
                    color_continuous_scale=px.colors.cyclical.IceFire,
                    mapbox_style="open-street-map"
                )
            else:
                # No color column: plot in a single color
                fig = px.scatter_mapbox(
                    df_map,
                    lat="latitude",
                    lon="longitude",
                    hover_name="Well Name",
                    zoom=6,
                    height=600,
                    mapbox_style="open-street-map"
                )
                fig.update_traces(marker=dict(size=12))

            st.plotly_chart(fig, use_container_width=True)
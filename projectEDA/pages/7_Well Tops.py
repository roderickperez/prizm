import streamlit as st
import utils
import plotly.express as px
import pandas as pd

import utils
utils.render_grouped_sidebar_nav()

# ==========================  STREAMLIT CONFIG  ==========================
st.set_page_config(page_title='GeoPython',
                   #page_icon=':bar_chart:', 
                   layout='wide',
                   initial_sidebar_state='expanded')

# ============== Petrel connection ===============
petrel_project = utils.get_petrel_connection()

####################### MAIN ###################################
# ==========================  MAIN  ==========================
st.title("Well Tops")

# Build data once
tops = utils.load_tops_dataframe(petrel_project)

# Three-column layout
col1, col2, col3 = st.columns([1, 2, 2])

with col1:
    # Sidebar footer stays here
    utils.sidebar_footer(petrel_project, utils.LOGO_PATH, utils.APP_VERSION)

    # Wells multiselect (moved here, renamed)
    all_well_names = tops["Well identifier (Well name)"].unique().tolist()
    selected_wells = st.multiselect(
        "Select wells",
        options=all_well_names,
        default=all_well_names,
        key="select_wells_for_all"
    )

    # Matrix controls
    value_choice = st.selectbox(
        "Select value to show in matrix:",
        options=["X", "Y", "Z", "TWT picked", "TWT auto", "MD", "PVD auto", "TVDSS"],
        index=5
    )
    round_digits = st.number_input(
        "Select number of decimal places",
        min_value=0, max_value=10,
        value=2, step=1,
        format="%d",
        key="tops_round_digits"
    )

    # Pivot matrix
    pivot_df = tops.pivot_table(
        index="Well identifier (Well name)",
        columns="Surface",
        values=value_choice,
        aggfunc="first"
    ).round(round_digits).fillna("-")

    st.write(f"**Total Wells:** {pivot_df.shape[0]} | **Total Surfaces:** {pivot_df.shape[1]}")
    st.dataframe(pivot_df, use_container_width=True)

# Prepare filtered data for the other columns
filtered_tops = tops[tops["Well identifier (Well name)"].isin(selected_wells)].copy()

with col2:
    # Surface selector (now in col2)
    available_tops = sorted(filtered_tops["Surface"].unique().tolist())
    if not available_tops:
        st.warning("No tops available for the selected wells.")
    else:
        selected_top = st.selectbox("Select a well top to check presence", options=available_tops)

        # Identify presence per well for the selected top
        wells_with_top = filtered_tops[
            filtered_tops["Surface"] == selected_top
        ]["Well identifier (Well name)"].unique().tolist()

        # Map
        wells_all = utils.get_all_wells_flat(petrel_project)
        geo_df = utils.get_well_min_lat_long(wells_all)

        if geo_df.empty or "latitude" not in geo_df.columns or "longitude" not in geo_df.columns:
            st.error("Latitude/Longitude data not found in expected format (`latitude`, `longitude`).")
        else:
            geo_df.columns = ["latitude", "longitude"]
            geo_df["Well Name"] = [w.petrel_name for w in wells_all]

            # Keep only selected wells on the map
            geo_df = geo_df[geo_df["Well Name"].isin(selected_wells)]

            # Presence flag
            geo_df["Presence"] = geo_df["Well Name"].apply(
                lambda well: "Present" if well in wells_with_top else "Absent"
            )

            color_map = {"Present": "green", "Absent": "red"}
            fig = px.scatter_mapbox(
                geo_df,
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

with col3:
    st.markdown("### Depth distribution for selected top")

    # Choose depth metric
    depth_candidates = [c for c in ["MD", "TVDSS", "TWT picked", "TWT auto", "PVD auto", "Z"]
                        if c in filtered_tops.columns]
    if not depth_candidates:
        st.info("No depth-like columns found to plot.")
    else:
        depth_metric = st.selectbox(
            "Depth metric",
            options=depth_candidates,
            index=(depth_candidates.index("MD") if "MD" in depth_candidates else 0),
            key="tops_depth_metric"
        )

        # Data for scatter/box
        df_top = filtered_tops[(filtered_tops["Surface"] == selected_top)].copy()
        df_top = df_top[df_top["Well identifier (Well name)"].isin(selected_wells)]
        df_top = df_top.dropna(subset=[depth_metric])

        if df_top.empty:
            st.info("No data to plot for the chosen top, wells, and depth metric.")
        else:
            well_col = "Well identifier (Well name)"
            # Keep x order consistent with selection
            df_top[well_col] = pd.Categorical(df_top[well_col], categories=selected_wells, ordered=True)

            # Split plotting area in 3:1 ratio
            pleft, pright = st.columns([3, 1])

            # LEFT: Scatter (square symbols), depth vs well
            with pleft:
                scatter_fig = px.scatter(
                    df_top,
                    x=well_col,
                    y=depth_metric,
                    hover_data=[well_col, "Surface"] if "Surface" in df_top.columns else [well_col],
                )
                scatter_fig.update_traces(marker=dict(symbol="square", size=9, line=dict(width=0)))
                # Depth increases downward
                scatter_fig.update_yaxes(autorange="reversed")
                scatter_fig.update_layout(
                    xaxis_title="Well",
                    yaxis_title=depth_metric,
                    height=480,
                    margin=dict(l=40, r=10, t=30, b=60)
                )
                st.plotly_chart(scatter_fig, use_container_width=True)

            # RIGHT: Single box plot for all values (aggregate across wells)
            with pright:
                # One-column DF for aggregated box
                agg_df = df_top[[depth_metric]].copy()
                agg_df["All Wells"] = "All Wells"
                box_fig = px.box(
                    agg_df,
                    x="All Wells",
                    y=depth_metric,
                    points="outliers"
                )
                # Depth increases downward
                box_fig.update_yaxes(autorange="reversed")
                box_fig.update_layout(
                    xaxis_title=None,
                    yaxis_title=None,
                    height=480,
                    margin=dict(l=10, r=20, t=30, b=60),
                    showlegend=False
                )
                st.plotly_chart(box_fig, use_container_width=True)
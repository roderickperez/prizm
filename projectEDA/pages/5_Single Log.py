import streamlit as st
import numpy as np
import pandas as pd
import utils
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import plotly.graph_objects as go
import plotly.express as px

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

st.title("Single Well Log")

# Get wells and other well data from utils
wells = utils.get_all_wells_flat(petrel_project)  # Not cached
wells_summary_df, wells_full_df, stat_keys = utils.get_all_well_data(petrel_project)

tops_df = utils.load_tops_dataframe(petrel_project)

well_names = [w.petrel_name for w in wells]
well_dict = {w.petrel_name: w for w in wells}
well_logs = {
    w.petrel_name: [getattr(lg, "petrel_name", "") for lg in getattr(w, "logs", [])]
    for w in wells
}

# --- Mode selection ---
# mode = st.radio("Select Log Mode", ["All Logs", "Single Log"], horizontal=True)

# --- after st.title("Single Well Log") and after you've loaded wells/tops/utils ---

# Split main layout: left controls, right content
col1, col2 = st.columns([1, 4])

with col1:
    st.subheader("Selections")

    # Single well / log pickers
    selected_well = st.selectbox("Select a well", options=well_names, key="single_well")
    available_logs = well_logs.get(selected_well, [])
    selected_log  = st.selectbox("Select a log", options=available_logs, key="single_log")

    # Prepare defaults in case nothing is selected yet
    depth_min, depth_max = None, None
    df_log_filtered = pd.DataFrame()
    log_obj = None

    if selected_well and selected_log:
        well_obj = well_dict[selected_well]
        log_obj = next((lg for lg in well_obj.logs if lg.petrel_name == selected_log), None)

        if log_obj:
            # Defaults based on this well+log
            md_min_default, md_max_default = utils.get_global_md_range([selected_well], [selected_log], well_dict)

            # Reset applied range if user changed well or log
            single_key = (selected_well, selected_log)
            if st.session_state.get("single_last_key") != single_key:
                st.session_state["single_last_key"] = single_key
                st.session_state["single_depth_range_applied"] = (md_min_default, md_max_default)
                for k in ("single_depth_slider_pending", "single_top_pending", "single_base_pending",
                        "single_qc_df", "single_qc_meta", "single_qc_mask"):
                    st.session_state.pop(k, None)

            # Ensure applied exists
            if "single_depth_range_applied" not in st.session_state:
                st.session_state["single_depth_range_applied"] = (md_min_default, md_max_default)

            depth_selection_mode = st.radio(
                "How would you like to select the depth range?",
                ["Slider", "Tops"], horizontal=True, key="single_depth_mode"
            )

            if depth_selection_mode == "Slider":
                # Only apply when the form is submitted
                with st.form("single_depth_slider_form"):
                    pending = st.session_state.get("single_depth_slider_pending",
                                                st.session_state["single_depth_range_applied"])
                    temp_range = st.slider(
                        "Select depth range (MD)",
                        min_value=md_min_default,
                        max_value=md_max_default,
                        value=tuple(pending),
                        step=1,
                        key="single_depth_slider"
                    )
                    st.session_state["single_depth_slider_pending"] = temp_range
                    submitted = st.form_submit_button("Apply Depth Range")
                    if submitted:
                        st.session_state["single_depth_range_applied"] = temp_range
                        for k in ("single_qc_df", "single_qc_meta", "single_qc_mask"):
                            st.session_state.pop(k, None)

            else:
                # Tops mode  apply only on submit
                filtered_tops = tops_df[tops_df['Well identifier (Well name)'] == selected_well].copy()
                top_names = sorted(filtered_tops["Surface"].unique())

                with st.form("single_tops_form"):
                    default_top  = st.session_state.get("single_top_pending",  top_names[0] if top_names else None)
                    default_base = st.session_state.get("single_base_pending", top_names[min(len(top_names)-1, 1)] if top_names else None)

                    top_marker = st.selectbox(
                        "Select **top** marker",
                        top_names,
                        index=top_names.index(default_top) if default_top in top_names else 0,
                        key="single_top_marker"
                    )
                    base_marker = st.selectbox(
                        "Select **base** marker",
                        top_names,
                        index=top_names.index(default_base) if default_base in top_names else min(len(top_names)-1, 1),
                        key="single_base_marker"
                    )
                    st.session_state["single_top_pending"]  = top_marker
                    st.session_state["single_base_pending"] = base_marker

                    submitted = st.form_submit_button("Apply Tops Range")
                    if submitted:
                        dm, dx = utils.get_md_from_tops(tops_df, [selected_well], top_marker, base_marker)
                        if dm is not None and dx is not None:
                            st.session_state["single_depth_range_applied"] = (dm, dx)
                        else:
                            st.warning("Selected tops not found in this well. Using full range.")
                            st.session_state["single_depth_range_applied"] = (md_min_default, md_max_default)
                        for k in ("single_qc_df", "single_qc_meta", "single_qc_mask"):
                            st.session_state.pop(k, None)

            # Use the applied depth window below
            depth_min, depth_max = st.session_state["single_depth_range_applied"]
            st.caption(f"Applied depth range: {depth_min:.2f}  {depth_max:.2f} m")

            # Filter the selected log by depth
            df_log = log_obj.as_dataframe()
            if {"MD", "Value"}.issubset(df_log.columns):
                df_log_filtered = (
                    df_log[(df_log["MD"] >= depth_min) & (df_log["MD"] <= depth_max)]
                    .sort_values("MD")
                    .reset_index(drop=True)
                )
            else:
                st.warning("Log data missing 'MD' or 'Value' columns.")

with col2:
    # Only show tabs if we have a valid selection and filtered data context
    if not selected_well or not selected_log or log_obj is None:
        st.info("Please select a well and a log on the left.")
    else:
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            " Data Table", " Statistics", "Outlier Visualization",
            "Missing-Value Plot", "Quality Checks", "Well Log Viewer"
        ])

        with tab1:
            st.markdown("### Well Log Data Table")
            st.dataframe(df_log_filtered[["MD", "Value"]], hide_index=True)

        with tab2:
            st.markdown("### Log Statistics")

            if not df_log_filtered.empty:
                # Controls (above the table)
                stat_option = st.selectbox(
                    "Statistic",
                    ["All", "mean", "median", "min", "max", "std", "25%", "50%", "75%"],
                    index=0,
                    key="single_stat_select",
                )

                round_digits = st.number_input(
                    "Select number of decimal places",
                    min_value=0, max_value=10,
                    value=2, step=1,
                    format="%d",
                    key="single_round_digits"
                )

                # Build full stats table
                stats = {
                    "mean":   df_log_filtered["Value"].mean(),
                    "median": df_log_filtered["Value"].median(),
                    "min":    df_log_filtered["Value"].min(),
                    "max":    df_log_filtered["Value"].max(),
                    "std":    df_log_filtered["Value"].std(),
                    "25%":    df_log_filtered["Value"].quantile(0.25),
                    "50%":    df_log_filtered["Value"].quantile(0.50),
                    "75%":    df_log_filtered["Value"].quantile(0.75),
                }
                stats_df = pd.DataFrame(stats, index=[selected_log]).T.round(round_digits)

                # If a single stat is chosen, show just that row
                if stat_option != "All":
                    # Only keep the row if it exists (e.g., user picks std or a percentile)
                    if stat_option in stats_df.index:
                        stats_df = stats_df.loc[[stat_option]]
                    else:
                        st.info(f"Statistic '{stat_option}' not available; showing all.")
                st.dataframe(stats_df, use_container_width=True)

            else:
                st.info("No data available in selected depth range.")

        with tab3:
            st.markdown("### Outlier Visualization")

            if df_log_filtered.empty:
                st.info("No data available in selected depth range.")
            else:
                plot_kind = st.radio(
                    "Plot type",
                    ["Box", "Violin"],
                    horizontal=True,
                    key="single_outlier_plot_kind"
                )

                if plot_kind == "Box":
                    # Vertical box: y = Value; no points
                    fig = px.box(
                        df_log_filtered,
                        y="Value",
                        points=False,
                        title=f"Box Plot for {selected_log}"
                    )
                else:
                    # Vertical violin: y = Value; no scatter points
                    fig = px.violin(
                        df_log_filtered,
                        y="Value",
                        box=True,        # show inner box inside the violin
                        points=False,    # <-- no dots
                        title=f"Violin Plot for {selected_log}"
                    )

                fig.update_layout(
                    xaxis_title=None,
                    yaxis_title=selected_log,
                    height=450,
                    margin=dict(l=60, r=20, t=40, b=40),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab4:
            st.markdown("### Missing-Value Plot")
            if not df_log_filtered.empty:
                fig, ax = plt.subplots()
                msno.matrix(df_log_filtered, ax=ax)
                st.pyplot(fig)
            else:
                st.info("No data to plot missing values.")

        with tab5:
            st.markdown("### Quality Checks")

            if df_log_filtered.empty:
                st.info("No data available in selected depth range.")
                st.stop()

            # == Layout
            colA, colB = st.columns([3, 1])

            # == Tests + params (left)
            with colA:
                available_tests = {
                    "all_positive": {},
                    "all_above": {"threshold": 50},
                    "mean_below": {"threshold": 100},
                    "no_nans": {},
                    "range": {"min": 0, "max": 200},
                    "no_flat": {},
                    "no_monotonic": {},
                }

                selected_test_name = st.selectbox(
                    "Select a quality check",
                    options=list(available_tests.keys()),
                    index=0,
                    key="single_qc_test_select"
                )

                # Param UI (only when needed)
                params = {}
                if selected_test_name == "all_above":
                    params["threshold"] = st.slider(
                        "Threshold (all_above)",
                        min_value=-1000, max_value=1000,
                        value=available_tests["all_above"]["threshold"],
                        key="single_qc_thr_above"
                    )
                elif selected_test_name == "mean_below":
                    params["threshold"] = st.slider(
                        "Threshold (mean_below)",
                        min_value=-1000, max_value=1000,
                        value=available_tests["mean_below"]["threshold"],
                        key="single_qc_thr_mean"
                    )
                elif selected_test_name == "range":
                    rmin, rmax = st.slider(
                        "Range (min, max)",
                        min_value=-1000, max_value=1000,
                        value=(available_tests["range"]["min"], available_tests["range"]["max"]),
                        key="single_qc_range"
                    )
                    params["min"], params["max"] = rmin, rmax

                # Row-wise violation mask (for simple tests)
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
                    # aggregate-only tests: no per-row failures
                    elif test_name in {"mean_below", "no_flat", "no_monotonic"}:
                        return pd.Series(False, index=values.index)
                    else:
                        return pd.Series(False, index=values.index)

                # Run QC
                if st.button("Run Quality Check", type="primary", key="single_qc_run"):
                    qc_df = utils.run_quality_checks(
                        [well_dict[selected_well]],
                        depth_min, depth_max,
                        {selected_test_name: params},
                        [selected_log]
                    )
                    st.session_state["single_qc_df"] = qc_df
                    st.session_state["single_qc_meta"] = (selected_test_name, params)

                    # For row-addressable tests, compute & store mask for plotting
                    if selected_test_name in {"all_positive", "all_above", "range", "no_nans"}:
                        st.session_state["single_qc_mask"] = _violation_mask(
                            df_log_filtered["Value"], selected_test_name, params
                        )
                    else:
                        st.session_state["single_qc_mask"] = None  # aggregate-only

                # Results + failing table (only if we have a run)
                mask = st.session_state.get("single_qc_mask", None)
                # realign mask to current df_log_filtered
                if isinstance(mask, pd.Series):
                    if not mask.index.equals(df_log_filtered.index):
                        mask = mask.reindex(df_log_filtered.index, fill_value=False)
                elif isinstance(mask, np.ndarray):
                    if mask.shape[0] != len(df_log_filtered):
                        mask = np.zeros(len(df_log_filtered), dtype=bool)

                selected_test_name, params = st.session_state.get("single_qc_meta", (selected_test_name, params))

                if "single_qc_df" in st.session_state and st.session_state["single_qc_df"] is not None:
                    if selected_test_name in {"all_positive", "all_above", "range", "no_nans"}:
                        if mask is not None and mask.any():
                            st.error(f"[FAIL] **{selected_log}** failed **{selected_test_name}** for **{selected_well}**.")
                            out = df_log_filtered.loc[mask, ["MD", "Value"]].reset_index(drop=True)
                            st.markdown("**Failing rows (MD, Value):**")
                            st.dataframe(out, hide_index=True, use_container_width=True)
                        elif mask is not None:
                            st.success(f" **{selected_log}** passed **{selected_test_name}** for **{selected_well}**.")
                    else:
                        # Aggregate-only tests: show simple message derived from QC table if possible
                        qc_df = st.session_state["single_qc_df"]
                        try:
                            well_col = next((c for c in qc_df.columns if str(c).lower().startswith(("well","bore","hole"))), None)
                            if well_col is None:
                                qc_df = qc_df.reset_index().rename(columns={"index":"Well"})
                                well_col = "Well"
                            is_long = {"Log","Result"}.issubset(qc_df.columns)
                            if is_long:
                                row = qc_df[(qc_df[well_col] == selected_well) & (qc_df["Log"].astype(str) == selected_log)]
                                result_val = row["Result"].iloc[0] if not row.empty else None
                            else:
                                candidates = [c for c in qc_df.columns if c != well_col and str(c).split(" - ",1)[0] == selected_log]
                                result_val = qc_df.loc[qc_df[well_col] == selected_well, candidates[0]].iloc[0] if candidates else None
                            s = str(result_val).strip().lower() if result_val is not None else ""
                            if s in ("pass","","true","1","yes","ok","success"):
                                st.success(f" **{selected_log}** passed **{selected_test_name}** for **{selected_well}**.")
                            elif s in ("fail","","false","0","no","error"):
                                st.error(f"[FAIL] **{selected_log}** failed **{selected_test_name}** for **{selected_well}**.")
                            else:
                                st.info(f"Result for **{selected_test_name}** is inconclusive; row-level breakdown isn't defined.")
                        except Exception:
                            st.info(f"Ran **{selected_test_name}**. Row-level breakdown isn't defined for this test.")

            # == Plot (right): curve + highlighted anomalies
            with colB:
                st.markdown("#### Log curve (highlighting failing depths)")

                # Toggle to show/hide tops (default True)
                show_tops_qc = st.checkbox("Show well tops on plot", value=True, key="qc_show_tops")

                fig = go.Figure()

                # Base curve (Value vs MD)
                fig.add_trace(go.Scatter(
                    x=df_log_filtered["Value"],
                    y=df_log_filtered["MD"],
                    mode="lines",
                    name=selected_log
                ))

                # Highlight failing depths
                mask = st.session_state.get("single_qc_mask", None)
                # realign mask to current df_log_filtered
                if isinstance(mask, pd.Series):
                    if not mask.index.equals(df_log_filtered.index):
                        mask = mask.reindex(df_log_filtered.index, fill_value=False)
                elif isinstance(mask, np.ndarray):
                    if mask.shape[0] != len(df_log_filtered):
                        mask = np.zeros(len(df_log_filtered), dtype=bool)

                if mask is not None and mask.any():
                    bad = df_log_filtered.loc[mask].copy()

                    if selected_test_name == "no_nans":
                        # Value is NaN --> add horizontal dotted lines at those MDs
                        x0 = pd.to_numeric(df_log_filtered["Value"], errors="coerce").min()
                        x1 = pd.to_numeric(df_log_filtered["Value"], errors="coerce").max()
                        if pd.notna(x0) and pd.notna(x1) and x0 != x1:
                            for md in bad["MD"].dropna().tolist():
                                fig.add_shape(
                                    type="line", xref="x", yref="y",
                                    x0=x0, x1=x1, y0=md, y1=md,
                                    line=dict(color="red", width=2, dash="dot")
                                )
                    else:
                        # Plot anomaly points as red markers (Value must be non-NaN)
                        bad = bad.dropna(subset=["Value"])
                        if not bad.empty:
                            fig.add_trace(go.Scatter(
                                x=bad["Value"],
                                y=bad["MD"],
                                mode="markers",
                                marker=dict(color="red", size=8),
                                name="Failing rows"
                            ))

                # Optional well tops overlay (all tops for this well)
                if show_tops_qc:
                    wt = tops_df[tops_df['Well identifier (Well name)'] == selected_well]
                    if not wt.empty:
                        xmin = float(pd.to_numeric(df_log_filtered["Value"], errors="coerce").min())
                        xmax = float(pd.to_numeric(df_log_filtered["Value"], errors="coerce").max())
                        if pd.notna(xmin) and pd.notna(xmax) and xmin != xmax:
                            for _, r in wt.iterrows():
                                md_val = float(r["MD"])
                                top_name = str(r["Surface"])
                                fig.add_shape(
                                    type="line", xref='x', yref='y',
                                    x0=xmin, x1=xmax, y0=md_val, y1=md_val,
                                    line=dict(color="red", width=1, dash="dash")
                                )
                                fig.add_annotation(
                                    x=xmax, y=md_val,
                                    text=top_name,
                                    showarrow=False,
                                    xanchor='left', yanchor='middle',
                                    font=dict(color="red", size=10)
                                )

                # Layout: legend on the right
                fig.update_layout(
                    yaxis=dict(autorange="reversed", title="Measured Depth (MD)"),
                    xaxis=dict(title=selected_log),
                    height=800,
                    width=350,  # keep it narrow
                    margin=dict(l=50, r=140, t=30, b=50),  # extra right margin for legend
                    legend=dict(
                        orientation="v",
                        yanchor="top", y=1,
                        xanchor="left", x=1.02   # place legend to the right of the plot
                    )
                )
                st.plotly_chart(fig, use_container_width=False)

        with tab6:
            st.markdown("### Well Log Viewer")
            show_tops = st.checkbox("Show well tops", key="single_show_tops")

            selected_tops = []
            if show_tops:
                well_tops = tops_df[tops_df['Well identifier (Well name)'] == selected_well]
                top_options = well_tops["Surface"].unique().tolist()
                selected_tops = st.multiselect(
                    "Select well tops to display",
                    options=top_options,
                    default=top_options,
                    key="single_tops_pick"
                )

            if not df_log_filtered.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_log_filtered["Value"],
                    y=df_log_filtered["MD"],
                    mode='lines',
                    name=selected_log,
                    line=dict(color='blue')
                ))

                # Horizontal tops
                if show_tops and len(selected_tops):
                    for top_name in selected_tops:
                        md_vals = well_tops.loc[well_tops["Surface"] == top_name, "MD"].values
                        for md_val in md_vals:
                            fig.add_shape(
                                type="line",
                                x0=float(df_log_filtered["Value"].min()),
                                x1=float(df_log_filtered["Value"].max()),
                                y0=float(md_val), y1=float(md_val),
                                line=dict(color="red", width=2, dash="dash"),
                                xref='x', yref='y',
                            )
                            fig.add_annotation(
                                x=float(df_log_filtered["Value"].max()),
                                y=float(md_val),
                                text=top_name,
                                showarrow=False,
                                xanchor='left',
                                yanchor='middle',
                                font=dict(color="red")
                            )

                fig.update_layout(
                    yaxis=dict(autorange="reversed", title="Measured Depth (MD)"),
                    xaxis=dict(title=selected_log),
                    height=700,
                    width=250,   # 👈 Fix width here (adjust to taste)
                    margin=dict(l=60, r=20, t=40, b=60),
                    title=f"Well Log Viewer for {selected_log} in {selected_well}",
                )

                # 👇 Prevent auto-stretch
                st.plotly_chart(fig, use_container_width=False)
            else:
                st.info("No data available to plot.")
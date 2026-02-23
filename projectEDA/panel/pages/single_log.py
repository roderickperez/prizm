from __future__ import annotations

import panel as pn
import plotly.express as px


def build_page(service) -> pn.Column:
    wells_df = service.get_wells()
    log_catalog = service.get_logs_catalog()
    well_names = wells_df["well_name"].dropna().astype(str).tolist() if not wells_df.empty else []
    log_names = log_catalog["log_name"].dropna().astype(str).tolist() if not log_catalog.empty else []

    well_select = pn.widgets.Select(name="Well", options=well_names)
    log_select = pn.widgets.Select(name="Log", options=log_names)

    dmin, dmax = service.get_depth_range(log_names)
    if dmax <= dmin:
        dmax = dmin + 1.0
    depth_slider = pn.widgets.RangeSlider(name="Depth range (MD)", start=dmin, end=dmax, value=(dmin, dmax), step=1)

    @pn.depends(well_select, log_select, depth_slider)
    def table_view(well, log, drange):
        if not well or not log:
            return pn.pane.Markdown("Select a well and a log.")
        df, _ = service.get_single_log_stats(well, log, drange[0], drange[1])
        if df.empty:
            return pn.pane.Markdown("No data available.")
        return pn.widgets.Tabulator(df[["md", "value"]], pagination="local", page_size=30, height=450, sizing_mode="stretch_both")

    @pn.depends(well_select, log_select, depth_slider)
    def stats_view(well, log, drange):
        if not well or not log:
            return pn.pane.Markdown("Select a well and a log.")
        _, stats = service.get_single_log_stats(well, log, drange[0], drange[1])
        if stats.empty:
            return pn.pane.Markdown("No statistics available.")
        return pn.widgets.Tabulator(stats, pagination="local", page_size=20, height=420, sizing_mode="stretch_both")

    @pn.depends(well_select, log_select, depth_slider)
    def plot_view(well, log, drange):
        if not well or not log:
            return pn.pane.Markdown("Select a well and a log.")
        df, _ = service.get_single_log_stats(well, log, drange[0], drange[1])
        if df.empty:
            return pn.pane.Markdown("No data to plot.")
        fig = px.line(df, x="value", y="md", title=f"{well} - {log}")
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(height=430, margin={"l": 0, "r": 0, "t": 40, "b": 0})
        return pn.pane.Plotly(fig, sizing_mode="stretch_both")

    tabs = pn.Tabs(
        ("Data Table", table_view),
        ("Statistics", stats_view),
        ("Log Curve", plot_view),
        sizing_mode="stretch_both",
        dynamic=True,
    )

    return pn.Column(
        pn.pane.Markdown("# Single Log"),
        pn.Row(well_select, log_select, depth_slider, sizing_mode="stretch_width"),
        tabs,
        sizing_mode="stretch_both",
    )

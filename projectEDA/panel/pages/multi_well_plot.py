from __future__ import annotations

import panel as pn
import plotly.express as px


def build_page(service) -> pn.Column:
    log_catalog = service.get_logs_catalog()
    log_names = log_catalog["log_name"].dropna().astype(str).tolist() if not log_catalog.empty else []

    x_select = pn.widgets.Select(name="X log", options=log_names)
    y_select = pn.widgets.Select(name="Y log", options=log_names)

    dmin, dmax = service.get_depth_range(log_names)
    if dmax <= dmin:
        dmax = dmin + 1.0
    depth_slider = pn.widgets.RangeSlider(name="Depth range (MD)", start=dmin, end=dmax, value=(dmin, dmax), step=1)

    @pn.depends(x_select, y_select, depth_slider)
    def crossplot(x_log, y_log, drange):
        if not x_log or not y_log:
            return pn.pane.Markdown("Select X and Y logs.")
        df = service.get_multi_well_crossplot(x_log, y_log, depth_min=drange[0], depth_max=drange[1])
        if df.empty:
            return pn.pane.Markdown("No data to build multi-well crossplot.")
        fig = px.scatter(df, x="x_value", y="y_value", color="well_name", hover_name="well_name")
        fig.update_layout(height=620, margin={"l": 0, "r": 0, "t": 20, "b": 0})
        return pn.pane.Plotly(fig, sizing_mode="stretch_both")

    return pn.Column(
        pn.pane.Markdown("# Multi Well Plot"),
        pn.Row(x_select, y_select, depth_slider, sizing_mode="stretch_width"),
        crossplot,
        sizing_mode="stretch_both",
    )

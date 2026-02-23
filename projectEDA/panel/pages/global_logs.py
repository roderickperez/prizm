from __future__ import annotations

import panel as pn
import pandas as pd
import plotly.express as px


def _safe_float_pair(value):
    if not value:
        return (0.0, 0.0)
    return float(value[0]), float(value[1])


def build_page(service) -> pn.Column:
    catalog = service.get_logs_catalog()
    well_names = service.get_wells()["well_name"].dropna().astype(str).tolist() if not service.get_wells().empty else []
    log_names = catalog["log_name"].dropna().astype(str).tolist() if not catalog.empty else []

    wells_selector = pn.widgets.MultiChoice(name="Wells", options=well_names, value=well_names)
    logs_selector = pn.widgets.MultiChoice(name="Logs", options=log_names, value=log_names[: min(6, len(log_names))])

    depth_range = service.get_depth_range(log_names if log_names else None)
    dmin, dmax = _safe_float_pair(depth_range)
    if dmax <= dmin:
        dmax = dmin + 1.0
    depth_slider = pn.widgets.RangeSlider(name="Depth range (MD)", start=dmin, end=dmax, value=(dmin, dmax), step=1)

    top_names = service.get_top_names()
    depth_mode = pn.widgets.RadioButtonGroup(name="Depth mode", options=["Slider", "Tops"], value="Slider")
    top_select = pn.widgets.Select(name="Top", options=top_names)
    base_select = pn.widgets.Select(name="Base", options=top_names)

    def _resolve_depth_window(mode, drange, top, base):
        if mode == "Tops" and top and base:
            win = service.get_depth_window_from_tops(top, base)
            if win:
                return float(win[0]), float(win[1])
        return float(drange[0]), float(drange[1])

    @pn.depends(wells_selector, logs_selector, depth_slider, depth_mode, top_select, base_select)
    def data_table_view(wells, logs, drange, mode, top, base):
        if not logs:
            return pn.pane.Markdown("Select at least one log.")
        depth_min, depth_max = _resolve_depth_window(mode, drange, top, base)
        df = service.get_logs_data(well_names=wells or None, log_names=logs, depth_min=depth_min, depth_max=depth_max)
        if df.empty:
            return pn.pane.Markdown("No log data in selected range.")
        return pn.widgets.Tabulator(df.head(3000), pagination="local", page_size=25, height=450, sizing_mode="stretch_both")

    @pn.depends(logs_selector, depth_slider, depth_mode, top_select, base_select)
    def completeness_view(logs, drange, mode, top, base):
        if not logs:
            return pn.pane.Markdown("Select at least one log.")
        depth_min, depth_max = _resolve_depth_window(mode, drange, top, base)
        comp = service.get_completeness(logs, depth_min, depth_max)
        if comp.empty:
            return pn.pane.Markdown("No completeness results.")
        return pn.widgets.Tabulator(comp, pagination="local", page_size=20, height=420, sizing_mode="stretch_both")

    @pn.depends(logs_selector, depth_slider, depth_mode, top_select, base_select)
    def stats_view(logs, drange, mode, top, base):
        if not logs:
            return pn.pane.Markdown("Select at least one log.")
        depth_min, depth_max = _resolve_depth_window(mode, drange, top, base)
        df = service.get_logs_data(log_names=logs, depth_min=depth_min, depth_max=depth_max)
        if df.empty:
            return pn.pane.Markdown("No data for statistics.")
        stats = (
            df.groupby("log_name")["value"]
            .describe(percentiles=[0.25, 0.5, 0.75])
            .reset_index()
        )
        return pn.widgets.Tabulator(stats, pagination="local", page_size=20, height=420, sizing_mode="stretch_both")

    @pn.depends(logs_selector, depth_slider, depth_mode, top_select, base_select)
    def hist_view(logs, drange, mode, top, base):
        if not logs:
            return pn.pane.Markdown("Select at least one log.")
        depth_min, depth_max = _resolve_depth_window(mode, drange, top, base)
        df = service.get_logs_data(log_names=logs, depth_min=depth_min, depth_max=depth_max)
        if df.empty:
            return pn.pane.Markdown("No data for histogram.")
        fig = px.histogram(df, x="value", color="log_name", nbins=40, barmode="overlay", opacity=0.65)
        fig.update_layout(height=420, margin={"l": 0, "r": 0, "t": 10, "b": 0})
        return pn.pane.Plotly(fig, sizing_mode="stretch_both")

    tabs = pn.Tabs(
        ("Data Table", data_table_view),
        ("Completeness", completeness_view),
        ("Statistics", stats_view),
        ("Histogram", hist_view),
        sizing_mode="stretch_both",
        dynamic=True,
    )

    controls = pn.Column(
        pn.pane.Markdown("### Filters"),
        wells_selector,
        logs_selector,
        depth_mode,
        depth_slider,
        top_select,
        base_select,
        sizing_mode="stretch_width",
    )

    return pn.Column(
        pn.pane.Markdown("# Global Logs"),
        pn.Row(controls, tabs, sizing_mode="stretch_both"),
        sizing_mode="stretch_both",
    )

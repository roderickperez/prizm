from __future__ import annotations

import pandas as pd
import panel as pn
import plotly.express as px


def _kpi_card(title: str, value: int, color: str) -> pn.Column:
    return pn.Column(
        pn.pane.Markdown(
            f"""
            <div style='background:white;border-radius:8px;padding:12px 16px;border:1px solid #d5d6d6;'>
              <div style='font-size:14px;color:#6b7280;'>{title}</div>
              <div style='font-size:32px;font-weight:700;color:{color};'>{value}</div>
            </div>
            """
        ),
        sizing_mode="stretch_width",
    )


def build_page(service) -> pn.Column:
    counts = service.get_counts()
    wells = service.get_wells()

    map_df = wells.copy()
    if "latitude" not in map_df.columns or "longitude" not in map_df.columns:
        map_df["latitude"] = map_df.get("y")
        map_df["longitude"] = map_df.get("x")

    map_df = map_df.dropna(subset=["latitude", "longitude"])
    if map_df.empty:
        fig = px.scatter_mapbox(lat=[], lon=[])
        fig.update_layout(title="No well coordinates available", mapbox_style="open-street-map")
    else:
        fig = px.scatter_mapbox(
            map_df,
            lat="latitude",
            lon="longitude",
            hover_name="well_name",
            mapbox_style="open-street-map",
            zoom=4,
            height=650,
        )
        fig.update_traces(marker={"size": 10, "color": "#007733"})
        fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})

    kpis = pn.Row(
        _kpi_card("Number of wells in project", counts["wells"], "#007733"),
        _kpi_card("Number of well logs", counts["logs"], "#335290"),
        _kpi_card("Number of well tops", counts["tops"], "#8848ea"),
        sizing_mode="stretch_width",
    )

    return pn.Column(
        pn.pane.Markdown("# Project Summary"),
        kpis,
        pn.pane.Plotly(fig, sizing_mode="stretch_both", min_height=650),
        sizing_mode="stretch_both",
    )

from __future__ import annotations

import pandas as pd
import panel as pn
import plotly.express as px


def _kpi_card(title: str, value: int, color: str) -> pn.Card:
    body = pn.pane.HTML(
        f"""
        <div style='padding: 8px 6px 4px 6px;'>
            <div style='font-size:16px; color:#6b7280; margin-bottom:10px;'>{title}</div>
            <div style='font-size:44px; font-weight:700; color:{color}; line-height:1.0;'>{value}</div>
        </div>
        """,
        sizing_mode="stretch_both",
    )
    return pn.Card(
        body,
        title="",
        hide_header=True,
        sizing_mode="stretch_width",
        min_height=170,
        styles={"background": "white", "border": "1px solid #d5d6d6", "border-radius": "8px"},
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
            height=500,
        )
        fig.update_traces(marker={"size": 10, "color": "#007733"})
        fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})

    kpis = pn.Row(
        _kpi_card("Number of wells in project", counts["wells"], "#007733"),
        _kpi_card("Number of well logs", counts["logs"], "#335290"),
        _kpi_card("Number of well tops", counts["tops"], "#8848ea"),
        sizing_mode="stretch_width",
        margin=(4, 0, 14, 0),
        styles={"gap": "24px"},
    )

    map_box = pn.Card(
        pn.pane.Plotly(fig, sizing_mode="stretch_width", min_height=500),
        title="Well Locations",
        collapsed=False,
        hide_header=False,
        sizing_mode="stretch_width",
        styles={"background": "white", "border": "1px solid #d5d6d6", "border-radius": "8px"},
    )

    return pn.Column(
        pn.pane.Markdown("# Project Summary"),
        kpis,
        map_box,
        sizing_mode="stretch_width",
    )

from __future__ import annotations

import panel as pn
import plotly.express as px


def build_page(service) -> pn.Column:
    wells = service.get_wells()
    display_cols = [c for c in ["well_name", "uwi", "x", "y", "latitude", "longitude", "location", "spud_date"] if c in wells.columns]

    map_df = wells.copy()
    map_df["latitude"] = map_df.get("latitude").fillna(map_df.get("y"))
    map_df["longitude"] = map_df.get("longitude").fillna(map_df.get("x"))
    map_df = map_df.dropna(subset=["latitude", "longitude"]) if not map_df.empty else map_df

    if map_df.empty:
        fig = px.scatter_mapbox(lat=[], lon=[])
        fig.update_layout(title="No well coordinates available", mapbox_style="open-street-map")
    else:
        fig = px.scatter_mapbox(
            map_df,
            lat="latitude",
            lon="longitude",
            hover_name="well_name",
            color="well_name",
            mapbox_style="open-street-map",
            zoom=4,
            height=600,
        )
        fig.update_layout(showlegend=False, margin={"l": 0, "r": 0, "t": 0, "b": 0})

    table = pn.widgets.Tabulator(
        wells[display_cols] if display_cols else wells,
        pagination="local",
        page_size=20,
        layout="fit_data_stretch",
        sizing_mode="stretch_width",
        height=320,
    )

    return pn.Column(
        pn.pane.Markdown("# Wells"),
        pn.Row(
            pn.Column(pn.pane.Plotly(fig, sizing_mode="stretch_both"), sizing_mode="stretch_both"),
            pn.Column(table, sizing_mode="stretch_width", width=650),
            sizing_mode="stretch_both",
        ),
        sizing_mode="stretch_both",
    )

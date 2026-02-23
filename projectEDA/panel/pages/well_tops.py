from __future__ import annotations

import panel as pn
import plotly.express as px


def build_page(service) -> pn.Column:
    tops = service.get_tops()
    wells = service.get_wells()

    table = pn.widgets.Tabulator(
        tops,
        pagination="local",
        page_size=25,
        layout="fit_data_stretch",
        sizing_mode="stretch_both",
        height=500,
    )

    if tops.empty or wells.empty:
        fig = px.scatter_mapbox(lat=[], lon=[])
        fig.update_layout(title="No tops or coordinates available", mapbox_style="open-street-map")
    else:
        per_well = tops.groupby("well_name", as_index=False).size().rename(columns={"size": "tops_count"})
        map_df = wells.merge(per_well, on="well_name", how="left").fillna({"tops_count": 0})
        map_df["latitude"] = map_df.get("latitude").fillna(map_df.get("y"))
        map_df["longitude"] = map_df.get("longitude").fillna(map_df.get("x"))
        map_df = map_df.dropna(subset=["latitude", "longitude"])
        fig = px.scatter_mapbox(
            map_df,
            lat="latitude",
            lon="longitude",
            size="tops_count",
            color="tops_count",
            hover_name="well_name",
            mapbox_style="open-street-map",
            height=600,
        )
        fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})

    return pn.Column(
        pn.pane.Markdown("# Well Tops"),
        pn.Row(pn.pane.Plotly(fig, sizing_mode="stretch_both"), table, sizing_mode="stretch_both"),
        sizing_mode="stretch_both",
    )

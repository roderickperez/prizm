from __future__ import annotations

import panel as pn
import plotly.express as px


def build_page(service) -> pn.Column:
    df = service.get_gantt()
    if df.empty:
        fig = px.scatter(title="No spud dates available")
    else:
        fig = px.timeline(
            df,
            x_start="start",
            x_end="finish",
            y="well_name",
            color="well_name",
            title="Spud Dates Gantt Chart",
        )
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(showlegend=False, height=700, margin={"l": 0, "r": 0, "t": 40, "b": 0})

    return pn.Column(
        pn.pane.Markdown("# Gantt Chart"),
        pn.pane.Markdown("Horizontal chart of spud dates with date on the X-axis."),
        pn.pane.Plotly(fig, sizing_mode="stretch_both"),
        sizing_mode="stretch_both",
    )

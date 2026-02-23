from __future__ import annotations

import panel as pn


def build_page(service) -> pn.Column:
    std_df = service.get_log_standardization()
    if std_df.empty:
        std_df = service.get_logs_catalog()[["log_name", "log_group"]].copy()
        std_df["priority"] = 1

    editor = pn.widgets.Tabulator(
        std_df,
        editors={"log_group": "input", "priority": {"type": "number"}},
        pagination="local",
        page_size=20,
        layout="fit_data_stretch",
        sizing_mode="stretch_both",
        height=500,
    )

    message = pn.pane.Markdown("", sizing_mode="stretch_width")

    def _save(_=None):
        service.save_log_standardization(editor.value)
        message.object = "✅ Log grouping and priorities saved to DuckDB."

    save_btn = pn.widgets.Button(name="Save Grouping and Priority", button_type="primary")
    save_btn.on_click(_save)

    summary = pn.bind(
        lambda df: pn.widgets.Tabulator(
            df.sort_values(["log_group", "priority", "log_name"]),
            disabled=True,
            pagination="local",
            page_size=15,
            layout="fit_data_stretch",
            sizing_mode="stretch_width",
            height=320,
        ),
        editor.param.value,
    )

    return pn.Column(
        pn.pane.Markdown("# Log Standardization"),
        pn.pane.Markdown(
            "Current logs can be organized in groups and ordered by priority. Priority `1` is treated as the standard/first log in each group."
        ),
        pn.Row(save_btn, message),
        editor,
        pn.pane.Markdown("## Current Grouped View"),
        summary,
        sizing_mode="stretch_both",
    )

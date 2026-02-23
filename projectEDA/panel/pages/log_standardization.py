from __future__ import annotations

import panel as pn


def build_page(service) -> pn.Column:
    std_df = service.get_log_standardization()
    if std_df.empty:
        std_df = service.get_logs_catalog()[["log_name", "log_group"]].copy()
        std_df["priority"] = 1

    group_options = service.get_mnemonic_groups()

    editor = pn.widgets.Tabulator(
        std_df,
        editors={
            "log_group": {"type": "list", "values": group_options} if group_options else "input",
            "priority": {"type": "number"},
        },
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

    def _apply_master(_=None):
        catalog = service.get_logs_catalog()
        if catalog.empty:
            message.object = "ℹ️ No logs available to map."
            return
        updated = catalog[["log_name", "mnemonic_group"]].rename(columns={"mnemonic_group": "log_group"}).copy()
        updated["priority"] = (
            updated.groupby("log_group").cumcount() + 1
        )
        editor.value = updated[["log_name", "log_group", "priority"]]
        message.object = "✅ Groups refreshed from mnemonics master aliases."

    save_btn = pn.widgets.Button(name="Save Grouping and Priority", button_type="primary")
    save_btn.on_click(_save)
    master_btn = pn.widgets.Button(name="Apply Mnemonics Master", button_type="default")
    master_btn.on_click(_apply_master)

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
            "Current logs can be organized in groups and ordered by priority. Priority `1` is treated as the standard/first log in each group. Group suggestions come from `mnemonics_master.json`."
        ),
        pn.Row(master_btn, save_btn, message),
        editor,
        pn.pane.Markdown("## Current Grouped View"),
        summary,
        sizing_mode="stretch_both",
    )

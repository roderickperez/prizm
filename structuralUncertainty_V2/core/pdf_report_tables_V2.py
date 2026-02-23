from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@lru_cache(maxsize=8)
def _load_logo(path_str: str):
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        return plt.imread(path)
    except Exception:
        return None


def add_pdf_logo(fig: plt.Figure, logo_path: str | Path) -> None:
    logo_img = _load_logo(str(logo_path))
    if logo_img is None:
        return
    logo_ax = fig.add_axes([0.79, 0.90, 0.18, 0.08])
    logo_ax.imshow(logo_img)
    logo_ax.axis("off")


def style_pdf_table_header(table, header_color: str) -> None:
    header_cells = [cell for (row, _col), cell in table.get_celld().items() if row == 0]
    for cell in header_cells:
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", weight="bold")


def build_pdf_tables_pages(
    report_title: str,
    summary_df: pd.DataFrame,
    realization_df: pd.DataFrame,
    logo_path: str | Path,
    header_color: str,
    rows_per_page: int = 40,
) -> list[plt.Figure]:
    summary_display = summary_df.copy()
    for col in ["P90", "P50", "P10", "Mean"]:
        if col in summary_display.columns:
            summary_display[col] = summary_display[col].astype(float).round(3)

    realization_display = realization_df.copy()
    numeric_cols = realization_display.select_dtypes(include=[np.number]).columns
    realization_display[numeric_cols] = realization_display[numeric_cols].round(3)

    figures: list[plt.Figure] = []

    fig_summary = plt.figure(figsize=(8.27, 11.69))
    add_pdf_logo(fig_summary, logo_path)
    fig_summary.suptitle(f"{report_title} — Summary Table", fontsize=14, fontweight="bold", x=0.45, y=0.97)
    ax_summary = fig_summary.add_subplot(111)
    ax_summary.axis("off")

    summary_tbl = ax_summary.table(
        cellText=summary_display.values,
        colLabels=summary_display.columns,
        loc="center",
        cellLoc="center",
        bbox=[0.04, 0.08, 0.92, 0.78],
    )
    style_pdf_table_header(summary_tbl, header_color)
    summary_tbl.auto_set_font_size(False)
    summary_tbl.set_fontsize(9)
    summary_tbl.scale(1.0, 1.35)
    ax_summary.set_title("Summary Statistics (P90 / P50 / P10 / Mean)", fontsize=11, fontweight="bold", pad=12)
    fig_summary.tight_layout(rect=[0, 0, 1, 0.95])
    figures.append(fig_summary)

    total_rows = len(realization_display)
    total_pages = max(1, int(np.ceil(total_rows / max(rows_per_page, 1))))

    for page_idx in range(total_pages):
        start = page_idx * rows_per_page
        end = min((page_idx + 1) * rows_per_page, total_rows)
        page_df = realization_display.iloc[start:end]

        fig_detail = plt.figure(figsize=(8.27, 11.69))
        add_pdf_logo(fig_detail, logo_path)
        fig_detail.suptitle(
            f"{report_title} — Per-Realization Results ({start + 1}-{end} of {total_rows})",
            fontsize=13,
            fontweight="bold",
            x=0.45,
            y=0.97,
        )
        ax_detail = fig_detail.add_subplot(111)
        ax_detail.axis("off")

        detail_tbl = ax_detail.table(
            cellText=page_df.values,
            colLabels=page_df.columns,
            loc="center",
            cellLoc="center",
            bbox=[0.02, 0.05, 0.96, 0.87],
        )
        style_pdf_table_header(detail_tbl, header_color)
        detail_tbl.auto_set_font_size(False)
        detail_tbl.set_fontsize(7.2)
        detail_tbl.scale(1.0, 1.08)
        ax_detail.set_title(f"Page {page_idx + 1} of {total_pages}", fontsize=10, fontweight="bold", pad=6)

        fig_detail.tight_layout(rect=[0, 0, 1, 0.95])
        figures.append(fig_detail)

    return figures

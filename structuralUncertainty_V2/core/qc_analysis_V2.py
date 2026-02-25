# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .pdf_report_tables_V2 import add_pdf_logo


def model_semivariance(h: np.ndarray, model: str, range_val: float, sill: float, nugget: float) -> np.ndarray:
    h_arr = np.asarray(h, dtype=float)
    safe_range = max(float(range_val), 1e-6)
    safe_sill = max(float(sill), 1e-12)
    safe_nugget = max(float(nugget), 0.0)

    if model == "Gaussian":
        gamma = safe_nugget + safe_sill * (1.0 - np.exp(-3.0 * (h_arr / safe_range) ** 2))
    elif model == "Exponential":
        gamma = safe_nugget + safe_sill * (1.0 - np.exp(-3.0 * h_arr / safe_range))
    else:
        ratio = h_arr / safe_range
        spherical = safe_nugget + safe_sill * (1.5 * ratio - 0.5 * ratio**3)
        gamma = np.where(h_arr <= safe_range, spherical, safe_nugget + safe_sill)
    return gamma


def empirical_variogram(
    values_2d: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    n_lags: int = 14,
    max_pairs: int = 60000,
    seed: int = 42,
) -> pd.DataFrame:
    val = np.asarray(values_2d, dtype=float)
    yy, xx = np.where(np.isfinite(val))
    if len(xx) < 2:
        return pd.DataFrame(columns=["LagDistance_m", "PairCount", "Gamma_empirical"])

    rng = np.random.default_rng(seed)
    n_pairs = min(max_pairs, len(xx) * 4)
    idx_a = rng.integers(0, len(xx), size=n_pairs)
    idx_b = rng.integers(0, len(xx), size=n_pairs)

    xa = x_coords[xx[idx_a]]
    ya = y_coords[yy[idx_a]]
    xb = x_coords[xx[idx_b]]
    yb = y_coords[yy[idx_b]]

    va = val[yy[idx_a], xx[idx_a]]
    vb = val[yy[idx_b], xx[idx_b]]

    dist = np.sqrt((xa - xb) ** 2 + (ya - yb) ** 2)
    gamma = 0.5 * (va - vb) ** 2

    valid = np.isfinite(dist) & np.isfinite(gamma) & (dist > 0)
    dist = dist[valid]
    gamma = gamma[valid]
    if dist.size == 0:
        return pd.DataFrame(columns=["LagDistance_m", "PairCount", "Gamma_empirical"])

    max_dist = float(np.percentile(dist, 95.0))
    if max_dist <= 0:
        return pd.DataFrame(columns=["LagDistance_m", "PairCount", "Gamma_empirical"])

    bins = np.linspace(0.0, max_dist, n_lags + 1)
    lag_center = 0.5 * (bins[:-1] + bins[1:])
    pair_count = np.zeros(n_lags, dtype=int)
    gamma_mean = np.full(n_lags, np.nan, dtype=float)

    bin_ids = np.digitize(dist, bins) - 1
    for lag_idx in range(n_lags):
        mask = bin_ids == lag_idx
        if not np.any(mask):
            continue
        pair_count[lag_idx] = int(np.sum(mask))
        gamma_mean[lag_idx] = float(np.nanmean(gamma[mask]))

    out = pd.DataFrame(
        {
            "LagDistance_m": lag_center,
            "PairCount": pair_count,
            "Gamma_empirical": gamma_mean,
        }
    )
    return out[out["PairCount"] > 0].reset_index(drop=True)


def _covariance_diagnostics(cov_df: pd.DataFrame) -> pd.DataFrame:
    if cov_df.empty:
        return pd.DataFrame(
            {
                "Metric": [
                    "CovarianceConditionNumber",
                    "CovarianceMinEigen",
                    "CovarianceMaxEigen",
                    "CovarianceOffDiagonalMean",
                    "CovarianceOffDiagonalStd",
                ],
                "Value": [np.nan, np.nan, np.nan, np.nan, np.nan],
            }
        )

    cov = cov_df.to_numpy(dtype=float)
    eigvals = np.linalg.eigvalsh(cov)
    off_diag = cov[np.triu_indices(cov.shape[0], k=1)] if cov.shape[0] > 1 else np.array([], dtype=float)

    return pd.DataFrame(
        {
            "Metric": [
                "CovarianceConditionNumber",
                "CovarianceMinEigen",
                "CovarianceMaxEigen",
                "CovarianceOffDiagonalMean",
                "CovarianceOffDiagonalStd",
            ],
            "Value": [
                float(np.linalg.cond(cov)),
                float(np.min(eigvals)),
                float(np.max(eigvals)),
                float(np.mean(off_diag)) if off_diag.size else np.nan,
                float(np.std(off_diag)) if off_diag.size else np.nan,
            ],
        }
    )


def _coherence_checks(
    velocity_stack: np.ndarray,
    depth_stack: np.ndarray,
    final_depth_map: np.ndarray,
    trap_masks: np.ndarray,
) -> pd.DataFrame:
    vel = np.asarray(velocity_stack, dtype=float)
    dep = np.asarray(depth_stack, dtype=float)
    dep_final = np.asarray(final_depth_map, dtype=float)
    masks = np.asarray(trap_masks, dtype=bool)

    finite_velocity_pct = float(np.isfinite(vel).mean() * 100.0)
    finite_depth_pct = float(np.isfinite(dep).mean() * 100.0)
    closure_frequency_pct = float(np.mean(np.any(masks.reshape(masks.shape[0], -1), axis=1)) * 100.0) if masks.size else np.nan

    rows = [
        ("FiniteVelocityCoverage_pct", finite_velocity_pct, "Target >= 99%"),
        ("FiniteDepthCoverage_pct", finite_depth_pct, "Target >= 99%"),
        ("VelocityMin_mps", float(np.nanmin(vel)), ">= 1200 m/s safeguard"),
        ("VelocityMax_mps", float(np.nanmax(vel)), "Check for realistic AV"),
        ("FinalDepthMin_m", float(np.nanmin(dep_final)), "Interpretation specific"),
        ("FinalDepthMax_m", float(np.nanmax(dep_final)), "Interpretation specific"),
        ("ClosurePresenceFrequency_pct", closure_frequency_pct, "Avoid 0% unless open structure"),
    ]

    return pd.DataFrame(rows, columns=["Metric", "Value", "QC_Note"])


def _recommended_variogram_parameters(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    empirical_df: pd.DataFrame,
    current_range: float,
    current_sill: float,
    current_nugget: float,
) -> pd.DataFrame:
    x_span = float(np.max(x_coords) - np.min(x_coords)) if len(x_coords) else np.nan
    y_span = float(np.max(y_coords) - np.min(y_coords)) if len(y_coords) else np.nan
    diag = float(np.sqrt(x_span**2 + y_span**2)) if np.isfinite(x_span) and np.isfinite(y_span) else np.nan

    rec_range = np.clip(0.25 * diag if np.isfinite(diag) else current_range, 500.0, 5000.0)
    if empirical_df.empty:
        empirical_sill = np.nan
        rec_sill = max(float(current_sill), 1e-3)
    else:
        empirical_sill = float(np.nanmedian(empirical_df["Gamma_empirical"].tail(min(4, len(empirical_df)))))
        rec_sill = float(np.clip(empirical_sill if np.isfinite(empirical_sill) else current_sill, 0.1, 5.0))

    rec_nugget = 0.1

    return pd.DataFrame(
        {
            "Parameter": ["Range_m", "Sill", "Nugget"],
            "Current": [float(current_range), float(current_sill), float(current_nugget)],
            "Recommended": [float(rec_range), float(rec_sill), float(rec_nugget)],
            "Reason": [
                "25% of map diagonal is a robust default correlation distance",
                "Tail median of empirical semivariogram stabilizes sill estimate",
                "Default engineering-safe microscale noise floor",
            ],
        }
    )


def build_geostat_qc_bundle(
    twt_ms: np.ndarray,
    base_velocity: np.ndarray,
    velocity_stack: np.ndarray,
    depth_stack: np.ndarray,
    final_depth_map: np.ndarray,
    trap_masks: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    wells_cov_df: pd.DataFrame,
    model: str,
    range_val: float,
    sill: float,
    nugget: float,
) -> dict[str, pd.DataFrame]:
    _ = twt_ms
    empirical_df = empirical_variogram(base_velocity, x_coords, y_coords)

    max_h = float(max(np.max(x_coords) - np.min(x_coords), np.max(y_coords) - np.min(y_coords))) if len(x_coords) and len(y_coords) else 1.0
    lag_theory = np.linspace(0.0, max(max_h, 1.0), 60)
    theory_df = pd.DataFrame(
        {
            "LagDistance_m": lag_theory,
            "Gamma_theoretical": model_semivariance(lag_theory, model, range_val, sill, nugget),
        }
    )

    cov_diag_df = _covariance_diagnostics(wells_cov_df)
    coherence_df = _coherence_checks(velocity_stack, depth_stack, final_depth_map, trap_masks)
    recommendation_df = _recommended_variogram_parameters(
        np.asarray(x_coords, dtype=float),
        np.asarray(y_coords, dtype=float),
        empirical_df,
        current_range=float(range_val),
        current_sill=float(sill),
        current_nugget=float(nugget),
    )

    return {
        "empirical_variogram_df": empirical_df,
        "theoretical_variogram_df": theory_df,
        "covariance_diagnostics_df": cov_diag_df,
        "coherence_checks_df": coherence_df,
        "recommended_parameters_df": recommendation_df,
    }


def build_pdf_geostat_qc_pages(
    report_title: str,
    logo_path: str | Path,
    wells_cov_df: pd.DataFrame,
    qc_bundle: dict[str, pd.DataFrame],
) -> list[plt.Figure]:
    figures: list[plt.Figure] = []

    fig_cov = plt.figure(figsize=(11.69, 8.27))
    add_pdf_logo(fig_cov, logo_path)
    fig_cov.suptitle(f"{report_title} - Covariance QC", fontsize=13, fontweight="bold", x=0.45, y=0.97)
    ax_cov = fig_cov.add_subplot(1, 2, 1)
    ax_txt = fig_cov.add_subplot(1, 2, 2)

    if wells_cov_df.empty:
        ax_cov.text(0.5, 0.5, "No wells covariance available", ha="center", va="center")
        ax_cov.axis("off")
    else:
        cov_np = wells_cov_df.to_numpy(dtype=float)
        im = ax_cov.imshow(cov_np, cmap="viridis", aspect="auto")
        ax_cov.set_title("Wells Covariance Matrix")
        ax_cov.set_xticks(range(len(wells_cov_df.columns)))
        ax_cov.set_yticks(range(len(wells_cov_df.index)))
        ax_cov.set_xticklabels(wells_cov_df.columns, rotation=45, ha="right", fontsize=7)
        ax_cov.set_yticklabels(wells_cov_df.index, fontsize=7)
        fig_cov.colorbar(im, ax=ax_cov, fraction=0.046, pad=0.04)

    cov_diag_df = qc_bundle.get("covariance_diagnostics_df", pd.DataFrame())
    ax_txt.axis("off")
    if cov_diag_df.empty:
        ax_txt.text(0.02, 0.98, "No covariance diagnostics", va="top")
    else:
        lines = ["Covariance Diagnostics", ""] + [f"- {row.Metric}: {row.Value:.6g}" for row in cov_diag_df.itertuples(index=False)]
        ax_txt.text(0.02, 0.98, "\n".join(lines), va="top", fontsize=10)

    fig_cov.tight_layout(rect=[0, 0, 1, 0.95])
    figures.append(fig_cov)

    fig_vario = plt.figure(figsize=(11.69, 8.27))
    add_pdf_logo(fig_vario, logo_path)
    fig_vario.suptitle(f"{report_title} - Variogram QC", fontsize=13, fontweight="bold", x=0.45, y=0.97)
    ax_v = fig_vario.add_subplot(1, 2, 1)
    ax_r = fig_vario.add_subplot(1, 2, 2)

    empirical_df = qc_bundle.get("empirical_variogram_df", pd.DataFrame())
    theory_df = qc_bundle.get("theoretical_variogram_df", pd.DataFrame())
    if not theory_df.empty:
        ax_v.plot(theory_df["LagDistance_m"], theory_df["Gamma_theoretical"], color="navy", linewidth=2, label="Theoretical")
    if not empirical_df.empty:
        ax_v.scatter(empirical_df["LagDistance_m"], empirical_df["Gamma_empirical"], color="darkorange", s=18, alpha=0.8, label="Empirical")
    ax_v.set_xlabel("Lag distance (m)")
    ax_v.set_ylabel("Semivariance")
    ax_v.set_title("Empirical vs Theoretical Variogram")
    ax_v.grid(alpha=0.25)
    ax_v.legend(loc="best", fontsize=9)

    rec_df = qc_bundle.get("recommended_parameters_df", pd.DataFrame())
    ax_r.axis("off")
    if rec_df.empty:
        ax_r.text(0.02, 0.98, "No recommendations", va="top")
    else:
        lines = ["Parameter Recommendations", ""]
        for row in rec_df.itertuples(index=False):
            lines.append(f"- {row.Parameter}: current={row.Current:.4g}, recommended={row.Recommended:.4g}")
            lines.append(f"  reason: {row.Reason}")
        ax_r.text(0.02, 0.98, "\n".join(lines), va="top", fontsize=10)

    fig_vario.tight_layout(rect=[0, 0, 1, 0.95])
    figures.append(fig_vario)

    fig_coh = plt.figure(figsize=(11.69, 8.27))
    add_pdf_logo(fig_coh, logo_path)
    fig_coh.suptitle(f"{report_title} - Engineering Coherence QC", fontsize=13, fontweight="bold", x=0.45, y=0.97)
    ax_c = fig_coh.add_subplot(111)
    ax_c.axis("off")
    coh_df = qc_bundle.get("coherence_checks_df", pd.DataFrame())
    if coh_df.empty:
        ax_c.text(0.02, 0.98, "No coherence checks", va="top")
    else:
        table = ax_c.table(
            cellText=coh_df.round(6).values,
            colLabels=coh_df.columns,
            loc="center",
            cellLoc="center",
            bbox=[0.03, 0.07, 0.94, 0.84],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.2)

    fig_coh.tight_layout(rect=[0, 0, 1, 0.95])
    figures.append(fig_coh)

    return figures
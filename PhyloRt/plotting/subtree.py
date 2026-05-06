"""Per-subtree R summaries and breakpoint utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .rt import DEFAULT_BUCKETS, filter_to_bucket

def _alpha_norm(series: pd.Series) -> np.ndarray:
    return np.clip(series.astype(float).fillna(0.0), 0.1, 1.0).to_numpy(float)

def _plot_one_df_on_ax(
    ax,
    df: pd.DataFrame,
    start_col: str,
    end_col: str,
    weight_col: str,
    change_threshold: float,
    lw: float = 1.0,
):
    required = ["subtree_size", start_col, end_col, "R1", "R2", "change", "breakpoint", weight_col]
    if any(column not in df.columns for column in required):
        return ax

    starts = pd.to_datetime(df[start_col], errors="coerce")
    ends = pd.to_datetime(df[end_col], errors="coerce")
    starts_num = mdates.date2num(starts)
    ends_num = mdates.date2num(ends)
    alphas = _alpha_norm(pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0))

    color_single = "#662d91"
    color_before = "#3182bd"
    color_after = "#e6550d"

    for idx, (_, row) in enumerate(df.iterrows()):
        start_num = starts_num[idx]
        end_num = ends_num[idx]
        if not (np.isfinite(start_num) and np.isfinite(end_num)):
            continue
        if end_num < start_num:
            start_num, end_num = end_num, start_num

        r1 = pd.to_numeric(row.get("R1"), errors="coerce")
        r2 = pd.to_numeric(row.get("R2"), errors="coerce")
        change = pd.to_numeric(row.get("change"), errors="coerce")
        bp_frac = pd.to_numeric(row.get("breakpoint"), errors="coerce")
        if not np.isfinite(r1):
            continue
        if not np.isfinite(bp_frac):
            bp_frac = 0.5
        bp_frac = min(max(float(bp_frac), 0.0), 1.0)

        total = end_num - start_num
        bp_time = start_num + bp_frac * total if total != 0 else start_num
        two_stage = bool(np.isfinite(change) and change >= change_threshold)
        if two_stage and (bp_frac < 0.2 or bp_frac > 0.8):
            continue

        alpha = float(alphas[idx]) if idx < len(alphas) else 0.6
        if total == 0:
            ax.plot([start_num - 0.5, start_num + 0.5], [r1, r1], color=color_single, linewidth=lw, alpha=alpha)
        elif not two_stage:
            if np.isfinite(change) and change < 0.1:
                ax.plot([start_num, end_num], [r1, r1], color=color_single, linewidth=lw, alpha=alpha)
        elif np.isfinite(r2):
            ax.plot([start_num, bp_time], [r1, r1], color=color_before, linewidth=lw, alpha=alpha)
            ax.plot([bp_time, end_num], [r2, r2], color=color_after, linewidth=lw, alpha=alpha)
    return ax

def smooth_hist_curve(xvals, kernel_sigma=1.0, normalize_max=True):
    xvals = np.asarray(xvals, dtype=float)
    if len(xvals) == 0:
        return np.array([]), np.array([])
    nbins = 200
    x_min, x_max = np.min(xvals), np.max(xvals)
    pad = max(0.01 * (x_max - x_min if x_max > x_min else 1.0), 1e-6)
    bins = np.linspace(x_min - pad, x_max + pad, nbins + 1)
    hist, edges = np.histogram(xvals, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dx = centers[1] - centers[0] if centers.size > 1 else 1.0
    sigma_bins = kernel_sigma / dx if kernel_sigma > 0 and dx > 0 else 1.0
    sigma_bins = min(max(sigma_bins, 1.0), max(1.0, nbins / 4.0))
    half = int(min(max(3, np.ceil(4 * sigma_bins)), max(3, nbins // 2)))
    kx = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (kx / sigma_bins) ** 2)
    kernel = kernel / np.sum(kernel)
    smooth = np.convolve(hist, kernel, mode="same")
    if smooth.shape[0] != centers.shape[0]:
        offset = max(0, (smooth.shape[0] - centers.shape[0]) // 2)
        smooth = smooth[offset : offset + centers.shape[0]]
    if normalize_max and smooth.size:
        maxv = np.nanmax(smooth)
        if maxv > 0:
            smooth = smooth / maxv
    return centers, smooth

def breakpoint_times(
    df: pd.DataFrame,
    start_col: str = "node_age",
    end_col: str = "last_sample_date",
    change_threshold: float = 0.9,
) -> np.ndarray:
    if df.empty:
        return np.array([], dtype=float)
    change = pd.to_numeric(df.get("change", pd.Series(index=df.index)), errors="coerce")
    mask = change >= change_threshold
    mask &= pd.to_datetime(df[start_col], errors="coerce").notna()
    mask &= pd.to_datetime(df[end_col], errors="coerce").notna()
    if not mask.any():
        return np.array([], dtype=float)

    times = []
    starts = pd.to_datetime(df.loc[mask, start_col], errors="coerce")
    ends = pd.to_datetime(df.loc[mask, end_col], errors="coerce")
    bps = pd.to_numeric(df.loc[mask, "breakpoint"], errors="coerce")
    for start, end, bp_frac in zip(starts, ends, bps):
        if pd.isna(start) or pd.isna(end) or pd.isna(bp_frac):
            continue
        if end < start:
            start, end = end, start
        frac = min(max(float(bp_frac), 0.0), 1.0)
        bp = start + pd.to_timedelta(frac * (end - start).total_seconds(), unit="s")
        times.append(pd.to_datetime(bp))
    if not times:
        return np.array([], dtype=float)
    return mdates.date2num(np.array(times, dtype="datetime64[ns]"))

def _plot_subtree_r(
    df: pd.DataFrame,
    out_path: Path,
    start_col: str,
    end_col: str,
    weight_col: str,
    change_threshold: float,
    show_change_points: bool = False,
    figsize=(11.69, 8.27),
) -> plt.Figure:
    combined = df.copy()
    combined[start_col] = pd.to_datetime(combined[start_col], errors="coerce")
    combined[end_col] = pd.to_datetime(combined[end_col], errors="coerce")

    starts_num = mdates.date2num(combined[start_col])
    ends_num = mdates.date2num(combined[end_col])
    valid = np.isfinite(starts_num) & np.isfinite(ends_num)
    if valid.any():
        x_min = np.nanmin(np.minimum(starts_num[valid], ends_num[valid]))
        x_max = np.nanmax(np.maximum(starts_num[valid], ends_num[valid]))
    else:
        today_num = mdates.date2num(pd.Timestamp.today())
        x_min, x_max = today_num - 60, today_num + 60

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
    axes = axes.reshape((2, 2))
    yticks = [0, 1, 2, 3, 4, 5, 6]

    for idx, (label, lo, hi) in enumerate(DEFAULT_BUCKETS):
        row_idx, col_idx = divmod(idx, 2)
        ax = axes[row_idx, col_idx]
        df_bucket = filter_to_bucket(combined, lo, hi)
        _plot_one_df_on_ax(ax, df_bucket, start_col, end_col, weight_col, change_threshold)

        if show_change_points:
            bp_times_num = breakpoint_times(df_bucket, start_col, end_col, change_threshold)
            if bp_times_num.size:
                centers, smooth = smooth_hist_curve(bp_times_num, kernel_sigma=5, normalize_max=True)
                if len(centers) and len(smooth):
                    ax2 = ax.twinx()
                    ax2.set_xlim(ax.get_xlim())
                    in_mask = (centers >= x_min) & (centers <= x_max)
                    if in_mask.any():
                        ax2.plot(centers[in_mask], smooth[in_mask], color="darkred", linestyle="--", linewidth=1.2)
                    else:
                        ax2.plot(centers, smooth, color="darkred", linestyle="--", linewidth=1.2)
                    ax2.set_ylim(0, 1)
                    ax2.set_yticks([])
                    ax2.set_ylabel("")
                    ax2.patch.set_alpha(0.0)

        ax.set_title(label, fontsize="medium")
        ax.set_ylabel("R" if col_idx == 0 else "")
        ax.set_xlabel("Date" if row_idx == 1 else "")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, 6)
        ax.set_yticks(yticks)
        ax.grid(True, linestyle=":", alpha=0.25)
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        for tick in ax.get_xticklabels():
            tick.set_rotation(30)
            tick.set_ha("right")

    handles = [
        Line2D([0], [0], color="#662d91", lw=1.5, label=r"$R_{NC}$"),
        Line2D([0], [0], color="#3182bd", lw=1.5, label=r"$R_{C1}$"),
        Line2D([0], [0], color="#e6550d", lw=1.5, label=r"$R_{C2}$"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.035), ncol=3, frameon=False, fontsize="small")
    fig.tight_layout(rect=[0, 0.03, 1.0, 0.96])
    fig.subplots_adjust(hspace=0.12, wspace=0.12)
    fig.savefig(out_path, bbox_inches="tight")
    return fig

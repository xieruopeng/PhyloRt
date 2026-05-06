"""Rt time-series aggregation and plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_BUCKETS = [
    ("21-50", 21, 50),
    ("51-100", 51, 100),
    ("101-200", 101, 200),
    ("201-400", 201, 400),
]


def _parse_rt_quantiles(rt_quantiles: str | tuple[float, float] | list[float]) -> tuple[float, float]:
    if isinstance(rt_quantiles, str):
        parts = [item.strip() for item in rt_quantiles.split(",") if item.strip()]
        if len(parts) != 2:
            raise ValueError("rt_quantiles must contain exactly two comma-separated percentiles, e.g. 25,75")
        values = [float(parts[0]), float(parts[1])]
    else:
        values = [float(item) for item in rt_quantiles]
        if len(values) != 2:
            raise ValueError("rt_quantiles must contain exactly two percentile values")

    low, high = values
    if not (0 <= low < high <= 100):
        raise ValueError("rt_quantiles must satisfy 0 <= low < high <= 100")
    return low / 100.0, high / 100.0


def _quantile_label(percentile: float) -> str:
    if float(percentile).is_integer():
        return str(int(percentile))
    return f"{percentile:g}".replace(".", "p")

def weighted_quantile(values, quantiles, sample_weight):
    values = np.asarray(values, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    weights = np.asarray(sample_weight, dtype=float)
    if values.size == 0:
        return np.full_like(quantiles, np.nan, dtype=float)

    sorter = np.argsort(values)
    values_sorted = values[sorter]
    weights_sorted = weights[sorter]
    cdf = np.cumsum(weights_sorted)
    if cdf[-1] <= 0:
        return np.full_like(quantiles, np.nan, dtype=float)
    cdf = cdf / cdf[-1]
    return np.interp(quantiles, cdf, values_sorted)

def filter_to_bucket(df: pd.DataFrame, lo: int, hi: int, col: str = "subtree_size") -> pd.DataFrame:
    if col not in df.columns:
        return df.iloc[0:0]
    return df[pd.to_numeric(df[col], errors="coerce").between(lo, hi, inclusive="both")]

def _bucket_defs(buckets=None):
    if buckets is None:
        buckets = DEFAULT_BUCKETS
    return [(label, lambda x, lo=lo, hi=hi: (x >= lo) & (x <= hi)) for label, lo, hi in buckets]

def _require_node_weight(df: pd.DataFrame) -> str:
    if "node_weight" not in df.columns:
        raise ValueError("Metadata is missing required node_weight column. Regenerate predictions with this PhyloRt version.")
    return "node_weight"

def compute_bucket_time_series_piecewise_weighted_quantile_dates(
    df: pd.DataFrame,
    buckets=None,
    start_col: str = "node_age",
    end_col: str = "last_sample_date",
    r1_col: str = "R1",
    r2_col: str = "R2",
    change_col: str = "change",
    bp_col: str = "breakpoint",
    weight_col: str = "node_weight",
    change_threshold: float = 0.9,
    bin_days: int = 14,
    distance_function: str = "exponential",
    recency_strength: float = 0.0,
    weight_by_fraction: bool = True,
    date_mean_col: str | None = None,
    rt_quantiles: tuple[float, float] = (0.25, 0.75),
):
    bucket_defs = _bucket_defs(buckets)
    df = df.copy()
    df[start_col] = pd.to_datetime(df[start_col], errors="coerce")
    df[end_col] = pd.to_datetime(df[end_col], errors="coerce")
    if date_mean_col is not None and date_mean_col in df.columns:
        df["span_mean"] = pd.to_datetime(df[date_mean_col], errors="coerce")

    for column in (r1_col, r2_col, change_col, bp_col, weight_col, "subtree_size"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    valid = df[start_col].notna() & df[end_col].notna()
    empty = {label: (np.array([]), np.array([]), np.array([]), np.array([], dtype=int)) for label, _ in bucket_defs}
    if not valid.any():
        return pd.DatetimeIndex([]), np.array([]), empty

    df = df.loc[valid].copy().reset_index(drop=True)
    s = df[start_col].to_numpy(dtype="datetime64[ns]")
    e = df[end_col].to_numpy(dtype="datetime64[ns]")
    if "span_mean" in df.columns:
        m_dt = pd.to_datetime(df["span_mean"], errors="coerce").to_numpy(dtype="datetime64[ns]")
    else:
        mid_ns = (s.astype(np.int64) + e.astype(np.int64)) // 2
        m_dt = mid_ns.astype("datetime64[ns]")

    r1 = pd.to_numeric(df[r1_col], errors="coerce").to_numpy(float)
    r2 = pd.to_numeric(df[r2_col], errors="coerce").to_numpy(float)
    change = pd.to_numeric(df[change_col], errors="coerce").to_numpy(float)
    breakpoint_frac = pd.to_numeric(df[bp_col], errors="coerce").to_numpy(float)
    weights = pd.to_numeric(df[weight_col], errors="coerce").to_numpy(float)
    weights = np.where(np.isfinite(weights), np.maximum(weights, 0.0), 0.0)
    if "replicate_id" in df.columns:
        replicate_ids = df["replicate_id"].fillna("replicate_0001").astype(str).to_numpy()
    else:
        replicate_ids = np.repeat("replicate_0001", len(df))

    t_min = pd.to_datetime(s.min())
    t_max = pd.to_datetime(e.max())
    edges = pd.date_range(
        start=t_min,
        end=t_max + pd.Timedelta(days=1),
        freq=pd.Timedelta(days=bin_days),
    )
    if len(edges) < 2 or edges[-1] < t_max:
        edges = edges.append(pd.DatetimeIndex([t_max]))

    starts = edges[:-1].to_numpy(dtype="datetime64[ns]")
    ends = edges[1:].to_numpy(dtype="datetime64[ns]")
    bin_centers_ns = (starts.astype(np.int64) + ends.astype(np.int64)) // 2
    bin_centers = bin_centers_ns.astype("datetime64[ns]")
    nbins = len(bin_centers)

    per_bucket = {
        label: {
            "vals": [[] for _ in range(nbins)],
            "wts": [[] for _ in range(nbins)],
            "reps": [[] for _ in range(nbins)],
        }
        for label, _fn in bucket_defs
    }

    def overlap_days(start_ts, end_ts, a_ts, b_ts):
        left = max(start_ts, a_ts)
        right = min(end_ts, b_ts)
        if right <= left:
            return 0.0
        return float((right - left).astype("timedelta64[ns]").astype(float) / 1e9 / 86400.0)

    for idx in range(len(s)):
        start_ts = s[idx]
        end_ts = e[idx]
        if np.isnat(start_ts) or np.isnat(end_ts) or weights[idx] <= 0:
            continue
        if end_ts < start_ts:
            start_ts, end_ts = end_ts, start_ts

        seg_len = (end_ts - start_ts).astype("timedelta64[ns]").astype(float) / 1e9 / 86400.0
        if seg_len < 0:
            continue

        bp_frac = breakpoint_frac[idx] if np.isfinite(breakpoint_frac[idx]) else 0.5
        bp_frac = min(max(float(bp_frac), 0.0), 1.0)
        bp_time = start_ts + np.timedelta64(int(bp_frac * seg_len * 86400), "s") if seg_len > 0 else start_ts
        two_stage = bool(np.isfinite(change[idx]) and change[idx] >= change_threshold)
        bin_indices = np.where((starts < end_ts) & (ends > start_ts))[0]
        if bin_indices.size == 0:
            continue

        mid_ts = m_dt[idx] if not np.isnat(m_dt[idx]) else start_ts + (end_ts - start_ts) / 2
        del mid_ts

        subtree_size = df.loc[idx, "subtree_size"] if "subtree_size" in df.columns else np.nan
        if pd.isna(subtree_size):
            continue

        bucket_label = None
        for label, fn in bucket_defs:
            if fn(subtree_size):
                bucket_label = label
                break
        if bucket_label is None:
            continue

        for bin_idx in bin_indices:
            a = starts[bin_idx]
            b = ends[bin_idx]
            if weight_by_fraction:
                overlap_len = overlap_days(start_ts, end_ts, a, b)
                bin_width_days = (b - a).astype("timedelta64[ns]").astype(float) / 1e9 / 86400.0
                overlap_frac = overlap_len / bin_width_days if bin_width_days > 0 else 0.0
            else:
                overlap_len = seg_len
                overlap_frac = 1.0
            if overlap_frac <= 0:
                continue

            bin_mid_ts = np.datetime64(bin_centers[bin_idx])
            if (not two_stage) or seg_len == 0:
                if change[idx] <= (1 - change_threshold):
                    value = r1[idx]
                    outside_frac = max(0.0, (seg_len - overlap_len) / seg_len) if seg_len > 0 else 0.0
                else:
                    continue
            elif bin_mid_ts <= bp_time:
                value = r1[idx]
                denom = (bp_time - start_ts).astype("timedelta64[ns]").astype(float) / 1e9 / 86400.0
                outside_frac = max(0.0, (denom - overlap_len) / denom) if denom > 0 else 0.0
            else:
                value = r2[idx]
                denom = (end_ts - bp_time).astype("timedelta64[ns]").astype(float) / 1e9 / 86400.0
                outside_frac = max(0.0, (denom - overlap_len) / denom) if denom > 0 else 0.0

            if not np.isfinite(value):
                continue

            if distance_function == "linear":
                recency_w = np.clip(1.0 - recency_strength * outside_frac, 0.01, 1.0)
            elif distance_function == "exponential":
                recency_w = np.exp(-recency_strength * outside_frac)
            elif distance_function == "gaussian":
                recency_w = np.exp(-recency_strength * (outside_frac**2))
            elif distance_function == "power":
                recency_w = 1.0 / (1.0 + recency_strength * outside_frac) ** 2
            elif distance_function == "step":
                recency_w = 1.0 if outside_frac <= 0.5 else 1.0 / max(recency_strength, 1.0)
            else:
                raise ValueError(f"Unknown distance_function: {distance_function}")

            eff_w = weights[idx] * overlap_frac * (recency_w**2)
            if eff_w <= 0:
                continue
            per_bucket[bucket_label]["vals"][bin_idx].append(value)
            per_bucket[bucket_label]["wts"][bin_idx].append(eff_w)
            per_bucket[bucket_label]["reps"][bin_idx].append(replicate_ids[idx])

    final = {}
    for label, _fn in bucket_defs:
        means = np.full(nbins, np.nan, dtype=float)
        lo = np.full(nbins, np.nan, dtype=float)
        hi = np.full(nbins, np.nan, dtype=float)
        n_replicates = np.zeros(nbins, dtype=int)
        for bin_idx in range(nbins):
            vals = np.asarray(per_bucket[label]["vals"][bin_idx], dtype=float)
            wts = np.asarray(per_bucket[label]["wts"][bin_idx], dtype=float)
            reps = np.asarray(per_bucket[label]["reps"][bin_idx], dtype=object)
            if vals.size == 0 or np.sum(wts) <= 0:
                continue
            balanced_wts = np.zeros_like(wts, dtype=float)
            for replicate in np.unique(reps):
                mask = reps == replicate
                replicate_total = float(np.sum(wts[mask]))
                if replicate_total <= 0:
                    continue
                balanced_wts[mask] = wts[mask] / replicate_total
            positive = balanced_wts > 0
            if not positive.any():
                continue
            n_replicates[bin_idx] = int(len(np.unique(reps[positive])))
            means[bin_idx] = np.average(vals[positive], weights=balanced_wts[positive])
            q = weighted_quantile(vals[positive], [rt_quantiles[0], rt_quantiles[1]], balanced_wts[positive])
            lo[bin_idx], hi[bin_idx] = q[0], q[1]
        final[label] = (means, lo, hi, n_replicates)

    return edges, bin_centers, final

def _plot_bucket_rt(
    df: pd.DataFrame,
    out_path: Path,
    weight_col: str,
    change_threshold: float,
    bin_days: int,
    recency_strength: float,
    rt_quantiles: str | tuple[float, float] = "25,75",
    start_col: str = "node_age",
    end_col: str = "last_sample_date",
    figsize=(12, 9),
) -> plt.Figure:
    quantiles = _parse_rt_quantiles(rt_quantiles)
    edges, _bin_centers, final = compute_bucket_time_series_piecewise_weighted_quantile_dates(
        df,
        start_col=start_col,
        end_col=end_col,
        change_threshold=change_threshold,
        bin_days=bin_days,
        distance_function="exponential",
        recency_strength=recency_strength,
        weight_col=weight_col,
        weight_by_fraction=True,
        rt_quantiles=quantiles,
    )

    fig, axes = plt.subplots(2, 2, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()
    colors = [plt.get_cmap("tab10")(idx) for idx in range(len(DEFAULT_BUCKETS))]
    edges_num = mdates.date2num(edges.to_pydatetime()) if len(edges) else np.array([])

    for ax, (bucket, color) in zip(axes_flat, zip(DEFAULT_BUCKETS, colors)):
        label = bucket[0]
        mean, lo, hi, _n_replicates = final[label]
        if len(mean) == 0 or np.all(np.isnan(mean)) or len(edges_num) == 0:
            ax.set_title(label + " (no data)")
            ax.set_ylim(0, 6)
            ax.grid(True, linestyle=":", alpha=0.35)
            continue

        mean_edges = np.append(mean, mean[-1])
        lo_edges = np.append(lo, lo[-1])
        hi_edges = np.append(hi, hi[-1])
        ax.step(edges_num, mean_edges, where="post", label=label, color=color, lw=2)
        ax.fill_between(edges_num, lo_edges, hi_edges, step="post", color=color, alpha=0.25, linewidth=0.0)
        ax.set_ylim(0, 6)
        ax.set_title(label)
        ax.set_xlabel("Date")
        ax.set_ylabel("R")
        ax.grid(True, linestyle=":", alpha=0.35)
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        for tick in ax.get_xticklabels():
            tick.set_rotation(30)
            tick.set_ha("right")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, bbox_inches="tight")
    return fig

def rt_summary_table(
    df: pd.DataFrame,
    weight_col: str = "node_weight",
    change_threshold: float = 0.9,
    bin_days: int = 14,
    recency_strength: float = 0.0,
    rt_quantiles: str | tuple[float, float] = "25,75",
    start_col: str = "node_age",
    end_col: str = "last_sample_date",
) -> pd.DataFrame:
    """Return the Rt time-bin summary used by the default Rt plot."""
    quantiles = _parse_rt_quantiles(rt_quantiles)
    low_pct, high_pct = quantiles[0] * 100.0, quantiles[1] * 100.0
    low_label = _quantile_label(low_pct)
    high_label = _quantile_label(high_pct)
    edges, _bin_centers, final = compute_bucket_time_series_piecewise_weighted_quantile_dates(
        df,
        start_col=start_col,
        end_col=end_col,
        change_threshold=change_threshold,
        bin_days=bin_days,
        distance_function="exponential",
        recency_strength=recency_strength,
        weight_col=weight_col,
        weight_by_fraction=True,
        rt_quantiles=quantiles,
    )
    columns = [
        "subtree_size",
        "bin_start",
        "bin_end",
        "Rt",
        f"Rt_q{low_label}",
        f"Rt_q{high_label}",
    ]
    if len(edges) < 2:
        return pd.DataFrame(columns=columns)

    rows = []
    for label, _lo, _hi in DEFAULT_BUCKETS:
        mean, q25, q75, _n_replicates = final[label]
        for idx in range(len(mean)):
            values = (mean[idx], q25[idx], q75[idx])
            if not any(np.isfinite(value) for value in values):
                continue
            rows.append(
                {
                    "subtree_size": label,
                    "bin_start": pd.Timestamp(edges[idx]).strftime("%Y-%m-%d"),
                    "bin_end": pd.Timestamp(edges[idx + 1]).strftime("%Y-%m-%d"),
                    "Rt": round(float(mean[idx]), 4),
                    f"Rt_q{low_label}": round(float(q25[idx]), 4),
                    f"Rt_q{high_label}": round(float(q75[idx]), 4),
                }
            )
    return pd.DataFrame(rows, columns=columns)

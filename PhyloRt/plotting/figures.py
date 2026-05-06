"""Public plotting workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd

from .rt import _plot_bucket_rt, _require_node_weight, rt_summary_table
from .subtree import _plot_subtree_r

def _coerce_metadata(metadata: str | Path | pd.DataFrame) -> tuple[pd.DataFrame, Path | None]:
    if isinstance(metadata, pd.DataFrame):
        return metadata.copy(), None
    path = Path(metadata)
    return pd.read_csv(path), path

def plot(
    metadata: str | Path | pd.DataFrame,
    out_dir: str | Path | None = None,
    outputs: Literal["both", "rt", "subtree_r"] = "both",
    start_col: str = "node_age",
    end_col: str = "last_sample_date",
    change_threshold: float = 0.9,
    bin_days: int = 14,
    recency_strength: float = 0.0,
    rt_quantiles: str | tuple[float, float] = "25,75",
    show_change_points: bool = False,
) -> dict[str, Path]:
    """Generate PhyloRt plot PDFs from a metadata table."""
    df, metadata_path = _coerce_metadata(metadata)
    if out_dir is None:
        out_path = metadata_path.parent if metadata_path is not None else Path.cwd()
    else:
        out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    selected_weight_col = _require_node_weight(df)
    written: dict[str, Path] = {}

    if outputs in ("both", "rt"):
        rt_path = out_path / "Rt.pdf"
        fig = _plot_bucket_rt(
            df,
            rt_path,
            selected_weight_col,
            change_threshold,
            bin_days,
            recency_strength,
            rt_quantiles,
            start_col=start_col,
            end_col=end_col,
        )
        plt.close(fig)
        written["rt"] = rt_path
        rt_table_path = out_path / "Rt.csv"
        rt_summary_table(
            df,
            weight_col=selected_weight_col,
            change_threshold=change_threshold,
            bin_days=bin_days,
            recency_strength=recency_strength,
            rt_quantiles=rt_quantiles,
            start_col=start_col,
            end_col=end_col,
        ).to_csv(rt_table_path, index=False, float_format="%.4f")
        written["rt_table"] = rt_table_path

    if outputs in ("both", "subtree_r"):
        subtree_path = out_path / "R0_subtree.pdf"
        fig = _plot_subtree_r(
            df,
            subtree_path,
            start_col,
            end_col,
            selected_weight_col,
            change_threshold,
            show_change_points=show_change_points,
        )
        plt.close(fig)
        written["subtree_r"] = subtree_path

    if outputs not in ("both", "rt", "subtree_r"):
        raise ValueError("outputs must be one of 'both', 'rt', or 'subtree_r'")
    return written

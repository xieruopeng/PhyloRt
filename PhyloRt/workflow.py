"""Combined PhyloRt workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

import pandas as pd

from .plotting import plot
from .prediction import predict


def run(
    tree: str | Path,
    sampling: str | Iterable[float],
    pre_model: str,
    out_dir: str | Path,
    subtree_sizes: str | Iterable[int] = (50, 100, 200, 400),
    sampling_times: str | Iterable[str | float | int] | None = None,
    time_anchor: str = "mid",
    n_jobs: int = 10,
    batch_size: int = 100,
    resume: bool = True,
    keep_tree_files: bool = False,
    n_replicates: int = 1,
    model_dir: str | Path | None = None,
    outputs: Literal["both", "rt", "subtree_r"] = "both",
    start_col: str = "node_age",
    end_col: str = "last_sample_date",
    change_threshold: float = 0.9,
    bin_days: int = 14,
    recency_strength: float = 0.0,
    rt_quantiles: str | tuple[float, float] = "25,75",
    show_change_points: bool = False,
    date_file: str | Path | None = None,
    lsd_lower_rate: float = 0.0001141552511,
    lsd_upper_rate: float = 0.0001141552511,
    lsd_rooting: str = "l",
    lsd_lambda: str | float | int = -1,
    lsd2_bin: str = "lsd2",
    gotree_bin: str = "gotree",
    polytomy_seed: int | None = None,
) -> dict[str, pd.DataFrame | dict[str, Path]]:
    """Run prediction and then generate the default PhyloRt plots."""
    metadata = predict(
        tree=tree,
        sampling=sampling,
        pre_model=pre_model,
        out_dir=out_dir,
        subtree_sizes=subtree_sizes,
        sampling_times=sampling_times,
        time_anchor=time_anchor,
        n_jobs=n_jobs,
        batch_size=batch_size,
        resume=resume,
        keep_tree_files=keep_tree_files,
        n_replicates=n_replicates,
        model_dir=model_dir,
        date_file=date_file,
        lsd_lower_rate=lsd_lower_rate,
        lsd_upper_rate=lsd_upper_rate,
        lsd_rooting=lsd_rooting,
        lsd_lambda=lsd_lambda,
        lsd2_bin=lsd2_bin,
        gotree_bin=gotree_bin,
        polytomy_seed=polytomy_seed,
    )
    plots = plot(
        metadata,
        out_dir=out_dir,
        outputs=outputs,
        start_col=start_col,
        end_col=end_col,
        change_threshold=change_threshold,
        bin_days=bin_days,
        recency_strength=recency_strength,
        rt_quantiles=rt_quantiles,
        show_change_points=show_change_points,
    )
    return {"metadata": metadata, "plots": plots}

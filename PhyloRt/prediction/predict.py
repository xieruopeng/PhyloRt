"""Main PhyloRt prediction workflow."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

from .. import transform
from .extraction import extract_subtrees
from .metadata import _format_metadata_for_output
from .models import _predict_metadata_rows, load_model_bundle
from .sampling import _parse_floats, _parse_ints, _parse_sampling_times
from .state import (
    build_replicate_run_config,
    clear_generated_outputs,
    ensure_resume_manifest,
    load_manifest,
    remove_replicate_work_files,
    remove_tree_files,
    replicate_id,
    replicate_seed,
    save_manifest,
    setup_logging,
)

def predict(
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
    load_model_func: Optional[Callable[[str], object]] = None,
    date_file: str | Path | None = None,
    lsd_lower_rate: float = 0.0001141552511,
    lsd_upper_rate: float = 0.0001141552511,
    lsd_rooting: str = "l",
    lsd_lambda: str | float | int = -1,
    lsd2_bin: str = "lsd2",
    gotree_bin: str = "gotree",
    polytomy_seed: int | None = None,
    timescale_command_func: Optional[Callable] = None,
) -> pd.DataFrame:
    """Run the PhyloRt subtree extraction and prediction workflow."""
    del time_anchor  # Kept for API compatibility with earlier scripts.

    n_replicates = int(n_replicates)
    if n_replicates < 1:
        raise ValueError("n_replicates must be at least 1")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    if not resume:
        clear_generated_outputs(out_path)

    thresholds = sorted(_parse_ints(subtree_sizes))
    allowed_subtree_sizes = {50, 100, 200, 400}
    invalid = [threshold for threshold in thresholds if threshold not in allowed_subtree_sizes]
    if invalid:
        invalid_text = ",".join(str(value) for value in invalid)
        raise ValueError(f"unsupported subtree size(s): {invalid_text}; allowed values are 50,100,200,400")

    sampling_props = _parse_floats(sampling)
    if any(prop < 0.001 or prop >= 1.0 for prop in sampling_props):
        raise ValueError("sampling proportions must be >=0.1% and <100%")

    parsed_sampling_times = _parse_sampling_times(sampling_times)
    if parsed_sampling_times and len(parsed_sampling_times) != len(sampling_props) - 1:
        raise ValueError("sampling_times must have one fewer entry than sampling")

    if date_file is None:
        raise ValueError("date_file is required because PhyloRt expects a genetic-distance tree and time-scales it with LSD2")

    replicate_seeds = [replicate_seed(polytomy_seed, index) for index in range(1, n_replicates + 1)]
    root_logger = setup_logging(out_path)
    root_logger.info(
        "Starting prediction run (replicates=%s, resume=%s, out_dir=%s)",
        n_replicates,
        resume,
        out_path,
    )
    if n_replicates > 1:
        root_config = build_replicate_run_config(
            tree,
            date_file,
            thresholds,
            sampling_props,
            parsed_sampling_times,
            "years",
            pre_model,
            n_replicates,
            replicate_seeds,
        )
        ensure_resume_manifest(out_path, root_config, resume=resume, logger=root_logger)
        existing_manifest = load_manifest(out_path)
        metadata_path = out_path / "metadata.csv"
        if (
            resume
            and existing_manifest is not None
            and existing_manifest.get("status") == "prediction_complete"
            and metadata_path.exists()
        ):
            root_logger.info("Resume fast-path: existing prediction_complete metadata.csv found; reusing prior results.")
            return pd.read_csv(metadata_path)
        (out_path / "time_scaled_trees.nwk").unlink(missing_ok=True)
        for stale_tree in out_path.glob("time_scaled_trees_*.nwk"):
            stale_tree.unlink(missing_ok=True)

    all_metadata: list[pd.DataFrame] = []

    for replicate_index, seed in enumerate(replicate_seeds, start=1):
        current_replicate_id = replicate_id(replicate_index)
        replicate_out_path = out_path if n_replicates == 1 else out_path / "replicates" / current_replicate_id
        replicate_out_path.mkdir(parents=True, exist_ok=True)
        root_logger.info(
            "Replicate %s/%s (%s) writing logs under %s",
            replicate_index,
            n_replicates,
            current_replicate_id,
            replicate_out_path,
        )

        dated_trees = transform.timescale_tree_set(
            tree=tree,
            date_file=date_file,
            out_dir=replicate_out_path,
            pre_model=pre_model,
            lsd_lower_rate=lsd_lower_rate,
            lsd_upper_rate=lsd_upper_rate,
            lsd_rooting=lsd_rooting,
            lsd_lambda=lsd_lambda,
            lsd2_bin=lsd2_bin,
            gotree_bin=gotree_bin,
            keep_tree_files=keep_tree_files,
            random_seed=seed,
            run_command_func=timescale_command_func,
        )

        replicate_dated_trees = replicate_out_path / "time_scaled_trees.nwk"
        if n_replicates > 1:
            root_dated_trees = out_path / f"time_scaled_trees_{replicate_index}.nwk"
            shutil.copyfile(replicate_dated_trees, root_dated_trees)

        meta = extract_subtrees(
            dated_trees,
            tree,
            date_file,
            thresholds,
            sampling_props,
            parsed_sampling_times,
            "years",
            replicate_out_path,
            n_jobs=n_jobs,
            batch_size=batch_size,
            resume=resume,
            subtree_prefix=current_replicate_id if n_replicates > 1 else None,
        )
        if "replicate_id" not in meta.columns:
            meta.insert(0, "replicate_id", current_replicate_id)
        if "replicate_seed" not in meta.columns:
            meta.insert(1, "replicate_seed", np.nan if seed is None else seed)
        all_metadata.append(meta)

    meta = pd.concat(all_metadata, ignore_index=True) if all_metadata else pd.DataFrame()

    if meta.empty:
        meta = _format_metadata_for_output(meta, out_path, keep_tree_files)
        metadata_path = out_path / "metadata.csv"
        meta.to_csv(metadata_path, index=False, float_format="%.4f")
        if n_replicates == 1:
            manifest = load_manifest(out_path)
            if manifest is not None:
                save_manifest(out_path, manifest["extraction_config"], status="prediction_complete")
        else:
            save_manifest(out_path, root_config, status="prediction_complete")
        if not keep_tree_files:
            if n_replicates == 1:
                remove_tree_files(out_path)
            else:
                remove_replicate_work_files(out_path)
        return meta

    if "subtree_id" in meta.columns:
        meta = meta.drop_duplicates(subset=["subtree_id"], keep="last")
    else:
        dupe_cols = [
            "subtree_size",
            "node_age",
            "first_sample_date",
            "last_sample_date",
            "mean_sample_date",
            "sampling_proportion",
            "node_weight",
        ]
        meta = meta.drop_duplicates(subset=[col for col in dupe_cols if col in meta.columns], keep="last")
    meta = meta.reset_index(drop=True)
    meta = meta.drop(columns=["tip_weight"], errors="ignore")

    bundle = load_model_bundle(pre_model, model_dir=model_dir, load_model_func=load_model_func)
    meta = _predict_metadata_rows(meta, bundle, n_jobs=n_jobs)
    meta = _format_metadata_for_output(meta, out_path, keep_tree_files)

    metadata_path = out_path / "metadata.csv"
    meta.to_csv(metadata_path, index=False, float_format="%.4f")
    root_logger.info("Prediction complete: wrote metadata to %s", metadata_path)
    if n_replicates == 1:
        manifest = load_manifest(out_path)
        if manifest is not None:
            save_manifest(out_path, manifest["extraction_config"], status="prediction_complete")
    else:
        save_manifest(out_path, root_config, status="prediction_complete")
    if not keep_tree_files:
        if n_replicates == 1:
            remove_tree_files(out_path)
        else:
            remove_replicate_work_files(out_path)
    return meta

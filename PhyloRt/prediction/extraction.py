"""Subtree extraction and batch processing."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from .encoding import _tree_class
from .sampling import _np_datetime, get_prop_for_date
from .state import (
    build_extraction_config,
    ensure_resume_manifest,
    get_memory_usage,
    load_existing_results,
    load_state,
    save_manifest,
    save_state,
    setup_logging,
)


@dataclass
class ExtractionContext:
    tree_id: str
    tree_file: Path
    iterate_dir: Path
    latest_dt: datetime
    root_date: datetime
    max_rel: float
    tip_date_map: dict[str, float]
    tip_abs_date_map: dict[str, datetime]
    prop_map: dict[str, float]
    tree: object

def _annotate_relative_dates(tree) -> None:
    for idx, node in enumerate(tree.traverse("levelorder")):
        parent_date = 0 if node.is_root() else getattr(node.up, "date")[0]
        node.add_feature("date", (parent_date + node.dist, idx))

def _build_extraction_context(
    info,
    tree_units: str,
    sampling_times: Sequence[str | float | int | datetime],
    sampling_props: Sequence[float],
) -> tuple[ExtractionContext, list[datetime]]:
    Tree = _tree_class()
    date_units = 365 if tree_units == "years" else 1
    tree = Tree(str(info.dated_tree), format=1)
    tree.dist = 0
    _annotate_relative_dates(tree)

    all_leaves = list(tree.iter_leaves())
    all_internal_nodes = [node for node in tree.traverse() if not node.is_leaf()]
    tip_date_map = {leaf.name: leaf.date[0] for leaf in all_leaves}
    max_rel = max(tip_date_map.values())
    root_date = info.latest_date - timedelta(days=max_rel * date_units)

    tip_names = list(tip_date_map.keys())
    all_tip_rel_dates = np.array([tip_date_map[name] for name in tip_names])
    tip_abs_dates = [root_date + timedelta(days=rel * date_units) for rel in all_tip_rel_dates]
    tip_abs_date_map = dict(zip(tip_names, tip_abs_dates))
    prop_map = {
        name: get_prop_for_date(date, sampling_times, sampling_props)
        for name, date in tip_abs_date_map.items()
    }

    node_rel_dates = np.array([node.date[0] for node in all_internal_nodes])
    node_abs_dates = [root_date + timedelta(days=rel * date_units) for rel in node_rel_dates]
    context = ExtractionContext(
        tree_id=info.tree_id,
        tree_file=Path(info.dated_tree),
        iterate_dir=Path(info.tree_dir) / "iterate",
        latest_dt=info.latest_date,
        root_date=root_date,
        max_rel=max_rel,
        tip_date_map=tip_date_map,
        tip_abs_date_map=tip_abs_date_map,
        prop_map=prop_map,
        tree=tree,
    )
    return context, [*tip_abs_dates, *node_abs_dates]

def extract_subtrees(
    dated_trees: Sequence[object],
    source_tree_file: str | Path,
    date_file: str | Path,
    subtree_sizes: Sequence[int],
    sampling_props: Sequence[float],
    sampling_times: Sequence[str | float | int | datetime],
    tree_units: str,
    out_dir: str | Path,
    n_jobs: int = 10,
    batch_size: int = 100,
    resume: bool = True,
    subtree_prefix: str | None = None,
) -> pd.DataFrame:
    """Extract pruned subtrees from dated trees and return combined metadata."""
    Tree = _tree_class()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    logger = setup_logging(out_path)
    state_file = out_path / "processing_state.json"

    existing_results = pd.DataFrame()
    last_batch_completed = 0
    if resume:
        existing_results = load_existing_results(out_path)
        state = load_state(state_file)
        if state:
            last_batch_completed = int(state.get("batch_completed", 0))
        if len(existing_results) > 0:
            logger.info("Resuming with %s existing rows", len(existing_results))

    date_units = 365 if tree_units == "years" else 1
    thresholds = sorted(set(int(size) for size in subtree_sizes), reverse=True)
    min_threshold = min(thresholds)
    extraction_config = build_extraction_config(
        source_tree_file,
        date_file,
        dated_trees,
        thresholds,
        sampling_props,
        sampling_times,
        tree_units,
    )
    ensure_resume_manifest(out_path, extraction_config, resume=resume, logger=logger)

    logger.info("Starting subtree extraction from %s dated tree(s)", len(dated_trees))
    logger.info("Initial memory usage: %s", get_memory_usage())
    logger.info("Processing thresholds: %s", thresholds)

    contexts: list[ExtractionContext] = []
    all_calendar_dates: list[datetime] = []
    for info in dated_trees:
        context, context_dates = _build_extraction_context(info, tree_units, sampling_times, sampling_props)
        context.iterate_dir.mkdir(parents=True, exist_ok=True)
        contexts.append(context)
        all_calendar_dates.extend(context_dates)

    global_sorted_dates = np.sort(np.array([_np_datetime(date) for date in all_calendar_dates], dtype="datetime64[ns]"))

    candidate_nodes = []
    node_info = []
    for context in contexts:
        for node_index, node in enumerate(tqdm(context.tree.traverse("postorder"), desc=f"Scanning {context.tree_id}")):
            if node.is_leaf():
                continue
            n_tips = len(node.get_leaves())
            if n_tips < min_threshold:
                continue
            candidate_nodes.append((context, node, node_index))
            node_info.append({"n_tips": n_tips, "node_index": node_index, "tree_id": context.tree_id})

    total_nodes = len(candidate_nodes)
    logger.info("Found %s candidate nodes to process", total_nodes)

    def process_node(args):
        context, node, node_index, node_tip_count = args
        if node_tip_count < min_threshold:
            return []

        valid_sizes = [size for size in thresholds if size <= node_tip_count]
        if len(valid_sizes) < len(thresholds):
            valid_sizes.append(node_tip_count)
        valid_sizes = sorted(set(valid_sizes), reverse=True)

        tips = node.get_leaves()
        node_tip_names = [tip.name for tip in tips]
        node_tip_rel_dates = np.array([context.tip_date_map[name] for name in node_tip_names])
        sorted_indices = np.argsort(node_tip_rel_dates)
        sorted_node_tip_names = [node_tip_names[i] for i in sorted_indices]
        sorted_node_rel_dates = node_tip_rel_dates[sorted_indices]
        sorted_node_abs_dates = [context.tip_abs_date_map[name] for name in sorted_node_tip_names]
        sorted_node_props = np.array([context.prop_map[name] for name in sorted_node_tip_names])

        rows = []
        max_subtree = None
        for idx, size in enumerate(valid_sizes):
            selected_names = sorted_node_tip_names[:size]
            selected_rel = sorted_node_rel_dates[:size]
            selected_abs = sorted_node_abs_dates[:size]
            selected_props = sorted_node_props[:size]

            variant_first_date = min(selected_abs)
            variant_last_date = max(selected_abs)
            variant_mean_date = pd.to_datetime(selected_abs).mean().to_pydatetime()
            subtree_sampling_prop = float(np.mean(selected_props))

            node_age = context.root_date + timedelta(days=node.date[0] * date_units)
            date_max = max(selected_abs)
            node_left_idx = np.searchsorted(global_sorted_dates, _np_datetime(node_age), side="left")
            node_right_idx = np.searchsorted(global_sorted_dates, _np_datetime(date_max), side="right")
            node_in_range_count = node_right_idx - node_left_idx
            node_weight = (size * 2 - 1) / node_in_range_count if node_in_range_count > 0 else np.nan

            subtree_stem = f"{context.tree_id}_{node_index}_{size}"
            if subtree_prefix:
                subtree_stem = f"{subtree_prefix}_{subtree_stem}"
            nwk_filename = context.iterate_dir / f"subtree_{subtree_stem}.nwk"
            if idx == 0:
                max_subtree = Tree(node.write(format=1), format=1)
                max_subtree.prune(selected_names, preserve_branch_length=True)
                subtree_variant = max_subtree
            else:
                subtree_variant = max_subtree.copy()
                subtree_variant.prune(selected_names, preserve_branch_length=True)

            subtree_variant.write(outfile=str(nwk_filename), format=1)
            last_to_full = (context.max_rel - max(selected_rel)) / context.max_rel if context.max_rel > 0 else 0.0
            mean_to_full = (context.max_rel - float(np.mean(selected_rel))) / context.max_rel if context.max_rel > 0 else 0.0

            rows.append(
                {
                    "tree_id": context.tree_id,
                    "node_index": node_index,
                    "subtree_id": subtree_stem,
                    "subtree_size": size,
                    "node_age": node_age.strftime("%Y-%m-%d"),
                    "first_sample_date": variant_first_date.strftime("%Y-%m-%d"),
                    "last_sample_date": variant_last_date.strftime("%Y-%m-%d"),
                    "mean_sample_date": variant_mean_date.strftime("%Y-%m-%d"),
                    "sampling_proportion": subtree_sampling_prop,
                    "node_weight": node_weight,
                    "subtree_nwk_file": str(nwk_filename),
                    "last_sample_to_full_tree": last_to_full,
                    "mean_sample_to_full_tree": mean_to_full,
                }
            )
        return rows

    all_rows = existing_results.to_dict("records") if len(existing_results) else []
    processed_nodes = 0
    current_batch_number = last_batch_completed + 1

    for batch_start in range(0, total_nodes, batch_size):
        batch_end = min(batch_start + batch_size, total_nodes)
        batch_nodes_info = candidate_nodes[batch_start:batch_end]
        batch_info = node_info[batch_start:batch_end]

        nodes_processed_start = batch_start + 1
        nodes_processed_end = batch_end
        logger.info(
            "Processing batch %s (nodes %s-%s)",
            current_batch_number,
            nodes_processed_start,
            nodes_processed_end,
        )

        batch_args = [
            (context, node, node_index, info["n_tips"])
            for (context, node, node_index), info in zip(batch_nodes_info, batch_info)
        ]

        if n_jobs == 1:
            batch_results = [process_node(args) for args in tqdm(batch_args, desc=f"Batch {current_batch_number}", leave=False)]
        else:
            batch_results = Parallel(
                n_jobs=n_jobs,
                backend="threading",
                batch_size=max(1, len(batch_args) // max(1, n_jobs * 2)),
            )(delayed(process_node)(args) for args in tqdm(batch_args, desc=f"Batch {current_batch_number}", leave=False))

        batch_rows = [row for result in batch_results for row in result]
        all_rows.extend(batch_rows)
        processed_nodes += len(batch_nodes_info)

        if batch_rows:
            batch_df = pd.DataFrame(batch_rows)
            node_range = f"processed_{nodes_processed_start}_to_{nodes_processed_end}"
            batch_file = out_path / f"batch_{current_batch_number:04d}_{node_range}.csv"
            batch_df.to_csv(batch_file, index=False)
            logger.info("Saved %s rows to %s", len(batch_rows), batch_file.name)

        last_node_index = batch_nodes_info[-1][2] if batch_nodes_info else -1
        last_tree_id = batch_nodes_info[-1][0].tree_id if batch_nodes_info else None
        save_state(
            state_file,
            {
                "last_processed_tree": last_tree_id,
                "last_processed_node": last_node_index,
                "processed_nodes": processed_nodes,
                "total_nodes": total_nodes,
                "timestamp": datetime.now().isoformat(),
                "batch_completed": current_batch_number,
            },
        )
        current_batch_number += 1
        del batch_results, batch_rows
        gc.collect()

    final_df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
    total_time = time.time() - start_time
    logger.info("Extraction complete in %.2f seconds", total_time)
    logger.info("Total subtrees generated: %s", len(final_df))

    if state_file.exists():
        state_file.unlink()
    save_manifest(out_path, extraction_config, status="extraction_complete")

    return final_df

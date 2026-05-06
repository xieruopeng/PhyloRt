"""Metadata formatting for public PhyloRt outputs."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

def _compact_id(value, prefix: str):
    if pd.isna(value):
        return value
    text = str(value)
    if text.startswith(prefix):
        suffix = text[len(prefix) :]
        if suffix.isdigit():
            return int(suffix)
    return value

def _relative_path_from_out(path_value, out_path: Path):
    if pd.isna(path_value):
        return path_value
    path = Path(str(path_value))
    try:
        return str(path.resolve().relative_to(out_path.resolve()))
    except (OSError, ValueError):
        try:
            return str(path.relative_to(out_path))
        except ValueError:
            return str(path)

def _compact_subtree_id(value):
    if pd.isna(value):
        return value
    text = str(value)
    match = re.fullmatch(r"replicate_0*\d+_tree_0*\d+_(.+)", text)
    if match:
        return match.group(1)
    match = re.fullmatch(r"tree_0*\d+_(.+)", text)
    if match:
        return match.group(1)
    match = re.fullmatch(r"\d+_\d+_(\d+_\d+)", text)
    if match:
        return match.group(1)
    return text

def _format_subtree_ids(meta: pd.DataFrame) -> pd.Series:
    if meta.empty:
        return pd.Series(index=meta.index, dtype=object)
    required = {"replicate_id", "tree_id", "node_index", "subtree_size"}
    if required <= set(meta.columns):
        return meta.apply(
            lambda row: f"{int(row['node_index'])}_{int(row['subtree_size'])}",
            axis=1,
        )
    if "subtree_id" in meta.columns:
        return meta["subtree_id"].map(_compact_subtree_id)
    return pd.Series(index=meta.index, dtype=object)

def _round_metadata_numeric_columns(meta: pd.DataFrame) -> pd.DataFrame:
    if "sampling_proportion" not in meta.columns:
        return meta
    start_idx = list(meta.columns).index("sampling_proportion")
    for column in meta.columns[start_idx:]:
        if pd.api.types.is_numeric_dtype(meta[column]):
            meta[column] = meta[column].round(4)
    return meta

def _format_metadata_for_output(meta: pd.DataFrame, out_path: Path, keep_tree_files: bool) -> pd.DataFrame:
    meta = meta.copy()
    if "replicate_id" in meta.columns:
        meta["replicate_id"] = meta["replicate_id"].map(lambda value: _compact_id(value, "replicate_"))
    if "tree_id" in meta.columns:
        meta["tree_id"] = meta["tree_id"].map(lambda value: _compact_id(value, "tree_"))
    if "subtree_id" in meta.columns:
        meta["subtree_id"] = _format_subtree_ids(meta)
    if "replicate_seed" in meta.columns:
        seed_values = meta["replicate_seed"].replace("", np.nan)
        if seed_values.isna().all():
            meta = meta.drop(columns=["replicate_seed"])
        else:
            meta["replicate_seed"] = pd.to_numeric(seed_values, errors="coerce").astype("Int64")
    meta = meta.drop(columns=["R_naught", "tip_weight"], errors="ignore")
    if "subtree_nwk_file" in meta.columns:
        if keep_tree_files:
            meta["subtree_nwk_file"] = meta["subtree_nwk_file"].map(lambda value: _relative_path_from_out(value, out_path))
        else:
            meta = meta.drop(columns=["subtree_nwk_file"])
    meta = _round_metadata_numeric_columns(meta)
    return meta

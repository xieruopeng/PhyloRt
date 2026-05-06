"""Resume manifests and generated-output cleanup."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import psutil

MANIFEST_FILENAME = "phylort_run_manifest.json"

def setup_logging(out_dir: str | Path) -> logging.Logger:
    log_file = Path(out_dir) / "extraction_log.txt"
    logger = logging.getLogger(f"PhyloRt.prediction.{Path(out_dir).resolve()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # Rebuild handlers each invocation so logs always target the current file.
    # This avoids stale handlers when runs are resumed/restarted in the same process.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

def get_memory_usage() -> str:
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return f"{mem_info.rss / 1024 / 1024:.1f} MB"

def save_state(state_file: str | Path, state_data: dict) -> None:
    with open(state_file, "w", encoding="utf-8") as handle:
        json.dump(state_data, handle, indent=2, default=str)

def load_state(state_file: str | Path) -> Optional[dict]:
    state_path = Path(state_file)
    if not state_path.exists():
        return None
    try:
        with open(state_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        logging.warning("Could not load state file %s: %s", state_file, exc)
    return None

def file_sha256(path: str | Path) -> str:
    """Return a SHA-256 fingerprint for a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def build_extraction_config(
    tree_file: str | Path,
    date_file: str | Path,
    dated_trees: Sequence[object],
    subtree_sizes: Sequence[int],
    sampling_props: Sequence[float],
    sampling_times: Sequence[str | float | int | datetime],
    tree_units: str,
) -> dict:
    """Build the resume-safety config stored with generated outputs."""
    tree_path = Path(tree_file)
    if not tree_path.exists():
        raise FileNotFoundError(f"Input tree does not exist: {tree_file}")
    date_path = Path(date_file)
    if not date_path.exists():
        raise FileNotFoundError(f"Date file does not exist: {date_file}")
    return {
        "tree_path": str(tree_path.resolve()),
        "tree_sha256": file_sha256(tree_path),
        "date_file_path": str(date_path.resolve()),
        "date_file_sha256": file_sha256(date_path),
        "dated_trees": [
            {
                "tree_id": info.tree_id,
                "latest_date": info.latest_date.strftime("%Y-%m-%d"),
                "tip_count": int(info.tip_count),
            }
            for info in dated_trees
        ],
        "subtree_sizes": sorted({int(size) for size in subtree_sizes}),
        "sampling_props": [float(prop) for prop in sampling_props],
        "sampling_times": [str(item) for item in sampling_times],
        "tree_units": tree_units,
    }

def replicate_id(index: int) -> str:
    return f"replicate_{index:04d}"

def replicate_seed(polytomy_seed: int | None, index: int) -> int | None:
    if polytomy_seed is None:
        return None
    return int(polytomy_seed) + index - 1

def build_replicate_run_config(
    tree_file: str | Path,
    date_file: str | Path,
    subtree_sizes: Sequence[int],
    sampling_props: Sequence[float],
    sampling_times: Sequence[str | float | int | datetime],
    tree_units: str,
    pre_model: str,
    n_replicates: int,
    replicate_seeds: Sequence[int | None],
) -> dict:
    """Build the root resume-safety config for multi-replicate runs."""
    tree_path = Path(tree_file)
    if not tree_path.exists():
        raise FileNotFoundError(f"Input tree does not exist: {tree_file}")
    date_path = Path(date_file)
    if not date_path.exists():
        raise FileNotFoundError(f"Date file does not exist: {date_file}")
    return {
        "tree_path": str(tree_path.resolve()),
        "tree_sha256": file_sha256(tree_path),
        "date_file_path": str(date_path.resolve()),
        "date_file_sha256": file_sha256(date_path),
        "subtree_sizes": sorted({int(size) for size in subtree_sizes}),
        "sampling_props": [float(prop) for prop in sampling_props],
        "sampling_times": [str(item) for item in sampling_times],
        "tree_units": tree_units,
        "pre_model": pre_model,
        "n_replicates": int(n_replicates),
        "replicate_seeds": [None if seed is None else int(seed) for seed in replicate_seeds],
    }

def _manifest_path(out_dir: str | Path) -> Path:
    return Path(out_dir) / MANIFEST_FILENAME

def load_manifest(out_dir: str | Path) -> Optional[dict]:
    path = _manifest_path(out_dir)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)

def save_manifest(out_dir: str | Path, extraction_config: dict, status: str) -> None:
    payload = {
        "extraction_config": extraction_config,
        "status": status,
        "updated_at": datetime.now().isoformat(),
    }
    with open(_manifest_path(out_dir), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

def _has_generated_outputs(out_dir: str | Path) -> bool:
    out_path = Path(out_dir)
    return (
        any(out_path.glob("batch_*.csv"))
        or (out_path / "metadata.csv").exists()
        or (out_path / "processing_state.json").exists()
    )

def ensure_resume_manifest(
    out_dir: str | Path,
    extraction_config: dict,
    resume: bool,
    logger: logging.Logger,
) -> None:
    """Validate or initialize the resume manifest for an output directory."""
    manifest = load_manifest(out_dir)
    if resume and manifest is not None:
        previous = manifest.get("extraction_config", {})
        if previous != extraction_config:
            changed = sorted(
                key for key in set(previous) | set(extraction_config)
                if previous.get(key) != extraction_config.get(key)
            )
            raise ValueError(
                "Cannot resume because the input tree or extraction options differ "
                f"from the previous run. Changed fields: {', '.join(changed)}. "
                "Use --no-resume to clear PhyloRt-generated files and start fresh, "
                "or choose a new --out-dir."
            )
        logger.info("Resume manifest matches current input tree and options")
        save_manifest(out_dir, extraction_config, status=manifest.get("status", "running"))
        return

    if resume and manifest is None and _has_generated_outputs(out_dir):
        raise ValueError(
            "Cannot safely resume: existing generated outputs were found but no "
            f"{MANIFEST_FILENAME} is available to verify the input tree. "
            "Use --no-resume to clear PhyloRt-generated files and start fresh, "
            "or choose a new --out-dir."
        )

    save_manifest(out_dir, extraction_config, status="running")

def clear_generated_outputs(out_dir: str | Path) -> None:
    """Remove PhyloRt-generated outputs before a fresh non-resume run."""
    out_path = Path(out_dir)
    for path in out_path.glob("batch_*.csv"):
        path.unlink(missing_ok=True)
    for path in out_path.glob("time_scaled_trees_*.nwk"):
        path.unlink(missing_ok=True)
    for name in (
        "metadata.csv",
        "processing_state.json",
        MANIFEST_FILENAME,
        "time_scaled_tree.nwk",
        "time_scaled_trees.nwk",
        "Rt.pdf",
        "Rt.csv",
        "R0_subtree.pdf",
        "per_tip_2x2_dates_Rt.pdf",
        "per_tip_2x2_dates_change_points.pdf",
    ):
        (out_path / name).unlink(missing_ok=True)
    iterate_dir = out_path / "iterate"
    if iterate_dir.exists():
        shutil.rmtree(iterate_dir)
    trees_dir = out_path / "trees"
    if trees_dir.exists():
        shutil.rmtree(trees_dir)
    replicates_dir = out_path / "replicates"
    if replicates_dir.exists():
        shutil.rmtree(replicates_dir)

def remove_tree_files(out_dir: str | Path) -> None:
    trees_dir = Path(out_dir) / "trees"
    if trees_dir.exists():
        shutil.rmtree(trees_dir)

def remove_replicate_work_files(out_dir: str | Path) -> None:
    replicates_dir = Path(out_dir) / "replicates"
    if replicates_dir.exists():
        shutil.rmtree(replicates_dir)

def load_existing_results(out_dir: str | Path) -> pd.DataFrame:
    out_path = Path(out_dir)
    all_results = []
    for path in sorted(out_path.glob("batch_*.csv")):
        try:
            all_results.append(pd.read_csv(path))
            logging.info("Loaded %s results from %s", len(all_results[-1]), path.name)
        except Exception as exc:
            logging.warning("Could not load %s: %s", path, exc)

    if not all_results:
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)
    if "subtree_id" in combined.columns:
        combined = combined.drop_duplicates(subset=["subtree_id"], keep="last")
    return combined

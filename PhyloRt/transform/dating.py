"""Tree dating workflow for genetic-distance Newick trees."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

from .lsd import (
    DEFAULT_LSD_LOWER_RATE,
    DEFAULT_LSD_UPPER_RATE,
    LEGACY_TIMESCALE_MANIFEST,
    _run_external_command,
    cleanup_dating_intermediates,
    strip_newick_annotations,
)
from .newick import DatedTreeInfo, parse_date_file, split_newick_records, tree_tip_names, write_lsd_date_file
from .pathogens import default_clock_file, pathogen_timescale_defaults
from .polytomies import resolve_polytomies

def timescale_genetic_tree(
    tree: str | Path,
    date_file: str | Path,
    out_dir: str | Path,
    pre_model: str = "FLUB",
    lsd_lower_rate: float = DEFAULT_LSD_LOWER_RATE,
    lsd_upper_rate: float = DEFAULT_LSD_UPPER_RATE,
    lsd_rooting: str = "l",
    lsd_lambda: str | float | int = -1,
    lsd2_bin: str = "lsd2",
    gotree_bin: str = "gotree",
    resume: bool | None = None,
    cleanup_intermediates: bool = True,
    random_seed: int | None = None,
    run_command_func: Callable | None = None,
) -> Path:
    """Convert a genetic-distance tree to a dated Newick tree with LSD2."""
    del resume  # Time-scaling is intentionally stateless; kept for API compatibility.

    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    defaults = pathogen_timescale_defaults(pre_model)
    sequence_length = defaults.genome_length
    clock_path = default_clock_file(pre_model)

    resolved_tree = out_path / "resolved_tree.nwk"
    dated_nexus = out_path / "resolved_tree.nwk.result.date.nexus"
    dated_tree = out_path / "time_scaled_tree.nwk"
    log_path = out_path / "extraction_log.txt"

    (out_path / LEGACY_TIMESCALE_MANIFEST).unlink(missing_ok=True)
    if cleanup_intermediates:
        cleanup_dating_intermediates(out_path)
    dated_tree.unlink(missing_ok=True)

    resolve_polytomies(
        in_nwk=tree,
        out_nwk=resolved_tree,
        sequence_length=sequence_length,
        random_seed=random_seed,
    )

    lsd2_command = [
        lsd2_bin,
        "-i",
        resolved_tree.name,
        "-d",
        str(Path(date_file).resolve()),
        "-l",
        str(lsd_lambda),
        "-s",
        str(defaults.genome_length),
        "-w",
        str(clock_path.resolve()),
        "-q",
        str(defaults.clock_sd),
        "-u",
        str(lsd_lower_rate),
        "-U",
        str(lsd_upper_rate),
        "-r",
        str(lsd_rooting),
    ]
    _run_external_command(lsd2_command, cwd=out_path, log_path=log_path, run_command_func=run_command_func)
    if not dated_nexus.exists():
        raise FileNotFoundError(f"LSD2 did not produce the expected dated Nexus tree: {dated_nexus}")

    gotree_command = [
        gotree_bin,
        "reformat",
        "newick",
        "-f",
        "nexus",
        "-i",
        str(dated_nexus),
        "-o",
        str(dated_tree),
    ]
    _run_external_command(gotree_command, cwd=out_path, log_path=log_path, run_command_func=run_command_func)
    if not dated_tree.exists():
        raise FileNotFoundError(f"gotree did not produce the expected dated Newick tree: {dated_tree}")
    strip_newick_annotations(dated_tree)
    if cleanup_intermediates:
        cleanup_dating_intermediates(out_path)
    return dated_tree

def _append_tree_log(root_log: Path, tree_id: str, tree_log: Path) -> None:
    if not tree_log.exists():
        return
    with open(root_log, "a", encoding="utf-8") as handle:
        handle.write(f"\n[{tree_id} time-scaling]\n")
        handle.write(tree_log.read_text(encoding="utf-8"))

def timescale_tree_set(
    tree: str | Path,
    date_file: str | Path,
    out_dir: str | Path,
    pre_model: str = "FLUB",
    lsd_lower_rate: float = DEFAULT_LSD_LOWER_RATE,
    lsd_upper_rate: float = DEFAULT_LSD_UPPER_RATE,
    lsd_rooting: str = "l",
    lsd_lambda: str | float | int = -1,
    lsd2_bin: str = "lsd2",
    gotree_bin: str = "gotree",
    keep_tree_files: bool = False,
    random_seed: int | None = None,
    run_command_func: Callable | None = None,
) -> list[DatedTreeInfo]:
    """Time-scale one or more Newick trees and write a combined dated tree file."""
    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    trees_dir = out_path / "trees"
    if trees_dir.exists():
        shutil.rmtree(trees_dir)
    trees_dir.mkdir(parents=True, exist_ok=True)

    records = split_newick_records(tree)
    date_map = parse_date_file(date_file)
    root_log = out_path / "extraction_log.txt"
    combined_path = out_path / "time_scaled_trees.nwk"
    combined_path.unlink(missing_ok=True)

    dated_infos: list[DatedTreeInfo] = []
    for record in records:
        tree_dir = trees_dir / record.tree_id
        tree_dir.mkdir(parents=True, exist_ok=True)
        source_tree = tree_dir / "genetic_tree.nwk"
        source_tree.write_text(record.newick + "\n", encoding="utf-8")

        tips = tree_tip_names(source_tree)
        lsd_dates = tree_dir / "lsd_dates.tsv"
        latest_date = write_lsd_date_file(date_map, tips, lsd_dates, record.tree_id)
        dated_tree = timescale_genetic_tree(
            tree=source_tree,
            date_file=lsd_dates,
            out_dir=tree_dir,
            pre_model=pre_model,
            lsd_lower_rate=lsd_lower_rate,
            lsd_upper_rate=lsd_upper_rate,
            lsd_rooting=lsd_rooting,
            lsd_lambda=lsd_lambda,
            lsd2_bin=lsd2_bin,
            gotree_bin=gotree_bin,
            cleanup_intermediates=not keep_tree_files,
            random_seed=random_seed,
            run_command_func=run_command_func,
        )
        _append_tree_log(root_log, record.tree_id, tree_dir / "extraction_log.txt")
        dated_infos.append(
            DatedTreeInfo(
                tree_id=record.tree_id,
                source_tree=source_tree,
                dated_tree=dated_tree,
                tree_dir=tree_dir,
                latest_date=latest_date,
                tip_count=len(tips),
            )
        )

    with open(combined_path, "w", encoding="utf-8") as handle:
        for info in dated_infos:
            handle.write(info.dated_tree.read_text(encoding="utf-8").strip() + "\n")

    return dated_infos

timescale_tree = timescale_genetic_tree

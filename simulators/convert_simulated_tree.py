#!/usr/bin/env python3
"""Convert a treesimulator time tree into an LSD2-dated training tree."""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PhyloRt.transform import (  # noqa: E402
    DEFAULT_LSD_LOWER_RATE,
    DEFAULT_LSD_UPPER_RATE,
    resolve_polytomies,
    strip_newick_annotations,
)


def _tree_class():
    try:
        from ete3 import Tree
    except ImportError as exc:
        raise ImportError(
            "convert_simulated_tree.py requires ete3. Install PhyloRt dependencies before running it."
        ) from exc
    return Tree


def _validate_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _decimal_year(value: float) -> str:
    return f"{value:.10f}".rstrip("0").rstrip(".")


def _lognormal_parameters(mean: float, sd: float) -> tuple[float, float] | None:
    _validate_positive("clock-rate-mean", mean)
    if sd < 0:
        raise ValueError("clock-rate-sd must be non-negative")
    if sd == 0:
        return None
    variance_ratio = (sd * sd) / (mean * mean)
    sigma_squared = math.log1p(variance_ratio)
    mu = math.log(mean) - 0.5 * sigma_squared
    sigma = math.sqrt(sigma_squared)
    return mu, sigma


def write_lsd_dates_from_simulated_tree(tree, out_file: str | Path, root_year: float = 2000.0) -> Path:
    """Write an LSD2 date file from root-to-tip distances in days."""
    leaves = list(tree.iter_leaves())
    out_path = Path(out_file)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(f"{len(leaves)}\n")
        for leaf in leaves:
            sample_year = root_year + leaf.get_distance(tree) / 365.0
            handle.write(f"{leaf.name}\t{_decimal_year(sample_year)}\n")
    return out_path


def simulate_genetic_distance_tree(
    tree,
    seq_length: int,
    clock_rate_mean: float,
    clock_rate_sd: float,
    rng: np.random.Generator | None = None,
):
    """Return mutation-count and substitutions/site copies of a day-scaled tree."""
    if seq_length <= 0:
        raise ValueError("seq-length must be a positive integer")
    clock_params = _lognormal_parameters(clock_rate_mean, clock_rate_sd)
    rng = rng or np.random.default_rng()
    mutation_tree = tree.copy()

    for node in mutation_tree.traverse():
        if node.up is None:
            node.dist = 0.0
            continue
        branch_days = max(float(node.dist), 0.0)
        branch_years = branch_days / 365.0
        if branch_years == 0:
            mutation_count = 0
        else:
            if clock_params is None:
                branch_rate = clock_rate_mean
            else:
                branch_rate = float(rng.lognormal(mean=clock_params[0], sigma=clock_params[1]))
            mutation_probability = branch_rate * branch_years
            if mutation_probability > 1:
                raise ValueError(
                    "clock rate and branch length imply a mutation probability greater than 1; "
                    "check that the simulated tree branch lengths are in days"
                )
            mutation_count = int(rng.binomial(seq_length, mutation_probability))
        node.dist = float(mutation_count)

    genetic_tree = mutation_tree.copy()
    for node in genetic_tree.traverse():
        node.dist = float(node.dist) / seq_length

    return mutation_tree, genetic_tree


def _run_external_command(
    command: list[str],
    cwd: Path,
    run_command_func: Callable | None = None,
) -> None:
    runner = run_command_func or subprocess.run
    try:
        runner(
            command,
            cwd=str(cwd),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Could not find external command {command[0]!r}. Install it and make sure it is on PATH.") from exc
    except subprocess.CalledProcessError as exc:
        output = exc.stdout or ""
        detail = f"\n{output}" if output else ""
        raise RuntimeError(f"Command failed: {' '.join(command)}{detail}") from exc


def convert_simulated_tree(
    tree: str | Path,
    output_tree: str | Path,
    seq_length: int,
    clock_rate_mean: float,
    clock_rate_sd: float,
    run_command_func: Callable | None = None,
    rng: np.random.Generator | None = None,
) -> Path:
    """Convert one simulated day-scaled tree to an LSD2-dated Newick tree."""
    tree_path = Path(tree).expanduser().resolve()
    output_path = Path(output_tree).expanduser().resolve()
    if tree_path == output_path:
        raise ValueError("--output-tree must be different from --tree")
    if not tree_path.exists():
        raise FileNotFoundError(f"Input tree does not exist: {tree_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    Tree = _tree_class()
    simulated_tree = Tree(str(tree_path), format=1)

    with tempfile.TemporaryDirectory(prefix=".phylort_sim_", dir=str(output_path.parent)) as tmp:
        work_dir = Path(tmp)
        lsd_dates = work_dir / "lsd_dates.tsv"
        mutation_count_tree = work_dir / "mutation_count_tree.nwk"
        genetic_tree = work_dir / "genetic_tree.nwk"
        resolved_tree = work_dir / "resolved_tree.nwk"
        clock_file = work_dir / "clock.txt"
        dated_nexus = work_dir / "resolved_tree.nwk.result.date.nexus"
        dated_tree = work_dir / "time_scaled_tree.nwk"

        write_lsd_dates_from_simulated_tree(simulated_tree, lsd_dates)
        mutation_tree, lsd_input_tree = simulate_genetic_distance_tree(
            simulated_tree,
            seq_length=seq_length,
            clock_rate_mean=clock_rate_mean,
            clock_rate_sd=clock_rate_sd,
            rng=rng,
        )
        mutation_tree.write(outfile=str(mutation_count_tree), format=5)
        lsd_input_tree.write(outfile=str(genetic_tree), format=5)
        clock_file.write_text(f"{clock_rate_mean:.12g}\n", encoding="utf-8")

        resolve_polytomies(genetic_tree, resolved_tree, sequence_length=seq_length)

        lsd2_command = [
            "lsd2",
            "-i",
            resolved_tree.name,
            "-d",
            lsd_dates.name,
            "-l",
            "-1",
            "-s",
            str(seq_length),
            "-w",
            clock_file.name,
            "-q",
            str(clock_rate_sd),
            "-u",
            str(DEFAULT_LSD_LOWER_RATE),
            "-U",
            str(DEFAULT_LSD_UPPER_RATE),
            "-r",
            "l",
        ]
        _run_external_command(lsd2_command, cwd=work_dir, run_command_func=run_command_func)
        if not dated_nexus.exists():
            raise FileNotFoundError(f"LSD2 did not produce the expected dated Nexus tree: {dated_nexus}")

        gotree_command = [
            "gotree",
            "reformat",
            "newick",
            "-f",
            "nexus",
            "-i",
            str(dated_nexus),
            "-o",
            str(dated_tree),
        ]
        _run_external_command(gotree_command, cwd=work_dir, run_command_func=run_command_func)
        if not dated_tree.exists():
            raise FileNotFoundError(f"gotree did not produce the expected dated Newick tree: {dated_tree}")
        strip_newick_annotations(dated_tree)
        shutil.copyfile(dated_tree, output_path)

    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a treesimulator day-scaled tree into an LSD2-dated PhyloRt training tree."
    )
    parser.add_argument("--tree", required=True, help="Input simulated Newick tree from treesimulator.")
    parser.add_argument("--output-tree", required=True, help="Output LSD2-dated Newick tree.")
    parser.add_argument("--seq-length", required=True, type=int, help="Genome or sequence length used for mutation counts.")
    parser.add_argument("--clock-rate-mean", required=True, type=float, help="Mean molecular clock rate in substitutions/site/year.")
    parser.add_argument("--clock-rate-sd", required=True, type=float, help="Standard deviation of the molecular clock rate.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = convert_simulated_tree(
        tree=args.tree,
        output_tree=args.output_tree,
        seq_length=args.seq_length,
        clock_rate_mean=args.clock_rate_mean,
        clock_rate_sd=args.clock_rate_sd,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

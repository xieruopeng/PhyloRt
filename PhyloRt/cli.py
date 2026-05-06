"""Command-line interface for PhyloRt."""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="PhyloRt", description="PhyloRt toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="{run}")

    predict_parser = subparsers.add_parser(
        "predict",
        help=argparse.SUPPRESS,
        description="Extract pruned subtrees and generate PhyloRt prediction metadata.",
    )
    _add_predict_args(predict_parser)
    _add_timescale_args(predict_parser)
    predict_parser.set_defaults(func=_run_predict)

    run_parser = subparsers.add_parser(
        "run",
        help="Predict and plot in one command.",
        description="Run PhyloRt prediction and then generate the default plot PDFs.",
    )
    _add_predict_args(run_parser)
    _add_timescale_args(run_parser)
    _add_plot_args(run_parser, include_metadata=False)
    run_parser.set_defaults(func=_run_combined)

    plot_parser = subparsers.add_parser(
        "plot",
        help=argparse.SUPPRESS,
        description="Generate Rt and per-subtree R plots from a PhyloRt metadata CSV.",
    )
    _add_plot_args(plot_parser, include_metadata=True)
    plot_parser.set_defaults(func=_run_plot)

    _hide_subcommands_from_help(subparsers, {"predict", "plot"})

    return parser


def build_public_parser() -> argparse.ArgumentParser:
    """Build a single-command public parser equivalent to `run`."""
    parser = argparse.ArgumentParser(prog="PhyloRt", description="PhyloRt toolkit")
    _add_predict_args(parser)
    _add_timescale_args(parser)
    _add_plot_args(parser, include_metadata=False)
    return parser


def _hide_subcommands_from_help(subparsers_action: argparse._SubParsersAction, hidden_names: set[str]) -> None:
    subparsers_action._choices_actions = [
        action for action in subparsers_action._choices_actions if action.dest not in hidden_names
    ]


def _add_predict_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tree",
        required=True,
        help="Input genetic-distance Newick tree file (one or multiple semicolon-terminated trees).",
    )
    parser.add_argument(
        "--subtree-sizes",
        default="50,100,200,400",
        help="Comma-separated subtree-size options. Supported values: 50,100,200,400.",
    )
    parser.add_argument(
        "--sampling",
        required=True,
        help="Comma-separated sampling proportions (min 0.001 / 0.1%%, max <1).",
    )
    parser.add_argument(
        "--sampling-times",
        default=None,
        help="Comma-separated sampling change boundaries (YYYY-MM-DD or decimal year). Must have one fewer value than --sampling.",
    )
    parser.add_argument(
        "--pre-model",
        required=True,
        choices=["COVID", "H3N2", "H1N1", "FLUB", "RSVA", "RSVB"],
        help=(
            "Pathogen model profile for tree dating and prediction compatibility. "
            "H1N1/RSVA/RSVB use their own clock rates for dating but compatible H3N2 neural-network weights."
        ),
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help=(
            "Output directory. Writes run logs (`extraction_log.txt`), dated trees (`time_scaled_trees.nwk` for single runs; "
            "`time_scaled_trees_n.nwk` for replicate n), prediction metadata (`metadata.csv`), and plotting outputs "
            "(`Rt.csv`, `Rt.pdf`, `R0_subtree.pdf`, and per-tip PDF summaries). "
            "With `--keep-tree-files`, retains intermediate dated trees and extracted subtree Newick files."
        ),
    )
    parser.add_argument("--n-jobs", type=int, default=4, help="Number of parallel workers for subtree extraction (recommended: >=1; default: 4).")
    parser.add_argument("--batch-size", type=int, default=100, help="Internal-node batch size for extraction (positive integer; default: 100).")
    parser.add_argument("--no-resume", action="store_true", help="Start a fresh run and ignore resumable batch outputs in --out-dir.")
    parser.add_argument("--keep-tree-files", action="store_true", help="Keep intermediate per-tree and subtree Newick files.")
    parser.add_argument("--n-replicates", type=int, default=1, help="Number of stochastic polytomy-resolution replicates (minimum: 1; default: 1).")


def _add_timescale_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--date-file", required=True, help="Tip-date TSV for dating (tip_name<TAB>YYYY-MM-DD).")
    parser.add_argument("--lsd2-bin", default="lsd2", help="LSD2 executable name or absolute path.")
    parser.add_argument("--gotree-bin", default="gotree", help="gotree executable name or absolute path.")
    parser.add_argument("--polytomy-seed", type=int, default=None, help="Optional random seed for reproducible stochastic polytomy resolving.")


def _add_plot_args(parser: argparse.ArgumentParser, include_metadata: bool) -> None:
    if include_metadata:
        parser.add_argument("--metadata", required=True, help="Path to metadata.csv.")
        parser.add_argument("--out-dir", default=None, help="Directory for output PDFs. Defaults to metadata directory.")
    parser.add_argument(
        "--change-threshold",
        type=float,
        default=0.9,
        help=(
            "Threshold for classifying a subtree as change-point positive. "
            "Subtrees >= threshold use two-stage inference; <= (1-threshold) use one-stage inference. Default: 0.9."
        ),
    )
    parser.add_argument("--bin-days", type=int, default=14, help="Time-bin width in days for R_t aggregation (positive integer; default: 14).")
    parser.add_argument(
        "--recency-strength",
        type=float,
        default=0.0,
        help=(
            "Recency-weighting strength for R_t aggregation (>=0). "
            "0 disables recency weighting; larger values give more weight to contributions closer to each bin (default: 0)."
        ),
    )
    parser.add_argument(
        "--rt-quantiles",
        default="25,75",
        help="Two comma-separated percentile values for Rt uncertainty output (default: 25,75).",
    )
    parser.add_argument(
        "--show-change-point-distribution",
        dest="show_change_points",
        action="store_true",
        help="Overlay the inferred change-point distribution on R0_subtree.pdf.",
    )


def _run_predict(args: argparse.Namespace) -> int:
    from .prediction import predict

    predict(
        tree=args.tree,
        sampling=args.sampling,
        pre_model=args.pre_model,
        out_dir=args.out_dir,
        subtree_sizes=args.subtree_sizes,
        sampling_times=args.sampling_times,
        n_jobs=args.n_jobs,
        batch_size=args.batch_size,
        resume=not args.no_resume,
        keep_tree_files=args.keep_tree_files,
        n_replicates=args.n_replicates,
        date_file=args.date_file,
        lsd2_bin=args.lsd2_bin,
        gotree_bin=args.gotree_bin,
        polytomy_seed=args.polytomy_seed,
    )
    return 0


def _run_combined(args: argparse.Namespace) -> int:
    from .workflow import run

    result = run(
        tree=args.tree,
        sampling=args.sampling,
        pre_model=args.pre_model,
        out_dir=args.out_dir,
        subtree_sizes=args.subtree_sizes,
        sampling_times=args.sampling_times,
        n_jobs=args.n_jobs,
        batch_size=args.batch_size,
        resume=not args.no_resume,
        keep_tree_files=args.keep_tree_files,
        n_replicates=args.n_replicates,
        date_file=args.date_file,
        lsd2_bin=args.lsd2_bin,
        gotree_bin=args.gotree_bin,
        polytomy_seed=args.polytomy_seed,
        outputs="both",
        start_col="node_age",
        end_col="last_sample_date",
        change_threshold=args.change_threshold,
        bin_days=args.bin_days,
        recency_strength=args.recency_strength,
        rt_quantiles=args.rt_quantiles,
        show_change_points=args.show_change_points,
    )
    for name, path in result["plots"].items():
        print(f"{name}: {path}")
    return 0


def _run_plot(args: argparse.Namespace) -> int:
    from .plotting import plot

    written = plot(
        metadata=args.metadata,
        out_dir=args.out_dir,
        outputs="both",
        start_col="node_age",
        end_col="last_sample_date",
        change_threshold=args.change_threshold,
        bin_days=args.bin_days,
        recency_strength=args.recency_strength,
        rt_quantiles=args.rt_quantiles,
        show_change_points=args.show_change_points,
    )
    for name, path in written.items():
        print(f"{name}: {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else argv
    if raw_argv in (["-h"], ["--help"]):
        public_parser = build_public_parser()
        public_parser.print_help()
        return 0

    argv = _normalize_argv(raw_argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def _normalize_argv(argv: list[str] | None) -> list[str]:
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        return argv

    known_subcommands = {"run", "predict", "plot"}
    help_flags = {"-h", "--help"}

    first = argv[0]
    if first in known_subcommands or first in help_flags:
        return argv

    # Public UX: treat bare options as the default combined pipeline.
    return ["run", *argv]


if __name__ == "__main__":
    raise SystemExit(main())

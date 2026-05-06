import pytest

from PhyloRt.cli import build_parser
from PhyloRt.cli import _normalize_argv


def test_predict_help_smoke(capsys):
    parser = build_parser()
    assert parser.prog == "PhyloRt"
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["predict", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "predict" in out
    assert "--date-file" in out
    assert "--tree-input" not in out
    assert "--aln-len" not in out
    assert "--tree-units" not in out
    assert "H1N1" in out
    assert "RSVA" in out
    assert "RSVB" in out
    assert "--contact-file" not in out
    assert "--lsd-clock-sd" not in out
    assert "--genome-length" not in out
    assert "--clock-file" not in out
    assert "--lsd-root-date" not in out
    assert "--timescale-dir" not in out
    assert "--latest-date" not in out
    assert "--keep-iterate" not in out
    assert "--keep-tree-files" in out
    assert "--subtree-sizes" in out
    assert "--n-replicates" in out
    assert "--tip-sizes" not in out


def test_plot_help_smoke(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["plot", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "plot" in out
    assert "--show-change-point-distribution" in out


def test_run_help_smoke(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["run", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "run" in out
    assert "--keep-tree-files" in out
    assert "--date-file" in out
    assert "--latest-date" not in out
    assert "--keep-iterate" not in out
    assert "--subtree-sizes" in out
    assert "--n-replicates" in out
    assert "--tip-sizes" not in out
    assert "--tree-input" not in out
    assert "--aln-len" not in out
    assert "--show-change-point-distribution" in out
    assert "--weight-col" not in out
    assert "--timescale-dir" not in out


def test_predict_default_n_replicates():
    parser = build_parser()
    args = parser.parse_args(
        [
            "predict",
            "--tree",
            "trees.nwk",
            "--date-file",
            "dates.tsv",
            "--sampling",
            "0.1",
            "--pre-model",
            "FLUB",
            "--out-dir",
            "out",
        ]
    )
    assert args.n_replicates == 1


def test_default_command_normalizes_to_run():
    normalized = _normalize_argv(
        [
            "--tree",
            "trees.nwk",
            "--date-file",
            "dates.tsv",
            "--sampling",
            "0.1",
            "--pre-model",
            "FLUB",
            "--out-dir",
            "out",
        ]
    )
    assert normalized[0] == "run"


def test_explicit_predict_not_rewritten():
    normalized = _normalize_argv(["predict", "--tree", "trees.nwk"])
    assert normalized[0] == "predict"

import importlib
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from PhyloRt import transform
from PhyloRt.transform import dating


def test_public_transform_imports():
    from PhyloRt import plot, predict, run, timescale_tree
    from PhyloRt.transform import timescale_tree_set

    assert callable(predict)
    assert callable(plot)
    assert callable(run)
    assert callable(timescale_tree)
    assert callable(timescale_tree_set)
    assert importlib.util.find_spec("PhyloRt.timescaling") is None


def test_default_clock_file_resolution():
    assert transform.default_clock_file("FLUB").name == "FLUB_clock.txt"
    assert transform.default_clock_file("H3N2").exists()
    assert "clock_rates" in str(transform.default_clock_file("H3N2"))
    assert transform.default_clock_file("RSVB").name == "RSVB_clock.txt"


def test_pathogen_timescale_defaults_from_table():
    flub = transform.pathogen_timescale_defaults("FLUB")
    h3n2 = transform.pathogen_timescale_defaults("H3N2")
    h1n1 = transform.pathogen_timescale_defaults("H1N1")
    rsva = transform.pathogen_timescale_defaults("RSVA")
    rsvb = transform.pathogen_timescale_defaults("RSVB")
    covid = transform.pathogen_timescale_defaults("COVID")

    assert flub.clock_rate == pytest.approx(2e-3)
    assert flub.clock_sd == pytest.approx(2.12e-4)
    assert flub.genome_length == 1885
    assert h3n2.clock_rate == pytest.approx(5e-3)
    assert h3n2.clock_sd == pytest.approx(1.02e-4)
    assert h3n2.genome_length == 1737
    assert h1n1.clock_rate == pytest.approx(4.4e-3)
    assert h1n1.clock_sd == pytest.approx(1.02e-4)
    assert h1n1.genome_length == 1752
    assert rsva.clock_rate == pytest.approx(6.47e-4)
    assert rsva.clock_sd == pytest.approx(4.64e-5)
    assert rsva.genome_length == 15225
    assert rsvb.clock_rate == pytest.approx(7.76e-4)
    assert rsvb.clock_sd == pytest.approx(4.31e-5)
    assert rsvb.genome_length == 15222
    assert covid.clock_rate == pytest.approx(8e-4)
    assert covid.clock_sd == pytest.approx(4e-4)
    assert covid.genome_length == 29903


def test_split_newick_records_assigns_stable_tree_ids(tmp_path):
    tree_file = tmp_path / "trees.nwk"
    tree_file.write_text("(A:1,B:1);\n(C:1,D:1);\n")

    records = transform.split_newick_records(tree_file)

    assert [record.tree_id for record in records] == ["tree_0001", "tree_0002"]
    assert records[0].newick == "(A:1,B:1);"
    assert records[1].newick == "(C:1,D:1);"


def test_parse_user_date_file_without_count_row(tmp_path):
    date_file = tmp_path / "dates.tsv"
    date_file.write_text("A\t2020-01-01\nB\t2020-01-03\n")

    parsed = transform.parse_date_file(date_file)

    assert parsed["A"].strftime("%Y-%m-%d") == "2020-01-01"
    assert parsed["B"].strftime("%Y-%m-%d") == "2020-01-03"


def test_write_lsd_date_file_adds_count_row(tmp_path):
    date_map = {
        "A": datetime.strptime("2020-01-01", "%Y-%m-%d"),
        "B": datetime.strptime("2020-01-03", "%Y-%m-%d"),
    }
    out_file = tmp_path / "lsd_dates.tsv"

    latest = transform.write_lsd_date_file(date_map, ["A", "B"], out_file, "tree_0001")

    assert latest.strftime("%Y-%m-%d") == "2020-01-03"
    assert out_file.read_text().splitlines() == ["2", "A\t2020-01-01", "B\t2020-01-03"]


def test_write_lsd_date_file_reports_missing_tips(tmp_path):
    date_map = {"A": datetime.strptime("2020-01-01", "%Y-%m-%d")}

    with pytest.raises(ValueError, match="missing 1 tips"):
        transform.write_lsd_date_file(date_map, ["A", "B"], tmp_path / "lsd_dates.tsv", "tree_0001")


def test_timescale_tree_set_writes_per_tree_lsd_dates_and_combined_output(tmp_path, monkeypatch):
    tree_file = tmp_path / "trees.nwk"
    date_file = tmp_path / "dates.tsv"
    tree_file.write_text("(A:1,B:1);\n(C:1,D:1);\n")
    date_file.write_text("A\t2020-01-01\nB\t2020-01-03\nC\t2020-02-01\nD\t2020-02-04\n")

    def fake_tip_names(path):
        text = Path(path).read_text()
        return ["A", "B"] if "A:" in text else ["C", "D"]

    def fake_timescale_genetic_tree(**kwargs):
        dated = Path(kwargs["out_dir"]) / "time_scaled_tree.nwk"
        if "tree_0001" in str(kwargs["out_dir"]):
            dated.write_text("(A:1,B:1);\n")
        else:
            dated.write_text("(C:1,D:1);\n")
        return dated

    monkeypatch.setattr(dating, "tree_tip_names", fake_tip_names)
    monkeypatch.setattr(dating, "timescale_genetic_tree", fake_timescale_genetic_tree)

    infos = transform.timescale_tree_set(
        tree=tree_file,
        date_file=date_file,
        out_dir=tmp_path / "out",
        pre_model="FLUB",
    )

    assert [info.tree_id for info in infos] == ["tree_0001", "tree_0002"]
    assert infos[0].latest_date.strftime("%Y-%m-%d") == "2020-01-03"
    assert (tmp_path / "out" / "trees" / "tree_0001" / "lsd_dates.tsv").read_text().splitlines()[0] == "2"
    assert (tmp_path / "out" / "time_scaled_trees.nwk").read_text().splitlines() == ["(A:1,B:1);", "(C:1,D:1);"]


def test_resolve_polytomies_writes_binary_tree(tmp_path):
    Tree = pytest.importorskip("ete3").Tree
    in_tree = tmp_path / "polytomy.nwk"
    out_tree = tmp_path / "resolved.nwk"
    in_tree.write_text("(A:0.1,B:0.1,C:0.1,D:0.1);")

    stats = transform.resolve_polytomies(in_tree, out_tree, sequence_length=1000, random_seed=7)
    resolved = Tree(str(out_tree), format=1)

    assert stats.tips == 4
    assert out_tree.exists()
    assert all(len(node.children) <= 2 for node in resolved.traverse())


def test_strip_newick_annotations_removes_date_comments(tmp_path):
    tree = tmp_path / "annotated.nwk"
    tree.write_text('(A[&date="2020-01-01"]:1,(B[&date="2020-01-02"]:1,C:1):1);\n')

    assert transform.strip_newick_annotations(tree) is True
    assert tree.read_text() == "(A:1,(B:1,C:1):1);\n"
    assert transform.strip_newick_annotations(tree) is False


def test_timescale_genetic_tree_runs_lsd2_and_gotree(tmp_path):
    pytest.importorskip("ete3")
    tree = tmp_path / "tree.nwk"
    dates = tmp_path / "dates.tsv"
    tree.write_text("((A:0.1,B:0.1):0.1,C:0.2);")
    dates.write_text("A\t2020-01-01\nB\t2020-01-02\nC\t2020-01-03\n")
    commands = []

    def fake_run(command, cwd, check, stdout, stderr, text):
        del check, stdout, stderr, text
        commands.append(command)
        if command[0] == "fake-lsd2":
            (tmp_path / "timescale" / "resolved_tree.nwk.result.date.nexus").write_text("#NEXUS\n")
        elif command[0] == "fake-gotree":
            out_path = command[command.index("-o") + 1]
            Path(out_path).write_text('((A[&date="2020-01-01"]:1,B[&date="2020-01-02"]:1):1,C:2);\n')
        return SimpleNamespace(stdout="ok")

    dated_tree = transform.timescale_genetic_tree(
        tree=tree,
        date_file=dates,
        out_dir=tmp_path / "timescale",
        pre_model="FLUB",
        lsd2_bin="fake-lsd2",
        gotree_bin="fake-gotree",
        run_command_func=fake_run,
    )

    assert dated_tree.exists()
    assert commands[0][0] == "fake-lsd2"
    assert commands[0][commands[0].index("-i") + 1] == "resolved_tree.nwk"
    assert commands[0][commands[0].index("-d") + 1] == str(dates.resolve())
    assert commands[0][commands[0].index("-s") + 1] == "1885"
    assert commands[0][commands[0].index("-q") + 1] == "0.000212"
    assert commands[0][commands[0].index("-w") + 1].endswith("FLUB_clock.txt")
    assert commands[1][:3] == ["fake-gotree", "reformat", "newick"]
    assert "[&date" not in dated_tree.read_text()
    assert "A:1" in dated_tree.read_text()
    assert not (tmp_path / "timescale" / "phylort_timescale_manifest.json").exists()
    assert (tmp_path / "timescale" / "extraction_log.txt").exists()
    assert not (tmp_path / "timescale" / "timescaling_log.txt").exists()
    assert not (tmp_path / "timescale" / "resolved_tree.nwk").exists()
    assert not (tmp_path / "timescale" / "resolved_tree.nwk.result.date.nexus").exists()


def test_timescale_genetic_tree_uses_absolute_paths_for_relative_outdir(tmp_path, monkeypatch):
    pytest.importorskip("ete3")
    monkeypatch.chdir(tmp_path)
    tree = tmp_path / "tree.nwk"
    dates = tmp_path / "dates.tsv"
    tree.write_text("((A:0.1,B:0.1):0.1,C:0.2);")
    dates.write_text("A\t2020-01-01\nB\t2020-01-02\nC\t2020-01-03\n")
    commands = []

    def fake_run(command, cwd, check, stdout, stderr, text):
        del check, stdout, stderr, text
        commands.append(command)
        if command[0] == "fake-lsd2":
            Path(cwd, "resolved_tree.nwk.result.date.nexus").write_text("#NEXUS\n")
        elif command[0] == "fake-gotree":
            out_path = Path(command[command.index("-o") + 1])
            assert out_path.is_absolute()
            out_path.write_text("((A:1,B:1):1,C:2);\n")
        return SimpleNamespace(stdout="ok")

    dated_tree = transform.timescale_genetic_tree(
        tree=tree,
        date_file=dates,
        out_dir="timescale",
        pre_model="FLUB",
        lsd2_bin="fake-lsd2",
        gotree_bin="fake-gotree",
        run_command_func=fake_run,
    )

    assert dated_tree == tmp_path / "timescale" / "time_scaled_tree.nwk"
    assert Path(commands[1][commands[1].index("-i") + 1]).is_absolute()
    assert dated_tree.exists()
    assert not (tmp_path / "timescale" / "resolved_tree.nwk").exists()


def test_timescale_genetic_tree_uses_rsvb_defaults(tmp_path):
    pytest.importorskip("ete3")
    tree = tmp_path / "tree.nwk"
    dates = tmp_path / "dates.tsv"
    tree.write_text("((A:0.1,B:0.1):0.1,C:0.2);")
    dates.write_text("A\t2020-01-01\nB\t2020-01-02\nC\t2020-01-03\n")
    commands = []

    def fake_run(command, cwd, check, stdout, stderr, text):
        del check, stdout, stderr, text
        commands.append(command)
        if command[0] == "fake-lsd2":
            (tmp_path / "timescale" / "resolved_tree.nwk.result.date.nexus").write_text("#NEXUS\n")
        elif command[0] == "fake-gotree":
            out_path = command[command.index("-o") + 1]
            Path(out_path).write_text("((A:1,B:1):1,C:2);\n")
        return SimpleNamespace(stdout="ok")

    transform.timescale_genetic_tree(
        tree=tree,
        date_file=dates,
        out_dir=tmp_path / "timescale",
        pre_model="RSVB",
        lsd2_bin="fake-lsd2",
        gotree_bin="fake-gotree",
        run_command_func=fake_run,
    )

    assert commands[0][commands[0].index("-s") + 1] == "15222"
    assert commands[0][commands[0].index("-q") + 1] == "4.31e-05"
    assert commands[0][commands[0].index("-w") + 1].endswith("RSVB_clock.txt")


def test_timescale_genetic_tree_reruns_without_manifest_state(tmp_path):
    pytest.importorskip("ete3")
    tree = tmp_path / "tree.nwk"
    dates = tmp_path / "dates.tsv"
    out_dir = tmp_path / "timescale"
    out_dir.mkdir()
    tree.write_text("((A:0.1,B:0.1):0.1,C:0.2);")
    dates.write_text("A\t2020-01-01\nB\t2020-01-02\nC\t2020-01-03\n")
    (out_dir / "phylort_timescale_manifest.json").write_text("{}\n")
    (out_dir / "time_scaled_tree.nwk").write_text("(OLD:1);\n")
    (out_dir / "resolved_tree.nwk.result").write_text("old intermediate\n")
    commands = []

    def fake_run(command, cwd, check, stdout, stderr, text):
        del check, stdout, stderr, text
        commands.append(command)
        if command[0] == "fake-lsd2":
            (out_dir / "resolved_tree.nwk.result.date.nexus").write_text("#NEXUS\n")
        elif command[0] == "fake-gotree":
            out_path = command[command.index("-o") + 1]
            Path(out_path).write_text("((A:1,B:1):1,C:2);\n")
        return SimpleNamespace(stdout="ok")

    dated_tree = transform.timescale_genetic_tree(
        tree=tree,
        date_file=dates,
        out_dir=out_dir,
        lsd2_bin="fake-lsd2",
        gotree_bin="fake-gotree",
        run_command_func=fake_run,
    )

    assert [command[0] for command in commands] == ["fake-lsd2", "fake-gotree"]
    assert dated_tree.read_text() == "((A:1,B:1):1,C:2);\n"
    assert not (out_dir / "phylort_timescale_manifest.json").exists()
    assert not (out_dir / "resolved_tree.nwk.result").exists()

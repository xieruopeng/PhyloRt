import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "simulators" / "convert_simulated_tree.py"
README = PROJECT_ROOT / "simulators" / "README.md"


def _load_converter():
    spec = importlib.util.spec_from_file_location("convert_simulated_tree", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_converter_help_smoke():
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=PROJECT_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert "--tree" in completed.stdout
    assert "--output-tree" in completed.stdout
    assert "--seq-length" in completed.stdout
    assert "--pre-model" not in completed.stdout
    assert "--out-dir" not in completed.stdout
    assert "--seed" not in completed.stdout


def test_write_lsd_dates_uses_tip_count_row(tmp_path):
    pytest.importorskip("ete3")
    converter = _load_converter()
    Tree = converter._tree_class()
    tree = Tree("((A:365,B:730):0,C:1095);", format=1)

    out_file = tmp_path / "lsd_dates.tsv"
    converter.write_lsd_dates_from_simulated_tree(tree, out_file)

    lines = out_file.read_text().splitlines()
    assert lines[0] == "3"
    assert "A\t2001" in lines
    assert "B\t2002" in lines
    assert "C\t2003" in lines


def test_simulate_genetic_distance_tree_is_deterministic_with_rng():
    pytest.importorskip("ete3")
    converter = _load_converter()
    Tree = converter._tree_class()
    tree = Tree("(A:365,B:730);", format=1)

    first_counts, first_genetic = converter.simulate_genetic_distance_tree(
        tree,
        seq_length=1000,
        clock_rate_mean=0.5,
        clock_rate_sd=0.0,
        rng=np.random.default_rng(7),
    )
    second_counts, second_genetic = converter.simulate_genetic_distance_tree(
        tree,
        seq_length=1000,
        clock_rate_mean=0.5,
        clock_rate_sd=0.0,
        rng=np.random.default_rng(7),
    )

    assert first_counts.write(format=5) == second_counts.write(format=5)
    assert first_genetic.write(format=5) == second_genetic.write(format=5)


def test_convert_simulated_tree_runs_lsd2_and_gotree_with_cleanup(tmp_path):
    pytest.importorskip("ete3")
    converter = _load_converter()
    tree = tmp_path / "simulated_tree.nwk"
    output = tmp_path / "simulated_tree.lsd2.nwk"
    tree.write_text("((A:365,B:365):0,C:730);\n")

    commands = []
    observed_dates = []
    observed_clock = []

    def fake_run(command, cwd, check, stdout, stderr, text):
        del check, stdout, stderr, text
        commands.append(command)
        work_dir = Path(cwd)
        if command[0] == "lsd2":
            observed_dates.extend((work_dir / command[command.index("-d") + 1]).read_text().splitlines())
            observed_clock.append((work_dir / command[command.index("-w") + 1]).read_text().strip())
            (work_dir / "resolved_tree.nwk.result.date.nexus").write_text("#NEXUS\n")
        elif command[0] == "gotree":
            out_path = Path(command[command.index("-o") + 1])
            out_path.write_text('(A[&date="2001"]:1,(B:1,C:1):1);\n')
        return SimpleNamespace(stdout="ok")

    result = converter.convert_simulated_tree(
        tree=tree,
        output_tree=output,
        seq_length=1737,
        clock_rate_mean=0.005,
        clock_rate_sd=0.000102,
        run_command_func=fake_run,
        rng=np.random.default_rng(11),
    )

    assert result == output.resolve()
    assert output.read_text() == "(A:1,(B:1,C:1):1);\n"
    assert [command[0] for command in commands] == ["lsd2", "gotree"]
    assert commands[0][commands[0].index("-s") + 1] == "1737"
    assert commands[0][commands[0].index("-q") + 1] == "0.000102"
    assert commands[1][:3] == ["gotree", "reformat", "newick"]
    assert observed_dates[0] == "3"
    assert observed_clock == ["0.005"]
    assert not any(path.name.startswith(".phylort_sim_") for path in tmp_path.iterdir())


def test_simulators_readme_matches_converter_interface():
    text = README.read_text()

    assert "generate_bdei" in text
    assert "generate_mtbd" in text
    assert "--output-tree" in text
    assert "--seq-length 1737" in text
    assert "--clock-rate-mean 0.005" in text
    assert "--clock-rate-sd 0.000102" in text
    assert "--seq-length 29903" in text
    assert "--clock-rate-mean 0.0008" in text
    assert "--clock-rate-sd 0.0004" in text
    assert "--pre-model" not in text
    assert "--out-dir" not in text
    assert "--seed" not in text

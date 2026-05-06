from datetime import datetime
import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from PhyloRt import prediction


def test_parse_latest_date_calendar_and_decimal():
    assert prediction.parse_latest_date("2021-09-01") == datetime(2021, 9, 1)
    parsed = prediction.parse_latest_date("2020.5")
    assert parsed.year == 2020
    assert parsed.month in (7, 6)


def test_get_prop_for_date():
    props = [0.1, 0.2, 0.3]
    times = ["2020-03-01", "2020-06-01"]
    assert prediction.get_prop_for_date(datetime(2020, 1, 1), times, props) == 0.1
    assert prediction.get_prop_for_date(datetime(2020, 4, 1), times, props) == 0.2
    assert prediction.get_prop_for_date(datetime(2020, 7, 1), times, props) == 0.3


def test_model_path_resolution():
    path = prediction.model_path("FLUB", 50, "classification", model_dir="/models")
    assert str(path).endswith("BDEI_flub_Rt_50_tips_classification.h5")
    assert str(prediction.model_path("H1N1", 50, "classification", model_dir="/models")).endswith(
        "BDEI_H3N2_Rt_50_tips_classification.h5"
    )
    assert str(prediction.model_path("RSVA", 50, "classification", model_dir="/models")).endswith(
        "BDEI_H3N2_Rt_50_tips_classification.h5"
    )
    assert str(prediction.model_path("RSVB", 50, "classification", model_dir="/models")).endswith(
        "BDEI_H3N2_Rt_50_tips_classification.h5"
    )


def test_process_tree_shape_if_ete3_available():
    pytest.importorskip("ete3")
    encoded = prediction.process_tree("((A:1,B:1):1,(C:1,D:1):1);")
    assert encoded.shape == (1000, 18)


def test_predict_with_mocked_models(monkeypatch, tmp_path):
    metadata = pd.DataFrame(
        {
            "subtree_id": ["1_50"],
            "subtree_size": [50],
            "node_age": ["2020-01-01"],
            "first_sample_date": ["2020-01-01"],
            "last_sample_date": ["2020-02-01"],
            "mean_sample_date": ["2020-01-15"],
            "sampling_proportion": [0.123456],
            "node_weight": [0.54321],
            "subtree_nwk_file": ["dummy.nwk"],
            "last_sample_to_full_tree": [0.23456],
            "mean_sample_to_full_tree": [0.34567],
        }
    )

    def fake_extract(*args, **kwargs):
        out_dir = args[7]
        iterate_dir = out_dir / "trees" / "tree_0001" / "iterate"
        iterate_dir.mkdir(parents=True, exist_ok=True)
        (iterate_dir / "dummy.nwk").write_text("(A:1,B:1);")
        df = metadata.copy()
        df["tree_id"] = ["tree_0001"]
        df["subtree_nwk_file"] = [str(iterate_dir / "dummy.nwk")]
        return df

    def fake_timescale_set(**kwargs):
        tree_dir = kwargs["out_dir"] / "trees" / "tree_0001"
        tree_dir.mkdir(parents=True, exist_ok=True)
        dated_tree = tree_dir / "time_scaled_tree.nwk"
        dated_tree.write_text("(A:1,B:1);\n")
        (kwargs["out_dir"] / "time_scaled_trees.nwk").write_text("(A:1,B:1);\n")
        return [SimpleNamespace(tree_id="tree_0001", dated_tree=dated_tree, tree_dir=tree_dir, latest_date=datetime(2020, 2, 1), tip_count=2)]

    from PhyloRt import transform
    predict_module = importlib.import_module("PhyloRt.prediction.predict")
    models_module = importlib.import_module("PhyloRt.prediction.models")

    monkeypatch.setattr(transform, "timescale_tree_set", fake_timescale_set)
    monkeypatch.setattr(predict_module, "extract_subtrees", fake_extract)
    monkeypatch.setattr(models_module, "process_tree", lambda _path: pd.DataFrame(np.ones((1000, 18))))

    class FakeModel:
        def __init__(self, path):
            self.path = path

        def predict(self, _array, verbose=0):
            if "classification" in self.path:
                return np.array([[0.95]])
            if "R2_1" in self.path:
                return np.array([[1.4]])
            if "R2_2" in self.path:
                return np.array([[0.7]])
            if "breakpoint" in self.path:
                return np.array([[0.35]])
            return np.array([[1.1]])

    result = prediction.predict(
        tree="tree.nwk",
        sampling="0.1",
        pre_model="FLUB",
        out_dir=tmp_path,
        subtree_sizes=(50,),
        date_file="dates.tsv",
        n_jobs=1,
        load_model_func=FakeModel,
    )

    assert result.loc[0, "change"] == pytest.approx(0.95)
    assert result.loc[0, "R1"] == pytest.approx(1.4)
    assert result.loc[0, "R2"] == pytest.approx(0.7)
    assert (tmp_path / "metadata.csv").exists()
    assert "tip_weight" not in result.columns
    assert "R_naught" not in result.columns
    assert "subtree_nwk_file" not in result.columns
    assert "replicate_seed" not in result.columns
    assert result.loc[0, "sampling_proportion"] == pytest.approx(0.1235)
    assert result.loc[0, "node_weight"] == pytest.approx(0.5432)
    assert "0.1235" in (tmp_path / "metadata.csv").read_text()
    assert not (tmp_path / "trees").exists()


def test_keep_tree_files_preserves_tree_folder(monkeypatch, tmp_path):
    metadata = pd.DataFrame(
        {
            "subtree_id": ["1_50"],
            "subtree_size": [50],
            "node_age": ["2020-01-01"],
            "first_sample_date": ["2020-01-01"],
            "last_sample_date": ["2020-02-01"],
            "mean_sample_date": ["2020-01-15"],
            "sampling_proportion": [0.1],
            "node_weight": [0.5],
            "subtree_nwk_file": ["dummy.nwk"],
            "last_sample_to_full_tree": [0.2],
            "mean_sample_to_full_tree": [0.3],
        }
    )

    def fake_extract(*args, **kwargs):
        out_dir = args[7]
        iterate_dir = out_dir / "trees" / "tree_0001" / "iterate"
        iterate_dir.mkdir(parents=True, exist_ok=True)
        (iterate_dir / "dummy.nwk").write_text("(A:1,B:1);")
        df = metadata.copy()
        df["tree_id"] = ["tree_0001"]
        df["subtree_nwk_file"] = [str(iterate_dir / "dummy.nwk")]
        return df

    def fake_timescale_set(**kwargs):
        tree_dir = kwargs["out_dir"] / "trees" / "tree_0001"
        tree_dir.mkdir(parents=True, exist_ok=True)
        dated_tree = tree_dir / "time_scaled_tree.nwk"
        dated_tree.write_text("(A:1,B:1);\n")
        (kwargs["out_dir"] / "time_scaled_trees.nwk").write_text("(A:1,B:1);\n")
        return [SimpleNamespace(tree_id="tree_0001", dated_tree=dated_tree, tree_dir=tree_dir, latest_date=datetime(2020, 2, 1), tip_count=2)]

    from PhyloRt import transform
    predict_module = importlib.import_module("PhyloRt.prediction.predict")
    models_module = importlib.import_module("PhyloRt.prediction.models")

    monkeypatch.setattr(transform, "timescale_tree_set", fake_timescale_set)
    monkeypatch.setattr(predict_module, "extract_subtrees", fake_extract)
    monkeypatch.setattr(models_module, "process_tree", lambda _path: pd.DataFrame(np.ones((1000, 18))))

    class FakeModel:
        def __init__(self, path):
            self.path = path

        def predict(self, _array, verbose=0):
            return np.array([[0.1]])

    result = prediction.predict(
        tree="tree.nwk",
        sampling="0.1",
        pre_model="FLUB",
        out_dir=tmp_path,
        subtree_sizes=(50,),
        date_file="dates.tsv",
        keep_tree_files=True,
        load_model_func=FakeModel,
    )

    assert (tmp_path / "trees" / "tree_0001" / "iterate").exists()
    assert result["subtree_id"].tolist() == ["1_50"]
    assert result["subtree_nwk_file"].tolist() == ["trees/tree_0001/iterate/dummy.nwk"]


def test_predict_multiple_replicates_uses_derived_seeds_and_numbered_trees(monkeypatch, tmp_path):
    tree = tmp_path / "trees.nwk"
    date_file = tmp_path / "dates.tsv"
    tree.write_text("(A:1,B:1);")
    date_file.write_text("A\t2020-01-01\nB\t2020-01-02\n")

    metadata = pd.DataFrame(
        {
            "tree_id": ["tree_0001"],
            "node_index": [1],
            "subtree_size": [50],
            "node_age": ["2020-01-01"],
            "first_sample_date": ["2020-01-01"],
            "last_sample_date": ["2020-02-01"],
            "mean_sample_date": ["2020-01-15"],
            "sampling_proportion": [0.1],
            "node_weight": [0.5],
            "last_sample_to_full_tree": [0.2],
            "mean_sample_to_full_tree": [0.3],
        }
    )
    calls = []

    def fake_timescale_set(**kwargs):
        calls.append(kwargs)
        tree_dir = kwargs["out_dir"] / "trees" / "tree_0001"
        tree_dir.mkdir(parents=True, exist_ok=True)
        dated_tree = tree_dir / "time_scaled_tree.nwk"
        dated_tree.write_text("(A:1,B:1);\n")
        (kwargs["out_dir"] / "time_scaled_trees.nwk").write_text(f"(A:{len(calls)},B:1);\n")
        return [
            SimpleNamespace(
                tree_id="tree_0001",
                dated_tree=dated_tree,
                tree_dir=tree_dir,
                latest_date=datetime(2020, 2, 1),
                tip_count=2,
            )
        ]

    def fake_extract(dated_trees, *args, **kwargs):
        out_dir = args[6]
        prefix = kwargs["subtree_prefix"]
        iterate_dir = out_dir / "trees" / "tree_0001" / "iterate"
        iterate_dir.mkdir(parents=True, exist_ok=True)
        subtree_path = iterate_dir / f"subtree_{prefix}_tree_0001_1_50.nwk"
        subtree_path.write_text("(A:1,B:1);")
        df = metadata.copy()
        df["subtree_id"] = [f"{prefix}_tree_0001_1_50"]
        df["subtree_nwk_file"] = [str(subtree_path)]
        return df

    from PhyloRt import transform
    predict_module = importlib.import_module("PhyloRt.prediction.predict")
    models_module = importlib.import_module("PhyloRt.prediction.models")

    monkeypatch.setattr(transform, "timescale_tree_set", fake_timescale_set)
    monkeypatch.setattr(predict_module, "extract_subtrees", fake_extract)
    monkeypatch.setattr(models_module, "process_tree", lambda _path: pd.DataFrame(np.ones((1000, 18))))

    class FakeModel:
        def __init__(self, path):
            self.path = path

        def predict(self, _array, verbose=0):
            return np.array([[0.2]])

    result = prediction.predict(
        tree=tree,
        sampling="0.1",
        pre_model="FLUB",
        out_dir=tmp_path / "out",
        subtree_sizes=(50,),
        date_file=date_file,
        n_replicates=3,
        polytomy_seed=7,
        load_model_func=FakeModel,
    )

    assert [call["random_seed"] for call in calls] == [7, 8, 9]
    assert result["replicate_id"].tolist() == [1, 2, 3]
    assert result["replicate_seed"].tolist() == [7, 8, 9]
    assert result["tree_id"].tolist() == [1, 1, 1]
    assert result["subtree_id"].tolist() == ["1_50", "1_50", "1_50"]
    assert "subtree_nwk_file" not in result.columns
    assert "R_naught" not in result.columns
    assert (tmp_path / "out" / "time_scaled_trees_1.nwk").read_text() == "(A:1,B:1);\n"
    assert (tmp_path / "out" / "time_scaled_trees_2.nwk").read_text() == "(A:2,B:1);\n"
    assert (tmp_path / "out" / "time_scaled_trees_3.nwk").read_text() == "(A:3,B:1);\n"
    assert not (tmp_path / "out" / "replicates").exists()
    assert (tmp_path / "out" / "metadata.csv").exists()


def test_predict_multiple_replicates_keep_tree_files_uses_relative_subtree_paths(monkeypatch, tmp_path):
    tree = tmp_path / "trees.nwk"
    date_file = tmp_path / "dates.tsv"
    tree.write_text("(A:1,B:1);")
    date_file.write_text("A\t2020-01-01\nB\t2020-01-02\n")

    def fake_timescale_set(**kwargs):
        tree_dir = kwargs["out_dir"] / "trees" / "tree_0001"
        tree_dir.mkdir(parents=True, exist_ok=True)
        dated_tree = tree_dir / "time_scaled_tree.nwk"
        dated_tree.write_text("(A:1,B:1);\n")
        (kwargs["out_dir"] / "time_scaled_trees.nwk").write_text("(A:1,B:1);\n")
        return [
            SimpleNamespace(
                tree_id="tree_0001",
                dated_tree=dated_tree,
                tree_dir=tree_dir,
                latest_date=datetime(2020, 2, 1),
                tip_count=2,
            )
        ]

    def fake_extract(dated_trees, *args, **kwargs):
        out_dir = args[6]
        prefix = kwargs["subtree_prefix"]
        iterate_dir = out_dir / "trees" / "tree_0001" / "iterate"
        iterate_dir.mkdir(parents=True, exist_ok=True)
        subtree_path = iterate_dir / f"subtree_{prefix}_tree_0001_1_50.nwk"
        subtree_path.write_text("(A:1,B:1);")
        return pd.DataFrame(
            {
                "tree_id": ["tree_0001"],
                "node_index": [1],
                "subtree_id": [f"{prefix}_tree_0001_1_50"],
                "subtree_size": [50],
                "node_age": ["2020-01-01"],
                "first_sample_date": ["2020-01-01"],
                "last_sample_date": ["2020-02-01"],
                "mean_sample_date": ["2020-01-15"],
                "sampling_proportion": [0.1],
                "node_weight": [0.5],
                "subtree_nwk_file": [str(subtree_path)],
                "last_sample_to_full_tree": [0.2],
                "mean_sample_to_full_tree": [0.3],
            }
        )

    from PhyloRt import transform
    predict_module = importlib.import_module("PhyloRt.prediction.predict")
    models_module = importlib.import_module("PhyloRt.prediction.models")

    monkeypatch.setattr(transform, "timescale_tree_set", fake_timescale_set)
    monkeypatch.setattr(predict_module, "extract_subtrees", fake_extract)
    monkeypatch.setattr(models_module, "process_tree", lambda _path: pd.DataFrame(np.ones((1000, 18))))

    class FakeModel:
        def __init__(self, path):
            self.path = path

        def predict(self, _array, verbose=0):
            return np.array([[0.2]])

    result = prediction.predict(
        tree=tree,
        sampling="0.1",
        pre_model="FLUB",
        out_dir=tmp_path / "out",
        subtree_sizes=(50,),
        date_file=date_file,
        n_replicates=2,
        keep_tree_files=True,
        load_model_func=FakeModel,
    )

    assert (tmp_path / "out" / "replicates" / "replicate_0001" / "trees").exists()
    assert result["subtree_id"].tolist() == ["1_50", "1_50"]
    assert result["subtree_nwk_file"].tolist() == [
        "replicates/replicate_0001/trees/tree_0001/iterate/subtree_replicate_0001_tree_0001_1_50.nwk",
        "replicates/replicate_0002/trees/tree_0001/iterate/subtree_replicate_0002_tree_0001_1_50.nwk",
    ]


def test_predict_multiple_replicates_resume_prediction_complete_skips_rerun(monkeypatch, tmp_path):
    tree = tmp_path / "trees.nwk"
    date_file = tmp_path / "dates.tsv"
    out_dir = tmp_path / "out"
    tree.write_text("(A:1,B:1);")
    date_file.write_text("A\t2020-01-01\nB\t2020-01-02\n")
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.DataFrame(
        {
            "replicate_id": [1],
            "replicate_seed": [7],
            "tree_id": [1],
            "subtree_id": ["1_50"],
            "subtree_size": [50],
            "change": [0.2],
            "R1": [1.1],
            "R2": [0.0],
            "breakpoint": [0.0],
        }
    )
    metadata.to_csv(out_dir / "metadata.csv", index=False)

    run_config = prediction.build_replicate_run_config(
        tree_file=tree,
        date_file=date_file,
        subtree_sizes=[50],
        sampling_props=[0.1],
        sampling_times=[],
        tree_units="years",
        pre_model="FLUB",
        n_replicates=3,
        replicate_seeds=[7, 8, 9],
    )
    prediction.save_manifest(out_dir, run_config, status="prediction_complete")

    from PhyloRt import transform
    predict_module = importlib.import_module("PhyloRt.prediction.predict")

    monkeypatch.setattr(
        transform,
        "timescale_tree_set",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("timescale_tree_set should not run")),
    )
    monkeypatch.setattr(
        predict_module,
        "extract_subtrees",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("extract_subtrees should not run")),
    )

    result = prediction.predict(
        tree=tree,
        sampling="0.1",
        pre_model="FLUB",
        out_dir=out_dir,
        subtree_sizes=(50,),
        date_file=date_file,
        n_replicates=3,
        polytomy_seed=7,
        resume=True,
    )

    pd.testing.assert_frame_equal(result, metadata)


def test_resume_manifest_rejects_changed_tree(tmp_path):
    tree = tmp_path / "tree.nwk"
    tree.write_text("(A:1,B:1);")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    config = prediction.build_extraction_config(
        tree,
        tree,
        [SimpleNamespace(tree_id="tree_0001", latest_date=datetime(2020, 1, 1), tip_count=2)],
        [50],
        [0.1],
        [],
        "years",
    )
    logger = prediction.setup_logging(out_dir)
    prediction.ensure_resume_manifest(out_dir, config, resume=True, logger=logger)

    tree.write_text("(A:1,C:1);")
    changed_config = prediction.build_extraction_config(
        tree,
        tree,
        [SimpleNamespace(tree_id="tree_0001", latest_date=datetime(2020, 1, 1), tip_count=2)],
        [50],
        [0.1],
        [],
        "years",
    )
    with pytest.raises(ValueError, match="Cannot resume"):
        prediction.ensure_resume_manifest(out_dir, changed_config, resume=True, logger=logger)


def test_resume_manifest_rejects_generated_outputs_without_manifest(tmp_path):
    tree = tmp_path / "tree.nwk"
    tree.write_text("(A:1,B:1);")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "metadata.csv").write_text("subtree_id\n")

    config = prediction.build_extraction_config(
        tree,
        tree,
        [SimpleNamespace(tree_id="tree_0001", latest_date=datetime(2020, 1, 1), tip_count=2)],
        [50],
        [0.1],
        [],
        "years",
    )
    logger = prediction.setup_logging(out_dir)

    with pytest.raises(ValueError, match="--no-resume"):
        prediction.ensure_resume_manifest(out_dir, config, resume=True, logger=logger)


def test_resume_manifest_allows_fresh_dating_outputs_without_manifest(tmp_path):
    tree = tmp_path / "tree.nwk"
    tree.write_text("(A:1,B:1);")
    out_dir = tmp_path / "out"
    (out_dir / "trees" / "tree_0001").mkdir(parents=True)
    (out_dir / "time_scaled_trees.nwk").write_text("(A:1,B:1);\n")

    config = prediction.build_extraction_config(
        tree,
        tree,
        [SimpleNamespace(tree_id="tree_0001", latest_date=datetime(2020, 1, 1), tip_count=2)],
        [50],
        [0.1],
        [],
        "years",
    )
    logger = prediction.setup_logging(out_dir)

    prediction.ensure_resume_manifest(out_dir, config, resume=True, logger=logger)

    assert (out_dir / prediction.MANIFEST_FILENAME).exists()


def test_clear_generated_outputs_removes_rt_table(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "Rt.csv").write_text("subtree_size,Rt\n")
    (out_dir / "time_scaled_trees_1.nwk").write_text("(A:1,B:1);\n")
    (out_dir / "replicates" / "replicate_0001").mkdir(parents=True)

    prediction.clear_generated_outputs(out_dir)

    assert not (out_dir / "Rt.csv").exists()
    assert not (out_dir / "time_scaled_trees_1.nwk").exists()
    assert not (out_dir / "replicates").exists()


def test_extract_subtrees_computes_node_weight_globally(tmp_path):
    pytest.importorskip("ete3")
    source_tree = tmp_path / "trees.nwk"
    date_file = tmp_path / "dates.tsv"
    tree1 = tmp_path / "trees" / "tree_0001" / "time_scaled_tree.nwk"
    tree2 = tmp_path / "trees" / "tree_0002" / "time_scaled_tree.nwk"
    tree1.parent.mkdir(parents=True)
    tree2.parent.mkdir(parents=True)
    source_tree.write_text("(A:1,B:1);\n(C:1,D:1);\n")
    date_file.write_text("A\t2020-01-10\nB\t2020-01-10\nC\t2020-01-11\nD\t2020-01-11\n")
    tree1.write_text("(A:1,B:1);\n")
    tree2.write_text("(C:1,D:1);\n")

    metadata = prediction.extract_subtrees(
        [
            SimpleNamespace(tree_id="tree_0001", dated_tree=tree1, tree_dir=tree1.parent, latest_date=datetime(2020, 1, 10), tip_count=2),
            SimpleNamespace(tree_id="tree_0002", dated_tree=tree2, tree_dir=tree2.parent, latest_date=datetime(2020, 1, 11), tip_count=2),
        ],
        source_tree,
        date_file,
        [2],
        [0.1],
        [],
        "years",
        tmp_path / "out",
        n_jobs=1,
        resume=False,
    )

    assert set(metadata["tree_id"]) == {"tree_0001", "tree_0002"}
    assert metadata.loc[metadata["tree_id"] == "tree_0001", "node_weight"].iloc[0] == pytest.approx(0.75)


def test_predict_genetic_tree_timescales_before_extraction(monkeypatch, tmp_path):
    from PhyloRt import transform
    predict_module = importlib.import_module("PhyloRt.prediction.predict")

    dated_tree = tmp_path / "out" / "trees" / "tree_0001" / "time_scaled_tree.nwk"
    dated_tree.parent.mkdir(parents=True, exist_ok=True)
    dated_tree.write_text("(A:1,B:1);")
    captured = {}

    def fake_timescale_set(**kwargs):
        captured["timescale"] = kwargs
        return [SimpleNamespace(tree_id="tree_0001", dated_tree=dated_tree, tree_dir=dated_tree.parent, latest_date=datetime(2020, 2, 1), tip_count=2)]

    def fake_extract(dated_trees, *args, **kwargs):
        captured["extract_tree"] = dated_trees[0].dated_tree
        return pd.DataFrame()

    monkeypatch.setattr(transform, "timescale_tree_set", fake_timescale_set)
    monkeypatch.setattr(predict_module, "extract_subtrees", fake_extract)

    result = prediction.predict(
        tree="genetic_tree.nwk",
        sampling="0.1",
        pre_model="FLUB",
        out_dir=tmp_path / "out",
        subtree_sizes=(50,),
        date_file="dates.tsv",
    )

    assert result.empty
    assert captured["timescale"]["tree"] == "genetic_tree.nwk"
    assert captured["timescale"]["date_file"] == "dates.tsv"
    assert captured["timescale"]["out_dir"] == tmp_path / "out"
    assert "aln_len" not in captured["timescale"]
    assert "timescale_dir" not in captured["timescale"]
    assert "resume" not in captured["timescale"]
    assert "latest_date" not in captured["timescale"]
    assert captured["extract_tree"] == dated_tree

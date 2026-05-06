from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from PhyloRt.plotting import (
    _plot_subtree_r,
    breakpoint_times,
    filter_to_bucket,
    plot,
    rt_summary_table,
    weighted_quantile,
)


FIXTURE = Path(__file__).parent / "fixtures" / "metadata.csv"


def test_weighted_quantile():
    q = weighted_quantile([1, 2, 3], [0.5], [1, 1, 1])
    assert q[0] == pytest.approx(1.5)


def test_filter_to_bucket():
    df = pd.DataFrame({"subtree_size": [20, 30, 80]})
    filtered = filter_to_bucket(df, 21, 50)
    assert filtered["subtree_size"].tolist() == [30]


def test_breakpoint_times():
    df = pd.read_csv(FIXTURE)
    times = breakpoint_times(df)
    assert len(times) == 2
    assert np.all(np.isfinite(times))


def test_plot_writes_default_pdfs(tmp_path):
    written = plot(FIXTURE, out_dir=tmp_path)
    assert set(written) == {"rt", "rt_table", "subtree_r"}
    assert written["rt"].exists()
    assert written["rt_table"].exists()
    assert written["subtree_r"].exists()
    assert written["rt"].name == "Rt.pdf"
    assert written["rt_table"].name == "Rt.csv"
    assert written["subtree_r"].name == "R0_subtree.pdf"
    assert "1.2000" in written["rt_table"].read_text()


def test_rt_summary_table_contains_iqr_bounds():
    table = rt_summary_table(pd.read_csv(FIXTURE), bin_days=30)
    assert list(table.columns) == ["subtree_size", "bin_start", "bin_end", "Rt", "Rt_q25", "Rt_q75"]
    assert not table.empty
    assert table["Rt"].notna().any()


def test_rt_summary_table_balances_replicates():
    df = pd.DataFrame(
        {
            "replicate_id": ["replicate_0001", "replicate_0002", "replicate_0002"],
            "subtree_size": [30, 30, 30],
            "node_age": ["2020-01-01", "2020-01-01", "2020-01-01"],
            "last_sample_date": ["2020-01-10", "2020-01-10", "2020-01-10"],
            "R1": [1.0, 3.0, 3.0],
            "R2": [0.0, 0.0, 0.0],
            "change": [0.0, 0.0, 0.0],
            "breakpoint": [0.0, 0.0, 0.0],
            "node_weight": [1.0, 1.0, 1.0],
        }
    )
    table = rt_summary_table(df, bin_days=30)
    row = table.loc[table["subtree_size"] == "21-50"].iloc[0]
    assert row["Rt"] == pytest.approx(2.0)


def test_subtree_r_hides_change_point_distribution_by_default(tmp_path):
    df = pd.read_csv(FIXTURE)
    fig = _plot_subtree_r(df, tmp_path / "R0_subtree.pdf", "node_age", "last_sample_date", "node_weight", 0.9)
    labels = [text.get_text() for text in fig.legends[0].get_texts()]
    assert len(fig.axes) == 4
    assert labels == [r"$R_{NC}$", r"$R_{C1}$", r"$R_{C2}$"]
    plt.close(fig)


def test_subtree_r_can_show_change_point_distribution(tmp_path):
    df = pd.read_csv(FIXTURE)
    fig = _plot_subtree_r(
        df,
        tmp_path / "R0_subtree_with_change_points.pdf",
        "node_age",
        "last_sample_date",
        "node_weight",
        0.9,
        show_change_points=True,
    )
    assert len(fig.axes) > 4
    plt.close(fig)

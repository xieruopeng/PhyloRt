"""PhyloRt public API."""

__all__ = ["predict", "plot", "run", "timescale_tree"]


def __getattr__(name):
    if name == "predict":
        from .prediction import predict

        return predict
    if name == "plot":
        from .plotting import plot

        return plot
    if name == "run":
        from .workflow import run

        return run
    if name == "timescale_tree":
        from .transform import timescale_tree

        return timescale_tree
    raise AttributeError(name)

"""Pytest configuration: headless Matplotlib before any test or package import."""

import matplotlib

matplotlib.use("Agg", force=True)

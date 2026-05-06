"""Compatibility wrapper for running the PhyloRt CLI from a source checkout."""

from PhyloRt.cli import main


if __name__ == "__main__":
    raise SystemExit(main())

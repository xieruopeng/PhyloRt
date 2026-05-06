"""Pathogen-specific time-scaling defaults for PhyloRt."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PathogenTimescaleDefaults:
    clock_file: str
    clock_rate: float
    clock_sd: float
    genome_length: int

PATHOGEN_TIMESCALE_DEFAULTS = {
    "COVID": PathogenTimescaleDefaults(
        clock_file="COVID_clock.txt",
        clock_rate=8e-4,
        clock_sd=4e-4,
        genome_length=29903,
    ),
    "H3N2": PathogenTimescaleDefaults(
        clock_file="H3N2_clock.txt",
        clock_rate=5e-3,
        clock_sd=1.02e-4,
        genome_length=1737,
    ),
    "FLUB": PathogenTimescaleDefaults(
        clock_file="FLUB_clock.txt",
        clock_rate=2e-3,
        clock_sd=2.12e-4,
        genome_length=1885,
    ),
    "H1N1": PathogenTimescaleDefaults(
        clock_file="H1N1_clock.txt",
        clock_rate=4.4e-3,
        clock_sd=1.02e-4,
        genome_length=1752,
    ),
    "RSVA": PathogenTimescaleDefaults(
        clock_file="RSVA_clock.txt",
        clock_rate=6.47e-4,
        clock_sd=4.64e-5,
        genome_length=15225,
    ),
    "RSVB": PathogenTimescaleDefaults(
        clock_file="RSVB_clock.txt",
        clock_rate=7.76e-4,
        clock_sd=4.31e-5,
        genome_length=15222,
    ),
}

def pathogen_timescale_defaults(pre_model: str) -> PathogenTimescaleDefaults:
    """Return pathogen-specific LSD2 defaults from the simulation table."""
    key = pre_model.upper()
    if key not in PATHOGEN_TIMESCALE_DEFAULTS:
        raise ValueError("pre_model must be one of COVID, H3N2, H1N1, FLUB, RSVA, or RSVB")
    return PATHOGEN_TIMESCALE_DEFAULTS[key]

def default_clock_file(pre_model: str) -> Path:
    """Return the bundled LSD2 clock file for a pretrained model family."""
    return Path(__file__).resolve().parents[1] / "clock_rates" / pathogen_timescale_defaults(pre_model).clock_file

"""Newick record handling and LSD2 date-file preparation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class NewickRecord:
    tree_id: str
    newick: str


@dataclass(frozen=True)
class DatedTreeInfo:
    tree_id: str
    source_tree: Path
    dated_tree: Path
    tree_dir: Path
    latest_date: datetime
    tip_count: int

def _tree_class():
    try:
        from ete3 import Tree
    except ImportError as exc:
        raise ImportError(
            "PhyloRt genetic tree preprocessing requires ete3. Install the "
            "package dependencies before running genetic tree time-scaling."
        ) from exc
    return Tree

def split_newick_records(tree_file: str | Path) -> list[NewickRecord]:
    """Split a Newick file containing one or more semicolon-terminated trees."""
    path = Path(tree_file)
    text = path.read_text(encoding="utf-8")
    records: list[str] = []
    buffer: list[str] = []
    bracket_depth = 0
    quote_char: str | None = None

    for char in text:
        buffer.append(char)
        if quote_char is not None:
            if char == quote_char:
                quote_char = None
            continue
        if char in {"'", '"'}:
            quote_char = char
            continue
        if char == "[":
            bracket_depth += 1
            continue
        if char == "]" and bracket_depth > 0:
            bracket_depth -= 1
            continue
        if char == ";" and bracket_depth == 0:
            record = "".join(buffer).strip()
            if record:
                records.append(record)
            buffer = []

    trailing = "".join(buffer).strip()
    if trailing:
        raise ValueError(f"Newick file has trailing text after the last complete tree: {tree_file}")
    if not records:
        raise ValueError(f"No semicolon-terminated Newick trees found in {tree_file}")

    return [
        NewickRecord(tree_id=f"tree_{idx:04d}", newick=record)
        for idx, record in enumerate(records, start=1)
    ]

def parse_date_file(date_file: str | Path) -> dict[str, datetime]:
    """Parse a user tip-date file without requiring an LSD2 count header."""
    path = Path(date_file)
    date_map: dict[str, datetime] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) == 1 and not date_map and parts[0].isdigit():
                continue
            if len(parts) < 2:
                raise ValueError(f"Invalid date-file row {line_number}: expected '<tip> <YYYY-MM-DD>'")
            tip_name, date_text = parts[0], parts[1]
            try:
                parsed = datetime.strptime(date_text, "%Y-%m-%d")
            except ValueError as exc:
                raise ValueError(
                    f"Invalid date for tip {tip_name!r} on row {line_number}: {date_text!r}; expected YYYY-MM-DD"
                ) from exc
            if tip_name in date_map and date_map[tip_name] != parsed:
                raise ValueError(f"Conflicting dates for tip {tip_name!r} in {date_file}")
            date_map[tip_name] = parsed

    if not date_map:
        raise ValueError(f"No tip dates found in {date_file}")
    return date_map

def tree_tip_names(tree_file: str | Path) -> list[str]:
    Tree = _tree_class()
    tree = Tree(str(tree_file), format=1)
    return [leaf.name for leaf in tree.iter_leaves()]

def write_lsd_date_file(
    date_map: dict[str, datetime],
    tip_names: list[str],
    out_file: str | Path,
    tree_id: str,
) -> datetime:
    """Write a per-tree LSD2 date file and return that tree's latest tip date."""
    missing = [name for name in tip_names if name not in date_map]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise ValueError(f"Date file is missing {len(missing)} tips for {tree_id}: {preview}{suffix}")

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    latest_date = max(date_map[name] for name in tip_names)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(f"{len(tip_names)}\n")
        for name in tip_names:
            handle.write(f"{name}\t{date_map[name].strftime('%Y-%m-%d')}\n")
    return latest_date

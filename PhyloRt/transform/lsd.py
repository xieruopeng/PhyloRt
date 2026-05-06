"""External LSD2/gotree execution helpers."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Callable

LEGACY_TIMESCALE_MANIFEST = "phylort_timescale_manifest.json"
DEFAULT_LSD_LOWER_RATE = 0.0001141552511
DEFAULT_LSD_UPPER_RATE = 0.0001141552511
NEWICK_ANNOTATION_RE = re.compile(r"\[&[^\]]*\]")

def _run_external_command(
    command: list[str],
    cwd: Path,
    log_path: Path,
    run_command_func: Callable | None = None,
) -> None:
    runner = run_command_func or subprocess.run
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")

    try:
        completed = runner(
            command,
            cwd=str(cwd),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Could not find external command {command[0]!r}. Install it or pass the correct binary path."
        ) from exc
    except subprocess.CalledProcessError as exc:
        output = exc.stdout or ""
        with open(log_path, "a", encoding="utf-8") as handle:
            if output:
                handle.write(output)
            handle.write("\n")
        raise RuntimeError(f"Command failed: {' '.join(command)}. See {log_path}.") from exc

    output = getattr(completed, "stdout", "") if completed is not None else ""
    with open(log_path, "a", encoding="utf-8") as handle:
        if output:
            handle.write(output)
        handle.write("\n")

def strip_newick_annotations(tree_file: str | Path) -> bool:
    """Remove BEAST/Nexus annotations that gotree may leave in Newick labels."""
    path = Path(tree_file)
    text = path.read_text(encoding="utf-8")
    cleaned = NEWICK_ANNOTATION_RE.sub("", text)
    if cleaned == text:
        return False
    path.write_text(cleaned, encoding="utf-8")
    return True

def cleanup_dating_intermediates(out_dir: str | Path) -> None:
    """Remove temporary LSD2/gotree inputs while keeping the final dated tree."""
    out_path = Path(out_dir)
    for path in out_path.glob("resolved_tree.nwk*"):
        path.unlink(missing_ok=True)

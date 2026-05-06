"""Polytomy cleanup and stochastic binary resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .newick import _tree_class


@dataclass
class ResolveStats:
    collapsed_branches: int
    resolved_polytomies: int
    tips: int

def collapse_zero_branches(tree, threshold: float) -> int:
    """Collapse internal branches shorter than ``threshold``."""
    n_collapsed = 0
    for node in list(tree.traverse("postorder")):
        for child in list(node.children):
            if child.dist < threshold and not child.is_leaf():
                node.remove_child(child)
                for grandchild in list(child.children):
                    node.add_child(grandchild, dist=grandchild.dist + child.dist)
                n_collapsed += 1
    return n_collapsed

def _collapse_unary_nodes(tree):
    for node in list(tree.traverse("postorder")):
        if node.is_leaf() or len(node.children) != 1:
            continue
        child = node.children[0]
        parent = node.up
        if parent:
            parent.remove_child(node)
            parent.add_child(child, dist=child.dist + node.dist)
        else:
            child.detach()
            tree = child
    return tree

def _resolve_remaining_polytomies(tree, rng: np.random.Generator) -> int:
    resolved = 0
    for node in tree.traverse("postorder"):
        if node.is_leaf() or len(node.children) <= 2:
            continue
        resolved += 1
        while len(node.children) > 2:
            children = list(node.children)
            first_idx, second_idx = rng.choice(len(children), 2, replace=False)
            child1 = children[int(first_idx)]
            child2 = children[int(second_idx)]
            node.remove_child(child1)
            node.remove_child(child2)
            parent = node.add_child(dist=0)
            parent.add_child(child1, dist=child1.dist)
            parent.add_child(child2, dist=child2.dist)
    return resolved

def resolve_polytomies(
    in_nwk: str | Path,
    out_nwk: str | Path,
    sequence_length: int,
    random_seed: int | None = None,
) -> ResolveStats:
    """Resolve a genetic-distance tree without modifying the input file."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be a positive integer")

    Tree = _tree_class()
    tree = Tree(str(in_nwk), format=1)
    one_mutation = 1 / sequence_length
    collapsed = collapse_zero_branches(tree, one_mutation / 2)

    tree = _collapse_unary_nodes(tree)
    rng = np.random.default_rng(random_seed)
    resolved = _resolve_remaining_polytomies(tree, rng)

    out_path = Path(out_nwk)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(outfile=str(out_path), format=5)
    return ResolveStats(
        collapsed_branches=collapsed,
        resolved_polytomies=resolved,
        tips=len(tree),
    )

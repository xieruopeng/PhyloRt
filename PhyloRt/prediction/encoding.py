"""Tree encoding for PhyloRt neural-network inputs."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.setrecursionlimit(100000)

TARGET_AVG_BL = 1

def _tree_class():
    try:
        from ete3 import Tree
    except ImportError as exc:
        raise ImportError(
            "PhyloRt prediction requires ete3. Install the prediction "
            "dependencies before running PhyloRt predict."
        ) from exc
    return Tree

def add_dist_to_root(tree) -> None:
    for node in tree.traverse("levelorder"):
        node.dist_to_root = 0 if node.is_root() else node.up.dist_to_root + node.dist

def add_number_of_leaves(tree) -> None:
    for node in tree.traverse("postorder"):
        node.leaves = 1 if node.is_leaf() else sum(child.leaves for child in node.children)

def name_tree_nodes(tree) -> None:
    for idx, node in enumerate(tree.traverse("preorder")):
        node.name = str(idx)

def add_dist_to_present(tree) -> None:
    max_depth = max(leaf.dist_to_root for leaf in tree.iter_leaves())
    for node in tree.traverse():
        node.dist_to_present = max_depth - node.dist_to_root

def rescale_tree(tree, target_avg_length: float = TARGET_AVG_BL) -> float:
    dist_all = [node.dist for node in tree.traverse("levelorder")]
    all_bl_mean = float(np.mean(dist_all))
    if all_bl_mean == 0:
        return 1.0
    rescale_factor = all_bl_mean / target_avg_length
    for node in tree.traverse():
        node.dist = node.dist / rescale_factor
    return rescale_factor

def add_distances_to_ancestors(tree) -> None:
    for node in tree.traverse("levelorder"):
        node.dist_to_anc = -1
        node.dist_to_grand_anc = -1
        node.dist_to_great_grand_anc = -1

        if node.up:
            node.dist_to_anc = node.dist
            if node.up.up:
                node.dist_to_grand_anc = node.dist_to_root - node.up.up.dist_to_root
                if node.up.up.up:
                    node.dist_to_great_grand_anc = (
                        node.dist_to_root - node.up.up.up.dist_to_root
                    )

def _binary_children(node):
    if len(node.children) != 2:
        raise ValueError(
            "PhyloRt tree encoding expects binary trees. Resolve polytomies "
            "before prediction."
        )
    return node.children

def add_distances_to_children(tree) -> None:
    for node in tree.traverse("postorder"):
        if node.is_leaf():
            node.dist_to_minor_child = -1
            node.dist_to_major_child = -1
            node.minor_leaves = -1
            node.major_leaves = -1
            continue

        child_1, child_2 = _binary_children(node)
        if child_1.leaves >= child_2.leaves:
            minor_child, major_child = child_2, child_1
        else:
            minor_child, major_child = child_1, child_2

        node.dist_to_minor_child = minor_child.dist
        node.dist_to_major_child = major_child.dist
        node.minor_leaves = minor_child.leaves
        node.major_leaves = major_child.leaves

def add_distances_to_grandchildren(tree) -> None:
    attrs = [
        "dist_to_minor_minor_grandchild",
        "dist_to_minor_major_grandchild",
        "dist_to_major_minor_grandchild",
        "dist_to_major_major_grandchild",
        "minor_minor_leaves",
        "minor_major_leaves",
        "major_minor_leaves",
        "major_major_leaves",
    ]

    for node in tree.traverse("postorder"):
        if node.is_leaf():
            for attr in attrs:
                setattr(node, attr, -1)
            continue

        child_1, child_2 = _binary_children(node)
        if child_1.leaves >= child_2.leaves:
            minor_child, major_child = child_2, child_1
        else:
            minor_child, major_child = child_1, child_2

        if not minor_child.is_leaf():
            gc_1, gc_2 = _binary_children(minor_child)
            if gc_1.leaves >= gc_2.leaves:
                minor_minor_gc, minor_major_gc = gc_2, gc_1
            else:
                minor_minor_gc, minor_major_gc = gc_1, gc_2
            node.dist_to_minor_minor_grandchild = (
                minor_minor_gc.dist_to_root - node.dist_to_root
            )
            node.dist_to_minor_major_grandchild = (
                minor_major_gc.dist_to_root - node.dist_to_root
            )
            node.minor_minor_leaves = minor_minor_gc.leaves
            node.minor_major_leaves = minor_major_gc.leaves
        else:
            node.dist_to_minor_minor_grandchild = -1
            node.dist_to_minor_major_grandchild = -1
            node.minor_minor_leaves = -1
            node.minor_major_leaves = -1

        if not major_child.is_leaf():
            gc_1, gc_2 = _binary_children(major_child)
            if gc_1.leaves >= gc_2.leaves:
                major_minor_gc, major_major_gc = gc_2, gc_1
            else:
                major_minor_gc, major_major_gc = gc_1, gc_2
            node.dist_to_major_minor_grandchild = (
                major_minor_gc.dist_to_root - node.dist_to_root
            )
            node.dist_to_major_major_grandchild = (
                major_major_gc.dist_to_root - node.dist_to_root
            )
            node.major_minor_leaves = major_minor_gc.leaves
            node.major_major_leaves = major_major_gc.leaves
        else:
            node.dist_to_major_minor_grandchild = -1
            node.dist_to_major_major_grandchild = -1
            node.major_minor_leaves = -1
            node.major_major_leaves = -1

def process_tree(tree_or_path: str | Path) -> pd.DataFrame:
    """Encode one Newick tree as the fixed-size PhyloRt model input table."""
    Tree = _tree_class()
    tree = Tree(str(tree_or_path), format=1)

    if len(tree.children) == 1:
        tree = tree.children[0]
        tree.up = None

    for node in tree.traverse("levelorder"):
        node.visited = 0

    name_tree_nodes(tree)
    rescale_factor = rescale_tree(tree, TARGET_AVG_BL)
    add_number_of_leaves(tree)
    add_dist_to_root(tree)
    add_dist_to_present(tree)
    add_distances_to_ancestors(tree)
    add_distances_to_children(tree)
    add_distances_to_grandchildren(tree)

    tree_embedding = []
    for node in tree.traverse("levelorder"):
        tree_embedding.append(
            [
                node.dist_to_present,
                node.dist_to_root,
                node.dist,
                node.leaves,
                node.dist_to_grand_anc,
                node.dist_to_minor_child,
                node.dist_to_major_child,
                node.minor_leaves,
                node.major_leaves,
                node.dist_to_great_grand_anc,
                node.dist_to_minor_minor_grandchild,
                node.dist_to_minor_major_grandchild,
                node.dist_to_major_minor_grandchild,
                node.dist_to_major_major_grandchild,
                node.minor_minor_leaves,
                node.minor_major_leaves,
                node.major_minor_leaves,
                node.major_major_leaves,
            ]
        )

    df = pd.DataFrame(tree_embedding)
    df = df.reindex(range(999), fill_value=0)
    df = df.reindex(range(1000), fill_value=rescale_factor)
    return df

def get_para_encoding_test(
    sampling_proportion: float,
    mean_sample_to_full_tree: float,
    last_sample_to_full_tree: float,
    encoding_csv: np.ndarray,
) -> np.ndarray:
    encoding_test = np.delete(encoding_csv, -1, axis=1)
    encoding_test = np.concatenate(
        (encoding_test, np.repeat(np.array(sampling_proportion), 999).reshape(-1, 999, 1)),
        axis=2,
    )
    encoding_test = np.concatenate(
        (encoding_test, np.repeat(np.array(last_sample_to_full_tree), 999).reshape(-1, 999, 1)),
        axis=2,
    )
    encoding_test = np.concatenate(
        (encoding_test, np.repeat(np.array(mean_sample_to_full_tree), 999).reshape(-1, 999, 1)),
        axis=2,
    )

    enc_pad_test = []
    for idx in range(encoding_test.shape[0]):
        leaves = encoding_test[idx][encoding_test[idx, :, 3] == 1]
        leaves = leaves[np.argsort(leaves[:, 1])]
        leaves = leaves[:500]
        leaves = np.pad(leaves, [(0, max(0, 500 - leaves.shape[0])), (0, 0)], mode="constant")

        nodes = encoding_test[idx][encoding_test[idx, :, 3] > 1]
        nodes = nodes[np.argsort(nodes[:, 1])]
        if nodes.shape[0] > 0:
            nodes = np.append(nodes, nodes[-1].reshape(1, -1), axis=0)
        nodes = nodes[:500]
        nodes = np.pad(nodes, [(0, max(0, 500 - nodes.shape[0])), (0, 0)], mode="constant")
        enc_pad_test.append(np.stack((leaves, nodes), axis=2))

    return np.array(enc_pad_test)

"""Prediction workflow and helpers."""

from .encoding import (
    TARGET_AVG_BL,
    add_dist_to_present,
    add_dist_to_root,
    add_distances_to_ancestors,
    add_distances_to_children,
    add_distances_to_grandchildren,
    add_number_of_leaves,
    get_para_encoding_test,
    name_tree_nodes,
    process_tree,
    rescale_tree,
)
from .extraction import ExtractionContext, extract_subtrees
from .metadata import _format_metadata_for_output
from .models import MODEL_SIZES, MODEL_SUFFIXES, PATHOGEN_MODELS, ModelBundle, load_model_bundle, model_path, resolve_model_prefix
from .predict import predict
from .sampling import datetime_to_decimal_year, get_prop_for_date, parse_latest_date
from .state import (
    MANIFEST_FILENAME,
    build_extraction_config,
    build_replicate_run_config,
    clear_generated_outputs,
    ensure_resume_manifest,
    file_sha256,
    load_existing_results,
    load_manifest,
    load_state,
    remove_replicate_work_files,
    remove_tree_files,
    replicate_id,
    replicate_seed,
    save_manifest,
    save_state,
    setup_logging,
)

__all__ = [name for name in globals() if not name.startswith("_")]

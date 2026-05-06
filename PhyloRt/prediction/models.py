"""Model path resolution, loading, and inference."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .encoding import get_para_encoding_test, process_tree

MODEL_SIZES = (50, 100, 200, 400)
MODEL_SUFFIXES = {
    "classification": "classification",
    "r1": "regression_R1",
    "r2_1": "regression_R2_1",
    "r2_2": "regression_R2_2",
    "breakpoint": "regression_breakpoint",
}
PATHOGEN_MODELS = {
    "H3N2": "BDEI_H3N2",
    "H1N1": "BDEI_H3N2",
    "FLUB": "BDEI_flub",
    "RSVA": "BDEI_H3N2",
    "RSVB": "BDEI_H3N2",
    "COVID": "BDEISS_COVID",
}


def _default_model_dir() -> Path:
    package_root = Path(__file__).resolve().parents[1]
    preferred = package_root / "pretrained_models"
    legacy = Path(__file__).resolve().parent / "pretrained_models"
    if preferred.exists():
        return preferred
    return legacy

def resolve_model_prefix(pre_model: str) -> str:
    key = pre_model.upper()
    if key not in PATHOGEN_MODELS:
        raise ValueError("pre_model must be one of COVID, H3N2, H1N1, FLUB, RSVA, or RSVB")
    return PATHOGEN_MODELS[key]

def model_path(
    pre_model: str,
    subtree_size: int,
    suffix: str,
    model_dir: str | Path | None = None,
) -> Path:
    if subtree_size not in MODEL_SIZES:
        raise ValueError(f"No bundled model for subtree size {subtree_size}")
    if suffix not in MODEL_SUFFIXES:
        raise ValueError(f"Unknown model suffix {suffix!r}")
    model_root = Path(model_dir) if model_dir is not None else _default_model_dir()
    prefix = resolve_model_prefix(pre_model)
    filename = f"{prefix}_Rt_{subtree_size}_tips_{MODEL_SUFFIXES[suffix]}.h5"
    return model_root / filename


@dataclass
class ModelBundle:
    models: dict[int, dict[str, object]]

    def for_subtree_size(self, subtree_size: int) -> Optional[dict[str, object]]:
        for size in MODEL_SIZES:
            if subtree_size <= size:
                return self.models[size]
        return None

def _load_model_func():
    try:
        from tensorflow.keras.models import load_model
    except ImportError as exc:
        raise ImportError(
            "PhyloRt prediction requires TensorFlow. Install with "
            "`pip install -e .` or provide load_model_func for testing."
        ) from exc
    return load_model


def _validate_pretrained_models(pre_model: str, model_dir: str | Path | None = None) -> None:
    model_root = Path(model_dir) if model_dir is not None else _default_model_dir()
    if not model_root.exists():
        raise FileNotFoundError(
            "Pretrained model directory not found: "
            f"{model_root}. Place pretrained .h5 files under this directory."
        )

    missing: list[Path] = []
    for subtree_size in MODEL_SIZES:
        for key in MODEL_SUFFIXES:
            path = model_path(pre_model, subtree_size, key, model_dir=model_root)
            if not path.exists():
                missing.append(path)

    if missing:
        preview = "\n".join(f"- {path}" for path in missing[:5])
        suffix = "\n- ..." if len(missing) > 5 else ""
        raise FileNotFoundError(
            "Missing pretrained model file(s). "
            f"Expected {len(missing)} .h5 file(s) for pre_model={pre_model!r} under {model_root}.\n"
            f"{preview}{suffix}"
        )


def load_model_bundle(
    pre_model: str,
    model_dir: str | Path | None = None,
    load_model_func: Optional[Callable[[str], object]] = None,
) -> ModelBundle:
    if load_model_func is None:
        _validate_pretrained_models(pre_model, model_dir=model_dir)
    loader = load_model_func or _load_model_func()
    models: dict[int, dict[str, object]] = {}
    for subtree_size in MODEL_SIZES:
        models[subtree_size] = {
            key: loader(str(model_path(pre_model, subtree_size, key, model_dir)))
            for key in MODEL_SUFFIXES
        }
    return ModelBundle(models=models)

def _predict_array(model, array: np.ndarray) -> np.ndarray:
    try:
        return np.asarray(model.predict(array, verbose=0))
    except TypeError:
        return np.asarray(model.predict(array))

def _predict_row(row: pd.Series, encoding_test: np.ndarray, bundle: ModelBundle) -> dict[str, float]:
    size = int(row["subtree_size"])
    if size <= 20:
        return {"change": np.nan, "R1": np.nan, "R2": np.nan, "breakpoint": np.nan}

    models = bundle.for_subtree_size(size)
    if models is None:
        return {"change": np.nan, "R1": np.nan, "R2": np.nan, "breakpoint": np.nan}

    prediction_c = _predict_array(models["classification"], encoding_test)
    change = float(prediction_c[0, 0])
    if change > 0.5:
        prediction_r2_1 = _predict_array(models["r2_1"], encoding_test)
        prediction_r2_2 = _predict_array(models["r2_2"], encoding_test)
        prediction_b = _predict_array(models["breakpoint"], encoding_test)
        return {
            "change": change,
            "R1": float(prediction_r2_1[0, 0]),
            "R2": float(prediction_r2_2[0, 0]),
            "breakpoint": float(prediction_b[0, 0]),
        }

    prediction_r1 = _predict_array(models["r1"], encoding_test)
    return {
        "change": change,
        "R1": float(prediction_r1[0, 0]),
        "R2": 0.0,
        "breakpoint": 0.0,
    }

def _bucket_for_subtree_size(subtree_size: int) -> Optional[int]:
    for size in MODEL_SIZES:
        if subtree_size <= size:
            return size
    return None


def _normalize_prediction_rows(values: np.ndarray, n_rows: int) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 0:
        return np.repeat(float(values), n_rows)
    if values.ndim == 1:
        if values.shape[0] == n_rows:
            return values.astype(float)
        if values.shape[0] == 1:
            return np.repeat(float(values[0]), n_rows)
    if values.ndim >= 2:
        first_col = values[:, 0].astype(float)
        if first_col.shape[0] == n_rows:
            return first_col
        if first_col.shape[0] == 1:
            return np.repeat(float(first_col[0]), n_rows)
    raise ValueError(f"Model output shape {values.shape} does not match batch size {n_rows}")


def _encode_row(record: dict[str, object]) -> tuple[int, np.ndarray]:
    idx = int(record["idx"])
    subtree_path = Path(str(record["subtree_nwk_file"]))
    if not subtree_path.exists():
        raise FileNotFoundError(
            "Subtree file not found during prediction: "
            f"{subtree_path}. This usually happens when resuming from old batch/metadata files "
            "after temporary tree files were cleaned. Re-run with --no-resume, or keep subtree files with "
            "--keep-tree-files."
        )
    try:
        encoding_csv = process_tree(subtree_path)
    except Exception as exc:
        raise ValueError(
            "Failed to parse subtree Newick during prediction: "
            f"{subtree_path}. The file may be malformed or stale from an older run. "
            "Re-run with --no-resume to regenerate subtree files."
        ) from exc
    encoding_array = np.expand_dims(encoding_csv.values.astype("float32"), axis=0)
    encoding_test = get_para_encoding_test(
        float(record["sampling_proportion"]),
        float(record["mean_sample_to_full_tree"]),
        float(record["last_sample_to_full_tree"]),
        encoding_array,
    )
    return idx, encoding_test


def _predict_metadata_rows(
    meta: pd.DataFrame,
    bundle: ModelBundle,
    n_jobs: int = 1,
) -> pd.DataFrame:
    if meta.empty:
        return meta
    meta = meta.copy()
    workers = max(1, int(n_jobs))
    records = []
    for idx, row in meta.iterrows():
        size = int(row["subtree_size"])
        if size <= 20 or _bucket_for_subtree_size(size) is None:
            meta.at[idx, "change"] = np.nan
            meta.at[idx, "R1"] = np.nan
            meta.at[idx, "R2"] = np.nan
            meta.at[idx, "breakpoint"] = np.nan
            continue
        records.append(
            {
                "idx": idx,
                "subtree_nwk_file": row["subtree_nwk_file"],
                "sampling_proportion": row["sampling_proportion"],
                "mean_sample_to_full_tree": row["mean_sample_to_full_tree"],
                "last_sample_to_full_tree": row["last_sample_to_full_tree"],
            }
        )

    encodings: dict[int, np.ndarray] = {}
    if workers == 1:
        for record in tqdm(records, total=len(records), desc="Encoding trees"):
            idx, encoding_test = _encode_row(record)
            encodings[idx] = encoding_test
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_encode_row, record) for record in records]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Encoding trees"):
                idx, encoding_test = future.result()
                encodings[idx] = encoding_test

    for model_size in MODEL_SIZES:
        bucket_idxs = [
            idx
            for idx, row in meta.iterrows()
            if idx in encodings and _bucket_for_subtree_size(int(row["subtree_size"])) == model_size
        ]
        if not bucket_idxs:
            continue
        batch = np.concatenate([encodings[idx] for idx in bucket_idxs], axis=0)
        models = bundle.models[model_size]

        change_scores = _normalize_prediction_rows(_predict_array(models["classification"], batch), len(bucket_idxs))
        r1_values = np.zeros(len(bucket_idxs), dtype=float)
        r2_values = np.zeros(len(bucket_idxs), dtype=float)
        breakpoint_values = np.zeros(len(bucket_idxs), dtype=float)

        change_mask = change_scores > 0.5
        if np.any(change_mask):
            change_batch = batch[change_mask]
            r2_1 = _normalize_prediction_rows(_predict_array(models["r2_1"], change_batch), int(change_mask.sum()))
            r2_2 = _normalize_prediction_rows(_predict_array(models["r2_2"], change_batch), int(change_mask.sum()))
            brk = _normalize_prediction_rows(
                _predict_array(models["breakpoint"], change_batch),
                int(change_mask.sum()),
            )
            r1_values[change_mask] = r2_1
            r2_values[change_mask] = r2_2
            breakpoint_values[change_mask] = brk

        if np.any(~change_mask):
            static_batch = batch[~change_mask]
            r1 = _normalize_prediction_rows(_predict_array(models["r1"], static_batch), int((~change_mask).sum()))
            r1_values[~change_mask] = r1

        for pos, idx in enumerate(bucket_idxs):
            meta.at[idx, "change"] = float(change_scores[pos])
            meta.at[idx, "R1"] = float(r1_values[pos])
            meta.at[idx, "R2"] = float(r2_values[pos])
            meta.at[idx, "breakpoint"] = float(breakpoint_values[pos])
    return meta

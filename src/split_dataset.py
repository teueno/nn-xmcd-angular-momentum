#!/usr/bin/env python3
"""Create reproducible train/validation/test assignments with B_ext auditing.

The prepared NPZ must contain X, Y, sample_id, prefix, b_ext_t,
external_field_enabled, external_field_magnitude_t, and
external_field_direction.  The accepted metadata CSV must contain
sample_id, prefix, sz_label, lz_label, and b_ext_t.
"""

import argparse
import csv
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .utils import (
        check_output_files,
        ensure_directory,
        ensure_finite,
        get_repository_root,
        load_json,
        public_path,
        resolve_path,
        sha256_file,
        software_versions,
        utc_now_iso,
        write_json,
    )
except ImportError:
    from utils import (
        check_output_files,
        ensure_directory,
        ensure_finite,
        get_repository_root,
        load_json,
        public_path,
        resolve_path,
        sha256_file,
        software_versions,
        utc_now_iso,
        write_json,
    )

SUBSET_NAMES = ("train", "validation", "test")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Create reproducible external-field-aware dataset splits."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_public_path(path, repository_root):
    if path is None:
        return None
    try:
        return public_path(path, repository_root)
    except ValueError:
        return Path(path).name


def validate_ratio(name, value, allow_zero):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric")
    ratio = float(value)
    if not np.isfinite(ratio):
        raise ValueError(f"{name} must be finite")
    if allow_zero and ratio < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    if not allow_zero and ratio <= 0.0:
        raise ValueError(f"{name} must be positive")
    if ratio > 1.0:
        raise ValueError(f"{name} must not exceed 1.0")
    return ratio


def validate_seed(value):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("split.random_seed must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError("split.random_seed must be nonnegative")
    return value


def validate_nonnegative_float(name, value):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def validate_config(config):
    required_sections = {
        "system", "paths", "split", "columns", "external_magnetic_field"
    }
    missing = required_sections - set(config)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    if "name" not in config["system"]:
        raise ValueError("Config must contain system.name")
    system_name = str(config["system"]["name"]).strip()
    if not system_name:
        raise ValueError("system.name must not be empty")

    required_paths = {
        "repository_root", "prepared_dataset_npz", "accepted_metadata_csv",
        "split_output_directory",
    }
    missing = required_paths - set(config["paths"])
    if missing:
        raise ValueError(f"Missing path settings: {sorted(missing)}")

    required_columns = {"prefix", "sz", "lz", "b_ext_t"}
    missing = required_columns - set(config["columns"])
    if missing:
        raise ValueError(f"Missing column settings: {sorted(missing)}")

    split = config["split"]
    train = validate_ratio("split.train_ratio", split.get("train_ratio", 0.70), False)
    validation = validate_ratio(
        "split.validation_ratio", split.get("validation_ratio", 0.15), True
    )
    test_value = split.get("test_ratio", 1.0 - train - validation)
    test = validate_ratio("split.test_ratio", test_value, False)
    if not np.isclose(train + validation + test, 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("train + validation + test ratios must equal 1")

    field = config["external_magnetic_field"]
    for key in ("enabled", "magnitude_t", "direction"):
        if key not in field:
            raise ValueError(f"Missing external_magnetic_field.{key}")
    magnitude = validate_nonnegative_float(
        "external_magnetic_field.magnitude_t", field["magnitude_t"]
    )
    tolerance = validate_nonnegative_float(
        "external_magnetic_field.tolerance_t", field.get("tolerance_t", 1e-9)
    )
    direction = np.asarray(field["direction"], dtype=np.float64)
    if direction.shape != (3,):
        raise ValueError("external_magnetic_field.direction must have three values")
    ensure_finite("External magnetic-field direction", direction)
    norm = float(np.linalg.norm(direction))
    if norm <= 0:
        raise ValueError("external magnetic-field direction must be nonzero")
    direction /= norm

    return {
        "system_name": system_name,
        "train_ratio": train,
        "validation_ratio": validation,
        "test_ratio": test,
        "random_seed": validate_seed(split.get("random_seed", 1234)),
        "field_enabled": bool(field["enabled"]),
        "field_magnitude_t": magnitude,
        "field_tolerance_t": tolerance,
        "field_direction": direction,
        "metadata_prefix_column": str(config["columns"]["prefix"]),
        "metadata_sz_column": str(config["columns"]["sz"]),
        "metadata_lz_column": str(config["columns"]["lz"]),
        "metadata_field_column": str(config["columns"]["b_ext_t"]),
    }


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("Summary input must be one-dimensional")
    if values.size == 0:
        return {"count": 0, "min": None, "max": None, "mean": None,
                "std_population": None}
    ensure_finite("Summary values", values)
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std_population": float(np.std(values, ddof=0)),
    }


def validate_dataset(X, Y, sample_ids, prefixes, b_ext_t, expected_field, tolerance):
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError(f"Invalid X shape: {X.shape}")
    n = int(X.shape[0])
    if Y.shape != (n, 2):
        raise ValueError(f"Expected Y shape {(n, 2)}, found {Y.shape}")
    for name, array in (("sample_id", sample_ids), ("prefix", prefixes),
                        ("b_ext_t", b_ext_t)):
        if array.shape != (n,):
            raise ValueError(f"{name} shape differs from sample count")
    ensure_finite("Prepared X", X)
    ensure_finite("Prepared Y", Y)
    ensure_finite("Prepared B_ext_T", b_ext_t)
    if len(set(sample_ids.tolist())) != n:
        raise ValueError("Duplicate sample_id values in NPZ")
    if len(set(prefixes.tolist())) != n:
        raise ValueError("Duplicate Prefix values in NPZ")
    if not np.allclose(b_ext_t, expected_field, rtol=0.0, atol=tolerance):
        raise ValueError("NPZ contains external-field values inconsistent with config")
    return n


def align_metadata(metadata, sample_ids, prefixes, Y, b_ext_t, normalized):
    prefix_col = normalized["metadata_prefix_column"]
    sz_col = normalized["metadata_sz_column"]
    lz_col = normalized["metadata_lz_column"]
    field_col = normalized["metadata_field_column"]

    # prepare_dataset adds normalized aliases; accept either aliases or source names.
    required = {"sample_id", "prefix", "sz_label", "lz_label", "b_ext_t"}
    if not required.issubset(metadata.columns):
        source_required = {"sample_id", prefix_col, sz_col, lz_col, field_col}
        missing = source_required - set(metadata.columns)
        if missing:
            raise ValueError(f"Accepted metadata is missing columns: {sorted(missing)}")
        metadata = metadata.copy()
        metadata["prefix"] = metadata[prefix_col]
        metadata["sz_label"] = metadata[sz_col]
        metadata["lz_label"] = metadata[lz_col]
        metadata["b_ext_t"] = metadata[field_col]

    if metadata["sample_id"].duplicated().any():
        raise ValueError("Accepted metadata contains duplicate sample_id values")
    by_id = metadata.assign(sample_id=metadata["sample_id"].astype(str)).set_index(
        "sample_id", drop=False
    )
    if set(by_id.index) != set(sample_ids.tolist()):
        raise ValueError("Prepared dataset and metadata have different sample IDs")
    ordered = by_id.loc[sample_ids].reset_index(drop=True).copy()

    metadata_prefixes = ordered["prefix"].astype(str).to_numpy()
    metadata_sz = pd.to_numeric(ordered["sz_label"], errors="coerce").to_numpy(float)
    metadata_lz = pd.to_numeric(ordered["lz_label"], errors="coerce").to_numpy(float)
    metadata_field = pd.to_numeric(ordered["b_ext_t"], errors="coerce").to_numpy(float)
    for name, values in (("Sz", metadata_sz), ("Lz", metadata_lz),
                         ("B_ext_T", metadata_field)):
        ensure_finite(f"Metadata {name}", values)
    if not np.array_equal(metadata_prefixes, prefixes):
        raise ValueError("Prefix order differs between NPZ and metadata")
    if not np.allclose(metadata_sz, Y[:, 0], rtol=1e-12, atol=1e-12):
        raise ValueError("Sz labels differ between NPZ and metadata")
    if not np.allclose(metadata_lz, Y[:, 1], rtol=1e-12, atol=1e-12):
        raise ValueError("Lz labels differ between NPZ and metadata")
    if not np.allclose(
        metadata_field, b_ext_t, rtol=0.0, atol=normalized["field_tolerance_t"]
    ):
        raise ValueError("B_ext_T differs between NPZ and metadata")
    return ordered


def calculate_counts(n, train_ratio, validation_ratio):
    n_train = int(train_ratio * n)
    n_validation = int(validation_ratio * n)
    n_test = n - n_train - n_validation
    if n_train <= 0 or n_test <= 0 or (validation_ratio > 0 and n_validation <= 0):
        raise ValueError("One or more split subsets are empty")
    return n_train, n_validation, n_test


def validate_indices(train, validation, test, n):
    arrays = [np.asarray(values, dtype=np.int64) for values in (train, validation, test)]
    for name, values in zip(SUBSET_NAMES, arrays):
        if values.ndim != 1 or np.any(values < 0) or np.any(values >= n):
            raise RuntimeError(f"Invalid {name} indices")
        if len(np.unique(values)) != len(values):
            raise RuntimeError(f"Duplicate {name} indices")
    all_indices = np.concatenate(arrays)
    if len(all_indices) != n or not np.array_equal(np.sort(np.unique(all_indices)), np.arange(n)):
        raise RuntimeError("Split indices do not exactly cover the dataset")


def subset_summaries(frame):
    output = {}
    for subset in SUBSET_NAMES:
        part = frame.loc[frame["subset"] == subset]
        output[subset] = {
            "count": int(len(part)),
            "targets": {
                "sz_label": summarize(part["sz_label"].to_numpy(float)),
                "lz_label": summarize(part["lz_label"].to_numpy(float)),
            },
            "external_field_t": summarize(part["b_ext_t"].to_numpy(float)),
        }
    return output


def main():
    args = parse_arguments()
    config_path = args.config.expanduser().resolve()
    config = load_json(config_path)
    normalized = validate_config(config)
    config_dir = config_path.parent
    repository_root = get_repository_root(config, config_path)
    paths = config["paths"]
    system_name = normalized["system_name"]

    dataset_file = resolve_path(paths["prepared_dataset_npz"], config_dir)
    metadata_file = resolve_path(paths["accepted_metadata_csv"], config_dir)
    output_dir = ensure_directory(resolve_path(paths["split_output_directory"], config_dir))
    split_csv = output_dir / f"{system_name}_split.csv"
    indices_npz = output_dir / f"{system_name}_split_indices.npz"
    summary_json = output_dir / f"{system_name}_split_summary.json"
    check_output_files(
        [split_csv, indices_npz, summary_json], overwrite=args.overwrite,
        label="Split output files",
    )
    if not dataset_file.is_file():
        raise FileNotFoundError(f"Prepared dataset not found: {safe_public_path(dataset_file, repository_root)}")
    if not metadata_file.is_file():
        raise FileNotFoundError(f"Accepted metadata not found: {safe_public_path(metadata_file, repository_root)}")

    with np.load(dataset_file, allow_pickle=False) as data:
        required = {
            "X", "Y", "sample_id", "prefix", "b_ext_t",
            "external_field_enabled", "external_field_magnitude_t",
            "external_field_direction",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Prepared dataset is missing arrays: {sorted(missing)}")
        X = np.asarray(data["X"], dtype=np.float64)
        Y = np.asarray(data["Y"], dtype=np.float64)
        sample_ids = data["sample_id"].astype(str)
        prefixes = data["prefix"].astype(str)
        b_ext_t = np.asarray(data["b_ext_t"], dtype=np.float64)
        npz_field_enabled = bool(np.asarray(data["external_field_enabled"]).reshape(-1)[0])
        npz_field_magnitude = float(np.asarray(data["external_field_magnitude_t"]).reshape(-1)[0])
        npz_field_direction = np.asarray(data["external_field_direction"], dtype=np.float64)

    if npz_field_enabled != normalized["field_enabled"]:
        raise ValueError("NPZ external_field_enabled differs from config")
    if not np.isclose(
        npz_field_magnitude, normalized["field_magnitude_t"], rtol=0.0,
        atol=normalized["field_tolerance_t"],
    ):
        raise ValueError("NPZ external-field magnitude differs from config")
    if not np.allclose(
        npz_field_direction, normalized["field_direction"], rtol=0.0, atol=1e-12
    ):
        raise ValueError("NPZ external-field direction differs from config")

    n = validate_dataset(
        X, Y, sample_ids, prefixes, b_ext_t,
        normalized["field_magnitude_t"], normalized["field_tolerance_t"],
    )
    metadata = pd.read_csv(metadata_file)
    if len(metadata) != n:
        raise ValueError(f"Metadata count differs from NPZ count: {len(metadata)} != {n}")
    metadata = align_metadata(metadata, sample_ids, prefixes, Y, b_ext_t, normalized)

    rng = np.random.default_rng(normalized["random_seed"])
    shuffled = rng.permutation(n).astype(np.int64, copy=False)
    n_train, n_validation, n_test = calculate_counts(
        n, normalized["train_ratio"], normalized["validation_ratio"]
    )
    idx_train = shuffled[:n_train]
    idx_validation = shuffled[n_train:n_train + n_validation]
    idx_test = shuffled[n_train + n_validation:]
    validate_indices(idx_train, idx_validation, idx_test, n)

    subset = np.full(n, "", dtype="U10")
    subset[idx_train] = "train"
    subset[idx_validation] = "validation"
    subset[idx_test] = "test"

    frame = metadata.copy()
    frame.insert(0, "dataset_index", np.arange(n, dtype=np.int64))
    frame.insert(1, "subset", subset)
    frame.insert(2, "split_seed", normalized["random_seed"])
    preferred = [
        "dataset_index", "sample_id", "subset", "split_seed", "prefix",
        "sz_label", "lz_label", "b_ext_t",
    ]
    frame = frame[preferred + [column for column in frame.columns if column not in preferred]]
    frame.to_csv(split_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    np.savez_compressed(
        indices_npz,
        idx_train=idx_train,
        idx_validation=idx_validation,
        idx_test=idx_test,
        sample_id_train=sample_ids[idx_train],
        sample_id_validation=sample_ids[idx_validation],
        sample_id_test=sample_ids[idx_test],
        prefix_train=prefixes[idx_train],
        prefix_validation=prefixes[idx_validation],
        prefix_test=prefixes[idx_test],
        b_ext_t_train=b_ext_t[idx_train],
        b_ext_t_validation=b_ext_t[idx_validation],
        b_ext_t_test=b_ext_t[idx_test],
        external_field_enabled=np.asarray([normalized["field_enabled"]], dtype=bool),
        external_field_magnitude_t=np.asarray([normalized["field_magnitude_t"]], dtype=np.float64),
        external_field_direction=np.asarray(normalized["field_direction"], dtype=np.float64),
        split_seed=np.asarray([normalized["random_seed"]], dtype=np.int64),
        ratios=np.asarray([
            normalized["train_ratio"], normalized["validation_ratio"],
            normalized["test_ratio"],
        ], dtype=np.float64),
        subset_names=np.asarray(SUBSET_NAMES, dtype=str),
    )

    train_ids = set(sample_ids[idx_train].tolist())
    validation_ids = set(sample_ids[idx_validation].tolist())
    test_ids = set(sample_ids[idx_test].tolist())
    overlap = {
        "train_validation": len(train_ids & validation_ids),
        "train_test": len(train_ids & test_ids),
        "validation_test": len(validation_ids & test_ids),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Cross-subset overlap detected: {overlap}")

    summary = {
        "created_at_utc": utc_now_iso(),
        "system": config["system"],
        "config_file": safe_public_path(config_path, repository_root),
        "config_sha256": sha256_file(config_path),
        "prepared_dataset_npz": safe_public_path(dataset_file, repository_root),
        "prepared_dataset_sha256": sha256_file(dataset_file),
        "accepted_metadata_csv": safe_public_path(metadata_file, repository_root),
        "accepted_metadata_sha256": sha256_file(metadata_file),
        "split_method": "random permutation without replacement",
        "random_number_generator": "numpy.random.default_rng",
        "random_seed": normalized["random_seed"],
        "external_magnetic_field": {
            "enabled": normalized["field_enabled"],
            "magnitude_t": normalized["field_magnitude_t"],
            "direction": normalized["field_direction"].tolist(),
            "tolerance_t": normalized["field_tolerance_t"],
            "complete_consistency_check": True,
        },
        "ratios": {
            "train": normalized["train_ratio"],
            "validation": normalized["validation_ratio"],
            "test": normalized["test_ratio"],
        },
        "counts": {
            "total": n, "train": n_train,
            "validation": n_validation, "test": n_test,
        },
        "rounding_rule": {
            "train": "int(train_ratio * N)",
            "validation": "int(validation_ratio * N)",
            "test": "N - n_train - n_validation",
        },
        "integrity_checks": {
            "complete_dataset_coverage": True,
            "unique_dataset_indices": True,
            "unique_sample_ids": True,
            "external_field_consistent_across_all_samples": True,
            "cross_subset_sample_id_overlap": overlap,
            "duplicate_sample_id_rows": int(frame["sample_id"].duplicated().sum()),
            "duplicate_prefix_rows": int(frame["prefix"].duplicated(keep=False).sum()),
        },
        "subsets": subset_summaries(frame),
        "output_files": {
            "split_csv": safe_public_path(split_csv, repository_root),
            "split_indices_npz": safe_public_path(indices_npz, repository_root),
            "summary_json": safe_public_path(summary_json, repository_root),
        },
        "output_sha256": {
            "split_csv": sha256_file(split_csv),
            "split_indices_npz": sha256_file(indices_npz),
        },
        "software": software_versions({"pandas": pd.__version__}),
    }
    write_json(summary_json, summary)

    print("=" * 80)
    print(f"System         : {system_name}")
    print(f"Dataset        : {safe_public_path(dataset_file, repository_root)}")
    print(f"External field : {normalized['field_magnitude_t']:.6f} T")
    print(f"Field direction: {normalized['field_direction'].tolist()}")
    print(f"Random seed    : {normalized['random_seed']}")
    print(f"Total          : {n}")
    print(f"Train          : {n_train}  X={X[idx_train].shape}  Y={Y[idx_train].shape}")
    print(f"Validation     : {n_validation}  X={X[idx_validation].shape}  Y={Y[idx_validation].shape}")
    print(f"Test           : {n_test}  X={X[idx_test].shape}  Y={Y[idx_test].shape}")
    print(f"Split CSV      : {safe_public_path(split_csv, repository_root)}")
    print(f"Indices NPZ    : {safe_public_path(indices_npz, repository_root)}")
    print(f"Summary JSON   : {safe_public_path(summary_json, repository_root)}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

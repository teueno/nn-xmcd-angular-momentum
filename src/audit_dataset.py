#!/usr/bin/env python3

"""
Audit a prepared dataset for:

- duplicated sample IDs,
- duplicated Prefix values,
- duplicated parameter configurations,
- exactly identical spectra,
- numerically identical spectra after rounding,
- duplicates that cross train/validation/test boundaries,
- identical inputs associated with different labels.

The audit is performed after prepare_dataset.py and split_dataset.py.

Internal calculations use absolute paths. Paths written to JSON files
and displayed in normal console output are converted to paths relative
to the repository root.
"""

import argparse
import csv
from collections import Counter
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# Support both:
#
#   python src/audit_dataset.py
#
# and:
#
#   python -m src.audit_dataset

try:
    from .utils import (
        check_output_files,
        ensure_directory,
        ensure_finite,
        get_repository_root,
        load_json,
        public_path,
        resolve_path,
        sha256_array,
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
        sha256_array,
        sha256_file,
        software_versions,
        utc_now_iso,
        write_json,
    )


# ============================================================
# Arguments
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Audit duplicate samples, "
            "split leakage, and label consistency."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Dataset JSON configuration file.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing audit outputs.",
    )

    return parser.parse_args()


# ============================================================
# Safe public path
# ============================================================

def safe_public_path(
    path,
    repository_root,
):
    """
    Return a path suitable for public output.

    Paths inside the repository are represented relative to the
    repository root. If a path is outside the repository, only its
    file name is returned to avoid exposing the local directory
    structure.
    """

    try:
        return public_path(
            path,
            repository_root,
        )

    except ValueError:
        return Path(
            path
        ).name


# ============================================================
# Duplicate grouping
# ============================================================

def grouped_rows(values):
    """
    Group row indices by a hash or signature.

    Only groups containing two or more rows are returned.
    """

    groups = defaultdict(
        list
    )

    for index, value in enumerate(
        values
    ):
        groups[
            value
        ].append(
            index
        )

    return {
        key: rows
        for key, rows in groups.items()
        if len(rows) > 1
    }


# ============================================================
# Parameter-column detection
# ============================================================

def parameter_columns_from_config(
    config,
    metadata,
):
    """
    Determine the columns defining a Quanty parameter configuration.

    Explicit columns in config.audit.parameter_columns take priority.
    Otherwise, common column names are detected automatically.
    """

    explicit_columns = (
        config.get(
            "audit",
            {},
        ).get(
            "parameter_columns"
        )
    )

    if explicit_columns:
        missing_columns = [
            column
            for column in explicit_columns
            if column not in metadata.columns
        ]

        if missing_columns:
            raise ValueError(
                "Configured parameter columns "
                "are missing from accepted metadata: "
                f"{missing_columns}"
            )

        return list(
            explicit_columns
        )

    candidate_columns = [
        "10Dq",
        "Slater",
        "SOC2p",
        "SOC3d",
        "Hex",

        "ten_dq",
        "slater",
        "soc_2p",
        "soc_3d",
        "h_ex",

        "parameter_ten_dq",
        "parameter_slater",
        "parameter_soc_2p",
        "parameter_soc_3d",
        "parameter_h_ex",
    ]

    return [
        column
        for column in candidate_columns
        if column in metadata.columns
    ]


def parameter_signatures(
    metadata,
    parameter_columns,
    decimals,
):
    """
    Build a tuple representing each sample's parameter configuration.
    """

    signatures = []

    for _, row in metadata.iterrows():
        values = []

        for column in parameter_columns:
            value = row[
                column
            ]

            if pd.isna(value):
                values.append(
                    None
                )

            elif isinstance(
                value,
                (
                    float,
                    np.floating,
                    int,
                    np.integer,
                ),
            ):
                values.append(
                    round(
                        float(value),
                        decimals,
                    )
                )

            else:
                values.append(
                    str(value)
                )

        signatures.append(
            tuple(values)
        )

    return signatures


# ============================================================
# Audit helpers
# ============================================================

def group_to_records(
    groups,
    metadata,
    duplicate_type,
):
    """
    Convert duplicate groups to rows for the audit CSV.
    """

    records = []

    for group_index, (
        signature,
        row_indices,
    ) in enumerate(
        groups.items()
    ):
        group_metadata = (
            metadata.iloc[
                row_indices
            ]
        )

        subsets = sorted(
            set(
                group_metadata[
                    "subset"
                ].astype(str)
            )
        )

        sample_ids = (
            group_metadata[
                "sample_id"
            ]
            .astype(str)
            .tolist()
        )

        prefixes = (
            group_metadata[
                "prefix"
            ]
            .astype(str)
            .tolist()
        )

        records.append(
            {
                "duplicate_type": (
                    duplicate_type
                ),

                "group_index": (
                    group_index
                ),

                "signature": str(
                    signature
                ),

                "row_count": len(
                    row_indices
                ),

                "cross_split": (
                    len(subsets) > 1
                ),

                "subsets": ";".join(
                    subsets
                ),

                "sample_ids": ";".join(
                    sample_ids
                ),

                "prefixes": ";".join(
                    prefixes
                ),
            }
        )

    return records


def label_conflict_groups(
    duplicate_groups,
    target_array,
    tolerance,
):
    """
    Find duplicate inputs associated with inconsistent target values.
    """

    conflicts = {}

    for signature, row_indices in (
        duplicate_groups.items()
    ):
        labels = target_array[
            row_indices
        ]

        reference = labels[
            0
        ]

        if np.any(
            np.abs(
                labels
                - reference
            )
            > tolerance
        ):
            conflicts[
                signature
            ] = row_indices

    return conflicts


def cross_split_count(
    duplicate_groups,
    metadata,
):
    """
    Count duplicate groups spanning multiple subsets.
    """

    group_count = 0
    row_count = 0

    for row_indices in (
        duplicate_groups.values()
    ):
        subsets = set(
            metadata.iloc[
                row_indices
            ][
                "subset"
            ].astype(str)
        )

        if len(subsets) > 1:
            group_count += 1

            row_count += len(
                row_indices
            )

    return (
        group_count,
        row_count,
    )


def duplicated_row_count(groups):
    """
    Return the number of rows belonging to duplicate groups.
    """

    return int(
        sum(
            len(rows)
            for rows in groups.values()
        )
    )


# ============================================================
# Main
# ============================================================

def main():
    arguments = parse_arguments()

    config_path = (
        arguments.config
        .expanduser()
        .resolve()
    )

    config = load_json(
        config_path
    )

    config_directory = (
        config_path.parent
    )

    # --------------------------------------------------------
    # Validate configuration
    # --------------------------------------------------------

    if "system" not in config:
        raise ValueError(
            "Config must contain "
            "the system section"
        )

    if "name" not in config["system"]:
        raise ValueError(
            "Config must contain "
            "system.name"
        )

    if "paths" not in config:
        raise ValueError(
            "Config must contain "
            "the paths section"
        )

    repository_root = get_repository_root(
        config,
        config_path,
    )

    system_name = (
        config["system"]["name"]
    )

    paths = config["paths"]

    audit_config = config.get(
        "audit",
        {},
    )

    # --------------------------------------------------------
    # Input and output paths
    # --------------------------------------------------------

    required_path_keys = (
        "prepared_dataset_npz",
        "accepted_metadata_csv",
        "split_csv",
    )

    missing_path_keys = [
        key
        for key in required_path_keys
        if key not in paths
    ]

    if missing_path_keys:
        raise ValueError(
            "Config paths section is missing: "
            f"{missing_path_keys}"
        )

    dataset_file = resolve_path(
        paths[
            "prepared_dataset_npz"
        ],
        config_directory,
    )

    accepted_metadata_file = resolve_path(
        paths[
            "accepted_metadata_csv"
        ],
        config_directory,
    )

    split_file = resolve_path(
        paths[
            "split_csv"
        ],
        config_directory,
    )

    audit_directory = ensure_directory(
        resolve_path(
            paths.get(
                "audit_output_directory",
                "../../results/audit",
            ),
            config_directory,
        )
    )

    summary_file = (
        audit_directory
        / (
            f"{system_name}_"
            "audit_summary.json"
        )
    )

    duplicate_file = (
        audit_directory
        / (
            f"{system_name}_"
            "duplicate_groups.csv"
        )
    )

    check_output_files(
        [
            summary_file,
            duplicate_file,
        ],
        overwrite=(
            arguments.overwrite
        ),
        label="Audit outputs",
    )

    for required_file in (
        dataset_file,
        accepted_metadata_file,
        split_file,
    ):
        if not required_file.is_file():
            raise FileNotFoundError(
                "Required input file was not found: "
                f"{safe_public_path(required_file, repository_root)}"
            )

    # --------------------------------------------------------
    # Load prepared dataset
    # --------------------------------------------------------

    with np.load(
        dataset_file,
        allow_pickle=False,
    ) as dataset:
        required_arrays = {
            "X",
            "Y",
            "sample_id",
            "prefix",
        }

        missing_arrays = (
            required_arrays
            - set(dataset.files)
        )

        if missing_arrays:
            raise ValueError(
                "Prepared dataset is missing "
                f"arrays: {sorted(missing_arrays)}"
            )

        X = np.asarray(
            dataset["X"],
            dtype=np.float64,
        )

        Y = np.asarray(
            dataset["Y"],
            dtype=np.float64,
        )

        sample_ids = (
            dataset[
                "sample_id"
            ]
            .astype(str)
        )

        prefixes = (
            dataset[
                "prefix"
            ]
            .astype(str)
        )

    ensure_finite(
        "X",
        X,
    )

    ensure_finite(
        "Y",
        Y,
    )

    if X.ndim != 2:
        raise ValueError(
            f"Unexpected X shape: {X.shape}"
        )

    if Y.shape != (
        len(X),
        2,
    ):
        raise ValueError(
            f"Unexpected Y shape: {Y.shape}"
        )

    if len(sample_ids) != len(X):
        raise ValueError(
            "sample_id count differs "
            "from dataset sample count"
        )

    if len(prefixes) != len(X):
        raise ValueError(
            "Prefix count differs "
            "from dataset sample count"
        )

    if len(set(sample_ids.tolist())) != len(
        sample_ids
    ):
        raise ValueError(
            "Prepared dataset contains "
            "duplicate sample_id values"
        )

    # --------------------------------------------------------
    # Load metadata and split
    # --------------------------------------------------------

    accepted_metadata = pd.read_csv(
        accepted_metadata_file
    )

    split_dataframe = pd.read_csv(
        split_file
    )

    required_metadata_columns = {
        "sample_id",
        "prefix",
    }

    if not required_metadata_columns.issubset(
        accepted_metadata.columns
    ):
        raise ValueError(
            "Accepted metadata must contain "
            "sample_id and prefix"
        )

    required_split_columns = {
        "sample_id",
        "subset",
        "dataset_index",
    }

    if not required_split_columns.issubset(
        split_dataframe.columns
    ):
        raise ValueError(
            "Split CSV must contain "
            "sample_id, subset, and dataset_index"
        )

    if accepted_metadata[
        "sample_id"
    ].duplicated().any():
        duplicated_ids = (
            accepted_metadata.loc[
                accepted_metadata[
                    "sample_id"
                ].duplicated(
                    keep=False
                ),
                "sample_id",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "Accepted metadata contains "
            "duplicate sample_id values: "
            f"{duplicated_ids[:10]}"
        )

    if split_dataframe[
        "sample_id"
    ].duplicated().any():
        duplicated_ids = (
            split_dataframe.loc[
                split_dataframe[
                    "sample_id"
                ].duplicated(
                    keep=False
                ),
                "sample_id",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "Split CSV contains duplicate "
            "sample_id values: "
            f"{duplicated_ids[:10]}"
        )

    # --------------------------------------------------------
    # Align metadata to NPZ order
    # --------------------------------------------------------

    accepted_by_id = (
        accepted_metadata.set_index(
            "sample_id",
            drop=False,
        )
    )

    split_by_id = (
        split_dataframe.set_index(
            "sample_id",
            drop=False,
        )
    )

    dataset_id_set = set(
        sample_ids.tolist()
    )

    accepted_id_set = set(
        accepted_by_id.index.astype(
            str
        )
    )

    split_id_set = set(
        split_by_id.index.astype(
            str
        )
    )

    if dataset_id_set != accepted_id_set:
        missing_from_metadata = sorted(
            dataset_id_set
            - accepted_id_set
        )

        extra_in_metadata = sorted(
            accepted_id_set
            - dataset_id_set
        )

        raise ValueError(
            "Prepared dataset and accepted metadata "
            "have different sample IDs. "
            f"Missing={len(missing_from_metadata)}, "
            f"extra={len(extra_in_metadata)}"
        )

    if dataset_id_set != split_id_set:
        missing_from_split = sorted(
            dataset_id_set
            - split_id_set
        )

        extra_in_split = sorted(
            split_id_set
            - dataset_id_set
        )

        raise ValueError(
            "Prepared dataset and split CSV "
            "have different sample IDs. "
            f"Missing={len(missing_from_split)}, "
            f"extra={len(extra_in_split)}"
        )

    metadata = (
        accepted_by_id
        .loc[sample_ids]
        .reset_index(drop=True)
        .copy()
    )

    split_ordered = (
        split_by_id
        .loc[sample_ids]
        .reset_index(drop=True)
    )

    metadata[
        "subset"
    ] = (
        split_ordered[
            "subset"
        ]
        .astype(str)
        .to_numpy()
    )

    metadata[
        "dataset_index"
    ] = (
        split_ordered[
            "dataset_index"
        ]
        .to_numpy(
            dtype=int
        )
    )

    valid_subsets = {
        "train",
        "validation",
        "test",
    }

    observed_subsets = set(
        metadata[
            "subset"
        ].astype(str)
    )

    invalid_subsets = (
        observed_subsets
        - valid_subsets
    )

    if invalid_subsets:
        raise ValueError(
            "Unknown subset names were found: "
            f"{sorted(invalid_subsets)}"
        )

    if not np.array_equal(
        metadata[
            "prefix"
        ]
        .astype(str)
        .to_numpy(),
        prefixes,
    ):
        raise ValueError(
            "Prefix order differs between "
            "the NPZ dataset and metadata"
        )

    if not np.array_equal(
        metadata[
            "dataset_index"
        ].to_numpy(),
        np.arange(
            len(X)
        ),
    ):
        raise ValueError(
            "dataset_index does not match "
            "the NPZ row order"
        )

    # --------------------------------------------------------
    # Exact and rounded spectrum hashes
    # --------------------------------------------------------

    exact_hashes = [
        sha256_array(
            spectrum
        )
        for spectrum in X
    ]

    spectrum_round_decimals = int(
        audit_config.get(
            "spectrum_round_decimals",
            12,
        )
    )

    if spectrum_round_decimals < 0:
        raise ValueError(
            "spectrum_round_decimals "
            "must be nonnegative"
        )

    rounded_hashes = [
        sha256_array(
            spectrum,
            decimals=(
                spectrum_round_decimals
            ),
        )
        for spectrum in X
    ]

    exact_spectrum_groups = grouped_rows(
        exact_hashes
    )

    rounded_spectrum_groups = grouped_rows(
        rounded_hashes
    )

    # --------------------------------------------------------
    # Parameter duplicates
    # --------------------------------------------------------

    parameter_columns = (
        parameter_columns_from_config(
            config,
            metadata,
        )
    )

    parameter_round_decimals = int(
        audit_config.get(
            "parameter_round_decimals",
            12,
        )
    )

    if parameter_round_decimals < 0:
        raise ValueError(
            "parameter_round_decimals "
            "must be nonnegative"
        )

    if parameter_columns:
        signatures = parameter_signatures(
            metadata,
            parameter_columns,
            parameter_round_decimals,
        )

        parameter_groups = grouped_rows(
            signatures
        )

    else:
        parameter_groups = {}

    # --------------------------------------------------------
    # Label consistency
    # --------------------------------------------------------

    label_tolerance = float(
        audit_config.get(
            "label_tolerance",
            1.0e-12,
        )
    )

    if label_tolerance < 0:
        raise ValueError(
            "label_tolerance "
            "must be nonnegative"
        )

    exact_label_conflicts = (
        label_conflict_groups(
            exact_spectrum_groups,
            Y,
            label_tolerance,
        )
    )

    rounded_label_conflicts = (
        label_conflict_groups(
            rounded_spectrum_groups,
            Y,
            label_tolerance,
        )
    )

    # --------------------------------------------------------
    # Cross-split leakage
    # --------------------------------------------------------

    (
        exact_cross_split_groups,
        exact_cross_split_rows,
    ) = cross_split_count(
        exact_spectrum_groups,
        metadata,
    )

    (
        rounded_cross_split_groups,
        rounded_cross_split_rows,
    ) = cross_split_count(
        rounded_spectrum_groups,
        metadata,
    )

    (
        parameter_cross_split_groups,
        parameter_cross_split_rows,
    ) = cross_split_count(
        parameter_groups,
        metadata,
    )

    # --------------------------------------------------------
    # Detailed duplicate CSV
    # --------------------------------------------------------

    duplicate_records = []

    duplicate_records.extend(
        group_to_records(
            parameter_groups,
            metadata,
            "parameter_configuration",
        )
    )

    duplicate_records.extend(
        group_to_records(
            exact_spectrum_groups,
            metadata,
            "exact_spectrum",
        )
    )

    duplicate_records.extend(
        group_to_records(
            rounded_spectrum_groups,
            metadata,
            "rounded_spectrum",
        )
    )

    duplicate_records.extend(
        group_to_records(
            exact_label_conflicts,
            metadata,
            "exact_input_different_labels",
        )
    )

    duplicate_records.extend(
        group_to_records(
            rounded_label_conflicts,
            metadata,
            "rounded_input_different_labels",
        )
    )

    duplicate_fieldnames = [
        "duplicate_type",
        "group_index",
        "signature",
        "row_count",
        "cross_split",
        "subsets",
        "sample_ids",
        "prefixes",
    ]

    with duplicate_file.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                duplicate_fieldnames
            ),
        )

        writer.writeheader()

        writer.writerows(
            duplicate_records
        )

    # --------------------------------------------------------
    # Pass/fail judgment
    # --------------------------------------------------------

    failure_reasons = []

    if exact_cross_split_groups > 0:
        failure_reasons.append(
            "exact_spectrum_duplicates_cross_split"
        )

    if rounded_cross_split_groups > 0:
        failure_reasons.append(
            "rounded_spectrum_duplicates_cross_split"
        )

    if len(exact_label_conflicts) > 0:
        failure_reasons.append(
            "identical_inputs_with_different_labels"
        )

    if len(rounded_label_conflicts) > 0:
        failure_reasons.append(
            "rounded_identical_inputs_with_different_labels"
        )

    audit_status = (
        "pass"
        if not failure_reasons
        else "fail"
    )

    subset_counts = Counter(
        metadata[
            "subset"
        ].astype(str)
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "created_at_utc": (
            utc_now_iso()
        ),

        "system": (
            config["system"]
        ),

        "config_file": public_path(
            config_path,
            repository_root,
        ),

        "config_sha256": (
            sha256_file(
                config_path
            )
        ),

        "input_files": {
            "prepared_dataset_npz": public_path(
                dataset_file,
                repository_root,
            ),

            "accepted_metadata_csv": public_path(
                accepted_metadata_file,
                repository_root,
            ),

            "split_csv": public_path(
                split_file,
                repository_root,
            ),
        },

        "input_sha256": {
            "prepared_dataset_npz": (
                sha256_file(
                    dataset_file
                )
            ),

            "accepted_metadata_csv": (
                sha256_file(
                    accepted_metadata_file
                )
            ),

            "split_csv": (
                sha256_file(
                    split_file
                )
            ),
        },

        "counts": {
            "samples": int(
                len(X)
            ),

            "features": int(
                X.shape[1]
            ),

            "train": int(
                subset_counts.get(
                    "train",
                    0,
                )
            ),

            "validation": int(
                subset_counts.get(
                    "validation",
                    0,
                )
            ),

            "test": int(
                subset_counts.get(
                    "test",
                    0,
                )
            ),

            "unique_sample_ids": int(
                metadata[
                    "sample_id"
                ].nunique()
            ),

            "unique_prefixes": int(
                metadata[
                    "prefix"
                ].nunique()
            ),
        },

        "parameter_audit": {
            "columns": (
                parameter_columns
            ),

            "round_decimals": (
                parameter_round_decimals
            ),

            "duplicate_groups": int(
                len(
                    parameter_groups
                )
            ),

            "duplicate_rows": (
                duplicated_row_count(
                    parameter_groups
                )
            ),

            "cross_split_groups": int(
                parameter_cross_split_groups
            ),

            "cross_split_rows": int(
                parameter_cross_split_rows
            ),
        },

        "spectrum_audit": {
            "exact_duplicate_groups": int(
                len(
                    exact_spectrum_groups
                )
            ),

            "exact_duplicate_rows": (
                duplicated_row_count(
                    exact_spectrum_groups
                )
            ),

            "exact_cross_split_groups": int(
                exact_cross_split_groups
            ),

            "exact_cross_split_rows": int(
                exact_cross_split_rows
            ),

            "rounded_decimals": (
                spectrum_round_decimals
            ),

            "rounded_duplicate_groups": int(
                len(
                    rounded_spectrum_groups
                )
            ),

            "rounded_duplicate_rows": (
                duplicated_row_count(
                    rounded_spectrum_groups
                )
            ),

            "rounded_cross_split_groups": int(
                rounded_cross_split_groups
            ),

            "rounded_cross_split_rows": int(
                rounded_cross_split_rows
            ),
        },

        "label_consistency": {
            "tolerance": (
                label_tolerance
            ),

            "exact_input_different_label_groups": int(
                len(
                    exact_label_conflicts
                )
            ),

            "rounded_input_different_label_groups": int(
                len(
                    rounded_label_conflicts
                )
            ),
        },

        "audit_status": (
            audit_status
        ),

        "failure_reasons": (
            failure_reasons
        ),

        "output_files": {
            "summary_json": public_path(
                summary_file,
                repository_root,
            ),

            "duplicate_groups_csv": public_path(
                duplicate_file,
                repository_root,
            ),
        },

        "software": (
            software_versions(
                {
                    "pandas": (
                        pd.__version__
                    )
                }
            )
        ),
    }

    write_json(
        summary_file,
        summary,
    )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print("=" * 80)

    print(
        f"System                       : {system_name}"
    )

    print(
        f"Samples                      : {len(X)}"
    )

    print(
        "Parameter duplicate groups   : "
        f"{len(parameter_groups)}"
    )

    print(
        "Parameter cross-split groups : "
        f"{parameter_cross_split_groups}"
    )

    print(
        "Exact spectrum duplicates    : "
        f"{len(exact_spectrum_groups)}"
    )

    print(
        "Rounded spectrum duplicates  : "
        f"{len(rounded_spectrum_groups)}"
    )

    print(
        "Exact cross-split groups      : "
        f"{exact_cross_split_groups}"
    )

    print(
        "Rounded cross-split groups    : "
        f"{rounded_cross_split_groups}"
    )

    print(
        "Exact input/label conflicts   : "
        f"{len(exact_label_conflicts)}"
    )

    print(
        "Rounded input/label conflicts : "
        f"{len(rounded_label_conflicts)}"
    )

    print(
        f"Audit status                 : {audit_status}"
    )

    print(
        "Summary                      : "
        f"{public_path(summary_file, repository_root)}"
    )

    print(
        "Duplicate details            : "
        f"{public_path(duplicate_file, repository_root)}"
    )

    print("=" * 80)

    # Nonzero exit status when split leakage or label
    # inconsistencies have been detected.
    if failure_reasons:
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3

"""
Create reproducible train, validation, and test assignments.

The split is created from the unnormalized prepared dataset.
All data-dependent preprocessing must be fitted later using the
training subset only.

Outputs
-------
<system>_split.csv
    Human-readable assignment table containing the original dataset
    index, sample ID, subset name, random seed, Prefix, labels, and
    source metadata.

<system>_split_indices.npz
    Machine-readable train, validation, and test index arrays and
    corresponding sample IDs.

<system>_split_summary.json
    Split method, ratios, counts, target distributions, overlap
    checks, hashes, and software versions.

Privacy
-------
Internal calculations use absolute paths. Paths written to public
JSON files and displayed in normal console output are converted to
paths relative to the repository root.
"""

import argparse
import csv
from numbers import Integral
from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd


# Support both:
#
#   python src/split_dataset.py
#
# and:
#
#   python -m src.split_dataset

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


# ============================================================
# Constants
# ============================================================

SUBSET_NAMES = (
    "train",
    "validation",
    "test",
)


# ============================================================
# Command-line arguments
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Create reproducible train, validation, "
            "and test assignments."
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
        help="Overwrite existing split files.",
    )

    return parser.parse_args()


# ============================================================
# Public-path utility
# ============================================================

def safe_public_path(
    path,
    repository_root,
):
    """
    Return a path suitable for public output.

    Paths inside the repository are represented relative to the
    repository root. If a path is outside the repository, only its
    final file or directory name is returned.
    """

    if path is None:
        return None

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
# Configuration validation
# ============================================================

def validate_ratio(
    name,
    value,
    allow_zero,
):
    """
    Validate a finite split ratio.

    Parameters
    ----------
    name : str
        Setting name used in error messages.

    value : object
        Ratio value.

    allow_zero : bool
        Whether zero is allowed.

    Returns
    -------
    float
        Validated ratio.
    """

    if isinstance(
        value,
        bool,
    ):
        raise TypeError(
            f"{name} must be numeric, not bool"
        )

    if not isinstance(
        value,
        Real,
    ):
        raise TypeError(
            f"{name} must be numeric"
        )

    ratio = float(
        value
    )

    if not np.isfinite(
        ratio
    ):
        raise ValueError(
            f"{name} must be finite"
        )

    if allow_zero:
        if ratio < 0.0:
            raise ValueError(
                f"{name} must be nonnegative"
            )

    else:
        if ratio <= 0.0:
            raise ValueError(
                f"{name} must be positive"
            )

    if ratio > 1.0:
        raise ValueError(
            f"{name} must not exceed 1.0"
        )

    return ratio


def validate_random_seed(
    value,
):
    """
    Validate and normalize the random seed.
    """

    if isinstance(
        value,
        bool,
    ):
        raise TypeError(
            "split.random_seed must be "
            "an integer, not bool"
        )

    if not isinstance(
        value,
        Integral,
    ):
        raise TypeError(
            "split.random_seed must be "
            "an integer"
        )

    seed = int(
        value
    )

    if seed < 0:
        raise ValueError(
            "split.random_seed must be "
            "nonnegative"
        )

    return seed


def validate_config(
    config,
):
    """
    Validate and normalize split configuration.

    Returns
    -------
    dict
        Normalized split settings.
    """

    if not isinstance(
        config,
        dict,
    ):
        raise TypeError(
            "Configuration must be a dictionary"
        )

    required_sections = {
        "system",
        "paths",
        "split",
    }

    missing_sections = (
        required_sections
        - set(config)
    )

    if missing_sections:
        raise ValueError(
            "Missing configuration sections: "
            f"{sorted(missing_sections)}"
        )

    if "name" not in config[
        "system"
    ]:
        raise ValueError(
            "Config must contain system.name"
        )

    system_name = str(
        config[
            "system"
        ][
            "name"
        ]
    ).strip()

    if not system_name:
        raise ValueError(
            "system.name must not be empty"
        )

    required_path_settings = {
        "repository_root",
        "prepared_dataset_npz",
        "accepted_metadata_csv",
        "split_output_directory",
    }

    missing_path_settings = (
        required_path_settings
        - set(
            config[
                "paths"
            ]
        )
    )

    if missing_path_settings:
        raise ValueError(
            "Missing path settings: "
            f"{sorted(missing_path_settings)}"
        )

    split_config = config[
        "split"
    ]

    if not isinstance(
        split_config,
        dict,
    ):
        raise TypeError(
            "split must be a dictionary"
        )

    train_ratio = validate_ratio(
        "split.train_ratio",
        split_config.get(
            "train_ratio",
            0.70,
        ),
        allow_zero=False,
    )

    validation_ratio = validate_ratio(
        "split.validation_ratio",
        split_config.get(
            "validation_ratio",
            0.15,
        ),
        allow_zero=True,
    )

    if "test_ratio" in split_config:
        test_ratio_value = split_config[
            "test_ratio"
        ]

    else:
        test_ratio_value = (
            1.0
            - train_ratio
            - validation_ratio
        )

    test_ratio = validate_ratio(
        "split.test_ratio",
        test_ratio_value,
        allow_zero=False,
    )

    ratio_sum = (
        train_ratio
        + validation_ratio
        + test_ratio
    )

    if not np.isclose(
        ratio_sum,
        1.0,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            "train_ratio + validation_ratio "
            "+ test_ratio must equal 1.0. "
            f"Received {ratio_sum:.17g}"
        )

    random_seed = validate_random_seed(
        split_config.get(
            "random_seed",
            1234,
        )
    )

    return {
        "system_name": (
            system_name
        ),

        "train_ratio": (
            train_ratio
        ),

        "validation_ratio": (
            validation_ratio
        ),

        "test_ratio": (
            test_ratio
        ),

        "random_seed": (
            random_seed
        ),
    }


# ============================================================
# Numeric summaries
# ============================================================

def summarize_numeric_values(
    values,
):
    """
    Return a JSON-serializable numeric summary.
    """

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.ndim != 1:
        raise ValueError(
            "Summary input must be "
            "one-dimensional"
        )

    if array.size == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std_population": None,
        }

    ensure_finite(
        "Summary values",
        array,
    )

    return {
        "count": int(
            array.size
        ),

        "min": float(
            np.min(
                array
            )
        ),

        "max": float(
            np.max(
                array
            )
        ),

        "mean": float(
            np.mean(
                array
            )
        ),

        "std_population": float(
            np.std(
                array,
                ddof=0,
            )
        ),
    }


def summarize_subsets(
    split_dataframe,
    target_columns,
):
    """
    Return sample counts and target summaries for each subset.
    """

    summaries = {}

    for subset_name in SUBSET_NAMES:
        subset_dataframe = (
            split_dataframe.loc[
                split_dataframe[
                    "subset"
                ]
                == subset_name
            ]
        )

        target_summaries = {}

        for target_column in target_columns:
            values = (
                subset_dataframe[
                    target_column
                ]
                .to_numpy(
                    dtype=np.float64
                )
            )

            target_summaries[
                target_column
            ] = summarize_numeric_values(
                values
            )

        summaries[
            subset_name
        ] = {
            "count": int(
                len(
                    subset_dataframe
                )
            ),

            "targets": (
                target_summaries
            ),
        }

    return summaries


# ============================================================
# Dataset and metadata validation
# ============================================================

def validate_prepared_dataset(
    X,
    Y,
    sample_ids,
    prefixes,
):
    """
    Validate the prepared NPZ dataset.

    Returns
    -------
    int
        Number of samples.
    """

    if X.ndim != 2:
        raise ValueError(
            "X must be two-dimensional. "
            f"Found shape {X.shape}"
        )

    sample_count = int(
        X.shape[
            0
        ]
    )

    if sample_count == 0:
        raise ValueError(
            "Prepared dataset contains no samples"
        )

    if Y.shape != (
        sample_count,
        2,
    ):
        raise ValueError(
            "Expected Y shape "
            f"({sample_count}, 2), "
            f"found {Y.shape}"
        )

    if sample_ids.shape != (
        sample_count,
    ):
        raise ValueError(
            "sample_id array shape differs "
            "from the X sample count"
        )

    if prefixes.shape != (
        sample_count,
    ):
        raise ValueError(
            "Prefix array shape differs "
            "from the X sample count"
        )

    ensure_finite(
        "Prepared X",
        X,
    )

    ensure_finite(
        "Prepared Y",
        Y,
    )

    if any(
        not str(value).strip()
        for value in sample_ids
    ):
        raise ValueError(
            "Prepared dataset contains "
            "empty sample_id values"
        )

    if any(
        not str(value).strip()
        for value in prefixes
    ):
        raise ValueError(
            "Prepared dataset contains "
            "empty Prefix values"
        )

    if len(
        set(
            sample_ids.tolist()
        )
    ) != sample_count:
        raise ValueError(
            "Duplicate sample_id values were "
            "found in the NPZ dataset"
        )

    if len(
        set(
            prefixes.tolist()
        )
    ) != sample_count:
        raise ValueError(
            "Duplicate Prefix values were "
            "found in the NPZ dataset"
        )

    return sample_count


def align_and_validate_metadata(
    metadata_dataframe,
    sample_ids,
    prefixes,
    Y,
):
    """
    Align accepted metadata to NPZ sample order and validate labels.

    Returns
    -------
    pandas.DataFrame
        Metadata ordered identically to the NPZ sample arrays.
    """

    required_metadata_columns = {
        "sample_id",
        "prefix",
        "sz_label",
        "lz_label",
    }

    missing_metadata_columns = (
        required_metadata_columns
        - set(
            metadata_dataframe.columns
        )
    )

    if missing_metadata_columns:
        raise ValueError(
            "Accepted metadata is missing columns: "
            f"{sorted(missing_metadata_columns)}"
        )

    if metadata_dataframe[
        "sample_id"
    ].duplicated().any():
        raise ValueError(
            "Accepted metadata contains "
            "duplicate sample_id values"
        )

    metadata_sample_ids = (
        metadata_dataframe[
            "sample_id"
        ]
        .astype(str)
    )

    metadata_by_id = (
        metadata_dataframe.assign(
            sample_id=(
                metadata_sample_ids
            )
        )
        .set_index(
            "sample_id",
            drop=False,
        )
    )

    metadata_id_set = set(
        metadata_by_id.index
    )

    dataset_id_set = set(
        sample_ids.tolist()
    )

    missing_ids = sorted(
        dataset_id_set
        - metadata_id_set
    )

    extra_ids = sorted(
        metadata_id_set
        - dataset_id_set
    )

    if missing_ids or extra_ids:
        raise ValueError(
            "Prepared dataset and accepted metadata "
            "have different sample IDs. "
            f"Missing={len(missing_ids)}, "
            f"extra={len(extra_ids)}"
        )

    metadata_ordered = (
        metadata_by_id
        .loc[
            sample_ids
        ]
        .reset_index(
            drop=True
        )
        .copy()
    )

    metadata_prefixes = (
        metadata_ordered[
            "prefix"
        ]
        .astype(str)
        .to_numpy()
    )

    if not np.array_equal(
        metadata_prefixes,
        prefixes,
    ):
        raise ValueError(
            "Prefix order differs between "
            "the NPZ dataset and metadata"
        )

    metadata_sz = pd.to_numeric(
        metadata_ordered[
            "sz_label"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=np.float64
    )

    metadata_lz = pd.to_numeric(
        metadata_ordered[
            "lz_label"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=np.float64
    )

    ensure_finite(
        "Metadata Sz labels",
        metadata_sz,
    )

    ensure_finite(
        "Metadata Lz labels",
        metadata_lz,
    )

    if not np.allclose(
        metadata_sz,
        Y[
            :,
            0
        ],
        rtol=1.0e-12,
        atol=1.0e-12,
        equal_nan=False,
    ):
        maximum_difference = float(
            np.max(
                np.abs(
                    metadata_sz
                    - Y[
                        :,
                        0
                    ]
                )
            )
        )

        raise ValueError(
            "Sz labels differ between the NPZ "
            "dataset and metadata. "
            f"Maximum absolute difference: "
            f"{maximum_difference:.8e}"
        )

    if not np.allclose(
        metadata_lz,
        Y[
            :,
            1
        ],
        rtol=1.0e-12,
        atol=1.0e-12,
        equal_nan=False,
    ):
        maximum_difference = float(
            np.max(
                np.abs(
                    metadata_lz
                    - Y[
                        :,
                        1
                    ]
                )
            )
        )

        raise ValueError(
            "Lz labels differ between the NPZ "
            "dataset and metadata. "
            f"Maximum absolute difference: "
            f"{maximum_difference:.8e}"
        )

    return metadata_ordered


# ============================================================
# Split counts
# ============================================================

def calculate_subset_counts(
    sample_count,
    train_ratio,
    validation_ratio,
):
    """
    Calculate subset counts using the original notebook rule.

    n_train = int(train_ratio * N)
    n_validation = int(validation_ratio * N)
    n_test = N - n_train - n_validation
    """

    n_train = int(
        train_ratio
        * sample_count
    )

    n_validation = int(
        validation_ratio
        * sample_count
    )

    n_test = (
        sample_count
        - n_train
        - n_validation
    )

    if n_train <= 0:
        raise ValueError(
            "Training subset is empty"
        )

    if (
        validation_ratio > 0.0
        and n_validation <= 0
    ):
        raise ValueError(
            "Validation ratio is nonzero, but "
            "the resulting validation subset is empty"
        )

    if n_test <= 0:
        raise ValueError(
            "Test subset is empty"
        )

    return (
        n_train,
        n_validation,
        n_test,
    )


# ============================================================
# Split integrity
# ============================================================

def validate_split_indices(
    train_indices,
    validation_indices,
    test_indices,
    sample_count,
):
    """
    Validate complete, unique, non-overlapping dataset coverage.
    """

    subset_indices = {
        "train": np.asarray(
            train_indices,
            dtype=np.int64,
        ),

        "validation": np.asarray(
            validation_indices,
            dtype=np.int64,
        ),

        "test": np.asarray(
            test_indices,
            dtype=np.int64,
        ),
    }

    for subset_name, indices in (
        subset_indices.items()
    ):
        if indices.ndim != 1:
            raise RuntimeError(
                f"{subset_name} indices must "
                "be one-dimensional"
            )

        if np.any(
            indices < 0
        ):
            raise RuntimeError(
                f"{subset_name} indices contain "
                "negative values"
            )

        if np.any(
            indices >= sample_count
        ):
            raise RuntimeError(
                f"{subset_name} indices exceed "
                "the dataset range"
            )

        if len(
            np.unique(
                indices
            )
        ) != len(indices):
            raise RuntimeError(
                f"{subset_name} indices contain "
                "duplicates"
            )

    all_assigned_indices = np.concatenate(
        [
            subset_indices[
                "train"
            ],
            subset_indices[
                "validation"
            ],
            subset_indices[
                "test"
            ],
        ]
    )

    if len(
        all_assigned_indices
    ) != sample_count:
        raise RuntimeError(
            "Split does not contain exactly "
            "one assignment per dataset sample"
        )

    unique_assigned_indices = np.unique(
        all_assigned_indices
    )

    if len(
        unique_assigned_indices
    ) != sample_count:
        raise RuntimeError(
            "Split indices overlap or contain "
            "duplicate assignments"
        )

    expected_indices = np.arange(
        sample_count,
        dtype=np.int64,
    )

    if not np.array_equal(
        np.sort(
            unique_assigned_indices
        ),
        expected_indices,
    ):
        missing_indices = np.setdiff1d(
            expected_indices,
            unique_assigned_indices,
        )

        raise RuntimeError(
            "Split indices do not exactly cover "
            "the dataset. Missing indices: "
            f"{missing_indices[:20].tolist()}"
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

    normalized_config = validate_config(
        config
    )

    config_directory = (
        config_path.parent
    )

    repository_root = get_repository_root(
        config,
        config_path,
    )

    system_name = normalized_config[
        "system_name"
    ]

    train_ratio = normalized_config[
        "train_ratio"
    ]

    validation_ratio = normalized_config[
        "validation_ratio"
    ]

    test_ratio = normalized_config[
        "test_ratio"
    ]

    random_seed = normalized_config[
        "random_seed"
    ]

    paths = config[
        "paths"
    ]

    # --------------------------------------------------------
    # Resolve input and output paths
    # --------------------------------------------------------

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

    output_directory = ensure_directory(
        resolve_path(
            paths[
                "split_output_directory"
            ],
            config_directory,
        )
    )

    split_file = (
        output_directory
        / f"{system_name}_split.csv"
    )

    split_indices_file = (
        output_directory
        / (
            f"{system_name}_"
            "split_indices.npz"
        )
    )

    summary_file = (
        output_directory
        / (
            f"{system_name}_"
            "split_summary.json"
        )
    )

    output_files = [
        split_file,
        split_indices_file,
        summary_file,
    ]

    check_output_files(
        output_files,
        overwrite=(
            arguments.overwrite
        ),
        label="Split output files",
    )

    if not dataset_file.is_file():
        raise FileNotFoundError(
            "Prepared dataset was not found: "
            f"{safe_public_path(dataset_file, repository_root)}"
        )

    if not accepted_metadata_file.is_file():
        raise FileNotFoundError(
            "Accepted metadata was not found: "
            f"{safe_public_path(accepted_metadata_file, repository_root)}"
        )

    # --------------------------------------------------------
    # Load prepared NPZ dataset
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
            - set(
                dataset.files
            )
        )

        if missing_arrays:
            raise ValueError(
                "Prepared dataset is missing arrays: "
                f"{sorted(missing_arrays)}"
            )

        X = np.asarray(
            dataset[
                "X"
            ],
            dtype=np.float64,
        )

        Y = np.asarray(
            dataset[
                "Y"
            ],
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

    sample_count = validate_prepared_dataset(
        X,
        Y,
        sample_ids,
        prefixes,
    )

    # --------------------------------------------------------
    # Load and align accepted metadata
    # --------------------------------------------------------

    metadata_dataframe = pd.read_csv(
        accepted_metadata_file
    )

    if len(
        metadata_dataframe
    ) != sample_count:
        raise ValueError(
            "Accepted metadata count differs "
            "from prepared-dataset count: "
            f"{len(metadata_dataframe)} != "
            f"{sample_count}"
        )

    metadata_ordered = (
        align_and_validate_metadata(
            metadata_dataframe,
            sample_ids,
            prefixes,
            Y,
        )
    )

    # --------------------------------------------------------
    # Reproducible random split
    # --------------------------------------------------------

    random_generator = (
        np.random.default_rng(
            random_seed
        )
    )

    shuffled_indices = (
        random_generator.permutation(
            sample_count
        )
        .astype(
            np.int64,
            copy=False,
        )
    )

    (
        n_train,
        n_validation,
        n_test,
    ) = calculate_subset_counts(
        sample_count,
        train_ratio,
        validation_ratio,
    )

    train_indices = (
        shuffled_indices[
            :n_train
        ]
    )

    validation_indices = (
        shuffled_indices[
            n_train:
            n_train + n_validation
        ]
    )

    test_indices = (
        shuffled_indices[
            n_train + n_validation:
        ]
    )

    if len(
        test_indices
    ) != n_test:
        raise RuntimeError(
            "Calculated test count differs "
            "from the test-index array length"
        )

    validate_split_indices(
        train_indices,
        validation_indices,
        test_indices,
        sample_count,
    )

    # --------------------------------------------------------
    # Assign subset labels in original dataset order
    # --------------------------------------------------------

    subset_by_dataset_index = np.full(
        sample_count,
        "",
        dtype="U10",
    )

    subset_by_dataset_index[
        train_indices
    ] = "train"

    subset_by_dataset_index[
        validation_indices
    ] = "validation"

    subset_by_dataset_index[
        test_indices
    ] = "test"

    if np.any(
        subset_by_dataset_index
        == ""
    ):
        raise RuntimeError(
            "One or more dataset rows were "
            "not assigned to a subset"
        )

    observed_subsets = set(
        subset_by_dataset_index.tolist()
    )

    expected_subsets = {
        "train",
        "test",
    }

    if validation_ratio > 0.0:
        expected_subsets.add(
            "validation"
        )

    if observed_subsets != expected_subsets:
        raise RuntimeError(
            "Unexpected subset names after assignment: "
            f"{sorted(observed_subsets)}"
        )

    # --------------------------------------------------------
    # Human-readable split table
    # --------------------------------------------------------

    split_dataframe = (
        metadata_ordered.copy()
    )

    split_dataframe.insert(
        0,
        "dataset_index",
        np.arange(
            sample_count,
            dtype=np.int64,
        ),
    )

    split_dataframe.insert(
        1,
        "subset",
        subset_by_dataset_index,
    )

    split_dataframe.insert(
        2,
        "split_seed",
        random_seed,
    )

    preferred_columns = [
        "dataset_index",
        "sample_id",
        "subset",
        "split_seed",
        "prefix",
        "sz_label",
        "lz_label",
    ]

    remaining_columns = [
        column
        for column in (
            split_dataframe.columns
        )
        if column not in preferred_columns
    ]

    split_dataframe = (
        split_dataframe[
            preferred_columns
            + remaining_columns
        ]
    )

    split_dataframe.to_csv(
        split_file,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )

    # --------------------------------------------------------
    # Machine-readable split indices
    # --------------------------------------------------------

    np.savez_compressed(
        split_indices_file,

        idx_train=(
            train_indices
        ),

        idx_validation=(
            validation_indices
        ),

        idx_test=(
            test_indices
        ),

        sample_id_train=(
            sample_ids[
                train_indices
            ]
        ),

        sample_id_validation=(
            sample_ids[
                validation_indices
            ]
        ),

        sample_id_test=(
            sample_ids[
                test_indices
            ]
        ),

        prefix_train=(
            prefixes[
                train_indices
            ]
        ),

        prefix_validation=(
            prefixes[
                validation_indices
            ]
        ),

        prefix_test=(
            prefixes[
                test_indices
            ]
        ),

        split_seed=np.asarray(
            [
                random_seed
            ],
            dtype=np.int64,
        ),

        ratios=np.asarray(
            [
                train_ratio,
                validation_ratio,
                test_ratio,
            ],
            dtype=np.float64,
        ),

        subset_names=np.asarray(
            SUBSET_NAMES,
            dtype=str,
        ),
    )

    # --------------------------------------------------------
    # Explicit sample-ID overlap audit
    # --------------------------------------------------------

    train_ids = set(
        sample_ids[
            train_indices
        ].tolist()
    )

    validation_ids = set(
        sample_ids[
            validation_indices
        ].tolist()
    )

    test_ids = set(
        sample_ids[
            test_indices
        ].tolist()
    )

    cross_subset_overlap = {
        "train_validation": int(
            len(
                train_ids
                & validation_ids
            )
        ),

        "train_test": int(
            len(
                train_ids
                & test_ids
            )
        ),

        "validation_test": int(
            len(
                validation_ids
                & test_ids
            )
        ),
    }

    if any(
        cross_subset_overlap.values()
    ):
        raise RuntimeError(
            "Cross-subset sample-ID overlap "
            "was detected: "
            f"{cross_subset_overlap}"
        )

    # --------------------------------------------------------
    # Verify saved split files
    # --------------------------------------------------------

    saved_split_dataframe = pd.read_csv(
        split_file
    )

    if len(
        saved_split_dataframe
    ) != sample_count:
        raise RuntimeError(
            "Saved split CSV row count differs "
            "from the dataset sample count"
        )

    if saved_split_dataframe[
        "sample_id"
    ].duplicated().any():
        raise RuntimeError(
            "Saved split CSV contains duplicate "
            "sample_id values"
        )

    with np.load(
        split_indices_file,
        allow_pickle=False,
    ) as saved_indices:
        validate_split_indices(
            saved_indices[
                "idx_train"
            ],
            saved_indices[
                "idx_validation"
            ],
            saved_indices[
                "idx_test"
            ],
            sample_count,
        )

    # --------------------------------------------------------
    # Split summary
    # --------------------------------------------------------

    summary = {
        "created_at_utc": (
            utc_now_iso()
        ),

        "system": (
            config[
                "system"
            ]
        ),

        "config_file": (
            safe_public_path(
                config_path,
                repository_root,
            )
        ),

        "config_sha256": (
            sha256_file(
                config_path
            )
        ),

        "prepared_dataset_npz": (
            safe_public_path(
                dataset_file,
                repository_root,
            )
        ),

        "prepared_dataset_sha256": (
            sha256_file(
                dataset_file
            )
        ),

        "accepted_metadata_csv": (
            safe_public_path(
                accepted_metadata_file,
                repository_root,
            )
        ),

        "accepted_metadata_sha256": (
            sha256_file(
                accepted_metadata_file
            )
        ),

        "split_method": (
            "random permutation without replacement"
        ),

        "random_number_generator": (
            "numpy.random.default_rng"
        ),

        "random_seed": (
            random_seed
        ),

        "ratios": {
            "train": (
                train_ratio
            ),

            "validation": (
                validation_ratio
            ),

            "test": (
                test_ratio
            ),
        },

        "counts": {
            "total": int(
                sample_count
            ),

            "train": int(
                n_train
            ),

            "validation": int(
                n_validation
            ),

            "test": int(
                n_test
            ),
        },

        "rounding_rule": {
            "train": (
                "int(train_ratio * N)"
            ),

            "validation": (
                "int(validation_ratio * N)"
            ),

            "test": (
                "N - n_train - n_validation"
            ),
        },

        "integrity_checks": {
            "complete_dataset_coverage": True,

            "unique_dataset_indices": True,

            "unique_sample_ids": True,

            "cross_subset_sample_id_overlap": (
                cross_subset_overlap
            ),

            "duplicate_sample_id_rows": int(
                split_dataframe[
                    "sample_id"
                ]
                .duplicated()
                .sum()
            ),

            "duplicate_prefix_rows": int(
                split_dataframe[
                    "prefix"
                ]
                .duplicated(
                    keep=False
                )
                .sum()
            ),
        },

        "subsets": (
            summarize_subsets(
                split_dataframe,
                [
                    "sz_label",
                    "lz_label",
                ],
            )
        ),

        "output_files": {
            "split_csv": (
                safe_public_path(
                    split_file,
                    repository_root,
                )
            ),

            "split_indices_npz": (
                safe_public_path(
                    split_indices_file,
                    repository_root,
                )
            ),

            "summary_json": (
                safe_public_path(
                    summary_file,
                    repository_root,
                )
            ),
        },

        "output_sha256": {
            "split_csv": (
                sha256_file(
                    split_file
                )
            ),

            "split_indices_npz": (
                sha256_file(
                    split_indices_file
                )
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
    # Console report
    # --------------------------------------------------------

    print("=" * 80)

    print(
        f"System       : {system_name}"
    )

    print(
        "Dataset      : "
        f"{safe_public_path(dataset_file, repository_root)}"
    )

    print(
        f"Random seed  : {random_seed}"
    )

    print(
        f"Total        : {sample_count}"
    )

    print(
        "Train        : "
        f"{n_train}  "
        f"X={X[train_indices].shape}  "
        f"Y={Y[train_indices].shape}"
    )

    print(
        "Validation   : "
        f"{n_validation}  "
        f"X={X[validation_indices].shape}  "
        f"Y={Y[validation_indices].shape}"
    )

    print(
        "Test         : "
        f"{n_test}  "
        f"X={X[test_indices].shape}  "
        f"Y={Y[test_indices].shape}"
    )

    print(
        "Split CSV    : "
        f"{safe_public_path(split_file, repository_root)}"
    )

    print(
        "Indices NPZ  : "
        f"{safe_public_path(split_indices_file, repository_root)}"
    )

    print(
        "Summary JSON : "
        f"{safe_public_path(summary_file, repository_root)}"
    )

    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
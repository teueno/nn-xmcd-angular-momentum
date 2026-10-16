#!/usr/bin/env python3

"""
Fit preprocessing statistics on the training subset only.

The XAS and XMCD channels are standardized independently at each
energy point. The target variables Sz and Lz are also standardized
using statistics calculated from the training subset only.

The fitted transforms are frozen and applied unchanged to the
validation and test subsets.

Outputs
-------
<system>_preprocessed.npz
    Standardized and raw train/validation/test arrays, identifiers,
    fixed split indices, energy axes, and target names.

<system>_scalers.npz
    Training-subset means and standard deviations required to
    reproduce preprocessing and convert predictions back to
    physical units.

<system>_preprocess_summary.json
    Preprocessing method, sample counts, array summaries, file
    hashes, and software versions.

Privacy
-------
Internal calculations use absolute paths. Paths written to public
JSON files and displayed in normal console output are converted to
paths relative to the repository root.
"""

import argparse
from pathlib import Path

import numpy as np


# Support both:
#
#   python src/preprocess.py
#
# and:
#
#   python -m src.preprocess

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
# Command-line arguments
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Fit preprocessing statistics on the training "
            "subset only and transform all fixed subsets."
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
        help="Overwrite existing preprocessing outputs.",
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
    repository root. If a path is outside the repository, only
    its final file or directory name is returned.
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

def validate_positive_float(
    name,
    value,
):
    """
    Validate a finite positive floating-point setting.
    """

    if isinstance(
        value,
        bool,
    ):
        raise TypeError(
            f"{name} must be numeric, not bool"
        )

    try:
        numeric_value = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as exception:
        raise TypeError(
            f"{name} must be numeric"
        ) from exception

    if not np.isfinite(
        numeric_value
    ):
        raise ValueError(
            f"{name} must be finite"
        )

    if numeric_value <= 0.0:
        raise ValueError(
            f"{name} must be positive"
        )

    return numeric_value


def validate_boolean(
    name,
    value,
):
    """
    Validate a strict Boolean configuration setting.
    """

    if not isinstance(
        value,
        bool,
    ):
        raise TypeError(
            f"{name} must be true or false"
        )

    return value


def validate_config(
    config,
):
    """
    Validate and normalize preprocessing configuration.

    Returns
    -------
    dict
        Normalized preprocessing settings.
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
        "preprocessing",
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

    required_path_settings = {
        "repository_root",
        "prepared_dataset_npz",
        "split_indices_npz",
        "preprocess_output_directory",
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

    preprocessing = config[
        "preprocessing"
    ]

    if not isinstance(
        preprocessing,
        dict,
    ):
        raise TypeError(
            "preprocessing must be a dictionary"
        )

    input_mode = str(
        preprocessing.get(
            "input_mode",
            "channel_wise_feature_standardization",
        )
    )

    if (
        input_mode
        != "channel_wise_feature_standardization"
    ):
        raise ValueError(
            "Only "
            "'channel_wise_feature_standardization' "
            "is implemented"
        )

    fit_statistics_on = str(
        preprocessing.get(
            "fit_statistics_on",
            "training_only",
        )
    )

    if fit_statistics_on != "training_only":
        raise ValueError(
            "preprocessing.fit_statistics_on "
            "must be 'training_only'"
        )

    nonfinite_policy = str(
        preprocessing.get(
            "nonfinite_policy",
            "error",
        )
    )

    allowed_nonfinite_policies = {
        "error",
        "replace_with_zero",
    }

    if (
        nonfinite_policy
        not in allowed_nonfinite_policies
    ):
        raise ValueError(
            "preprocessing.nonfinite_policy must be "
            "'error' or 'replace_with_zero'"
        )

    epsilon = validate_positive_float(
        "preprocessing.epsilon",
        preprocessing.get(
            "epsilon",
            1.0e-6,
        ),
    )

    standardize_inputs = validate_boolean(
        "preprocessing.standardize_inputs",
        preprocessing.get(
            "standardize_inputs",
            True,
        ),
    )

    standardize_targets = validate_boolean(
        "preprocessing.standardize_targets",
        preprocessing.get(
            "standardize_targets",
            True,
        ),
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

    return {
        "system_name": (
            system_name
        ),

        "input_mode": (
            input_mode
        ),

        "fit_statistics_on": (
            fit_statistics_on
        ),

        "nonfinite_policy": (
            nonfinite_policy
        ),

        "epsilon": (
            epsilon
        ),

        "standardize_inputs": (
            standardize_inputs
        ),

        "standardize_targets": (
            standardize_targets
        ),
    }


# ============================================================
# Summary utilities
# ============================================================

def finite_summary(
    array,
):
    """
    Return a JSON-serializable numeric summary.

    The input array must contain only finite values.
    """

    values = np.asarray(
        array
    )

    ensure_finite(
        "Summary array",
        values,
    )

    if values.size == 0:
        raise ValueError(
            "Cannot summarize an empty array"
        )

    return {
        "shape": list(
            values.shape
        ),

        "dtype": str(
            values.dtype
        ),

        "nan_count": int(
            np.isnan(
                values
            ).sum()
        ),

        "positive_inf_count": int(
            np.isposinf(
                values
            ).sum()
        ),

        "negative_inf_count": int(
            np.isneginf(
                values
            ).sum()
        ),

        "min": float(
            np.min(
                values
            )
        ),

        "max": float(
            np.max(
                values
            )
        ),

        "mean": float(
            np.mean(
                values
            )
        ),

        "std_population": float(
            np.std(
                values,
                ddof=0,
            )
        ),
    }


def target_summary(
    array,
    names,
):
    """
    Return per-target numeric summaries.
    """

    values = np.asarray(
        array,
        dtype=np.float64,
    )

    target_names = [
        str(name)
        for name in names
    ]

    if values.ndim != 2:
        raise ValueError(
            "Target summary input must be "
            "two-dimensional"
        )

    if values.shape[1] != len(
        target_names
    ):
        raise ValueError(
            "Target-name count differs from "
            "the target-array dimension"
        )

    ensure_finite(
        "Target summary array",
        values,
    )

    summary = {}

    for target_index, target_name in enumerate(
        target_names
    ):
        target_values = values[
            :,
            target_index,
        ]

        summary[
            target_name
        ] = {
            "count": int(
                len(
                    target_values
                )
            ),

            "min": float(
                np.min(
                    target_values
                )
            ),

            "max": float(
                np.max(
                    target_values
                )
            ),

            "mean": float(
                np.mean(
                    target_values
                )
            ),

            "std_population": float(
                np.std(
                    target_values,
                    ddof=0,
                )
            ),
        }

    return summary


# ============================================================
# Index validation
# ============================================================

def validate_subset_indices(
    name,
    indices,
    sample_count,
):
    """
    Validate one split-index array.
    """

    values = np.asarray(
        indices
    )

    if values.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional"
        )

    if values.size == 0:
        raise ValueError(
            f"{name} is empty"
        )

    if not np.issubdtype(
        values.dtype,
        np.integer,
    ):
        raise TypeError(
            f"{name} must contain integers"
        )

    values = values.astype(
        np.int64,
        copy=False,
    )

    if np.any(
        values < 0
    ):
        raise ValueError(
            f"{name} contains negative indices"
        )

    if np.any(
        values >= sample_count
    ):
        raise ValueError(
            f"{name} contains indices outside "
            "the dataset range"
        )

    if len(
        np.unique(
            values
        )
    ) != len(values):
        raise ValueError(
            f"{name} contains duplicate indices"
        )

    return values


def validate_complete_split(
    idx_train,
    idx_validation,
    idx_test,
    sample_count,
):
    """
    Validate that fixed split arrays cover every sample exactly once.
    """

    idx_train = validate_subset_indices(
        "idx_train",
        idx_train,
        sample_count,
    )

    idx_validation = validate_subset_indices(
        "idx_validation",
        idx_validation,
        sample_count,
    )

    idx_test = validate_subset_indices(
        "idx_test",
        idx_test,
        sample_count,
    )

    all_indices = np.concatenate(
        [
            idx_train,
            idx_validation,
            idx_test,
        ]
    )

    if len(
        all_indices
    ) != sample_count:
        raise ValueError(
            "Split indices do not contain "
            "exactly one entry per dataset sample"
        )

    if len(
        np.unique(
            all_indices
        )
    ) != sample_count:
        raise ValueError(
            "Split indices overlap or contain "
            "duplicate assignments"
        )

    sorted_indices = np.sort(
        all_indices
    )

    expected_indices = np.arange(
        sample_count,
        dtype=np.int64,
    )

    if not np.array_equal(
        sorted_indices,
        expected_indices,
    ):
        missing_indices = np.setdiff1d(
            expected_indices,
            sorted_indices,
        )

        raise ValueError(
            "Split indices do not cover every "
            "dataset row. Missing indices: "
            f"{missing_indices[:20].tolist()}"
        )

    return (
        idx_train,
        idx_validation,
        idx_test,
    )


# ============================================================
# Transform utilities
# ============================================================

def fit_standardization(
    array,
    epsilon,
    enabled,
):
    """
    Fit feature-wise mean and population standard deviation.

    Parameters
    ----------
    array : numpy.ndarray
        Training data with shape (n_samples, n_features).

    epsilon : float
        Standard deviations below epsilon are replaced with 1.0
        in the effective transform.

    enabled : bool
        If False, identity-transform parameters are returned.

    Returns
    -------
    tuple
        mean, original_std, safe_std, small_std_mask
    """

    values = np.asarray(
        array,
        dtype=np.float64,
    )

    if values.ndim != 2:
        raise ValueError(
            "Standardization input must be "
            "two-dimensional"
        )

    if values.shape[0] == 0:
        raise ValueError(
            "Cannot fit standardization "
            "on an empty training subset"
        )

    ensure_finite(
        "Standardization input",
        values,
    )

    feature_count = values.shape[
        1
    ]

    if enabled:
        mean = np.mean(
            values,
            axis=0,
        )

        standard_deviation = np.std(
            values,
            axis=0,
            ddof=0,
        )

        small_std_mask = (
            standard_deviation
            < epsilon
        )

        safe_standard_deviation = np.where(
            small_std_mask,
            1.0,
            standard_deviation,
        )

    else:
        mean = np.zeros(
            feature_count,
            dtype=np.float64,
        )

        standard_deviation = np.ones(
            feature_count,
            dtype=np.float64,
        )

        safe_standard_deviation = np.ones(
            feature_count,
            dtype=np.float64,
        )

        small_std_mask = np.zeros(
            feature_count,
            dtype=bool,
        )

    ensure_finite(
        "Fitted mean",
        mean,
    )

    ensure_finite(
        "Fitted standard deviation",
        standard_deviation,
    )

    ensure_finite(
        "Safe standard deviation",
        safe_standard_deviation,
    )

    if np.any(
        safe_standard_deviation <= 0.0
    ):
        raise RuntimeError(
            "Safe standard deviation contains "
            "nonpositive values"
        )

    return (
        mean,
        standard_deviation,
        safe_standard_deviation,
        small_std_mask,
    )


def apply_standardization(
    array,
    mean,
    safe_standard_deviation,
):
    """
    Apply a previously fitted standardization transform.
    """

    values = np.asarray(
        array,
        dtype=np.float64,
    )

    transformed = (
        values
        - mean
    ) / safe_standard_deviation

    ensure_finite(
        "Transformed array",
        transformed,
    )

    return transformed


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

    paths = config[
        "paths"
    ]

    input_mode = normalized_config[
        "input_mode"
    ]

    fit_statistics_on = normalized_config[
        "fit_statistics_on"
    ]

    nonfinite_policy = normalized_config[
        "nonfinite_policy"
    ]

    epsilon = normalized_config[
        "epsilon"
    ]

    standardize_inputs = normalized_config[
        "standardize_inputs"
    ]

    standardize_targets = normalized_config[
        "standardize_targets"
    ]

    # --------------------------------------------------------
    # Resolve paths
    # --------------------------------------------------------

    dataset_file = resolve_path(
        paths[
            "prepared_dataset_npz"
        ],
        config_directory,
    )

    split_indices_file = resolve_path(
        paths[
            "split_indices_npz"
        ],
        config_directory,
    )

    output_directory = ensure_directory(
        resolve_path(
            paths[
                "preprocess_output_directory"
            ],
            config_directory,
        )
    )

    preprocessed_file = (
        output_directory
        / (
            f"{system_name}_"
            "preprocessed.npz"
        )
    )

    scaler_file = (
        output_directory
        / (
            f"{system_name}_"
            "scalers.npz"
        )
    )

    summary_file = (
        output_directory
        / (
            f"{system_name}_"
            "preprocess_summary.json"
        )
    )

    output_files = [
        preprocessed_file,
        scaler_file,
        summary_file,
    ]

    check_output_files(
        output_files,
        overwrite=(
            arguments.overwrite
        ),
        label="Preprocessing output files",
    )

    if not dataset_file.is_file():
        raise FileNotFoundError(
            "Prepared dataset was not found: "
            f"{safe_public_path(dataset_file, repository_root)}"
        )

    if not split_indices_file.is_file():
        raise FileNotFoundError(
            "Split indices were not found: "
            f"{safe_public_path(split_indices_file, repository_root)}"
        )

    # --------------------------------------------------------
    # Load prepared raw dataset
    # --------------------------------------------------------

    with np.load(
        dataset_file,
        allow_pickle=False,
    ) as dataset:
        required_dataset_arrays = {
            "X",
            "Y",
            "sample_id",
            "prefix",
            "energy_xas",
            "energy_xmcd",
        }

        missing_dataset_arrays = (
            required_dataset_arrays
            - set(
                dataset.files
            )
        )

        if missing_dataset_arrays:
            raise ValueError(
                "Prepared dataset is missing arrays: "
                f"{sorted(missing_dataset_arrays)}"
            )

        X_source = np.asarray(
            dataset[
                "X"
            ],
            dtype=np.float64,
        )

        Y_source = np.asarray(
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

        energy_xas = np.asarray(
            dataset[
                "energy_xas"
            ],
            dtype=np.float64,
        )

        energy_xmcd = np.asarray(
            dataset[
                "energy_xmcd"
            ],
            dtype=np.float64,
        )

        if (
            "target_names"
            in dataset.files
        ):
            target_names = (
                dataset[
                    "target_names"
                ]
                .astype(str)
            )

        else:
            target_names = np.asarray(
                [
                    "Sz",
                    "Lz",
                ],
                dtype=str,
            )

    # --------------------------------------------------------
    # Validate prepared dataset
    # --------------------------------------------------------

    if X_source.ndim != 2:
        raise ValueError(
            "X must be two-dimensional. "
            f"Found shape {X_source.shape}"
        )

    sample_count = int(
        X_source.shape[
            0
        ]
    )

    if sample_count == 0:
        raise ValueError(
            "Prepared dataset contains no samples"
        )

    if Y_source.shape != (
        sample_count,
        2,
    ):
        raise ValueError(
            "Expected Y shape "
            f"({sample_count}, 2), "
            f"found {Y_source.shape}"
        )

    if sample_ids.shape != (
        sample_count,
    ):
        raise ValueError(
            "sample_id count differs "
            "from X sample count"
        )

    if prefixes.shape != (
        sample_count,
    ):
        raise ValueError(
            "Prefix count differs "
            "from X sample count"
        )

    if len(
        set(
            sample_ids.tolist()
        )
    ) != sample_count:
        raise ValueError(
            "Prepared dataset contains duplicate "
            "sample_id values"
        )

    if len(
        set(
            prefixes.tolist()
        )
    ) != sample_count:
        raise ValueError(
            "Prepared dataset contains duplicate "
            "Prefix values"
        )

    if target_names.ndim != 1:
        raise ValueError(
            "target_names must be one-dimensional"
        )

    if target_names.tolist() != [
        "Sz",
        "Lz",
    ]:
        raise ValueError(
            "Target order must be ['Sz', 'Lz']. "
            f"Found {target_names.tolist()}"
        )

    if energy_xas.ndim != 1:
        raise ValueError(
            "energy_xas must be one-dimensional"
        )

    if energy_xmcd.ndim != 1:
        raise ValueError(
            "energy_xmcd must be one-dimensional"
        )

    if len(
        energy_xas
    ) == 0:
        raise ValueError(
            "energy_xas is empty"
        )

    if len(
        energy_xmcd
    ) == 0:
        raise ValueError(
            "energy_xmcd is empty"
        )

    ensure_finite(
        "energy_xas",
        energy_xas,
    )

    ensure_finite(
        "energy_xmcd",
        energy_xmcd,
    )

    n_xas = int(
        len(
            energy_xas
        )
    )

    n_xmcd = int(
        len(
            energy_xmcd
        )
    )

    expected_total_features = (
        n_xas
        + n_xmcd
    )

    if (
        X_source.shape[1]
        != expected_total_features
    ):
        raise ValueError(
            "Feature count does not equal "
            "n_xas + n_xmcd: "
            f"{X_source.shape[1]} != "
            f"{n_xas} + {n_xmcd}"
        )

    # --------------------------------------------------------
    # Load fixed split
    # --------------------------------------------------------

    with np.load(
        split_indices_file,
        allow_pickle=False,
    ) as split:
        required_split_arrays = {
            "idx_train",
            "idx_validation",
            "idx_test",
        }

        missing_split_arrays = (
            required_split_arrays
            - set(
                split.files
            )
        )

        if missing_split_arrays:
            raise ValueError(
                "Split file is missing arrays: "
                f"{sorted(missing_split_arrays)}"
            )

        idx_train = np.asarray(
            split[
                "idx_train"
            ],
            dtype=np.int64,
        )

        idx_validation = np.asarray(
            split[
                "idx_validation"
            ],
            dtype=np.int64,
        )

        idx_test = np.asarray(
            split[
                "idx_test"
            ],
            dtype=np.int64,
        )

        if (
            "split_seed"
            in split.files
        ):
            split_seed_array = np.asarray(
                split[
                    "split_seed"
                ]
            ).reshape(-1)

            if split_seed_array.size != 1:
                raise ValueError(
                    "split_seed must contain "
                    "exactly one value"
                )

            stored_split_seed = int(
                split_seed_array[
                    0
                ]
            )

        else:
            stored_split_seed = None

    (
        idx_train,
        idx_validation,
        idx_test,
    ) = validate_complete_split(
        idx_train,
        idx_validation,
        idx_test,
        sample_count,
    )

    # --------------------------------------------------------
    # Apply non-finite-value policy
    # --------------------------------------------------------

    source_input_nonfinite_count = int(
        (
            ~np.isfinite(
                X_source
            )
        ).sum()
    )

    source_target_nonfinite_count = int(
        (
            ~np.isfinite(
                Y_source
            )
        ).sum()
    )

    if nonfinite_policy == "error":
        ensure_finite(
            "X_source",
            X_source,
        )

        ensure_finite(
            "Y_source",
            Y_source,
        )

        X_work = X_source.copy()
        Y_work = Y_source.copy()

        replaced_input_count = 0
        replaced_target_count = 0

    else:
        replaced_input_count = (
            source_input_nonfinite_count
        )

        replaced_target_count = (
            source_target_nonfinite_count
        )

        X_work = np.nan_to_num(
            X_source,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        Y_work = np.nan_to_num(
            Y_source,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        ensure_finite(
            "X_work",
            X_work,
        )

        ensure_finite(
            "Y_work",
            Y_work,
        )

    # --------------------------------------------------------
    # Raw split arrays
    # --------------------------------------------------------

    X_train_raw = X_work[
        idx_train
    ].copy()

    X_validation_raw = X_work[
        idx_validation
    ].copy()

    X_test_raw = X_work[
        idx_test
    ].copy()

    Y_train_raw = Y_work[
        idx_train
    ].copy()

    Y_validation_raw = Y_work[
        idx_validation
    ].copy()

    Y_test_raw = Y_work[
        idx_test
    ].copy()

    # --------------------------------------------------------
    # Separate training XAS and XMCD channels
    # --------------------------------------------------------

    X_train_xas = X_train_raw[
        :,
        :n_xas,
    ]

    X_train_xmcd = X_train_raw[
        :,
        n_xas:
        n_xas + n_xmcd,
    ]

    # --------------------------------------------------------
    # Fit input statistics on the training subset only
    # --------------------------------------------------------

    (
        mean_xas,
        std_xas,
        std_xas_safe,
        small_std_xas_mask,
    ) = fit_standardization(
        X_train_xas,
        epsilon,
        standardize_inputs,
    )

    (
        mean_xmcd,
        std_xmcd,
        std_xmcd_safe,
        small_std_xmcd_mask,
    ) = fit_standardization(
        X_train_xmcd,
        epsilon,
        standardize_inputs,
    )

    # --------------------------------------------------------
    # Frozen input transform
    # --------------------------------------------------------

    def transform_inputs(
        raw_inputs,
    ):
        """
        Apply the training-fitted channel-wise transform.
        """

        values = np.asarray(
            raw_inputs,
            dtype=np.float64,
        )

        if values.ndim != 2:
            raise ValueError(
                "Input transform expects "
                "a two-dimensional array"
            )

        if values.shape[1] != (
            n_xas
            + n_xmcd
        ):
            raise ValueError(
                "Input transform feature count mismatch"
            )

        channel_xas = values[
            :,
            :n_xas,
        ]

        channel_xmcd = values[
            :,
            n_xas:
            n_xas + n_xmcd,
        ]

        normalized_xas = (
            channel_xas
            - mean_xas
        ) / std_xas_safe

        normalized_xmcd = (
            channel_xmcd
            - mean_xmcd
        ) / std_xmcd_safe

        transformed = np.concatenate(
            [
                normalized_xas,
                normalized_xmcd,
            ],
            axis=1,
        )

        ensure_finite(
            "Transformed inputs",
            transformed,
        )

        return transformed

    X_train = transform_inputs(
        X_train_raw
    )

    X_validation = transform_inputs(
        X_validation_raw
    )

    X_test = transform_inputs(
        X_test_raw
    )

    # --------------------------------------------------------
    # Fit target statistics on the training subset only
    # --------------------------------------------------------

    (
        Y_mean,
        Y_std,
        Y_std_safe,
        small_std_target_mask,
    ) = fit_standardization(
        Y_train_raw,
        epsilon,
        standardize_targets,
    )

    # --------------------------------------------------------
    # Frozen target transform
    # --------------------------------------------------------

    Y_train = apply_standardization(
        Y_train_raw,
        Y_mean,
        Y_std_safe,
    )

    Y_validation = apply_standardization(
        Y_validation_raw,
        Y_mean,
        Y_std_safe,
    )

    Y_test = apply_standardization(
        Y_test_raw,
        Y_mean,
        Y_std_safe,
    )

    # --------------------------------------------------------
    # Final safety checks
    # --------------------------------------------------------

    arrays_to_validate = (
        (
            "X_train",
            X_train,
        ),
        (
            "X_validation",
            X_validation,
        ),
        (
            "X_test",
            X_test,
        ),
        (
            "Y_train",
            Y_train,
        ),
        (
            "Y_validation",
            Y_validation,
        ),
        (
            "Y_test",
            Y_test,
        ),
        (
            "X_train_raw",
            X_train_raw,
        ),
        (
            "X_validation_raw",
            X_validation_raw,
        ),
        (
            "X_test_raw",
            X_test_raw,
        ),
        (
            "Y_train_raw",
            Y_train_raw,
        ),
        (
            "Y_validation_raw",
            Y_validation_raw,
        ),
        (
            "Y_test_raw",
            Y_test_raw,
        ),
    )

    for array_name, array in arrays_to_validate:
        ensure_finite(
            array_name,
            array,
        )

    expected_subset_shapes = {
        "train": (
            len(
                idx_train
            ),
            X_source.shape[
                1
            ],
        ),

        "validation": (
            len(
                idx_validation
            ),
            X_source.shape[
                1
            ],
        ),

        "test": (
            len(
                idx_test
            ),
            X_source.shape[
                1
            ],
        ),
    }

    if X_train.shape != expected_subset_shapes[
        "train"
    ]:
        raise RuntimeError(
            "Unexpected X_train shape"
        )

    if X_validation.shape != expected_subset_shapes[
        "validation"
    ]:
        raise RuntimeError(
            "Unexpected X_validation shape"
        )

    if X_test.shape != expected_subset_shapes[
        "test"
    ]:
        raise RuntimeError(
            "Unexpected X_test shape"
        )

    # Training standardized arrays should have means close to
    # zero except for features using the small-std safeguard.
    if standardize_inputs:
        nonconstant_feature_mask = np.concatenate(
            [
                ~small_std_xas_mask,
                ~small_std_xmcd_mask,
            ]
        )

        if np.any(
            nonconstant_feature_mask
        ):
            train_feature_means = np.mean(
                X_train,
                axis=0,
            )

            maximum_abs_train_mean = float(
                np.max(
                    np.abs(
                        train_feature_means[
                            nonconstant_feature_mask
                        ]
                    )
                )
            )

        else:
            maximum_abs_train_mean = 0.0

    else:
        maximum_abs_train_mean = None

    # --------------------------------------------------------
    # Save preprocessed arrays
    # --------------------------------------------------------

    np.savez_compressed(
        preprocessed_file,

        X_train=(
            X_train
        ),

        X_validation=(
            X_validation
        ),

        X_test=(
            X_test
        ),

        Y_train=(
            Y_train
        ),

        Y_validation=(
            Y_validation
        ),

        Y_test=(
            Y_test
        ),

        X_train_raw=(
            X_train_raw
        ),

        X_validation_raw=(
            X_validation_raw
        ),

        X_test_raw=(
            X_test_raw
        ),

        Y_train_raw=(
            Y_train_raw
        ),

        Y_validation_raw=(
            Y_validation_raw
        ),

        Y_test_raw=(
            Y_test_raw
        ),

        idx_train=(
            idx_train
        ),

        idx_validation=(
            idx_validation
        ),

        idx_test=(
            idx_test
        ),

        sample_id_train=(
            sample_ids[
                idx_train
            ]
        ),

        sample_id_validation=(
            sample_ids[
                idx_validation
            ]
        ),

        sample_id_test=(
            sample_ids[
                idx_test
            ]
        ),

        prefix_train=(
            prefixes[
                idx_train
            ]
        ),

        prefix_validation=(
            prefixes[
                idx_validation
            ]
        ),

        prefix_test=(
            prefixes[
                idx_test
            ]
        ),

        energy_xas=(
            energy_xas
        ),

        energy_xmcd=(
            energy_xmcd
        ),

        target_names=(
            target_names
        ),
    )

    # --------------------------------------------------------
    # Save fitted scaler parameters
    # --------------------------------------------------------

    np.savez_compressed(
        scaler_file,

        mean_xas=(
            mean_xas
        ),

        std_xas=(
            std_xas
        ),

        std_xas_safe=(
            std_xas_safe
        ),

        small_std_xas_mask=(
            small_std_xas_mask
        ),

        mean_xmcd=(
            mean_xmcd
        ),

        std_xmcd=(
            std_xmcd
        ),

        std_xmcd_safe=(
            std_xmcd_safe
        ),

        small_std_xmcd_mask=(
            small_std_xmcd_mask
        ),

        Y_mean=(
            Y_mean
        ),

        Y_std=(
            Y_std
        ),

        Y_std_safe=(
            Y_std_safe
        ),

        small_std_target_mask=(
            small_std_target_mask
        ),

        epsilon=np.asarray(
            [
                epsilon
            ],
            dtype=np.float64,
        ),

        n_xas=np.asarray(
            [
                n_xas
            ],
            dtype=np.int64,
        ),

        n_xmcd=np.asarray(
            [
                n_xmcd
            ],
            dtype=np.int64,
        ),

        standardize_inputs=np.asarray(
            [
                standardize_inputs
            ],
            dtype=bool,
        ),

        standardize_targets=np.asarray(
            [
                standardize_targets
            ],
            dtype=bool,
        ),

        target_names=(
            target_names
        ),

        fit_sample_ids=(
            sample_ids[
                idx_train
            ]
        ),
    )

    # --------------------------------------------------------
    # Preprocessing audit summary
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

        "split_indices_npz": (
            safe_public_path(
                split_indices_file,
                repository_root,
            )
        ),

        "split_indices_sha256": (
            sha256_file(
                split_indices_file
            )
        ),

        "method": {
            "fit_statistics_on": (
                fit_statistics_on
            ),

            "input_mode": (
                input_mode
            ),

            "feature_axis": (
                "energy point within each channel"
            ),

            "channel_order": (
                "XAS followed by XMCD"
            ),

            "target_order": (
                target_names.tolist()
            ),

            "standardize_inputs": (
                standardize_inputs
            ),

            "standardize_targets": (
                standardize_targets
            ),

            "population_standard_deviation_ddof": 0,

            "epsilon": (
                epsilon
            ),

            "small_std_replacement": 1.0,

            "nonfinite_policy": (
                nonfinite_policy
            ),

            "split_seed": (
                stored_split_seed
            ),
        },

        "counts": {
            "total": (
                sample_count
            ),

            "train": int(
                len(
                    idx_train
                )
            ),

            "validation": int(
                len(
                    idx_validation
                )
            ),

            "test": int(
                len(
                    idx_test
                )
            ),

            "n_xas_features": (
                n_xas
            ),

            "n_xmcd_features": (
                n_xmcd
            ),

            "n_total_features": int(
                X_source.shape[
                    1
                ]
            ),
        },

        "source_nonfinite_values": {
            "input_values": (
                source_input_nonfinite_count
            ),

            "target_values": (
                source_target_nonfinite_count
            ),
        },

        "nonfinite_replacements": {
            "input_values": (
                replaced_input_count
            ),

            "target_values": (
                replaced_target_count
            ),
        },

        "constant_or_near_constant_features": {
            "xas_below_epsilon": int(
                np.sum(
                    small_std_xas_mask
                )
            ),

            "xmcd_below_epsilon": int(
                np.sum(
                    small_std_xmcd_mask
                )
            ),

            "targets_below_epsilon": int(
                np.sum(
                    small_std_target_mask
                )
            ),
        },

        "training_transform_checks": {
            "maximum_absolute_training_feature_mean": (
                maximum_abs_train_mean
            ),

            "training_target_mean": (
                np.mean(
                    Y_train,
                    axis=0,
                ).tolist()
            ),

            "training_target_std_population": (
                np.std(
                    Y_train,
                    axis=0,
                    ddof=0,
                ).tolist()
            ),
        },

        "raw_target_summary": {
            "train": (
                target_summary(
                    Y_train_raw,
                    target_names,
                )
            ),

            "validation": (
                target_summary(
                    Y_validation_raw,
                    target_names,
                )
            ),

            "test": (
                target_summary(
                    Y_test_raw,
                    target_names,
                )
            ),
        },

        "standardized_target_summary": {
            "train": (
                target_summary(
                    Y_train,
                    target_names,
                )
            ),

            "validation": (
                target_summary(
                    Y_validation,
                    target_names,
                )
            ),

            "test": (
                target_summary(
                    Y_test,
                    target_names,
                )
            ),
        },

        "standardized_array_summary": {
            "X_train": (
                finite_summary(
                    X_train
                )
            ),

            "X_validation": (
                finite_summary(
                    X_validation
                )
            ),

            "X_test": (
                finite_summary(
                    X_test
                )
            ),

            "Y_train": (
                finite_summary(
                    Y_train
                )
            ),

            "Y_validation": (
                finite_summary(
                    Y_validation
                )
            ),

            "Y_test": (
                finite_summary(
                    Y_test
                )
            ),
        },

        "output_files": {
            "preprocessed_npz": (
                safe_public_path(
                    preprocessed_file,
                    repository_root,
                )
            ),

            "scalers_npz": (
                safe_public_path(
                    scaler_file,
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
            "preprocessed_npz": (
                sha256_file(
                    preprocessed_file
                )
            ),

            "scalers_npz": (
                sha256_file(
                    scaler_file
                )
            ),
        },

        "software": (
            software_versions()
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
        f"System             : {system_name}"
    )

    print(
        "Scaler fit subset  : training only"
    )

    print(
        f"Input mode         : {input_mode}"
    )

    print(
        f"Epsilon            : {epsilon}"
    )

    print(
        "Train X/Y          : "
        f"{X_train.shape} / "
        f"{Y_train.shape}"
    )

    print(
        "Validation X/Y     : "
        f"{X_validation.shape} / "
        f"{Y_validation.shape}"
    )

    print(
        "Test X/Y           : "
        f"{X_test.shape} / "
        f"{Y_test.shape}"
    )

    print(
        "Small-std XAS bins : "
        f"{int(np.sum(small_std_xas_mask))}"
    )

    print(
        "Small-std XMCD bins: "
        f"{int(np.sum(small_std_xmcd_mask))}"
    )

    print(
        "Small-std targets  : "
        f"{int(np.sum(small_std_target_mask))}"
    )

    print(
        "Train Y mean       : "
        f"{np.mean(Y_train, axis=0)}"
    )

    print(
        "Train Y std        : "
        f"{np.std(Y_train, axis=0, ddof=0)}"
    )

    print(
        "Preprocessed data  : "
        f"{safe_public_path(preprocessed_file, repository_root)}"
    )

    print(
        "Scalers            : "
        f"{safe_public_path(scaler_file, repository_root)}"
    )

    print(
        "Summary            : "
        f"{safe_public_path(summary_file, repository_root)}"
    )

    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
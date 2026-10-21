#!/usr/bin/env python3

"""Fit preprocessing statistics on training data only.

Expected neural-network inputs
------------------------------
XAS:
    (I_r + I_l) / 2

XMCD:
    I_r - I_l

The script validates the spectrum definitions and fixed external
magnetic field stored in the prepared dataset. The external field
is retained as metadata but is not appended to the neural-network
input vector.

XAS, XMCD, Sz, and Lz are standardized using statistics fitted
only on the training subset.
"""

import argparse
from pathlib import Path

import numpy as np

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

EXPECTED_XAS_DEFINITION = "(I_r + I_l) / 2"

EXPECTED_XMCD_DEFINITION = "I_r - I_l"

EXPECTED_EXTERNAL_FIELD_IS_MODEL_INPUT = False


# ============================================================
# Arguments
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Fit training-only preprocessing with spectrum-"
            "definition and external-field auditing."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# Basic utilities
# ============================================================

def safe_public_path(
    path,
    repository_root,
):
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


def validate_positive_float(
    name,
    value,
):
    if isinstance(
        value,
        bool,
    ):
        raise TypeError(
            f"{name} must be numeric, not bool"
        )

    try:
        value = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as exception:
        raise TypeError(
            f"{name} must be numeric"
        ) from exception

    if (
        not np.isfinite(
            value
        )
        or value <= 0.0
    ):
        raise ValueError(
            f"{name} must be finite and positive"
        )

    return value


def validate_nonnegative_float(
    name,
    value,
):
    if isinstance(
        value,
        bool,
    ):
        raise TypeError(
            f"{name} must be numeric, not bool"
        )

    try:
        value = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as exception:
        raise TypeError(
            f"{name} must be numeric"
        ) from exception

    if (
        not np.isfinite(
            value
        )
        or value < 0.0
    ):
        raise ValueError(
            f"{name} must be finite and nonnegative"
        )

    return value


def validate_boolean(
    name,
    value,
):
    if not isinstance(
        value,
        bool,
    ):
        raise TypeError(
            f"{name} must be true or false"
        )

    return value


def normalize_definition(
    value,
):
    return " ".join(
        str(
            value
        ).strip().split()
    )


def read_scalar_string(
    name,
    value,
):
    array = np.asarray(
        value
    ).reshape(
        -1
    )

    if array.size != 1:
        raise ValueError(
            f"{name} must contain exactly one value"
        )

    result = str(
        array[
            0
        ]
    ).strip()

    if not result:
        raise ValueError(
            f"{name} must not be empty"
        )

    return result


def read_scalar_boolean(
    name,
    value,
):
    array = np.asarray(
        value
    ).reshape(
        -1
    )

    if array.size != 1:
        raise ValueError(
            f"{name} must contain exactly one value"
        )

    return bool(
        array[
            0
        ]
    )


def read_scalar_integer(
    name,
    value,
):
    array = np.asarray(
        value
    ).reshape(
        -1
    )

    if array.size != 1:
        raise ValueError(
            f"{name} must contain exactly one value"
        )

    result = int(
        array[
            0
        ]
    )

    if result <= 0:
        raise ValueError(
            f"{name} must be positive"
        )

    return result


# ============================================================
# Configuration validation
# ============================================================

def validate_config(
    config,
):
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
        "spectrum",
        "preprocessing",
        "external_magnetic_field",
    }

    missing_sections = (
        required_sections
        - set(
            config
        )
    )

    if missing_sections:
        raise ValueError(
            "Missing configuration sections: "
            f"{sorted(missing_sections)}"
        )

    for section_name in required_sections:
        if not isinstance(
            config[
                section_name
            ],
            dict,
        ):
            raise TypeError(
                f"{section_name} must be a dictionary"
            )

    # --------------------------------------------------------
    # System
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    required_paths = {
        "repository_root",
        "prepared_dataset_npz",
        "split_indices_npz",
        "preprocess_output_directory",
    }

    missing_paths = (
        required_paths
        - set(
            config[
                "paths"
            ]
        )
    )

    if missing_paths:
        raise ValueError(
            "Missing path settings: "
            f"{sorted(missing_paths)}"
        )

    # --------------------------------------------------------
    # Spectrum settings
    # --------------------------------------------------------

    required_spectrum_settings = {
        "xas_right_suffix",
        "xas_left_suffix",
        "xmcd_suffix",
    }

    missing_spectrum_settings = (
        required_spectrum_settings
        - set(
            config[
                "spectrum"
            ]
        )
    )

    if missing_spectrum_settings:
        raise ValueError(
            "Missing spectrum settings: "
            f"{sorted(missing_spectrum_settings)}"
        )

    xas_right_suffix = str(
        config[
            "spectrum"
        ][
            "xas_right_suffix"
        ]
    ).strip()

    xas_left_suffix = str(
        config[
            "spectrum"
        ][
            "xas_left_suffix"
        ]
    ).strip()

    xmcd_suffix = str(
        config[
            "spectrum"
        ][
            "xmcd_suffix"
        ]
    ).strip()

    suffixes = (
        xas_right_suffix,
        xas_left_suffix,
        xmcd_suffix,
    )

    if any(
        not suffix
        for suffix in suffixes
    ):
        raise ValueError(
            "Spectrum suffixes must not be empty"
        )

    if len(
        set(
            suffixes
        )
    ) != len(
        suffixes
    ):
        raise ValueError(
            "Right-XAS, left-XAS, and XMCD suffixes "
            "must be different"
        )

    # --------------------------------------------------------
    # Preprocessing settings
    # --------------------------------------------------------

    preprocessing = config[
        "preprocessing"
    ]

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
            "Only 'channel_wise_feature_standardization' "
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
            "preprocessing.fit_statistics_on must be "
            "'training_only'"
        )

    nonfinite_policy = str(
        preprocessing.get(
            "nonfinite_policy",
            "error",
        )
    )

    if nonfinite_policy not in {
        "error",
        "replace_with_zero",
    }:
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

    # --------------------------------------------------------
    # External-field settings
    # --------------------------------------------------------

    field = config[
        "external_magnetic_field"
    ]

    for key in (
        "enabled",
        "magnitude_t",
        "direction",
    ):
        if key not in field:
            raise ValueError(
                "Missing external_magnetic_field."
                f"{key}"
            )

    field_enabled = validate_boolean(
        "external_magnetic_field.enabled",
        field[
            "enabled"
        ],
    )

    field_magnitude_t = validate_nonnegative_float(
        "external_magnetic_field.magnitude_t",
        field[
            "magnitude_t"
        ],
    )

    field_tolerance_t = validate_nonnegative_float(
        "external_magnetic_field.tolerance_t",
        field.get(
            "tolerance_t",
            1.0e-9,
        ),
    )

    field_direction = np.asarray(
        field[
            "direction"
        ],
        dtype=np.float64,
    )

    if field_direction.shape != (
        3,
    ):
        raise ValueError(
            "external_magnetic_field.direction must "
            "contain three values"
        )

    ensure_finite(
        "External magnetic-field direction",
        field_direction,
    )

    direction_norm = float(
        np.linalg.norm(
            field_direction
        )
    )

    if direction_norm <= 0.0:
        raise ValueError(
            "external magnetic-field direction must "
            "be nonzero"
        )

    field_direction = (
        field_direction
        / direction_norm
    )

    if (
        not field_enabled
        and not np.isclose(
            field_magnitude_t,
            0.0,
            rtol=0.0,
            atol=field_tolerance_t,
        )
    ):
        raise ValueError(
            "External-field magnitude must be zero "
            "when enabled is false"
        )

    return {
        "system_name": system_name,
        "input_mode": input_mode,
        "fit_statistics_on": fit_statistics_on,
        "nonfinite_policy": nonfinite_policy,
        "epsilon": epsilon,
        "standardize_inputs": standardize_inputs,
        "standardize_targets": standardize_targets,
        "xas_right_suffix": xas_right_suffix,
        "xas_left_suffix": xas_left_suffix,
        "xmcd_suffix": xmcd_suffix,
        "xas_definition": EXPECTED_XAS_DEFINITION,
        "xmcd_definition": EXPECTED_XMCD_DEFINITION,
        "external_field_is_model_input": (
            EXPECTED_EXTERNAL_FIELD_IS_MODEL_INPUT
        ),
        "field_enabled": field_enabled,
        "field_magnitude_t": field_magnitude_t,
        "field_tolerance_t": field_tolerance_t,
        "field_direction": field_direction,
    }


# ============================================================
# Summary utilities
# ============================================================

def finite_summary(
    array,
):
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
    values = np.asarray(
        array,
        dtype=np.float64,
    )

    names = [
        str(
            name
        )
        for name in names
    ]

    if (
        values.ndim != 2
        or values.shape[
            1
        ] != len(
            names
        )
    ):
        raise ValueError(
            "Target array and target names are inconsistent"
        )

    ensure_finite(
        "Target summary array",
        values,
    )

    return {
        name: {
            "count": int(
                len(
                    values[
                        :,
                        index
                    ]
                )
            ),
            "min": float(
                np.min(
                    values[
                        :,
                        index
                    ]
                )
            ),
            "max": float(
                np.max(
                    values[
                        :,
                        index
                    ]
                )
            ),
            "mean": float(
                np.mean(
                    values[
                        :,
                        index
                    ]
                )
            ),
            "std_population": float(
                np.std(
                    values[
                        :,
                        index
                    ],
                    ddof=0,
                )
            ),
        }
        for index, name in enumerate(
            names
        )
    }


def field_summary(
    values,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    ensure_finite(
        "External-field summary values",
        values,
    )

    if values.size == 0:
        raise ValueError(
            "Cannot summarize an empty field array"
        )

    return {
        "count": int(
            values.size
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


# ============================================================
# Split validation
# ============================================================

def validate_subset_indices(
    name,
    indices,
    sample_count,
):
    values = np.asarray(
        indices
    )

    if (
        values.ndim != 1
        or values.size == 0
    ):
        raise ValueError(
            f"{name} must be a nonempty "
            "one-dimensional array"
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

    if (
        np.any(
            values < 0
        )
        or np.any(
            values >= sample_count
        )
    ):
        raise ValueError(
            f"{name} contains indices outside "
            "the dataset range"
        )

    if len(
        np.unique(
            values
        )
    ) != len(
        values
    ):
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
            "Split does not contain one assignment "
            "per sample"
        )

    if not np.array_equal(
        np.sort(
            np.unique(
                all_indices
            )
        ),
        np.arange(
            sample_count
        ),
    ):
        raise ValueError(
            "Split indices overlap or do not cover "
            "every sample"
        )

    return (
        idx_train,
        idx_validation,
        idx_test,
    )


# ============================================================
# Standardization
# ============================================================

def fit_standardization(
    array,
    epsilon,
    enabled,
):
    values = np.asarray(
        array,
        dtype=np.float64,
    )

    if (
        values.ndim != 2
        or values.shape[
            0
        ] == 0
    ):
        raise ValueError(
            "Standardization input must be nonempty "
            "and two-dimensional"
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
        safe_standard_deviation
        <= 0.0
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
    transformed = (
        np.asarray(
            array,
            dtype=np.float64,
        )
        - mean
    ) / safe_standard_deviation

    ensure_finite(
        "Transformed array",
        transformed,
    )

    return transformed


# ============================================================
# Metadata validation
# ============================================================

def validate_spectrum_metadata(
    xas_definition,
    xmcd_definition,
    field_is_model_input,
    normalized,
):
    observed_xas = normalize_definition(
        xas_definition
    )

    observed_xmcd = normalize_definition(
        xmcd_definition
    )

    expected_xas = normalize_definition(
        normalized[
            "xas_definition"
        ]
    )

    expected_xmcd = normalize_definition(
        normalized[
            "xmcd_definition"
        ]
    )

    if observed_xas != expected_xas:
        raise ValueError(
            "Prepared-dataset XAS definition differs "
            "from the expected definition: "
            f"{xas_definition!r}"
        )

    if observed_xmcd != expected_xmcd:
        raise ValueError(
            "Prepared-dataset XMCD definition differs "
            "from the expected definition: "
            f"{xmcd_definition!r}"
        )

    if (
        bool(
            field_is_model_input
        )
        != normalized[
            "external_field_is_model_input"
        ]
    ):
        raise ValueError(
            "Prepared-dataset external-field input state "
            "differs from the expected value"
        )


def normalize_direction(
    name,
    direction,
):
    direction = np.asarray(
        direction,
        dtype=np.float64,
    ).reshape(
        -1
    )

    if direction.shape != (
        3,
    ):
        raise ValueError(
            f"{name} must contain three values"
        )

    ensure_finite(
        name,
        direction,
    )

    norm = float(
        np.linalg.norm(
            direction
        )
    )

    if norm <= 0.0:
        raise ValueError(
            f"{name} must be nonzero"
        )

    return (
        direction
        / norm
    )


def validate_field_metadata(
    dataset_enabled,
    dataset_magnitude,
    dataset_direction,
    dataset_fields,
    split_enabled,
    split_magnitude,
    split_direction,
    split_fields,
    normalized,
):
    expected_enabled = normalized[
        "field_enabled"
    ]

    expected_magnitude = normalized[
        "field_magnitude_t"
    ]

    expected_direction = normalized[
        "field_direction"
    ]

    tolerance = normalized[
        "field_tolerance_t"
    ]

    if (
        dataset_enabled != expected_enabled
        or split_enabled != expected_enabled
    ):
        raise ValueError(
            "External-field enabled state is inconsistent"
        )

    for name, magnitude in (
        (
            "prepared dataset",
            dataset_magnitude,
        ),
        (
            "split indices",
            split_magnitude,
        ),
    ):
        if not np.isclose(
            magnitude,
            expected_magnitude,
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError(
                "External-field magnitude in "
                f"{name} differs from config"
            )

    normalized_dataset_direction = normalize_direction(
        "Prepared-dataset field direction",
        dataset_direction,
    )

    normalized_split_direction = normalize_direction(
        "Split field direction",
        split_direction,
    )

    if not np.allclose(
        normalized_dataset_direction,
        expected_direction,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            "External-field direction in prepared "
            "dataset differs from config"
        )

    if not np.allclose(
        normalized_split_direction,
        expected_direction,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            "External-field direction in split indices "
            "differs from config"
        )

    ensure_finite(
        "Prepared dataset B_ext_T",
        dataset_fields,
    )

    ensure_finite(
        "Split B_ext_T",
        split_fields,
    )

    if not np.allclose(
        dataset_fields,
        expected_magnitude,
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError(
            "Prepared dataset contains inconsistent "
            "B_ext_T values"
        )

    if not np.allclose(
        split_fields,
        dataset_fields,
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError(
            "Split B_ext_T values differ from "
            "prepared dataset"
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

    normalized = validate_config(
        config
    )

    config_directory = (
        config_path.parent
    )

    repository_root = get_repository_root(
        config,
        config_path,
    )

    system_name = normalized[
        "system_name"
    ]

    paths = config[
        "paths"
    ]

    # --------------------------------------------------------
    # Paths
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
        / f"{system_name}_preprocessed.npz"
    )

    scaler_file = (
        output_directory
        / f"{system_name}_scalers.npz"
    )

    summary_file = (
        output_directory
        / f"{system_name}_preprocess_summary.json"
    )

    check_output_files(
        [
            preprocessed_file,
            scaler_file,
            summary_file,
        ],
        overwrite=arguments.overwrite,
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
            "energy_xas",
            "energy_xmcd",
            "b_ext_t",
            "external_field_enabled",
            "external_field_magnitude_t",
            "external_field_direction",
            "external_field_is_model_input",
            "xas_definition",
            "xmcd_definition",
            "n_xas",
            "n_xmcd",
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

        sample_ids = dataset[
            "sample_id"
        ].astype(
            str
        )

        prefixes = dataset[
            "prefix"
        ].astype(
            str
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

        b_ext_t = np.asarray(
            dataset[
                "b_ext_t"
            ],
            dtype=np.float64,
        )

        dataset_field_enabled = read_scalar_boolean(
            "external_field_enabled",
            dataset[
                "external_field_enabled"
            ],
        )

        dataset_field_magnitude = float(
            np.asarray(
                dataset[
                    "external_field_magnitude_t"
                ]
            ).reshape(
                -1
            )[
                0
            ]
        )

        dataset_field_direction = np.asarray(
            dataset[
                "external_field_direction"
            ],
            dtype=np.float64,
        )

        field_is_model_input = read_scalar_boolean(
            "external_field_is_model_input",
            dataset[
                "external_field_is_model_input"
            ],
        )

        xas_definition = read_scalar_string(
            "xas_definition",
            dataset[
                "xas_definition"
            ],
        )

        xmcd_definition = read_scalar_string(
            "xmcd_definition",
            dataset[
                "xmcd_definition"
            ],
        )

        stored_n_xas = read_scalar_integer(
            "n_xas",
            dataset[
                "n_xas"
            ],
        )

        stored_n_xmcd = read_scalar_integer(
            "n_xmcd",
            dataset[
                "n_xmcd"
            ],
        )

        target_names = (
            dataset[
                "target_names"
            ].astype(
                str
            )
            if "target_names" in dataset.files
            else np.asarray(
                [
                    "Sz",
                    "Lz",
                ],
                dtype=str,
            )
        )

    # --------------------------------------------------------
    # Validate prepared data
    # --------------------------------------------------------

    validate_spectrum_metadata(
        xas_definition=xas_definition,
        xmcd_definition=xmcd_definition,
        field_is_model_input=field_is_model_input,
        normalized=normalized,
    )

    if (
        X_source.ndim != 2
        or X_source.shape[
            0
        ] == 0
    ):
        raise ValueError(
            f"Unexpected X shape: {X_source.shape}"
        )

    sample_count = int(
        X_source.shape[
            0
        ]
    )

    if Y_source.shape != (
        sample_count,
        2,
    ):
        raise ValueError(
            f"Unexpected Y shape: {Y_source.shape}"
        )

    for name, values in (
        (
            "sample_id",
            sample_ids,
        ),
        (
            "prefix",
            prefixes,
        ),
        (
            "b_ext_t",
            b_ext_t,
        ),
    ):
        if values.shape != (
            sample_count,
        ):
            raise ValueError(
                f"{name} count differs from X sample count"
            )

    if len(
        set(
            sample_ids.tolist()
        )
    ) != sample_count:
        raise ValueError(
            "Prepared dataset contains duplicate sample IDs"
        )

    if len(
        set(
            prefixes.tolist()
        )
    ) != sample_count:
        raise ValueError(
            "Prepared dataset contains duplicate Prefix values"
        )

    if target_names.tolist() != [
        "Sz",
        "Lz",
    ]:
        raise ValueError(
            "Target order must be ['Sz', 'Lz']; found "
            f"{target_names.tolist()}"
        )

    if (
        energy_xas.ndim != 1
        or energy_xas.size == 0
    ):
        raise ValueError(
            "energy_xas must be nonempty and "
            "one-dimensional"
        )

    if (
        energy_xmcd.ndim != 1
        or energy_xmcd.size == 0
    ):
        raise ValueError(
            "energy_xmcd must be nonempty and "
            "one-dimensional"
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

    if n_xas != stored_n_xas:
        raise ValueError(
            "Stored n_xas differs from energy_xas length"
        )

    if n_xmcd != stored_n_xmcd:
        raise ValueError(
            "Stored n_xmcd differs from energy_xmcd length"
        )

    if X_source.shape[
        1
    ] != (
        n_xas
        + n_xmcd
    ):
        raise ValueError(
            "Feature count does not equal n_xas + n_xmcd"
        )

    # --------------------------------------------------------
    # Load split indices
    # --------------------------------------------------------

    with np.load(
        split_indices_file,
        allow_pickle=False,
    ) as split:
        required_split_arrays = {
            "idx_train",
            "idx_validation",
            "idx_test",
            "b_ext_t_train",
            "b_ext_t_validation",
            "b_ext_t_test",
            "external_field_enabled",
            "external_field_magnitude_t",
            "external_field_direction",
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

        split_field_enabled = read_scalar_boolean(
            "split.external_field_enabled",
            split[
                "external_field_enabled"
            ],
        )

        split_field_magnitude = float(
            np.asarray(
                split[
                    "external_field_magnitude_t"
                ]
            ).reshape(
                -1
            )[
                0
            ]
        )

        split_field_direction = np.asarray(
            split[
                "external_field_direction"
            ],
            dtype=np.float64,
        )

        b_ext_t_train_split = np.asarray(
            split[
                "b_ext_t_train"
            ],
            dtype=np.float64,
        )

        b_ext_t_validation_split = np.asarray(
            split[
                "b_ext_t_validation"
            ],
            dtype=np.float64,
        )

        b_ext_t_test_split = np.asarray(
            split[
                "b_ext_t_test"
            ],
            dtype=np.float64,
        )

        stored_split_seed = (
            int(
                np.asarray(
                    split[
                        "split_seed"
                    ]
                ).reshape(
                    -1
                )[
                    0
                ]
            )
            if "split_seed" in split.files
            else None
        )

    idx_train, idx_validation, idx_test = (
        validate_complete_split(
            idx_train,
            idx_validation,
            idx_test,
            sample_count,
        )
    )

    if b_ext_t_train_split.shape != (
        len(
            idx_train
        ),
    ):
        raise ValueError(
            "b_ext_t_train count differs from idx_train"
        )

    if b_ext_t_validation_split.shape != (
        len(
            idx_validation
        ),
    ):
        raise ValueError(
            "b_ext_t_validation count differs from "
            "idx_validation"
        )

    if b_ext_t_test_split.shape != (
        len(
            idx_test
        ),
    ):
        raise ValueError(
            "b_ext_t_test count differs from idx_test"
        )

    split_b_ext_t = np.empty(
        sample_count,
        dtype=np.float64,
    )

    split_b_ext_t[
        idx_train
    ] = b_ext_t_train_split

    split_b_ext_t[
        idx_validation
    ] = b_ext_t_validation_split

    split_b_ext_t[
        idx_test
    ] = b_ext_t_test_split

    validate_field_metadata(
        dataset_enabled=dataset_field_enabled,
        dataset_magnitude=dataset_field_magnitude,
        dataset_direction=dataset_field_direction,
        dataset_fields=b_ext_t,
        split_enabled=split_field_enabled,
        split_magnitude=split_field_magnitude,
        split_direction=split_field_direction,
        split_fields=split_b_ext_t,
        normalized=normalized,
    )

    # --------------------------------------------------------
    # Nonfinite-value policy
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

    if normalized[
        "nonfinite_policy"
    ] == "error":
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

    # --------------------------------------------------------
    # Extract raw subsets
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

    B_train = b_ext_t[
        idx_train
    ].copy()

    B_validation = b_ext_t[
        idx_validation
    ].copy()

    B_test = b_ext_t[
        idx_test
    ].copy()

    # --------------------------------------------------------
    # Fit XAS and XMCD scalers on training data only
    # --------------------------------------------------------

    (
        mean_xas,
        std_xas,
        std_xas_safe,
        small_std_xas_mask,
    ) = fit_standardization(
        X_train_raw[
            :,
            :n_xas
        ],
        normalized[
            "epsilon"
        ],
        normalized[
            "standardize_inputs"
        ],
    )

    (
        mean_xmcd,
        std_xmcd,
        std_xmcd_safe,
        small_std_xmcd_mask,
    ) = fit_standardization(
        X_train_raw[
            :,
            n_xas:
            n_xas + n_xmcd
        ],
        normalized[
            "epsilon"
        ],
        normalized[
            "standardize_inputs"
        ],
    )

    def transform_inputs(
        values,
    ):
        values = np.asarray(
            values,
            dtype=np.float64,
        )

        if (
            values.ndim != 2
            or values.shape[
                1
            ] != (
                n_xas
                + n_xmcd
            )
        ):
            raise ValueError(
                "Input transform feature count mismatch"
            )

        xas = (
            values[
                :,
                :n_xas
            ]
            - mean_xas
        ) / std_xas_safe

        xmcd = (
            values[
                :,
                n_xas:
                n_xas + n_xmcd
            ]
            - mean_xmcd
        ) / std_xmcd_safe

        transformed = np.concatenate(
            [
                xas,
                xmcd,
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
    # Fit target scaler on training data only
    # --------------------------------------------------------

    (
        Y_mean,
        Y_std,
        Y_std_safe,
        small_std_target_mask,
    ) = fit_standardization(
        Y_train_raw,
        normalized[
            "epsilon"
        ],
        normalized[
            "standardize_targets"
        ],
    )

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
    # Final finite checks
    # --------------------------------------------------------

    for name, values in (
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
        (
            "B_train",
            B_train,
        ),
        (
            "B_validation",
            B_validation,
        ),
        (
            "B_test",
            B_test,
        ),
    ):
        ensure_finite(
            name,
            values,
        )

    if normalized[
        "standardize_inputs"
    ]:
        nonconstant_mask = np.concatenate(
            [
                ~small_std_xas_mask,
                ~small_std_xmcd_mask,
            ]
        )

        maximum_abs_train_mean = (
            float(
                np.max(
                    np.abs(
                        np.mean(
                            X_train,
                            axis=0,
                        )[
                            nonconstant_mask
                        ]
                    )
                )
            )
            if np.any(
                nonconstant_mask
            )
            else 0.0
        )

    else:
        maximum_abs_train_mean = None

    # --------------------------------------------------------
    # Shared metadata payloads
    # --------------------------------------------------------

    field_payload = {
        "external_field_enabled": np.asarray(
            [
                normalized[
                    "field_enabled"
                ]
            ],
            dtype=bool,
        ),

        "external_field_magnitude_t": np.asarray(
            [
                normalized[
                    "field_magnitude_t"
                ]
            ],
            dtype=np.float64,
        ),

        "external_field_direction": np.asarray(
            normalized[
                "field_direction"
            ],
            dtype=np.float64,
        ),

        "external_field_is_model_input": np.asarray(
            [
                normalized[
                    "external_field_is_model_input"
                ]
            ],
            dtype=bool,
        ),
    }

    spectrum_payload = {
        "xas_definition": np.asarray(
            [
                xas_definition
            ],
            dtype=str,
        ),

        "xmcd_definition": np.asarray(
            [
                xmcd_definition
            ],
            dtype=str,
        ),

        "n_xas": np.asarray(
            [
                n_xas
            ],
            dtype=np.int64,
        ),

        "n_xmcd": np.asarray(
            [
                n_xmcd
            ],
            dtype=np.int64,
        ),
    }

    # --------------------------------------------------------
    # Save preprocessed data
    # --------------------------------------------------------

    np.savez_compressed(
        preprocessed_file,
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        Y_train=Y_train,
        Y_validation=Y_validation,
        Y_test=Y_test,
        X_train_raw=X_train_raw,
        X_validation_raw=X_validation_raw,
        X_test_raw=X_test_raw,
        Y_train_raw=Y_train_raw,
        Y_validation_raw=Y_validation_raw,
        Y_test_raw=Y_test_raw,
        b_ext_t_train=B_train,
        b_ext_t_validation=B_validation,
        b_ext_t_test=B_test,
        idx_train=idx_train,
        idx_validation=idx_validation,
        idx_test=idx_test,
        sample_id_train=sample_ids[
            idx_train
        ],
        sample_id_validation=sample_ids[
            idx_validation
        ],
        sample_id_test=sample_ids[
            idx_test
        ],
        prefix_train=prefixes[
            idx_train
        ],
        prefix_validation=prefixes[
            idx_validation
        ],
        prefix_test=prefixes[
            idx_test
        ],
        energy_xas=energy_xas,
        energy_xmcd=energy_xmcd,
        target_names=target_names,
        **field_payload,
        **spectrum_payload,
    )

    # --------------------------------------------------------
    # Save scalers
    # --------------------------------------------------------

    np.savez_compressed(
        scaler_file,
        mean_xas=mean_xas,
        std_xas=std_xas,
        std_xas_safe=std_xas_safe,
        small_std_xas_mask=small_std_xas_mask,
        mean_xmcd=mean_xmcd,
        std_xmcd=std_xmcd,
        std_xmcd_safe=std_xmcd_safe,
        small_std_xmcd_mask=small_std_xmcd_mask,
        Y_mean=Y_mean,
        Y_std=Y_std,
        Y_std_safe=Y_std_safe,
        small_std_target_mask=small_std_target_mask,
        epsilon=np.asarray(
            [
                normalized[
                    "epsilon"
                ]
            ],
            dtype=np.float64,
        ),
        standardize_inputs=np.asarray(
            [
                normalized[
                    "standardize_inputs"
                ]
            ],
            dtype=bool,
        ),
        standardize_targets=np.asarray(
            [
                normalized[
                    "standardize_targets"
                ]
            ],
            dtype=bool,
        ),
        target_names=target_names,
        fit_sample_ids=sample_ids[
            idx_train
        ],
        **field_payload,
        **spectrum_payload,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "created_at_utc": utc_now_iso(),

        "system": config[
            "system"
        ],

        "config_file": safe_public_path(
            config_path,
            repository_root,
        ),

        "config_sha256": sha256_file(
            config_path
        ),

        "prepared_dataset_npz": safe_public_path(
            dataset_file,
            repository_root,
        ),

        "prepared_dataset_sha256": sha256_file(
            dataset_file
        ),

        "split_indices_npz": safe_public_path(
            split_indices_file,
            repository_root,
        ),

        "split_indices_sha256": sha256_file(
            split_indices_file
        ),

        "spectrum_definitions": {
            "xas": xas_definition,

            "xmcd": xmcd_definition,

            "xas_right_suffix": normalized[
                "xas_right_suffix"
            ],

            "xas_left_suffix": normalized[
                "xas_left_suffix"
            ],

            "xmcd_suffix": normalized[
                "xmcd_suffix"
            ],

            "n_xas": n_xas,

            "n_xmcd": n_xmcd,

            "feature_order": (
                "[XAS channels, XMCD channels]"
            ),

            "validated_against_prepared_dataset": True,
        },

        "external_magnetic_field": {
            "enabled": normalized[
                "field_enabled"
            ],

            "magnitude_t": normalized[
                "field_magnitude_t"
            ],

            "direction": normalized[
                "field_direction"
            ].tolist(),

            "tolerance_t": normalized[
                "field_tolerance_t"
            ],

            "is_model_input": normalized[
                "external_field_is_model_input"
            ],

            "consistent_across_config_dataset_and_split": True,

            "train": field_summary(
                B_train
            ),

            "validation": field_summary(
                B_validation
            ),

            "test": field_summary(
                B_test
            ),
        },

        "method": {
            "fit_statistics_on": normalized[
                "fit_statistics_on"
            ],

            "input_mode": normalized[
                "input_mode"
            ],

            "feature_axis": (
                "energy point within each channel"
            ),

            "channel_order": (
                "XAS followed by XMCD"
            ),

            "target_order": target_names.tolist(),

            "standardize_inputs": normalized[
                "standardize_inputs"
            ],

            "standardize_targets": normalized[
                "standardize_targets"
            ],

            "population_standard_deviation_ddof": 0,

            "epsilon": normalized[
                "epsilon"
            ],

            "small_std_replacement": 1.0,

            "nonfinite_policy": normalized[
                "nonfinite_policy"
            ],

            "split_seed": stored_split_seed,
        },

        "counts": {
            "total": sample_count,

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

            "n_xas_features": n_xas,

            "n_xmcd_features": n_xmcd,

            "n_total_features": int(
                X_source.shape[
                    1
                ]
            ),
        },

        "source_nonfinite_values": {
            "input_values": source_input_nonfinite_count,

            "target_values": source_target_nonfinite_count,
        },

        "nonfinite_replacements": {
            "input_values": replaced_input_count,

            "target_values": replaced_target_count,
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

            "training_target_mean": np.mean(
                Y_train,
                axis=0,
            ).tolist(),

            "training_target_std_population": np.std(
                Y_train,
                axis=0,
                ddof=0,
            ).tolist(),
        },

        "raw_target_summary": {
            "train": target_summary(
                Y_train_raw,
                target_names,
            ),

            "validation": target_summary(
                Y_validation_raw,
                target_names,
            ),

            "test": target_summary(
                Y_test_raw,
                target_names,
            ),
        },

        "standardized_target_summary": {
            "train": target_summary(
                Y_train,
                target_names,
            ),

            "validation": target_summary(
                Y_validation,
                target_names,
            ),

            "test": target_summary(
                Y_test,
                target_names,
            ),
        },

        "standardized_array_summary": {
            "X_train": finite_summary(
                X_train
            ),

            "X_validation": finite_summary(
                X_validation
            ),

            "X_test": finite_summary(
                X_test
            ),

            "Y_train": finite_summary(
                Y_train
            ),

            "Y_validation": finite_summary(
                Y_validation
            ),

            "Y_test": finite_summary(
                Y_test
            ),
        },

        "output_files": {
            "preprocessed_npz": safe_public_path(
                preprocessed_file,
                repository_root,
            ),

            "scalers_npz": safe_public_path(
                scaler_file,
                repository_root,
            ),

            "summary_json": safe_public_path(
                summary_file,
                repository_root,
            ),
        },

        "output_sha256": {
            "preprocessed_npz": sha256_file(
                preprocessed_file
            ),

            "scalers_npz": sha256_file(
                scaler_file
            ),
        },

        "software": software_versions(),
    }

    write_json(
        summary_file,
        summary,
    )

    # --------------------------------------------------------
    # Console report
    # --------------------------------------------------------

    print(
        "=" * 80
    )

    print(
        f"System             : {system_name}"
    )

    print(
        "Scaler fit subset  : training only"
    )

    print(
        f"XAS definition     : {xas_definition}"
    )

    print(
        f"XMCD definition    : {xmcd_definition}"
    )

    print(
        "Field is NN input  : "
        f"{field_is_model_input}"
    )

    print(
        "External field     : "
        f"{normalized['field_magnitude_t']:.6f} T"
    )

    print(
        "Field direction    : "
        f"{normalized['field_direction'].tolist()}"
    )

    print(
        f"Input mode         : {normalized['input_mode']}"
    )

    print(
        f"Epsilon            : {normalized['epsilon']}"
    )

    print(
        f"Train X/Y          : "
        f"{X_train.shape} / {Y_train.shape}"
    )

    print(
        f"Validation X/Y     : "
        f"{X_validation.shape} / "
        f"{Y_validation.shape}"
    )

    print(
        f"Test X/Y           : "
        f"{X_test.shape} / {Y_test.shape}"
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
        f"Train Y mean       : "
        f"{np.mean(Y_train, axis=0)}"
    )

    print(
        f"Train Y std        : "
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

    print(
        "=" * 80
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
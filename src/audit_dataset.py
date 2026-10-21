#!/usr/bin/env python3

"""Audit an external-magnetic-field-aware prepared XAS/XMCD dataset.

Expected neural-network inputs
------------------------------
XAS:
    (I_r + I_l) / 2

XMCD:
    I_r - I_l

The external magnetic field is treated as a fixed calculation
condition. It is validated and recorded as metadata but is not
included in the neural-network input vector.

Checks
------
- XAS and XMCD definitions in config, NPZ, and accepted metadata,
- external-field model-input state,
- duplicate sample IDs and Prefix values,
- duplicate parameter configurations, including B_ext_T,
- exact and rounded-identical input spectra,
- duplicate inputs crossing train/validation/test boundaries,
- identical inputs associated with different labels,
- consistency of B_ext_T among config, NPZ, accepted metadata,
  and split CSV,
- consistency of external-field enabled state, magnitude,
  and direction.
"""

import argparse
import csv
from collections import Counter
from collections import defaultdict
from numbers import Real
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
# Constants
# ============================================================

VALID_SUBSETS = {
    "train",
    "validation",
    "test",
}

EXPECTED_XAS_DEFINITION = (
    "(I_r + I_l) / 2"
)

EXPECTED_XMCD_DEFINITION = (
    "I_r - I_l"
)

EXPECTED_EXTERNAL_FIELD_IS_MODEL_INPUT = (
    False
)


# ============================================================
# Arguments
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Audit spectrum definitions, duplicate samples, "
            "split leakage, label consistency, and external "
            "magnetic-field consistency."
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
# Path handling
# ============================================================

def safe_public_path(
    path,
    repository_root,
):
    """Return a public-safe path without exposing local paths."""

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
# Basic validation
# ============================================================

def validate_nonnegative_float(
    name,
    value,
):
    """Validate a finite, nonnegative floating-point value."""

    if (
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            Real,
        )
    ):
        raise TypeError(
            f"{name} must be numeric"
        )

    value = float(
        value
    )

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


def validate_nonnegative_integer(
    name,
    value,
):
    """Validate a nonnegative integer."""

    if (
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            int,
        )
    ):
        raise TypeError(
            f"{name} must be an integer"
        )

    if value < 0:
        raise ValueError(
            f"{name} must be nonnegative"
        )

    return value


def normalize_definition(
    value,
):
    """Normalize whitespace in a spectrum-definition string."""

    return " ".join(
        str(
            value
        ).strip().split()
    )


def scalar_string_from_array(
    name,
    values,
):
    """Read one scalar string from an NPZ array."""

    array = np.asarray(
        values
    ).reshape(
        -1
    )

    if array.size != 1:
        raise ValueError(
            f"{name} must contain exactly one value"
        )

    value = str(
        array[
            0
        ]
    ).strip()

    if not value:
        raise ValueError(
            f"{name} must not be empty"
        )

    return value


def scalar_bool_from_array(
    name,
    values,
):
    """Read one scalar Boolean from an NPZ array."""

    array = np.asarray(
        values
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


# ============================================================
# Configuration validation
# ============================================================

def validate_config(
    config,
):
    """Validate and normalize the audit configuration."""

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
        "columns",
        "audit",
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
        "accepted_metadata_csv",
        "split_csv",
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
    # Spectrum definitions
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
    # Columns
    # --------------------------------------------------------

    required_columns = {
        "prefix",
        "sz",
        "lz",
        "b_ext_t",
    }

    missing_columns = (
        required_columns
        - set(
            config[
                "columns"
            ]
        )
    )

    if missing_columns:
        raise ValueError(
            "Missing column settings: "
            f"{sorted(missing_columns)}"
        )

    prefix_column = str(
        config[
            "columns"
        ][
            "prefix"
        ]
    ).strip()

    sz_column = str(
        config[
            "columns"
        ][
            "sz"
        ]
    ).strip()

    lz_column = str(
        config[
            "columns"
        ][
            "lz"
        ]
    ).strip()

    field_column = str(
        config[
            "columns"
        ][
            "b_ext_t"
        ]
    ).strip()

    column_names = (
        prefix_column,
        sz_column,
        lz_column,
        field_column,
    )

    if any(
        not column
        for column in column_names
    ):
        raise ValueError(
            "Column names must not be empty"
        )

    # --------------------------------------------------------
    # External field
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

    if not isinstance(
        field[
            "enabled"
        ],
        bool,
    ):
        raise TypeError(
            "external_magnetic_field.enabled must be "
            "true or false"
        )

    magnitude = validate_nonnegative_float(
        "external_magnetic_field.magnitude_t",
        field[
            "magnitude_t"
        ],
    )

    tolerance = validate_nonnegative_float(
        "external_magnetic_field.tolerance_t",
        field.get(
            "tolerance_t",
            1.0e-9,
        ),
    )

    direction = np.asarray(
        field[
            "direction"
        ],
        dtype=np.float64,
    )

    if direction.shape != (
        3,
    ):
        raise ValueError(
            "external_magnetic_field.direction must "
            "have three values"
        )

    ensure_finite(
        "External magnetic-field direction",
        direction,
    )

    direction_norm = float(
        np.linalg.norm(
            direction
        )
    )

    if direction_norm <= 0.0:
        raise ValueError(
            "external magnetic-field direction must "
            "be nonzero"
        )

    direction = (
        direction
        / direction_norm
    )

    if (
        not field[
            "enabled"
        ]
        and not np.isclose(
            magnitude,
            0.0,
            rtol=0.0,
            atol=tolerance,
        )
    ):
        raise ValueError(
            "External-field magnitude must be zero "
            "when the field is disabled"
        )

    # --------------------------------------------------------
    # Audit
    # --------------------------------------------------------

    audit = config[
        "audit"
    ]

    spectrum_decimals = validate_nonnegative_integer(
        "audit.spectrum_round_decimals",
        int(
            audit.get(
                "spectrum_round_decimals",
                12,
            )
        ),
    )

    parameter_decimals = validate_nonnegative_integer(
        "audit.parameter_round_decimals",
        int(
            audit.get(
                "parameter_round_decimals",
                12,
            )
        ),
    )

    label_tolerance = validate_nonnegative_float(
        "audit.label_tolerance",
        audit.get(
            "label_tolerance",
            1.0e-12,
        ),
    )

    return {
        "system_name": system_name,

        "xas_right_suffix": xas_right_suffix,

        "xas_left_suffix": xas_left_suffix,

        "xmcd_suffix": xmcd_suffix,

        "expected_xas_definition": (
            EXPECTED_XAS_DEFINITION
        ),

        "expected_xmcd_definition": (
            EXPECTED_XMCD_DEFINITION
        ),

        "external_field_is_model_input": (
            EXPECTED_EXTERNAL_FIELD_IS_MODEL_INPUT
        ),

        "field_enabled": field[
            "enabled"
        ],

        "field_magnitude_t": magnitude,

        "field_tolerance_t": tolerance,

        "field_direction": direction,

        "prefix_column": prefix_column,

        "sz_column": sz_column,

        "lz_column": lz_column,

        "field_column": field_column,

        "spectrum_round_decimals": spectrum_decimals,

        "parameter_round_decimals": parameter_decimals,

        "label_tolerance": label_tolerance,
    }


# ============================================================
# Duplicate grouping
# ============================================================

def grouped_rows(
    values,
):
    """Return groups containing two or more matching values."""

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
        if len(
            rows
        ) > 1
    }


# ============================================================
# Parameter handling
# ============================================================

def parameter_columns_from_config(
    config,
    metadata,
):
    """Determine the parameter columns used for duplicate checks."""

    explicit_columns = config.get(
        "audit",
        {},
    ).get(
        "parameter_columns"
    )

    if explicit_columns:
        missing_columns = [
            column
            for column in explicit_columns
            if column not in metadata.columns
        ]

        if missing_columns:
            raise ValueError(
                "Configured parameter columns are missing "
                "from accepted metadata: "
                f"{missing_columns}"
            )

        return list(
            explicit_columns
        )

    candidates = [
        "TEN_DQ",
        "SlaterScale",
        "SOC_2p",
        "SOC_3d",
        "H_ex",
        "B_ext_T",
        "10Dq",
        "Slater",
        "SOC2p",
        "SOC3d",
        "Hex",
        "Bext",
        "ten_dq",
        "slater",
        "soc_2p",
        "soc_3d",
        "h_ex",
        "b_ext_t",
        "parameter_ten_dq",
        "parameter_slater",
        "parameter_soc_2p",
        "parameter_soc_3d",
        "parameter_h_ex",
        "parameter_b_ext_t",
    ]

    return [
        column
        for column in candidates
        if column in metadata.columns
    ]


def parameter_signatures(
    metadata,
    columns,
    decimals,
):
    """Create rounded signatures for parameter configurations."""

    signatures = []

    for _, row in metadata.iterrows():
        values = []

        for column in columns:
            value = row[
                column
            ]

            if pd.isna(
                value
            ):
                values.append(
                    None
                )

            elif isinstance(
                value,
                (
                    float,
                    int,
                    np.floating,
                    np.integer,
                ),
            ):
                values.append(
                    round(
                        float(
                            value
                        ),
                        decimals,
                    )
                )

            else:
                values.append(
                    str(
                        value
                    )
                )

        signatures.append(
            tuple(
                values
            )
        )

    return signatures


# ============================================================
# Duplicate reports
# ============================================================

def duplicate_group_records(
    groups,
    metadata,
    duplicate_type,
):
    """Convert duplicate groups to CSV records."""

    records = []

    for group_index, (
        signature,
        row_indices,
    ) in enumerate(
        groups.items()
    ):
        group = metadata.iloc[
            row_indices
        ]

        subsets = sorted(
            set(
                group[
                    "subset"
                ].astype(
                    str
                )
            )
        )

        records.append(
            {
                "duplicate_type": duplicate_type,

                "group_index": group_index,

                "signature": str(
                    signature
                ),

                "row_count": len(
                    row_indices
                ),

                "cross_split": (
                    len(
                        subsets
                    )
                    > 1
                ),

                "subsets": ";".join(
                    subsets
                ),

                "sample_ids": ";".join(
                    group[
                        "sample_id"
                    ].astype(
                        str
                    ).tolist()
                ),

                "prefixes": ";".join(
                    group[
                        "prefix"
                    ].astype(
                        str
                    ).tolist()
                ),

                "b_ext_t_values": ";".join(
                    f"{value:.12g}"
                    for value in group[
                        "b_ext_t"
                    ].to_numpy(
                        float
                    )
                ),
            }
        )

    return records


def label_conflict_groups(
    groups,
    targets,
    tolerance,
):
    """Find duplicate inputs associated with different labels."""

    conflicts = {}

    for signature, rows in groups.items():
        labels = targets[
            rows
        ]

        if np.any(
            np.abs(
                labels
                - labels[
                    0
                ]
            )
            > tolerance
        ):
            conflicts[
                signature
            ] = rows

    return conflicts


def cross_split_count(
    groups,
    metadata,
):
    """Count duplicate groups crossing dataset subsets."""

    group_count = 0
    row_count = 0

    for rows in groups.values():
        subsets = set(
            metadata.iloc[
                rows
            ][
                "subset"
            ].astype(
                str
            )
        )

        if len(
            subsets
        ) > 1:
            group_count += 1
            row_count += len(
                rows
            )

    return (
        group_count,
        row_count,
    )


def duplicate_row_count(
    groups,
):
    """Count all rows belonging to duplicate groups."""

    return int(
        sum(
            len(
                rows
            )
            for rows in groups.values()
        )
    )


# ============================================================
# Metadata normalization
# ============================================================

def normalize_metadata_columns(
    metadata,
    normalized,
):
    """Create canonical metadata columns from configured aliases."""

    metadata = metadata.copy()

    aliases = {
        "prefix": normalized[
            "prefix_column"
        ],

        "sz_label": normalized[
            "sz_column"
        ],

        "lz_label": normalized[
            "lz_column"
        ],

        "b_ext_t": normalized[
            "field_column"
        ],
    }

    for target_column, source_column in aliases.items():
        if target_column not in metadata.columns:
            if source_column not in metadata.columns:
                raise ValueError(
                    "Accepted metadata lacks both "
                    f"{target_column!r} and source column "
                    f"{source_column!r}"
                )

            metadata[
                target_column
            ] = metadata[
                source_column
            ]

    return metadata


# ============================================================
# Spectrum-definition validation
# ============================================================

def validate_spectrum_definitions(
    npz_xas_definition,
    npz_xmcd_definition,
    npz_field_is_model_input,
    metadata,
    normalized,
):
    """Validate input definitions across config, NPZ, and metadata."""

    expected_xas = normalize_definition(
        normalized[
            "expected_xas_definition"
        ]
    )

    expected_xmcd = normalize_definition(
        normalized[
            "expected_xmcd_definition"
        ]
    )

    observed_xas = normalize_definition(
        npz_xas_definition
    )

    observed_xmcd = normalize_definition(
        npz_xmcd_definition
    )

    if observed_xas != expected_xas:
        raise ValueError(
            "NPZ XAS definition differs from expected "
            f"definition: {npz_xas_definition!r}"
        )

    if observed_xmcd != expected_xmcd:
        raise ValueError(
            "NPZ XMCD definition differs from expected "
            f"definition: {npz_xmcd_definition!r}"
        )

    if (
        bool(
            npz_field_is_model_input
        )
        != normalized[
            "external_field_is_model_input"
        ]
    ):
        raise ValueError(
            "NPZ external_field_is_model_input differs "
            "from the expected value"
        )

    required_definition_columns = {
        "xas_definition",
        "xmcd_definition",
    }

    missing_columns = (
        required_definition_columns
        - set(
            metadata.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Accepted metadata is missing spectrum-definition "
            f"columns: {sorted(missing_columns)}"
        )

    metadata_xas_definitions = {
        normalize_definition(
            value
        )
        for value in metadata[
            "xas_definition"
        ].dropna().astype(
            str
        )
    }

    metadata_xmcd_definitions = {
        normalize_definition(
            value
        )
        for value in metadata[
            "xmcd_definition"
        ].dropna().astype(
            str
        )
    }

    if metadata_xas_definitions != {
        expected_xas
    }:
        raise ValueError(
            "Accepted metadata contains unexpected XAS "
            "definitions: "
            f"{sorted(metadata_xas_definitions)}"
        )

    if metadata_xmcd_definitions != {
        expected_xmcd
    }:
        raise ValueError(
            "Accepted metadata contains unexpected XMCD "
            "definitions: "
            f"{sorted(metadata_xmcd_definitions)}"
        )


# ============================================================
# External-field validation
# ============================================================

def validate_external_field(
    npz_enabled,
    npz_magnitude,
    npz_direction,
    b_ext_t,
    metadata_field,
    split_field,
    normalized,
):
    """Validate external-field consistency across all inputs."""

    tolerance = normalized[
        "field_tolerance_t"
    ]

    expected = normalized[
        "field_magnitude_t"
    ]

    expected_direction = normalized[
        "field_direction"
    ]

    if (
        bool(
            npz_enabled
        )
        != normalized[
            "field_enabled"
        ]
    ):
        raise ValueError(
            "NPZ external_field_enabled differs from config"
        )

    if not np.isclose(
        npz_magnitude,
        expected,
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError(
            "NPZ external-field magnitude differs from config"
        )

    npz_direction = np.asarray(
        npz_direction,
        dtype=np.float64,
    ).reshape(
        -1
    )

    ensure_finite(
        "NPZ external-field direction",
        npz_direction,
    )

    if npz_direction.shape != (
        3,
    ):
        raise ValueError(
            "NPZ external-field direction must have "
            "three values"
        )

    npz_direction_norm = float(
        np.linalg.norm(
            npz_direction
        )
    )

    if npz_direction_norm <= 0.0:
        raise ValueError(
            "NPZ external-field direction must be nonzero"
        )

    npz_direction = (
        npz_direction
        / npz_direction_norm
    )

    if not np.allclose(
        npz_direction,
        expected_direction,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            "NPZ external-field direction differs from config"
        )

    for name, values in (
        (
            "NPZ b_ext_t",
            b_ext_t,
        ),
        (
            "accepted metadata b_ext_t",
            metadata_field,
        ),
        (
            "split CSV b_ext_t",
            split_field,
        ),
    ):
        values = np.asarray(
            values,
            dtype=np.float64,
        )

        ensure_finite(
            name,
            values,
        )

        if not np.allclose(
            values,
            expected,
            rtol=0.0,
            atol=tolerance,
        ):
            maximum_difference = float(
                np.max(
                    np.abs(
                        values
                        - expected
                    )
                )
            )

            raise ValueError(
                f"{name} differs from configured field; "
                "maximum difference is "
                f"{maximum_difference:.8e} T"
            )

    if not np.allclose(
        metadata_field,
        b_ext_t,
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError(
            "External field differs between metadata and NPZ"
        )

    if not np.allclose(
        split_field,
        b_ext_t,
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError(
            "External field differs between split CSV and NPZ"
        )


# ============================================================
# Numerical summaries
# ============================================================

def numeric_summary(
    values,
):
    """Return JSON-safe numerical summary statistics."""

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    ensure_finite(
        "Summary values",
        values,
    )

    if values.size == 0:
        raise ValueError(
            "Cannot summarize an empty array"
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
    # Input and output paths
    # --------------------------------------------------------

    dataset_file = resolve_path(
        paths[
            "prepared_dataset_npz"
        ],
        config_directory,
    )

    metadata_file = resolve_path(
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
        / f"{system_name}_audit_summary.json"
    )

    duplicate_file = (
        audit_directory
        / f"{system_name}_duplicate_groups.csv"
    )

    check_output_files(
        [
            summary_file,
            duplicate_file,
        ],
        overwrite=arguments.overwrite,
        label="Audit outputs",
    )

    for input_file in (
        dataset_file,
        metadata_file,
        split_file,
    ):
        if not input_file.is_file():
            raise FileNotFoundError(
                "Required input file was not found: "
                f"{safe_public_path(input_file, repository_root)}"
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

        b_ext_t = np.asarray(
            dataset[
                "b_ext_t"
            ],
            dtype=np.float64,
        )

        npz_enabled = scalar_bool_from_array(
            "external_field_enabled",
            dataset[
                "external_field_enabled"
            ],
        )

        npz_magnitude = float(
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

        npz_direction = np.asarray(
            dataset[
                "external_field_direction"
            ],
            dtype=np.float64,
        )

        npz_field_is_model_input = scalar_bool_from_array(
            "external_field_is_model_input",
            dataset[
                "external_field_is_model_input"
            ],
        )

        npz_xas_definition = scalar_string_from_array(
            "xas_definition",
            dataset[
                "xas_definition"
            ],
        )

        npz_xmcd_definition = scalar_string_from_array(
            "xmcd_definition",
            dataset[
                "xmcd_definition"
            ],
        )

        n_xas = int(
            np.asarray(
                dataset[
                    "n_xas"
                ]
            ).reshape(
                -1
            )[
                0
            ]
        )

        n_xmcd = int(
            np.asarray(
                dataset[
                    "n_xmcd"
                ]
            ).reshape(
                -1
            )[
                0
            ]
        )

    # --------------------------------------------------------
    # Validate core arrays
    # --------------------------------------------------------

    ensure_finite(
        "X",
        X,
    )

    ensure_finite(
        "Y",
        Y,
    )

    ensure_finite(
        "b_ext_t",
        b_ext_t,
    )

    if X.ndim != 2:
        raise ValueError(
            f"X must be two-dimensional, got {X.shape}"
        )

    if Y.shape != (
        len(
            X
        ),
        2,
    ):
        raise ValueError(
            f"Unexpected target shape: Y={Y.shape}"
        )

    if n_xas <= 0:
        raise ValueError(
            "n_xas must be positive"
        )

    if n_xmcd <= 0:
        raise ValueError(
            "n_xmcd must be positive"
        )

    if X.shape[
        1
    ] != (
        n_xas
        + n_xmcd
    ):
        raise ValueError(
            "X feature count differs from n_xas + n_xmcd: "
            f"{X.shape[1]} != {n_xas} + {n_xmcd}"
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
            len(
                X
            ),
        ):
            raise ValueError(
                f"{name} count differs from dataset "
                "sample count"
            )

    if len(
        set(
            sample_ids.tolist()
        )
    ) != len(
        sample_ids
    ):
        raise ValueError(
            "Prepared dataset contains duplicate "
            "sample_id values"
        )

    if len(
        set(
            prefixes.tolist()
        )
    ) != len(
        prefixes
    ):
        raise ValueError(
            "Prepared dataset contains duplicate Prefix values"
        )

    # --------------------------------------------------------
    # Load metadata and split information
    # --------------------------------------------------------

    accepted_metadata = normalize_metadata_columns(
        pd.read_csv(
            metadata_file
        ),
        normalized,
    )

    split_dataframe = pd.read_csv(
        split_file
    )

    required_metadata_columns = {
        "sample_id",
        "prefix",
        "sz_label",
        "lz_label",
        "b_ext_t",
        "xas_definition",
        "xmcd_definition",
    }

    missing_metadata_columns = (
        required_metadata_columns
        - set(
            accepted_metadata.columns
        )
    )

    if missing_metadata_columns:
        raise ValueError(
            "Accepted metadata is missing columns: "
            f"{sorted(missing_metadata_columns)}"
        )

    required_split_columns = {
        "sample_id",
        "subset",
        "dataset_index",
        "b_ext_t",
    }

    missing_split_columns = (
        required_split_columns
        - set(
            split_dataframe.columns
        )
    )

    if missing_split_columns:
        raise ValueError(
            "Split CSV is missing columns: "
            f"{sorted(missing_split_columns)}"
        )

    if accepted_metadata[
        "sample_id"
    ].duplicated().any():
        raise ValueError(
            "Accepted metadata contains duplicate "
            "sample_id values"
        )

    if accepted_metadata[
        "prefix"
    ].astype(
        str
    ).duplicated().any():
        raise ValueError(
            "Accepted metadata contains duplicate Prefix values"
        )

    if split_dataframe[
        "sample_id"
    ].duplicated().any():
        raise ValueError(
            "Split CSV contains duplicate sample_id values"
        )

    # --------------------------------------------------------
    # Align metadata and split rows to NPZ order
    # --------------------------------------------------------

    accepted_by_id = accepted_metadata.assign(
        sample_id=accepted_metadata[
            "sample_id"
        ].astype(
            str
        )
    ).set_index(
        "sample_id",
        drop=False,
    )

    split_by_id = split_dataframe.assign(
        sample_id=split_dataframe[
            "sample_id"
        ].astype(
            str
        )
    ).set_index(
        "sample_id",
        drop=False,
    )

    dataset_ids = set(
        sample_ids.tolist()
    )

    if dataset_ids != set(
        accepted_by_id.index
    ):
        raise ValueError(
            "Prepared dataset and accepted metadata "
            "have different IDs"
        )

    if dataset_ids != set(
        split_by_id.index
    ):
        raise ValueError(
            "Prepared dataset and split CSV have different IDs"
        )

    metadata = (
        accepted_by_id.loc[
            sample_ids
        ]
        .reset_index(
            drop=True
        )
        .copy()
    )

    split_ordered = (
        split_by_id.loc[
            sample_ids
        ]
        .reset_index(
            drop=True
        )
        .copy()
    )

    metadata[
        "subset"
    ] = split_ordered[
        "subset"
    ].astype(
        str
    ).to_numpy()

    metadata[
        "dataset_index"
    ] = split_ordered[
        "dataset_index"
    ].to_numpy(
        int
    )

    observed_subsets = set(
        metadata[
            "subset"
        ]
    )

    invalid_subsets = (
        observed_subsets
        - VALID_SUBSETS
    )

    if invalid_subsets:
        raise ValueError(
            "Unknown subset names: "
            f"{sorted(invalid_subsets)}"
        )

    if not np.array_equal(
        metadata[
            "prefix"
        ].astype(
            str
        ).to_numpy(),
        prefixes,
    ):
        raise ValueError(
            "Prefix order differs between NPZ and metadata"
        )

    if not np.array_equal(
        metadata[
            "dataset_index"
        ].to_numpy(),
        np.arange(
            len(
                X
            )
        ),
    ):
        raise ValueError(
            "dataset_index does not match NPZ row order"
        )

    # --------------------------------------------------------
    # Validate labels
    # --------------------------------------------------------

    metadata_sz = pd.to_numeric(
        metadata[
            "sz_label"
        ],
        errors="coerce",
    ).to_numpy(
        float
    )

    metadata_lz = pd.to_numeric(
        metadata[
            "lz_label"
        ],
        errors="coerce",
    ).to_numpy(
        float
    )

    metadata_field = pd.to_numeric(
        metadata[
            "b_ext_t"
        ],
        errors="coerce",
    ).to_numpy(
        float
    )

    split_field = pd.to_numeric(
        split_ordered[
            "b_ext_t"
        ],
        errors="coerce",
    ).to_numpy(
        float
    )

    ensure_finite(
        "Metadata Sz",
        metadata_sz,
    )

    ensure_finite(
        "Metadata Lz",
        metadata_lz,
    )

    if not np.allclose(
        metadata_sz,
        Y[
            :,
            0,
        ],
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise ValueError(
            "Sz labels differ between NPZ and metadata"
        )

    if not np.allclose(
        metadata_lz,
        Y[
            :,
            1,
        ],
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise ValueError(
            "Lz labels differ between NPZ and metadata"
        )

    # --------------------------------------------------------
    # Validate spectrum definitions
    # --------------------------------------------------------

    validate_spectrum_definitions(
        npz_xas_definition=npz_xas_definition,
        npz_xmcd_definition=npz_xmcd_definition,
        npz_field_is_model_input=npz_field_is_model_input,
        metadata=metadata,
        normalized=normalized,
    )

    # --------------------------------------------------------
    # Validate external field
    # --------------------------------------------------------

    validate_external_field(
        npz_enabled=npz_enabled,
        npz_magnitude=npz_magnitude,
        npz_direction=npz_direction,
        b_ext_t=b_ext_t,
        metadata_field=metadata_field,
        split_field=split_field,
        normalized=normalized,
    )

    # --------------------------------------------------------
    # Duplicate spectrum hashes
    # --------------------------------------------------------

    exact_hashes = [
        sha256_array(
            input_vector
        )
        for input_vector in X
    ]

    rounded_hashes = [
        sha256_array(
            input_vector,
            decimals=normalized[
                "spectrum_round_decimals"
            ],
        )
        for input_vector in X
    ]

    exact_groups = grouped_rows(
        exact_hashes
    )

    rounded_groups = grouped_rows(
        rounded_hashes
    )

    # --------------------------------------------------------
    # Duplicate parameter configurations
    # --------------------------------------------------------

    parameter_columns = parameter_columns_from_config(
        config,
        metadata,
    )

    if parameter_columns:
        parameter_groups = grouped_rows(
            parameter_signatures(
                metadata,
                parameter_columns,
                normalized[
                    "parameter_round_decimals"
                ],
            )
        )

    else:
        parameter_groups = {}

    # --------------------------------------------------------
    # Label conflicts
    # --------------------------------------------------------

    exact_conflicts = label_conflict_groups(
        exact_groups,
        Y,
        normalized[
            "label_tolerance"
        ],
    )

    rounded_conflicts = label_conflict_groups(
        rounded_groups,
        Y,
        normalized[
            "label_tolerance"
        ],
    )

    # --------------------------------------------------------
    # Cross-split duplicates
    # --------------------------------------------------------

    (
        exact_cross_groups,
        exact_cross_rows,
    ) = cross_split_count(
        exact_groups,
        metadata,
    )

    (
        rounded_cross_groups,
        rounded_cross_rows,
    ) = cross_split_count(
        rounded_groups,
        metadata,
    )

    (
        parameter_cross_groups,
        parameter_cross_rows,
    ) = cross_split_count(
        parameter_groups,
        metadata,
    )

    # --------------------------------------------------------
    # Duplicate detail records
    # --------------------------------------------------------

    duplicate_records = []

    for duplicate_type, groups in (
        (
            "parameter_configuration",
            parameter_groups,
        ),
        (
            "exact_spectrum",
            exact_groups,
        ),
        (
            "rounded_spectrum",
            rounded_groups,
        ),
        (
            "exact_input_different_labels",
            exact_conflicts,
        ),
        (
            "rounded_input_different_labels",
            rounded_conflicts,
        ),
    ):
        duplicate_records.extend(
            duplicate_group_records(
                groups,
                metadata,
                duplicate_type,
            )
        )

    duplicate_fields = [
        "duplicate_type",
        "group_index",
        "signature",
        "row_count",
        "cross_split",
        "subsets",
        "sample_ids",
        "prefixes",
        "b_ext_t_values",
    ]

    with duplicate_file.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=duplicate_fields,
        )

        writer.writeheader()

        writer.writerows(
            duplicate_records
        )

    # --------------------------------------------------------
    # Audit status
    # --------------------------------------------------------

    failure_reasons = []

    if exact_cross_groups:
        failure_reasons.append(
            "exact_spectrum_duplicates_cross_split"
        )

    if rounded_cross_groups:
        failure_reasons.append(
            "rounded_spectrum_duplicates_cross_split"
        )

    if exact_conflicts:
        failure_reasons.append(
            "identical_inputs_with_different_labels"
        )

    if rounded_conflicts:
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
        ]
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

        "input_files": {
            "prepared_dataset_npz": safe_public_path(
                dataset_file,
                repository_root,
            ),

            "accepted_metadata_csv": safe_public_path(
                metadata_file,
                repository_root,
            ),

            "split_csv": safe_public_path(
                split_file,
                repository_root,
            ),
        },

        "input_sha256": {
            "prepared_dataset_npz": sha256_file(
                dataset_file
            ),

            "accepted_metadata_csv": sha256_file(
                metadata_file
            ),

            "split_csv": sha256_file(
                split_file
            ),
        },

        "spectrum_definitions": {
            "xas": npz_xas_definition,

            "xmcd": npz_xmcd_definition,

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

            "consistent_across_config_npz_and_metadata": (
                True
            ),
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

            "is_model_input": (
                npz_field_is_model_input
            ),

            "accepted_field_summary_t": numeric_summary(
                b_ext_t
            ),

            "consistent_across_config_npz_metadata_and_split": (
                True
            ),
        },

        "counts": {
            "samples": int(
                len(
                    X
                )
            ),

            "features": int(
                X.shape[
                    1
                ]
            ),

            "xas_features": int(
                n_xas
            ),

            "xmcd_features": int(
                n_xmcd
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
            "columns": parameter_columns,

            "round_decimals": normalized[
                "parameter_round_decimals"
            ],

            "duplicate_groups": int(
                len(
                    parameter_groups
                )
            ),

            "duplicate_rows": duplicate_row_count(
                parameter_groups
            ),

            "cross_split_groups": int(
                parameter_cross_groups
            ),

            "cross_split_rows": int(
                parameter_cross_rows
            ),
        },

        "spectrum_audit": {
            "hash_input": (
                "complete concatenated input vector "
                "[XAS, XMCD]"
            ),

            "exact_duplicate_groups": int(
                len(
                    exact_groups
                )
            ),

            "exact_duplicate_rows": duplicate_row_count(
                exact_groups
            ),

            "exact_cross_split_groups": int(
                exact_cross_groups
            ),

            "exact_cross_split_rows": int(
                exact_cross_rows
            ),

            "rounded_decimals": normalized[
                "spectrum_round_decimals"
            ],

            "rounded_duplicate_groups": int(
                len(
                    rounded_groups
                )
            ),

            "rounded_duplicate_rows": duplicate_row_count(
                rounded_groups
            ),

            "rounded_cross_split_groups": int(
                rounded_cross_groups
            ),

            "rounded_cross_split_rows": int(
                rounded_cross_rows
            ),
        },

        "label_consistency": {
            "tolerance": normalized[
                "label_tolerance"
            ],

            "exact_input_different_label_groups": int(
                len(
                    exact_conflicts
                )
            ),

            "rounded_input_different_label_groups": int(
                len(
                    rounded_conflicts
                )
            ),
        },

        "audit_status": audit_status,

        "failure_reasons": failure_reasons,

        "output_files": {
            "summary_json": safe_public_path(
                summary_file,
                repository_root,
            ),

            "duplicate_groups_csv": safe_public_path(
                duplicate_file,
                repository_root,
            ),
        },

        "software": software_versions(
            {
                "pandas": pd.__version__,
            }
        ),
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
        f"System                       : {system_name}"
    )

    print(
        f"Samples                      : {len(X)}"
    )

    print(
        f"Features                     : {X.shape[1]}"
    )

    print(
        f"XAS features                 : {n_xas}"
    )

    print(
        f"XMCD features                : {n_xmcd}"
    )

    print(
        "XAS definition               : "
        f"{npz_xas_definition}"
    )

    print(
        "XMCD definition              : "
        f"{npz_xmcd_definition}"
    )

    print(
        "External field is NN input   : "
        f"{npz_field_is_model_input}"
    )

    print(
        "External field               : "
        f"{normalized['field_magnitude_t']:.6f} T"
    )

    print(
        "Field direction              : "
        f"{normalized['field_direction'].tolist()}"
    )

    print(
        "Parameter duplicate groups   : "
        f"{len(parameter_groups)}"
    )

    print(
        "Parameter cross-split groups : "
        f"{parameter_cross_groups}"
    )

    print(
        "Exact spectrum duplicates    : "
        f"{len(exact_groups)}"
    )

    print(
        "Rounded spectrum duplicates  : "
        f"{len(rounded_groups)}"
    )

    print(
        "Exact cross-split groups      : "
        f"{exact_cross_groups}"
    )

    print(
        "Rounded cross-split groups    : "
        f"{rounded_cross_groups}"
    )

    print(
        "Exact input/label conflicts   : "
        f"{len(exact_conflicts)}"
    )

    print(
        "Rounded input/label conflicts : "
        f"{len(rounded_conflicts)}"
    )

    print(
        f"Audit status                 : {audit_status}"
    )

    print(
        "Summary                      : "
        f"{safe_public_path(summary_file, repository_root)}"
    )

    print(
        "Duplicate details            : "
        f"{safe_public_path(duplicate_file, repository_root)}"
    )

    print(
        "=" * 80
    )

    return (
        2
        if failure_reasons
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
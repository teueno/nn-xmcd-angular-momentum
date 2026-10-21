#!/usr/bin/env python3

"""Prepare a neural-network dataset from Quanty XAS/XMCD spectra.

This version validates and records a fixed external magnetic field.

Outputs
-------
<system>_dataset.npz
    X, Y, sample IDs, prefixes, energy axes, spectrum definitions,
    and external-field metadata.

<system>_metadata.csv
    Complete audit table containing all source samples.

<system>_accepted.csv
    Samples accepted for neural-network processing.

<system>_excluded.csv
    Excluded samples and their exclusion reasons.

<system>_summary.json
    Counts, label ranges, spectrum definitions, field conditions,
    hashes, and software versions.

Learning inputs
---------------
    XAS  = (I_r + I_l) / 2
    XMCD = I_r - I_l, read from *_cd.spec

The external magnetic field is validated as a fixed calculation
condition. It is recorded in the outputs but is not appended to
the neural-network input vector.
"""

import argparse
import csv
from collections import Counter
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


# ============================================================
# Arguments
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare an external-field-aware XAS/XMCD dataset "
            "using XAS = (I_r + I_l) / 2."
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
        help="Overwrite existing output files.",
    )

    return parser.parse_args()


# ============================================================
# Public path handling
# ============================================================

def safe_public_path(
    path,
    repository_root,
):
    """Return a public-safe repository-relative path."""

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
# Numeric validation
# ============================================================

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

    value = int(
        value
    )

    if value < 0:
        raise ValueError(
            f"{name} must be nonnegative"
        )

    return value


def validate_nonnegative_float(
    name,
    value,
):
    """Validate a finite, nonnegative numeric value."""

    if isinstance(
        value,
        bool,
    ):
        raise TypeError(
            f"{name} must be numeric, not bool"
        )

    try:
        result = float(
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
            result
        )
        or result < 0.0
    ):
        raise ValueError(
            f"{name} must be finite and nonnegative"
        )

    return result


# ============================================================
# Configuration validation
# ============================================================

def validate_config(
    config,
):
    """Validate and normalize the dataset configuration."""

    if not isinstance(
        config,
        dict,
    ):
        raise TypeError(
            "Configuration must be a dictionary"
        )

    required_sections = (
        "system",
        "paths",
        "spectrum",
        "columns",
        "exclusion",
        "external_magnetic_field",
    )

    missing_sections = [
        section
        for section in required_sections
        if section not in config
    ]

    if missing_sections:
        raise ValueError(
            "Missing configuration sections: "
            f"{missing_sections}"
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

    required_system_settings = (
        "name",
        "element",
        "n_electrons_3d",
    )

    missing_system_settings = [
        setting
        for setting in required_system_settings
        if setting not in config[
            "system"
        ]
    ]

    if missing_system_settings:
        raise ValueError(
            "Missing system settings: "
            f"{missing_system_settings}"
        )

    system_name = str(
        config[
            "system"
        ][
            "name"
        ]
    ).strip()

    element_name = str(
        config[
            "system"
        ][
            "element"
        ]
    ).strip()

    if not system_name:
        raise ValueError(
            "system.name must not be empty"
        )

    if not element_name:
        raise ValueError(
            "system.element must not be empty"
        )

    n_electrons_3d = validate_nonnegative_integer(
        "system.n_electrons_3d",
        config[
            "system"
        ][
            "n_electrons_3d"
        ],
    )

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    required_path_settings = (
        "repository_root",
        "spec_directory",
        "quanty_results_csv",
        "output_directory",
    )

    missing_path_settings = [
        setting
        for setting in required_path_settings
        if setting not in config[
            "paths"
        ]
    ]

    if missing_path_settings:
        raise ValueError(
            "Missing path settings: "
            f"{missing_path_settings}"
        )

    # --------------------------------------------------------
    # Spectrum
    # --------------------------------------------------------

    required_spectrum_settings = (
        "xas_right_suffix",
        "xas_left_suffix",
        "xmcd_suffix",
        "skip_header",
        "energy_column",
        "value_column",
    )

    missing_spectrum_settings = [
        setting
        for setting in required_spectrum_settings
        if setting not in config[
            "spectrum"
        ]
    ]

    if missing_spectrum_settings:
        raise ValueError(
            "Missing spectrum settings: "
            f"{missing_spectrum_settings}. "
            "XAS requires xas_right_suffix and "
            "xas_left_suffix."
        )

    right_suffix = str(
        config[
            "spectrum"
        ][
            "xas_right_suffix"
        ]
    ).strip()

    left_suffix = str(
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

    spectrum_suffixes = (
        right_suffix,
        left_suffix,
        xmcd_suffix,
    )

    if any(
        not suffix
        for suffix in spectrum_suffixes
    ):
        raise ValueError(
            "Spectrum suffixes must not be empty"
        )

    if len(
        set(
            spectrum_suffixes
        )
    ) != len(
        spectrum_suffixes
    ):
        raise ValueError(
            "Right-XAS, left-XAS, and XMCD suffixes "
            "must be different"
        )

    skip_header = validate_nonnegative_integer(
        "spectrum.skip_header",
        config[
            "spectrum"
        ][
            "skip_header"
        ],
    )

    energy_column = validate_nonnegative_integer(
        "spectrum.energy_column",
        config[
            "spectrum"
        ][
            "energy_column"
        ],
    )

    value_column = validate_nonnegative_integer(
        "spectrum.value_column",
        config[
            "spectrum"
        ][
            "value_column"
        ],
    )

    energy_rtol = validate_nonnegative_float(
        "spectrum.energy_rtol",
        config[
            "spectrum"
        ].get(
            "energy_rtol",
            1.0e-10,
        ),
    )

    energy_atol = validate_nonnegative_float(
        "spectrum.energy_atol",
        config[
            "spectrum"
        ].get(
            "energy_atol",
            1.0e-12,
        ),
    )

    require_equal_length = config[
        "spectrum"
    ].get(
        "require_equal_length",
        True,
    )

    require_same_energy_grid = config[
        "spectrum"
    ].get(
        "require_same_energy_grid",
        True,
    )

    if not isinstance(
        require_equal_length,
        bool,
    ):
        raise TypeError(
            "spectrum.require_equal_length "
            "must be true or false"
        )

    if not isinstance(
        require_same_energy_grid,
        bool,
    ):
        raise TypeError(
            "spectrum.require_same_energy_grid "
            "must be true or false"
        )

    # --------------------------------------------------------
    # Columns
    # --------------------------------------------------------

    required_column_settings = (
        "prefix",
        "sz",
        "lz",
        "b_ext_t",
    )

    missing_column_settings = [
        setting
        for setting in required_column_settings
        if setting not in config[
            "columns"
        ]
    ]

    if missing_column_settings:
        raise ValueError(
            "Missing column settings: "
            f"{missing_column_settings}"
        )

    column_names = [
        str(
            config[
                "columns"
            ][
                setting
            ]
        ).strip()
        for setting in required_column_settings
    ]

    if any(
        not column_name
        for column_name in column_names
    ):
        raise ValueError(
            "Column names must not be empty"
        )

    if len(
        set(
            column_names
        )
    ) != len(
        column_names
    ):
        raise ValueError(
            "Prefix, Sz, Lz, and B_ext_T columns "
            "must be different"
        )

    # --------------------------------------------------------
    # Exclusion
    # --------------------------------------------------------

    zero_tolerance = validate_nonnegative_float(
        "exclusion.zero_tolerance",
        config[
            "exclusion"
        ].get(
            "zero_tolerance",
            1.0e-6,
        ),
    )

    exclude_both_zero = config[
        "exclusion"
    ].get(
        "exclude_both_moments_zero",
        False,
    )

    if not isinstance(
        exclude_both_zero,
        bool,
    ):
        raise TypeError(
            "exclusion.exclude_both_moments_zero "
            "must be true or false"
        )

    # --------------------------------------------------------
    # External magnetic field
    # --------------------------------------------------------

    field = config[
        "external_magnetic_field"
    ]

    required_field_settings = (
        "enabled",
        "magnitude_t",
        "direction",
    )

    missing_field_settings = [
        setting
        for setting in required_field_settings
        if setting not in field
    ]

    if missing_field_settings:
        raise ValueError(
            "Missing external-field settings: "
            f"{missing_field_settings}"
        )

    if not isinstance(
        field[
            "enabled"
        ],
        bool,
    ):
        raise TypeError(
            "external_magnetic_field.enabled "
            "must be true or false"
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
            "external_magnetic_field.direction "
            "must contain exactly three values"
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
            "external_magnetic_field.direction "
            "must be nonzero"
        )

    field_direction = (
        field_direction
        / direction_norm
    )

    if (
        not field[
            "enabled"
        ]
        and not np.isclose(
            field_magnitude_t,
            0.0,
            rtol=0.0,
            atol=field_tolerance_t,
        )
    ):
        raise ValueError(
            "external_magnetic_field.magnitude_t "
            "must be zero when enabled is false"
        )

    return {
        "system_name": system_name,
        "element_name": element_name,
        "n_electrons_3d": n_electrons_3d,
        "xas_right_suffix": right_suffix,
        "xas_left_suffix": left_suffix,
        "xmcd_suffix": xmcd_suffix,
        "skip_header": skip_header,
        "energy_column": energy_column,
        "value_column": value_column,
        "energy_rtol": energy_rtol,
        "energy_atol": energy_atol,
        "require_equal_length": require_equal_length,
        "require_same_energy_grid": require_same_energy_grid,
        "exclude_both_moments_zero": exclude_both_zero,
        "zero_tolerance": zero_tolerance,
        "external_field_enabled": field[
            "enabled"
        ],
        "external_field_magnitude_t": field_magnitude_t,
        "external_field_tolerance_t": field_tolerance_t,
        "external_field_direction": field_direction.tolist(),
    }


# ============================================================
# Spectrum loading
# ============================================================

def load_spectrum(
    path,
    skip_header,
    energy_column,
    value_column,
):
    """Load one Quanty spectrum file."""

    path = Path(
        path
    )

    if not path.is_file():
        raise FileNotFoundError(
            "Spectrum file was not found: "
            f"{path.name}"
        )

    raw = np.genfromtxt(
        path,
        comments="#",
        skip_header=skip_header,
        dtype=np.float64,
    )

    if (
        raw.size == 0
        or raw.ndim == 0
    ):
        raise ValueError(
            "Empty or invalid spectrum: "
            f"{path.name}"
        )

    if raw.ndim == 1:
        raw = raw.reshape(
            1,
            -1,
        )

    required_column = max(
        energy_column,
        value_column,
    )

    if (
        raw.ndim != 2
        or raw.shape[
            1
        ] <= required_column
    ):
        raise ValueError(
            f"Requested column {required_column}; "
            f"spectrum shape is {raw.shape}"
        )

    finite_mask = np.isfinite(
        raw
    ).all(
        axis=1
    )

    cleaned = raw[
        finite_mask
    ]

    removed_rows = int(
        raw.shape[
            0
        ]
        - cleaned.shape[
            0
        ]
    )

    if cleaned.shape[
        0
    ] == 0:
        raise ValueError(
            "Spectrum contains no finite rows: "
            f"{path.name}"
        )

    energy = np.asarray(
        cleaned[
            :,
            energy_column,
        ],
        dtype=np.float64,
    )

    values = np.asarray(
        cleaned[
            :,
            value_column,
        ],
        dtype=np.float64,
    )

    ensure_finite(
        "Spectrum energy",
        energy,
    )

    ensure_finite(
        "Spectrum values",
        values,
    )

    if (
        energy.ndim != 1
        or values.ndim != 1
        or len(
            energy
        ) != len(
            values
        )
    ):
        raise ValueError(
            "Invalid energy/value arrays"
        )

    if np.any(
        np.diff(
            energy
        )
        <= 0.0
    ):
        raise ValueError(
            "Energy grid is not strictly increasing: "
            f"{path.name}"
        )

    return (
        energy,
        values,
        removed_rows,
    )


# ============================================================
# Spectrum validation
# ============================================================

def same_energy_grid(
    energy_a,
    energy_b,
    rtol,
    atol,
):
    """Return whether two energy grids match."""

    energy_a = np.asarray(
        energy_a,
        dtype=np.float64,
    )

    energy_b = np.asarray(
        energy_b,
        dtype=np.float64,
    )

    if energy_a.shape != energy_b.shape:
        return False

    return bool(
        np.allclose(
            energy_a,
            energy_b,
            rtol=rtol,
            atol=atol,
            equal_nan=False,
        )
    )


def add_reason(
    reasons,
    reason,
):
    """Append one unique exclusion reason."""

    reason = str(
        reason
    )

    if reason not in reasons:
        reasons.append(
            reason
        )


def exception_reason(
    exception,
):
    """Convert an exception to a compact audit-safe reason."""

    message = " ".join(
        str(
            exception
        ).split()
    ).replace(
        ";",
        ",",
    )

    if len(
        message
    ) > 200:
        message = (
            message[
                :200
            ]
            + "..."
        )

    return (
        "spectrum_read_error:"
        f"{type(exception).__name__}:"
        f"{message}"
    )


def numeric_summary(
    values,
):
    """Return JSON-safe numerical summary statistics."""

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    ensure_finite(
        "Summary array",
        array,
    )

    if array.size == 0:
        raise ValueError(
            "Cannot summarize an empty array"
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

    system = config[
        "system"
    ]

    paths = config[
        "paths"
    ]

    columns = config[
        "columns"
    ]

    system_name = normalized[
        "system_name"
    ]

    # --------------------------------------------------------
    # Resolve paths
    # --------------------------------------------------------

    spec_directory = resolve_path(
        paths[
            "spec_directory"
        ],
        config_directory,
    )

    results_csv = resolve_path(
        paths[
            "quanty_results_csv"
        ],
        config_directory,
    )

    output_directory = ensure_directory(
        resolve_path(
            paths[
                "output_directory"
            ],
            config_directory,
        )
    )

    if not spec_directory.is_dir():
        raise NotADirectoryError(
            "Spectrum directory was not found: "
            f"{safe_public_path(spec_directory, repository_root)}"
        )

    if not results_csv.is_file():
        raise FileNotFoundError(
            "Quanty results CSV was not found: "
            f"{safe_public_path(results_csv, repository_root)}"
        )

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    dataset_file = (
        output_directory
        / f"{system_name}_dataset.npz"
    )

    metadata_file = (
        output_directory
        / f"{system_name}_metadata.csv"
    )

    accepted_file = (
        output_directory
        / f"{system_name}_accepted.csv"
    )

    excluded_file = (
        output_directory
        / f"{system_name}_excluded.csv"
    )

    summary_file = (
        output_directory
        / f"{system_name}_summary.json"
    )

    check_output_files(
        [
            dataset_file,
            metadata_file,
            accepted_file,
            excluded_file,
            summary_file,
        ],
        overwrite=arguments.overwrite,
        label="Dataset output files",
    )

    # --------------------------------------------------------
    # Load source metadata
    # --------------------------------------------------------

    source = pd.read_csv(
        results_csv
    )

    if source.empty:
        raise ValueError(
            "Quanty results CSV is empty"
        )

    prefix_column = str(
        columns[
            "prefix"
        ]
    )

    sz_column = str(
        columns[
            "sz"
        ]
    )

    lz_column = str(
        columns[
            "lz"
        ]
    )

    b_ext_column = str(
        columns[
            "b_ext_t"
        ]
    )

    required_csv_columns = [
        prefix_column,
        sz_column,
        lz_column,
        b_ext_column,
    ]

    missing_csv_columns = [
        column
        for column in required_csv_columns
        if column not in source.columns
    ]

    if missing_csv_columns:
        raise ValueError(
            "Required CSV columns are missing: "
            f"{missing_csv_columns}"
        )

    # --------------------------------------------------------
    # Effective settings
    # --------------------------------------------------------

    right_suffix = normalized[
        "xas_right_suffix"
    ]

    left_suffix = normalized[
        "xas_left_suffix"
    ]

    xmcd_suffix = normalized[
        "xmcd_suffix"
    ]

    skip_header = normalized[
        "skip_header"
    ]

    energy_column = normalized[
        "energy_column"
    ]

    value_column = normalized[
        "value_column"
    ]

    energy_rtol = normalized[
        "energy_rtol"
    ]

    energy_atol = normalized[
        "energy_atol"
    ]

    require_equal_length = normalized[
        "require_equal_length"
    ]

    require_same_grid = normalized[
        "require_same_energy_grid"
    ]

    exclude_both_zero = normalized[
        "exclude_both_moments_zero"
    ]

    zero_tolerance = normalized[
        "zero_tolerance"
    ]

    field_enabled = normalized[
        "external_field_enabled"
    ]

    expected_field = normalized[
        "external_field_magnitude_t"
    ]

    field_tolerance = normalized[
        "external_field_tolerance_t"
    ]

    field_direction = normalized[
        "external_field_direction"
    ]

    # --------------------------------------------------------
    # Accumulators
    # --------------------------------------------------------

    records = []

    accepted_x = []
    accepted_y = []
    accepted_ids = []
    accepted_prefixes = []
    accepted_fields = []

    observed_prefixes = set()

    expected_feature_count = None

    reference_energy_xas = None
    reference_energy_xmcd = None

    # --------------------------------------------------------
    # Process every source row
    # --------------------------------------------------------

    for row_position, (
        source_index,
        row,
    ) in enumerate(
        source.iterrows()
    ):
        sample_id = (
            f"{system_name}_"
            f"{row_position:06d}"
        )

        raw_prefix = row[
            prefix_column
        ]

        prefix = (
            ""
            if pd.isna(
                raw_prefix
            )
            else str(
                raw_prefix
            ).strip()
        )

        sz_value = pd.to_numeric(
            pd.Series(
                [
                    row[
                        sz_column
                    ]
                ]
            ),
            errors="coerce",
        ).iloc[
            0
        ]

        lz_value = pd.to_numeric(
            pd.Series(
                [
                    row[
                        lz_column
                    ]
                ]
            ),
            errors="coerce",
        ).iloc[
            0
        ]

        b_ext_value = pd.to_numeric(
            pd.Series(
                [
                    row[
                        b_ext_column
                    ]
                ]
            ),
            errors="coerce",
        ).iloc[
            0
        ]

        right_file = (
            spec_directory
            / f"{prefix}{right_suffix}"
        )

        left_file = (
            spec_directory
            / f"{prefix}{left_suffix}"
        )

        xmcd_file = (
            spec_directory
            / f"{prefix}{xmcd_suffix}"
        )

        audit = row.to_dict()

        audit.update(
            {
                "sample_id": sample_id,
                "source_index": str(
                    source_index
                ),
                "source_row_position": int(
                    row_position
                ),
                "system": system_name,
                "element": normalized[
                    "element_name"
                ],
                "n_electrons_3d": int(
                    normalized[
                        "n_electrons_3d"
                    ]
                ),
                "prefix": prefix,
                "sz_label": sz_value,
                "lz_label": lz_value,
                "b_ext_t": b_ext_value,
                "expected_b_ext_t": expected_field,
                "external_field_direction_x": field_direction[
                    0
                ],
                "external_field_direction_y": field_direction[
                    1
                ],
                "external_field_direction_z": field_direction[
                    2
                ],
                "xas_right_file": safe_public_path(
                    right_file,
                    repository_root,
                ),
                "xas_left_file": safe_public_path(
                    left_file,
                    repository_root,
                ),
                "xmcd_file": safe_public_path(
                    xmcd_file,
                    repository_root,
                ),
                "xas_definition": "(I_r + I_l) / 2",
                "xmcd_definition": "I_r - I_l",
                "accepted": False,
                "exclusion_reason": "",
                "xas_right_rows": 0,
                "xas_left_rows": 0,
                "xas_rows": 0,
                "xmcd_rows": 0,
                "xas_right_removed_nonfinite_rows": 0,
                "xas_left_removed_nonfinite_rows": 0,
                "xmcd_removed_nonfinite_rows": 0,
                "feature_count": 0,
            }
        )

        reasons = []

        # ----------------------------------------------------
        # Prefix checks
        # ----------------------------------------------------

        if not prefix:
            add_reason(
                reasons,
                "empty_prefix",
            )

        elif prefix in observed_prefixes:
            add_reason(
                reasons,
                "duplicate_prefix",
            )

        else:
            observed_prefixes.add(
                prefix
            )

        # ----------------------------------------------------
        # Label checks
        # ----------------------------------------------------

        if (
            not np.isfinite(
                sz_value
            )
            or not np.isfinite(
                lz_value
            )
        ):
            add_reason(
                reasons,
                "nonfinite_label",
            )

        elif (
            exclude_both_zero
            and abs(
                float(
                    sz_value
                )
            ) < zero_tolerance
            and abs(
                float(
                    lz_value
                )
            ) < zero_tolerance
        ):
            add_reason(
                reasons,
                "both_moments_below_tolerance",
            )

        # ----------------------------------------------------
        # External field checks
        # ----------------------------------------------------

        if not np.isfinite(
            b_ext_value
        ):
            add_reason(
                reasons,
                "nonfinite_external_magnetic_field",
            )

        elif field_enabled:
            if not np.isclose(
                float(
                    b_ext_value
                ),
                expected_field,
                rtol=0.0,
                atol=field_tolerance,
            ):
                add_reason(
                    reasons,
                    "external_magnetic_field_mismatch",
                )

        elif not np.isclose(
            float(
                b_ext_value
            ),
            0.0,
            rtol=0.0,
            atol=field_tolerance,
        ):
            add_reason(
                reasons,
                "unexpected_nonzero_external_magnetic_field",
            )

        # ----------------------------------------------------
        # File checks
        # ----------------------------------------------------

        if (
            prefix
            and not right_file.is_file()
        ):
            add_reason(
                reasons,
                "missing_xas_right_file",
            )

        if (
            prefix
            and not left_file.is_file()
        ):
            add_reason(
                reasons,
                "missing_xas_left_file",
            )

        if (
            prefix
            and not xmcd_file.is_file()
        ):
            add_reason(
                reasons,
                "missing_xmcd_file",
            )

        features = None

        # ----------------------------------------------------
        # Load spectra
        # ----------------------------------------------------

        if not reasons:
            try:
                (
                    energy_right,
                    right_values,
                    removed_right,
                ) = load_spectrum(
                    right_file,
                    skip_header,
                    energy_column,
                    value_column,
                )

                (
                    energy_left,
                    left_values,
                    removed_left,
                ) = load_spectrum(
                    left_file,
                    skip_header,
                    energy_column,
                    value_column,
                )

                (
                    energy_xmcd,
                    xmcd_values,
                    removed_xmcd,
                ) = load_spectrum(
                    xmcd_file,
                    skip_header,
                    energy_column,
                    value_column,
                )

                audit[
                    "xas_right_rows"
                ] = int(
                    len(
                        right_values
                    )
                )

                audit[
                    "xas_left_rows"
                ] = int(
                    len(
                        left_values
                    )
                )

                audit[
                    "xmcd_rows"
                ] = int(
                    len(
                        xmcd_values
                    )
                )

                audit[
                    "xas_right_removed_nonfinite_rows"
                ] = int(
                    removed_right
                )

                audit[
                    "xas_left_removed_nonfinite_rows"
                ] = int(
                    removed_left
                )

                audit[
                    "xmcd_removed_nonfinite_rows"
                ] = int(
                    removed_xmcd
                )

                # --------------------------------------------
                # Length checks
                # --------------------------------------------

                if require_equal_length:
                    if len(
                        right_values
                    ) != len(
                        left_values
                    ):
                        add_reason(
                            reasons,
                            "right_left_length_mismatch",
                        )

                    if len(
                        right_values
                    ) != len(
                        xmcd_values
                    ):
                        add_reason(
                            reasons,
                            "xas_xmcd_length_mismatch",
                        )

                # --------------------------------------------
                # Use common length if permitted
                # --------------------------------------------

                if not reasons:
                    common_length = min(
                        len(
                            right_values
                        ),
                        len(
                            left_values
                        ),
                        len(
                            xmcd_values
                        ),
                    )

                    if common_length <= 0:
                        add_reason(
                            reasons,
                            "empty_common_spectrum",
                        )

                    else:
                        energy_right = energy_right[
                            :common_length
                        ]

                        right_values = right_values[
                            :common_length
                        ]

                        energy_left = energy_left[
                            :common_length
                        ]

                        left_values = left_values[
                            :common_length
                        ]

                        energy_xmcd = energy_xmcd[
                            :common_length
                        ]

                        xmcd_values = xmcd_values[
                            :common_length
                        ]

                # --------------------------------------------
                # Energy-grid checks
                # --------------------------------------------

                if (
                    not reasons
                    and require_same_grid
                ):
                    if not same_energy_grid(
                        energy_right,
                        energy_left,
                        energy_rtol,
                        energy_atol,
                    ):
                        add_reason(
                            reasons,
                            "right_left_energy_grid_mismatch",
                        )

                    if not same_energy_grid(
                        energy_right,
                        energy_xmcd,
                        energy_rtol,
                        energy_atol,
                    ):
                        add_reason(
                            reasons,
                            "xas_xmcd_energy_grid_mismatch",
                        )

                # --------------------------------------------
                # Construct XAS and feature vector
                # --------------------------------------------

                if not reasons:
                    xas_values = (
                        right_values
                        + left_values
                    ) / 2.0

                    energy_xas = (
                        energy_right
                    )

                    ensure_finite(
                        "Constructed XAS",
                        xas_values,
                    )

                    audit[
                        "xas_rows"
                    ] = int(
                        len(
                            xas_values
                        )
                    )

                    features = np.concatenate(
                        [
                            xas_values,
                            xmcd_values,
                        ]
                    ).astype(
                        np.float64,
                        copy=False,
                    )

                    audit[
                        "feature_count"
                    ] = int(
                        len(
                            features
                        )
                    )

                    if not np.isfinite(
                        features
                    ).all():
                        add_reason(
                            reasons,
                            "nonfinite_feature_after_loading",
                        )

                # --------------------------------------------
                # Reference feature/grid checks
                # --------------------------------------------

                if (
                    features is not None
                    and not reasons
                ):
                    if expected_feature_count is None:
                        expected_feature_count = int(
                            len(
                                features
                            )
                        )

                        reference_energy_xas = (
                            energy_xas.copy()
                        )

                        reference_energy_xmcd = (
                            energy_xmcd.copy()
                        )

                    elif len(
                        features
                    ) != expected_feature_count:
                        add_reason(
                            reasons,
                            "feature_count_mismatch",
                        )

                    elif require_same_grid:
                        if not same_energy_grid(
                            energy_xas,
                            reference_energy_xas,
                            energy_rtol,
                            energy_atol,
                        ):
                            add_reason(
                                reasons,
                                "xas_energy_grid_differs_from_reference",
                            )

                        if not same_energy_grid(
                            energy_xmcd,
                            reference_energy_xmcd,
                            energy_rtol,
                            energy_atol,
                        ):
                            add_reason(
                                reasons,
                                "xmcd_energy_grid_differs_from_reference",
                            )

                # --------------------------------------------
                # Accept the sample
                # --------------------------------------------

                if (
                    features is not None
                    and not reasons
                ):
                    accepted_x.append(
                        features
                    )

                    accepted_y.append(
                        [
                            float(
                                sz_value
                            ),
                            float(
                                lz_value
                            ),
                        ]
                    )

                    accepted_ids.append(
                        sample_id
                    )

                    accepted_prefixes.append(
                        prefix
                    )

                    accepted_fields.append(
                        float(
                            b_ext_value
                        )
                    )

            except Exception as exception:
                add_reason(
                    reasons,
                    exception_reason(
                        exception
                    ),
                )

        audit[
            "accepted"
        ] = (
            len(
                reasons
            )
            == 0
        )

        audit[
            "exclusion_reason"
        ] = ";".join(
            reasons
        )

        records.append(
            audit
        )

    # --------------------------------------------------------
    # Build metadata tables
    # --------------------------------------------------------

    metadata = pd.DataFrame(
        records
    )

    if metadata.empty:
        raise RuntimeError(
            "No source rows were processed"
        )

    accepted_mask = metadata[
        "accepted"
    ].astype(
        bool
    )

    accepted_metadata = (
        metadata.loc[
            accepted_mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    excluded_metadata = (
        metadata.loc[
            ~accepted_mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if not accepted_x:
        raise RuntimeError(
            "No samples were accepted"
        )

    # --------------------------------------------------------
    # Build final arrays
    # --------------------------------------------------------

    X = np.asarray(
        accepted_x,
        dtype=np.float64,
    )

    Y = np.asarray(
        accepted_y,
        dtype=np.float64,
    )

    sample_ids = np.asarray(
        accepted_ids,
        dtype=str,
    )

    prefixes = np.asarray(
        accepted_prefixes,
        dtype=str,
    )

    b_ext_values = np.asarray(
        accepted_fields,
        dtype=np.float64,
    )

    if (
        X.ndim != 2
        or Y.ndim != 2
        or Y.shape[
            1
        ] != 2
    ):
        raise RuntimeError(
            "Unexpected final array shapes: "
            f"X={X.shape}, Y={Y.shape}"
        )

    if not (
        len(
            X
        )
        == len(
            Y
        )
        == len(
            sample_ids
        )
        == len(
            prefixes
        )
        == len(
            b_ext_values
        )
    ):
        raise RuntimeError(
            "Accepted arrays have inconsistent sample counts"
        )

    if len(
        set(
            sample_ids.tolist()
        )
    ) != len(
        sample_ids
    ):
        raise RuntimeError(
            "Duplicate sample IDs were detected"
        )

    if len(
        set(
            prefixes.tolist()
        )
    ) != len(
        prefixes
    ):
        raise RuntimeError(
            "Duplicate accepted prefixes were detected"
        )

    ensure_finite(
        "Final X",
        X,
    )

    ensure_finite(
        "Final Y",
        Y,
    )

    ensure_finite(
        "Accepted external fields",
        b_ext_values,
    )

    if (
        reference_energy_xas is None
        or reference_energy_xmcd is None
    ):
        raise RuntimeError(
            "Reference energy grids were not created"
        )

    ensure_finite(
        "Reference XAS energy",
        reference_energy_xas,
    )

    ensure_finite(
        "Reference XMCD energy",
        reference_energy_xmcd,
    )

    if (
        field_enabled
        and not np.allclose(
            b_ext_values,
            expected_field,
            rtol=0.0,
            atol=field_tolerance,
        )
    ):
        raise RuntimeError(
            "Accepted samples contain inconsistent "
            "external-field values"
        )

    # --------------------------------------------------------
    # Save NPZ
    # --------------------------------------------------------

    np.savez_compressed(
        dataset_file,
        X=X,
        Y=Y,
        sample_id=sample_ids,
        prefix=prefixes,
        energy_xas=reference_energy_xas,
        energy_xmcd=reference_energy_xmcd,
        b_ext_t=b_ext_values,
        external_field_enabled=np.asarray(
            [
                field_enabled
            ],
            dtype=bool,
        ),
        external_field_magnitude_t=np.asarray(
            [
                expected_field
            ],
            dtype=np.float64,
        ),
        external_field_direction=np.asarray(
            field_direction,
            dtype=np.float64,
        ),
        external_field_is_model_input=np.asarray(
            [
                False
            ],
            dtype=bool,
        ),
        target_names=np.asarray(
            [
                "Sz",
                "Lz",
            ],
            dtype=str,
        ),
        n_xas=np.asarray(
            [
                len(
                    reference_energy_xas
                )
            ],
            dtype=np.int64,
        ),
        n_xmcd=np.asarray(
            [
                len(
                    reference_energy_xmcd
                )
            ],
            dtype=np.int64,
        ),
        xas_definition=np.asarray(
            [
                "(I_r + I_l) / 2"
            ],
            dtype=str,
        ),
        xmcd_definition=np.asarray(
            [
                "I_r - I_l"
            ],
            dtype=str,
        ),
    )

    # --------------------------------------------------------
    # Save metadata CSV files
    # --------------------------------------------------------

    metadata.to_csv(
        metadata_file,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )

    accepted_metadata.to_csv(
        accepted_file,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )

    excluded_metadata.to_csv(
        excluded_file,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )

    # --------------------------------------------------------
    # Count exclusion reasons
    # --------------------------------------------------------

    reason_counter = Counter()

    for reason_text in excluded_metadata[
        "exclusion_reason"
    ].fillna(
        ""
    ):
        reason_counter.update(
            [
                reason
                for reason in str(
                    reason_text
                ).split(
                    ";"
                )
                if reason
            ]
        )

    # --------------------------------------------------------
    # Build summary
    # --------------------------------------------------------

    summary = {
        "created_at_utc": utc_now_iso(),

        "system": system,

        "config_file": safe_public_path(
            config_path,
            repository_root,
        ),

        "config_sha256": sha256_file(
            config_path
        ),

        "source_results_csv": safe_public_path(
            results_csv,
            repository_root,
        ),

        "source_results_csv_sha256": sha256_file(
            results_csv
        ),

        "spec_directory": safe_public_path(
            spec_directory,
            repository_root,
        ),

        "external_magnetic_field": {
            "enabled": field_enabled,

            "magnitude_t": expected_field,

            "direction": field_direction,

            "field_column": b_ext_column,

            "field_tolerance_t": field_tolerance,

            "accepted_field_summary_t": numeric_summary(
                b_ext_values
            ),

            "is_model_input": False,

            "hamiltonian_term": (
                "mu_B * B_ext * "
                "(g_L * Lz + g_S * Sz)"
            ),
        },

        "spectrum_settings": {
            "xas_right_suffix": right_suffix,

            "xas_left_suffix": left_suffix,

            "xmcd_suffix": xmcd_suffix,

            "xas_definition": "(I_r + I_l) / 2",

            "xmcd_definition": "I_r - I_l",

            "skip_header": skip_header,

            "energy_column": energy_column,

            "value_column": value_column,

            "require_equal_length": (
                require_equal_length
            ),

            "require_same_energy_grid": (
                require_same_grid
            ),

            "energy_rtol": energy_rtol,

            "energy_atol": energy_atol,
        },

        "output_files": {
            "dataset_npz": safe_public_path(
                dataset_file,
                repository_root,
            ),

            "metadata_csv": safe_public_path(
                metadata_file,
                repository_root,
            ),

            "accepted_csv": safe_public_path(
                accepted_file,
                repository_root,
            ),

            "excluded_csv": safe_public_path(
                excluded_file,
                repository_root,
            ),

            "summary_json": safe_public_path(
                summary_file,
                repository_root,
            ),
        },

        "output_sha256": {
            "dataset_npz": sha256_file(
                dataset_file
            ),

            "metadata_csv": sha256_file(
                metadata_file
            ),

            "accepted_csv": sha256_file(
                accepted_file
            ),

            "excluded_csv": sha256_file(
                excluded_file
            ),
        },

        "selection": {
            "exclude_both_moments_zero": (
                exclude_both_zero
            ),

            "zero_tolerance": zero_tolerance,

            "criterion": (
                "abs(Sz) < tolerance and "
                "abs(Lz) < tolerance"
            ),
        },

        "counts": {
            "source_rows": int(
                len(
                    source
                )
            ),

            "accepted": int(
                len(
                    accepted_metadata
                )
            ),

            "excluded": int(
                len(
                    excluded_metadata
                )
            ),
        },

        "exclusion_reason_counts": dict(
            sorted(
                reason_counter.items()
            )
        ),

        "arrays": {
            "X_shape": list(
                X.shape
            ),

            "Y_shape": list(
                Y.shape
            ),

            "X_dtype": str(
                X.dtype
            ),

            "Y_dtype": str(
                Y.dtype
            ),

            "n_xas": int(
                len(
                    reference_energy_xas
                )
            ),

            "n_xmcd": int(
                len(
                    reference_energy_xmcd
                )
            ),

            "feature_order": (
                "[XAS channels, XMCD channels]"
            ),

            "target_order": [
                "Sz",
                "Lz",
            ],
        },

        "accepted_label_summary": {
            "Sz": numeric_summary(
                Y[
                    :,
                    0,
                ]
            ),

            "Lz": numeric_summary(
                Y[
                    :,
                    1,
                ]
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
        f"System             : {system_name}"
    )

    print(
        "Source CSV         : "
        f"{safe_public_path(results_csv, repository_root)}"
    )

    print(
        "Spectrum directory : "
        f"{safe_public_path(spec_directory, repository_root)}"
    )

    print(
        "XAS definition     : (I_r + I_l) / 2"
    )

    print(
        "XMCD definition    : I_r - I_l"
    )

    print(
        f"External field     : {expected_field:.6f} T"
    )

    print(
        f"Field direction    : {field_direction}"
    )

    print(
        "Field is NN input  : False"
    )

    print(
        f"Original samples   : {len(source)}"
    )

    print(
        f"Accepted samples   : {len(accepted_metadata)}"
    )

    print(
        f"Excluded samples   : {len(excluded_metadata)}"
    )

    print(
        f"X shape            : {X.shape}"
    )

    print(
        f"Y shape            : {Y.shape}"
    )

    print(
        "Dataset            : "
        f"{safe_public_path(dataset_file, repository_root)}"
    )

    print(
        "Metadata           : "
        f"{safe_public_path(metadata_file, repository_root)}"
    )

    print(
        "Accepted records   : "
        f"{safe_public_path(accepted_file, repository_root)}"
    )

    print(
        "Excluded records   : "
        f"{safe_public_path(excluded_file, repository_root)}"
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
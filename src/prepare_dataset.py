#!/usr/bin/env python3

"""
Prepare a neural-network dataset from Quanty XAS/XMCD spectra.

Outputs
-------
<system>_dataset.npz
    X, Y, sample IDs, prefixes, and energy axes.

<system>_metadata.csv
    Complete audit table containing all source samples.

<system>_accepted.csv
    Samples accepted for neural-network processing.

<system>_excluded.csv
    Excluded samples and their exclusion reasons.

<system>_summary.json
    Counts, label ranges, file hashes, and software versions.

Privacy
-------
Internal calculations use absolute paths. Paths written to public
JSON and CSV files and displayed in normal console output are converted
to paths relative to the repository root.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


# Support both:
#
#   python src/prepare_dataset.py
#
# and:
#
#   python -m src.prepare_dataset

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
            "Prepare an XAS/XMCD neural-network dataset "
            "from Quanty outputs."
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
# Public-path utility
# ============================================================

def safe_public_path(
    path,
    repository_root,
):
    """
    Return a public-safe path.

    Paths inside the repository are displayed relative to the
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

def validate_nonnegative_integer(
    name,
    value,
):
    """
    Validate a nonnegative integer configuration value.
    """

    if isinstance(
        value,
        bool,
    ):
        raise TypeError(
            f"{name} must be an integer, not bool"
        )

    if not isinstance(
        value,
        int,
    ):
        raise TypeError(
            f"{name} must be an integer"
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
    """
    Validate a finite nonnegative floating-point value.
    """

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

    if not np.isfinite(
        result
    ):
        raise ValueError(
            f"{name} must be finite"
        )

    if result < 0.0:
        raise ValueError(
            f"{name} must be nonnegative"
        )

    return result


def validate_config(
    config,
):
    """
    Validate the required dataset configuration.
    """

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

    required_paths = (
        "repository_root",
        "spec_directory",
        "quanty_results_csv",
        "output_directory",
    )

    missing_path_settings = [
        setting
        for setting in required_paths
        if setting not in config[
            "paths"
        ]
    ]

    if missing_path_settings:
        raise ValueError(
            "Missing path settings: "
            f"{missing_path_settings}"
        )

    required_spectrum_settings = (
        "xas_suffix",
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
            f"{missing_spectrum_settings}"
        )

    required_column_settings = (
        "prefix",
        "sz",
        "lz",
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

    zero_tolerance = validate_nonnegative_float(
        "exclusion.zero_tolerance",
        config[
            "exclusion"
        ].get(
            "zero_tolerance",
            1.0e-6,
        ),
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

    xas_suffix = str(
        config[
            "spectrum"
        ][
            "xas_suffix"
        ]
    )

    xmcd_suffix = str(
        config[
            "spectrum"
        ][
            "xmcd_suffix"
        ]
    )

    if not xas_suffix:
        raise ValueError(
            "spectrum.xas_suffix must not be empty"
        )

    if not xmcd_suffix:
        raise ValueError(
            "spectrum.xmcd_suffix must not be empty"
        )

    if xas_suffix == xmcd_suffix:
        raise ValueError(
            "XAS and XMCD suffixes must differ"
        )

    column_values = [
        str(
            config[
                "columns"
            ][
                setting
            ]
        )
        for setting in required_column_settings
    ]

    if any(
        not value
        for value in column_values
    ):
        raise ValueError(
            "Column names must not be empty"
        )

    if len(
        set(
            column_values
        )
    ) != len(column_values):
        raise ValueError(
            "Prefix, Sz, and Lz columns "
            "must have different names"
        )

    return {
        "system_name": (
            system_name
        ),
        "element_name": (
            element_name
        ),
        "n_electrons_3d": (
            n_electrons_3d
        ),
        "skip_header": (
            skip_header
        ),
        "energy_column": (
            energy_column
        ),
        "value_column": (
            value_column
        ),
        "zero_tolerance": (
            zero_tolerance
        ),
        "energy_rtol": (
            energy_rtol
        ),
        "energy_atol": (
            energy_atol
        ),
        "xas_suffix": (
            xas_suffix
        ),
        "xmcd_suffix": (
            xmcd_suffix
        ),
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
    """
    Read a Quanty .spec file.

    Rows containing NaN or Inf in any column are removed. The
    number of removed rows is returned for auditing.

    Parameters
    ----------
    path : Path
        Spectrum file.

    skip_header : int
        Number of initial lines to skip.

    energy_column : int
        Zero-based energy-column index.

    value_column : int
        Zero-based spectral-value-column index.

    Returns
    -------
    tuple
        energy, values, removed_nonfinite_row_count
    """

    path = Path(
        path
    )

    if not path.is_file():
        raise FileNotFoundError(
            "Spectrum file was not found"
        )

    raw = np.genfromtxt(
        path,
        comments="#",
        skip_header=skip_header,
        dtype=np.float64,
    )

    if raw.size == 0:
        raise ValueError(
            "Empty spectrum"
        )

    if raw.ndim == 0:
        raise ValueError(
            "Spectrum contains no tabular data"
        )

    if raw.ndim == 1:
        raw = raw.reshape(
            1,
            -1,
        )

    if raw.ndim != 2:
        raise ValueError(
            "Spectrum data must be two-dimensional"
        )

    required_column = max(
        energy_column,
        value_column,
    )

    if raw.shape[1] <= required_column:
        raise ValueError(
            f"Requested column {required_column}, "
            f"but the spectrum contains only "
            f"{raw.shape[1]} columns"
        )

    finite_row_mask = np.isfinite(
        raw
    ).all(
        axis=1
    )

    cleaned = raw[
        finite_row_mask
    ]

    removed_row_count = int(
        raw.shape[0]
        - cleaned.shape[0]
    )

    if cleaned.shape[0] == 0:
        raise ValueError(
            "No finite spectrum rows"
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

    if energy.ndim != 1:
        raise ValueError(
            "Energy axis is not one-dimensional"
        )

    if values.ndim != 1:
        raise ValueError(
            "Spectrum values are not one-dimensional"
        )

    if len(energy) != len(values):
        raise ValueError(
            "Energy and value arrays have different lengths"
        )

    return (
        energy,
        values,
        removed_row_count,
    )


# ============================================================
# Exclusion utilities
# ============================================================

def add_reason(
    reasons,
    reason,
):
    """
    Add a unique exclusion reason.
    """

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
    """
    Convert an exception to a compact CSV-safe reason.

    Local absolute paths are not included.
    """

    message = str(
        exception
    )

    message = " ".join(
        message.split()
    )

    message = message.replace(
        ";",
        ",",
    )

    if len(message) > 200:
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


# ============================================================
# Summary helpers
# ============================================================

def numeric_summary(
    values,
):
    """
    Return a JSON-serializable numeric summary.
    """

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
# Main dataset preparation
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

    system = config[
        "system"
    ]

    paths = config[
        "paths"
    ]

    spectrum_config = config[
        "spectrum"
    ]

    columns = config[
        "columns"
    ]

    exclusion_config = config[
        "exclusion"
    ]

    system_name = normalized_config[
        "system_name"
    ]

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
    # Output files
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

    output_files = [
        dataset_file,
        metadata_file,
        accepted_file,
        excluded_file,
        summary_file,
    ]

    check_output_files(
        output_files,
        overwrite=(
            arguments.overwrite
        ),
        label="Dataset output files",
    )

    # --------------------------------------------------------
    # Read Quanty labels and parameters
    # --------------------------------------------------------

    source_dataframe = pd.read_csv(
        results_csv
    )

    if source_dataframe.empty:
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

    required_csv_columns = [
        prefix_column,
        sz_column,
        lz_column,
    ]

    missing_csv_columns = [
        column
        for column in required_csv_columns
        if column not in source_dataframe.columns
    ]

    if missing_csv_columns:
        raise ValueError(
            "Required columns are missing from "
            "the Quanty results CSV: "
            f"{missing_csv_columns}"
        )

    # --------------------------------------------------------
    # Normalized configuration values
    # --------------------------------------------------------

    zero_tolerance = normalized_config[
        "zero_tolerance"
    ]

    exclude_both_zero = bool(
        exclusion_config.get(
            "exclude_both_moments_zero",
            True,
        )
    )

    xas_suffix = normalized_config[
        "xas_suffix"
    ]

    xmcd_suffix = normalized_config[
        "xmcd_suffix"
    ]

    skip_header = normalized_config[
        "skip_header"
    ]

    energy_column = normalized_config[
        "energy_column"
    ]

    value_column = normalized_config[
        "value_column"
    ]

    require_equal_length = bool(
        spectrum_config.get(
            "require_equal_length",
            True,
        )
    )

    require_same_energy_grid = bool(
        spectrum_config.get(
            "require_same_energy_grid",
            True,
        )
    )

    energy_relative_tolerance = (
        normalized_config[
            "energy_rtol"
        ]
    )

    energy_absolute_tolerance = (
        normalized_config[
            "energy_atol"
        ]
    )

    # --------------------------------------------------------
    # Output containers
    # --------------------------------------------------------

    metadata_records = []

    accepted_x = []
    accepted_y = []
    accepted_sample_ids = []
    accepted_prefixes = []

    expected_feature_count = None

    reference_energy_xas = None
    reference_energy_xmcd = None

    observed_prefixes = set()

    # --------------------------------------------------------
    # Process every source row
    # --------------------------------------------------------

    for row_position, (
        source_index,
        source_row,
    ) in enumerate(
        source_dataframe.iterrows()
    ):
        sample_id = (
            f"{system_name}_"
            f"{row_position:06d}"
        )

        raw_prefix = source_row[
            prefix_column
        ]

        if pd.isna(
            raw_prefix
        ):
            prefix = ""

        else:
            prefix = str(
                raw_prefix
            ).strip()

        sz_value = pd.to_numeric(
            pd.Series(
                [
                    source_row[
                        sz_column
                    ]
                ]
            ),
            errors="coerce",
        ).iloc[0]

        lz_value = pd.to_numeric(
            pd.Series(
                [
                    source_row[
                        lz_column
                    ]
                ]
            ),
            errors="coerce",
        ).iloc[0]

        xas_file = (
            spec_directory
            / f"{prefix}{xas_suffix}"
        )

        xmcd_file = (
            spec_directory
            / f"{prefix}{xmcd_suffix}"
        )

        audit_record = (
            source_row.to_dict()
        )

        audit_record.update(
            {
                "sample_id": (
                    sample_id
                ),

                "source_index": (
                    str(
                        source_index
                    )
                ),

                "source_row_position": int(
                    row_position
                ),

                "system": (
                    system_name
                ),

                "element": (
                    normalized_config[
                        "element_name"
                    ]
                ),

                "n_electrons_3d": int(
                    normalized_config[
                        "n_electrons_3d"
                    ]
                ),

                "prefix": (
                    prefix
                ),

                "sz_label": (
                    sz_value
                ),

                "lz_label": (
                    lz_value
                ),

                "xas_file": (
                    safe_public_path(
                        xas_file,
                        repository_root,
                    )
                ),

                "xmcd_file": (
                    safe_public_path(
                        xmcd_file,
                        repository_root,
                    )
                ),

                "accepted": False,

                "exclusion_reason": "",

                "xas_rows": 0,

                "xmcd_rows": 0,

                "xas_removed_nonfinite_rows": 0,

                "xmcd_removed_nonfinite_rows": 0,

                "feature_count": 0,
            }
        )

        exclusion_reasons = []

        # ----------------------------------------------------
        # Prefix validity
        # ----------------------------------------------------

        if not prefix:
            add_reason(
                exclusion_reasons,
                "empty_prefix",
            )

        elif prefix in observed_prefixes:
            add_reason(
                exclusion_reasons,
                "duplicate_prefix",
            )

        else:
            observed_prefixes.add(
                prefix
            )

        # ----------------------------------------------------
        # Label validity and zero-label criterion
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
                exclusion_reasons,
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
                exclusion_reasons,
                "both_moments_below_tolerance",
            )

        # ----------------------------------------------------
        # File existence
        # ----------------------------------------------------

        if (
            prefix
            and not xas_file.is_file()
        ):
            add_reason(
                exclusion_reasons,
                "missing_xas_file",
            )

        if (
            prefix
            and not xmcd_file.is_file()
        ):
            add_reason(
                exclusion_reasons,
                "missing_xmcd_file",
            )

        # ----------------------------------------------------
        # Load spectra if not already excluded
        # ----------------------------------------------------

        if not exclusion_reasons:
            try:
                (
                    energy_xas,
                    xas_values,
                    removed_xas_rows,
                ) = load_spectrum(
                    xas_file,
                    skip_header,
                    energy_column,
                    value_column,
                )

                (
                    energy_xmcd,
                    xmcd_values,
                    removed_xmcd_rows,
                ) = load_spectrum(
                    xmcd_file,
                    skip_header,
                    energy_column,
                    value_column,
                )

                audit_record[
                    "xas_rows"
                ] = int(
                    len(
                        xas_values
                    )
                )

                audit_record[
                    "xmcd_rows"
                ] = int(
                    len(
                        xmcd_values
                    )
                )

                audit_record[
                    "xas_removed_nonfinite_rows"
                ] = int(
                    removed_xas_rows
                )

                audit_record[
                    "xmcd_removed_nonfinite_rows"
                ] = int(
                    removed_xmcd_rows
                )

                # --------------------------------------------
                # XAS/XMCD length consistency
                # --------------------------------------------

                if (
                    require_equal_length
                    and len(
                        xas_values
                    ) != len(
                        xmcd_values
                    )
                ):
                    add_reason(
                        exclusion_reasons,
                        "xas_xmcd_length_mismatch",
                    )

                else:
                    common_length = min(
                        len(
                            xas_values
                        ),
                        len(
                            xmcd_values
                        ),
                    )

                    if common_length <= 0:
                        add_reason(
                            exclusion_reasons,
                            "empty_common_spectrum",
                        )

                    else:
                        xas_values = (
                            xas_values[
                                :common_length
                            ]
                        )

                        xmcd_values = (
                            xmcd_values[
                                :common_length
                            ]
                        )

                        energy_xas = (
                            energy_xas[
                                :common_length
                            ]
                        )

                        energy_xmcd = (
                            energy_xmcd[
                                :common_length
                            ]
                        )

                # --------------------------------------------
                # XAS/XMCD energy-grid consistency
                # --------------------------------------------

                if (
                    not exclusion_reasons
                    and require_same_energy_grid
                ):
                    same_grid = np.allclose(
                        energy_xas,
                        energy_xmcd,
                        rtol=(
                            energy_relative_tolerance
                        ),
                        atol=(
                            energy_absolute_tolerance
                        ),
                        equal_nan=False,
                    )

                    if not same_grid:
                        add_reason(
                            exclusion_reasons,
                            "xas_xmcd_energy_grid_mismatch",
                        )

                # --------------------------------------------
                # Concatenated NN input
                # --------------------------------------------

                features = None

                if not exclusion_reasons:
                    features = np.concatenate(
                        [
                            xas_values,
                            xmcd_values,
                        ]
                    ).astype(
                        np.float64,
                        copy=False,
                    )

                    audit_record[
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
                            exclusion_reasons,
                            "nonfinite_feature_after_loading",
                        )

                # --------------------------------------------
                # Dataset-wide shape/grid consistency
                # --------------------------------------------

                if (
                    features is not None
                    and not exclusion_reasons
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

                    elif (
                        len(
                            features
                        )
                        != expected_feature_count
                    ):
                        add_reason(
                            exclusion_reasons,
                            "feature_count_mismatch",
                        )

                    elif require_same_energy_grid:
                        xas_grid_matches = np.allclose(
                            energy_xas,
                            reference_energy_xas,
                            rtol=(
                                energy_relative_tolerance
                            ),
                            atol=(
                                energy_absolute_tolerance
                            ),
                            equal_nan=False,
                        )

                        xmcd_grid_matches = np.allclose(
                            energy_xmcd,
                            reference_energy_xmcd,
                            rtol=(
                                energy_relative_tolerance
                            ),
                            atol=(
                                energy_absolute_tolerance
                            ),
                            equal_nan=False,
                        )

                        if not xas_grid_matches:
                            add_reason(
                                exclusion_reasons,
                                "xas_energy_grid_differs_from_reference",
                            )

                        if not xmcd_grid_matches:
                            add_reason(
                                exclusion_reasons,
                                "xmcd_energy_grid_differs_from_reference",
                            )

                # --------------------------------------------
                # Accept sample
                # --------------------------------------------

                if (
                    features is not None
                    and not exclusion_reasons
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

                    accepted_sample_ids.append(
                        sample_id
                    )

                    accepted_prefixes.append(
                        prefix
                    )

            except Exception as exception:
                add_reason(
                    exclusion_reasons,
                    exception_reason(
                        exception
                    ),
                )

        audit_record[
            "accepted"
        ] = (
            len(
                exclusion_reasons
            )
            == 0
        )

        audit_record[
            "exclusion_reason"
        ] = ";".join(
            exclusion_reasons
        )

        metadata_records.append(
            audit_record
        )

    # --------------------------------------------------------
    # Construct audit tables
    # --------------------------------------------------------

    metadata_dataframe = pd.DataFrame(
        metadata_records
    )

    if metadata_dataframe.empty:
        raise RuntimeError(
            "No source rows were processed"
        )

    accepted_mask = (
        metadata_dataframe[
            "accepted"
        ].astype(bool)
    )

    accepted_metadata = (
        metadata_dataframe.loc[
            accepted_mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    excluded_metadata = (
        metadata_dataframe.loc[
            ~accepted_mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if not accepted_x:
        raise RuntimeError(
            "No samples were accepted. "
            "Dataset was not created."
        )

    # --------------------------------------------------------
    # Final numeric arrays
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
        accepted_sample_ids,
        dtype=str,
    )

    prefixes = np.asarray(
        accepted_prefixes,
        dtype=str,
    )

    if X.ndim != 2:
        raise RuntimeError(
            f"Unexpected X shape: {X.shape}"
        )

    if (
        Y.ndim != 2
        or Y.shape[1] != 2
    ):
        raise RuntimeError(
            f"Unexpected Y shape: {Y.shape}"
        )

    if not (
        len(X)
        == len(Y)
        == len(sample_ids)
        == len(prefixes)
    ):
        raise RuntimeError(
            "Accepted arrays have inconsistent "
            "sample counts"
        )

    if len(
        set(
            sample_ids.tolist()
        )
    ) != len(sample_ids):
        raise RuntimeError(
            "Duplicate sample_id values were detected"
        )

    if len(
        set(
            prefixes.tolist()
        )
    ) != len(prefixes):
        raise RuntimeError(
            "Duplicate accepted Prefix values were detected"
        )

    ensure_finite(
        "Final X",
        X,
    )

    ensure_finite(
        "Final Y",
        Y,
    )

    if reference_energy_xas is None:
        raise RuntimeError(
            "Reference XAS energy grid was not created"
        )

    if reference_energy_xmcd is None:
        raise RuntimeError(
            "Reference XMCD energy grid was not created"
        )

    ensure_finite(
        "Reference XAS energy grid",
        reference_energy_xas,
    )

    ensure_finite(
        "Reference XMCD energy grid",
        reference_energy_xmcd,
    )

    # --------------------------------------------------------
    # Save compressed NN dataset
    # --------------------------------------------------------

    np.savez_compressed(
        dataset_file,

        X=(
            X
        ),

        Y=(
            Y
        ),

        sample_id=(
            sample_ids
        ),

        prefix=(
            prefixes
        ),

        energy_xas=(
            reference_energy_xas
        ),

        energy_xmcd=(
            reference_energy_xmcd
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
    )

    # --------------------------------------------------------
    # Save audit CSV files
    # --------------------------------------------------------

    metadata_dataframe.to_csv(
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
    # Count every exclusion reason
    # --------------------------------------------------------

    exclusion_reason_counter = Counter()

    for reason_text in excluded_metadata[
        "exclusion_reason"
    ].fillna(
        ""
    ):
        reasons = [
            reason
            for reason in str(
                reason_text
            ).split(
                ";"
            )
            if reason
        ]

        exclusion_reason_counter.update(
            reasons
        )

    exclusion_reason_counts = dict(
        sorted(
            exclusion_reason_counter.items()
        )
    )

    # --------------------------------------------------------
    # Summary JSON
    # --------------------------------------------------------

    summary = {
        "created_at_utc": (
            utc_now_iso()
        ),

        "system": (
            system
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

        "source_results_csv": (
            safe_public_path(
                results_csv,
                repository_root,
            )
        ),

        "source_results_csv_sha256": (
            sha256_file(
                results_csv
            )
        ),

        "spec_directory": (
            safe_public_path(
                spec_directory,
                repository_root,
            )
        ),

        "spectrum_settings": {
            "xas_suffix": (
                xas_suffix
            ),

            "xmcd_suffix": (
                xmcd_suffix
            ),

            "skip_header": (
                skip_header
            ),

            "energy_column": (
                energy_column
            ),

            "value_column": (
                value_column
            ),

            "require_equal_length": (
                require_equal_length
            ),

            "require_same_energy_grid": (
                require_same_energy_grid
            ),

            "energy_rtol": (
                energy_relative_tolerance
            ),

            "energy_atol": (
                energy_absolute_tolerance
            ),
        },

        "output_files": {
            "dataset_npz": (
                safe_public_path(
                    dataset_file,
                    repository_root,
                )
            ),

            "metadata_csv": (
                safe_public_path(
                    metadata_file,
                    repository_root,
                )
            ),

            "accepted_csv": (
                safe_public_path(
                    accepted_file,
                    repository_root,
                )
            ),

            "excluded_csv": (
                safe_public_path(
                    excluded_file,
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
            "dataset_npz": (
                sha256_file(
                    dataset_file
                )
            ),

            "metadata_csv": (
                sha256_file(
                    metadata_file
                )
            ),

            "accepted_csv": (
                sha256_file(
                    accepted_file
                )
            ),

            "excluded_csv": (
                sha256_file(
                    excluded_file
                )
            ),
        },

        "selection": {
            "exclude_both_moments_zero": (
                exclude_both_zero
            ),

            "zero_tolerance": (
                zero_tolerance
            ),

            "criterion": (
                "abs(Sz) < tolerance and "
                "abs(Lz) < tolerance"
            ),
        },

        "counts": {
            "source_rows": int(
                len(
                    source_dataframe
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

        "exclusion_reason_counts": (
            exclusion_reason_counts
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
            "Sz": (
                numeric_summary(
                    Y[
                        :,
                        0,
                    ]
                )
            ),

            "Lz": (
                numeric_summary(
                    Y[
                        :,
                        1,
                    ]
                )
            ),
        },

        "software": software_versions(
            {
                "pandas": (
                    pd.__version__
                )
            }
        ),
    }

    write_json(
        summary_file,
        summary,
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print("=" * 80)

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
        "Original samples   : "
        f"{len(source_dataframe)}"
    )

    print(
        "Accepted samples   : "
        f"{len(accepted_metadata)}"
    )

    print(
        "Excluded samples   : "
        f"{len(excluded_metadata)}"
    )

    print(
        f"X shape            : {X.shape}"
    )

    print(
        f"Y shape            : {Y.shape}"
    )

    print(
        "Dataset             : "
        f"{safe_public_path(dataset_file, repository_root)}"
    )

    print(
        "Metadata            : "
        f"{safe_public_path(metadata_file, repository_root)}"
    )

    print(
        "Accepted records    : "
        f"{safe_public_path(accepted_file, repository_root)}"
    )

    print(
        "Excluded records    : "
        f"{safe_public_path(excluded_file, repository_root)}"
    )

    print(
        "Summary             : "
        f"{safe_public_path(summary_file, repository_root)}"
    )

    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
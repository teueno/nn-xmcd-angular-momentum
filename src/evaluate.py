#!/usr/bin/env python3

"""Train MultiOutputNN reproducibly under a fixed external field.

Expected neural-network inputs
------------------------------
XAS:
    (I_r + I_l) / 2

XMCD:
    I_r - I_l

The script validates the spectrum definitions stored in the
preprocessed dataset before training.

The fixed external magnetic field is validated by data_loader.py
and stored in checkpoints and training summaries. The external
field is metadata and is not appended to the XAS/XMCD input.
"""

import argparse
import copy
import csv
import os
import random
import shutil
import time
from numbers import Integral
from numbers import Real
from pathlib import Path

import numpy as np
import torch

try:
    from .data_loader import (
        build_dataloaders,
        dataloader_summary,
    )

    from .model import build_model

    from .utils import (
        check_output_files,
        ensure_directory,
        ensure_finite,
        get_repository_root,
        load_json,
        public_path,
        resolve_path,
        select_torch_device,
        set_random_seeds,
        sha256_file,
        software_versions,
        utc_now_iso,
        write_json,
    )

except ImportError:
    from data_loader import (
        build_dataloaders,
        dataloader_summary,
    )

    from model import build_model

    from utils import (
        check_output_files,
        ensure_directory,
        ensure_finite,
        get_repository_root,
        load_json,
        public_path,
        resolve_path,
        select_torch_device,
        set_random_seeds,
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
            "Train the XAS/XMCD multi-output neural network "
            "with spectrum-definition and external-field "
            "validation."
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
# General utilities
# ============================================================

def safe_public_path(
    path,
    repository_root,
):
    """Return a public-safe path."""

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


def public_config_copy(
    config,
):
    """Return a deep copy of the configuration."""

    return copy.deepcopy(
        config
    )


def normalize_definition(
    value,
):
    """Normalize whitespace in a definition string."""

    return " ".join(
        str(
            value
        ).strip().split()
    )


def read_scalar_string(
    name,
    values,
):
    """Read one nonempty scalar string from an array."""

    array = np.asarray(
        values
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
    values,
):
    """Read one scalar Boolean from an array."""

    array = np.asarray(
        values
    ).reshape(
        -1
    )

    if array.size != 1:
        raise ValueError(
            f"{name} must contain exactly one value"
        )

    value = array[
        0
    ]

    if isinstance(
        value,
        (
            bool,
            np.bool_,
        ),
    ):
        return bool(
            value
        )

    if (
        isinstance(
            value,
            (
                int,
                np.integer,
            ),
        )
        and int(
            value
        ) in (
            0,
            1,
        )
    ):
        return bool(
            value
        )

    raise TypeError(
        f"{name} must contain a Boolean value"
    )


def read_scalar_integer(
    name,
    values,
):
    """Read one positive scalar integer from an array."""

    array = np.asarray(
        values
    ).reshape(
        -1
    )

    if array.size != 1:
        raise ValueError(
            f"{name} must contain exactly one value"
        )

    value = int(
        array[
            0
        ]
    )

    if value <= 0:
        raise ValueError(
            f"{name} must be positive"
        )

    return value


def read_scalar_float(
    name,
    values,
):
    """Read one finite scalar float from an array."""

    array = np.asarray(
        values
    ).reshape(
        -1
    )

    if array.size != 1:
        raise ValueError(
            f"{name} must contain exactly one value"
        )

    value = float(
        array[
            0
        ]
    )

    if not np.isfinite(
        value
    ):
        raise ValueError(
            f"{name} must be finite"
        )

    return value


def normalize_direction(
    name,
    direction,
):
    """Validate and normalize a three-component direction."""

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
            f"{name} must contain exactly three values"
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


# ============================================================
# Numeric validation
# ============================================================

def validate_positive_integer(
    name,
    value,
):
    if (
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            Integral,
        )
    ):
        raise TypeError(
            f"{name} must be an integer, not bool"
        )

    value = int(
        value
    )

    if value <= 0:
        raise ValueError(
            f"{name} must be positive"
        )

    return value


def validate_nonnegative_integer(
    name,
    value,
):
    if (
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            Integral,
        )
    ):
        raise TypeError(
            f"{name} must be an integer, not bool"
        )

    value = int(
        value
    )

    if value < 0:
        raise ValueError(
            f"{name} must be nonnegative"
        )

    return value


def validate_positive_float(
    name,
    value,
):
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
            f"{name} must be numeric, not bool"
        )

    value = float(
        value
    )

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
            f"{name} must be numeric, not bool"
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


def validate_probability_open(
    name,
    value,
):
    value = validate_positive_float(
        name,
        value,
    )

    if value >= 1.0:
        raise ValueError(
            f"{name} must be smaller than 1.0"
        )

    return value


# ============================================================
# Spectrum configuration
# ============================================================

def validate_spectrum_config(
    config,
):
    """Validate the XAS/XMCD settings in the JSON config."""

    if "spectrum" not in config:
        raise ValueError(
            "Config must contain the spectrum section"
        )

    spectrum = config[
        "spectrum"
    ]

    if not isinstance(
        spectrum,
        dict,
    ):
        raise TypeError(
            "spectrum must be a dictionary"
        )

    required_settings = {
        "xas_right_suffix",
        "xas_left_suffix",
        "xmcd_suffix",
    }

    missing_settings = (
        required_settings
        - set(
            spectrum
        )
    )

    if missing_settings:
        raise ValueError(
            "Missing spectrum settings: "
            f"{sorted(missing_settings)}"
        )

    if "xas_suffix" in spectrum:
        raise ValueError(
            "Obsolete spectrum.xas_suffix is present. "
            "Use xas_right_suffix and xas_left_suffix."
        )

    xas_right_suffix = str(
        spectrum[
            "xas_right_suffix"
        ]
    ).strip()

    xas_left_suffix = str(
        spectrum[
            "xas_left_suffix"
        ]
    ).strip()

    xmcd_suffix = str(
        spectrum[
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
            "must differ"
        )

    return {
        "xas": EXPECTED_XAS_DEFINITION,
        "xmcd": EXPECTED_XMCD_DEFINITION,
        "xas_definition": EXPECTED_XAS_DEFINITION,
        "xmcd_definition": EXPECTED_XMCD_DEFINITION,
        "xas_right_suffix": xas_right_suffix,
        "xas_left_suffix": xas_left_suffix,
        "xmcd_suffix": xmcd_suffix,
        "external_field_is_model_input": (
            EXPECTED_EXTERNAL_FIELD_IS_MODEL_INPUT
        ),
    }


# ============================================================
# External-field configuration
# ============================================================

def validate_external_field(
    config,
):
    """Validate and normalize the external-field config."""

    if "external_magnetic_field" not in config:
        raise ValueError(
            "Config must contain external_magnetic_field"
        )

    field = config[
        "external_magnetic_field"
    ]

    if not isinstance(
        field,
        dict,
    ):
        raise TypeError(
            "external_magnetic_field must be a dictionary"
        )

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
            "external_magnetic_field.enabled must "
            "be true or false"
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

    direction = normalize_direction(
        "external_magnetic_field.direction",
        field[
            "direction"
        ],
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

    return {
        "enabled": field[
            "enabled"
        ],
        "magnitude_t": magnitude,
        "direction": direction.tolist(),
        "tolerance_t": tolerance,
        "is_model_input": False,
    }


# ============================================================
# Main configuration validation
# ============================================================

def validate_config(
    config,
):
    """Validate the full training configuration."""

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
        "model",
        "training",
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

    system_name = str(
        config[
            "system"
        ].get(
            "name",
            "",
        )
    ).strip()

    if not system_name:
        raise ValueError(
            "system.name must not be empty"
        )

    path_keys = {
        "repository_root",
        "preprocessed_dataset_npz",
        "model_output_directory",
        "result_output_directory",
    }

    missing_paths = (
        path_keys
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

    training = config[
        "training"
    ]

    if not isinstance(
        training,
        dict,
    ):
        raise TypeError(
            "training must be a dictionary"
        )

    for subsection in (
        "optimizer",
        "loss",
        "data_loader",
        "early_stopping",
        "checkpoint",
    ):
        if subsection not in training:
            raise ValueError(
                "Missing training subsection: "
                f"{subsection}"
            )

    spectrum_definitions = validate_spectrum_config(
        config
    )

    external_field = validate_external_field(
        config
    )

    return (
        system_name,
        spectrum_definitions,
        external_field,
    )


# ============================================================
# Preprocessed spectrum metadata
# ============================================================

def load_preprocessed_metadata(
    dataset_file,
    expected_spectrum,
    expected_field,
):
    """Load and validate metadata from the preprocessed NPZ."""

    with np.load(
        dataset_file,
        allow_pickle=False,
    ) as dataset:
        required_arrays = {
            "X_train",
            "X_validation",
            "X_test",
            "target_names",
            "xas_definition",
            "xmcd_definition",
            "n_xas",
            "n_xmcd",
            "external_field_enabled",
            "external_field_magnitude_t",
            "external_field_direction",
            "external_field_is_model_input",
        }

        missing_arrays = (
            required_arrays
            - set(
                dataset.files
            )
        )

        if missing_arrays:
            raise ValueError(
                "Preprocessed dataset is missing arrays: "
                f"{sorted(missing_arrays)}"
            )

        x_train_shape = np.asarray(
            dataset[
                "X_train"
            ]
        ).shape

        x_validation_shape = np.asarray(
            dataset[
                "X_validation"
            ]
        ).shape

        x_test_shape = np.asarray(
            dataset[
                "X_test"
            ]
        ).shape

        target_names = dataset[
            "target_names"
        ].astype(
            str
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

        n_xas = read_scalar_integer(
            "n_xas",
            dataset[
                "n_xas"
            ],
        )

        n_xmcd = read_scalar_integer(
            "n_xmcd",
            dataset[
                "n_xmcd"
            ],
        )

        field_is_model_input = read_scalar_boolean(
            "external_field_is_model_input",
            dataset[
                "external_field_is_model_input"
            ],
        )

        field_enabled = read_scalar_boolean(
            "external_field_enabled",
            dataset[
                "external_field_enabled"
            ],
        )

        field_magnitude = read_scalar_float(
            "external_field_magnitude_t",
            dataset[
                "external_field_magnitude_t"
            ],
        )

        field_direction = normalize_direction(
            "preprocessed external_field_direction",
            dataset[
                "external_field_direction"
            ],
        )

    for name, shape in (
        (
            "X_train",
            x_train_shape,
        ),
        (
            "X_validation",
            x_validation_shape,
        ),
        (
            "X_test",
            x_test_shape,
        ),
    ):
        if len(
            shape
        ) != 2:
            raise ValueError(
                f"{name} must be two-dimensional"
            )

        if shape[
            0
        ] <= 0:
            raise ValueError(
                f"{name} must contain samples"
            )

        if shape[
            1
        ] != (
            n_xas
            + n_xmcd
        ):
            raise ValueError(
                f"{name} feature count differs from "
                "n_xas + n_xmcd"
            )

    if target_names.tolist() != [
        "Sz",
        "Lz",
    ]:
        raise ValueError(
            "Target order must be ['Sz', 'Lz']; found "
            f"{target_names.tolist()}"
        )

    if normalize_definition(
        xas_definition
    ) != normalize_definition(
        expected_spectrum[
            "xas_definition"
        ]
    ):
        raise ValueError(
            "Preprocessed XAS definition differs from "
            "the expected definition: "
            f"{xas_definition!r}"
        )

    if normalize_definition(
        xmcd_definition
    ) != normalize_definition(
        expected_spectrum[
            "xmcd_definition"
        ]
    ):
        raise ValueError(
            "Preprocessed XMCD definition differs from "
            "the expected definition: "
            f"{xmcd_definition!r}"
        )

    if field_is_model_input:
        raise ValueError(
            "Preprocessed dataset incorrectly treats "
            "the external field as a model input"
        )

    if field_enabled != expected_field[
        "enabled"
    ]:
        raise ValueError(
            "External-field enabled state differs between "
            "config and preprocessed dataset"
        )

    if not np.isclose(
        field_magnitude,
        expected_field[
            "magnitude_t"
        ],
        rtol=0.0,
        atol=expected_field[
            "tolerance_t"
        ],
    ):
        raise ValueError(
            "External-field magnitude differs between "
            "config and preprocessed dataset"
        )

    if not np.allclose(
        field_direction,
        expected_field[
            "direction"
        ],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            "External-field direction differs between "
            "config and preprocessed dataset"
        )

    return {
        "xas": xas_definition,
        "xmcd": xmcd_definition,
        "xas_definition": xas_definition,
        "xmcd_definition": xmcd_definition,
        "xas_right_suffix": expected_spectrum[
            "xas_right_suffix"
        ],
        "xas_left_suffix": expected_spectrum[
            "xas_left_suffix"
        ],
        "xmcd_suffix": expected_spectrum[
            "xmcd_suffix"
        ],
        "n_xas": n_xas,
        "n_xmcd": n_xmcd,
        "total_input_features": (
            n_xas
            + n_xmcd
        ),
        "feature_order": (
            "[XAS channels, XMCD channels]"
        ),
        "external_field_is_model_input": (
            field_is_model_input
        ),
        "target_order": target_names.tolist(),
    }


# ============================================================
# Torch dtype and CPU threads
# ============================================================

def select_torch_dtype(
    dtype_name,
):
    dtype_name = str(
        dtype_name
    ).lower()

    mapping = {
        "float32": torch.float32,
        "float64": torch.float64,
    }

    if dtype_name not in mapping:
        raise ValueError(
            "training.dtype must be 'float32' or 'float64'"
        )

    return (
        dtype_name,
        mapping[
            dtype_name
        ],
    )


def detect_available_cpu_count():
    try:
        count = len(
            os.sched_getaffinity(
                0
            )
        )

    except (
        AttributeError,
        OSError,
    ):
        count = os.cpu_count()

    return max(
        1,
        int(
            count
            or 1
        ),
    )


def parse_thread_setting(
    name,
    value,
    automatic_value,
):
    if (
        value is None
        or str(
            value
        ).strip().lower() == "auto"
    ):
        return int(
            automatic_value
        )

    if isinstance(
        value,
        bool,
    ):
        raise TypeError(
            f"{name} must be a positive integer or 'auto'"
        )

    value = int(
        value
    )

    if value <= 0:
        raise ValueError(
            f"{name} must be positive"
        )

    return value


def configure_cpu_threads(
    training_config,
    device,
):
    cpu_config = training_config.get(
        "cpu_threads",
        {},
    )

    if not isinstance(
        cpu_config,
        dict,
    ):
        raise TypeError(
            "training.cpu_threads must be a dictionary"
        )

    available = detect_available_cpu_count()

    requested_num = cpu_config.get(
        "num_threads",
        "auto",
    )

    requested_interop = cpu_config.get(
        "interop_threads",
        "auto",
    )

    num_threads = parse_thread_setting(
        "cpu_threads.num_threads",
        requested_num,
        available,
    )

    interop_threads = parse_thread_setting(
        "cpu_threads.interop_threads",
        requested_interop,
        max(
            1,
            min(
                8,
                available // 4,
            ),
        ),
    )

    warn_on_oversubscription = cpu_config.get(
        "warn_on_oversubscription",
        True,
    )

    if not isinstance(
        warn_on_oversubscription,
        bool,
    ):
        raise TypeError(
            "cpu_threads.warn_on_oversubscription "
            "must be true or false"
        )

    if warn_on_oversubscription:
        if num_threads > available:
            print(
                "Warning: intra-op threads "
                f"{num_threads} exceed CPUs {available}"
            )

        if interop_threads > available:
            print(
                "Warning: inter-op threads "
                f"{interop_threads} exceed CPUs {available}"
            )

    for environment_variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[
            environment_variable
        ] = str(
            num_threads
        )

    torch.set_num_threads(
        num_threads
    )

    try:
        torch.set_num_interop_threads(
            interop_threads
        )

    except RuntimeError as exception:
        raise RuntimeError(
            "Configure inter-op threads before parallel "
            "work starts"
        ) from exception

    return {
        "device": str(
            device
        ),
        "available_logical_cpus": available,
        "requested_num_threads": requested_num,
        "requested_interop_threads": requested_interop,
        "effective_num_threads": int(
            torch.get_num_threads()
        ),
        "effective_interop_threads": int(
            torch.get_num_interop_threads()
        ),
    }


# ============================================================
# Optimizer and loss
# ============================================================

def build_optimizer(
    model,
    config,
):
    if str(
        config.get(
            "name",
            "adam",
        )
    ).lower() != "adam":
        raise ValueError(
            "Only Adam is implemented"
        )

    learning_rate = validate_positive_float(
        "optimizer.learning_rate",
        config.get(
            "learning_rate",
            2.0e-5,
        ),
    )

    beta1 = validate_probability_open(
        "optimizer.beta1",
        config.get(
            "beta1",
            0.9,
        ),
    )

    beta2 = validate_probability_open(
        "optimizer.beta2",
        config.get(
            "beta2",
            0.999,
        ),
    )

    epsilon = validate_positive_float(
        "optimizer.epsilon",
        config.get(
            "epsilon",
            1.0e-8,
        ),
    )

    weight_decay = validate_nonnegative_float(
        "optimizer.weight_decay",
        config.get(
            "weight_decay",
            0.0,
        ),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(
            beta1,
            beta2,
        ),
        eps=epsilon,
        weight_decay=weight_decay,
    )

    return (
        optimizer,
        {
            "name": "adam",
            "learning_rate": learning_rate,
            "beta1": beta1,
            "beta2": beta2,
            "epsilon": epsilon,
            "weight_decay": weight_decay,
        },
    )


def build_loss(
    config,
    device,
    dtype,
):
    if str(
        config.get(
            "name",
            "mse",
        )
    ).lower() != "mse":
        raise ValueError(
            "Only MSE loss is implemented"
        )

    loss_space = str(
        config.get(
            "space",
            "standardized_target_space",
        )
    )

    if loss_space != "standardized_target_space":
        raise ValueError(
            "Loss must be evaluated in "
            "standardized_target_space"
        )

    output_reduction = str(
        config.get(
            "output_reduction",
            "mean",
        )
    ).lower()

    batch_reduction = str(
        config.get(
            "batch_reduction",
            "mean",
        )
    ).lower()

    if output_reduction not in {
        "mean",
        "sum",
    }:
        raise ValueError(
            "loss.output_reduction must be 'mean' or 'sum'"
        )

    if batch_reduction not in {
        "mean",
        "sum",
    }:
        raise ValueError(
            "loss.batch_reduction must be 'mean' or 'sum'"
        )

    weights = torch.as_tensor(
        config.get(
            "target_weights",
            [
                1.0,
                1.0,
            ],
        ),
        device=device,
        dtype=dtype,
    )

    if weights.shape != (
        2,
    ):
        raise ValueError(
            "loss.target_weights must contain two values"
        )

    if not torch.isfinite(
        weights
    ).all():
        raise ValueError(
            "loss.target_weights must be finite"
        )

    if torch.any(
        weights < 0.0
    ):
        raise ValueError(
            "loss.target_weights must be nonnegative"
        )

    if weights.sum() <= 0.0:
        raise ValueError(
            "At least one target weight must be positive"
        )

    def per_sample(
        prediction,
        target,
    ):
        weighted_squared_error = (
            (
                prediction
                - target
            ) ** 2
            * weights.reshape(
                1,
                2,
            )
        )

        if output_reduction == "mean":
            return weighted_squared_error.mean(
                dim=1
            )

        return weighted_squared_error.sum(
            dim=1
        )

    def loss_function(
        prediction,
        target,
    ):
        values = per_sample(
            prediction,
            target,
        )

        if batch_reduction == "mean":
            return values.mean()

        return values.sum()

    return (
        loss_function,
        per_sample,
        batch_reduction,
        {
            "name": "mse",
            "space": "standardized_target_space",
            "output_reduction": output_reduction,
            "batch_reduction": batch_reduction,
            "effective_target_weights": (
                weights.detach().cpu().tolist()
            ),
        },
    )


# ============================================================
# Epoch execution
# ============================================================

def run_epoch(
    model,
    loader,
    loss_function,
    per_sample_loss,
    batch_reduction,
    device,
    dtype,
    optimizer=None,
    gradient_clip_norm=None,
):
    training = (
        optimizer is not None
    )

    model.train(
        training
    )

    loss_sum = 0.0
    sample_count = 0

    gradient_norm_sum = 0.0
    gradient_norm_count = 0
    maximum_gradient_norm = None

    context = (
        torch.enable_grad()
        if training
        else torch.inference_mode()
    )

    with context:
        for inputs, targets in loader:
            inputs = inputs.to(
                device=device,
                dtype=dtype,
                non_blocking=True,
            )

            targets = targets.to(
                device=device,
                dtype=dtype,
                non_blocking=True,
            )

            if training:
                optimizer.zero_grad(
                    set_to_none=True
                )

            predictions = model(
                inputs
            )

            if not torch.isfinite(
                predictions
            ).all():
                raise RuntimeError(
                    "Non-finite prediction detected"
                )

            loss = loss_function(
                predictions,
                targets,
            )

            per_sample = per_sample_loss(
                predictions,
                targets,
            )

            if (
                not torch.isfinite(
                    loss
                )
                or not torch.isfinite(
                    per_sample
                ).all()
            ):
                raise RuntimeError(
                    "Non-finite loss detected"
                )

            if training:
                loss.backward()

                if any(
                    parameter.grad is not None
                    and not torch.isfinite(
                        parameter.grad
                    ).all()
                    for parameter in model.parameters()
                ):
                    raise RuntimeError(
                        "Non-finite gradient detected"
                    )

                if gradient_clip_norm is not None:
                    gradient_norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            gradient_clip_norm,
                            error_if_nonfinite=True,
                        ).detach().cpu()
                    )

                    gradient_norm_sum += (
                        gradient_norm
                    )

                    gradient_norm_count += 1

                    maximum_gradient_norm = (
                        gradient_norm
                        if maximum_gradient_norm is None
                        else max(
                            maximum_gradient_norm,
                            gradient_norm,
                        )
                    )

                optimizer.step()

            loss_sum += float(
                per_sample.detach().sum().cpu()
            )

            sample_count += int(
                inputs.shape[
                    0
                ]
            )

    if sample_count == 0:
        raise RuntimeError(
            "DataLoader produced no samples"
        )

    return {
        "mean_per_sample_loss": (
            loss_sum
            / sample_count
        ),

        "processed_samples": sample_count,

        "batch_reduction": batch_reduction,

        "mean_gradient_norm_before_clipping": (
            gradient_norm_sum
            / gradient_norm_count
            if gradient_norm_count
            else None
        ),

        "maximum_gradient_norm_before_clipping": (
            maximum_gradient_norm
        ),
    }


# ============================================================
# Checkpoint handling
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    validation_loss,
    config,
    repository_root,
    config_path,
    dataset_file,
    cpu_threads,
    external_field,
    spectrum_definitions,
):
    """Save a reproducible training checkpoint."""

    model_configuration = dict(
        model.configuration()
    )

    model_configuration[
        "external_field_is_model_input"
    ] = False

    model_configuration[
        "input_feature_count"
    ] = int(
        model.n_in
    )

    checkpoint = {
        "checkpoint_format_version": 3,

        "created_at_utc": utc_now_iso(),

        "epoch": int(
            epoch
        ),

        "validation_loss": float(
            validation_loss
        ),

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict(),

        "model_configuration": model_configuration,

        "training_configuration": public_config_copy(
            config
        ),

        "spectrum_definitions": {
            "xas": spectrum_definitions[
                "xas_definition"
            ],

            "xmcd": spectrum_definitions[
                "xmcd_definition"
            ],

            "xas_definition": spectrum_definitions[
                "xas_definition"
            ],

            "xmcd_definition": spectrum_definitions[
                "xmcd_definition"
            ],

            "xas_right_suffix": spectrum_definitions[
                "xas_right_suffix"
            ],

            "xas_left_suffix": spectrum_definitions[
                "xas_left_suffix"
            ],

            "xmcd_suffix": spectrum_definitions[
                "xmcd_suffix"
            ],

            "n_xas": int(
                spectrum_definitions[
                    "n_xas"
                ]
            ),

            "n_xmcd": int(
                spectrum_definitions[
                    "n_xmcd"
                ]
            ),

            "total_input_features": int(
                spectrum_definitions[
                    "total_input_features"
                ]
            ),

            "feature_order": spectrum_definitions[
                "feature_order"
            ],

            "target_order": spectrum_definitions[
                "target_order"
            ],

            "external_field_is_model_input": False,
        },

        "external_magnetic_field": external_field,

        "cpu_threads": cpu_threads,

        "source_files": {
            "config": safe_public_path(
                config_path,
                repository_root,
            ),

            "preprocessed_dataset": safe_public_path(
                dataset_file,
                repository_root,
            ),
        },

        "source_sha256": {
            "config": sha256_file(
                config_path
            ),

            "preprocessed_dataset": sha256_file(
                dataset_file
            ),
        },

        "torch_rng_state": torch.get_rng_state(),

        "numpy_random_state": np.random.get_state(),

        "python_random_state": random.getstate(),
    }

    if torch.cuda.is_available():
        checkpoint[
            "cuda_rng_state_all"
        ] = torch.cuda.get_rng_state_all()

    torch.save(
        checkpoint,
        path,
    )


def load_checkpoint(
    path,
    device,
):
    """Load a checkpoint across supported PyTorch versions."""

    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )

    except TypeError:
        return torch.load(
            path,
            map_location=device,
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

    (
        system_name,
        configured_spectrum,
        configured_field,
    ) = validate_config(
        config
    )

    config_directory = (
        config_path.parent
    )

    repository_root = get_repository_root(
        config,
        config_path,
    )

    training = config[
        "training"
    ]

    paths = config[
        "paths"
    ]

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    dataset_file = resolve_path(
        paths[
            "preprocessed_dataset_npz"
        ],
        config_directory,
    )

    model_directory = ensure_directory(
        resolve_path(
            paths[
                "model_output_directory"
            ],
            config_directory,
        )
    )

    result_directory = ensure_directory(
        resolve_path(
            paths[
                "result_output_directory"
            ],
            config_directory,
        )
    )

    if not dataset_file.is_file():
        raise FileNotFoundError(
            "Preprocessed dataset was not found: "
            f"{safe_public_path(dataset_file, repository_root)}"
        )

    # --------------------------------------------------------
    # Validate preprocessed input definitions
    # --------------------------------------------------------

    spectrum_definitions = load_preprocessed_metadata(
        dataset_file=dataset_file,
        expected_spectrum=configured_spectrum,
        expected_field=configured_field,
    )

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    checkpoint_config = training[
        "checkpoint"
    ]

    checkpoint_filename = str(
        checkpoint_config.get(
            "filename",
            "best_model.pt",
        )
    )

    if Path(
        checkpoint_filename
    ).name != checkpoint_filename:
        raise ValueError(
            "checkpoint.filename must not contain directories"
        )

    best_checkpoint = (
        model_directory
        / checkpoint_filename
    )

    final_checkpoint = (
        model_directory
        / "final_model.pt"
    )

    selected_checkpoint = (
        model_directory
        / "selected_model.pt"
    )

    history_file = (
        result_directory
        / "training_history.csv"
    )

    summary_file = (
        result_directory
        / "training_summary.json"
    )

    config_copy = (
        model_directory
        / "training_config.json"
    )

    check_output_files(
        [
            best_checkpoint,
            final_checkpoint,
            selected_checkpoint,
            history_file,
            summary_file,
            config_copy,
        ],
        overwrite=arguments.overwrite,
        label="Training output files",
    )

    # --------------------------------------------------------
    # Reproducibility settings
    # --------------------------------------------------------

    random_seed = validate_nonnegative_integer(
        "training.random_seed",
        training.get(
            "random_seed",
            1234,
        ),
    )

    deterministic = training.get(
        "deterministic_algorithms",
        True,
    )

    if not isinstance(
        deterministic,
        bool,
    ):
        raise TypeError(
            "training.deterministic_algorithms must be "
            "true or false"
        )

    device = select_torch_device(
        training.get(
            "device",
            "auto",
        )
    )

    cpu_threads = configure_cpu_threads(
        training,
        device,
    )

    set_random_seeds(
        random_seed,
        deterministic_algorithms=deterministic,
    )

    (
        dtype_name,
        dtype,
    ) = select_torch_dtype(
        training.get(
            "dtype",
            "float32",
        )
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    loader_bundle = build_dataloaders(
        config,
        config_directory=config_directory,
    )

    dataset = loader_bundle[
        "dataset"
    ]

    required_dataset_keys = {
        "X_train",
        "Y_train",
        "X_validation",
        "Y_validation",
        "X_test",
        "Y_test",
        "target_names",
    }

    missing_dataset_keys = (
        required_dataset_keys
        - set(
            dataset
        )
    )

    if missing_dataset_keys:
        raise ValueError(
            "DataLoader dataset is missing keys: "
            f"{sorted(missing_dataset_keys)}"
        )

    input_dimension = int(
        dataset[
            "X_train"
        ].shape[
            1
        ]
    )

    if input_dimension != spectrum_definitions[
        "total_input_features"
    ]:
        raise RuntimeError(
            "DataLoader input dimension differs from "
            "n_xas + n_xmcd"
        )

    target_names = dataset[
        "target_names"
    ].astype(
        str
    )

    if target_names.tolist() != spectrum_definitions[
        "target_order"
    ]:
        raise RuntimeError(
            "DataLoader target order differs from "
            "preprocessed metadata"
        )

    field_from_loader = {
        "enabled": bool(
            loader_bundle[
                "external_field_enabled"
            ]
        ),

        "magnitude_t": float(
            loader_bundle[
                "external_field_magnitude_t"
            ]
        ),

        "direction": normalize_direction(
            "DataLoader external-field direction",
            loader_bundle[
                "external_field_direction"
            ],
        ).tolist(),

        "tolerance_t": configured_field[
            "tolerance_t"
        ],

        "is_model_input": False,
    }

    if field_from_loader[
        "enabled"
    ] != configured_field[
        "enabled"
    ]:
        raise RuntimeError(
            "External-field enabled state is inconsistent"
        )

    if not np.isclose(
        field_from_loader[
            "magnitude_t"
        ],
        configured_field[
            "magnitude_t"
        ],
        rtol=0.0,
        atol=configured_field[
            "tolerance_t"
        ],
    ):
        raise RuntimeError(
            "External-field magnitude is inconsistent"
        )

    if not np.allclose(
        field_from_loader[
            "direction"
        ],
        configured_field[
            "direction"
        ],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            "External-field direction is inconsistent"
        )

    if str(
        loader_bundle[
            "device"
        ]
    ) != str(
        device
    ):
        raise RuntimeError(
            "DataLoader and training devices are inconsistent"
        )

    # --------------------------------------------------------
    # Model, optimizer, and loss
    # --------------------------------------------------------

    model = build_model(
        input_dimension,
        config[
            "model"
        ],
    ).to(
        device=device,
        dtype=dtype,
    )

    if int(
        model.n_in
    ) != spectrum_definitions[
        "total_input_features"
    ]:
        raise RuntimeError(
            "Model input dimension differs from "
            "n_xas + n_xmcd"
        )

    (
        optimizer,
        optimizer_effective,
    ) = build_optimizer(
        model,
        training[
            "optimizer"
        ],
    )

    (
        loss_function,
        per_sample_loss,
        batch_reduction,
        loss_effective,
    ) = build_loss(
        training[
            "loss"
        ],
        device,
        dtype,
    )

    maximum_epochs = validate_positive_integer(
        "training.maximum_epochs",
        training.get(
            "maximum_epochs",
            300,
        ),
    )

    gradient_clip_setting = training.get(
        "gradient_clip_norm",
        1.0,
    )

    gradient_clip_norm = (
        None
        if gradient_clip_setting is None
        else validate_positive_float(
            "training.gradient_clip_norm",
            gradient_clip_setting,
        )
    )

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    early = training[
        "early_stopping"
    ]

    early_enabled = early.get(
        "enabled",
        True,
    )

    if not isinstance(
        early_enabled,
        bool,
    ):
        raise TypeError(
            "early_stopping.enabled must be true or false"
        )

    patience = validate_positive_integer(
        "early_stopping.patience",
        early.get(
            "patience",
            30,
        ),
    )

    minimum_delta = validate_nonnegative_float(
        "early_stopping.minimum_delta",
        early.get(
            "minimum_delta",
            0.0,
        ),
    )

    restore_best = early.get(
        "restore_best_weights",
        True,
    )

    save_best = checkpoint_config.get(
        "save_best",
        True,
    )

    if not isinstance(
        restore_best,
        bool,
    ):
        raise TypeError(
            "early_stopping.restore_best_weights "
            "must be true or false"
        )

    if not isinstance(
        save_best,
        bool,
    ):
        raise TypeError(
            "checkpoint.save_best must be true or false"
        )

    if str(
        early.get(
            "monitor",
            "validation_loss",
        )
    ) != "validation_loss":
        raise ValueError(
            "Only validation_loss monitoring is supported"
        )

    if str(
        early.get(
            "mode",
            "min",
        )
    ) != "min":
        raise ValueError(
            "Only early-stopping mode 'min' is supported"
        )

    if (
        restore_best
        and not save_best
    ):
        raise ValueError(
            "restore_best_weights=true requires "
            "save_best=true"
        )

    # --------------------------------------------------------
    # Training state
    # --------------------------------------------------------

    history = []

    best_loss = float(
        "inf"
    )

    best_epoch = None

    patience_counter = 0

    stopped_early = False

    start_time = time.perf_counter()

    print(
        "=" * 80
    )

    print(
        f"System             : {system_name}"
    )

    print(
        f"Device             : {device}"
    )

    print(
        f"Dtype              : {dtype_name}"
    )

    print(
        "XAS definition     : "
        f"{spectrum_definitions['xas_definition']}"
    )

    print(
        "XMCD definition    : "
        f"{spectrum_definitions['xmcd_definition']}"
    )

    print(
        "XAS features       : "
        f"{spectrum_definitions['n_xas']}"
    )

    print(
        "XMCD features      : "
        f"{spectrum_definitions['n_xmcd']}"
    )

    print(
        "External field     : "
        f"{field_from_loader['magnitude_t']:.6f} T"
    )

    print(
        "Field direction    : "
        f"{field_from_loader['direction']}"
    )

    print(
        "Field is NN input  : False"
    )

    print(
        f"Input dimension    : {model.n_in}"
    )

    print(
        f"Maximum epochs     : {maximum_epochs}"
    )

    print(
        "=" * 80
    )

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(
        maximum_epochs
    ):
        train_result = run_epoch(
            model=model,
            loader=loader_bundle[
                "train_loader"
            ],
            loss_function=loss_function,
            per_sample_loss=per_sample_loss,
            batch_reduction=batch_reduction,
            device=device,
            dtype=dtype,
            optimizer=optimizer,
            gradient_clip_norm=gradient_clip_norm,
        )

        validation_result = run_epoch(
            model=model,
            loader=loader_bundle[
                "validation_loader"
            ],
            loss_function=loss_function,
            per_sample_loss=per_sample_loss,
            batch_reduction=batch_reduction,
            device=device,
            dtype=dtype,
        )

        train_loss = train_result[
            "mean_per_sample_loss"
        ]

        validation_loss = validation_result[
            "mean_per_sample_loss"
        ]

        improved = (
            best_epoch is None
            or validation_loss
            < best_loss
            - minimum_delta
        )

        if improved:
            best_loss = validation_loss

            best_epoch = epoch

            patience_counter = 0

            if save_best:
                save_checkpoint(
                    path=best_checkpoint,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    validation_loss=validation_loss,
                    config=config,
                    repository_root=repository_root,
                    config_path=config_path,
                    dataset_file=dataset_file,
                    cpu_threads=cpu_threads,
                    external_field=field_from_loader,
                    spectrum_definitions=spectrum_definitions,
                )

        else:
            patience_counter += 1

        history.append(
            {
                "epoch": epoch,

                "train_loss": train_loss,

                "validation_loss": validation_loss,

                "train_samples_seen": train_result[
                    "processed_samples"
                ],

                "validation_samples_seen": validation_result[
                    "processed_samples"
                ],

                "mean_gradient_norm_before_clipping": (
                    train_result[
                        "mean_gradient_norm_before_clipping"
                    ]
                ),

                "maximum_gradient_norm_before_clipping": (
                    train_result[
                        "maximum_gradient_norm_before_clipping"
                    ]
                ),

                "best_validation_loss_so_far": best_loss,

                "best_epoch_so_far": best_epoch,

                "patience_counter": patience_counter,

                "improved": improved,
            }
        )

        if (
            epoch % 20 == 0
            or epoch == maximum_epochs - 1
            or improved
        ):
            print(
                f"Epoch {epoch:4d} | "
                f"train={train_loss:.6e} | "
                f"val={validation_loss:.6e} | "
                f"best={best_loss:.6e}"
            )

        if (
            early_enabled
            and patience_counter >= patience
        ):
            stopped_early = True

            print(
                "Early stopping triggered at "
                f"epoch {epoch}"
            )

            break

    # --------------------------------------------------------
    # Final checkpoint selection
    # --------------------------------------------------------

    elapsed_seconds = float(
        time.perf_counter()
        - start_time
    )

    if not history:
        raise RuntimeError(
            "Training completed without epochs"
        )

    save_checkpoint(
        path=final_checkpoint,
        model=model,
        optimizer=optimizer,
        epoch=history[
            -1
        ][
            "epoch"
        ],
        validation_loss=history[
            -1
        ][
            "validation_loss"
        ],
        config=config,
        repository_root=repository_root,
        config_path=config_path,
        dataset_file=dataset_file,
        cpu_threads=cpu_threads,
        external_field=field_from_loader,
        spectrum_definitions=spectrum_definitions,
    )

    if restore_best:
        if not best_checkpoint.is_file():
            raise RuntimeError(
                "No best checkpoint exists"
            )

        best_checkpoint_data = load_checkpoint(
            best_checkpoint,
            device,
        )

        model.load_state_dict(
            best_checkpoint_data[
                "model_state_dict"
            ],
            strict=True,
        )

        shutil.copy2(
            best_checkpoint,
            selected_checkpoint,
        )

        selected_source = "best"

    else:
        shutil.copy2(
            final_checkpoint,
            selected_checkpoint,
        )

        selected_source = "final"

    # --------------------------------------------------------
    # Save history and configuration
    # --------------------------------------------------------

    with history_file.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                history[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            history
        )

    write_json(
        config_copy,
        public_config_copy(
            config
        ),
    )

    # --------------------------------------------------------
    # Training summary
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

        "preprocessed_dataset_npz": safe_public_path(
            dataset_file,
            repository_root,
        ),

        "preprocessed_dataset_sha256": sha256_file(
            dataset_file
        ),

        "spectrum_definitions": {
            "xas": spectrum_definitions[
                "xas_definition"
            ],

            "xmcd": spectrum_definitions[
                "xmcd_definition"
            ],

            "xas_right_suffix": spectrum_definitions[
                "xas_right_suffix"
            ],

            "xas_left_suffix": spectrum_definitions[
                "xas_left_suffix"
            ],

            "xmcd_suffix": spectrum_definitions[
                "xmcd_suffix"
            ],

            "n_xas": spectrum_definitions[
                "n_xas"
            ],

            "n_xmcd": spectrum_definitions[
                "n_xmcd"
            ],

            "total_input_features": spectrum_definitions[
                "total_input_features"
            ],

            "feature_order": spectrum_definitions[
                "feature_order"
            ],

            "target_order": spectrum_definitions[
                "target_order"
            ],

            "external_field_is_model_input": False,

            "validated_against_preprocessed_dataset": True,
        },

        "external_magnetic_field": (
            field_from_loader
        ),

        "device": str(
            device
        ),

        "dtype": dtype_name,

        "cpu_threads": cpu_threads,

        "random_seed": random_seed,

        "deterministic_algorithms": deterministic,

        "model": model.configuration(),

        "parameter_counts": model.parameter_counts(),

        "data_loader": dataloader_summary(
            loader_bundle
        ),

        "optimizer": optimizer_effective,

        "loss": loss_effective,

        "training": {
            "maximum_epochs": maximum_epochs,

            "epochs_completed": len(
                history
            ),

            "last_epoch": history[
                -1
            ][
                "epoch"
            ],

            "gradient_clip_norm": gradient_clip_norm,

            "stopped_early": stopped_early,

            "best_epoch": best_epoch,

            "best_validation_loss": best_loss,

            "elapsed_seconds": elapsed_seconds,
        },

        "early_stopping": {
            "enabled": early_enabled,

            "monitor": "validation_loss",

            "mode": "min",

            "patience": patience,

            "minimum_delta": minimum_delta,

            "restore_best_weights": restore_best,
        },

        "checkpoints": {
            "best": (
                safe_public_path(
                    best_checkpoint,
                    repository_root,
                )
                if best_checkpoint.is_file()
                else None
            ),

            "final": safe_public_path(
                final_checkpoint,
                repository_root,
            ),

            "selected_for_evaluation_and_figures": (
                safe_public_path(
                    selected_checkpoint,
                    repository_root,
                )
            ),

            "selected_source": selected_source,

            "selected_sha256": sha256_file(
                selected_checkpoint
            ),
        },

        "files": {
            "history_csv": safe_public_path(
                history_file,
                repository_root,
            ),

            "summary_json": safe_public_path(
                summary_file,
                repository_root,
            ),

            "training_config_copy": safe_public_path(
                config_copy,
                repository_root,
            ),
        },

        "sha256": {
            "history_csv": sha256_file(
                history_file
            ),

            "training_config_copy": sha256_file(
                config_copy
            ),

            "final_checkpoint": sha256_file(
                final_checkpoint
            ),

            "selected_checkpoint": sha256_file(
                selected_checkpoint
            ),
        },

        "software": software_versions(
            {
                "torch": torch.__version__,

                "cuda_version": torch.version.cuda,
            }
        ),
    }

    write_json(
        summary_file,
        summary,
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print(
        "=" * 80
    )

    print(
        "Training completed"
    )

    print(
        "XAS definition      : "
        f"{spectrum_definitions['xas_definition']}"
    )

    print(
        "XMCD definition     : "
        f"{spectrum_definitions['xmcd_definition']}"
    )

    print(
        "Field is NN input   : False"
    )

    print(
        f"Epochs completed    : {len(history)}"
    )

    print(
        f"Best epoch          : {best_epoch}"
    )

    print(
        "Best validation loss: "
        f"{best_loss:.8e}"
    )

    print(
        "Selected checkpoint : "
        f"{safe_public_path(selected_checkpoint, repository_root)}"
    )

    print(
        "History             : "
        f"{safe_public_path(history_file, repository_root)}"
    )

    print(
        "Summary             : "
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
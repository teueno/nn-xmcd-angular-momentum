#!/usr/bin/env python3

"""
Evaluate the selected checkpoint on the complete saved test subset.

The model predicts standardized target values. Predictions are converted
back to physical units using the training-subset statistics saved by
preprocess.py.

Outputs
-------
predictions.csv
    Every test target/prediction pair in standardized and physical units.

metrics.json
    Metrics calculated from exactly the rows saved in predictions.csv.

residual_summary.csv
    Physical-unit regression and residual metrics, including SSE, SST,
    RMSE, NRMSE, bias, R2, and Pearson correlation.

evaluation_summary.json
    Checkpoint, input files, hashes, software versions, and inference details.

Privacy
-------
Internal calculations use absolute paths. Paths written to public output
files and displayed in normal console output are converted to paths relative
to the repository root.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# Support both:
#
#   python src/evaluate.py
#
# and:
#
#   python -m src.evaluate

try:
    from .data_loader import (
        build_dataloaders,
    )

    from .metrics import (
        multioutput_regression_metrics,
    )

    from .model import (
        build_model,
    )

    from .utils import (
        check_output_files,
        ensure_directory,
        ensure_finite,
        get_repository_root,
        load_json,
        public_path,
        resolve_path,
        select_torch_device,
        sha256_file,
        software_versions,
        utc_now_iso,
        write_json,
    )

except ImportError:
    from data_loader import (
        build_dataloaders,
    )

    from metrics import (
        multioutput_regression_metrics,
    )

    from model import (
        build_model,
    )

    from utils import (
        check_output_files,
        ensure_directory,
        ensure_finite,
        get_repository_root,
        load_json,
        public_path,
        resolve_path,
        select_torch_device,
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
            "Evaluate the selected model "
            "on the complete saved test subset."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Training JSON configuration file.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing evaluation outputs.",
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

    A path inside the repository is represented relative to the
    repository root. If it is outside the repository, only the
    final file name is returned.
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
# Checkpoint utilities
# ============================================================

def load_checkpoint(
    checkpoint_path,
    device,
):
    """
    Load a PyTorch checkpoint.

    The fallback provides compatibility with older PyTorch versions
    that do not support the weights_only argument.
    """

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    return checkpoint


def extract_model_state_dict(
    checkpoint,
):
    """
    Extract the model state dictionary.

    Supports:
    - complete checkpoint dictionaries,
    - legacy files containing state_dict only.
    """

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise TypeError(
            "Checkpoint must be a dictionary"
        )

    if "model_state_dict" in checkpoint:
        return checkpoint[
            "model_state_dict"
        ]

    # Compatibility with legacy state_dict-only files.
    if checkpoint and all(
        torch.is_tensor(value)
        for value in checkpoint.values()
    ):
        return checkpoint

    raise ValueError(
        "Checkpoint does not contain "
        "'model_state_dict' and does not appear "
        "to be a state_dict-only file"
    )


def validate_checkpoint_configuration(
    checkpoint,
    config,
):
    """
    Compare checkpoint model configuration with the current config.

    Legacy state_dict-only files do not contain configuration
    metadata and therefore cannot be checked here.
    """

    if not isinstance(
        checkpoint,
        dict,
    ):
        return

    saved_model_config = checkpoint.get(
        "model_configuration"
    )

    if saved_model_config is None:
        return

    current_model_config = config[
        "model"
    ]

    comparisons = {
        "hidden_units": (
            list(
                current_model_config.get(
                    "hidden_units",
                    [
                        256,
                        128,
                    ],
                )
            ),
            saved_model_config.get(
                "hidden_units"
            ),
        ),

        "dropout_rate": (
            float(
                current_model_config.get(
                    "dropout_rate",
                    0.1,
                )
            ),
            saved_model_config.get(
                "dropout_rate"
            ),
        ),

        "batch_normalization": (
            bool(
                current_model_config.get(
                    "use_batch_normalization",
                    True,
                )
            ),
            saved_model_config.get(
                "batch_normalization"
            ),
        ),

        "output_order": (
            list(
                current_model_config.get(
                    "output_names",
                    [
                        "Sz",
                        "Lz",
                    ],
                )
            ),
            saved_model_config.get(
                "output_order"
            ),
        ),
    }

    mismatches = []

    for setting_name, (
        current_value,
        saved_value,
    ) in comparisons.items():
        if (
            saved_value is not None
            and current_value != saved_value
        ):
            mismatches.append(
                {
                    "setting": (
                        setting_name
                    ),
                    "current": (
                        current_value
                    ),
                    "checkpoint": (
                        saved_value
                    ),
                }
            )

    if mismatches:
        raise ValueError(
            "The current model configuration "
            "does not match the selected checkpoint: "
            f"{mismatches}"
        )


# ============================================================
# Metadata merge
# ============================================================

def merge_accepted_metadata(
    prediction_dataframe,
    accepted_metadata_file,
):
    """
    Merge accepted-sample metadata into predictions by sample_id.
    """

    accepted_metadata = pd.read_csv(
        accepted_metadata_file
    )

    required_columns = {
        "sample_id",
        "prefix",
    }

    missing_columns = (
        required_columns
        - set(
            accepted_metadata.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Accepted metadata is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if accepted_metadata[
        "sample_id"
    ].duplicated().any():
        raise ValueError(
            "accepted_metadata_csv contains "
            "duplicate sample_id values"
        )

    excluded_columns = {
        "accepted",
        "exclusion_reason",
    }

    metadata_columns = [
        column
        for column in (
            accepted_metadata.columns
        )
        if (
            column
            not in prediction_dataframe.columns
            and column
            not in excluded_columns
        )
    ]

    merged_dataframe = (
        prediction_dataframe.merge(
            accepted_metadata[
                [
                    "sample_id"
                ]
                + metadata_columns
            ],
            on="sample_id",
            how="left",
            validate="one_to_one",
            indicator=True,
        )
    )

    unmatched_rows = (
        merged_dataframe[
            "_merge"
        ]
        != "both"
    )

    if unmatched_rows.any():
        missing_sample_ids = (
            merged_dataframe.loc[
                unmatched_rows,
                "sample_id",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "Some test sample IDs were not found "
            "in accepted metadata: "
            f"{missing_sample_ids[:10]}"
        )

    merged_dataframe = (
        merged_dataframe.drop(
            columns=[
                "_merge"
            ]
        )
    )

    return merged_dataframe


# ============================================================
# Prediction table
# ============================================================

def build_prediction_dataframe(
    sample_ids,
    prefixes,
    target_names,
    y_true_standardized,
    y_pred_standardized,
    y_true_physical,
    y_pred_physical,
):
    """
    Build the complete test-prediction table.
    """

    prediction_dataframe = pd.DataFrame(
        {
            "sample_id": (
                sample_ids
            ),

            "prefix": (
                prefixes
            ),

            "subset": (
                "test"
            ),
        }
    )

    for target_index, target_name in enumerate(
        target_names
    ):
        lower_name = str(
            target_name
        ).lower()

        prediction_dataframe[
            f"{lower_name}_true_standardized"
        ] = y_true_standardized[
            :,
            target_index,
        ]

        prediction_dataframe[
            f"{lower_name}_pred_standardized"
        ] = y_pred_standardized[
            :,
            target_index,
        ]

        prediction_dataframe[
            f"{lower_name}_true"
        ] = y_true_physical[
            :,
            target_index,
        ]

        prediction_dataframe[
            f"{lower_name}_pred"
        ] = y_pred_physical[
            :,
            target_index,
        ]

        prediction_dataframe[
            f"{lower_name}_residual"
        ] = (
            y_pred_physical[
                :,
                target_index,
            ]
            - y_true_physical[
                :,
                target_index,
            ]
        )

    return prediction_dataframe


def arrays_from_prediction_dataframe(
    prediction_dataframe,
    target_names,
):
    """
    Reconstruct metric arrays from predictions.csv columns.

    This ensures that metrics are calculated from exactly the rows
    and values represented in the public prediction table.
    """

    true_physical_columns = []
    predicted_physical_columns = []

    true_standardized_columns = []
    predicted_standardized_columns = []

    for target_name in target_names:
        lower_name = str(
            target_name
        ).lower()

        true_physical_columns.append(
            f"{lower_name}_true"
        )

        predicted_physical_columns.append(
            f"{lower_name}_pred"
        )

        true_standardized_columns.append(
            f"{lower_name}_true_standardized"
        )

        predicted_standardized_columns.append(
            f"{lower_name}_pred_standardized"
        )

    y_true_physical = (
        prediction_dataframe[
            true_physical_columns
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    y_pred_physical = (
        prediction_dataframe[
            predicted_physical_columns
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    y_true_standardized = (
        prediction_dataframe[
            true_standardized_columns
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    y_pred_standardized = (
        prediction_dataframe[
            predicted_standardized_columns
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    return (
        y_true_physical,
        y_pred_physical,
        y_true_standardized,
        y_pred_standardized,
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

    required_sections = {
        "system",
        "paths",
        "model",
        "training",
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

    repository_root = (
        get_repository_root(
            config,
            config_path,
        )
    )

    system_name = (
        config[
            "system"
        ][
            "name"
        ]
    )

    paths = config[
        "paths"
    ]

    training_config = config[
        "training"
    ]

    required_path_keys = {
        "preprocessed_dataset_npz",
        "scalers_npz",
        "selected_model_checkpoint",
        "result_output_directory",
    }

    missing_path_keys = (
        required_path_keys
        - set(paths)
    )

    if missing_path_keys:
        raise ValueError(
            "Config paths section is missing: "
            f"{sorted(missing_path_keys)}"
        )

    # --------------------------------------------------------
    # Resolve input and output paths
    # --------------------------------------------------------

    preprocessed_file = resolve_path(
        paths[
            "preprocessed_dataset_npz"
        ],
        config_directory,
    )

    scaler_file = resolve_path(
        paths[
            "scalers_npz"
        ],
        config_directory,
    )

    selected_checkpoint = resolve_path(
        paths[
            "selected_model_checkpoint"
        ],
        config_directory,
    )

    result_directory = ensure_directory(
        resolve_path(
            paths[
                "result_output_directory"
            ],
            config_directory,
        )
    )

    if "accepted_metadata_csv" in paths:
        accepted_metadata_file = resolve_path(
            paths[
                "accepted_metadata_csv"
            ],
            config_directory,
        )

    else:
        accepted_metadata_file = None

    predictions_file = (
        result_directory
        / "predictions.csv"
    )

    metrics_file = (
        result_directory
        / "metrics.json"
    )

    residual_summary_file = (
        result_directory
        / "residual_summary.csv"
    )

    evaluation_summary_file = (
        result_directory
        / "evaluation_summary.json"
    )

    output_files = [
        predictions_file,
        metrics_file,
        residual_summary_file,
        evaluation_summary_file,
    ]

    check_output_files(
        output_files,
        overwrite=(
            arguments.overwrite
        ),
        label="Evaluation outputs",
    )

    required_input_files = [
        preprocessed_file,
        scaler_file,
        selected_checkpoint,
    ]

    if accepted_metadata_file is not None:
        required_input_files.append(
            accepted_metadata_file
        )

    for required_file in required_input_files:
        if not required_file.is_file():
            raise FileNotFoundError(
                "Required input file was not found: "
                f"{safe_public_path(required_file, repository_root)}"
            )

    # --------------------------------------------------------
    # Require complete test-set evaluation
    # --------------------------------------------------------

    if "data_loader" not in training_config:
        raise ValueError(
            "Training config must contain "
            "training.data_loader"
        )

    data_loader_config = (
        training_config[
            "data_loader"
        ]
    )

    if bool(
        data_loader_config.get(
            "drop_last_evaluation",
            False,
        )
    ):
        raise ValueError(
            "drop_last_evaluation must be false "
            "for complete test-set evaluation"
        )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    loader_bundle = build_dataloaders(
        config,
        config_directory=(
            config_directory
        ),
    )

    test_loader = (
        loader_bundle[
            "test_loader"
        ]
    )

    dataset = (
        loader_bundle[
            "dataset"
        ]
    )

    expected_test_count = len(
        dataset[
            "sample_id_test"
        ]
    )

    if len(
        loader_bundle[
            "test_dataset"
        ]
    ) != expected_test_count:
        raise RuntimeError(
            "Test TensorDataset count differs "
            "from sample_id_test count"
        )

    # --------------------------------------------------------
    # Load target scaler
    # --------------------------------------------------------

    with np.load(
        scaler_file,
        allow_pickle=False,
    ) as scaler:
        required_scaler_arrays = {
            "Y_mean",
            "Y_std_safe",
            "target_names",
        }

        missing_scaler_arrays = (
            required_scaler_arrays
            - set(
                scaler.files
            )
        )

        if missing_scaler_arrays:
            raise ValueError(
                "Scaler file is missing arrays: "
                f"{sorted(missing_scaler_arrays)}"
            )

        y_mean = np.asarray(
            scaler[
                "Y_mean"
            ],
            dtype=np.float64,
        )

        y_std_safe = np.asarray(
            scaler[
                "Y_std_safe"
            ],
            dtype=np.float64,
        )

        scaler_target_names = (
            scaler[
                "target_names"
            ]
            .astype(str)
        )

    target_names = (
        dataset[
            "target_names"
        ]
        .astype(str)
    )

    if not np.array_equal(
        target_names,
        scaler_target_names,
    ):
        raise ValueError(
            "Target order differs between "
            "the preprocessed dataset and scaler"
        )

    if y_mean.shape != (
        len(target_names),
    ):
        raise ValueError(
            "Unexpected Y_mean shape: "
            f"{y_mean.shape}"
        )

    if y_std_safe.shape != (
        len(target_names),
    ):
        raise ValueError(
            "Unexpected Y_std_safe shape: "
            f"{y_std_safe.shape}"
        )

    ensure_finite(
        "Y_mean",
        y_mean,
    )

    ensure_finite(
        "Y_std_safe",
        y_std_safe,
    )

    if np.any(
        y_std_safe <= 0
    ):
        raise ValueError(
            "Y_std_safe must contain "
            "strictly positive values"
        )

    # --------------------------------------------------------
    # Device and model
    # --------------------------------------------------------

    device = select_torch_device(
        training_config.get(
            "device",
            "auto",
        )
    )

    dtype_name = str(
        training_config.get(
            "dtype",
            "float32",
        )
    ).lower()

    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
    }

    if dtype_name not in dtype_map:
        raise ValueError(
            "Unsupported training dtype: "
            f"{dtype_name}"
        )

    model_dtype = (
        dtype_map[
            dtype_name
        ]
    )

    input_dimension = int(
        dataset[
            "X_train"
        ].shape[1]
    )

    model = build_model(
        n_in=(
            input_dimension
        ),
        model_config=(
            config["model"]
        ),
    )

    model = model.to(
        device=device,
        dtype=model_dtype,
    )

    checkpoint = load_checkpoint(
        selected_checkpoint,
        device,
    )

    validate_checkpoint_configuration(
        checkpoint,
        config,
    )

    model_state_dict = (
        extract_model_state_dict(
            checkpoint
        )
    )

    model.load_state_dict(
        model_state_dict,
        strict=True,
    )

    # Evaluation mode disables Dropout and uses saved
    # BatchNorm running statistics.
    model.eval()

    # --------------------------------------------------------
    # Test-only inference in standardized target space
    # --------------------------------------------------------

    prediction_batches = []

    with torch.inference_mode():
        for inputs, _targets in test_loader:
            inputs = inputs.to(
                device=device,
                dtype=model_dtype,
                non_blocking=True,
            )

            predicted_batch = model(
                inputs
            )

            if not torch.isfinite(
                predicted_batch
            ).all():
                raise RuntimeError(
                    "Model produced NaN or Inf "
                    "during test inference"
                )

            prediction_batches.append(
                predicted_batch
                .detach()
                .cpu()
                .numpy()
            )

    if not prediction_batches:
        raise RuntimeError(
            "Test DataLoader produced no samples"
        )

    y_pred_standardized = np.vstack(
        prediction_batches
    ).astype(
        np.float64
    )

    # --------------------------------------------------------
    # Saved test targets and identifiers
    # --------------------------------------------------------

    y_true_standardized = np.asarray(
        dataset[
            "Y_test"
        ],
        dtype=np.float64,
    )

    y_true_physical = np.asarray(
        dataset[
            "Y_test_raw"
        ],
        dtype=np.float64,
    )

    sample_ids = (
        dataset[
            "sample_id_test"
        ]
        .astype(str)
    )

    prefixes = (
        dataset[
            "prefix_test"
        ]
        .astype(str)
    )

    expected_shape = (
        expected_test_count,
        len(target_names),
    )

    arrays_to_validate = (
        (
            "Y_pred_standardized",
            y_pred_standardized,
        ),
        (
            "Y_true_standardized",
            y_true_standardized,
        ),
        (
            "Y_true_physical",
            y_true_physical,
        ),
    )

    for array_name, array in arrays_to_validate:
        if array.shape != expected_shape:
            raise RuntimeError(
                f"{array_name} shape mismatch: "
                f"expected {expected_shape}, "
                f"found {array.shape}"
            )

        ensure_finite(
            array_name,
            array,
        )

    if len(prefixes) != expected_test_count:
        raise RuntimeError(
            "Prefix count differs from "
            "the saved test sample count"
        )

    # --------------------------------------------------------
    # Convert predictions to physical units
    # --------------------------------------------------------

    y_pred_physical = (
        y_pred_standardized
        * y_std_safe
        + y_mean
    )

    ensure_finite(
        "Y_pred_physical",
        y_pred_physical,
    )

    # --------------------------------------------------------
    # Build and enrich prediction table
    # --------------------------------------------------------

    prediction_dataframe = (
        build_prediction_dataframe(
            sample_ids=(
                sample_ids
            ),
            prefixes=(
                prefixes
            ),
            target_names=(
                target_names
            ),
            y_true_standardized=(
                y_true_standardized
            ),
            y_pred_standardized=(
                y_pred_standardized
            ),
            y_true_physical=(
                y_true_physical
            ),
            y_pred_physical=(
                y_pred_physical
            ),
        )
    )

    if accepted_metadata_file is not None:
        prediction_dataframe = (
            merge_accepted_metadata(
                prediction_dataframe,
                accepted_metadata_file,
            )
        )

    if len(
        prediction_dataframe
    ) != expected_test_count:
        raise RuntimeError(
            "Prediction table row count changed "
            "during metadata merge"
        )

    if prediction_dataframe[
        "sample_id"
    ].duplicated().any():
        raise RuntimeError(
            "Prediction table contains "
            "duplicate sample_id values"
        )

    prediction_dataframe.to_csv(
        predictions_file,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )

    # --------------------------------------------------------
    # Reload predictions.csv and calculate metrics from it
    # --------------------------------------------------------

    saved_predictions = pd.read_csv(
        predictions_file
    )

    if len(
        saved_predictions
    ) != expected_test_count:
        raise RuntimeError(
            "Saved predictions.csv does not contain "
            "the complete test subset"
        )

    if saved_predictions[
        "sample_id"
    ].duplicated().any():
        raise RuntimeError(
            "Saved predictions.csv contains "
            "duplicate sample_id values"
        )

    (
        metric_y_true_physical,
        metric_y_pred_physical,
        metric_y_true_standardized,
        metric_y_pred_standardized,
    ) = arrays_from_prediction_dataframe(
        saved_predictions,
        target_names,
    )

    metrics_physical = (
        multioutput_regression_metrics(
            metric_y_true_physical,
            metric_y_pred_physical,
            target_names,
        )
    )

    metrics_standardized = (
        multioutput_regression_metrics(
            metric_y_true_standardized,
            metric_y_pred_standardized,
            target_names,
        )
    )

    metrics = {
        "evaluation_subset": (
            "test"
        ),

        "prediction_count": int(
            len(
                saved_predictions
            )
        ),

        "metric_source": (
            "predictions.csv"
        ),

        "physical_target_space": (
            metrics_physical
        ),

        "standardized_target_space": (
            metrics_standardized
        ),
    }

    write_json(
        metrics_file,
        metrics,
    )

    # --------------------------------------------------------
    # Physical-unit residual summary
    # --------------------------------------------------------

    residual_rows = []

    for target_name in target_names:
        target_metrics = (
            metrics_physical[
                "per_target"
            ][
                str(target_name)
            ]
        )

        residual_rows.append(
            {
                "target": str(
                    target_name
                ),

                "count": (
                    target_metrics[
                        "count"
                    ]
                ),

                "sse": (
                    target_metrics[
                        "sse"
                    ]
                ),

                "sst": (
                    target_metrics[
                        "sst"
                    ]
                ),

                "mse": (
                    target_metrics[
                        "mse"
                    ]
                ),

                "rmse": (
                    target_metrics[
                        "rmse"
                    ]
                ),

                "nrmse_by_target_std": (
                    target_metrics[
                        "nrmse_by_target_std"
                    ]
                ),

                "mae": (
                    target_metrics[
                        "mae"
                    ]
                ),

                "bias": (
                    target_metrics[
                        "bias"
                    ]
                ),

                "residual_mean": (
                    target_metrics[
                        "residual_mean"
                    ]
                ),

                "residual_std_population": (
                    target_metrics[
                        "residual_std_population"
                    ]
                ),

                "residual_min": (
                    target_metrics[
                        "residual_min"
                    ]
                ),

                "residual_max": (
                    target_metrics[
                        "residual_max"
                    ]
                ),

                "r2_coefficient_of_determination": (
                    target_metrics[
                        "r2_coefficient_of_determination"
                    ]
                ),

                "pearson_r": (
                    target_metrics[
                        "pearson_r"
                    ]
                ),

                "pearson_correlation_squared": (
                    target_metrics[
                        "pearson_correlation_squared"
                    ]
                ),
            }
        )

    pd.DataFrame(
        residual_rows
    ).to_csv(
        residual_summary_file,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )

    # --------------------------------------------------------
    # Checkpoint metadata
    # --------------------------------------------------------

    if isinstance(
        checkpoint,
        dict,
    ):
        checkpoint_epoch = (
            checkpoint.get(
                "epoch"
            )
        )

        checkpoint_validation_loss = (
            checkpoint.get(
                "validation_loss"
            )
        )

    else:
        checkpoint_epoch = None
        checkpoint_validation_loss = None

    # --------------------------------------------------------
    # Evaluation audit summary
    # --------------------------------------------------------

    evaluation_summary = {
        "created_at_utc": (
            utc_now_iso()
        ),

        "system": (
            config["system"]
        ),

        "evaluation_subset": (
            "complete saved test subset"
        ),

        "model_mode": (
            "evaluation"
        ),

        "dropout_active": False,

        "batch_normalization_mode": (
            "evaluation"
        ),

        "device": str(
            device
        ),

        "dtype": (
            dtype_name
        ),

        "target_order": (
            target_names.tolist()
        ),

        "inverse_transform": (
            "Y_physical = "
            "Y_standardized * "
            "Y_std_safe + Y_mean"
        ),

        "metric_definitions": {
            "residual": (
                "y_pred - y_true"
            ),

            "sse": (
                "sum((y_pred-y_true)^2)"
            ),

            "sst": (
                "sum((y_true-mean(y_true))^2)"
            ),

            "r2": (
                "1 - SSE/SST"
            ),

            "nrmse": (
                "RMSE/std_population(y_true)"
            ),
        },

        "prediction_count": int(
            len(
                saved_predictions
            )
        ),

        "metrics_calculated_from": (
            safe_public_path(
                predictions_file,
                repository_root,
            )
        ),

        "checkpoint_epoch": (
            checkpoint_epoch
        ),

        "checkpoint_validation_loss": (
            checkpoint_validation_loss
        ),

        "files": {
            "config": (
                safe_public_path(
                    config_path,
                    repository_root,
                )
            ),

            "preprocessed_dataset": (
                safe_public_path(
                    preprocessed_file,
                    repository_root,
                )
            ),

            "scaler": (
                safe_public_path(
                    scaler_file,
                    repository_root,
                )
            ),

            "selected_checkpoint": (
                safe_public_path(
                    selected_checkpoint,
                    repository_root,
                )
            ),

            "accepted_metadata": (
                safe_public_path(
                    accepted_metadata_file,
                    repository_root,
                )
                if accepted_metadata_file
                is not None
                else None
            ),

            "predictions_csv": (
                safe_public_path(
                    predictions_file,
                    repository_root,
                )
            ),

            "metrics_json": (
                safe_public_path(
                    metrics_file,
                    repository_root,
                )
            ),

            "residual_summary_csv": (
                safe_public_path(
                    residual_summary_file,
                    repository_root,
                )
            ),

            "evaluation_summary_json": (
                safe_public_path(
                    evaluation_summary_file,
                    repository_root,
                )
            ),
        },

        "sha256": {
            "config": (
                sha256_file(
                    config_path
                )
            ),

            "preprocessed_dataset": (
                sha256_file(
                    preprocessed_file
                )
            ),

            "scaler": (
                sha256_file(
                    scaler_file
                )
            ),

            "selected_checkpoint": (
                sha256_file(
                    selected_checkpoint
                )
            ),

            "predictions_csv": (
                sha256_file(
                    predictions_file
                )
            ),

            "metrics_json": (
                sha256_file(
                    metrics_file
                )
            ),

            "residual_summary_csv": (
                sha256_file(
                    residual_summary_file
                )
            ),
        },

        "software": software_versions(
            {
                "pandas": (
                    pd.__version__
                ),

                "torch": (
                    torch.__version__
                ),

                "cuda_version": (
                    torch.version.cuda
                ),
            }
        ),
    }

    write_json(
        evaluation_summary_file,
        evaluation_summary,
    )

    # --------------------------------------------------------
    # Console report
    # --------------------------------------------------------

    print("=" * 80)

    print(
        f"System          : {system_name}"
    )

    print(
        f"Device          : {device}"
    )

    print(
        f"Test samples    : {len(saved_predictions)}"
    )

    for target_name in target_names:
        target_metrics = (
            metrics_physical[
                "per_target"
            ][
                str(target_name)
            ]
        )

        r2_value = target_metrics[
            "r2_coefficient_of_determination"
        ]

        nrmse_value = target_metrics[
            "nrmse_by_target_std"
        ]

        print(
            f"{target_name} RMSE       : "
            f"{target_metrics['rmse']:.8e}"
        )

        if nrmse_value is None:
            print(
                f"{target_name} NRMSE      : undefined"
            )

        else:
            print(
                f"{target_name} NRMSE      : "
                f"{nrmse_value:.8e}"
            )

        print(
            f"{target_name} SSE        : "
            f"{target_metrics['sse']:.8e}"
        )

        print(
            f"{target_name} SST        : "
            f"{target_metrics['sst']:.8e}"
        )

        if r2_value is None:
            print(
                f"{target_name} R2         : undefined"
            )

        else:
            print(
                f"{target_name} R2         : "
                f"{r2_value:.8f}"
            )

    print(
        "Predictions     : "
        f"{safe_public_path(predictions_file, repository_root)}"
    )

    print(
        "Metrics         : "
        f"{safe_public_path(metrics_file, repository_root)}"
    )

    print(
        "Residual summary: "
        f"{safe_public_path(residual_summary_file, repository_root)}"
    )

    print(
        "Summary         : "
        f"{safe_public_path(evaluation_summary_file, repository_root)}"
    )

    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
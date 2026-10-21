#!/usr/bin/env python3

"""
Regression metrics used by evaluate.py.

Definitions
-----------
Residual:
    y_pred - y_true

SSE:
    sum((y_pred - y_true)^2)

SST:
    sum((y_true - mean(y_true))^2)

R2:
    1 - SSE / SST

NRMSE:
    RMSE / population standard deviation of y_true

The R2 value is the coefficient of determination, not the square
of the Pearson correlation coefficient.
"""

import numpy as np


# ============================================================
# Input validation
# ============================================================

def _as_1d(
    name,
    values,
):
    """
    Convert input to a finite one-dimensional float64 array.

    Parameters
    ----------
    name : str
        Variable name used in error messages.

    values : array-like
        Input numerical values.

    Returns
    -------
    numpy.ndarray
        One-dimensional float64 array.
    """

    array = np.asarray(
        values,
        dtype=np.float64,
    ).reshape(-1)

    if array.size == 0:
        raise ValueError(
            f"{name} is empty"
        )

    if not np.isfinite(
        array
    ).all():
        nan_count = int(
            np.isnan(
                array
            ).sum()
        )

        inf_count = int(
            np.isinf(
                array
            ).sum()
        )

        raise ValueError(
            f"{name} contains non-finite values: "
            f"NaN={nan_count}, Inf={inf_count}"
        )

    return array


def validate_pair(
    y_true,
    y_pred,
):
    """
    Validate a target/prediction pair.

    Returns
    -------
    tuple of numpy.ndarray
        Validated true and predicted arrays.
    """

    true = _as_1d(
        "y_true",
        y_true,
    )

    predicted = _as_1d(
        "y_pred",
        y_pred,
    )

    if true.shape != predicted.shape:
        raise ValueError(
            "Shape mismatch: "
            f"y_true={true.shape}, "
            f"y_pred={predicted.shape}"
        )

    return true, predicted


# ============================================================
# Basic error quantities
# ============================================================

def residuals(
    y_true,
    y_pred,
):
    """
    Calculate signed residuals.

    Residual is defined as:

        y_pred - y_true
    """

    true, predicted = validate_pair(
        y_true,
        y_pred,
    )

    return (
        predicted
        - true
    )


def sum_squared_error(
    y_true,
    y_pred,
):
    """
    Calculate the sum of squared errors.

    SSE = sum((y_pred - y_true)^2)
    """

    residual = residuals(
        y_true,
        y_pred,
    )

    return float(
        np.sum(
            residual ** 2
        )
    )


def total_sum_of_squares(
    y_true,
):
    """
    Calculate the total sum of squares.

    SST = sum((y_true - mean(y_true))^2)
    """

    true = _as_1d(
        "y_true",
        y_true,
    )

    centered = (
        true
        - np.mean(true)
    )

    return float(
        np.sum(
            centered ** 2
        )
    )


def mean_squared_error(
    y_true,
    y_pred,
):
    """
    Calculate mean squared error.

    MSE = mean((y_pred - y_true)^2)
    """

    true, predicted = validate_pair(
        y_true,
        y_pred,
    )

    return float(
        np.mean(
            (
                predicted
                - true
            ) ** 2
        )
    )


def root_mean_squared_error(
    y_true,
    y_pred,
):
    """
    Calculate root mean squared error.

    RMSE = sqrt(MSE)
    """

    return float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )
    )


def mean_absolute_error(
    y_true,
    y_pred,
):
    """
    Calculate mean absolute error.

    MAE = mean(abs(y_pred - y_true))
    """

    true, predicted = validate_pair(
        y_true,
        y_pred,
    )

    return float(
        np.mean(
            np.abs(
                predicted
                - true
            )
        )
    )


# ============================================================
# Normalized error and goodness of fit
# ============================================================

def normalized_root_mean_squared_error(
    y_true,
    y_pred,
):
    """
    Calculate NRMSE normalized by the population standard
    deviation of the target values.

    NRMSE = RMSE / std_population(y_true)

    Returns
    -------
    float or None
        None is returned when the true target has zero variance.
    """

    true, predicted = validate_pair(
        y_true,
        y_pred,
    )

    target_std = float(
        np.std(
            true,
            ddof=0,
        )
    )

    if target_std == 0.0:
        return None

    rmse = root_mean_squared_error(
        true,
        predicted,
    )

    return float(
        rmse
        / target_std
    )


def coefficient_of_determination(
    y_true,
    y_pred,
):
    """
    Calculate the coefficient of determination.

    R2 = 1 - SSE / SST

    Returns
    -------
    float or None
        None is returned when the true target has zero variance
        and therefore SST is zero.
    """

    true, predicted = validate_pair(
        y_true,
        y_pred,
    )

    sse = sum_squared_error(
        true,
        predicted,
    )

    sst = total_sum_of_squares(
        true
    )

    if sst == 0.0:
        return None

    return float(
        1.0
        - (
            sse
            / sst
        )
    )


def pearson_correlation(
    y_true,
    y_pred,
):
    """
    Calculate the Pearson correlation coefficient.

    Returns
    -------
    float or None
        None is returned if either array has zero variance.
    """

    true, predicted = validate_pair(
        y_true,
        y_pred,
    )

    true_std = float(
        np.std(
            true,
            ddof=0,
        )
    )

    predicted_std = float(
        np.std(
            predicted,
            ddof=0,
        )
    )

    if true_std == 0.0:
        return None

    if predicted_std == 0.0:
        return None

    return float(
        np.corrcoef(
            true,
            predicted,
        )[0, 1]
    )


# ============================================================
# Single-target metrics
# ============================================================

def regression_metrics(
    y_true,
    y_pred,
):
    """
    Calculate regression metrics for one target.

    Returns
    -------
    dict
        JSON-serializable regression metrics.
    """

    true, predicted = validate_pair(
        y_true,
        y_pred,
    )

    residual = (
        predicted
        - true
    )

    squared_error = (
        residual ** 2
    )

    absolute_error = np.abs(
        residual
    )

    sample_count = int(
        true.size
    )

    sse = float(
        np.sum(
            squared_error
        )
    )

    sst = total_sum_of_squares(
        true
    )

    mse = float(
        np.mean(
            squared_error
        )
    )

    rmse = float(
        np.sqrt(
            mse
        )
    )

    mae = float(
        np.mean(
            absolute_error
        )
    )

    target_std = float(
        np.std(
            true,
            ddof=0,
        )
    )

    prediction_std = float(
        np.std(
            predicted,
            ddof=0,
        )
    )

    bias = float(
        np.mean(
            residual
        )
    )

    if sst == 0.0:
        r2 = None
    else:
        r2 = float(
            1.0
            - (
                sse
                / sst
            )
        )

    if target_std == 0.0:
        nrmse = None
    else:
        nrmse = float(
            rmse
            / target_std
        )

    correlation = pearson_correlation(
        true,
        predicted,
    )

    if correlation is None:
        correlation_squared = None
    else:
        correlation_squared = float(
            correlation ** 2
        )

    return {
        "count": (
            sample_count
        ),

        "sse": (
            sse
        ),

        "sst": (
            sst
        ),

        "mse": (
            mse
        ),

        "rmse": (
            rmse
        ),

        "nrmse_by_target_std": (
            nrmse
        ),

        "mae": (
            mae
        ),

        "bias": (
            bias
        ),

        "r2_coefficient_of_determination": (
            r2
        ),

        "pearson_r": (
            correlation
        ),

        "pearson_correlation_squared": (
            correlation_squared
        ),

        "residual_mean": (
            bias
        ),

        "residual_std_population": float(
            np.std(
                residual,
                ddof=0,
            )
        ),

        "residual_min": float(
            np.min(
                residual
            )
        ),

        "residual_max": float(
            np.max(
                residual
            )
        ),

        "residual_absolute_max": float(
            np.max(
                absolute_error
            )
        ),

        "target_min": float(
            np.min(
                true
            )
        ),

        "target_max": float(
            np.max(
                true
            )
        ),

        "target_range": float(
            np.max(true)
            - np.min(true)
        ),

        "target_mean": float(
            np.mean(
                true
            )
        ),

        "target_std_population": (
            target_std
        ),

        "prediction_min": float(
            np.min(
                predicted
            )
        ),

        "prediction_max": float(
            np.max(
                predicted
            )
        ),

        "prediction_range": float(
            np.max(predicted)
            - np.min(predicted)
        ),

        "prediction_mean": float(
            np.mean(
                predicted
            )
        ),

        "prediction_std_population": (
            prediction_std
        ),

        "all_values_finite": True,
    }


# ============================================================
# Multi-output metrics
# ============================================================

def multioutput_regression_metrics(
    y_true,
    y_pred,
    target_names=("Sz", "Lz"),
):
    """
    Calculate metrics for a multi-output regression problem.

    Parameters
    ----------
    y_true : array-like
        Shape (n_samples, n_targets).

    y_pred : array-like
        Shape (n_samples, n_targets).

    target_names : sequence of str
        Target names in column order.

    Returns
    -------
    dict
        Per-target metrics and uniform-average multi-output
        summary metrics.
    """

    true = np.asarray(
        y_true,
        dtype=np.float64,
    )

    predicted = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    names = [
        str(name)
        for name in target_names
    ]

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if true.ndim != 2:
        raise ValueError(
            "y_true must be two-dimensional"
        )

    if predicted.ndim != 2:
        raise ValueError(
            "y_pred must be two-dimensional"
        )

    if true.shape != predicted.shape:
        raise ValueError(
            "Shape mismatch: "
            f"y_true={true.shape}, "
            f"y_pred={predicted.shape}"
        )

    if true.shape[0] == 0:
        raise ValueError(
            "Metric arrays contain no samples"
        )

    if true.shape[1] != len(names):
        raise ValueError(
            "target_names count differs "
            "from output dimension"
        )

    if len(
        set(
            names
        )
    ) != len(names):
        raise ValueError(
            "target_names must be unique"
        )

    if not np.isfinite(
        true
    ).all():
        raise ValueError(
            "y_true contains NaN or Inf"
        )

    if not np.isfinite(
        predicted
    ).all():
        raise ValueError(
            "y_pred contains NaN or Inf"
        )

    # --------------------------------------------------------
    # Per-target metrics
    # --------------------------------------------------------

    per_target = {}

    for index, target_name in enumerate(
        names
    ):
        per_target[
            target_name
        ] = regression_metrics(
            true[
                :,
                index,
            ],
            predicted[
                :,
                index,
            ],
        )

    # --------------------------------------------------------
    # Multi-output uniform averages
    # --------------------------------------------------------

    residual = (
        predicted
        - true
    )

    squared_error = (
        residual ** 2
    )

    absolute_error = np.abs(
        residual
    )

    uniform_average_mse = float(
        np.mean(
            squared_error
        )
    )

    uniform_average_rmse = float(
        np.sqrt(
            uniform_average_mse
        )
    )

    uniform_average_mae = float(
        np.mean(
            absolute_error
        )
    )

    total_sse = float(
        np.sum(
            squared_error
        )
    )

    per_target_r2_values = [
        per_target[
            target_name
        ][
            "r2_coefficient_of_determination"
        ]
        for target_name in names
    ]

    finite_r2_values = [
        value
        for value in per_target_r2_values
        if value is not None
    ]

    if len(
        finite_r2_values
    ) == len(names):
        uniform_average_r2 = float(
            np.mean(
                finite_r2_values
            )
        )
    else:
        uniform_average_r2 = None

    per_target_nrmse_values = [
        per_target[
            target_name
        ][
            "nrmse_by_target_std"
        ]
        for target_name in names
    ]

    finite_nrmse_values = [
        value
        for value in per_target_nrmse_values
        if value is not None
    ]

    if len(
        finite_nrmse_values
    ) == len(names):
        uniform_average_nrmse = float(
            np.mean(
                finite_nrmse_values
            )
        )
    else:
        uniform_average_nrmse = None

    return {
        "definition": {
            "residual": (
                "y_pred - y_true"
            ),

            "sse": (
                "sum((y_pred-y_true)^2)"
            ),

            "sst": (
                "sum((y_true-mean(y_true))^2)"
            ),

            "r2_coefficient_of_determination": (
                "1 - SSE/SST"
            ),

            "nrmse_by_target_std": (
                "RMSE/std_population(y_true)"
            ),

            "pearson_correlation_squared": (
                "pearson_r^2; not the coefficient "
                "of determination"
            ),

            "population_standard_deviation_ddof": 0,
        },

        "n_samples": int(
            true.shape[0]
        ),

        "n_targets": int(
            true.shape[1]
        ),

        "target_order": names,

        "per_target": (
            per_target
        ),

        "multioutput_uniform_average_mse": (
            uniform_average_mse
        ),

        "multioutput_uniform_average_rmse": (
            uniform_average_rmse
        ),

        "multioutput_uniform_average_mae": (
            uniform_average_mae
        ),

        "multioutput_uniform_average_r2": (
            uniform_average_r2
        ),

        "multioutput_uniform_average_nrmse_by_target_std": (
            uniform_average_nrmse
        ),

        "multioutput_total_sse": (
            total_sse
        ),

        "all_values_finite": True,
    }

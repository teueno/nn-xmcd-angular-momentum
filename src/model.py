#!/usr/bin/env python3

"""
Neural-network model for simultaneous prediction of Sz and Lz.

Architecture
------------
Input
  -> Linear
  -> BatchNorm
  -> ReLU
  -> Dropout
  -> Linear
  -> BatchNorm
  -> ReLU
  -> Two independent linear output heads:
       head_sz -> Sz
       head_lz -> Lz

The two output-head values are concatenated in the order:

    [Sz, Lz]
"""

from collections import OrderedDict
from numbers import Integral
from numbers import Real

import torch
import torch.nn as nn


# ============================================================
# Constants
# ============================================================

SUPPORTED_CLASS_NAME = "MultiOutputNN"

SUPPORTED_OUTPUT_NAMES = (
    "Sz",
    "Lz",
)

SUPPORTED_WEIGHT_INITIALIZATION = (
    "kaiming_normal",
    "kaiming_normal_fan_in_relu",
)

SUPPORTED_BIAS_INITIALIZATION = (
    "zeros",
)


# ============================================================
# Validation helpers
# ============================================================

def validate_positive_integer(
    name,
    value,
):
    """
    Validate a positive integer.

    Parameters
    ----------
    name : str
        Name used in error messages.

    value : object
        Value to validate.

    Returns
    -------
    int
        Validated integer.
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
        Integral,
    ):
        raise TypeError(
            f"{name} must be an integer"
        )

    value = int(
        value
    )

    if value <= 0:
        raise ValueError(
            f"{name} must be positive"
        )

    return value


def validate_probability(
    name,
    value,
):
    """
    Validate a probability in the half-open interval [0, 1).
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

    value = float(
        value
    )

    if not 0.0 <= value < 1.0:
        raise ValueError(
            f"{name} must satisfy "
            "0.0 <= value < 1.0"
        )

    return value


def normalize_hidden_units(
    hidden_units,
):
    """
    Validate and normalize the two hidden-layer dimensions.
    """

    if not isinstance(
        hidden_units,
        (
            tuple,
            list,
        ),
    ):
        raise TypeError(
            "hidden_units must be a tuple "
            "or list of two positive integers"
        )

    if len(hidden_units) != 2:
        raise ValueError(
            "hidden_units must contain "
            "exactly two values"
        )

    hidden_1 = validate_positive_integer(
        "hidden_units[0]",
        hidden_units[0],
    )

    hidden_2 = validate_positive_integer(
        "hidden_units[1]",
        hidden_units[1],
    )

    return (
        hidden_1,
        hidden_2,
    )


def normalize_output_names(
    output_names,
):
    """
    Validate and normalize output names.

    The implemented heads have fixed physical meanings and order:

        output 0 -> Sz
        output 1 -> Lz
    """

    if not isinstance(
        output_names,
        (
            tuple,
            list,
        ),
    ):
        raise TypeError(
            "output_names must be a tuple "
            "or list"
        )

    if len(output_names) != 2:
        raise ValueError(
            "output_names must contain "
            "exactly two names"
        )

    normalized_names = tuple(
        str(name)
        for name in output_names
    )

    if normalized_names != SUPPORTED_OUTPUT_NAMES:
        raise ValueError(
            "output_names must have the fixed order "
            f"{SUPPORTED_OUTPUT_NAMES}. "
            f"Received {normalized_names}."
        )

    return normalized_names


# ============================================================
# Model
# ============================================================

class MultiOutputNN(nn.Module):
    """
    Fully connected multi-output neural network.

    The model consists of a shared feature extractor followed by
    two independent linear output heads for the standardized target
    variables Sz and Lz.

    Parameters
    ----------
    n_in : int
        Number of input features. For concatenated XAS and XMCD
        spectra, this is n_xas + n_xmcd.

    hidden_units : tuple of int, optional
        Number of units in the two shared hidden layers.

        Default:
            (256, 128)

    dropout_rate : float, optional
        Dropout probability applied after the first hidden-layer
        activation.

        Default:
            0.1

    use_batch_normalization : bool, optional
        If True, BatchNorm1d is applied between each shared linear
        layer and its ReLU activation.

        Default:
            True

    output_names : tuple of str, optional
        Names and order of the prediction targets.

        The implemented output order is fixed as:

            ("Sz", "Lz")

    initialize_weights : bool, optional
        If True, linear-layer weights are initialized using Kaiming
        normal initialization and biases are initialized to zero.

        Default:
            True

    Notes
    -----
    The default architecture is:

        Linear(n_in, 256)
        BatchNorm1d(256)
        ReLU
        Dropout(p=0.1)

        Linear(256, 128)
        BatchNorm1d(128)
        ReLU

        Linear(128, 1) for Sz
        Linear(128, 1) for Lz
    """

    def __init__(
        self,
        n_in,
        hidden_units=(256, 128),
        dropout_rate=0.1,
        use_batch_normalization=True,
        output_names=("Sz", "Lz"),
        initialize_weights=True,
    ):
        super().__init__()

        # ----------------------------------------------------
        # Validate configuration
        # ----------------------------------------------------

        n_in = validate_positive_integer(
            "n_in",
            n_in,
        )

        hidden_1, hidden_2 = (
            normalize_hidden_units(
                hidden_units
            )
        )

        dropout_rate = validate_probability(
            "dropout_rate",
            dropout_rate,
        )

        if not isinstance(
            use_batch_normalization,
            bool,
        ):
            raise TypeError(
                "use_batch_normalization "
                "must be bool"
            )

        if not isinstance(
            initialize_weights,
            bool,
        ):
            raise TypeError(
                "initialize_weights must be bool"
            )

        output_names = normalize_output_names(
            output_names
        )

        # ----------------------------------------------------
        # Store model configuration
        # ----------------------------------------------------

        self.n_in = (
            n_in
        )

        self.hidden_units = (
            hidden_1,
            hidden_2,
        )

        self.dropout_rate = (
            dropout_rate
        )

        self.use_batch_normalization = (
            use_batch_normalization
        )

        self.output_names = (
            output_names
        )

        self.initialize_weights_enabled = (
            initialize_weights
        )

        # ----------------------------------------------------
        # Shared feature extractor
        # ----------------------------------------------------

        shared_layers = OrderedDict()

        shared_layers[
            "linear_1"
        ] = nn.Linear(
            in_features=n_in,
            out_features=hidden_1,
            bias=True,
        )

        if self.use_batch_normalization:
            shared_layers[
                "batch_norm_1"
            ] = nn.BatchNorm1d(
                num_features=hidden_1,
            )

        shared_layers[
            "relu_1"
        ] = nn.ReLU()

        # Dropout is included only after the first
        # hidden-layer activation.
        if dropout_rate > 0.0:
            shared_layers[
                "dropout_1"
            ] = nn.Dropout(
                p=dropout_rate
            )

        shared_layers[
            "linear_2"
        ] = nn.Linear(
            in_features=hidden_1,
            out_features=hidden_2,
            bias=True,
        )

        if self.use_batch_normalization:
            shared_layers[
                "batch_norm_2"
            ] = nn.BatchNorm1d(
                num_features=hidden_2,
            )

        shared_layers[
            "relu_2"
        ] = nn.ReLU()

        self.shared = nn.Sequential(
            shared_layers
        )

        # ----------------------------------------------------
        # Independent output heads
        # ----------------------------------------------------

        self.head_sz = nn.Linear(
            in_features=hidden_2,
            out_features=1,
            bias=True,
        )

        self.head_lz = nn.Linear(
            in_features=hidden_2,
            out_features=1,
            bias=True,
        )

        # ----------------------------------------------------
        # Weight initialization
        # ----------------------------------------------------

        if self.initialize_weights_enabled:
            self.initialize_weights()

    # ========================================================
    # Weight initialization
    # ========================================================

    def initialize_weights(self):
        """
        Initialize trainable model parameters.

        Linear layers
        -------------
        Weights:
            Kaiming normal initialization with fan-in mode for
            a ReLU-based network.

        Biases:
            Zero initialization.

        Batch-normalization layers
        --------------------------
        Scale:
            One.

        Bias:
            Zero.
        """

        for module in self.modules():
            if isinstance(
                module,
                nn.Linear,
            ):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_in",
                    nonlinearity="relu",
                )

                if module.bias is not None:
                    nn.init.zeros_(
                        module.bias
                    )

            elif isinstance(
                module,
                nn.BatchNorm1d,
            ):
                if module.weight is not None:
                    nn.init.ones_(
                        module.weight
                    )

                if module.bias is not None:
                    nn.init.zeros_(
                        module.bias
                    )

    # ========================================================
    # Input validation
    # ========================================================

    def validate_input(
        self,
        x,
    ):
        """
        Validate a model input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor expected to have shape:

                (batch_size, n_in)
        """

        if not torch.is_tensor(
            x
        ):
            raise TypeError(
                "Model input must be "
                "a torch.Tensor"
            )

        if x.ndim != 2:
            raise RuntimeError(
                "Model input must be "
                "two-dimensional with shape "
                "(batch_size, n_in). "
                f"Received shape {tuple(x.shape)}."
            )

        if x.shape[0] <= 0:
            raise RuntimeError(
                "Model input batch is empty"
            )

        if x.shape[1] != self.n_in:
            raise RuntimeError(
                "Input dimension mismatch: "
                f"expected {self.n_in}, "
                f"received {x.shape[1]}."
            )

        if not x.is_floating_point():
            raise TypeError(
                "Model input must have "
                "a floating-point dtype. "
                f"Received {x.dtype}."
            )

        # Obtain model device and dtype from the first parameter.
        reference_parameter = next(
            self.parameters()
        )

        if x.device != reference_parameter.device:
            raise RuntimeError(
                "Input and model are on different devices: "
                f"input={x.device}, "
                f"model={reference_parameter.device}."
            )

        if x.dtype != reference_parameter.dtype:
            raise RuntimeError(
                "Input and model use different dtypes: "
                f"input={x.dtype}, "
                f"model={reference_parameter.dtype}."
            )

        if not torch.isfinite(
            x
        ).all():
            nan_count = int(
                torch.isnan(
                    x
                ).sum().item()
            )

            inf_count = int(
                torch.isinf(
                    x
                ).sum().item()
            )

            raise RuntimeError(
                "Model input contains "
                "non-finite values: "
                f"NaN={nan_count}, "
                f"Inf={inf_count}."
            )

        # BatchNorm1d cannot estimate batch statistics from a
        # single training sample.
        if (
            self.training
            and self.use_batch_normalization
            and x.shape[0] < 2
        ):
            raise RuntimeError(
                "BatchNorm1d requires at least two "
                "samples per batch in training mode. "
                f"Received batch size {x.shape[0]}."
            )

    # ========================================================
    # Forward propagation
    # ========================================================

    def forward(
        self,
        x,
    ):
        """
        Run forward propagation.

        Parameters
        ----------
        x : torch.Tensor
            Standardized input spectra with shape:

                (batch_size, n_in)

            Expected feature ordering:

                [XAS energy channels, XMCD energy channels]

        Returns
        -------
        torch.Tensor
            Predicted standardized target values with shape:

                (batch_size, 2)

            Output ordering:

                [:, 0] = Sz
                [:, 1] = Lz
        """

        self.validate_input(
            x
        )

        shared_representation = (
            self.shared(
                x
            )
        )

        predicted_sz = (
            self.head_sz(
                shared_representation
            )
        )

        predicted_lz = (
            self.head_lz(
                shared_representation
            )
        )

        predictions = torch.cat(
            [
                predicted_sz,
                predicted_lz,
            ],
            dim=1,
        )

        if predictions.shape != (
            x.shape[0],
            2,
        ):
            raise RuntimeError(
                "Unexpected model output shape: "
                f"{tuple(predictions.shape)}."
            )

        if not torch.isfinite(
            predictions
        ).all():
            nan_count = int(
                torch.isnan(
                    predictions
                ).sum().item()
            )

            inf_count = int(
                torch.isinf(
                    predictions
                ).sum().item()
            )

            raise RuntimeError(
                "Model output contains "
                "non-finite values: "
                f"NaN={nan_count}, "
                f"Inf={inf_count}."
            )

        return predictions

    # ========================================================
    # Model information
    # ========================================================

    def configuration(self):
        """
        Return model configuration as a JSON-serializable dictionary.
        """

        output_head_configuration = {}

        for output_name in self.output_names:
            output_head_configuration[
                output_name
            ] = {
                "input_features": (
                    self.hidden_units[1]
                ),
                "output_features": 1,
                "activation": None,
            }

        return {
            "class_name": (
                self.__class__.__name__
            ),

            "input_features": (
                self.n_in
            ),

            "hidden_units": list(
                self.hidden_units
            ),

            "dropout_rate": (
                self.dropout_rate
            ),

            "dropout_position": (
                "after_first_relu"
                if self.dropout_rate > 0.0
                else None
            ),

            "batch_normalization": (
                self.use_batch_normalization
            ),

            "batch_normalization_position": (
                "after_each_shared_linear_layer"
                if self.use_batch_normalization
                else None
            ),

            "activation": (
                "ReLU"
            ),

            "output_heads": (
                output_head_configuration
            ),

            "output_order": list(
                self.output_names
            ),

            "output_activation": (
                None
            ),

            "weight_initialization": (
                "kaiming_normal_fan_in_relu"
                if self.initialize_weights_enabled
                else "pytorch_default"
            ),

            "bias_initialization": (
                "zeros"
                if self.initialize_weights_enabled
                else "pytorch_default"
            ),
        }

    def parameter_counts(self):
        """
        Return model parameter counts.

        Notes
        -----
        The running mean and running variance stored by BatchNorm
        are buffers, not parameters, and are therefore not included.
        """

        total_parameters = sum(
            parameter.numel()
            for parameter in self.parameters()
        )

        trainable_parameters = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

        nontrainable_parameters = (
            total_parameters
            - trainable_parameters
        )

        total_buffers = sum(
            buffer.numel()
            for buffer in self.buffers()
        )

        return {
            "total_parameters": int(
                total_parameters
            ),

            "trainable_parameters": int(
                trainable_parameters
            ),

            "nontrainable_parameters": int(
                nontrainable_parameters
            ),

            "registered_buffer_values": int(
                total_buffers
            ),
        }


# ============================================================
# Config-driven builder
# ============================================================

def build_model(
    n_in,
    model_config=None,
):
    """
    Build MultiOutputNN from a configuration dictionary.

    Parameters
    ----------
    n_in : int
        Number of input features.

    model_config : dict or None
        Model settings. Missing settings use the defaults.

        Example:

            {
              "class_name": "MultiOutputNN",
              "hidden_units": [256, 128],
              "dropout_rate": 0.1,
              "use_batch_normalization": true,
              "output_names": ["Sz", "Lz"],
              "initialize_weights": true,
              "weight_initialization": "kaiming_normal",
              "bias_initialization": "zeros"
            }

    Returns
    -------
    MultiOutputNN
        Constructed model instance.
    """

    if model_config is None:
        model_config = {}

    if not isinstance(
        model_config,
        dict,
    ):
        raise TypeError(
            "model_config must be "
            "a dictionary or None"
        )

    class_name = str(
        model_config.get(
            "class_name",
            SUPPORTED_CLASS_NAME,
        )
    )

    if class_name != SUPPORTED_CLASS_NAME:
        raise ValueError(
            "Unsupported model class: "
            f"{class_name}. "
            f"Supported class is {SUPPORTED_CLASS_NAME}."
        )

    unknown_keys = (
        set(
            model_config
        )
        - {
            "class_name",
            "hidden_units",
            "dropout_rate",
            "use_batch_normalization",
            "output_names",
            "initialize_weights",
            "weight_initialization",
            "bias_initialization",
        }
    )

    if unknown_keys:
        raise ValueError(
            "Unknown model configuration keys: "
            f"{sorted(unknown_keys)}"
        )

    hidden_units = (
        model_config.get(
            "hidden_units",
            [
                256,
                128,
            ],
        )
    )

    dropout_rate = (
        model_config.get(
            "dropout_rate",
            0.1,
        )
    )

    use_batch_normalization = (
        model_config.get(
            "use_batch_normalization",
            True,
        )
    )

    output_names = (
        model_config.get(
            "output_names",
            [
                "Sz",
                "Lz",
            ],
        )
    )

    initialize_weights = (
        model_config.get(
            "initialize_weights",
            True,
        )
    )

    weight_initialization = str(
        model_config.get(
            "weight_initialization",
            "kaiming_normal",
        )
    )

    bias_initialization = str(
        model_config.get(
            "bias_initialization",
            "zeros",
        )
    )

    if initialize_weights:
        if (
            weight_initialization
            not in SUPPORTED_WEIGHT_INITIALIZATION
        ):
            raise ValueError(
                "Unsupported weight initialization: "
                f"{weight_initialization}. "
                "Supported values are "
                f"{SUPPORTED_WEIGHT_INITIALIZATION}."
            )

        if (
            bias_initialization
            not in SUPPORTED_BIAS_INITIALIZATION
        ):
            raise ValueError(
                "Unsupported bias initialization: "
                f"{bias_initialization}. "
                "Supported values are "
                f"{SUPPORTED_BIAS_INITIALIZATION}."
            )

    model = MultiOutputNN(
        n_in=(
            n_in
        ),

        hidden_units=(
            hidden_units
        ),

        dropout_rate=(
            dropout_rate
        ),

        use_batch_normalization=(
            use_batch_normalization
        ),

        output_names=(
            output_names
        ),

        initialize_weights=(
            initialize_weights
        ),
    )

    return model


# ============================================================
# Standalone model test
# ============================================================

def run_self_test():
    """
    Perform standalone model tests.

    This function is executed only when model.py is run directly.
    It is not executed when model.py is imported by train.py or
    evaluate.py.
    """

    test_input_features = 4000
    test_batch_size = 8
    random_seed = 1234

    torch.manual_seed(
        random_seed
    )

    model = build_model(
        n_in=(
            test_input_features
        ),

        model_config={
            "class_name": (
                "MultiOutputNN"
            ),

            "hidden_units": [
                256,
                128,
            ],

            "dropout_rate": (
                0.1
            ),

            "use_batch_normalization": (
                True
            ),

            "output_names": [
                "Sz",
                "Lz",
            ],

            "initialize_weights": (
                True
            ),

            "weight_initialization": (
                "kaiming_normal"
            ),

            "bias_initialization": (
                "zeros"
            ),
        },
    )

    test_input = torch.randn(
        test_batch_size,
        test_input_features,
        dtype=torch.float32,
    )

    # --------------------------------------------------------
    # Training-mode forward pass
    # --------------------------------------------------------

    model.train()

    output_training = model(
        test_input
    )

    if output_training.shape != (
        test_batch_size,
        2,
    ):
        raise RuntimeError(
            "Unexpected training output shape: "
            f"{tuple(output_training.shape)}"
        )

    if not torch.isfinite(
        output_training
    ).all():
        raise RuntimeError(
            "Training-mode output contains "
            "non-finite values"
        )

    # --------------------------------------------------------
    # Evaluation-mode forward pass
    # --------------------------------------------------------

    model.eval()

    with torch.inference_mode():
        output_evaluation = model(
            test_input
        )

    if output_evaluation.shape != (
        test_batch_size,
        2,
    ):
        raise RuntimeError(
            "Unexpected evaluation output shape: "
            f"{tuple(output_evaluation.shape)}"
        )

    if not torch.isfinite(
        output_evaluation
    ).all():
        raise RuntimeError(
            "Evaluation-mode output contains "
            "non-finite values"
        )

    # --------------------------------------------------------
    # Incorrect input-dimension test
    # --------------------------------------------------------

    incorrect_input = torch.randn(
        test_batch_size,
        test_input_features - 1,
        dtype=torch.float32,
    )

    try:
        model(
            incorrect_input
        )

    except RuntimeError:
        pass

    else:
        raise RuntimeError(
            "Input-dimension validation did not "
            "reject an invalid input"
        )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print("=" * 80)

    print(
        "Model self-test completed successfully."
    )

    print(
        "Input shape : "
        f"{tuple(test_input.shape)}"
    )

    print(
        "Output shape: "
        f"{tuple(output_evaluation.shape)}"
    )

    print(
        "Configuration:"
    )

    print(
        model.configuration()
    )

    print(
        "Parameter counts:"
    )

    print(
        model.parameter_counts()
    )

    print("=" * 80)


if __name__ == "__main__":
    run_self_test()
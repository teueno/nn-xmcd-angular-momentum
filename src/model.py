#!/usr/bin/env python3
"""Neural-network model for simultaneous prediction of Sz and Lz.

External-field treatment
------------------------
The external magnetic field is a fixed forward-model condition and is not an
additional neural-network input feature.  The model input remains the
concatenated standardized spectral vector:

    [XAS energy channels, XMCD energy channels]

The external-field magnitude and direction are validated and preserved by the
dataset, split, preprocessing, DataLoader, training, and evaluation workflows.
Changing the external field requires a newly generated, trained, and validated
dataset/model combination.

Architecture
------------
Input -> Linear -> BatchNorm -> ReLU -> Dropout
      -> Linear -> BatchNorm -> ReLU
      -> independent Sz and Lz linear heads

Output order: [Sz, Lz]
"""

from collections import OrderedDict
from numbers import Integral, Real

import torch
import torch.nn as nn

SUPPORTED_CLASS_NAME = "MultiOutputNN"
SUPPORTED_OUTPUT_NAMES = ("Sz", "Lz")
SUPPORTED_INPUT_CHANNELS = ("XAS", "XMCD")
EXTERNAL_FIELD_IS_MODEL_INPUT = False
SUPPORTED_WEIGHT_INITIALIZATION = (
    "kaiming_normal",
    "kaiming_normal_fan_in_relu",
)
SUPPORTED_BIAS_INITIALIZATION = ("zeros",)


def validate_positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def validate_probability(name, value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric")
    value = float(value)
    if not 0.0 <= value < 1.0:
        raise ValueError(f"{name} must satisfy 0.0 <= value < 1.0")
    return value


def normalize_hidden_units(hidden_units):
    if not isinstance(hidden_units, (tuple, list)):
        raise TypeError("hidden_units must be a tuple or list")
    if len(hidden_units) != 2:
        raise ValueError("hidden_units must contain exactly two values")
    return (
        validate_positive_integer("hidden_units[0]", hidden_units[0]),
        validate_positive_integer("hidden_units[1]", hidden_units[1]),
    )


def normalize_output_names(output_names):
    if not isinstance(output_names, (tuple, list)):
        raise TypeError("output_names must be a tuple or list")
    names = tuple(str(name) for name in output_names)
    if names != SUPPORTED_OUTPUT_NAMES:
        raise ValueError(
            f"output_names must have fixed order {SUPPORTED_OUTPUT_NAMES}; "
            f"received {names}"
        )
    return names


class MultiOutputNN(nn.Module):
    """Fully connected spectral-regression model with Sz and Lz heads.

    Notes
    -----
    ``n_in`` must equal ``n_xas + n_xmcd`` from the preprocessed dataset.
    ``B_ext_T`` is deliberately not appended to the feature vector because the
    current workflow uses one fixed external-field condition for every sample.
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
        n_in = validate_positive_integer("n_in", n_in)
        hidden_1, hidden_2 = normalize_hidden_units(hidden_units)
        dropout_rate = validate_probability("dropout_rate", dropout_rate)
        if not isinstance(use_batch_normalization, bool):
            raise TypeError("use_batch_normalization must be bool")
        if not isinstance(initialize_weights, bool):
            raise TypeError("initialize_weights must be bool")
        output_names = normalize_output_names(output_names)

        self.n_in = n_in
        self.hidden_units = (hidden_1, hidden_2)
        self.dropout_rate = dropout_rate
        self.use_batch_normalization = use_batch_normalization
        self.output_names = output_names
        self.input_channels = SUPPORTED_INPUT_CHANNELS
        self.external_field_is_model_input = EXTERNAL_FIELD_IS_MODEL_INPUT
        self.initialize_weights_enabled = initialize_weights

        shared_layers = OrderedDict()
        shared_layers["linear_1"] = nn.Linear(n_in, hidden_1, bias=True)
        if use_batch_normalization:
            shared_layers["batch_norm_1"] = nn.BatchNorm1d(hidden_1)
        shared_layers["relu_1"] = nn.ReLU()
        if dropout_rate > 0.0:
            shared_layers["dropout_1"] = nn.Dropout(p=dropout_rate)
        shared_layers["linear_2"] = nn.Linear(hidden_1, hidden_2, bias=True)
        if use_batch_normalization:
            shared_layers["batch_norm_2"] = nn.BatchNorm1d(hidden_2)
        shared_layers["relu_2"] = nn.ReLU()
        self.shared = nn.Sequential(shared_layers)

        self.head_sz = nn.Linear(hidden_2, 1, bias=True)
        self.head_lz = nn.Linear(hidden_2, 1, bias=True)

        if initialize_weights:
            self.initialize_weights()

    def initialize_weights(self):
        """Apply Kaiming-normal linear weights and standard BN initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_in", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def validate_input(self, x):
        if not torch.is_tensor(x):
            raise TypeError("Model input must be a torch.Tensor")
        if x.ndim != 2:
            raise RuntimeError(
                "Model input must have shape (batch_size, n_in); "
                f"received {tuple(x.shape)}"
            )
        if x.shape[0] <= 0:
            raise RuntimeError("Model input batch is empty")
        if x.shape[1] != self.n_in:
            raise RuntimeError(
                f"Input dimension mismatch: expected {self.n_in}, "
                f"received {x.shape[1]}. B_ext_T must not be appended to the "
                "spectral feature vector."
            )
        if not x.is_floating_point():
            raise TypeError(f"Model input must be floating point; received {x.dtype}")
        reference = next(self.parameters())
        if x.device != reference.device:
            raise RuntimeError(
                f"Input and model are on different devices: "
                f"input={x.device}, model={reference.device}"
            )
        if x.dtype != reference.dtype:
            raise RuntimeError(
                f"Input and model use different dtypes: "
                f"input={x.dtype}, model={reference.dtype}"
            )
        if not torch.isfinite(x).all():
            raise RuntimeError(
                "Model input contains non-finite values: "
                f"NaN={int(torch.isnan(x).sum())}, "
                f"Inf={int(torch.isinf(x).sum())}"
            )
        if self.training and self.use_batch_normalization and x.shape[0] < 2:
            raise RuntimeError(
                "BatchNorm1d requires at least two samples per training batch"
            )

    def forward(self, x):
        """Predict standardized targets in fixed order ``[Sz, Lz]``."""
        self.validate_input(x)
        shared = self.shared(x)
        predictions = torch.cat(
            [self.head_sz(shared), self.head_lz(shared)], dim=1
        )
        if predictions.shape != (x.shape[0], 2):
            raise RuntimeError(
                f"Unexpected model output shape: {tuple(predictions.shape)}"
            )
        if not torch.isfinite(predictions).all():
            raise RuntimeError(
                "Model output contains non-finite values: "
                f"NaN={int(torch.isnan(predictions).sum())}, "
                f"Inf={int(torch.isinf(predictions).sum())}"
            )
        return predictions

    def configuration(self):
        heads = {
            name: {
                "input_features": self.hidden_units[1],
                "output_features": 1,
                "activation": None,
            }
            for name in self.output_names
        }
        return {
            "class_name": self.__class__.__name__,
            "input_features": self.n_in,
            "input_channels": list(self.input_channels),
            "input_feature_order": "[XAS channels, XMCD channels]",
            "external_field_is_model_input": self.external_field_is_model_input,
            "external_field_treatment": (
                "fixed forward-model condition validated outside model.py"
            ),
            "hidden_units": list(self.hidden_units),
            "dropout_rate": self.dropout_rate,
            "dropout_position": (
                "after_first_relu" if self.dropout_rate > 0.0 else None
            ),
            "batch_normalization": self.use_batch_normalization,
            "batch_normalization_position": (
                "after_each_shared_linear_layer"
                if self.use_batch_normalization else None
            ),
            "activation": "ReLU",
            "output_heads": heads,
            "output_order": list(self.output_names),
            "output_activation": None,
            "weight_initialization": (
                "kaiming_normal_fan_in_relu"
                if self.initialize_weights_enabled else "pytorch_default"
            ),
            "bias_initialization": (
                "zeros" if self.initialize_weights_enabled else "pytorch_default"
            ),
        }

    def parameter_counts(self):
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        buffers = sum(buffer.numel() for buffer in self.buffers())
        return {
            "total_parameters": int(total),
            "trainable_parameters": int(trainable),
            "nontrainable_parameters": int(total - trainable),
            "registered_buffer_values": int(buffers),
        }


def build_model(n_in, model_config=None):
    """Build ``MultiOutputNN`` from the model section of a JSON config."""
    if model_config is None:
        model_config = {}
    if not isinstance(model_config, dict):
        raise TypeError("model_config must be a dictionary or None")

    class_name = str(model_config.get("class_name", SUPPORTED_CLASS_NAME))
    if class_name != SUPPORTED_CLASS_NAME:
        raise ValueError(
            f"Unsupported model class: {class_name}; "
            f"supported class is {SUPPORTED_CLASS_NAME}"
        )

    allowed_keys = {
        "class_name", "hidden_units", "dropout_rate",
        "use_batch_normalization", "output_names", "initialize_weights",
        "weight_initialization", "bias_initialization",
    }
    unknown_keys = set(model_config) - allowed_keys
    if unknown_keys:
        raise ValueError(f"Unknown model configuration keys: {sorted(unknown_keys)}")

    initialize_weights = model_config.get("initialize_weights", True)
    weight_initialization = str(
        model_config.get("weight_initialization", "kaiming_normal")
    )
    bias_initialization = str(model_config.get("bias_initialization", "zeros"))
    if initialize_weights:
        if weight_initialization not in SUPPORTED_WEIGHT_INITIALIZATION:
            raise ValueError(
                f"Unsupported weight initialization: {weight_initialization}"
            )
        if bias_initialization not in SUPPORTED_BIAS_INITIALIZATION:
            raise ValueError(f"Unsupported bias initialization: {bias_initialization}")

    return MultiOutputNN(
        n_in=n_in,
        hidden_units=model_config.get("hidden_units", [256, 128]),
        dropout_rate=model_config.get("dropout_rate", 0.1),
        use_batch_normalization=model_config.get(
            "use_batch_normalization", True
        ),
        output_names=model_config.get("output_names", ["Sz", "Lz"]),
        initialize_weights=initialize_weights,
    )


def run_self_test():
    test_input_features = 4000
    test_batch_size = 8
    torch.manual_seed(1234)
    model = build_model(
        test_input_features,
        {
            "class_name": "MultiOutputNN",
            "hidden_units": [256, 128],
            "dropout_rate": 0.1,
            "use_batch_normalization": True,
            "output_names": ["Sz", "Lz"],
            "initialize_weights": True,
            "weight_initialization": "kaiming_normal",
            "bias_initialization": "zeros",
        },
    )
    test_input = torch.randn(test_batch_size, test_input_features)
    model.train()
    training_output = model(test_input)
    model.eval()
    with torch.inference_mode():
        evaluation_output = model(test_input)
    if training_output.shape != (test_batch_size, 2):
        raise RuntimeError("Unexpected training output shape")
    if evaluation_output.shape != (test_batch_size, 2):
        raise RuntimeError("Unexpected evaluation output shape")
    try:
        model(torch.randn(test_batch_size, test_input_features + 1))
    except RuntimeError:
        pass
    else:
        raise RuntimeError("Invalid input dimension was not rejected")

    print("=" * 80)
    print("Model self-test completed successfully.")
    print(f"Input shape : {tuple(test_input.shape)}")
    print(f"Output shape: {tuple(evaluation_output.shape)}")
    print("External field is a model input: False")
    print("Configuration:")
    print(model.configuration())
    print("Parameter counts:")
    print(model.parameter_counts())
    print("=" * 80)


if __name__ == "__main__":
    run_self_test()

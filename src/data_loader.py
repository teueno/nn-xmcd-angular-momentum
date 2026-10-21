#!/usr/bin/env python3
"""PyTorch Dataset/DataLoader construction with external-field validation.

Input
-----
A preprocessed NPZ produced by the external-field-aware preprocess.py.

The external magnetic field is metadata, not a neural-network input feature.
The model input remains the concatenated standardized XAS and XMCD channels.
This module validates and preserves B_ext_T for reproducibility and auditing.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from .utils import ensure_finite, load_json, resolve_path, select_torch_device
except ImportError:
    from utils import ensure_finite, load_json, resolve_path, select_torch_device

SUBSET_NAMES = ("train", "validation", "test")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Construct external-field-aware PyTorch DataLoaders."
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def get_torch_dtype(dtype_name):
    dtype_name = str(dtype_name).lower()
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float64":
        return torch.float64
    raise ValueError(
        f"Unsupported training dtype: {dtype_name}. "
        "Supported values are float32 and float64."
    )


def resolve_dataset_path(training_config, config_directory=None):
    if "paths" not in training_config:
        raise ValueError("Training configuration must contain the 'paths' section")
    paths = training_config["paths"]
    if "preprocessed_dataset_npz" not in paths:
        raise ValueError(
            "Training configuration paths must contain preprocessed_dataset_npz"
        )
    path_text = paths["preprocessed_dataset_npz"]
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path.resolve()
    if config_directory is not None:
        return resolve_path(path_text, config_directory)
    return path.resolve()


def validate_external_field_config(training_config):
    if "external_magnetic_field" not in training_config:
        raise ValueError(
            "Training configuration must contain external_magnetic_field"
        )
    field = training_config["external_magnetic_field"]
    for key in ("enabled", "magnitude_t", "direction"):
        if key not in field:
            raise ValueError(f"Missing external_magnetic_field.{key}")
    if not isinstance(field["enabled"], bool):
        raise TypeError("external_magnetic_field.enabled must be true or false")
    magnitude = float(field["magnitude_t"])
    tolerance = float(field.get("tolerance_t", 1.0e-9))
    if not np.isfinite(magnitude) or magnitude < 0.0:
        raise ValueError("external_magnetic_field.magnitude_t must be nonnegative")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("external_magnetic_field.tolerance_t must be nonnegative")
    direction = np.asarray(field["direction"], dtype=np.float64)
    if direction.shape != (3,):
        raise ValueError("external_magnetic_field.direction must have three values")
    ensure_finite("External magnetic-field direction", direction)
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        raise ValueError("external magnetic-field direction must be nonzero")
    return {
        "enabled": field["enabled"],
        "magnitude_t": magnitude,
        "direction": direction / norm,
        "tolerance_t": tolerance,
    }


def load_preprocessed_dataset(npz_file, expected_field=None):
    npz_file = Path(npz_file).expanduser().resolve()
    if not npz_file.is_file():
        raise FileNotFoundError(f"Preprocessed dataset was not found: {npz_file.name}")

    required = {
        "X_train", "X_validation", "X_test",
        "Y_train", "Y_validation", "Y_test",
        "X_train_raw", "X_validation_raw", "X_test_raw",
        "Y_train_raw", "Y_validation_raw", "Y_test_raw",
        "idx_train", "idx_validation", "idx_test",
        "sample_id_train", "sample_id_validation", "sample_id_test",
        "prefix_train", "prefix_validation", "prefix_test",
        "b_ext_t_train", "b_ext_t_validation", "b_ext_t_test",
        "external_field_enabled", "external_field_magnitude_t",
        "external_field_direction", "energy_xas", "energy_xmcd",
        "target_names",
    }

    with np.load(npz_file, allow_pickle=False) as data:
        missing = required - set(data.files)
        if missing:
            raise ValueError(
                f"Preprocessed dataset is missing required arrays: {sorted(missing)}"
            )
        dataset = {}
        for subset in SUBSET_NAMES:
            dataset[f"X_{subset}"] = np.asarray(data[f"X_{subset}"])
            dataset[f"Y_{subset}"] = np.asarray(data[f"Y_{subset}"])
            dataset[f"X_{subset}_raw"] = np.asarray(data[f"X_{subset}_raw"])
            dataset[f"Y_{subset}_raw"] = np.asarray(data[f"Y_{subset}_raw"])
            dataset[f"idx_{subset}"] = np.asarray(
                data[f"idx_{subset}"], dtype=np.int64
            )
            dataset[f"sample_id_{subset}"] = data[f"sample_id_{subset}"].astype(str)
            dataset[f"prefix_{subset}"] = data[f"prefix_{subset}"].astype(str)
            dataset[f"b_ext_t_{subset}"] = np.asarray(
                data[f"b_ext_t_{subset}"], dtype=np.float64
            )
        dataset["energy_xas"] = np.asarray(data["energy_xas"], dtype=np.float64)
        dataset["energy_xmcd"] = np.asarray(data["energy_xmcd"], dtype=np.float64)
        dataset["target_names"] = data["target_names"].astype(str)
        dataset["external_field_enabled"] = bool(
            np.asarray(data["external_field_enabled"]).reshape(-1)[0]
        )
        dataset["external_field_magnitude_t"] = float(
            np.asarray(data["external_field_magnitude_t"]).reshape(-1)[0]
        )
        dataset["external_field_direction"] = np.asarray(
            data["external_field_direction"], dtype=np.float64
        )

    validate_preprocessed_dataset(dataset, expected_field=expected_field)
    return dataset


def validate_preprocessed_dataset(dataset, expected_field=None):
    expected_input_dimension = None
    all_sample_ids = []
    all_indices = []
    all_prefixes = []

    for subset in SUBSET_NAMES:
        X = dataset[f"X_{subset}"]
        Y = dataset[f"Y_{subset}"]
        X_raw = dataset[f"X_{subset}_raw"]
        Y_raw = dataset[f"Y_{subset}_raw"]
        indices = dataset[f"idx_{subset}"]
        sample_ids = dataset[f"sample_id_{subset}"]
        prefixes = dataset[f"prefix_{subset}"]
        fields = dataset[f"b_ext_t_{subset}"]

        if X.ndim != 2 or Y.ndim != 2 or X_raw.ndim != 2 or Y_raw.ndim != 2:
            raise ValueError(f"{subset} arrays must be two-dimensional")
        sample_count = X.shape[0]
        counts = {
            len(X), len(Y), len(X_raw), len(Y_raw), len(indices),
            len(sample_ids), len(prefixes), len(fields),
        }
        if len(counts) != 1:
            raise ValueError(f"{subset} sample-count mismatch")
        if sample_count == 0:
            raise ValueError(f"{subset} subset is empty")
        if X.shape != X_raw.shape or Y.shape != Y_raw.shape:
            raise ValueError(f"Raw and standardized shapes differ for {subset}")
        if Y.shape[1] != 2:
            raise ValueError(f"Y_{subset} must contain [Sz, Lz]")
        if expected_input_dimension is None:
            expected_input_dimension = X.shape[1]
        elif X.shape[1] != expected_input_dimension:
            raise ValueError("Input dimensions differ across subsets")

        for name, values in (
            (f"X_{subset}", X), (f"Y_{subset}", Y),
            (f"X_{subset}_raw", X_raw), (f"Y_{subset}_raw", Y_raw),
            (f"b_ext_t_{subset}", fields),
        ):
            ensure_finite(name, values)
        if indices.ndim != 1 or fields.ndim != 1:
            raise ValueError(f"Indices and B_ext_T must be one-dimensional for {subset}")
        if np.any(indices < 0) or len(np.unique(indices)) != len(indices):
            raise ValueError(f"Invalid indices in {subset}")
        if sample_ids.ndim != 1 or prefixes.ndim != 1:
            raise ValueError(f"Identifiers must be one-dimensional for {subset}")
        if any(not str(value).strip() for value in sample_ids):
            raise ValueError(f"Empty sample IDs in {subset}")
        if any(not str(value).strip() for value in prefixes):
            raise ValueError(f"Empty Prefix values in {subset}")
        if len(set(sample_ids.tolist())) != len(sample_ids):
            raise ValueError(f"Duplicate sample IDs in {subset}")
        if len(set(prefixes.tolist())) != len(prefixes):
            raise ValueError(f"Duplicate Prefix values in {subset}")

        all_sample_ids.extend(sample_ids.tolist())
        all_prefixes.extend(prefixes.tolist())
        all_indices.extend(indices.tolist())

    if len(set(all_sample_ids)) != len(all_sample_ids):
        raise ValueError("sample_id values overlap across subsets")
    if len(set(all_prefixes)) != len(all_prefixes):
        raise ValueError("Prefix values overlap across subsets")
    if len(set(all_indices)) != len(all_indices):
        raise ValueError("Dataset indices overlap across subsets")

    energy_xas = dataset["energy_xas"]
    energy_xmcd = dataset["energy_xmcd"]
    target_names = dataset["target_names"]
    if energy_xas.ndim != 1 or energy_xmcd.ndim != 1:
        raise ValueError("Energy axes must be one-dimensional")
    ensure_finite("energy_xas", energy_xas)
    ensure_finite("energy_xmcd", energy_xmcd)
    if expected_input_dimension != len(energy_xas) + len(energy_xmcd):
        raise ValueError("Input dimension does not match XAS+XMCD energy axes")
    if target_names.tolist() != ["Sz", "Lz"]:
        raise ValueError(f"Unexpected target order: {target_names.tolist()}")

    direction = dataset["external_field_direction"]
    if direction.shape != (3,):
        raise ValueError("external_field_direction must have shape (3,)")
    ensure_finite("external_field_direction", direction)
    if not np.isclose(np.linalg.norm(direction), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("external_field_direction must be normalized")

    if expected_field is not None:
        tolerance = expected_field["tolerance_t"]
        if dataset["external_field_enabled"] != expected_field["enabled"]:
            raise ValueError("External-field enabled state differs from config")
        if not np.isclose(
            dataset["external_field_magnitude_t"], expected_field["magnitude_t"],
            rtol=0.0, atol=tolerance,
        ):
            raise ValueError("External-field magnitude differs from config")
        if not np.allclose(
            direction, expected_field["direction"], rtol=0.0, atol=1e-12
        ):
            raise ValueError("External-field direction differs from config")
        for subset in SUBSET_NAMES:
            if not np.allclose(
                dataset[f"b_ext_t_{subset}"], expected_field["magnitude_t"],
                rtol=0.0, atol=tolerance,
            ):
                raise ValueError(f"B_ext_T values differ from config in {subset}")


def build_tensor_datasets(dataset, dtype=torch.float32):
    return {
        subset: TensorDataset(
            torch.as_tensor(dataset[f"X_{subset}"], dtype=dtype),
            torch.as_tensor(dataset[f"Y_{subset}"], dtype=dtype),
        )
        for subset in SUBSET_NAMES
    }


def validate_data_loader_config(config):
    train_batch_size = int(config.get("train_batch_size", 32))
    evaluation_batch_size = int(config.get("evaluation_batch_size", 64))
    num_workers = int(config.get("num_workers", 0))
    if train_batch_size <= 0 or evaluation_batch_size <= 0:
        raise ValueError("Batch sizes must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be nonnegative")
    pin_memory = config.get("pin_memory", "auto")
    if not (pin_memory == "auto" or isinstance(pin_memory, bool)):
        raise ValueError("pin_memory must be true, false, or 'auto'")
    return {
        "train_batch_size": train_batch_size,
        "evaluation_batch_size": evaluation_batch_size,
        "shuffle_training": bool(config.get("shuffle_training", True)),
        "drop_last_training": bool(config.get("drop_last_training", True)),
        "drop_last_evaluation": bool(config.get("drop_last_evaluation", False)),
        "num_workers": num_workers,
        "pin_memory_setting": pin_memory,
    }


def build_dataloaders(training_config, config_directory=None):
    if not isinstance(training_config, dict):
        raise TypeError("training_config must be a dictionary")
    if "training" not in training_config:
        raise ValueError("Training configuration must contain 'training'")
    training = training_config["training"]
    if "data_loader" not in training or not isinstance(training["data_loader"], dict):
        raise ValueError("Training configuration must contain training.data_loader")

    expected_field = validate_external_field_config(training_config)
    settings = validate_data_loader_config(training["data_loader"])
    random_seed = int(training.get("random_seed", 1234))
    tensor_dtype = get_torch_dtype(training.get("dtype", "float32"))
    device = select_torch_device(training.get("device", "auto"))
    dataset_file = resolve_dataset_path(training_config, config_directory)
    dataset = load_preprocessed_dataset(dataset_file, expected_field=expected_field)
    tensor_datasets = build_tensor_datasets(dataset, tensor_dtype)
    pin_memory = device.type == "cuda" if settings["pin_memory_setting"] == "auto" else bool(settings["pin_memory_setting"])

    use_batch_normalization = bool(
        training_config.get("model", {}).get("use_batch_normalization", False)
    )
    n_train = len(tensor_datasets["train"])
    batch_size = settings["train_batch_size"]
    if use_batch_normalization and n_train < 2:
        raise ValueError("Batch normalization requires at least two training samples")
    if (
        use_batch_normalization
        and not settings["drop_last_training"]
        and n_train % batch_size == 1
    ):
        raise ValueError("The final training batch would contain one sample")
    if settings["drop_last_training"] and n_train < batch_size:
        raise ValueError("drop_last_training would remove the entire training subset")

    generator = torch.Generator()
    generator.manual_seed(random_seed)
    train_loader = DataLoader(
        tensor_datasets["train"], batch_size=batch_size,
        shuffle=settings["shuffle_training"],
        drop_last=settings["drop_last_training"],
        num_workers=settings["num_workers"], pin_memory=pin_memory,
        generator=generator,
    )
    validation_loader = DataLoader(
        tensor_datasets["validation"],
        batch_size=settings["evaluation_batch_size"], shuffle=False,
        drop_last=settings["drop_last_evaluation"],
        num_workers=settings["num_workers"], pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        tensor_datasets["test"],
        batch_size=settings["evaluation_batch_size"], shuffle=False,
        drop_last=settings["drop_last_evaluation"],
        num_workers=settings["num_workers"], pin_memory=pin_memory,
    )

    effective_settings = {
        **settings,
        "shuffle_validation": False,
        "shuffle_test": False,
        "pin_memory_requested": settings["pin_memory_setting"],
        "pin_memory_effective": pin_memory,
    }
    bundle = {
        "device": device,
        "dtype": tensor_dtype,
        "random_seed": random_seed,
        "pin_memory": pin_memory,
        "settings": effective_settings,
        "dataset": dataset,
        "train_loader": train_loader,
        "validation_loader": validation_loader,
        "test_loader": test_loader,
        "train_dataset": tensor_datasets["train"],
        "validation_dataset": tensor_datasets["validation"],
        "test_dataset": tensor_datasets["test"],
    }
    for subset in SUBSET_NAMES:
        bundle[f"sample_id_{subset}"] = dataset[f"sample_id_{subset}"]
        bundle[f"prefix_{subset}"] = dataset[f"prefix_{subset}"]
        bundle[f"b_ext_t_{subset}"] = dataset[f"b_ext_t_{subset}"]
    bundle["external_field_enabled"] = dataset["external_field_enabled"]
    bundle["external_field_magnitude_t"] = dataset["external_field_magnitude_t"]
    bundle["external_field_direction"] = dataset["external_field_direction"]
    return bundle


def dataloader_summary(bundle):
    dataset = bundle["dataset"]
    settings = bundle["settings"]
    train_loader = bundle["train_loader"]
    validation_loader = bundle["validation_loader"]
    test_loader = bundle["test_loader"]
    n_train = len(dataset["sample_id_train"])
    n_validation = len(dataset["sample_id_validation"])
    n_test = len(dataset["sample_id_test"])
    train_samples = (
        n_train // train_loader.batch_size * train_loader.batch_size
        if settings["drop_last_training"] else n_train
    )
    validation_samples = (
        n_validation // validation_loader.batch_size * validation_loader.batch_size
        if settings["drop_last_evaluation"] else n_validation
    )
    test_samples = (
        n_test // test_loader.batch_size * test_loader.batch_size
        if settings["drop_last_evaluation"] else n_test
    )
    return {
        "n_train": n_train,
        "n_validation": n_validation,
        "n_test": n_test,
        "train_samples_per_epoch": int(train_samples),
        "validation_samples_evaluated": int(validation_samples),
        "test_samples_evaluated": int(test_samples),
        "input_dimension": int(dataset["X_train"].shape[1]),
        "output_dimension": int(dataset["Y_train"].shape[1]),
        "targets": dataset["target_names"].tolist(),
        "external_magnetic_field": {
            "enabled": bool(dataset["external_field_enabled"]),
            "magnitude_t": float(dataset["external_field_magnitude_t"]),
            "direction": dataset["external_field_direction"].tolist(),
        },
        "tensor_dtype": str(bundle["dtype"]),
        "device": str(bundle["device"]),
        "random_seed": int(bundle["random_seed"]),
        "train_batch_size": int(train_loader.batch_size),
        "evaluation_batch_size": int(validation_loader.batch_size),
        "train_batches_per_epoch": int(len(train_loader)),
        "validation_batches": int(len(validation_loader)),
        "test_batches": int(len(test_loader)),
        "shuffle_training": bool(settings["shuffle_training"]),
        "shuffle_validation": False,
        "shuffle_test": False,
        "drop_last_training": bool(settings["drop_last_training"]),
        "drop_last_evaluation": bool(settings["drop_last_evaluation"]),
        "num_workers": int(settings["num_workers"]),
        "pin_memory_requested": settings["pin_memory_requested"],
        "pin_memory_effective": bool(settings["pin_memory_effective"]),
    }


def run_self_test():
    args = parse_arguments()
    config_path = args.config.expanduser().resolve()
    training_config = load_json(config_path)
    loaders = build_dataloaders(training_config, config_path.parent)
    summary = dataloader_summary(loaders)
    field = summary["external_magnetic_field"]
    print("=" * 80)
    print("DataLoader setup completed successfully.")
    print(f"Device                    : {summary['device']}")
    print(f"Tensor dtype              : {summary['tensor_dtype']}")
    print(f"External field            : {field['magnitude_t']:.6f} T")
    print(f"Field direction           : {field['direction']}")
    print(f"Random seed               : {summary['random_seed']}")
    print(f"Train samples             : {summary['n_train']}")
    print(f"Validation samples        : {summary['n_validation']}")
    print(f"Test samples              : {summary['n_test']}")
    print(f"Train samples per epoch   : {summary['train_samples_per_epoch']}")
    print(f"Train batch size          : {summary['train_batch_size']}")
    print(f"Evaluation batch size     : {summary['evaluation_batch_size']}")
    print(f"Train batches             : {summary['train_batches_per_epoch']}")
    print(f"Validation batches        : {summary['validation_batches']}")
    print(f"Test batches              : {summary['test_batches']}")
    print(f"pin_memory                : {summary['pin_memory_effective']}")
    print(f"Input dimension           : {summary['input_dimension']}")
    print(f"Output dimension          : {summary['output_dimension']}")
    print(f"num_workers               : {summary['num_workers']}")
    print("=" * 80)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run_self_test()

#!/usr/bin/env python3

"""Train MultiOutputNN with reproducible, automatically threaded CPU execution."""

import argparse
import copy
import csv
import os
import random
import shutil
import time
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import torch

try:
    from .data_loader import build_dataloaders, dataloader_summary
    from .model import build_model
    from .utils import (
        check_output_files,
        ensure_directory,
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
    from data_loader import build_dataloaders, dataloader_summary
    from model import build_model
    from utils import (
        check_output_files,
        ensure_directory,
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


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train the XMCD multi-output neural network."
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
        help="Replace existing training outputs.",
    )
    return parser.parse_args()


def safe_public_path(path, repository_root):
    if path is None:
        return None
    try:
        return public_path(path, repository_root)
    except ValueError:
        return Path(path).name


def public_config_copy(config):
    return copy.deepcopy(config)


def validate_positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not bool")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def validate_nonnegative_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not bool")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def validate_positive_float(name, value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric, not bool")
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def validate_nonnegative_float(name, value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric, not bool")
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def validate_probability_open(name, value):
    value = validate_positive_float(name, value)
    if value >= 1.0:
        raise ValueError(f"{name} must be smaller than 1.0")
    return value


def validate_config(config):
    if not isinstance(config, dict):
        raise TypeError("Configuration must be a dictionary")

    required_sections = {"system", "paths", "model", "training"}
    missing_sections = required_sections - set(config)
    if missing_sections:
        raise ValueError(
            f"Missing configuration sections: {sorted(missing_sections)}"
        )

    if "name" not in config["system"]:
        raise ValueError("Config must contain system.name")
    system_name = str(config["system"]["name"]).strip()
    if not system_name:
        raise ValueError("system.name must not be empty")

    required_paths = {
        "repository_root",
        "preprocessed_dataset_npz",
        "model_output_directory",
        "result_output_directory",
    }
    missing_paths = required_paths - set(config["paths"])
    if missing_paths:
        raise ValueError(f"Missing path settings: {sorted(missing_paths)}")

    training = config["training"]
    if not isinstance(training, dict):
        raise TypeError("training must be a dictionary")
    for subsection in (
        "optimizer",
        "loss",
        "data_loader",
        "early_stopping",
        "checkpoint",
    ):
        if subsection not in training:
            raise ValueError(f"Missing training subsection: {subsection}")

    return system_name


def select_torch_dtype(dtype_name):
    dtype_name = str(dtype_name).lower()
    mapping = {
        "float32": torch.float32,
        "float64": torch.float64,
    }
    if dtype_name not in mapping:
        raise ValueError("training.dtype must be 'float32' or 'float64'")
    return dtype_name, mapping[dtype_name]


# ============================================================
# Automatic CPU threading
# ============================================================

def detect_available_cpu_count():
    """Return CPUs available to this process, respecting Linux affinity."""
    try:
        count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        count = os.cpu_count()

    if count is None or int(count) <= 0:
        return 1
    return int(count)


def parse_thread_setting(name, value, automatic_value):
    """Accept a positive integer, 'auto', or None."""
    if value is None or str(value).strip().lower() == "auto":
        return int(automatic_value)
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer or 'auto'")
    try:
        count = int(value)
    except (TypeError, ValueError) as exception:
        raise TypeError(
            f"{name} must be a positive integer or 'auto'"
        ) from exception
    if count <= 0:
        raise ValueError(f"{name} must be positive")
    return count


def configure_cpu_threads(training_config, device):
    """Configure PyTorch intra-op and inter-op CPU thread counts."""
    cpu_config = training_config.get("cpu_threads", {})
    if not isinstance(cpu_config, dict):
        raise TypeError("training.cpu_threads must be a dictionary")

    available_cpus = detect_available_cpu_count()
    requested_num_threads = cpu_config.get("num_threads", "auto")
    requested_interop_threads = cpu_config.get("interop_threads", "auto")
    warn_on_oversubscription = cpu_config.get(
        "warn_on_oversubscription", True
    )
    if not isinstance(warn_on_oversubscription, bool):
        raise TypeError(
            "cpu_threads.warn_on_oversubscription must be true or false"
        )

    automatic_num_threads = available_cpus
    automatic_interop_threads = max(1, min(8, available_cpus // 4))

    num_threads = parse_thread_setting(
        "cpu_threads.num_threads",
        requested_num_threads,
        automatic_num_threads,
    )
    interop_threads = parse_thread_setting(
        "cpu_threads.interop_threads",
        requested_interop_threads,
        automatic_interop_threads,
    )

    if warn_on_oversubscription and num_threads > available_cpus:
        print(
            "Warning: requested intra-op threads "
            f"({num_threads}) exceed available CPUs ({available_cpus})."
        )
    if warn_on_oversubscription and interop_threads > available_cpus:
        print(
            "Warning: requested inter-op threads "
            f"({interop_threads}) exceed available CPUs ({available_cpus})."
        )

    # torch.set_num_threads() is the effective runtime control here.
    # Environment values are recorded and supplied to compatible backends.
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)

    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(interop_threads)
    except RuntimeError as exception:
        raise RuntimeError(
            "PyTorch inter-op threads must be configured before parallel "
            "work starts. Call configure_cpu_threads() earlier in main()."
        ) from exception

    return {
        "device": str(device),
        "available_logical_cpus": available_cpus,
        "requested_num_threads": requested_num_threads,
        "requested_interop_threads": requested_interop_threads,
        "automatic_num_threads": automatic_num_threads,
        "automatic_interop_threads": automatic_interop_threads,
        "effective_num_threads": int(torch.get_num_threads()),
        "effective_interop_threads": int(torch.get_num_interop_threads()),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "numexpr_num_threads": os.environ.get("NUMEXPR_NUM_THREADS"),
    }


def build_optimizer(model, optimizer_config):
    if not isinstance(optimizer_config, dict):
        raise TypeError("training.optimizer must be a dictionary")
    name = str(optimizer_config.get("name", "adam")).lower()
    if name != "adam":
        raise ValueError("Only the Adam optimizer is implemented")

    learning_rate = validate_positive_float(
        "optimizer.learning_rate",
        optimizer_config.get("learning_rate", 2.0e-5),
    )
    beta1 = validate_probability_open(
        "optimizer.beta1", optimizer_config.get("beta1", 0.9)
    )
    beta2 = validate_probability_open(
        "optimizer.beta2", optimizer_config.get("beta2", 0.999)
    )
    epsilon = validate_positive_float(
        "optimizer.epsilon", optimizer_config.get("epsilon", 1.0e-8)
    )
    weight_decay = validate_nonnegative_float(
        "optimizer.weight_decay",
        optimizer_config.get("weight_decay", 0.0),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=epsilon,
        weight_decay=weight_decay,
    )
    effective = {
        "name": "adam",
        "learning_rate": learning_rate,
        "beta1": beta1,
        "beta2": beta2,
        "epsilon": epsilon,
        "weight_decay": weight_decay,
    }
    return optimizer, effective


def build_loss(loss_config, device, dtype):
    if not isinstance(loss_config, dict):
        raise TypeError("training.loss must be a dictionary")
    if str(loss_config.get("name", "mse")).lower() != "mse":
        raise ValueError("Only MSE loss is implemented")

    loss_space = str(
        loss_config.get("space", "standardized_target_space")
    )
    if loss_space != "standardized_target_space":
        raise ValueError(
            "Training loss must be evaluated in standardized_target_space"
        )

    output_reduction = str(
        loss_config.get("output_reduction", "mean")
    ).lower()
    batch_reduction = str(
        loss_config.get("batch_reduction", "mean")
    ).lower()
    if output_reduction not in {"mean", "sum"}:
        raise ValueError("loss.output_reduction must be 'mean' or 'sum'")
    if batch_reduction not in {"mean", "sum"}:
        raise ValueError("loss.batch_reduction must be 'mean' or 'sum'")

    target_weights = torch.as_tensor(
        loss_config.get("target_weights", [1.0, 1.0]),
        device=device,
        dtype=dtype,
    )
    if target_weights.shape != (2,):
        raise ValueError("loss.target_weights must contain two values")
    if not torch.isfinite(target_weights).all():
        raise ValueError("loss.target_weights contains NaN or Inf")
    if torch.any(target_weights < 0.0):
        raise ValueError("loss.target_weights must be nonnegative")
    if torch.sum(target_weights).item() <= 0.0:
        raise ValueError("At least one target weight must be positive")

    def unreduced_per_sample_loss(prediction, target):
        if prediction.shape != target.shape:
            raise RuntimeError(
                "Prediction/target shape mismatch: "
                f"{tuple(prediction.shape)} != {tuple(target.shape)}"
            )
        if prediction.ndim != 2 or prediction.shape[1] != 2:
            raise RuntimeError("Loss expects shape (batch_size, 2)")
        weighted = (prediction - target) ** 2 * target_weights.reshape(1, 2)
        if output_reduction == "mean":
            return weighted.mean(dim=1)
        return weighted.sum(dim=1)

    def loss_function(prediction, target):
        per_sample = unreduced_per_sample_loss(prediction, target)
        if batch_reduction == "mean":
            return per_sample.mean()
        return per_sample.sum()

    effective = {
        "name": "mse",
        "space": loss_space,
        "output_reduction": output_reduction,
        "batch_reduction": batch_reduction,
        "effective_target_weights": target_weights.detach().cpu().tolist(),
    }
    return (
        loss_function,
        unreduced_per_sample_loss,
        batch_reduction,
        effective,
    )


def gradients_are_finite(model):
    return all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def run_epoch(
    model,
    data_loader,
    loss_function,
    unreduced_per_sample_loss,
    batch_reduction,
    device,
    dtype,
    optimizer=None,
    gradient_clip_norm=None,
):
    training_mode = optimizer is not None
    model.train(training_mode)

    per_sample_loss_sum = 0.0
    processed_sample_count = 0
    gradient_norm_sum = 0.0
    gradient_step_count = 0
    maximum_gradient_norm = None

    context = torch.enable_grad() if training_mode else torch.inference_mode()
    with context:
        for inputs, targets in data_loader:
            inputs = inputs.to(
                device=device, dtype=dtype, non_blocking=True
            )
            targets = targets.to(
                device=device, dtype=dtype, non_blocking=True
            )

            if training_mode:
                optimizer.zero_grad(set_to_none=True)

            predictions = model(inputs)
            loss = loss_function(predictions, targets)
            per_sample_loss = unreduced_per_sample_loss(
                predictions, targets
            )

            if not torch.isfinite(loss):
                phase = "training" if training_mode else "validation"
                raise RuntimeError(
                    f"Non-finite loss detected during {phase}"
                )
            if not torch.isfinite(per_sample_loss).all():
                raise RuntimeError("Non-finite per-sample loss detected")

            if training_mode:
                loss.backward()
                if not gradients_are_finite(model):
                    raise RuntimeError("Non-finite gradient detected")

                if gradient_clip_norm is not None:
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=gradient_clip_norm,
                        error_if_nonfinite=True,
                    )
                    value = float(gradient_norm.detach().cpu().item())
                    gradient_norm_sum += value
                    gradient_step_count += 1
                    if maximum_gradient_norm is None or value > maximum_gradient_norm:
                        maximum_gradient_norm = value

                optimizer.step()

            per_sample_loss_sum += float(
                per_sample_loss.detach().sum().cpu().item()
            )
            processed_sample_count += int(inputs.shape[0])

    if processed_sample_count == 0:
        raise RuntimeError(
            "DataLoader produced no samples. Check batch size and drop_last."
        )

    epoch_loss = per_sample_loss_sum / processed_sample_count
    if not np.isfinite(epoch_loss):
        raise RuntimeError("Epoch loss is non-finite")

    mean_gradient_norm = (
        gradient_norm_sum / gradient_step_count
        if gradient_step_count > 0
        else None
    )
    return {
        "mean_per_sample_loss": float(epoch_loss),
        "processed_samples": processed_sample_count,
        "batch_reduction": batch_reduction,
        "mean_gradient_norm_before_clipping": mean_gradient_norm,
        "maximum_gradient_norm_before_clipping": maximum_gradient_norm,
    }


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    validation_loss,
    complete_config,
    model_configuration,
    repository_root,
    config_path,
    preprocessed_dataset_file,
    cpu_thread_configuration,
):
    checkpoint = {
        "checkpoint_format_version": 1,
        "created_at_utc": utc_now_iso(),
        "epoch": int(epoch),
        "validation_loss": float(validation_loss),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_configuration": model_configuration,
        "training_configuration": public_config_copy(complete_config),
        "cpu_threads": cpu_thread_configuration,
        "source_files": {
            "config": safe_public_path(config_path, repository_root),
            "preprocessed_dataset": safe_public_path(
                preprocessed_dataset_file, repository_root
            ),
        },
        "source_sha256": {
            "config": sha256_file(config_path),
            "preprocessed_dataset": sha256_file(
                preprocessed_dataset_file
            ),
        },
        "torch_rng_state": torch.get_rng_state(),
        "numpy_random_state": np.random.get_state(),
        "python_random_state": random.getstate(),
    }
    if torch.cuda.is_available():
        checkpoint["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    torch.save(checkpoint, path)


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main():
    arguments = parse_arguments()
    config_path = arguments.config.expanduser().resolve()
    config = load_json(config_path)
    system_name = validate_config(config)
    config_directory = config_path.parent
    repository_root = get_repository_root(config, config_path)
    training_config = config["training"]
    paths = config["paths"]

    preprocessed_dataset_file = resolve_path(
        paths["preprocessed_dataset_npz"], config_directory
    )
    model_directory = ensure_directory(
        resolve_path(paths["model_output_directory"], config_directory)
    )
    result_directory = ensure_directory(
        resolve_path(paths["result_output_directory"], config_directory)
    )
    if not preprocessed_dataset_file.is_file():
        raise FileNotFoundError(
            "Preprocessed dataset was not found: "
            f"{safe_public_path(preprocessed_dataset_file, repository_root)}"
        )

    checkpoint_config = training_config["checkpoint"]
    checkpoint_filename = str(
        checkpoint_config.get("filename", "best_model.pt")
    )
    if Path(checkpoint_filename).name != checkpoint_filename:
        raise ValueError(
            "checkpoint.filename must be a file name without directories"
        )

    best_checkpoint = model_directory / checkpoint_filename
    final_checkpoint = model_directory / "final_model.pt"
    selected_checkpoint = model_directory / "selected_model.pt"
    history_file = result_directory / "training_history.csv"
    summary_file = result_directory / "training_summary.json"
    config_copy = model_directory / "training_config.json"

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

    random_seed = validate_nonnegative_integer(
        "training.random_seed",
        training_config.get("random_seed", 1234),
    )
    deterministic_algorithms = training_config.get(
        "deterministic_algorithms", True
    )
    if not isinstance(deterministic_algorithms, bool):
        raise TypeError(
            "training.deterministic_algorithms must be true or false"
        )

    # Configure threading before DataLoader/model construction and training.
    device = select_torch_device(training_config.get("device", "auto"))
    cpu_thread_configuration = configure_cpu_threads(
        training_config, device
    )
    set_random_seeds(
        random_seed,
        deterministic_algorithms=deterministic_algorithms,
    )
    dtype_name, dtype = select_torch_dtype(
        training_config.get("dtype", "float32")
    )

    loader_bundle = build_dataloaders(
        config, config_directory=config_directory
    )
    train_loader = loader_bundle["train_loader"]
    validation_loader = loader_bundle["validation_loader"]
    dataset = loader_bundle["dataset"]

    if str(loader_bundle["device"]) != str(device):
        raise RuntimeError(
            "DataLoader and training device selection are inconsistent"
        )

    input_dimension = int(dataset["X_train"].shape[1])
    model = build_model(
        n_in=input_dimension,
        model_config=config["model"],
    ).to(device=device, dtype=dtype)

    optimizer, effective_optimizer_config = build_optimizer(
        model, training_config["optimizer"]
    )
    (
        loss_function,
        unreduced_per_sample_loss,
        batch_reduction,
        effective_loss_config,
    ) = build_loss(training_config["loss"], device, dtype)

    maximum_epochs = validate_positive_integer(
        "training.maximum_epochs",
        training_config.get("maximum_epochs", 300),
    )
    gradient_clip_setting = training_config.get("gradient_clip_norm", 1.0)
    gradient_clip_norm = (
        None
        if gradient_clip_setting is None
        else validate_positive_float(
            "training.gradient_clip_norm", gradient_clip_setting
        )
    )

    early = training_config["early_stopping"]
    if not isinstance(early, dict):
        raise TypeError("training.early_stopping must be a dictionary")
    early_stopping_enabled = early.get("enabled", True)
    if not isinstance(early_stopping_enabled, bool):
        raise TypeError("early_stopping.enabled must be true or false")
    monitor = str(early.get("monitor", "validation_loss"))
    if monitor != "validation_loss":
        raise ValueError("Only validation_loss monitoring is supported")
    mode = str(early.get("mode", "min"))
    if mode != "min":
        raise ValueError("Only early_stopping.mode='min' is supported")
    patience = validate_positive_integer(
        "early_stopping.patience", early.get("patience", 30)
    )
    minimum_delta = validate_nonnegative_float(
        "early_stopping.minimum_delta", early.get("minimum_delta", 0.0)
    )
    restore_best_weights = early.get("restore_best_weights", True)
    if not isinstance(restore_best_weights, bool):
        raise TypeError(
            "early_stopping.restore_best_weights must be true or false"
        )

    save_best = checkpoint_config.get("save_best", True)
    if not isinstance(save_best, bool):
        raise TypeError("checkpoint.save_best must be true or false")
    if restore_best_weights and not save_best:
        raise ValueError(
            "restore_best_weights=true requires checkpoint.save_best=true"
        )

    history = []
    best_validation_loss = float("inf")
    best_epoch = None
    patience_counter = 0
    stopped_early = False
    training_start_time = time.perf_counter()

    print("=" * 80)
    print(f"System             : {system_name}")
    print(f"Device             : {device}")
    print(f"Dtype              : {dtype_name}")
    print(
        "Available CPUs     : "
        f"{cpu_thread_configuration['available_logical_cpus']}"
    )
    print(
        "PyTorch threads    : "
        f"{cpu_thread_configuration['effective_num_threads']}"
    )
    print(
        "Interop threads    : "
        f"{cpu_thread_configuration['effective_interop_threads']}"
    )
    print(f"Input dimension    : {input_dimension}")
    print(f"Maximum epochs     : {maximum_epochs}")
    print(f"Early-stop patience: {patience}")
    print(
        "Best checkpoint    : "
        f"{safe_public_path(best_checkpoint, repository_root)}"
    )
    print("=" * 80)

    for epoch_index in range(maximum_epochs):
        training_result = run_epoch(
            model=model,
            data_loader=train_loader,
            loss_function=loss_function,
            unreduced_per_sample_loss=unreduced_per_sample_loss,
            batch_reduction=batch_reduction,
            device=device,
            dtype=dtype,
            optimizer=optimizer,
            gradient_clip_norm=gradient_clip_norm,
        )
        validation_result = run_epoch(
            model=model,
            data_loader=validation_loader,
            loss_function=loss_function,
            unreduced_per_sample_loss=unreduced_per_sample_loss,
            batch_reduction=batch_reduction,
            device=device,
            dtype=dtype,
        )

        train_loss = training_result["mean_per_sample_loss"]
        validation_loss = validation_result["mean_per_sample_loss"]
        improved = (
            best_epoch is None
            or validation_loss < best_validation_loss - minimum_delta
        )

        if improved:
            best_validation_loss = validation_loss
            best_epoch = epoch_index
            patience_counter = 0
            if save_best:
                save_checkpoint(
                    best_checkpoint,
                    model,
                    optimizer,
                    epoch_index,
                    validation_loss,
                    config,
                    model.configuration(),
                    repository_root,
                    config_path,
                    preprocessed_dataset_file,
                    cpu_thread_configuration,
                )
        else:
            patience_counter += 1

        history.append(
            {
                "epoch": int(epoch_index),
                "train_loss": float(train_loss),
                "validation_loss": float(validation_loss),
                "train_samples_seen": training_result["processed_samples"],
                "validation_samples_seen": validation_result[
                    "processed_samples"
                ],
                "mean_gradient_norm_before_clipping": training_result[
                    "mean_gradient_norm_before_clipping"
                ],
                "maximum_gradient_norm_before_clipping": training_result[
                    "maximum_gradient_norm_before_clipping"
                ],
                "best_validation_loss_so_far": float(best_validation_loss),
                "best_epoch_so_far": int(best_epoch),
                "patience_counter": int(patience_counter),
                "improved": bool(improved),
            }
        )

        if epoch_index % 20 == 0 or epoch_index == maximum_epochs - 1 or improved:
            print(
                f"Epoch {epoch_index:4d} | "
                f"train loss={train_loss:.6e} | "
                f"val loss={validation_loss:.6e} | "
                f"best={best_validation_loss:.6e}"
            )

        if early_stopping_enabled and patience_counter >= patience:
            stopped_early = True
            print(f"Early stopping triggered at epoch {epoch_index}")
            break

    elapsed_seconds = float(time.perf_counter() - training_start_time)
    if not history:
        raise RuntimeError("Training completed without any epochs")

    save_checkpoint(
        final_checkpoint,
        model,
        optimizer,
        history[-1]["epoch"],
        history[-1]["validation_loss"],
        config,
        model.configuration(),
        repository_root,
        config_path,
        preprocessed_dataset_file,
        cpu_thread_configuration,
    )

    if restore_best_weights:
        if not best_checkpoint.is_file():
            raise RuntimeError("No best checkpoint exists")
        best_checkpoint_data = load_checkpoint(best_checkpoint, device)
        model.load_state_dict(
            best_checkpoint_data["model_state_dict"], strict=True
        )
        shutil.copy2(best_checkpoint, selected_checkpoint)
        selected_checkpoint_source = "best"
    else:
        shutil.copy2(final_checkpoint, selected_checkpoint)
        selected_checkpoint_source = "final"

    with history_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    write_json(config_copy, public_config_copy(config))

    summary = {
        "created_at_utc": utc_now_iso(),
        "system": config["system"],
        "config_file": safe_public_path(config_path, repository_root),
        "config_sha256": sha256_file(config_path),
        "preprocessed_dataset_npz": safe_public_path(
            preprocessed_dataset_file, repository_root
        ),
        "preprocessed_dataset_sha256": sha256_file(
            preprocessed_dataset_file
        ),
        "device": str(device),
        "dtype": dtype_name,
        "cpu_threads": cpu_thread_configuration,
        "random_seed": random_seed,
        "deterministic_algorithms": deterministic_algorithms,
        "model": model.configuration(),
        "parameter_counts": model.parameter_counts(),
        "data_loader": dataloader_summary(loader_bundle),
        "optimizer": effective_optimizer_config,
        "loss": effective_loss_config,
        "training": {
            "maximum_epochs": maximum_epochs,
            "epochs_completed": len(history),
            "last_epoch": history[-1]["epoch"],
            "gradient_clip_norm": gradient_clip_norm,
            "stopped_early": stopped_early,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "elapsed_seconds": elapsed_seconds,
        },
        "early_stopping": {
            "enabled": early_stopping_enabled,
            "monitor": monitor,
            "mode": mode,
            "patience": patience,
            "minimum_delta": minimum_delta,
            "restore_best_weights": restore_best_weights,
        },
        "checkpoints": {
            "best": (
                safe_public_path(best_checkpoint, repository_root)
                if best_checkpoint.is_file()
                else None
            ),
            "final": safe_public_path(final_checkpoint, repository_root),
            "selected_for_evaluation_and_figures": safe_public_path(
                selected_checkpoint, repository_root
            ),
            "selected_source": selected_checkpoint_source,
            "selected_sha256": sha256_file(selected_checkpoint),
        },
        "files": {
            "history_csv": safe_public_path(history_file, repository_root),
            "summary_json": safe_public_path(summary_file, repository_root),
            "training_config_copy": safe_public_path(
                config_copy, repository_root
            ),
        },
        "sha256": {
            "history_csv": sha256_file(history_file),
            "training_config_copy": sha256_file(config_copy),
            "final_checkpoint": sha256_file(final_checkpoint),
            "selected_checkpoint": sha256_file(selected_checkpoint),
        },
        "software": software_versions(
            {
                "torch": torch.__version__,
                "cuda_version": torch.version.cuda,
            }
        ),
    }
    write_json(summary_file, summary)

    print("=" * 80)
    print("Training completed")
    print(f"Epochs completed    : {len(history)}")
    print(f"Best epoch          : {best_epoch}")
    print(f"Best validation loss: {best_validation_loss:.8e}")
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
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

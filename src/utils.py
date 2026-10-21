#!/usr/bin/env python3
"""Shared utilities for the external-field-aware XMCD neural-network workflow.

Internal calculations may use absolute paths. Public JSON/CSV output should use
``public_path`` or ``safe_public_path``. External magnetic-field conditions are
validated centrally so all workflow stages use identical conventions.
"""

import hashlib
import json
import os
import platform
import random
from datetime import datetime, timezone
from numbers import Integral, Real
from pathlib import Path

import numpy as np


# ============================================================
# JSON serialization
# ============================================================

def json_default(value):
    """Convert common pathlib and NumPy objects to strict JSON values."""
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("JSON output contains a non-finite float")
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError("JSON output contains a non-finite NumPy array")
        return value.tolist()
    if isinstance(value, (set, tuple)):
        return list(value)
    raise TypeError(f"Object is not JSON serializable: {type(value).__name__}")


def load_json(path):
    """Load UTF-8 JSON and report its line/column on syntax failure."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"JSON file was not found: {path.name}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON syntax in {path.name} at line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc


def write_json(path, data):
    """Atomically write deterministic, human-readable, strict UTF-8 JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(
                data,
                handle,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
                default=json_default,
            )
            handle.write("\n")
        temporary_path.replace(path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


# ============================================================
# Path handling
# ============================================================

def resolve_path(value, base_directory):
    """Resolve an absolute or configuration-relative path without requiring existence."""
    if value is None:
        raise ValueError("Path value must not be None")
    path = Path(value).expanduser()
    base_directory = Path(base_directory).expanduser()
    if not path.is_absolute():
        path = base_directory / path
    return path.resolve(strict=False)


def get_repository_root(config, config_path):
    """Resolve and validate ``config.paths.repository_root``."""
    if not isinstance(config, dict):
        raise TypeError("Config must be a dictionary")
    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise TypeError("Config paths section must be a dictionary")
    value = paths.get("repository_root")
    if value is None:
        raise ValueError("Config paths must contain repository_root")
    config_path = Path(config_path).expanduser().resolve(strict=False)
    root = resolve_path(value, config_path.parent)
    if not root.is_dir():
        raise NotADirectoryError("Configured repository root does not exist")
    missing = [entry for entry in ("src", "configs") if not (root / entry).exists()]
    if missing:
        raise FileNotFoundError(
            f"Configured repository root is missing required entries: {missing}"
        )
    if not (root / "src").is_dir() or not (root / "configs").is_dir():
        raise NotADirectoryError("Repository entries src and configs must be directories")
    return root


def public_path(path, repository_root):
    """Return a repository-relative POSIX path and reject external paths."""
    if path is None:
        raise ValueError("Path must not be None")
    path = Path(path).expanduser().resolve(strict=False)
    root = Path(repository_root).expanduser().resolve(strict=False)
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError("The requested path is outside the repository root") from exc
    return relative.as_posix()


def safe_public_path(path, repository_root):
    """Return repository-relative path, or only basename for an external path."""
    if path is None:
        return None
    try:
        return public_path(path, repository_root)
    except ValueError:
        return Path(path).name


def public_paths(paths, repository_root, allow_external=False):
    """Convert a mapping of paths to public-safe strings."""
    if not hasattr(paths, "items"):
        raise TypeError("paths must be a mapping")
    return {
        name: (
            None if path is None
            else safe_public_path(path, repository_root)
            if allow_external
            else public_path(path, repository_root)
        )
        for name, path in paths.items()
    }


def ensure_directory(path):
    """Create a directory if needed and return its absolute path."""
    path = Path(path).expanduser().resolve(strict=False)
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"A non-directory entry exists: {path.name}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def check_output_files(paths, overwrite=False, label="Output files"):
    """Reject or remove pre-existing output files; never remove directories."""
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be true or false")
    normalized = [Path(path).expanduser().resolve(strict=False) for path in paths]
    if not normalized:
        raise ValueError("No output files were specified")
    if len({str(path) for path in normalized}) != len(normalized):
        raise ValueError(f"{label} contains duplicate paths")
    directories = [
        path for path in normalized
        if path.exists() and path.is_dir() and not path.is_symlink()
    ]
    if directories:
        raise IsADirectoryError(
            f"{label} unexpectedly include directories: "
            + ", ".join(path.name for path in directories)
        )
    existing = [path for path in normalized if path.exists() or path.is_symlink()]
    if existing and not overwrite:
        listing = "\n".join(f"  {path.name}" for path in existing)
        raise FileExistsError(f"{label} already exist. Use --overwrite:\n{listing}")
    if overwrite:
        for path in existing:
            path.unlink()


# ============================================================
# External magnetic-field validation
# ============================================================

def validate_nonnegative_real(name, value):
    """Validate a finite, nonnegative real number."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric, not bool")
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def normalize_direction(direction, name="external magnetic-field direction"):
    """Validate and normalize a three-component direction vector."""
    values = np.asarray(direction, dtype=np.float64)
    if values.shape != (3,):
        raise ValueError(f"{name} must contain exactly three values")
    ensure_finite(name, values)
    norm = float(np.linalg.norm(values))
    if norm <= 0.0:
        raise ValueError(f"{name} must be nonzero")
    return values / norm


def validate_external_magnetic_field_config(config, required=True):
    """Normalize ``config.external_magnetic_field``.

    Returns a JSON-safe dictionary with ``enabled``, ``magnitude_t``,
    normalized ``direction``, ``tolerance_t``, and ``is_model_input=False``.
    """
    if not isinstance(config, dict):
        raise TypeError("Configuration must be a dictionary")
    if "external_magnetic_field" not in config:
        if required:
            raise ValueError("Config must contain external_magnetic_field")
        return None
    field = config["external_magnetic_field"]
    if not isinstance(field, dict):
        raise TypeError("external_magnetic_field must be a dictionary")
    for key in ("enabled", "magnitude_t", "direction"):
        if key not in field:
            raise ValueError(f"Missing external_magnetic_field.{key}")
    if not isinstance(field["enabled"], bool):
        raise TypeError("external_magnetic_field.enabled must be true or false")
    magnitude = validate_nonnegative_real(
        "external_magnetic_field.magnitude_t", field["magnitude_t"]
    )
    tolerance = validate_nonnegative_real(
        "external_magnetic_field.tolerance_t", field.get("tolerance_t", 1e-9)
    )
    direction = normalize_direction(
        field["direction"], "external_magnetic_field.direction"
    )
    if not field["enabled"] and not np.isclose(
        magnitude, 0.0, rtol=0.0, atol=tolerance
    ):
        raise ValueError(
            "external_magnetic_field.magnitude_t must be zero when enabled=false"
        )
    return {
        "enabled": field["enabled"],
        "magnitude_t": magnitude,
        "direction": direction.tolist(),
        "tolerance_t": tolerance,
        "is_model_input": False,
    }


def validate_external_magnetic_field_values(
    values,
    expected_field,
    name="B_ext_T",
    allow_empty=False,
):
    """Validate per-sample field values against a normalized field config."""
    if not isinstance(expected_field, dict):
        raise TypeError("expected_field must be a normalized dictionary")
    values = ensure_finite(name, np.asarray(values, dtype=np.float64), allow_empty)
    if values.size == 0:
        return values
    expected = float(expected_field["magnitude_t"])
    tolerance = float(expected_field["tolerance_t"])
    if not np.allclose(values, expected, rtol=0.0, atol=tolerance):
        difference = float(np.max(np.abs(values - expected)))
        raise ValueError(
            f"{name} differs from the configured external field; "
            f"maximum absolute difference={difference:.8e} T"
        )
    return values


def validate_external_magnetic_field_metadata(
    source_name,
    enabled,
    magnitude_t,
    direction,
    expected_field,
    values=None,
):
    """Cross-check stored field metadata and optional per-sample field values."""
    if bool(enabled) != bool(expected_field["enabled"]):
        raise ValueError(f"External-field enabled state differs in {source_name}")
    tolerance = float(expected_field["tolerance_t"])
    if not np.isclose(
        float(magnitude_t), float(expected_field["magnitude_t"]),
        rtol=0.0, atol=tolerance,
    ):
        raise ValueError(f"External-field magnitude differs in {source_name}")
    actual_direction = normalize_direction(direction, f"{source_name} field direction")
    expected_direction = np.asarray(expected_field["direction"], dtype=np.float64)
    if not np.allclose(
        actual_direction, expected_direction, rtol=0.0, atol=1e-12
    ):
        raise ValueError(f"External-field direction differs in {source_name}")
    if values is not None:
        validate_external_magnetic_field_values(
            values, expected_field, name=f"{source_name} B_ext_T"
        )
    return True


def external_magnetic_field_payload(expected_field):
    """Return NumPy arrays suitable for inclusion in NPZ files."""
    return {
        "external_field_enabled": np.asarray(
            [expected_field["enabled"]], dtype=bool
        ),
        "external_field_magnitude_t": np.asarray(
            [expected_field["magnitude_t"]], dtype=np.float64
        ),
        "external_field_direction": np.asarray(
            expected_field["direction"], dtype=np.float64
        ),
    }


def external_magnetic_field_summary(values, expected_field):
    """Build a JSON-safe statistical summary of per-sample field values."""
    values = validate_external_magnetic_field_values(values, expected_field)
    return {
        "enabled": bool(expected_field["enabled"]),
        "magnitude_t": float(expected_field["magnitude_t"]),
        "direction": list(expected_field["direction"]),
        "tolerance_t": float(expected_field["tolerance_t"]),
        "is_model_input": False,
        "values": {
            "count": int(values.size),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std_population": float(np.std(values, ddof=0)),
        },
    }


# ============================================================
# SHA-256
# ============================================================

def sha256_file(path, block_size=1024 * 1024):
    """Calculate a file's SHA-256 digest."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Cannot hash a missing file: {path.name}")
    if isinstance(block_size, bool) or not isinstance(block_size, Integral):
        raise TypeError("block_size must be an integer")
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array, decimals=None):
    """Hash array dtype, shape, and C-order bytes deterministically."""
    values = np.asarray(array)
    if values.dtype.hasobject:
        raise TypeError("Object arrays cannot be hashed deterministically")
    if decimals is not None:
        if isinstance(decimals, bool) or not isinstance(decimals, Integral):
            raise TypeError("decimals must be an integer or None")
        decimals = int(decimals)
        if decimals < 0:
            raise ValueError("decimals must be nonnegative")
        try:
            values = values.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise TypeError("Rounded hashing requires numeric values") from exc
        ensure_finite("Array to hash", values)
        values = np.round(values, decimals=decimals)
    if values.dtype.byteorder == ">" or (
        values.dtype.byteorder == "=" and not np.little_endian
    ):
        values = values.byteswap().view(values.dtype.newbyteorder("<"))
    values = values.astype(values.dtype.newbyteorder("<"), copy=False)
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("utf-8"))
    digest.update(repr(contiguous.shape).encode("utf-8"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


# ============================================================
# Time, software, and numeric validation
# ============================================================

def utc_now_iso():
    """Return the current UTC time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def software_versions(extra=None):
    """Return JSON-safe platform and scientific-software information."""
    versions = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
    }
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        versions.update({
            "torch": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "cudnn_available": bool(torch.backends.cudnn.is_available()),
            "cudnn_version": (
                torch.backends.cudnn.version()
                if torch.backends.cudnn.is_available() else None
            ),
            "mps_available": bool(
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ),
        })
    if extra is not None:
        if not hasattr(extra, "items"):
            raise TypeError("extra must be a mapping or None")
        versions.update(dict(extra))
    return versions


def ensure_finite(name, array, allow_empty=False):
    """Validate that an array contains only finite numeric values."""
    if not isinstance(allow_empty, bool):
        raise TypeError("allow_empty must be true or false")
    values = np.asarray(array)
    if values.size == 0:
        if allow_empty:
            return values
        raise ValueError(f"{name} is empty")
    if not (
        np.issubdtype(values.dtype, np.number)
        or np.issubdtype(values.dtype, np.bool_)
    ):
        raise TypeError(f"{name} must contain numeric values")
    finite = np.isfinite(values)
    if not finite.all():
        raise ValueError(
            f"{name} contains non-finite values: "
            f"NaN={int(np.isnan(values).sum())}, "
            f"+Inf={int(np.isposinf(values).sum())}, "
            f"-Inf={int(np.isneginf(values).sum())}"
        )
    return values


# ============================================================
# Reproducibility and device selection
# ============================================================

def validate_random_seed(seed):
    """Validate a NumPy/PyTorch-compatible unsigned 32-bit seed."""
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("Random seed must be an integer")
    seed = int(seed)
    if seed < 0 or seed > 2 ** 32 - 1:
        raise ValueError("Random seed must be between 0 and 2**32 - 1")
    return seed


def set_random_seeds(seed, deterministic_algorithms=False):
    """Set Python, NumPy, and PyTorch random seeds."""
    seed = validate_random_seed(seed)
    if not isinstance(deterministic_algorithms, bool):
        raise TypeError("deterministic_algorithms must be true or false")
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic_algorithms:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cuda"):
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
        except AttributeError:
            pass
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic_algorithms
        try:
            torch.backends.cudnn.allow_tf32 = False
        except AttributeError:
            pass
    torch.use_deterministic_algorithms(
        deterministic_algorithms,
        warn_only=True,
    )
    return seed


def select_torch_device(requested="auto"):
    """Select CPU, CUDA, or MPS and validate explicit requests."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for device selection") from exc
    requested = str(requested).strip().lower()
    if not requested:
        raise ValueError("Device setting must not be empty")
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps":
        if not (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            raise RuntimeError("MPS was requested but is unavailable")
        return torch.device("mps")
    if requested == "cuda" or requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        try:
            device = torch.device(requested)
        except RuntimeError as exc:
            raise ValueError("Invalid CUDA device specification") from exc
        if device.index is not None and not 0 <= device.index < torch.cuda.device_count():
            raise RuntimeError("Requested CUDA device index is unavailable")
        return device
    raise ValueError("Device must be auto, cpu, cuda, cuda:<index>, or mps")

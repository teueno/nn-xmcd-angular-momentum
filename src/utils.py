#!/usr/bin/env python3

"""
Shared utility functions for the XMCD neural-network workflow.

Internal calculations may use absolute paths. Paths written to publicly
shareable JSON and CSV files should be converted to repository-relative
paths using public_path() or safe_public_path().
"""

import hashlib
import json
import os
import platform
import random
from datetime import datetime
from datetime import timezone
from numbers import Integral
from pathlib import Path

import numpy as np


# ============================================================
# JSON serialization
# ============================================================

def json_default(
    value,
):
    """
    Convert common scientific-Python objects to JSON-safe values.

    Supported values
    ----------------
    pathlib.Path
        Converted to a POSIX-style string.

    numpy integer
        Converted to int.

    numpy floating point
        Converted to float.

    numpy Boolean
        Converted to bool.

    numpy.ndarray
        Converted to a nested list.

    set or tuple
        Converted to a list.
    """

    if isinstance(
        value,
        Path,
    ):
        return value.as_posix()

    if isinstance(
        value,
        np.integer,
    ):
        return int(
            value
        )

    if isinstance(
        value,
        np.floating,
    ):
        numeric_value = float(
            value
        )

        if not np.isfinite(
            numeric_value
        ):
            raise ValueError(
                "JSON output contains a non-finite "
                "floating-point value"
            )

        return numeric_value

    if isinstance(
        value,
        np.bool_,
    ):
        return bool(
            value
        )

    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    if isinstance(
        value,
        (
            set,
            tuple,
        ),
    ):
        return list(
            value
        )

    raise TypeError(
        "Object is not JSON serializable: "
        f"{type(value).__name__}"
    )


def load_json(
    path,
):
    """
    Load a UTF-8 JSON file.

    Parameters
    ----------
    path : str or Path
        JSON file path.

    Returns
    -------
    object
        Parsed JSON content.
    """

    path = Path(
        path
    ).expanduser()

    if not path.is_file():
        raise FileNotFoundError(
            "JSON file was not found: "
            f"{path.name}"
        )

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            return json.load(
                handle
            )

    except json.JSONDecodeError as exception:
        raise ValueError(
            "Invalid JSON syntax in "
            f"{path.name} at line "
            f"{exception.lineno}, column "
            f"{exception.colno}: "
            f"{exception.msg}"
        ) from exception


def write_json(
    path,
    data,
):
    """
    Write data as deterministic, human-readable UTF-8 JSON.

    Non-finite floating-point values are rejected because they are
    not part of the standard JSON specification.
    """

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_name(
        f".{path.name}.tmp"
    )

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                data,
                handle,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
                default=json_default,
            )

            handle.write(
                "\n"
            )

        temporary_path.replace(
            path
        )

    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()

        raise


# ============================================================
# Path handling
# ============================================================

def resolve_path(
    value,
    base_directory,
):
    """
    Resolve a path relative to a configuration-file directory.

    Parameters
    ----------
    value : str or Path
        Absolute path or path relative to base_directory.

    base_directory : str or Path
        Directory used to resolve relative paths.

    Returns
    -------
    pathlib.Path
        Absolute normalized path.

    Notes
    -----
    The path does not need to exist.
    """

    if value is None:
        raise ValueError(
            "Path value must not be None"
        )

    path = Path(
        value
    ).expanduser()

    base_directory = Path(
        base_directory
    ).expanduser()

    if not path.is_absolute():
        path = (
            base_directory
            / path
        )

    return path.resolve(
        strict=False
    )


def get_repository_root(
    config,
    config_path,
):
    """
    Obtain and validate the repository root.

    The repository root is read from:

        config["paths"]["repository_root"]

    For configuration files under configs/dataset or configs/training,
    repository_root is normally:

        "../.."

    Parameters
    ----------
    config : dict
        Loaded configuration dictionary.

    config_path : str or Path
        Path to the configuration file.

    Returns
    -------
    pathlib.Path
        Absolute repository root.
    """

    if not isinstance(
        config,
        dict,
    ):
        raise TypeError(
            "Config must be a dictionary"
        )

    if "paths" not in config:
        raise ValueError(
            "Config does not contain "
            "the paths section"
        )

    paths = config[
        "paths"
    ]

    if not isinstance(
        paths,
        dict,
    ):
        raise TypeError(
            "Config paths section must be "
            "a dictionary"
        )

    repository_root_value = (
        paths.get(
            "repository_root"
        )
    )

    if repository_root_value is None:
        raise ValueError(
            "Config paths section must contain "
            "'repository_root'. For configuration "
            "files under configs/dataset or "
            "configs/training, use '../..'."
        )

    config_path = Path(
        config_path
    ).expanduser().resolve(
        strict=False
    )

    repository_root = resolve_path(
        repository_root_value,
        config_path.parent,
    )

    if not repository_root.is_dir():
        raise NotADirectoryError(
            "Configured repository root "
            "does not exist"
        )

    required_entries = (
        "src",
        "configs",
    )

    missing_entries = [
        entry
        for entry in required_entries
        if not (
            repository_root
            / entry
        ).exists()
    ]

    if missing_entries:
        raise FileNotFoundError(
            "Configured repository root is invalid. "
            "Missing required entries: "
            f"{missing_entries}"
        )

    if not (
        repository_root
        / "src"
    ).is_dir():
        raise NotADirectoryError(
            "Repository entry 'src' "
            "must be a directory"
        )

    if not (
        repository_root
        / "configs"
    ).is_dir():
        raise NotADirectoryError(
            "Repository entry 'configs' "
            "must be a directory"
        )

    return repository_root


def public_path(
    path,
    repository_root,
):
    """
    Convert a path to a repository-relative POSIX path.

    Parameters
    ----------
    path : str or Path
        Path that must be inside repository_root.

    repository_root : str or Path
        Repository root directory.

    Returns
    -------
    str
        Repository-relative POSIX-style path.

    Raises
    ------
    ValueError
        If path is outside repository_root.

    Notes
    -----
    The exception message deliberately does not include absolute
    paths, preventing accidental exposure of a local directory
    hierarchy in logs or public metadata.
    """

    if path is None:
        raise ValueError(
            "Path must not be None"
        )

    path = Path(
        path
    ).expanduser().resolve(
        strict=False
    )

    repository_root = Path(
        repository_root
    ).expanduser().resolve(
        strict=False
    )

    try:
        relative_path = path.relative_to(
            repository_root
        )

    except ValueError as exception:
        raise ValueError(
            "The requested path is outside "
            "the repository root"
        ) from exception

    relative_text = relative_path.as_posix()

    if relative_text == ".":
        return "."

    return relative_text


def safe_public_path(
    path,
    repository_root,
):
    """
    Return a path suitable for public output.

    Paths inside the repository are represented relative to the
    repository root. For a path outside the repository, only the
    final file or directory name is returned.

    None is preserved as None.
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


def public_paths(
    paths,
    repository_root,
    allow_external=False,
):
    """
    Convert named paths to public-safe strings.

    Parameters
    ----------
    paths : mapping
        Mapping from logical names to paths.

    repository_root : str or Path
        Repository root.

    allow_external : bool, optional
        If False, an external path raises ValueError.
        If True, only the external path's final name is returned.

    Returns
    -------
    dict
        Mapping with public-safe path strings.
    """

    if not hasattr(
        paths,
        "items",
    ):
        raise TypeError(
            "paths must be a mapping"
        )

    converted = {}

    for name, path in paths.items():
        if path is None:
            converted[
                name
            ] = None

        elif allow_external:
            converted[
                name
            ] = safe_public_path(
                path,
                repository_root,
            )

        else:
            converted[
                name
            ] = public_path(
                path,
                repository_root,
            )

    return converted


def ensure_directory(
    path,
):
    """
    Create a directory if needed and return its absolute path.
    """

    path = Path(
        path
    ).expanduser().resolve(
        strict=False
    )

    if path.exists() and not path.is_dir():
        raise NotADirectoryError(
            "A non-directory entry already exists "
            f"with the requested name: {path.name}"
        )

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def check_output_files(
    paths,
    overwrite=False,
    label="Output files",
):
    """
    Check whether output files already exist.

    Parameters
    ----------
    paths : iterable of str or Path
        Expected output file paths.

    overwrite : bool
        If False, existing output files cause FileExistsError.
        If True, existing files and symbolic links are removed.

    label : str
        Description used in exceptions.

    Notes
    -----
    Existing directories are never removed. Passing a directory as
    an output file path raises IsADirectoryError.
    """

    if not isinstance(
        overwrite,
        bool,
    ):
        raise TypeError(
            "overwrite must be true or false"
        )

    normalized_paths = [
        Path(
            path
        ).expanduser().resolve(
            strict=False
        )
        for path in paths
    ]

    if not normalized_paths:
        raise ValueError(
            "No output files were specified"
        )

    normalized_text = [
        str(
            path
        )
        for path in normalized_paths
    ]

    if len(
        set(
            normalized_text
        )
    ) != len(
        normalized_text
    ):
        raise ValueError(
            f"{label} contains duplicate paths"
        )

    directory_paths = [
        path
        for path in normalized_paths
        if (
            path.exists()
            and path.is_dir()
            and not path.is_symlink()
        )
    ]

    if directory_paths:
        names = ", ".join(
            path.name
            for path in directory_paths
        )

        raise IsADirectoryError(
            f"{label} unexpectedly include "
            f"directories: {names}"
        )

    existing_paths = [
        path
        for path in normalized_paths
        if path.exists() or path.is_symlink()
    ]

    if (
        existing_paths
        and not overwrite
    ):
        listing = "\n".join(
            f"  {path.name}"
            for path in existing_paths
        )

        raise FileExistsError(
            f"{label} already exist. "
            "Use --overwrite:\n"
            f"{listing}"
        )

    if overwrite:
        for path in existing_paths:
            path.unlink()


# ============================================================
# SHA-256
# ============================================================

def sha256_file(
    path,
    block_size=1024 * 1024,
):
    """
    Calculate the SHA-256 digest of a file.
    """

    path = Path(
        path
    ).expanduser()

    if not path.is_file():
        raise FileNotFoundError(
            "Cannot hash a missing file: "
            f"{path.name}"
        )

    if isinstance(
        block_size,
        bool,
    ) or not isinstance(
        block_size,
        Integral,
    ):
        raise TypeError(
            "block_size must be an integer"
        )

    block_size = int(
        block_size
    )

    if block_size <= 0:
        raise ValueError(
            "block_size must be positive"
        )

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        while True:
            block = handle.read(
                block_size
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def sha256_array(
    array,
    decimals=None,
):
    """
    Calculate a deterministic SHA-256 digest of a NumPy array.

    Parameters
    ----------
    array : array-like
        Numeric array.

    decimals : int or None
        If provided, values are converted to float64 and rounded
        before hashing. This is useful for identifying numerically
        equivalent spectra within a configured precision.

    Returns
    -------
    str
        SHA-256 hexadecimal digest.
    """

    values = np.asarray(
        array
    )

    if values.dtype.hasobject:
        raise TypeError(
            "Object arrays cannot be hashed "
            "deterministically"
        )

    if decimals is not None:
        if isinstance(
            decimals,
            bool,
        ) or not isinstance(
            decimals,
            Integral,
        ):
            raise TypeError(
                "decimals must be an integer or None"
            )

        decimals = int(
            decimals
        )

        if decimals < 0:
            raise ValueError(
                "decimals must be nonnegative"
            )

        try:
            values = values.astype(
                np.float64
            )

        except (
            TypeError,
            ValueError,
        ) as exception:
            raise TypeError(
                "Rounded array hashing requires "
                "numeric values"
            ) from exception

        if not np.isfinite(
            values
        ).all():
            raise ValueError(
                "Array contains non-finite values"
            )

        values = np.round(
            values,
            decimals=decimals,
        )

    # Normalize non-native byte order so the same numerical array
    # produces the same digest on common little- and big-endian hosts.
    if (
        values.dtype.byteorder == ">"
        or (
            values.dtype.byteorder == "="
            and not np.little_endian
        )
    ):
        values = values.byteswap().view(
            values.dtype.newbyteorder("<")
        )

    normalized_dtype = values.dtype.newbyteorder(
        "<"
    )

    values = values.astype(
        normalized_dtype,
        copy=False,
    )

    contiguous = np.ascontiguousarray(
        values
    )

    digest = hashlib.sha256()

    digest.update(
        contiguous.dtype.str.encode(
            "utf-8"
        )
    )

    digest.update(
        repr(
            contiguous.shape
        ).encode(
            "utf-8"
        )
    )

    digest.update(
        contiguous.tobytes(
            order="C"
        )
    )

    return digest.hexdigest()


# ============================================================
# Time and software information
# ============================================================

def utc_now_iso():
    """
    Return the current UTC time in ISO 8601 format.
    """

    return datetime.now(
        timezone.utc
    ).isoformat()


def software_versions(
    extra=None,
):
    """
    Return JSON-serializable software and platform information.

    Local executable paths are intentionally omitted.
    """

    versions = {
        "python": (
            platform.python_version()
        ),

        "python_implementation": (
            platform.python_implementation()
        ),

        "platform": (
            platform.platform()
        ),

        "machine": (
            platform.machine()
        ),

        "numpy": (
            np.__version__
        ),
    }

    try:
        import torch

    except ImportError:
        torch = None

    if torch is not None:
        versions.update(
            {
                "torch": (
                    torch.__version__
                ),

                "cuda_version": (
                    torch.version.cuda
                ),

                "cuda_available": bool(
                    torch.cuda.is_available()
                ),

                "cudnn_available": bool(
                    torch.backends.cudnn.is_available()
                ),

                "cudnn_version": (
                    torch.backends.cudnn.version()
                    if torch.backends.cudnn.is_available()
                    else None
                ),

                "mps_available": bool(
                    hasattr(
                        torch.backends,
                        "mps",
                    )
                    and torch.backends.mps.is_available()
                ),
            }
        )

    if extra is not None:
        if not hasattr(
            extra,
            "items",
        ):
            raise TypeError(
                "extra must be a mapping or None"
            )

        versions.update(
            dict(
                extra
            )
        )

    return versions


# ============================================================
# Numeric validation
# ============================================================

def ensure_finite(
    name,
    array,
    allow_empty=False,
):
    """
    Validate that an array contains only finite numeric values.

    Parameters
    ----------
    name : str
        Name used in the exception message.

    array : array-like
        Numeric values to validate.

    allow_empty : bool
        If False, empty arrays are rejected.

    Returns
    -------
    numpy.ndarray
        NumPy view of the validated values.
    """

    if not isinstance(
        allow_empty,
        bool,
    ):
        raise TypeError(
            "allow_empty must be true or false"
        )

    values = np.asarray(
        array
    )

    if (
        values.size == 0
        and not allow_empty
    ):
        raise ValueError(
            f"{name} is empty"
        )

    if values.size == 0:
        return values

    if not (
        np.issubdtype(
            values.dtype,
            np.number,
        )
        or np.issubdtype(
            values.dtype,
            np.bool_,
        )
    ):
        raise TypeError(
            f"{name} must contain numeric values"
        )

    finite_mask = np.isfinite(
        values
    )

    if not finite_mask.all():
        nan_count = int(
            np.isnan(
                values
            ).sum()
        )

        positive_inf_count = int(
            np.isposinf(
                values
            ).sum()
        )

        negative_inf_count = int(
            np.isneginf(
                values
            ).sum()
        )

        raise ValueError(
            f"{name} contains non-finite values: "
            f"NaN={nan_count}, "
            f"+Inf={positive_inf_count}, "
            f"-Inf={negative_inf_count}"
        )

    return values


# ============================================================
# Reproducibility
# ============================================================

def validate_random_seed(
    seed,
):
    """
    Validate a random seed suitable for NumPy and PyTorch.
    """

    if isinstance(
        seed,
        bool,
    ) or not isinstance(
        seed,
        Integral,
    ):
        raise TypeError(
            "Random seed must be an integer"
        )

    seed = int(
        seed
    )

    if seed < 0:
        raise ValueError(
            "Random seed must be nonnegative"
        )

    # NumPy's legacy global RNG requires a 32-bit unsigned seed.
    if seed > (
        2 ** 32
        - 1
    ):
        raise ValueError(
            "Random seed must not exceed "
            "2**32 - 1"
        )

    return seed


def set_random_seeds(
    seed,
    deterministic_algorithms=False,
):
    """
    Set Python, NumPy, and PyTorch random seeds.

    Parameters
    ----------
    seed : int
        Nonnegative seed not exceeding 2**32 - 1.

    deterministic_algorithms : bool
        Request deterministic PyTorch algorithms where available.

    Returns
    -------
    int
        Validated seed.
    """

    seed = validate_random_seed(
        seed
    )

    if not isinstance(
        deterministic_algorithms,
        bool,
    ):
        raise TypeError(
            "deterministic_algorithms must "
            "be true or false"
        )

    os.environ[
        "PYTHONHASHSEED"
    ] = str(
        seed
    )

    # Helps deterministic CUDA matrix multiplication on supported
    # CUDA versions. It has no effect on CPU or MPS-only systems.
    if deterministic_algorithms:
        os.environ.setdefault(
            "CUBLAS_WORKSPACE_CONFIG",
            ":4096:8",
        )

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    try:
        import torch

    except ImportError:
        return seed

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )

    if hasattr(
        torch.backends,
        "cuda",
    ):
        try:
            torch.backends.cuda.matmul.allow_tf32 = False

        except AttributeError:
            pass

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False

        torch.backends.cudnn.deterministic = (
            deterministic_algorithms
        )

        try:
            torch.backends.cudnn.allow_tf32 = False

        except AttributeError:
            pass

    torch.use_deterministic_algorithms(
        deterministic_algorithms,
        warn_only=True,
    )

    return seed


# ============================================================
# PyTorch device selection
# ============================================================

def select_torch_device(
    requested="auto",
):
    """
    Select and validate a PyTorch device.

    Supported values
    ----------------
    auto
        CUDA if available, otherwise MPS if available,
        otherwise CPU.

    cpu
        CPU device.

    cuda
        Default CUDA device.

    cuda:<index>
        Explicit CUDA device index.

    mps
        Apple Metal Performance Shaders device.
    """

    try:
        import torch

    except ImportError as exception:
        raise RuntimeError(
            "PyTorch is required for "
            "device selection"
        ) from exception

    requested = str(
        requested
    ).strip().lower()

    if not requested:
        raise ValueError(
            "Device setting must not be empty"
        )

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device(
                "cuda"
            )

        if (
            hasattr(
                torch.backends,
                "mps",
            )
            and torch.backends.mps.is_available()
        ):
            return torch.device(
                "mps"
            )

        return torch.device(
            "cpu"
        )

    if requested == "cpu":
        return torch.device(
            "cpu"
        )

    if requested == "mps":
        mps_available = bool(
            hasattr(
                torch.backends,
                "mps",
            )
            and torch.backends.mps.is_available()
        )

        if not mps_available:
            raise RuntimeError(
                "MPS was requested but "
                "is unavailable"
            )

        return torch.device(
            "mps"
        )

    if (
        requested == "cuda"
        or requested.startswith(
            "cuda:"
        )
    ):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but "
                "is unavailable"
            )

        try:
            device = torch.device(
                requested
            )

        except RuntimeError as exception:
            raise ValueError(
                "Invalid CUDA device specification"
            ) from exception

        if device.index is not None:
            device_count = (
                torch.cuda.device_count()
            )

            if (
                device.index < 0
                or device.index >= device_count
            ):
                raise RuntimeError(
                    "Requested CUDA device index "
                    "is unavailable"
                )

        return device

    raise ValueError(
        "Device must be one of: "
        "auto, cpu, cuda, cuda:<index>, or mps"
    )
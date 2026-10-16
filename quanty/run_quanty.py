#!/usr/bin/env python3

"""
Unified Quanty parameter-sweep runner for Fe, Co, and Ni.

Examples
--------

Test without running Quanty:

    python run_quanty.py \
        --config configs/Fe_d6.json \
        --limit 1 \
        --dry-run

Run one calculation:

    python run_quanty.py \
        --config configs/Fe_d6.json \
        --limit 1 \
        --overwrite

Run all calculations:

    python run_quanty.py \
        --config configs/Fe_d6.json \
        --overwrite

Resume an interrupted calculation:

    python run_quanty.py \
        --config configs/Fe_d6.json \
        --resume

Run Fe, Co, and Ni sequentially:

    python run_quanty.py \
        --config \
        configs/Fe_d6.json \
        configs/Co_d7.json \
        configs/Ni_d8.json \
        --resume
"""

import argparse
import csv
import hashlib
import itertools
import json
import platform
import re
import shutil
import subprocess
import sys
import time

from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PLACEHOLDER_PATTERN = re.compile(r"__[A-Z0-9_]+__")


# ============================================================
# Command-line arguments
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Generate and execute Quanty parameter-sweep "
            "calculations."
        )
    )

    parser.add_argument(
        "--config",
        nargs="+",
        type=Path,
        required=True,
        help="One or more JSON configuration files.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip successfully completed parameter sets.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove and regenerate existing outputs.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate Lua files and metadata without running Quanty.",
    )

    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue after an individual calculation fails.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N parameter sets.",
    )

    return parser.parse_args()


# ============================================================
# JSON utilities
# ============================================================

def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            data,
            handle,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        handle.write("\n")


# ============================================================
# Path utilities
# ============================================================

def resolve_path(value, base_directory):
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = base_directory / path

    return path.resolve()


# ============================================================
# Configuration validation
# ============================================================

def validate_config(config, config_path):
    required_top_level = {
        "system",
        "quanty_executable",
        "template_lua",
        "output_directory",
        "spec_source_directory",
        "parameters",
    }

    missing = required_top_level - set(config)

    if missing:
        raise ValueError(
            f"{config_path}: missing configuration keys: "
            f"{', '.join(sorted(missing))}"
        )

    required_system = {
        "element",
        "edge",
        "n_electrons_3d",
    }

    missing_system = required_system - set(
        config["system"]
    )

    if missing_system:
        raise ValueError(
            f"{config_path}: missing system keys: "
            f"{', '.join(sorted(missing_system))}"
        )

    required_parameters = {
        "ten_dq",
        "slater",
        "soc_2p",
        "soc_3d",
        "h_ex",
    }

    missing_parameters = required_parameters - set(
        config["parameters"]
    )

    if missing_parameters:
        raise ValueError(
            f"{config_path}: missing parameter definitions: "
            f"{', '.join(sorted(missing_parameters))}"
        )


# ============================================================
# Parameter definitions
# ============================================================

def values_from_definition(definition, parameter_name):
    """
    Convert a JSON parameter definition into a NumPy array.

    Supported formats
    -----------------

    Explicit list:

        [0.8, 1.0, 1.2]

    Explicit values dictionary:

        {
          "type": "values",
          "values": [0.8, 1.0, 1.2]
        }

    Linspace:

        {
          "type": "linspace",
          "start": 0.8,
          "stop": 1.6,
          "num": 10
        }

    Arange:

        {
          "type": "arange",
          "start": 0.0,
          "stop": 0.011,
          "step": 0.002
        }
    """

    if isinstance(definition, list):
        values = np.asarray(
            definition,
            dtype=float,
        )

    elif isinstance(definition, dict):
        definition_type = definition.get("type")

        if definition_type == "linspace":
            values = np.linspace(
                float(definition["start"]),
                float(definition["stop"]),
                int(definition["num"]),
            )

        elif definition_type == "arange":
            values = np.arange(
                float(definition["start"]),
                float(definition["stop"]),
                float(definition["step"]),
            )

        elif definition_type == "values":
            values = np.asarray(
                definition["values"],
                dtype=float,
            )

        else:
            raise ValueError(
                f"Unsupported parameter definition for "
                f"{parameter_name}: {definition_type!r}"
            )

    else:
        raise TypeError(
            f"Invalid parameter definition for "
            f"{parameter_name}: {definition!r}"
        )

    if values.size == 0:
        raise ValueError(
            f"No values specified for {parameter_name}"
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"Non-finite values detected in {parameter_name}"
        )

    return values


def build_parameter_grid(config):
    definitions = config["parameters"]

    ten_dq_values = values_from_definition(
        definitions["ten_dq"],
        "ten_dq",
    )

    slater_values = values_from_definition(
        definitions["slater"],
        "slater",
    )

    soc_2p_values = values_from_definition(
        definitions["soc_2p"],
        "soc_2p",
    )

    soc_3d_values = values_from_definition(
        definitions["soc_3d"],
        "soc_3d",
    )

    h_ex_values = values_from_definition(
        definitions["h_ex"],
        "h_ex",
    )

    parameter_grid = []

    for values in itertools.product(
        ten_dq_values,
        slater_values,
        soc_2p_values,
        soc_3d_values,
        h_ex_values,
    ):
        parameter_grid.append(
            {
                "ten_dq": float(values[0]),
                "slater": float(values[1]),
                "soc_2p": float(values[2]),
                "soc_3d": float(values[3]),
                "h_ex": float(values[4]),
            }
        )

    return parameter_grid


# ============================================================
# Prefix generation
# ============================================================

def format_float(value, decimals):
    threshold = 0.5 * 10 ** (-decimals)

    if abs(value) < threshold:
        value = 0.0

    return f"{value:.{decimals}f}"


def make_prefix(config, parameters):
    system = config["system"]

    precision = config.get(
        "filename_precision",
        {},
    )

    element = system["element"]
    edge = system.get("edge", "L23")
    n_electrons = int(
        system["n_electrons_3d"]
    )

    ten_dq_text = format_float(
        parameters["ten_dq"],
        int(precision.get("ten_dq", 4)),
    )

    slater_text = format_float(
        parameters["slater"],
        int(precision.get("slater", 4)),
    )

    soc_2p_text = format_float(
        parameters["soc_2p"],
        int(precision.get("soc_2p", 4)),
    )

    soc_3d_text = format_float(
        parameters["soc_3d"],
        int(precision.get("soc_3d", 4)),
    )

    h_ex_text = format_float(
        parameters["h_ex"],
        int(precision.get("h_ex", 6)),
    )

    return (
        f"{element}_{edge}_d{n_electrons}"
        f"__10Dq{ten_dq_text}"
        f"__Sl{slater_text}"
        f"__SOC2p{soc_2p_text}"
        f"__SOC3d{soc_3d_text}"
        f"__Hex{h_ex_text}"
    )


# ============================================================
# Lua template
# ============================================================

def render_lua_template(
    template_text,
    config,
    parameters,
    prefix,
):
    replacements = {
        "__NELECTRONS_3D__": str(
            int(
                config["system"]["n_electrons_3d"]
            )
        ),
        "__TEN_DQ__": (
            f"{parameters['ten_dq']:.12g}"
        ),
        "__SCALE_SLATER_3D__": (
            f"{parameters['slater']:.12g}"
        ),
        "__SCALE_SLATER_2P3D__": (
            f"{parameters['slater']:.12g}"
        ),
        "__SCALE_SOC_2P__": (
            f"{parameters['soc_2p']:.12g}"
        ),
        "__SCALE_SOC_3D__": (
            f"{parameters['soc_3d']:.12g}"
        ),
        "__H_EX__": (
            f"{parameters['h_ex']:.12g}"
        ),
        "__PREFIX__": prefix,
    }

    lua_text = template_text

    for token, replacement in replacements.items():
        lua_text = lua_text.replace(
            token,
            replacement,
        )

    unreplaced = sorted(
        set(
            PLACEHOLDER_PATTERN.findall(
                lua_text
            )
        )
    )

    if unreplaced:
        raise ValueError(
            "Unreplaced Lua placeholders: "
            + ", ".join(unreplaced)
        )

    return lua_text


# ============================================================
# File checksum
# ============================================================

def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


# ============================================================
# Output directories
# ============================================================

def prepare_output_directories(output_directory):
    """
    Create:

        runs/<system>/lua/
        runs/<system>/logs/
        runs/<system>/status/
        runs/<system>/metadata/
        runs/<system>/spec/
    """

    directory_names = (
        "lua",
        "logs",
        "status",
        "metadata",
        "spec",
    )

    for directory_name in directory_names:
        directory = (
            output_directory
            / directory_name
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# Parameter-grid CSV
# ============================================================

def write_parameter_grid_csv(
    path,
    config,
    parameter_grid,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "sample_index",
        "prefix",
        "element",
        "edge",
        "n_electrons_3d",
        "ten_dq",
        "slater",
        "soc_2p",
        "soc_3d",
        "h_ex",
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for index, parameters in enumerate(
            parameter_grid
        ):
            writer.writerow(
                {
                    "sample_index": index,
                    "prefix": make_prefix(
                        config,
                        parameters,
                    ),
                    "element": (
                        config["system"]["element"]
                    ),
                    "edge": (
                        config["system"]["edge"]
                    ),
                    "n_electrons_3d": (
                        config["system"][
                            "n_electrons_3d"
                        ]
                    ),
                    "ten_dq": (
                        f"{parameters['ten_dq']:.16g}"
                    ),
                    "slater": (
                        f"{parameters['slater']:.16g}"
                    ),
                    "soc_2p": (
                        f"{parameters['soc_2p']:.16g}"
                    ),
                    "soc_3d": (
                        f"{parameters['soc_3d']:.16g}"
                    ),
                    "h_ex": (
                        f"{parameters['h_ex']:.16g}"
                    ),
                }
            )


# ============================================================
# Execution summary CSV
# ============================================================

def append_summary_csv(
    path,
    result,
    parameters,
):
    fieldnames = [
        "prefix",
        "status",
        "return_code",
        "elapsed_seconds",
        "ten_dq",
        "slater",
        "soc_2p",
        "soc_3d",
        "h_ex",
        "spec_file_count",
        "error_message",
    ]

    file_exists = path.is_file()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "prefix": result["prefix"],
                "status": result["status"],
                "return_code": (
                    result["return_code"]
                ),
                "elapsed_seconds": (
                    f"{result['elapsed_seconds']:.6f}"
                ),
                "ten_dq": (
                    f"{parameters['ten_dq']:.16g}"
                ),
                "slater": (
                    f"{parameters['slater']:.16g}"
                ),
                "soc_2p": (
                    f"{parameters['soc_2p']:.16g}"
                ),
                "soc_3d": (
                    f"{parameters['soc_3d']:.16g}"
                ),
                "h_ex": (
                    f"{parameters['h_ex']:.16g}"
                ),
                "spec_file_count": len(
                    result["spec_files"]
                ),
                "error_message": (
                    result["error_message"]
                ),
            }
        )


# ============================================================
# Resume check
# ============================================================

def is_completed(
    completion_file,
    output_directory,
):
    if not completion_file.is_file():
        return False

    try:
        record = load_json(
            completion_file
        )

    except (
        OSError,
        json.JSONDecodeError,
    ):
        return False

    if record.get("status") != "success":
        return False

    spec_files = record.get(
        "spec_files",
        [],
    )

    if not spec_files:
        return False

    spec_directory = (
        output_directory
        / "spec"
    )

    for filename in spec_files:
        if not (
            spec_directory
            / filename
        ).is_file():
            return False

    return True


# ============================================================
# Remove previous outputs
# ============================================================

def remove_previous_outputs(
    prefix,
    output_directory,
    spec_source_directory,
):
    fixed_files = [
        (
            output_directory
            / "lua"
            / f"{prefix}.lua"
        ),
        (
            output_directory
            / "logs"
            / f"{prefix}.stdout.txt"
        ),
        (
            output_directory
            / "logs"
            / f"{prefix}.stderr.txt"
        ),
        (
            output_directory
            / "status"
            / f"{prefix}.json"
        ),
    ]

    for path in fixed_files:
        if path.is_file():
            path.unlink()

    spec_output_directory = (
        output_directory
        / "spec"
    )

    for path in spec_output_directory.glob(
        f"{prefix}*.spec"
    ):
        if path.is_file():
            path.unlink()

    for path in spec_source_directory.glob(
        f"{prefix}*.spec"
    ):
        if path.is_file():
            path.unlink()


# ============================================================
# Move generated .spec files
# ============================================================

def move_generated_spec_files(
    prefix,
    source_directory,
    destination_directory,
    overwrite,
):
    destination_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    matched_spec_files = sorted(
        source_directory.glob(
            f"{prefix}*.spec"
        )
    )

    moved_files = []

    for source_file in matched_spec_files:
        destination_file = (
            destination_directory
            / source_file.name
        )

        if destination_file.exists():
            if overwrite:
                destination_file.unlink()
            else:
                raise FileExistsError(
                    "Destination spectrum "
                    "already exists: "
                    f"{destination_file}"
                )

        shutil.move(
            str(source_file),
            str(destination_file),
        )

        moved_files.append(
            destination_file
        )

    return moved_files


# ============================================================
# Single Quanty calculation
# ============================================================

def run_single_parameter_set(
    quanty_executable,
    template_text,
    config,
    parameters,
    output_directory,
    spec_source_directory,
    dry_run,
    overwrite,
):
    prefix = make_prefix(
        config,
        parameters,
    )

    lua_file = (
        output_directory
        / "lua"
        / f"{prefix}.lua"
    )

    stdout_file = (
        output_directory
        / "logs"
        / f"{prefix}.stdout.txt"
    )

    stderr_file = (
        output_directory
        / "logs"
        / f"{prefix}.stderr.txt"
    )

    status_file = (
        output_directory
        / "status"
        / f"{prefix}.json"
    )

    spec_output_directory = (
        output_directory
        / "spec"
    )

    if overwrite:
        remove_previous_outputs(
            prefix,
            output_directory,
            spec_source_directory,
        )

    lua_text = render_lua_template(
        template_text,
        config,
        parameters,
        prefix,
    )

    lua_file.write_text(
        lua_text,
        encoding="utf-8",
    )

    if dry_run:
        result = {
            "prefix": prefix,
            "status": "dry-run",
            "return_code": None,
            "elapsed_seconds": 0.0,
            "lua_file": str(lua_file),
            "stdout_file": str(stdout_file),
            "stderr_file": str(stderr_file),
            "spec_files": [],
            "error_message": "",
            "parameters": parameters,
        }

        write_json(
            status_file,
            result,
        )

        return result

    start_time = time.perf_counter()

    process = subprocess.run(
        [
            str(quanty_executable),
            str(lua_file.resolve()),
        ],
        cwd=str(
            spec_source_directory
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    stdout_file.write_text(
        process.stdout,
        encoding="utf-8",
    )

    stderr_file.write_text(
        process.stderr,
        encoding="utf-8",
    )

    if process.returncode != 0:
        result = {
            "prefix": prefix,
            "status": "failed",
            "return_code": (
                process.returncode
            ),
            "elapsed_seconds": (
                elapsed_seconds
            ),
            "lua_file": str(lua_file),
            "stdout_file": str(
                stdout_file
            ),
            "stderr_file": str(
                stderr_file
            ),
            "spec_files": [],
            "error_message": (
                "Quanty returned status "
                f"{process.returncode}"
            ),
            "parameters": parameters,
        }

        write_json(
            status_file,
            result,
        )

        return result

    try:
        moved_spec_files = (
            move_generated_spec_files(
                prefix,
                spec_source_directory,
                spec_output_directory,
                overwrite,
            )
        )

    except Exception as exception:
        result = {
            "prefix": prefix,
            "status": "failed",
            "return_code": (
                process.returncode
            ),
            "elapsed_seconds": (
                elapsed_seconds
            ),
            "lua_file": str(lua_file),
            "stdout_file": str(
                stdout_file
            ),
            "stderr_file": str(
                stderr_file
            ),
            "spec_files": [],
            "error_message": str(
                exception
            ),
            "parameters": parameters,
        }

        write_json(
            status_file,
            result,
        )

        return result

    expected_minimum = int(
        config.get(
            "expected_minimum_spec_files",
            1,
        )
    )

    if (
        len(moved_spec_files)
        >= expected_minimum
    ):
        status = "success"
        error_message = ""

    else:
        status = "failed"
        error_message = (
            "Expected at least "
            f"{expected_minimum} "
            ".spec files, but found "
            f"{len(moved_spec_files)}."
        )

    result = {
        "prefix": prefix,
        "status": status,
        "return_code": (
            process.returncode
        ),
        "elapsed_seconds": (
            elapsed_seconds
        ),
        "lua_file": str(lua_file),
        "stdout_file": str(
            stdout_file
        ),
        "stderr_file": str(
            stderr_file
        ),
        "spec_files": [
            path.name
            for path in moved_spec_files
        ],
        "error_message": (
            error_message
        ),
        "parameters": parameters,
        "lua_sha256": (
            sha256_file(lua_file)
        ),
        "completed_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
    }

    write_json(
        status_file,
        result,
    )

    return result


# ============================================================
# Run metadata
# ============================================================

def save_run_metadata(
    output_directory,
    config_path,
    config,
    template_lua,
    quanty_executable,
    parameter_count,
):
    metadata = {
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "config_file": str(
            config_path
        ),
        "template_lua": str(
            template_lua
        ),
        "template_sha256": (
            sha256_file(
                template_lua
            )
        ),
        "quanty_executable": str(
            quanty_executable
        ),
        "parameter_count": (
            parameter_count
        ),
        "python_version": (
            sys.version
        ),
        "python_executable": (
            sys.executable
        ),
        "platform": (
            platform.platform()
        ),
        "numpy_version": (
            np.__version__
        ),
        "configuration": config,
    }

    write_json(
        (
            output_directory
            / "metadata"
            / "run_metadata.json"
        ),
        metadata,
    )


# ============================================================
# Execute one configuration
# ============================================================

def run_configuration(
    config_path,
    resume,
    overwrite,
    dry_run,
    keep_going,
    limit,
):
    config_path = (
        config_path
        .expanduser()
        .resolve()
    )

    config = load_json(
        config_path
    )

    validate_config(
        config,
        config_path,
    )

    config_base_directory = (
        config_path.parent
    )

    quanty_executable = resolve_path(
        config["quanty_executable"],
        config_base_directory,
    )

    template_lua = resolve_path(
        config["template_lua"],
        config_base_directory,
    )

    output_directory = resolve_path(
        config["output_directory"],
        config_base_directory,
    )

    spec_source_directory = resolve_path(
        config["spec_source_directory"],
        config_base_directory,
    )

    if not dry_run:
        if not quanty_executable.exists():
            raise FileNotFoundError(
                "Quanty executable not found: "
                f"{quanty_executable}"
            )

    if not template_lua.is_file():
        raise FileNotFoundError(
            "Lua template not found: "
            f"{template_lua}"
        )

    spec_source_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    prepare_output_directories(
        output_directory
    )

    template_text = (
        template_lua.read_text(
            encoding="utf-8"
        )
    )

    parameter_grid = (
        build_parameter_grid(
            config
        )
    )

    if limit is not None:
        if limit <= 0:
            raise ValueError(
                "--limit must be "
                "a positive integer"
            )

        parameter_grid = (
            parameter_grid[:limit]
        )

    write_parameter_grid_csv(
        (
            output_directory
            / "metadata"
            / "parameter_grid.csv"
        ),
        config,
        parameter_grid,
    )

    save_run_metadata(
        output_directory,
        config_path,
        config,
        template_lua,
        quanty_executable,
        len(parameter_grid),
    )

    summary_file = (
        output_directory
        / "metadata"
        / "execution_summary.csv"
    )

    if overwrite:
        if summary_file.exists():
            summary_file.unlink()

    print("=" * 80)
    print(
        f"Configuration : {config_path}"
    )
    print(
        "System        : "
        f"{config['system']['element']} "
        f"d{config['system']['n_electrons_3d']}"
    )
    print(
        f"Template      : {template_lua}"
    )
    print(
        f"Output        : {output_directory}"
    )
    print(
        "Spec output   : "
        f"{output_directory / 'spec'}"
    )
    print(
        "Parameter sets: "
        f"{len(parameter_grid)}"
    )
    print(
        f"Resume        : {resume}"
    )
    print(
        f"Overwrite     : {overwrite}"
    )
    print(
        f"Dry run       : {dry_run}"
    )
    print("=" * 80)

    success_count = 0
    failure_count = 0
    skipped_count = 0

    for index, parameters in enumerate(
        parameter_grid,
        start=1,
    ):
        prefix = make_prefix(
            config,
            parameters,
        )

        completion_file = (
            output_directory
            / "status"
            / f"{prefix}.json"
        )

        if resume:
            if is_completed(
                completion_file,
                output_directory,
            ):
                skipped_count += 1

                print(
                    f"[{index:5d}/"
                    f"{len(parameter_grid):5d}] "
                    f"SKIP {prefix}"
                )

                continue

        print(
            f"[{index:5d}/"
            f"{len(parameter_grid):5d}] "
            f"RUN  {prefix}"
        )

        result = run_single_parameter_set(
            quanty_executable,
            template_text,
            config,
            parameters,
            output_directory,
            spec_source_directory,
            dry_run,
            overwrite,
        )

        append_summary_csv(
            summary_file,
            result,
            parameters,
        )

        if result["status"] in {
            "success",
            "dry-run",
        }:
            success_count += 1

            print(
                "  OK: "
                f"{len(result['spec_files'])} "
                "spectrum file(s), "
                f"{result['elapsed_seconds']:.2f} s"
            )

        else:
            failure_count += 1

            print(
                "  FAILED: "
                f"{result['error_message']}",
                file=sys.stderr,
            )

            print(
                "  stdout: "
                f"{result['stdout_file']}",
                file=sys.stderr,
            )

            print(
                "  stderr: "
                f"{result['stderr_file']}",
                file=sys.stderr,
            )

            if not keep_going:
                break

    final_summary = {
        "configuration": str(
            config_path
        ),
        "system": (
            config["system"]
        ),
        "total_parameter_sets": (
            len(parameter_grid)
        ),
        "successful_or_dry_run": (
            success_count
        ),
        "failed": (
            failure_count
        ),
        "skipped": (
            skipped_count
        ),
        "finished_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
    }

    write_json(
        (
            output_directory
            / "metadata"
            / "final_summary.json"
        ),
        final_summary,
    )

    print("=" * 80)
    print(
        f"Completed : {success_count}"
    )
    print(
        f"Failed    : {failure_count}"
    )
    print(
        f"Skipped   : {skipped_count}"
    )
    print(
        f"Output    : {output_directory}"
    )
    print("=" * 80)

    return failure_count == 0


# ============================================================
# Main
# ============================================================

def main():
    args = parse_arguments()

    if args.resume and args.overwrite:
        print(
            "ERROR: --resume and --overwrite "
            "cannot be used together.",
            file=sys.stderr,
        )

        return 2

    all_successful = True

    for config_path in args.config:
        try:
            successful = run_configuration(
                config_path=config_path,
                resume=args.resume,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                keep_going=args.keep_going,
                limit=args.limit,
            )

            all_successful = (
                all_successful
                and successful
            )

        except Exception as exception:
            all_successful = False

            print(
                "ERROR while processing "
                f"{config_path}: "
                f"{exception}",
                file=sys.stderr,
            )

            if not args.keep_going:
                return 1

    if all_successful:
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
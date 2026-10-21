#!/usr/bin/env python3
"""Parallel Quanty parameter-sweep runner for Fe, Co, and Ni.

Each Quanty subprocess runs in an isolated work directory.  This prevents
concurrent Lua processes from appending to the same quanty_results_*.csv file.
Generated spectra, per-job CSV files, logs, status records, and final merged
CSV files are collected under the configured output directory.

Examples
--------
Dry run:
    python run_quanty_parallel.py --config configs/Fe_d6.json --limit 4 --dry-run

Parallel run with 8 workers:
    python run_quanty_parallel.py --config configs/Fe_d6.json --workers 8 --overwrite

Resume:
    python run_quanty_parallel.py --config configs/Fe_d6.json --workers 8 --resume

Run several systems sequentially, while each system is internally parallel:
    python run_quanty_parallel.py --config configs/Fe_d6.json configs/Co_d7.json configs/Ni_d8.json --workers 16 --resume
"""

import argparse
import csv
import hashlib
import itertools
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PLACEHOLDER_PATTERN = re.compile(r"__[A-Z0-9_]+__")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate and execute Quanty parameter sweeps in parallel."
    )
    parser.add_argument("--config", nargs="+", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel Quanty processes. 0 = automatic.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--quanty-threads", type=int, default=1,
                        help="OMP/MKL/OpenBLAS threads allowed per Quanty process.")
    parser.add_argument("--keep-work-directories", action="store_true",
                        help="Keep isolated temporary work directories.")
    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def resolve_path(value, base_directory):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_directory / path
    return path.resolve()


def validate_config(config, config_path):
    required_top_level = {
        "system", "quanty_executable", "template_lua", "output_directory",
        "spec_source_directory", "parameters",
    }
    missing = required_top_level - set(config)
    if missing:
        raise ValueError(f"{config_path}: missing keys: {', '.join(sorted(missing))}")
    required_system = {"element", "edge", "n_electrons_3d"}
    missing = required_system - set(config["system"])
    if missing:
        raise ValueError(f"{config_path}: missing system keys: {', '.join(sorted(missing))}")
    required_parameters = {"ten_dq", "slater", "soc_2p", "soc_3d", "h_ex"}
    missing = required_parameters - set(config["parameters"])
    if missing:
        raise ValueError(f"{config_path}: missing parameter definitions: {', '.join(sorted(missing))}")


def values_from_definition(definition, parameter_name):
    if isinstance(definition, list):
        values = np.asarray(definition, dtype=float)
    elif isinstance(definition, dict):
        kind = definition.get("type")
        if kind == "linspace":
            values = np.linspace(float(definition["start"]), float(definition["stop"]), int(definition["num"]))
        elif kind == "arange":
            values = np.arange(float(definition["start"]), float(definition["stop"]), float(definition["step"]))
        elif kind == "values":
            values = np.asarray(definition["values"], dtype=float)
        else:
            raise ValueError(f"Unsupported definition for {parameter_name}: {kind!r}")
    else:
        raise TypeError(f"Invalid definition for {parameter_name}: {definition!r}")
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"Invalid values for {parameter_name}")
    return values


def build_parameter_grid(config):
    p = config["parameters"]
    arrays = [
        values_from_definition(p["ten_dq"], "ten_dq"),
        values_from_definition(p["slater"], "slater"),
        values_from_definition(p["soc_2p"], "soc_2p"),
        values_from_definition(p["soc_3d"], "soc_3d"),
        values_from_definition(p["h_ex"], "h_ex"),
    ]
    return [
        {"ten_dq": float(v[0]), "slater": float(v[1]),
         "soc_2p": float(v[2]), "soc_3d": float(v[3]), "h_ex": float(v[4])}
        for v in itertools.product(*arrays)
    ]


def format_float(value, decimals):
    if abs(value) < 0.5 * 10 ** (-decimals):
        value = 0.0
    return f"{value:.{decimals}f}"


def make_prefix(config, parameters):
    system = config["system"]
    precision = config.get("filename_precision", {})
    return (
        f"{system['element']}_{system.get('edge', 'L23')}_d{int(system['n_electrons_3d'])}"
        f"__10Dq{format_float(parameters['ten_dq'], int(precision.get('ten_dq', 4)))}"
        f"__Sl{format_float(parameters['slater'], int(precision.get('slater', 4)))}"
        f"__SOC2p{format_float(parameters['soc_2p'], int(precision.get('soc_2p', 4)))}"
        f"__SOC3d{format_float(parameters['soc_3d'], int(precision.get('soc_3d', 4)))}"
        f"__Hex{format_float(parameters['h_ex'], int(precision.get('h_ex', 6)))}"
    )


def render_lua_template(template_text, config, parameters, prefix):
    replacements = {
        "__NELECTRONS_3D__": str(int(config["system"]["n_electrons_3d"])),
        "__TEN_DQ__": f"{parameters['ten_dq']:.12g}",
        "__SCALE_SLATER_3D__": f"{parameters['slater']:.12g}",
        "__SCALE_SLATER_2P3D__": f"{parameters['slater']:.12g}",
        "__SCALE_SOC_2P__": f"{parameters['soc_2p']:.12g}",
        "__SCALE_SOC_3D__": f"{parameters['soc_3d']:.12g}",
        "__H_EX__": f"{parameters['h_ex']:.12g}",
        "__PREFIX__": prefix,
    }
    text = template_text
    for token, replacement in replacements.items():
        text = text.replace(token, replacement)
    unreplaced = sorted(set(PLACEHOLDER_PATTERN.findall(text)))
    if unreplaced:
        raise ValueError("Unreplaced Lua placeholders: " + ", ".join(unreplaced))
    return text


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_output_directories(output_directory):
    for name in ("lua", "logs", "status", "metadata", "spec", "job_results", "work"):
        (output_directory / name).mkdir(parents=True, exist_ok=True)


def write_parameter_grid_csv(path, config, parameter_grid):
    fields = ["sample_index", "prefix", "element", "edge", "n_electrons_3d",
              "ten_dq", "slater", "soc_2p", "soc_3d", "h_ex"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, p in enumerate(parameter_grid):
            writer.writerow({
                "sample_index": index, "prefix": make_prefix(config, p),
                "element": config["system"]["element"], "edge": config["system"]["edge"],
                "n_electrons_3d": config["system"]["n_electrons_3d"],
                **{key: f"{p[key]:.16g}" for key in ("ten_dq", "slater", "soc_2p", "soc_3d", "h_ex")},
            })


def write_summary_csv(path, completed_rows):
    fields = ["sample_index", "prefix", "status", "return_code", "elapsed_seconds",
              "ten_dq", "slater", "soc_2p", "soc_3d", "h_ex",
              "spec_file_count", "result_csv_file", "error_message"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, result, p in sorted(completed_rows, key=lambda row: row[0]):
            writer.writerow({
                "sample_index": index, "prefix": result["prefix"], "status": result["status"],
                "return_code": result["return_code"],
                "elapsed_seconds": f"{result['elapsed_seconds']:.6f}",
                **{key: f"{p[key]:.16g}" for key in ("ten_dq", "slater", "soc_2p", "soc_3d", "h_ex")},
                "spec_file_count": len(result["spec_files"]),
                "result_csv_file": result.get("result_csv_file", ""),
                "error_message": result["error_message"],
            })


def is_completed(completion_file, output_directory):
    if not completion_file.is_file():
        return False
    try:
        record = load_json(completion_file)
    except (OSError, json.JSONDecodeError):
        return False
    if record.get("status") != "success" or not record.get("spec_files"):
        return False
    if not all((output_directory / "spec" / name).is_file() for name in record["spec_files"]):
        return False
    result_csv = record.get("result_csv_file")
    return not result_csv or (output_directory / "job_results" / result_csv).is_file()


def remove_previous_outputs(prefix, output_directory):
    for path in [
        output_directory / "lua" / f"{prefix}.lua",
        output_directory / "logs" / f"{prefix}.stdout.txt",
        output_directory / "logs" / f"{prefix}.stderr.txt",
        output_directory / "status" / f"{prefix}.json",
        output_directory / "job_results" / f"{prefix}.csv",
    ]:
        if path.is_file():
            path.unlink()
    for path in (output_directory / "spec").glob(f"{prefix}*.spec"):
        if path.is_file():
            path.unlink()
    work = output_directory / "work" / prefix
    if work.exists():
        shutil.rmtree(work)


def atomic_move(source, destination, overwrite):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Destination exists: {destination}")
        destination.unlink()
    os.replace(source, destination)


def collect_quanty_result_csv(job_cwd, prefix, output_directory, overwrite):
    candidates = sorted(job_cwd.glob("quanty_results*.csv"))
    if not candidates:
        return ""
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple Quanty result CSV files found: {candidates}")
    destination = output_directory / "job_results" / f"{prefix}.csv"
    atomic_move(candidates[0], destination, overwrite)
    return destination.name


def merge_job_result_csvs(output_directory, system_label):
    files = sorted((output_directory / "job_results").glob("*.csv"))
    destination = output_directory / "metadata" / f"quanty_results_{system_label}.csv"
    if not files:
        return None
    header = None
    rows = []
    for path in files:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            local_header = next(reader, None)
            if local_header is None:
                continue
            if header is None:
                header = local_header
            elif local_header != header:
                raise ValueError(f"Inconsistent result CSV header: {path}")
            local_rows = list(reader)
            if len(local_rows) != 1:
                raise ValueError(f"Expected one data row in {path}, found {len(local_rows)}")
            rows.extend(local_rows)
    if header is None:
        return None
    prefix_index = header.index("Prefix") if "Prefix" in header else 0
    rows.sort(key=lambda row: row[prefix_index])
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return destination


def build_subprocess_environment(quanty_threads):
    env = os.environ.copy()
    thread_text = str(quanty_threads)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[name] = thread_text
    return env


def run_single_parameter_set(sample_index, quanty_executable, template_text, config,
                             parameters, output_directory, dry_run, overwrite,
                             quanty_threads, keep_work_directories):
    prefix = make_prefix(config, parameters)
    lua_file = output_directory / "lua" / f"{prefix}.lua"
    stdout_file = output_directory / "logs" / f"{prefix}.stdout.txt"
    stderr_file = output_directory / "logs" / f"{prefix}.stderr.txt"
    status_file = output_directory / "status" / f"{prefix}.json"
    spec_output_directory = output_directory / "spec"
    work_root = output_directory / "work" / prefix
    job_cwd = work_root / "working"
    job_runs = work_root / "runs"

    if overwrite:
        remove_previous_outputs(prefix, output_directory)
    job_cwd.mkdir(parents=True, exist_ok=True)
    job_runs.mkdir(parents=True, exist_ok=True)

    lua_text = render_lua_template(template_text, config, parameters, prefix)
    lua_file.write_text(lua_text, encoding="utf-8")
    job_lua_file = job_cwd / lua_file.name
    job_lua_file.write_text(lua_text, encoding="utf-8")

    base_result = {
        "sample_index": sample_index, "prefix": prefix, "parameters": parameters,
        "lua_file": str(lua_file), "stdout_file": str(stdout_file),
        "stderr_file": str(stderr_file), "spec_files": [], "result_csv_file": "",
    }
    if dry_run:
        result = {**base_result, "status": "dry-run", "return_code": None,
                  "elapsed_seconds": 0.0, "error_message": ""}
        write_json(status_file, result)
        if not keep_work_directories:
            shutil.rmtree(work_root, ignore_errors=True)
        return result

    start = time.perf_counter()
    try:
        process = subprocess.run(
            [str(quanty_executable), str(job_lua_file.resolve())],
            cwd=str(job_cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", check=False,
            env=build_subprocess_environment(quanty_threads),
        )
        elapsed = time.perf_counter() - start
        stdout_file.write_text(process.stdout, encoding="utf-8")
        stderr_file.write_text(process.stderr, encoding="utf-8")
        if process.returncode != 0:
            raise RuntimeError(f"Quanty returned status {process.returncode}")

        generated_specs = sorted(job_runs.glob(f"{prefix}*.spec"))
        if not generated_specs:
            # Supports templates that append a field suffix to Prefix.
            generated_specs = sorted(job_runs.glob(f"{prefix}*.spec"))
        moved = []
        for source in generated_specs:
            destination = spec_output_directory / source.name
            atomic_move(source, destination, overwrite)
            moved.append(destination)

        minimum = int(config.get("expected_minimum_spec_files", 1))
        if len(moved) < minimum:
            raise RuntimeError(f"Expected at least {minimum} .spec files, found {len(moved)}")

        result_csv_file = collect_quanty_result_csv(job_cwd, prefix, output_directory, overwrite)
        result = {
            **base_result, "status": "success", "return_code": process.returncode,
            "elapsed_seconds": elapsed, "spec_files": [p.name for p in moved],
            "result_csv_file": result_csv_file, "error_message": "",
            "lua_sha256": sha256_file(lua_file),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return_code = locals().get("process").returncode if "process" in locals() else None
        result = {**base_result, "status": "failed", "return_code": return_code,
                  "elapsed_seconds": elapsed, "error_message": str(exc)}
    write_json(status_file, result)
    if not keep_work_directories and result["status"] == "success":
        shutil.rmtree(work_root, ignore_errors=True)
    return result


def available_cpu_count():
    try:
        affinity = os.sched_getaffinity(0)
        if affinity:
            return len(affinity)
    except (AttributeError, OSError):
        pass
    return os.cpu_count() or 1


def choose_worker_count(requested, config, quanty_threads, task_count):
    if requested < 0:
        raise ValueError("--workers must be 0 or a positive integer")
    if quanty_threads <= 0:
        raise ValueError("--quanty-threads must be positive")
    cpus = available_cpu_count()
    configured = int(config.get("parallel_workers", 0))
    if requested > 0:
        workers = requested
    elif configured > 0:
        workers = configured
    else:
        workers = max(1, cpus // quanty_threads)
    return max(1, min(workers, task_count or 1))


def save_run_metadata(output_directory, config_path, config, template_lua,
                      quanty_executable, parameter_count, workers, quanty_threads):
    write_json(output_directory / "metadata" / "run_metadata.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_file": str(config_path), "template_lua": str(template_lua),
        "template_sha256": sha256_file(template_lua),
        "quanty_executable": str(quanty_executable),
        "parameter_count": parameter_count, "parallel_workers": workers,
        "quanty_threads_per_process": quanty_threads,
        "available_cpu_count": available_cpu_count(),
        "python_version": sys.version, "python_executable": sys.executable,
        "platform": platform.platform(), "numpy_version": np.__version__,
        "configuration": config,
    })


def run_configuration(config_path, args):
    config_path = config_path.expanduser().resolve()
    config = load_json(config_path)
    validate_config(config, config_path)
    base = config_path.parent
    quanty_executable = resolve_path(config["quanty_executable"], base)
    template_lua = resolve_path(config["template_lua"], base)
    output_directory = resolve_path(config["output_directory"], base)

    if not args.dry_run and not quanty_executable.exists():
        raise FileNotFoundError(f"Quanty executable not found: {quanty_executable}")
    if not template_lua.is_file():
        raise FileNotFoundError(f"Lua template not found: {template_lua}")

    prepare_output_directories(output_directory)
    template_text = template_lua.read_text(encoding="utf-8")
    parameter_grid = build_parameter_grid(config)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        parameter_grid = parameter_grid[:args.limit]

    workers = choose_worker_count(args.workers, config, args.quanty_threads, len(parameter_grid))
    write_parameter_grid_csv(output_directory / "metadata" / "parameter_grid.csv", config, parameter_grid)
    save_run_metadata(output_directory, config_path, config, template_lua,
                      quanty_executable, len(parameter_grid), workers, args.quanty_threads)

    tasks = []
    skipped = 0
    for index, parameters in enumerate(parameter_grid):
        prefix = make_prefix(config, parameters)
        status_file = output_directory / "status" / f"{prefix}.json"
        if args.resume and is_completed(status_file, output_directory):
            skipped += 1
        else:
            tasks.append((index, parameters))

    print("=" * 80)
    print(f"Configuration : {config_path}")
    print(f"System        : {config['system']['element']} d{config['system']['n_electrons_3d']}")
    print(f"Template      : {template_lua}")
    print(f"Output        : {output_directory}")
    print(f"Parameter sets: {len(parameter_grid)}")
    print(f"Pending tasks : {len(tasks)}")
    print(f"Workers       : {workers}")
    print(f"Threads/job   : {args.quanty_threads}")
    print(f"Resume        : {args.resume}")
    print(f"Overwrite     : {args.overwrite}")
    print(f"Dry run       : {args.dry_run}")
    print("=" * 80)

    completed_rows = []
    failures = 0
    stop_submitting = False
    start_all = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="quanty") as executor:
        future_map = {
            executor.submit(
                run_single_parameter_set, index, quanty_executable, template_text,
                config, parameters, output_directory, args.dry_run, args.overwrite,
                args.quanty_threads, args.keep_work_directories,
            ): (index, parameters)
            for index, parameters in tasks
        }
        try:
            for completed_number, future in enumerate(as_completed(future_map), start=1):
                index, parameters = future_map[future]
                try:
                    result = future.result()
                except Exception as exc:
                    prefix = make_prefix(config, parameters)
                    result = {
                        "prefix": prefix, "status": "failed", "return_code": None,
                        "elapsed_seconds": 0.0, "spec_files": [],
                        "result_csv_file": "", "error_message": str(exc),
                    }
                completed_rows.append((index, result, parameters))
                marker = "OK" if result["status"] in {"success", "dry-run"} else "FAILED"
                print(f"[{completed_number:5d}/{len(tasks):5d}] {marker:6s} "
                      f"{result['prefix']} ({result['elapsed_seconds']:.2f} s)")
                if marker == "FAILED":
                    failures += 1
                    print(f"  {result['error_message']}", file=sys.stderr)
                    if not args.keep_going:
                        stop_submitting = True
                        for pending in future_map:
                            if not pending.done():
                                pending.cancel()
                        break
        except KeyboardInterrupt:
            print("\nInterrupted. Cancelling pending jobs...", file=sys.stderr)
            for future in future_map:
                future.cancel()
            raise

    summary_file = output_directory / "metadata" / "execution_summary.csv"
    write_summary_csv(summary_file, completed_rows)

    system_label = f"{config['system']['element']}_d{config['system']['n_electrons_3d']}"
    merged_csv = None
    if not args.dry_run:
        merged_csv = merge_job_result_csvs(output_directory, system_label)

    success_count = sum(r[1]["status"] in {"success", "dry-run"} for r in completed_rows)
    total_elapsed = time.perf_counter() - start_all
    final_summary = {
        "configuration": str(config_path), "system": config["system"],
        "total_parameter_sets": len(parameter_grid), "submitted_this_run": len(tasks),
        "successful_or_dry_run": success_count, "failed": failures,
        "skipped": skipped, "parallel_workers": workers,
        "quanty_threads_per_process": args.quanty_threads,
        "elapsed_seconds": total_elapsed,
        "merged_quanty_results_csv": str(merged_csv) if merged_csv else None,
        "stopped_after_failure": stop_submitting,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_directory / "metadata" / "final_summary.json", final_summary)

    print("=" * 80)
    print(f"Completed : {success_count}")
    print(f"Failed    : {failures}")
    print(f"Skipped   : {skipped}")
    print(f"Elapsed   : {total_elapsed:.2f} s")
    print(f"Output    : {output_directory}")
    if merged_csv:
        print(f"Merged CSV: {merged_csv}")
    print("=" * 80)
    return failures == 0 and not stop_submitting


def main():
    args = parse_arguments()
    if args.resume and args.overwrite:
        print("ERROR: --resume and --overwrite cannot be used together.", file=sys.stderr)
        return 2
    all_successful = True
    for config_path in args.config:
        try:
            successful = run_configuration(config_path, args)
            all_successful = all_successful and successful
        except KeyboardInterrupt:
            return 130
        except Exception as exc:
            all_successful = False
            print(f"ERROR while processing {config_path}: {exc}", file=sys.stderr)
            if not args.keep_going:
                return 1
    return 0 if all_successful else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run first-pass Ultimate generation for all configured models, then split variants."""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import load_domain_specs, load_model_specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="final")
    parser.add_argument("--domains", nargs="+", default=None)
    parser.add_argument("--models-config", type=Path, default=PROJECT_ROOT / "configs" / "models.yaml")
    parser.add_argument("--datasets-config", type=Path, default=PROJECT_ROOT / "configs" / "datasets.yaml")
    parser.add_argument("--source-shard-root", type=Path, default=PROJECT_ROOT / "data" / "source_shards")
    parser.add_argument("--generated-root", type=Path, default=None)
    parser.add_argument("--log-root", type=Path, default=PROJECT_ROOT / "logs")
    parser.add_argument("--sample-count", type=int, default=30)
    parser.add_argument("--max-candidates", type=int, default=45)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--fact-max-tokens", type=int, default=1600)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument(
        "--schedule",
        choices=["sequential", "parallel", "hybrid"],
        default="sequential",
        help=(
            "Execution schedule. hybrid starts gemini/claude/deepseek/gpt-3.5 "
            "together, then runs gpt-4 and gpt-5.5 after gpt-3.5 finishes."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--echo-model-logs",
        action="store_true",
        help="Also echo per-model subprocess output to the runner console.",
    )
    parser.add_argument("--no-split", action="store_true")
    parser.add_argument("--strict-split", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.generated_root is None:
        args.generated_root = PROJECT_ROOT / "runs" / args.run_id / "generated"
    args.log_dir = args.log_root / args.run_id
    args.log_dir.mkdir(parents=True, exist_ok=True)
    domains = args.domains or list(load_domain_specs(args.datasets_config))
    models = load_model_specs(args.models_config)
    command_items = [(spec["model_label"], generation_command(args, domains, spec)) for spec in models]
    if args.dry_run:
        for _, command in command_items:
            print(" ".join(command))
        return
    schedule = "parallel" if args.parallel else args.schedule
    if schedule == "parallel":
        run_parallel(command_items, args)
    elif schedule == "hybrid":
        run_hybrid(command_items, args)
    else:
        for label, command in command_items:
            run_command(label, command, args)
    if not args.no_split:
        run_command("split_variants", split_command(args, domains), args)


def generation_command(args: argparse.Namespace, domains: list[str], spec: dict[str, Any]) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "generate_ultimate_controlled_fdg_dataset.py"),
        "--provider",
        spec["provider"],
        "--model",
        spec["model"],
        "--model-label",
        spec["model_label"],
        "--datasets",
        *domains,
        "--sample-count",
        str(args.sample_count),
        "--max-candidates",
        str(args.max_candidates),
        "--max-tokens",
        str(spec.get("max_tokens", args.max_tokens)),
        "--fact-max-tokens",
        str(spec.get("fact_max_tokens", args.fact_max_tokens)),
        "--shard-root",
        str(args.source_shard_root / args.run_id),
        "--output-root",
        str(args.generated_root),
        "--datasets-config",
        str(args.datasets_config),
        "--verbose-progress",
    ]
    if spec.get("openai_reasoning_effort"):
        command.extend(["--openai-reasoning-effort", str(spec["openai_reasoning_effort"])])
    if spec.get("openai_verbosity"):
        command.extend(["--openai-verbosity", str(spec["openai_verbosity"])])
    if spec.get("deepseek_reasoning_effort"):
        command.extend(["--deepseek-reasoning-effort", str(spec["deepseek_reasoning_effort"])])
    if spec.get("deepseek_thinking"):
        command.extend(["--deepseek-thinking", str(spec["deepseek_thinking"])])
    return command


def split_command(args: argparse.Namespace, domains: list[str]) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "split_variant_shards.py"),
        "--run-id",
        args.run_id,
        "--domains",
        *domains,
        "--generated-root",
        str(args.generated_root),
        "--source-shard-root",
        str(args.source_shard_root),
    ]
    if args.strict_split:
        command.append("--strict")
    return command


def run_parallel(command_items: list[tuple[str, list[str]]], args: argparse.Namespace) -> None:
    with ThreadPoolExecutor(max_workers=len(command_items)) as pool:
        futures = {pool.submit(run_command, label, command, args): label for label, command in command_items}
        for future in as_completed(futures):
            future.result()


def run_hybrid(command_items: list[tuple[str, list[str]]], args: argparse.Namespace) -> None:
    """Run the hybrid provider schedule."""
    first_wave_labels = [
        "gemini-2.5-pro",
        "claude-sonnet-4-6",
        "deepseek-v4-pro",
        "gpt-3.5-turbo",
    ]
    openai_chain_labels = ["gpt-4", "gpt-5.5"]
    by_label = {label: command for label, command in command_items}
    first_wave = [(label, by_label[label]) for label in first_wave_labels if label in by_label]
    openai_chain = [(label, by_label[label]) for label in openai_chain_labels if label in by_label]
    scheduled_labels = set(first_wave_labels + openai_chain_labels)
    remaining = [(label, command) for label, command in command_items if label not in scheduled_labels]
    first_wave.extend(remaining)

    if not first_wave:
        for _, command in openai_chain:
            run_command(command)
        return

    with ThreadPoolExecutor(max_workers=len(first_wave)) as pool:
        futures = {label: pool.submit(run_command, label, command, args) for label, command in first_wave}
        gpt35_future = futures.get("gpt-3.5-turbo")
        if gpt35_future is not None:
            try:
                gpt35_future.result()
            except Exception:
                wait_for_background(futures, skip_label="gpt-3.5-turbo")
                raise
        for label, command in openai_chain:
            run_command(label, command, args)
        wait_for_background(futures, skip_label="gpt-3.5-turbo")


def wait_for_background(futures: dict[str, object], skip_label: str | None = None) -> None:
    for label, future in futures.items():
        if label == skip_label:
            continue
        future.result()


def run_command(label: str, command: list[str], args: argparse.Namespace) -> None:
    log_path = args.log_dir / f"{label}.log"
    print(f"[run:{label}] {' '.join(command)}", flush=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"[start] {datetime.now(timezone.utc).isoformat()} label={label}\n")
        log.write("[cmd] " + " ".join(command) + "\n")
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            if args.echo_model_logs:
                print(f"[{label}] {line}", end="", flush=True)
        returncode = process.wait()
        log.write(f"[finish] {datetime.now(timezone.utc).isoformat()} label={label} returncode={returncode}\n")
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run AI edit pass for all configured domain/model curated shards."""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import load_domain_specs, load_model_specs, write_json
from scripts.validate_ai_edits import print_validation_summary, validate_ai_edits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="final")
    parser.add_argument("--domains", nargs="+", default=None)
    parser.add_argument("--models-config", type=Path, default=PROJECT_ROOT / "configs" / "models.yaml")
    parser.add_argument("--datasets-config", type=Path, default=PROJECT_ROOT / "configs" / "datasets.yaml")
    parser.add_argument("--input-root", type=Path, default=PROJECT_ROOT / "data" / "curated_shards")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "curated_shards")
    parser.add_argument("--log-root", type=Path, default=PROJECT_ROOT / "logs" / "final_ai_edit")
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--expected-count", type=int, default=10)
    parser.add_argument("--schedule", choices=["sequential", "parallel", "hybrid"], default="hybrid")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--echo-model-logs", action="store_true")
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--validation-manifest", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.log_root.mkdir(parents=True, exist_ok=True)
    domains = args.domains or list(load_domain_specs(args.datasets_config))
    models = load_model_specs(args.models_config)
    command_items = [(spec["model_label"], spec, edit_commands_for_model(args, domains, spec)) for spec in models]
    if args.dry_run:
        for label, _, commands in command_items:
            print(f"# {label}")
            for command in commands:
                print(" ".join(command))
        return

    schedule = "parallel" if args.parallel else args.schedule
    if schedule == "parallel":
        run_parallel(command_items, args)
    elif schedule == "hybrid":
        run_hybrid(command_items, args)
    else:
        for label, spec, commands in command_items:
            run_model(label, spec, commands, args)

    if not args.no_validate:
        summary = validate_ai_edits(
            run_id=args.run_id,
            domains=domains,
            models=models,
            input_root=args.output_root,
            expected_count=args.expected_count,
        )
        print_validation_summary(summary)
        manifest_path = args.validation_manifest or args.log_root / f"{args.run_id}_validation.json"
        write_json(manifest_path, summary)
        if summary["errors"]:
            raise SystemExit(1)


def edit_commands_for_model(args: argparse.Namespace, domains: list[str], spec: dict[str, Any]) -> list[list[str]]:
    commands = []
    edit_max_tokens = str(spec.get("edit_max_tokens", spec.get("max_tokens", args.max_tokens)))
    for domain in domains:
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "edit_ai.py"),
            "--run-id",
            args.run_id,
            "--domain",
            domain,
            "--provider",
            spec["provider"],
            "--model",
            spec["model"],
            "--model-label",
            spec["model_label"],
            "--input-root",
            str(args.input_root),
            "--output-root",
            str(args.output_root),
            "--temperature",
            str(args.temperature),
            "--top-p",
            str(args.top_p),
            "--max-tokens",
            edit_max_tokens,
        ]
        if spec.get("openai_reasoning_effort"):
            command.extend(["--openai-reasoning-effort", str(spec["openai_reasoning_effort"])])
        if spec.get("openai_verbosity"):
            command.extend(["--openai-verbosity", str(spec["openai_verbosity"])])
        if spec.get("deepseek_reasoning_effort"):
            command.extend(["--deepseek-reasoning-effort", str(spec["deepseek_reasoning_effort"])])
        if spec.get("deepseek_thinking"):
            command.extend(["--deepseek-thinking", str(spec["deepseek_thinking"])])
        if args.mock:
            command.append("--mock")
        commands.append(command)
    return commands


def run_parallel(command_items: list[tuple[str, dict[str, Any], list[list[str]]]], args: argparse.Namespace) -> None:
    with ThreadPoolExecutor(max_workers=len(command_items)) as pool:
        futures = {
            label: pool.submit(run_model, label, spec, commands, args)
            for label, spec, commands in command_items
        }
        for future in futures.values():
            future.result()


def run_hybrid(command_items: list[tuple[str, dict[str, Any], list[list[str]]]], args: argparse.Namespace) -> None:
    first_wave_labels = [
        "gemini-2.5-pro",
        "claude-sonnet-4-6",
        "deepseek-v4-pro",
        "gpt-3.5-turbo",
    ]
    openai_chain_labels = ["gpt-4", "gpt-5.5"]
    by_label = {label: (spec, commands) for label, spec, commands in command_items}
    first_wave = [(label, *by_label[label]) for label in first_wave_labels if label in by_label]
    openai_chain = [(label, *by_label[label]) for label in openai_chain_labels if label in by_label]
    scheduled = set(first_wave_labels + openai_chain_labels)
    first_wave.extend((label, spec, commands) for label, spec, commands in command_items if label not in scheduled)

    if not first_wave:
        for label, spec, commands in openai_chain:
            run_model(label, spec, commands, args)
        return

    with ThreadPoolExecutor(max_workers=len(first_wave)) as pool:
        futures = {
            label: pool.submit(run_model, label, spec, commands, args)
            for label, spec, commands in first_wave
        }
        gpt35_future = futures.get("gpt-3.5-turbo")
        if gpt35_future is not None:
            try:
                gpt35_future.result()
            except Exception:
                wait_for_background(futures, skip_label="gpt-3.5-turbo")
                raise
        for label, spec, commands in openai_chain:
            run_model(label, spec, commands, args)
        wait_for_background(futures, skip_label="gpt-3.5-turbo")


def wait_for_background(futures: dict[str, object], skip_label: str | None = None) -> None:
    for label, future in futures.items():
        if label == skip_label:
            continue
        future.result()


def run_model(label: str, spec: dict[str, Any], commands: list[list[str]], args: argparse.Namespace) -> None:
    log_path = args.log_root / f"{label}.log"
    print(f"[ai-edit-run:{label}] domains={len(commands)} log={log_path}", flush=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"[model-start] {datetime.now(timezone.utc).isoformat()} label={label}\n")
        log.write(f"[provider] {spec['provider']} model={spec['model']}\n")
        for command in commands:
            domain = command[command.index("--domain") + 1]
            log.write(f"[domain-start] {datetime.now(timezone.utc).isoformat()} domain={domain}\n")
            run_command(label, domain, command, log, args)
            log.write(f"[domain-finish] {datetime.now(timezone.utc).isoformat()} domain={domain}\n")
        log.write(f"[model-finish] {datetime.now(timezone.utc).isoformat()} label={label}\n")


def run_command(label: str, domain: str, command: list[str], log, args: argparse.Namespace) -> None:
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
            print(f"[{label}:{domain}] {line}", end="", flush=True)
    returncode = process.wait()
    if returncode:
        log.write(f"[domain-error] domain={domain} returncode={returncode}\n")
        raise subprocess.CalledProcessError(returncode, command)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Regenerate only ai_edited rows with word-count ratio warnings."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import iter_shard_files, load_model_specs, output_shard_dir, read_jsonl, word_count


LOW_RATIO = 0.65
HIGH_RATIO = 1.35


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="final")
    parser.add_argument("--input-root", type=Path, default=PROJECT_ROOT / "data" / "curated_shards")
    parser.add_argument("--models-config", type=Path, default=PROJECT_ROOT / "configs" / "models.yaml")
    parser.add_argument("--log-root", type=Path, default=PROJECT_ROOT / "logs" / "final_ai_reedit_warnings")
    parser.add_argument("--backup-root", type=Path, default=PROJECT_ROOT / "runs" / "ai_edit_reedit_backups")
    parser.add_argument("--low-ratio", type=float, default=LOW_RATIO)
    parser.add_argument("--high-ratio", type=float, default=HIGH_RATIO)
    parser.add_argument("--max-tokens", type=int, default=2200)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mock", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_specs = {spec["model_label"]: spec for spec in load_model_specs(args.models_config)}
    warnings = find_warning_rows(args.input_root, args.run_id, args.low_ratio, args.high_ratio)
    grouped = group_warnings(warnings)
    print(f"[reedit-warnings] warning_rows={len(warnings)} groups={len(grouped)}")
    for (domain, model_label), row_ids in sorted(grouped.items()):
        print(f"[reedit-warnings] {domain}/{model_label}: {len(row_ids)} rows")
        command = build_command(args, domain, model_specs[model_label], row_ids)
        print("[cmd] " + " ".join(command))
        if not args.dry_run:
            run_command(args, domain, model_label, command)


def find_warning_rows(root: Path, run_id: str, low_ratio: float, high_ratio: float) -> list[dict[str, Any]]:
    warnings = []
    for edited_path in sorted((root / run_id).glob("*/*/ai_edited/*_ai_edited_shard_*.jsonl")):
        domain = edited_path.parts[-4]
        model_label = edited_path.parts[-3]
        input_dir = output_shard_dir(root, run_id, domain, model_label, "ai_edit_input")
        input_path = input_dir / edited_path.name.replace("_ai_edited_", "_ai_edit_input_")
        input_rows = {row.get("id"): row for row in read_jsonl(input_path)}
        for edited in read_jsonl(edited_path):
            source_id = edited.get("source_input_id") or str(edited.get("id", "")).replace(":ai_edited:", ":ai_edit_input:")
            source = input_rows.get(source_id)
            if not source:
                continue
            source_words = word_count(str(source.get("sampled") or ""))
            edited_words = word_count(str(edited.get("sampled") or ""))
            if not source_words:
                continue
            ratio = edited_words / source_words
            if ratio < low_ratio or ratio > high_ratio:
                warnings.append(
                    {
                        "domain": domain,
                        "model_label": model_label,
                        "source_input_id": source_id,
                        "edited_id": edited.get("id"),
                        "source_words": source_words,
                        "edited_words": edited_words,
                        "ratio": ratio,
                    }
                )
    return warnings


def group_warnings(warnings: list[dict[str, Any]]) -> dict[tuple[str, str], list[str]]:
    grouped: dict[tuple[str, str], list[str]] = {}
    for warning in warnings:
        key = (warning["domain"], warning["model_label"])
        grouped.setdefault(key, []).append(warning["source_input_id"])
    return grouped


def build_command(args: argparse.Namespace, domain: str, model_spec: dict[str, Any], row_ids: list[str]) -> list[str]:
    max_tokens = str(model_spec.get("edit_max_tokens", model_spec.get("max_tokens", args.max_tokens)))
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "edit_ai.py"),
        "--run-id",
        args.run_id,
        "--domain",
        domain,
        "--provider",
        model_spec["provider"],
        "--model",
        model_spec["model"],
        "--model-label",
        model_spec["model_label"],
        "--input-root",
        str(args.input_root),
        "--output-root",
        str(args.input_root),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--max-tokens",
        max_tokens,
        "--length-guard",
        "--force-row-ids",
        *row_ids,
    ]
    if model_spec.get("openai_reasoning_effort"):
        command.extend(["--openai-reasoning-effort", str(model_spec["openai_reasoning_effort"])])
    if model_spec.get("openai_verbosity"):
        command.extend(["--openai-verbosity", str(model_spec["openai_verbosity"])])
    if model_spec.get("deepseek_reasoning_effort"):
        command.extend(["--deepseek-reasoning-effort", str(model_spec["deepseek_reasoning_effort"])])
    if model_spec.get("deepseek_thinking"):
        command.extend(["--deepseek-thinking", str(model_spec["deepseek_thinking"])])
    if args.mock:
        command.append("--mock")
    return command


def run_command(args: argparse.Namespace, domain: str, model_label: str, command: list[str]) -> None:
    args.log_root.mkdir(parents=True, exist_ok=True)
    backup_existing_ai_edited_shards(args, domain, model_label)
    log_path = args.log_root / f"{domain}_{model_label}.log"
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"[start] {datetime.now(timezone.utc).isoformat()} domain={domain} model={model_label}\n")
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
            print(f"[{domain}/{model_label}] {line}", end="", flush=True)
        returncode = process.wait()
        log.write(f"[finish] {datetime.now(timezone.utc).isoformat()} returncode={returncode}\n")
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def backup_existing_ai_edited_shards(args: argparse.Namespace, domain: str, model_label: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    edited_dir = output_shard_dir(args.input_root, args.run_id, domain, model_label, "ai_edited")
    for shard_path in iter_shard_files(edited_dir):
        backup_path = args.backup_root / f"{args.run_id}_{timestamp}" / shard_path.relative_to(args.input_root)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(shard_path, backup_path)


if __name__ == "__main__":
    main()

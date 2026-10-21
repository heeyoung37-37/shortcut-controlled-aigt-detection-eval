#!/usr/bin/env python3
"""Validate ai_edit_input -> ai_edited curated shards."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import (
    iter_shard_files,
    load_domain_specs,
    load_model_specs,
    output_shard_dir,
    read_jsonl,
    word_count,
    write_json,
)
from scripts.edit_ai import MAX_WORD_RATIO, MIN_WORD_RATIO, validate_edited_text
from scripts.ultimate_controlled.prompts import EDIT_PROMPT_VERSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="final")
    parser.add_argument("--domains", nargs="+", default=None)
    parser.add_argument("--models-config", type=Path, default=PROJECT_ROOT / "configs" / "models.yaml")
    parser.add_argument("--datasets-config", type=Path, default=PROJECT_ROOT / "configs" / "datasets.yaml")
    parser.add_argument("--input-root", type=Path, default=PROJECT_ROOT / "data" / "curated_shards")
    parser.add_argument("--expected-count", type=int, default=10)
    parser.add_argument("--manifest-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    domains = args.domains or list(load_domain_specs(args.datasets_config))
    models = load_model_specs(args.models_config)
    summary = validate_ai_edits(
        run_id=args.run_id,
        domains=domains,
        models=models,
        input_root=args.input_root,
        expected_count=args.expected_count,
    )
    print_validation_summary(summary)
    if args.manifest_path:
        write_json(args.manifest_path, summary)
    if summary["errors"]:
        raise SystemExit(1)


def validate_ai_edits(
    *,
    run_id: str,
    domains: list[str],
    models: list[dict[str, Any]],
    input_root: Path,
    expected_count: int = 10,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "run_id": run_id,
        "expected_count": expected_count,
        "checked_pairs": 0,
        "errors": [],
        "warnings": [],
        "counts": {},
    }
    for domain in domains:
        for model_spec in models:
            model_label = model_spec["model_label"]
            pair_key = f"{domain}/{model_label}"
            input_rows = read_variant_rows(input_root, run_id, domain, model_label, "ai_edit_input")
            edited_rows = read_variant_rows(input_root, run_id, domain, model_label, "ai_edited")
            summary["checked_pairs"] += 1
            summary["counts"][pair_key] = {
                "ai_edit_input": len(input_rows),
                "ai_edited": len(edited_rows),
            }
            if len(input_rows) != expected_count:
                add_error(summary, pair_key, f"ai_edit_input count is {len(input_rows)}, expected {expected_count}")
            if len(edited_rows) != expected_count:
                add_error(summary, pair_key, f"ai_edited count is {len(edited_rows)}, expected {expected_count}")
            validate_pair_rows(summary, pair_key, domain, model_label, input_rows, edited_rows)
    return summary


def read_variant_rows(root: Path, run_id: str, domain: str, model_label: str, variant: str) -> list[dict[str, Any]]:
    variant_dir = output_shard_dir(root, run_id, domain, model_label, variant)
    rows: list[dict[str, Any]] = []
    for shard_path in iter_shard_files(variant_dir):
        rows.extend(read_jsonl(shard_path))
    return rows


def validate_pair_rows(
    summary: dict[str, Any],
    pair_key: str,
    domain: str,
    model_label: str,
    input_rows: list[dict[str, Any]],
    edited_rows: list[dict[str, Any]],
) -> None:
    edited_by_id = {}
    for row in edited_rows:
        row_id = row.get("id")
        if row_id in edited_by_id:
            add_error(summary, pair_key, f"duplicate ai_edited id: {row_id}")
        edited_by_id[row_id] = row
    for source in input_rows:
        source_id = source.get("id")
        expected_id = str(source_id).replace(":ai_edit_input:", ":ai_edited:")
        edited = edited_by_id.get(expected_id)
        if edited is None:
            add_error(summary, pair_key, f"missing ai_edited row for {source_id}")
            continue
        validate_one_row(summary, pair_key, domain, model_label, source, edited, expected_id)


def validate_one_row(
    summary: dict[str, Any],
    pair_key: str,
    domain: str,
    model_label: str,
    source: dict[str, Any],
    edited: dict[str, Any],
    expected_id: str,
) -> None:
    if edited.get("id") != expected_id:
        add_error(summary, pair_key, f"id mismatch: {edited.get('id')} != {expected_id}")
    if edited.get("domain") != domain:
        add_error(summary, pair_key, f"domain mismatch at {expected_id}")
    if edited.get("model_label") != model_label:
        add_error(summary, pair_key, f"model_label mismatch at {expected_id}")
    if edited.get("variant_type") != "ai_edited":
        add_error(summary, pair_key, f"variant_type mismatch at {expected_id}")
    if edited.get("source_input_id") != source.get("id"):
        add_error(summary, pair_key, f"source_input_id mismatch at {expected_id}")
    if edited.get("original") != source.get("original"):
        add_error(summary, pair_key, f"original changed at {expected_id}")
    sampled = str(edited.get("sampled") or "")
    try:
        validate_edited_text(domain, sampled)
    except Exception as exc:
        add_error(summary, pair_key, f"invalid sampled text at {expected_id}: {exc}")
    metadata = edited.get("metadata") if isinstance(edited.get("metadata"), dict) else {}
    ai_edit = metadata.get("ai_edit") if isinstance(metadata.get("ai_edit"), dict) else {}
    required_ai_edit = ("provider", "model", "source_variant", "prompt_version", "edited_at")
    missing = [key for key in required_ai_edit if not ai_edit.get(key)]
    if missing:
        add_error(summary, pair_key, f"missing ai_edit metadata at {expected_id}: {missing}")
    if ai_edit.get("source_variant") != "ai_edit_input":
        add_error(summary, pair_key, f"bad source_variant at {expected_id}")
    if ai_edit.get("prompt_version") != EDIT_PROMPT_VERSION:
        add_error(summary, pair_key, f"bad prompt_version at {expected_id}: {ai_edit.get('prompt_version')}")
    source_words = word_count(str(source.get("sampled") or ""))
    edited_words = word_count(sampled)
    if source_words:
        ratio = edited_words / source_words
        if ratio < MIN_WORD_RATIO or ratio > MAX_WORD_RATIO:
            add_warning(summary, pair_key, f"word_count_ratio_review:{ratio:.2f} at {expected_id}")


def add_error(summary: dict[str, Any], pair_key: str, message: str) -> None:
    summary["errors"].append(f"{pair_key}: {message}")


def add_warning(summary: dict[str, Any], pair_key: str, message: str) -> None:
    summary["warnings"].append(f"{pair_key}: {message}")


def print_validation_summary(summary: dict[str, Any]) -> None:
    print(
        "[ai-edit-validation] "
        f"pairs={summary['checked_pairs']} errors={len(summary['errors'])} warnings={len(summary['warnings'])}"
    )
    for message in summary["errors"][:50]:
        print(f"[error] {message}")
    if len(summary["errors"]) > 50:
        print(f"[error] ... {len(summary['errors']) - 50} more")
    for message in summary["warnings"][:20]:
        print(f"[warning] {message}")
    if len(summary["warnings"]) > 20:
        print(f"[warning] ... {len(summary['warnings']) - 20} more")


if __name__ == "__main__":
    main()

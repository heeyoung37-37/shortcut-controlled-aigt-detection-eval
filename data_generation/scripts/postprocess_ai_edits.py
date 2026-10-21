#!/usr/bin/env python3
"""Postprocess AI-edited curated shards without changing source pairing."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
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
    write_jsonl,
)


PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")
SPACE_RE = re.compile(r"[ \t]{2,}")
WORD_RATIO_LOW = 0.65
WORD_RATIO_HIGH = 1.35


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="final")
    parser.add_argument("--domains", nargs="+", default=None)
    parser.add_argument("--models-config", type=Path, default=PROJECT_ROOT / "configs" / "models.yaml")
    parser.add_argument("--datasets-config", type=Path, default=PROJECT_ROOT / "configs" / "datasets.yaml")
    parser.add_argument("--input-root", type=Path, default=PROJECT_ROOT / "data" / "curated_shards")
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--backup-root", type=Path, default=PROJECT_ROOT / "runs" / "ai_edit_postprocess_backups")
    parser.add_argument("--write", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    domains = args.domains or list(load_domain_specs(args.datasets_config))
    models = load_model_specs(args.models_config)
    summary = postprocess_ai_edits(
        run_id=args.run_id,
        domains=domains,
        models=models,
        input_root=args.input_root,
        backup_root=args.backup_root,
        write=args.write,
    )
    print_summary(summary)
    if args.write or args.manifest_path:
        manifest_path = args.manifest_path or PROJECT_ROOT / "data" / "manifests" / f"{args.run_id}_ai_edit_postprocess_manifest.json"
        write_json(manifest_path, summary)


def postprocess_ai_edits(
    *,
    run_id: str,
    domains: list[str],
    models: list[dict[str, Any]],
    input_root: Path,
    backup_root: Path,
    write: bool,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "dry_run": not write,
        "changed_rows": 0,
        "changed_files": 0,
        "paragraph_drift_before": 0,
        "paragraph_drift_after": 0,
        "word_count_warnings": [],
        "changes_by_domain": {},
        "changes_by_model": {},
        "files": [],
    }
    backup_run_dir = backup_root / f"{run_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    for domain in domains:
        for model_spec in models:
            model_label = model_spec["model_label"]
            process_pair(summary, input_root, backup_run_dir, run_id, domain, model_label, write)
    summary["changes_by_domain"] = dict(sorted(summary["changes_by_domain"].items()))
    summary["changes_by_model"] = dict(sorted(summary["changes_by_model"].items()))
    return summary


def process_pair(
    summary: dict[str, Any],
    root: Path,
    backup_run_dir: Path,
    run_id: str,
    domain: str,
    model_label: str,
    write: bool,
) -> None:
    input_dir = output_shard_dir(root, run_id, domain, model_label, "ai_edit_input")
    edited_dir = output_shard_dir(root, run_id, domain, model_label, "ai_edited")
    for edited_path in iter_shard_files(edited_dir):
        input_path = input_dir / edited_path.name.replace("_ai_edited_", "_ai_edit_input_")
        input_rows = read_jsonl(input_path)
        edited_rows = read_jsonl(edited_path)
        input_by_id = {row.get("id"): row for row in input_rows}
        changed_rows = 0
        for edited in edited_rows:
            source_id = edited.get("source_input_id") or str(edited.get("id", "")).replace(":ai_edited:", ":ai_edit_input:")
            source = input_by_id.get(source_id)
            if not source:
                continue
            changed = postprocess_row(summary, source, edited, domain, model_label)
            changed_rows += int(changed)
        if changed_rows:
            summary["changed_files"] += 1
            summary["changed_rows"] += changed_rows
            file_record = {"path": str(edited_path), "changed_rows": changed_rows}
            if write:
                backup_path = backup_run_dir / edited_path.relative_to(root)
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(edited_path, backup_path)
                write_jsonl(edited_path, edited_rows)
                file_record["backup_path"] = str(backup_path)
            summary["files"].append(file_record)


def postprocess_row(
    summary: dict[str, Any],
    source: dict[str, Any],
    edited: dict[str, Any],
    domain: str,
    model_label: str,
) -> bool:
    source_text = str(source.get("sampled") or "")
    edited_text = str(edited.get("sampled") or "")
    source_paragraphs = paragraph_count(source_text)
    edited_paragraphs = paragraph_count(edited_text)
    source_lines = nonempty_line_count(source_text)
    edited_lines = nonempty_line_count(edited_text)
    if source_paragraphs == 1 and edited_paragraphs > 1:
        summary["paragraph_drift_before"] += 1
        flattened = flatten_paragraphs(edited_text)
        metadata = edited.get("metadata") if isinstance(edited.get("metadata"), dict) else {}
        metadata = dict(metadata)
        postprocess_meta = dict(metadata.get("ai_edit_postprocess") or {})
        postprocess_meta["paragraph_flattened_at"] = datetime.now(timezone.utc).isoformat()
        postprocess_meta["paragraph_count_before"] = edited_paragraphs
        postprocess_meta["paragraph_count_after"] = paragraph_count(flattened)
        postprocess_meta["source_paragraph_count"] = source_paragraphs
        postprocess_meta["line_count_before"] = edited_lines
        postprocess_meta["source_line_count"] = source_lines
        metadata["ai_edit_postprocess"] = postprocess_meta
        edited["metadata"] = metadata
        edited["sampled"] = flattened
        summary["changes_by_domain"][domain] = summary["changes_by_domain"].get(domain, 0) + 1
        summary["changes_by_model"][model_label] = summary["changes_by_model"].get(model_label, 0) + 1
        if paragraph_count(flattened) > 1:
            summary["paragraph_drift_after"] += 1
        add_word_count_warning_if_needed(summary, source, edited, domain, model_label)
        return True
    add_word_count_warning_if_needed(summary, source, edited, domain, model_label)
    return False


def add_word_count_warning_if_needed(
    summary: dict[str, Any],
    source: dict[str, Any],
    edited: dict[str, Any],
    domain: str,
    model_label: str,
) -> None:
    source_words = word_count(str(source.get("sampled") or ""))
    edited_words = word_count(str(edited.get("sampled") or ""))
    if not source_words:
        return
    ratio = edited_words / source_words
    if WORD_RATIO_LOW <= ratio <= WORD_RATIO_HIGH:
        return
    summary["word_count_warnings"].append(
        {
            "id": edited.get("id"),
            "domain": domain,
            "model_label": model_label,
            "source_words": source_words,
            "edited_words": edited_words,
            "ratio": round(ratio, 3),
            "direction": "too_short" if ratio < WORD_RATIO_LOW else "too_long",
        }
    )


def paragraph_count(text: str) -> int:
    parts = [part.strip() for part in PARAGRAPH_SPLIT_RE.split((text or "").strip()) if part.strip()]
    return len(parts) if parts else 0


def nonempty_line_count(text: str) -> int:
    return len([line for line in (text or "").splitlines() if line.strip()])


def flatten_paragraphs(text: str) -> str:
    flattened = re.sub(r"\s*\n+\s*", " ", text or "")
    flattened = SPACE_RE.sub(" ", flattened)
    return flattened.strip()


def print_summary(summary: dict[str, Any]) -> None:
    warning_counts = Counter(item["direction"] for item in summary["word_count_warnings"])
    print(
        "[ai-edit-postprocess] "
        f"dry_run={summary['dry_run']} changed_rows={summary['changed_rows']} "
        f"changed_files={summary['changed_files']} paragraph_drift_before={summary['paragraph_drift_before']} "
        f"paragraph_drift_after={summary['paragraph_drift_after']} word_count_warnings={len(summary['word_count_warnings'])}"
    )
    print(f"[ai-edit-postprocess] changes_by_domain={summary['changes_by_domain']}")
    print(f"[ai-edit-postprocess] changes_by_model={summary['changes_by_model']}")
    print(f"[ai-edit-postprocess] word_count_warning_directions={dict(warning_counts)}")


if __name__ == "__main__":
    main()

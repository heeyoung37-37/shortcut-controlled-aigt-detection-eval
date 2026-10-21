#!/usr/bin/env python3
"""Apply targeted meeting_summary replacement generations to curated shards."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import (
    output_shard_dir,
    read_json,
    read_jsonl,
    validate_output_row,
    write_json,
    write_jsonl,
)


DOMAIN = "meeting_summary"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replacement-run-id", default="meeting_summary_ordinance_replacements")
    parser.add_argument("--target-run-id", default="final")
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--generated-root", type=Path, default=None)
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT / "data" / "source_shards")
    parser.add_argument("--curated-root", type=Path, default=PROJECT_ROOT / "data" / "curated_shards")
    parser.add_argument("--output-manifest-path", type=Path, default=None)
    parser.add_argument("--write", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest_path or PROJECT_ROOT / "data" / "manifests" / f"{args.replacement_run_id}_manifest.json"
    generated_root = args.generated_root or PROJECT_ROOT / "runs" / args.replacement_run_id / "generated"
    output_manifest_path = (
        args.output_manifest_path
        or PROJECT_ROOT / "data" / "manifests" / f"{args.replacement_run_id}_apply_manifest.json"
    )
    manifest = read_json(manifest_path)
    applied: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": not args.write,
        "replacement_run_id": args.replacement_run_id,
        "target_run_id": args.target_run_id,
        "generated_root": str(generated_root),
        "files": [],
        "ai_edit_input_ids": [],
        "summary": {},
    }
    targets_by_model = manifest.get("targets_by_model") or {}
    for model_label, targets in sorted(targets_by_model.items()):
        accepted = load_replacement_records(generated_root, model_label)
        if len(accepted) < len(targets):
            raise SystemExit(f"{model_label}: need {len(targets)} accepted replacements, found {len(accepted)}")
        source_map = load_source_map(args.source_root / args.replacement_run_id / DOMAIN / model_label)
        changes_by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for target, replacement in zip(targets, accepted):
            row_change = apply_target(args, target, replacement, source_map, model_label, write=args.write)
            changes_by_file[str(target["shard_path"])].append(row_change)
            if target["variant_type"] == "ai_edit_input":
                applied["ai_edit_input_ids"].append(target["id"])
        for file_path, changes in sorted(changes_by_file.items()):
            applied["files"].append({"path": file_path, "changes": changes})
        applied["summary"][model_label] = {
            "targets": len(targets),
            "accepted_available": len(accepted),
            "ai_edit_input_replacements": sum(1 for item in targets if item["variant_type"] == "ai_edit_input"),
        }
        print(f"[meeting-replace-apply] {model_label}: replaced={len(targets)}")
    if args.write:
        write_json(output_manifest_path, applied)
    else:
        print("[meeting-replace-apply] dry-run only; pass --write to modify curated shards")
    print(f"[meeting-replace-apply] ai_edit_input_ids={len(applied['ai_edit_input_ids'])}")
    for row_id in applied["ai_edit_input_ids"]:
        print(row_id)


def load_replacement_records(generated_root: Path, model_label: str) -> list[dict[str, Any]]:
    accepted_path = generated_root / model_label / DOMAIN / "artifacts" / f"{DOMAIN}_{model_label}.accepted.jsonl"
    rows = read_jsonl(accepted_path)
    if not rows:
        raise SystemExit(f"No accepted replacement rows at {accepted_path}")
    return rows


def load_source_map(model_source_dir: Path) -> dict[int, dict[str, Any]]:
    rows = read_jsonl(model_source_dir / "structure.sources.jsonl")
    return {int(row["source_index"]): row for row in rows}


def apply_target(
    args: argparse.Namespace,
    target: dict[str, Any],
    replacement: dict[str, Any],
    source_map: dict[int, dict[str, Any]],
    model_label: str,
    *,
    write: bool,
) -> dict[str, Any]:
    shard_path = Path(target["shard_path"])
    rows = read_jsonl(shard_path)
    row_index = int(target["row_index"])
    row = rows[row_index]
    if row.get("id") != target["id"]:
        raise ValueError(f"{shard_path}:{row_index} expected id={target['id']}, found {row.get('id')}")
    source_index = int(replacement.get("structure_source_index", 0))
    source_info = source_map.get(source_index, {})
    old_metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    split_metadata = old_metadata.get("split") if isinstance(old_metadata.get("split"), dict) else None
    row["original"] = replacement["original"]
    row["sampled"] = replacement["sampled"]
    row["source_id"] = replacement.get("structure_source_id") or source_info.get("source_id") or row["source_id"]
    new_metadata = {
        "pair_id": replacement.get("pair_id"),
        "structure_source_index": replacement.get("structure_source_index"),
        "fact_source_index": replacement.get("fact_source_index"),
        "generation_attempt": replacement.get("attempt"),
        "metrics": replacement.get("metrics", {}),
        "source": source_info,
        "replacement": {
            "replaced_at": datetime.now(timezone.utc).isoformat(),
            "reason": "meeting_summary_ordinance_title_reference",
            "replacement_run_id": args.replacement_run_id,
            "old_source_id": target.get("old_source_id"),
            "old_original_preview": target.get("old_original_preview"),
            "replacement_pair_id": replacement.get("pair_id"),
            "replacement_model_label": model_label,
        },
    }
    if split_metadata:
        new_metadata["split"] = split_metadata
    row["metadata"] = new_metadata
    validate_output_row(row)
    if write:
        write_jsonl(shard_path, rows)
    return {
        "id": row["id"],
        "variant_type": row["variant_type"],
        "row_index": row_index,
        "old_source_id": target.get("old_source_id"),
        "new_source_id": row["source_id"],
        "replacement_pair_id": replacement.get("pair_id"),
    }


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare clean source shards for replacing bad meeting_summary rows.

The target issue is MeetingBank rows whose human reference is a legislative
title/citation blob, usually beginning with "AN ORDINANCE ...". Those are not
useful meeting summaries, so this script selects only affected final rows and
builds clean same-source replacement shards for targeted regeneration.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import (
    load_model_specs,
    output_shard_dir,
    read_jsonl,
    write_json,
    write_jsonl,
)
from scripts.build_source_shards import source_sidecar_rows


DOMAIN = "meeting_summary"
TARGET_VARIANTS = ("raw_ai", "ai_edit_input", "human_edit_input")
BAD_ORIGINAL_RE = re.compile(
    r"^\s*AN ORDINANCE\b|"
    r"\bK\.C\.C\.\b.*\bOrdinance\b|"
    r"\bOrdinance\b.*\bK\.C\.C\.\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="meeting_summary_ordinance_replacements")
    parser.add_argument("--source-run-id", default="final")
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--extra-candidates", type=int, default=18)
    parser.add_argument("--min-candidates-per-model", type=int, default=24)
    parser.add_argument("--curated-root", type=Path, default=PROJECT_ROOT / "data" / "curated_shards")
    parser.add_argument("--normalized-path", type=Path, default=PROJECT_ROOT / "data" / "normalized" / DOMAIN / "records.jsonl")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "source_shards")
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--models-config", type=Path, default=PROJECT_ROOT / "configs" / "models.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_specs = load_model_specs(args.models_config)
    targets_by_model = find_targets(args, model_specs)
    if not targets_by_model:
        print("[meeting-replace] no bad meeting_summary rows found")
        return
    normalized_rows = read_jsonl(args.normalized_path)
    used_source_ids = current_source_ids(args.curated_root / args.source_run_id / DOMAIN)
    available = [
        row for row in normalized_rows
        if row.get("source_id") not in used_source_ids and is_clean_source(row)
    ]
    if not available:
        raise SystemExit("No clean meeting_summary replacement sources available")
    rng = random.Random(f"{args.seed}:{args.run_id}:{DOMAIN}")
    rng.shuffle(available)

    cursor = 0
    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "source_run_id": args.source_run_id,
        "domain": DOMAIN,
        "bad_original_pattern": BAD_ORIGINAL_RE.pattern,
        "source_root": str(args.output_root / args.run_id),
        "targets_by_model": {},
        "candidate_counts_by_model": {},
    }
    for model_spec in model_specs:
        label = model_spec["model_label"]
        targets = targets_by_model.get(label, [])
        if not targets:
            continue
        candidate_count = max(args.min_candidates_per_model, len(targets) + args.extra_candidates)
        selected = available[cursor : cursor + candidate_count]
        if len(selected) < candidate_count:
            raise SystemExit(f"Need {candidate_count} clean sources for {label}, found {len(selected)}")
        cursor += candidate_count
        model_dir = args.output_root / args.run_id / DOMAIN / label
        write_json(model_dir / "structure.jsonl", {"original": [row["human_text"] for row in selected]})
        write_json(model_dir / "fact.jsonl", {"original": [row["fact_source_text"] for row in selected]})
        write_jsonl(model_dir / "structure.sources.jsonl", source_sidecar_rows(selected, "structure"))
        write_jsonl(model_dir / "fact.sources.jsonl", source_sidecar_rows(selected, "fact"))
        manifest["targets_by_model"][label] = targets
        manifest["candidate_counts_by_model"][label] = {
            "needed_replacements": len(targets),
            "candidate_count": candidate_count,
            "source_ids": [row["source_id"] for row in selected],
        }
        print(f"[meeting-replace-source] {label}: targets={len(targets)} candidates={candidate_count}")

    manifest_path = args.manifest_path or PROJECT_ROOT / "data" / "manifests" / f"{args.run_id}_manifest.json"
    write_json(manifest_path, manifest)
    print(f"[meeting-replace-source] manifest -> {manifest_path}")


def find_targets(args: argparse.Namespace, model_specs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    labels = [spec["model_label"] for spec in model_specs]
    targets_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counts = Counter()
    for label in labels:
        for variant in TARGET_VARIANTS:
            variant_dir = output_shard_dir(args.curated_root, args.source_run_id, DOMAIN, label, variant)
            for shard_path in sorted(variant_dir.glob("*.jsonl")):
                rows = read_jsonl(shard_path)
                for row_index, row in enumerate(rows):
                    if BAD_ORIGINAL_RE.search(str(row.get("original") or "")):
                        target = {
                            "id": row["id"],
                            "variant_type": variant,
                            "row_index": row_index,
                            "shard_path": str(shard_path),
                            "old_source_id": row.get("source_id"),
                            "old_original_preview": str(row.get("original") or "")[:240],
                        }
                        targets_by_model[label].append(target)
                        counts[(label, variant)] += 1
    for (label, variant), count in sorted(counts.items()):
        print(f"[meeting-replace-target] {label}/{variant}: {count}")
    return dict(targets_by_model)


def current_source_ids(meeting_root: Path) -> set[str]:
    ids: set[str] = set()
    for path in meeting_root.rglob("*.jsonl"):
        for row in read_jsonl(path):
            source_id = row.get("source_id")
            if source_id:
                ids.add(str(source_id))
    return ids


def is_clean_source(row: dict[str, Any]) -> bool:
    human_text = str(row.get("human_text") or "")
    if BAD_ORIGINAL_RE.search(human_text):
        return False
    if human_text.count("K.C.C.") >= 2 or human_text.count("Ordinance") >= 4:
        return False
    # Avoid source references that are mostly agenda title/citation rather than
    # a compact action summary.
    if len(re.findall(r"\bSection\s+\d+", human_text, flags=re.IGNORECASE)) >= 4:
        return False
    return True


if __name__ == "__main__":
    main()

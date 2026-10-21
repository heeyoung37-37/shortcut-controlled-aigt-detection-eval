#!/usr/bin/env python3
"""Build model-specific source shards for Ultimate generation."""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import load_domain_specs, load_model_specs, read_jsonl, write_json, write_jsonl


SAME_SOURCE_DOMAINS = {"meeting_summary", "product_review"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="final")
    parser.add_argument("--domains", nargs="+", default=None)
    parser.add_argument("--items-per-kind", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normalized-dir", type=Path, default=PROJECT_ROOT / "data" / "normalized")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "source_shards")
    parser.add_argument("--manifest-dir", type=Path, default=PROJECT_ROOT / "data" / "manifests")
    parser.add_argument("--datasets-config", type=Path, default=PROJECT_ROOT / "configs" / "datasets.yaml")
    parser.add_argument("--models-config", type=Path, default=PROJECT_ROOT / "configs" / "models.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    domain_specs = load_domain_specs(args.datasets_config)
    model_specs = load_model_specs(args.models_config)
    domains = args.domains or list(domain_specs)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "items_per_kind": args.items_per_kind,
        "seed": args.seed,
        "domains": {},
    }
    for domain in domains:
        rows = read_jsonl(args.normalized_dir / domain / "records.jsonl")
        domain_manifest = build_domain_shards(
            rows,
            domain=domain,
            model_specs=model_specs,
            output_root=args.output_root / args.run_id,
            items_per_kind=args.items_per_kind,
            seed=args.seed,
        )
        manifest["domains"][domain] = domain_manifest
        print(f"[sharded] {domain}: {len(model_specs)} models")
    write_json(args.manifest_dir / f"{args.run_id}_source_shards_manifest.json", manifest)


def build_domain_shards(
    rows: list[dict],
    *,
    domain: str,
    model_specs: list[dict[str, str]],
    output_root: Path,
    items_per_kind: int,
    seed: int,
) -> dict:
    if domain == "translation":
        return build_translation_domain_shards(
            rows,
            domain=domain,
            model_specs=model_specs,
            output_root=output_root,
            items_per_kind=items_per_kind,
            seed=seed,
        )
    if domain in SAME_SOURCE_DOMAINS:
        return build_same_source_domain_shards(
            rows,
            domain=domain,
            model_specs=model_specs,
            output_root=output_root,
            items_per_kind=items_per_kind,
            seed=seed,
        )
    needed = len(model_specs) * items_per_kind * 2
    if len(rows) < needed:
        raise ValueError(f"{domain} needs at least {needed} normalized rows, found {len(rows)}")
    rng = random.Random(f"{seed}:{domain}")
    pool = list(rows)
    rng.shuffle(pool)
    manifest = {"models": {}}
    cursor = 0
    for model_spec in model_specs:
        label = model_spec["model_label"]
        structure_rows = pool[cursor : cursor + items_per_kind]
        fact_rows = pool[cursor + items_per_kind : cursor + items_per_kind * 2]
        cursor += items_per_kind * 2
        model_dir = output_root / domain / label
        write_json(model_dir / "structure.jsonl", {"original": [row["human_text"] for row in structure_rows]})
        write_json(model_dir / "fact.jsonl", {"original": [row["fact_source_text"] for row in fact_rows]})
        write_jsonl(model_dir / "structure.sources.jsonl", source_sidecar_rows(structure_rows, "structure"))
        write_jsonl(model_dir / "fact.sources.jsonl", source_sidecar_rows(fact_rows, "fact"))
        manifest["models"][label] = {
            "provider": model_spec["provider"],
            "model": model_spec["model"],
            "structure_count": len(structure_rows),
            "fact_count": len(fact_rows),
            "structure_source_ids": [row["source_id"] for row in structure_rows],
            "fact_source_ids": [row["source_id"] for row in fact_rows],
        }
    return manifest


def build_same_source_domain_shards(
    rows: list[dict],
    *,
    domain: str,
    model_specs: list[dict[str, str]],
    output_root: Path,
    items_per_kind: int,
    seed: int,
) -> dict:
    needed = len(model_specs) * items_per_kind
    if len(rows) < needed:
        raise ValueError(f"{domain} needs at least {needed} normalized rows, found {len(rows)}")
    rng = random.Random(f"{seed}:{domain}")
    pool = list(rows)
    rng.shuffle(pool)
    manifest = {"models": {}}
    cursor = 0
    for model_spec in model_specs:
        label = model_spec["model_label"]
        selected = pool[cursor : cursor + items_per_kind]
        cursor += items_per_kind
        model_dir = output_root / domain / label
        write_json(model_dir / "structure.jsonl", {"original": [row["human_text"] for row in selected]})
        write_json(model_dir / "fact.jsonl", {"original": [row["fact_source_text"] for row in selected]})
        write_jsonl(model_dir / "structure.sources.jsonl", source_sidecar_rows(selected, "structure"))
        write_jsonl(model_dir / "fact.sources.jsonl", source_sidecar_rows(selected, "fact"))
        manifest["models"][label] = {
            "provider": model_spec["provider"],
            "model": model_spec["model"],
            "structure_count": len(selected),
            "fact_count": len(selected),
            "structure_source_ids": [row["source_id"] for row in selected],
            "fact_source_ids": [row["source_id"] for row in selected],
            "same_source_pairing": True,
        }
    return manifest


def build_translation_domain_shards(
    rows: list[dict],
    *,
    domain: str,
    model_specs: list[dict[str, str]],
    output_root: Path,
    items_per_kind: int,
    seed: int,
) -> dict:
    needed = len(model_specs) * items_per_kind
    if len(rows) < needed:
        raise ValueError(f"{domain} needs at least {needed} normalized rows, found {len(rows)}")
    by_language: dict[str, list[dict]] = {}
    for row in rows:
        language = str((row.get("metadata") or {}).get("language_config") or "unknown")
        by_language.setdefault(language, []).append(row)
    if len(by_language) < 2:
        raise ValueError(f"{domain} needs multiple languages, found {sorted(by_language)}")
    rng = random.Random(f"{seed}:{domain}")
    for language_rows in by_language.values():
        rng.shuffle(language_rows)

    language_order = sorted(by_language)
    cursors = {language: 0 for language in language_order}
    manifest = {"models": {}}
    for model_index, model_spec in enumerate(model_specs):
        label = model_spec["model_label"]
        selected: list[dict] = []
        language_counts = {language: 0 for language in language_order}
        offset = model_index % len(language_order)
        while len(selected) < items_per_kind:
            progress = False
            for step in range(len(language_order)):
                if len(selected) >= items_per_kind:
                    break
                language = language_order[(offset + step) % len(language_order)]
                cursor = cursors[language]
                if cursor >= len(by_language[language]):
                    continue
                selected.append(by_language[language][cursor])
                cursors[language] = cursor + 1
                language_counts[language] += 1
                progress = True
            if not progress:
                raise ValueError(f"{domain} ran out of rows while building balanced language shards")

        model_dir = output_root / domain / label
        write_json(model_dir / "structure.jsonl", {"original": [row["human_text"] for row in selected]})
        write_json(model_dir / "fact.jsonl", {"original": [row["fact_source_text"] for row in selected]})
        write_jsonl(model_dir / "structure.sources.jsonl", source_sidecar_rows(selected, "structure"))
        write_jsonl(model_dir / "fact.sources.jsonl", source_sidecar_rows(selected, "fact"))
        manifest["models"][label] = {
            "provider": model_spec["provider"],
            "model": model_spec["model"],
            "structure_count": len(selected),
            "fact_count": len(selected),
            "language_counts": language_counts,
            "structure_source_ids": [row["source_id"] for row in selected],
            "fact_source_ids": [row["source_id"] for row in selected],
        }
    return manifest


def source_sidecar_rows(rows: list[dict], shard_kind: str) -> list[dict]:
    output = []
    for idx, row in enumerate(rows):
        output.append(
            {
                "source_index": idx,
                "source_id": row["source_id"],
                "domain": row["domain"],
                "shard_kind": shard_kind,
                "task_input": row.get("task_input", ""),
                "metadata": row.get("metadata", {}),
            }
        )
    return output


if __name__ == "__main__":
    main()

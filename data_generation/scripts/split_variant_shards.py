#!/usr/bin/env python3
"""Split first-pass generation outputs into raw, AI-edit, and human-edit shards."""

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

from scripts.common import (
    load_domain_specs,
    load_model_specs,
    output_shard_dir,
    read_json,
    read_jsonl,
    shard_file_name,
    validate_output_row,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="final")
    parser.add_argument("--domains", nargs="+", default=None)
    parser.add_argument("--generated-root", type=Path, default=PROJECT_ROOT / "runs" / "final" / "generated")
    parser.add_argument("--source-shard-root", type=Path, default=PROJECT_ROOT / "data" / "source_shards")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "output_shards")
    parser.add_argument("--manifest-dir", type=Path, default=PROJECT_ROOT / "data" / "manifests")
    parser.add_argument("--datasets-config", type=Path, default=PROJECT_ROOT / "configs" / "datasets.yaml")
    parser.add_argument("--models-config", type=Path, default=PROJECT_ROOT / "configs" / "models.yaml")
    parser.add_argument("--raw-count", type=int, default=10)
    parser.add_argument("--ai-edit-count", type=int, default=10)
    parser.add_argument("--human-edit-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict", action="store_true", help="Fail unless every domain/model has the requested counts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    domains = args.domains or list(load_domain_specs(args.datasets_config))
    models = load_model_specs(args.models_config)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "output_root": str(args.output_root),
        "split_strategy": "deterministic_shuffle",
        "seed": args.seed,
        "domains": {},
    }
    for domain in domains:
        manifest["domains"][domain] = {}
        for model_spec in models:
            label = model_spec["model_label"]
            accepted = load_generation_records(args.generated_root, domain, label)
            if args.strict and len(accepted) < args.raw_count + args.ai_edit_count + args.human_edit_count:
                raise ValueError(f"{domain}/{label} has {len(accepted)} accepted records, not enough for strict split")
            source_map = load_structure_source_map(args.source_shard_root / args.run_id / domain / label)
            split_rows = split_records(
                accepted,
                domain=domain,
                model_label=label,
                run_id=args.run_id,
                source_map=source_map,
                raw_count=args.raw_count,
                ai_edit_count=args.ai_edit_count,
                human_edit_count=args.human_edit_count,
                seed=args.seed,
            )
            counts = {}
            for variant_type, rows in split_rows.items():
                shard_path = output_shard_dir(args.output_root, args.run_id, domain, label, variant_type) / shard_file_name(
                    domain,
                    label,
                    variant_type,
                )
                for row in rows:
                    validate_output_row(row)
                write_jsonl(shard_path, rows)
                counts[variant_type] = len(rows)
            manifest["domains"][domain][label] = counts
            print(f"[split] {domain}/{label}: {counts}")
    manifest_name = f"{args.run_id}_variant_split_manifest"
    if args.output_root.name != "output_shards":
        manifest_name += f"_{args.output_root.name}"
    write_json(args.manifest_dir / f"{manifest_name}.json", manifest)


def load_generation_records(generated_root: Path, domain: str, model_label: str) -> list[dict]:
    artifact_path = generated_root / model_label / domain / "artifacts" / f"{domain}_{model_label}.accepted.jsonl"
    if artifact_path.exists():
        rows = read_jsonl(artifact_path)
        if rows:
            return rows
    raw_path = generated_root / model_label / domain / f"{domain}_{model_label}.data.json"
    payload = read_json(raw_path)
    originals = payload.get("original", [])
    sampled = payload.get("sampled", [])
    return [
        {
            "pair_id": f"raw_{idx:06d}",
            "original": original,
            "sampled": sample,
            "structure_source_index": idx,
            "metrics": {},
        }
        for idx, (original, sample) in enumerate(zip(originals, sampled))
    ]


def load_structure_source_map(model_source_dir: Path) -> dict[int, dict]:
    rows = read_jsonl(model_source_dir / "structure.sources.jsonl")
    return {int(row["source_index"]): row for row in rows}


def split_records(
    accepted: list[dict],
    *,
    domain: str,
    model_label: str,
    run_id: str,
    source_map: dict[int, dict],
    raw_count: int,
    ai_edit_count: int,
    human_edit_count: int,
    seed: int = 42,
) -> dict[str, list[dict]]:
    shuffled = list(accepted)
    rng = random.Random(f"{seed}:{run_id}:{domain}:{model_label}:variant_split")
    rng.shuffle(shuffled)
    raw_records = shuffled[:raw_count]
    ai_records = shuffled[raw_count : raw_count + ai_edit_count]
    human_records = shuffled[raw_count + ai_edit_count : raw_count + ai_edit_count + human_edit_count]
    return {
        "raw_ai": make_rows(raw_records, domain, model_label, run_id, "raw_ai", source_map, split_seed=seed),
        "ai_edit_input": make_rows(ai_records, domain, model_label, run_id, "ai_edit_input", source_map, split_seed=seed),
        "human_edit_input": make_rows(
            human_records,
            domain,
            model_label,
            run_id,
            "human_edit_input",
            source_map,
            split_seed=seed,
        ),
    }


def make_rows(
    records: list[dict],
    domain: str,
    model_label: str,
    run_id: str,
    variant_type: str,
    source_map: dict[int, dict],
    split_seed: int,
) -> list[dict]:
    output = []
    for idx, record in enumerate(records):
        source_index = int(record.get("structure_source_index", idx))
        source_info = source_map.get(source_index, {})
        source_id = source_info.get("source_id") or record.get("pair_id") or f"{source_index}"
        row = {
            "id": f"{run_id}:{domain}:{model_label}:{variant_type}:{idx:04d}",
            "domain": domain,
            "model_label": model_label,
            "variant_type": variant_type,
            "source_id": source_id,
            "original": record["original"],
            "sampled": record["sampled"],
            "metadata": {
                "pair_id": record.get("pair_id"),
                "structure_source_index": source_index,
                "fact_source_index": record.get("fact_source_index"),
                "generation_attempt": record.get("attempt"),
                "metrics": record.get("metrics", {}),
                "source": source_info,
                "split": {
                    "strategy": "deterministic_shuffle",
                    "seed": split_seed,
                    "variant_rank": idx,
                },
            },
        }
        output.append(row)
    return output


if __name__ == "__main__":
    main()

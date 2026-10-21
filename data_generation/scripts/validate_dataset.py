#!/usr/bin/env python3
"""Validate normalized records, source shards, output shards, or final merged JSONL."""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import (
    FINAL_VARIANTS,
    iter_shard_files,
    load_domain_specs,
    load_model_specs,
    output_shard_dir,
    read_json,
    read_jsonl,
    validate_output_row,
    word_count,
)


SAME_SOURCE_SHARD_DOMAINS = {"meeting_summary", "product_review", "translation"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["normalized", "source_shards", "output_shards", "final"], required=True)
    parser.add_argument("--run-id", default="final")
    parser.add_argument("--domains", nargs="+", default=None)
    parser.add_argument("--datasets-config", type=Path, default=PROJECT_ROOT / "configs" / "datasets.yaml")
    parser.add_argument("--models-config", type=Path, default=PROJECT_ROOT / "configs" / "models.yaml")
    parser.add_argument("--normalized-dir", type=Path, default=PROJECT_ROOT / "data" / "normalized")
    parser.add_argument("--source-shard-root", type=Path, default=PROJECT_ROOT / "data" / "source_shards")
    parser.add_argument("--output-shard-root", type=Path, default=PROJECT_ROOT / "data" / "output_shards")
    parser.add_argument("--final-jsonl", type=Path, default=PROJECT_ROOT / "runs" / "final" / "our_data_aigt.jsonl")
    parser.add_argument("--expected-source-items-per-kind", type=int, default=60)
    parser.add_argument("--expected-final-per-variant", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = load_domain_specs(args.datasets_config)
    domains = args.domains or list(specs)
    models = load_model_specs(args.models_config)
    if args.mode == "normalized":
        validate_normalized(args, specs, domains)
    elif args.mode == "source_shards":
        validate_source_shards(args, domains, models)
    elif args.mode == "output_shards":
        validate_output_shards(args, domains, models)
    else:
        validate_final(args, domains, models)
    print(f"[valid] {args.mode}")


def validate_normalized(args: argparse.Namespace, specs: dict, domains: list[str]) -> None:
    required = {"source_id", "domain", "source_dataset", "split", "task_input", "human_text", "fact_source_text", "metadata"}
    for domain in domains:
        rows = read_jsonl(args.normalized_dir / domain / "records.jsonl")
        if not rows:
            raise ValueError(f"{domain}: no normalized rows")
        min_words = int(specs[domain].get("min_words") or 1)
        ids = []
        for row in rows:
            missing = required - set(row)
            if missing:
                raise ValueError(f"{domain}: missing normalized keys {missing}")
            if row["domain"] != domain:
                raise ValueError(f"{domain}: row has wrong domain {row['domain']}")
            if word_count(row["human_text"]) < min_words:
                raise ValueError(f"{domain}: row {row['source_id']} is shorter than min_words")
            ids.append(row["source_id"])
        ensure_no_duplicates(ids, f"{domain} source_id")


def validate_source_shards(args: argparse.Namespace, domains: list[str], models: list[dict[str, str]]) -> None:
    for domain in domains:
        model_sources = []
        for model in models:
            label = model["model_label"]
            model_dir = args.source_shard_root / args.run_id / domain / label
            structure = read_json(model_dir / "structure.jsonl").get("original", [])
            fact = read_json(model_dir / "fact.jsonl").get("original", [])
            if len(structure) != args.expected_source_items_per_kind or len(fact) != args.expected_source_items_per_kind:
                raise ValueError(f"{domain}/{label}: unexpected source shard counts {len(structure)}/{len(fact)}")
            structure_ids = [row["source_id"] for row in read_jsonl(model_dir / "structure.sources.jsonl")]
            fact_ids = [row["source_id"] for row in read_jsonl(model_dir / "fact.sources.jsonl")]
            if domain in SAME_SOURCE_SHARD_DOMAINS:
                if structure_ids != fact_ids:
                    raise ValueError(f"{domain}/{label}: structure and fact source ids must match by row")
            elif set(structure_ids) & set(fact_ids):
                raise ValueError(f"{domain}/{label}: structure and fact source ids overlap")
            model_sources.append((label, set(structure_ids + fact_ids)))
        for idx, (left_label, left_ids) in enumerate(model_sources):
            for right_label, right_ids in model_sources[idx + 1 :]:
                overlap = left_ids & right_ids
                if overlap:
                    raise ValueError(f"{domain}: model source overlap {left_label}/{right_label}: {list(overlap)[:3]}")


def validate_output_shards(args: argparse.Namespace, domains: list[str], models: list[dict[str, str]]) -> None:
    for domain in domains:
        for model in models:
            label = model["model_label"]
            for variant in FINAL_VARIANTS:
                rows = read_output_variant(args.output_shard_root, args.run_id, domain, label, variant)
                if len(rows) != args.expected_final_per_variant:
                    raise ValueError(f"{domain}/{label}/{variant}: expected {args.expected_final_per_variant}, found {len(rows)}")
                for row in rows:
                    validate_output_row(row)


def validate_final(args: argparse.Namespace, domains: list[str], models: list[dict[str, str]]) -> None:
    rows = read_jsonl(args.final_jsonl)
    ids = []
    counts = defaultdict(Counter)
    for row in rows:
        validate_output_row(row)
        ids.append(row["id"])
        counts[row["domain"]][(row["model_label"], row["variant_type"])] += 1
    ensure_no_duplicates(ids, "final id")
    for domain in domains:
        for model in models:
            for variant in FINAL_VARIANTS:
                count = counts[domain][(model["model_label"], variant)]
                if count != args.expected_final_per_variant:
                    raise ValueError(f"{domain}/{model['model_label']}/{variant}: expected {args.expected_final_per_variant}, found {count}")


def read_output_variant(root: Path, run_id: str, domain: str, model_label: str, variant: str) -> list[dict]:
    variant_dir = output_shard_dir(root, run_id, domain, model_label, variant)
    rows = []
    for path in iter_shard_files(variant_dir):
        rows.extend(read_jsonl(path))
    return rows


def ensure_no_duplicates(values: list[str], name: str) -> None:
    duplicates = [value for value, count in Counter(values).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate {name}: {duplicates[:5]}")


if __name__ == "__main__":
    main()

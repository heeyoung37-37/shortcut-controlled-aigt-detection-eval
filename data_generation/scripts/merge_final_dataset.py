#!/usr/bin/env python3
"""Merge raw_ai, ai_edited, and human_edited shards into the final JSONL dataset."""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import (
    ALL_OUTPUT_VARIANTS,
    FINAL_VARIANTS,
    iter_shard_files,
    load_domain_specs,
    load_model_specs,
    output_shard_dir,
    read_jsonl,
    safe_filename_component,
    validate_output_row,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="final")
    parser.add_argument("--domains", nargs="+", default=None)
    parser.add_argument("--input-root", type=Path, default=PROJECT_ROOT / "data" / "output_shards")
    parser.add_argument("--output-jsonl", type=Path, default=PROJECT_ROOT / "runs" / "final" / "our_data_aigt.jsonl")
    parser.add_argument("--fdg-output-root", type=Path, default=PROJECT_ROOT / "runs" / "final" / "fdg_format")
    parser.add_argument("--manifest-path", type=Path, default=PROJECT_ROOT / "data" / "manifests" / "final_dataset_manifest.json")
    parser.add_argument("--datasets-config", type=Path, default=PROJECT_ROOT / "configs" / "datasets.yaml")
    parser.add_argument("--models-config", type=Path, default=PROJECT_ROOT / "configs" / "models.yaml")
    parser.add_argument("--variants", nargs="+", default=list(FINAL_VARIANTS), choices=list(ALL_OUTPUT_VARIANTS))
    parser.add_argument("--expected-per-variant", type=int, default=10)
    parser.add_argument(
        "--omit-variant-dir",
        action="store_true",
        help="Write FDG exports directly under {domain}/filename when exporting a single variant.",
    )
    parser.add_argument("--no-fdg-export", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    domains = args.domains or list(load_domain_specs(args.datasets_config))
    models = load_model_specs(args.models_config)
    rows = []
    counts = defaultdict(Counter)
    for domain in domains:
        for model in models:
            label = model["model_label"]
            for variant in args.variants:
                variant_rows = read_variant_rows(args.input_root, args.run_id, domain, label, variant)
                if args.strict and len(variant_rows) != args.expected_per_variant:
                    raise ValueError(
                        f"{domain}/{label}/{variant}: expected {args.expected_per_variant}, found {len(variant_rows)}"
                    )
                for row in variant_rows:
                    validate_output_row(row)
                    if row["variant_type"] != variant:
                        raise ValueError(f"{row['id']} has variant_type={row['variant_type']}, expected {variant}")
                counts[domain][f"{label}:{variant}"] = len(variant_rows)
                rows.extend(variant_rows)
    validate_unique(rows, "id")
    write_jsonl(args.output_jsonl, rows)
    fdg_files = [] if args.no_fdg_export else write_fdg_exports(
        rows,
        args.fdg_output_root,
        omit_variant_dir=args.omit_variant_dir,
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "row_count": len(rows),
        "expected_per_variant": args.expected_per_variant,
        "variants": args.variants,
        "counts": {domain: dict(counter) for domain, counter in counts.items()},
        "output_jsonl": str(args.output_jsonl),
        "fdg_output_root": None if args.no_fdg_export else str(args.fdg_output_root),
        "fdg_files": fdg_files,
    }
    write_json(args.manifest_path, manifest)
    print(f"[merged] {len(rows)} rows -> {args.output_jsonl}")
    if not args.no_fdg_export:
        print(f"[fdg-export] {len(fdg_files)} files -> {args.fdg_output_root}")


def read_variant_rows(root: Path, run_id: str, domain: str, model_label: str, variant: str) -> list[dict]:
    variant_dir = output_shard_dir(root, run_id, domain, model_label, variant)
    rows = []
    for path in iter_shard_files(variant_dir):
        rows.extend(read_jsonl(path))
    return rows


def validate_unique(rows: list[dict], key: str) -> None:
    values = [row.get(key) for row in rows]
    duplicates = [value for value, count in Counter(values).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate {key} values: {duplicates[:5]}")


def write_fdg_exports(rows: list[dict], output_root: Path, *, omit_variant_dir: bool = False) -> list[str]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["domain"], row["model_label"], row["variant_type"])].append(row)

    written = []
    for (domain, model_label, variant), group_rows in sorted(grouped.items()):
        filename = (
            f"{safe_filename_component(domain)}_"
            f"{safe_filename_component(model_label)}_"
            f"{safe_filename_component(variant)}.json"
        )
        path = output_root / domain / filename if omit_variant_dir else output_root / variant / domain / filename
        payload = {
            "original": [row["original"] for row in group_rows],
            "sampled": [row["sampled"] for row in group_rows],
        }
        write_json(path, payload)
        written.append(str(path))
    return written


if __name__ == "__main__":
    main()

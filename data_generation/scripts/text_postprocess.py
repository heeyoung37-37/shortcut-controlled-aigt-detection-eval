#!/usr/bin/env python3
"""Sentence-safe cleanup for generated Our_data text artifacts."""

from __future__ import annotations

import argparse
import html
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import (
    ALL_OUTPUT_VARIANTS,
    clean_text,
    iter_shard_files,
    load_domain_specs,
    load_model_specs,
    read_json,
    read_jsonl,
    word_count,
    write_json,
    write_jsonl,
)


CLOSERS = r"[\]\})'\"\u201d\u2019\u00bb]*"
TERMINAL_RE = re.compile(rf"(?:[.!?\u3002\uff01\uff1f\u2026]+|[\U0001F300-\U0001FAFF]){CLOSERS}$")
EMOTICON_RE = re.compile(rf"(?:[:;=xX8]-?[)(DPpOo/]|<3){CLOSERS}$")
BRACKETED_DISCLOSURE_RE = re.compile(rf"\[[^\[\]]{{4,120}}\]{CLOSERS}$")
SENTENCE_BOUNDARY_RE = re.compile(rf"[.!?\u3002\uff01\uff1f\u2026]+{CLOSERS}")
HTML_BREAK_RE = re.compile(r"<\s*br\s*/?\s*>", re.IGNORECASE)
HTML_PARAGRAPH_RE = re.compile(r"</\s*p\s*>", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class TextPostprocessResult:
    text: str
    action: str
    original_word_count: int
    final_word_count: int
    removed_word_count: int
    complete_before: bool
    complete_after: bool


def surface_clean_text(text: str) -> str:
    """Normalize whitespace and common HTML wrappers without changing content intent."""
    cleaned = html.unescape(text or "")
    cleaned = HTML_BREAK_RE.sub("\n", cleaned)
    cleaned = HTML_PARAGRAPH_RE.sub("\n\n", cleaned)
    cleaned = HTML_TAG_RE.sub(" ", cleaned)
    return clean_text(cleaned)


def has_complete_ending(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    return bool(
        TERMINAL_RE.search(stripped)
        or EMOTICON_RE.search(stripped)
        or BRACKETED_DISCLOSURE_RE.search(stripped)
    )


def sentence_safe_postprocess_text(
    text: str,
    *,
    min_words: int = 1,
    max_loss_ratio: float = 0.35,
) -> TextPostprocessResult:
    """Trim only a clearly incomplete final sentence, and only when enough text remains."""
    cleaned = surface_clean_text(text)
    original_words = word_count(cleaned)
    complete_before = has_complete_ending(cleaned)
    if not cleaned:
        return TextPostprocessResult("", "empty", 0, 0, 0, False, False)
    if complete_before:
        return TextPostprocessResult(cleaned, "complete", original_words, original_words, 0, True, True)

    matches = list(SENTENCE_BOUNDARY_RE.finditer(cleaned))
    if not matches:
        return TextPostprocessResult(
            cleaned,
            "keep_no_boundary",
            original_words,
            original_words,
            0,
            False,
            has_complete_ending(cleaned),
        )

    candidate = cleaned[: matches[-1].end()].strip()
    candidate_words = word_count(candidate)
    removed_words = max(0, original_words - candidate_words)
    if not candidate:
        action = "keep_empty_after_trim"
    elif candidate_words < min_words:
        action = "keep_below_min"
    elif original_words and removed_words / original_words > max_loss_ratio:
        action = "keep_big_loss"
    else:
        return TextPostprocessResult(
            candidate,
            "trimmed",
            original_words,
            candidate_words,
            removed_words,
            False,
            has_complete_ending(candidate),
        )

    return TextPostprocessResult(
        cleaned,
        action,
        original_words,
        original_words,
        0,
        False,
        has_complete_ending(cleaned),
    )


def flatten_product_review_text(text: str) -> str:
    """Return product reviews as one paragraph in final detector-facing files."""
    flattened = re.sub(r"\s*\n+\s*", " ", text or "")
    flattened = re.sub(r"[ \t]{2,}", " ", flattened)
    return flattened.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="final")
    parser.add_argument("--domains", nargs="+", default=None)
    parser.add_argument("--model-labels", nargs="+", default=None)
    parser.add_argument("--mode", choices=["generated", "output_shards", "both"], default="both")
    parser.add_argument("--generated-root", type=Path, default=PROJECT_ROOT / "runs" / "final" / "generated")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "output_shards")
    parser.add_argument("--manifest-dir", type=Path, default=PROJECT_ROOT / "data" / "manifests")
    parser.add_argument("--datasets-config", type=Path, default=PROJECT_ROOT / "configs" / "datasets.yaml")
    parser.add_argument("--models-config", type=Path, default=PROJECT_ROOT / "configs" / "models.yaml")
    parser.add_argument("--variants", nargs="+", default=list(ALL_OUTPUT_VARIANTS))
    parser.add_argument("--max-loss-ratio", type=float, default=0.35)
    parser.add_argument("--min-trim-words", type=int, default=20)
    parser.add_argument("--write", action="store_true", help="Write cleaned files. Default is dry-run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    domain_specs = load_domain_specs(args.datasets_config)
    model_specs = load_model_specs(args.models_config)
    domains = args.domains or list(domain_specs)
    model_labels = args.model_labels or [spec["model_label"] for spec in model_specs]
    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "dry_run": not args.write,
        "mode": args.mode,
        "domains": {},
    }

    if args.mode in {"generated", "both"}:
        for domain in domains:
            for model_label in model_labels:
                summary = process_generated_artifacts(args, domain, model_label, domain_specs)
                if summary:
                    manifest["domains"].setdefault(domain, {}).setdefault(model_label, {})["generated"] = summary

    if args.mode in {"output_shards", "both"}:
        for domain in domains:
            for model_label in model_labels:
                summary = process_output_shards(args, domain, model_label, domain_specs)
                if summary:
                    manifest["domains"].setdefault(domain, {}).setdefault(model_label, {})["output_shards"] = summary

    for domain, models in manifest["domains"].items():
        for model_label, summary in models.items():
            print(f"[postprocess] {domain}/{model_label}: {summary}")

    if args.write:
        write_json(args.manifest_dir / f"{args.run_id}_sentence_postprocess_manifest.json", manifest)


def min_words_for_domain(domain: str, domain_specs: dict[str, dict[str, Any]], trim_floor: int) -> int:
    domain_min = int((domain_specs.get(domain) or {}).get("min_words") or 1)
    return max(1, min(domain_min, trim_floor))


def process_generated_artifacts(
    args: argparse.Namespace,
    domain: str,
    model_label: str,
    domain_specs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    domain_dir = args.generated_root / model_label / domain
    data_path = domain_dir / f"{domain}_{model_label}.data.json"
    accepted_path = domain_dir / "artifacts" / f"{domain}_{model_label}.accepted.jsonl"
    summary: dict[str, Any] = {}
    min_words = min_words_for_domain(domain, domain_specs, args.min_trim_words)
    if data_path.exists():
        payload = read_json(data_path)
        changed, stats = postprocess_raw_payload(payload, min_words=min_words, max_loss_ratio=args.max_loss_ratio, domain=domain)
        summary["data_json"] = stats
        if args.write and changed:
            write_json(data_path, payload)
    if accepted_path.exists():
        rows = read_jsonl(accepted_path)
        changed, stats = postprocess_rows(rows, min_words=min_words, max_loss_ratio=args.max_loss_ratio, domain=domain)
        summary["accepted_jsonl"] = stats
        if args.write and changed:
            write_jsonl(accepted_path, rows)
    return summary


def process_output_shards(
    args: argparse.Namespace,
    domain: str,
    model_label: str,
    domain_specs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    root = args.output_root / args.run_id / domain / model_label
    if not root.exists():
        return {}
    summary: dict[str, Any] = {}
    min_words = min_words_for_domain(domain, domain_specs, args.min_trim_words)
    for variant in args.variants:
        variant_dir = root / variant
        if not variant_dir.exists():
            continue
        variant_stats = []
        for shard_path in iter_shard_files(variant_dir):
            rows = read_jsonl(shard_path)
            changed, stats = postprocess_rows(rows, min_words=min_words, max_loss_ratio=args.max_loss_ratio, domain=domain)
            variant_stats.append({"path": str(shard_path), **stats})
            if args.write and changed:
                write_jsonl(shard_path, rows)
        if variant_stats:
            summary[variant] = variant_stats
    return summary


def postprocess_raw_payload(
    payload: dict[str, Any],
    *,
    min_words: int,
    max_loss_ratio: float,
    domain: str,
) -> tuple[bool, dict[str, Any]]:
    changed = False
    stats = new_stats()
    for field in ("original", "sampled"):
        values = payload.get(field, [])
        if not isinstance(values, list):
            continue
        for idx, value in enumerate(values):
            result = sentence_safe_postprocess_text(
                str(value),
                min_words=min_words,
                max_loss_ratio=max_loss_ratio,
            )
            final_text = result.text
            flattened_product_review = False
            if domain == "product_review":
                flattened = flatten_product_review_text(final_text)
                flattened_product_review = flattened != final_text
                final_text = flattened
            update_stats(stats, field, result)
            if flattened_product_review:
                stats["fields"][field]["product_review_flattened"] += 1
            if final_text != value:
                values[idx] = final_text
                changed = True
    return changed, compact_stats(stats)


def postprocess_rows(
    rows: list[dict[str, Any]],
    *,
    min_words: int,
    max_loss_ratio: float,
    domain: str,
) -> tuple[bool, dict[str, Any]]:
    changed = False
    stats = new_stats()
    for row in rows:
        row_postprocess: dict[str, Any] = {}
        for field in ("original", "sampled"):
            if field not in row:
                continue
            value = str(row.get(field) or "")
            result = sentence_safe_postprocess_text(
                value,
                min_words=min_words,
                max_loss_ratio=max_loss_ratio,
            )
            final_text = result.text
            flattened_product_review = False
            if domain == "product_review":
                flattened = flatten_product_review_text(final_text)
                flattened_product_review = flattened != final_text
                final_text = flattened
            update_stats(stats, field, result)
            if flattened_product_review:
                stats["fields"][field]["product_review_flattened"] += 1
            row_postprocess[field] = asdict(result) | {
                "text": None,
                "product_review_flattened": flattened_product_review,
            }
            if final_text != value:
                row[field] = final_text
                changed = True
        if row_postprocess:
            metadata = row.setdefault("metadata", {})
            metadata["sentence_postprocess"] = row_postprocess
            changed = True
    return changed, compact_stats(stats)


def new_stats() -> dict[str, Any]:
    return {
        "fields": defaultdict(Counter),
        "rows": 0,
    }


def update_stats(stats: dict[str, Any], field: str, result: TextPostprocessResult) -> None:
    stats["fields"][field][result.action] += 1
    if field == "sampled":
        stats["rows"] += 1


def compact_stats(stats: dict[str, Any]) -> dict[str, Any]:
    fields = {field: dict(counter) for field, counter in stats["fields"].items()}
    needs_review = 0
    for counter in fields.values():
        for action, count in counter.items():
            if action.startswith("keep_") or action == "empty":
                needs_review += count
    return {
        "rows": stats["rows"],
        "fields": fields,
        "needs_review_fields": needs_review,
    }


if __name__ == "__main__":
    main()

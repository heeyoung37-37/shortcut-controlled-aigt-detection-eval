#!/usr/bin/env python3
"""Build a length-matched detector benchmark export from FDG-format files."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import read_json, word_count, write_json
from scripts.text_postprocess import flatten_product_review_text, has_complete_ending, surface_clean_text


WORD_RE = re.compile(r"\b[\w'-]+\b")
SENTENCE_BOUNDARY_RE = re.compile(r"[.!?\u3002\uff01\uff1f\u2026]+[\]\})'\"\u201d\u2019\u00bb]*")
MEETING_BOUNDARY_RE = re.compile(r"(?:[.!?\u3002\uff01\uff1f\u2026]+|;)[\]\})'\"\u201d\u2019\u00bb]*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=PROJECT_ROOT / "Benchmark")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "Benchmark_length_matched")
    parser.add_argument("--variants", nargs="+", default=["raw_ai", "ai_edited"])
    parser.add_argument(
        "--match-ratio-min",
        type=float,
        default=0.8,
        help="Only trim original when sampled/original is below this ratio.",
    )
    parser.add_argument(
        "--match-ratio-max",
        type=float,
        default=1.25,
        help="Only trim sampled when sampled/original is above this ratio.",
    )
    parser.add_argument("--min-words", type=int, default=20)
    parser.add_argument(
        "--max-boundary-overshoot-ratio",
        type=float,
        default=0.15,
        help="Allow trimming at the next complete boundary when it exceeds the shorter side by this ratio.",
    )
    parser.add_argument(
        "--max-boundary-overshoot-words",
        type=int,
        default=30,
        help="Absolute cap for the next-boundary overshoot allowance.",
    )
    parser.add_argument(
        "--min-prev-boundary-ratio",
        type=float,
        default=0.65,
        help="Use the next complete boundary when the previous boundary would be shorter than this share of target.",
    )
    parser.add_argument(
        "--max-ending-loss-ratio",
        type=float,
        default=0.45,
        help="If final-ending cleanup would remove more than this share, keep text and flag it.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.output_root} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = build_benchmark(args)
    write_json(args.output_root / "manifest.json", manifest)
    print_summary(manifest)


def build_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "variants": args.variants,
        "files": [],
        "summary": {
            "files": 0,
            "pairs": 0,
            "length_trimmed_fields": 0,
            "ending_trimmed_fields": 0,
            "unresolved_incomplete_fields": 0,
            "empty_fields": 0,
            "ratio_bands_before": {},
            "ratio_bands_after": {},
        },
    }
    before_bands = Counter()
    after_bands = Counter()
    for input_path in sorted(args.input_root.glob("*/*/*.json")):
        variant = input_path.parts[-3]
        domain = input_path.parts[-2]
        if variant not in args.variants:
            continue
        payload = read_json(input_path)
        original_values = payload.get("original")
        sampled_values = payload.get("sampled")
        if not isinstance(original_values, list) or not isinstance(sampled_values, list):
            raise ValueError(f"{input_path} must contain original/sampled lists")
        if len(original_values) != len(sampled_values):
            raise ValueError(f"{input_path} original/sampled length mismatch")
        output_original: list[str] = []
        output_sampled: list[str] = []
        file_stats = new_file_stats(input_path, variant, domain)
        for index, (original, sampled) in enumerate(zip(original_values, sampled_values)):
            result = process_pair(str(original), str(sampled), domain, args)
            output_original.append(result["original"])
            output_sampled.append(result["sampled"])
            update_stats(file_stats, result, index)
            before_bands[ratio_band(result["before_ratio"])] += 1
            after_bands[ratio_band(result["after_ratio"])] += 1
        output_path = args.output_root / input_path.relative_to(args.input_root)
        write_json(output_path, {"original": output_original, "sampled": output_sampled})
        manifest["files"].append(file_stats)
    summary = manifest["summary"]
    summary["files"] = len(manifest["files"])
    summary["pairs"] = sum(file_item["pairs"] for file_item in manifest["files"])
    for key in ("length_trimmed_fields", "ending_trimmed_fields", "unresolved_incomplete_fields", "empty_fields"):
        summary[key] = sum(file_item[key] for file_item in manifest["files"])
    summary["ratio_bands_before"] = dict(sorted(before_bands.items()))
    summary["ratio_bands_after"] = dict(sorted(after_bands.items()))
    return manifest


def process_pair(original: str, sampled: str, domain: str, args: argparse.Namespace) -> dict[str, Any]:
    clean_original = cleanup_surface(original, domain)
    clean_sampled = cleanup_surface(sampled, domain)
    original_words = word_count(clean_original)
    sampled_words = word_count(clean_sampled)
    target_words = min(original_words, sampled_words)
    original_result = FieldResult(clean_original, False, False, False, "")
    sampled_result = FieldResult(clean_sampled, False, False, False, "")
    before_ratio = safe_ratio(sampled_words, original_words)
    if target_words > 0 and original_words > sampled_words and before_ratio < args.match_ratio_min:
        original_result = trim_to_shorter_side(clean_original, target_words, domain, args)
    elif target_words > 0 and sampled_words > original_words and before_ratio > args.match_ratio_max:
        sampled_result = trim_to_shorter_side(clean_sampled, target_words, domain, args)
    original_result = cleanup_ending(original_result.text, domain, args, already_length_trimmed=original_result.length_trimmed)
    sampled_result = cleanup_ending(sampled_result.text, domain, args, already_length_trimmed=sampled_result.length_trimmed)
    original_result, sampled_result = rebalance_after_cleanup(original_result, sampled_result, domain, args)
    if domain == "product_review":
        original_result.text = flatten_product_review_text(original_result.text)
        sampled_result.text = flatten_product_review_text(sampled_result.text)
    after_original_words = word_count(original_result.text)
    after_sampled_words = word_count(sampled_result.text)
    return {
        "original": original_result.text,
        "sampled": sampled_result.text,
        "before_ratio": before_ratio,
        "after_ratio": safe_ratio(after_sampled_words, after_original_words),
        "original_stats": original_result.as_dict(before_words=original_words, after_words=after_original_words),
        "sampled_stats": sampled_result.as_dict(before_words=sampled_words, after_words=after_sampled_words),
    }


class FieldResult:
    def __init__(self, text: str, length_trimmed: bool, ending_trimmed: bool, unresolved: bool, action: str) -> None:
        self.text = text
        self.length_trimmed = length_trimmed
        self.ending_trimmed = ending_trimmed
        self.unresolved = unresolved
        self.action = action

    def as_dict(self, *, before_words: int, after_words: int) -> dict[str, Any]:
        return {
            "before_words": before_words,
            "after_words": after_words,
            "length_trimmed": self.length_trimmed,
            "ending_trimmed": self.ending_trimmed,
            "unresolved_incomplete": self.unresolved,
            "action": self.action,
        }


def cleanup_surface(text: str, domain: str) -> str:
    cleaned = surface_clean_text(text)
    if domain == "product_review":
        return flatten_product_review_text(cleaned)
    return cleaned


def trim_to_shorter_side(text: str, target_words: int, domain: str, args: argparse.Namespace) -> FieldResult:
    if word_count(text) <= target_words:
        return FieldResult(text, False, False, False, "unchanged")
    trimmed, action = boundary_prefix_near_target(text, target_words, domain, args)
    return FieldResult(trimmed, True, False, False, action)


def cleanup_ending(text: str, domain: str, args: argparse.Namespace, *, already_length_trimmed: bool) -> FieldResult:
    cleaned = cleanup_surface(text, domain)
    if is_complete_enough(cleaned, domain):
        return FieldResult(cleaned, already_length_trimmed, False, False, "complete")
    before_words = word_count(cleaned)
    candidate = complete_prefix(cleaned, domain)
    candidate_words = word_count(candidate)
    if not candidate:
        return FieldResult(cleaned, already_length_trimmed, False, True, "unresolved_no_boundary")
    if candidate_words < min(args.min_words, max(before_words, 1)):
        return FieldResult(cleaned, already_length_trimmed, False, True, "unresolved_below_min")
    loss_ratio = (before_words - candidate_words) / before_words if before_words else 0.0
    if not already_length_trimmed and loss_ratio > args.max_ending_loss_ratio:
        return FieldResult(cleaned, already_length_trimmed, False, True, "unresolved_big_loss")
    return FieldResult(candidate, already_length_trimmed, True, False, "ending_trimmed")


def prefix_to_word_count(text: str, target_words: int) -> tuple[str, str]:
    matches = list(WORD_RE.finditer(text or ""))
    if len(matches) <= target_words:
        return (text or "").strip(), "unchanged"
    cutoff = matches[max(target_words - 1, 0)].end()
    return (text or "")[:cutoff].strip(), "length_trimmed_word_boundary"


def boundary_prefix_near_target(
    text: str,
    target_words: int,
    domain: str,
    args: argparse.Namespace,
) -> tuple[str, str]:
    boundary_options = boundary_prefix_options(text, domain)
    if not boundary_options:
        return prefix_to_word_count(text, target_words)
    min_words = min(args.min_words, max(target_words, 1))
    before = [item for item in boundary_options if min_words <= item[1] <= target_words]
    allowed_extra = min(
        args.max_boundary_overshoot_words,
        max(5, int(round(target_words * args.max_boundary_overshoot_ratio))),
    )
    max_allowed = target_words + allowed_extra
    after = [item for item in boundary_options if target_words < item[1] <= max_allowed]
    best_before = max(before, key=lambda item: item[1], default=None)
    best_after = min(after, key=lambda item: item[1], default=None)
    any_after = [item for item in boundary_options if item[1] > target_words]
    first_after = min(any_after, key=lambda item: item[1], default=None)
    if best_before and first_after and best_before[1] < target_words * args.min_prev_boundary_ratio:
        chosen = best_after or first_after
        return (text or "")[: chosen[0]].strip(), "length_trimmed_next_boundary_to_avoid_fragment"
    if best_before and best_after:
        before_gap = target_words - best_before[1]
        after_gap = best_after[1] - target_words
        chosen = best_after if after_gap < before_gap else best_before
        action = "length_trimmed_next_boundary" if chosen is best_after else "length_trimmed_prev_boundary"
        return (text or "")[: chosen[0]].strip(), action
    if best_before:
        return (text or "")[: best_before[0]].strip(), "length_trimmed_prev_boundary"
    if best_after:
        return (text or "")[: best_after[0]].strip(), "length_trimmed_next_boundary"
    any_before = [item for item in boundary_options if item[1] <= target_words]
    if first_after:
        best_any_before = max(any_before, key=lambda item: item[1], default=None)
        if best_any_before is None or best_any_before[1] < target_words * args.min_prev_boundary_ratio:
            return (text or "")[: first_after[0]].strip(), "length_trimmed_first_boundary_overshoot"
    if any_before:
        chosen = max(any_before, key=lambda item: item[1])
        return (text or "")[: chosen[0]].strip(), "length_trimmed_prev_boundary_big_loss"
    if first_after:
        return (text or "")[: first_after[0]].strip(), "length_trimmed_first_boundary_overshoot"
    chosen = min(boundary_options, key=lambda item: item[1])
    return (text or "")[: chosen[0]].strip(), "length_trimmed_first_boundary_overshoot"


def boundary_prefix_options(text: str, domain: str) -> list[tuple[int, int]]:
    pattern = MEETING_BOUNDARY_RE if domain == "meeting_summary" else SENTENCE_BOUNDARY_RE
    options = []
    for match in pattern.finditer(text or ""):
        if not is_valid_boundary(text or "", match):
            continue
        prefix = (text or "")[: match.end()].strip()
        words = word_count(prefix)
        if words:
            options.append((match.end(), words))
    return options


def rebalance_after_cleanup(
    original_result: FieldResult,
    sampled_result: FieldResult,
    domain: str,
    args: argparse.Namespace,
) -> tuple[FieldResult, FieldResult]:
    for _ in range(2):
        original_words = word_count(original_result.text)
        sampled_words = word_count(sampled_result.text)
        ratio = safe_ratio(sampled_words, original_words)
        if original_words and sampled_words and ratio < args.match_ratio_min and original_words > sampled_words:
            trimmed = trim_to_shorter_side(original_result.text, sampled_words, domain, args)
            cleaned = cleanup_ending(trimmed.text, domain, args, already_length_trimmed=trimmed.length_trimmed)
            original_result = merge_field_results(original_result, cleaned)
            continue
        if original_words and sampled_words and ratio > args.match_ratio_max and sampled_words > original_words:
            trimmed = trim_to_shorter_side(sampled_result.text, original_words, domain, args)
            cleaned = cleanup_ending(trimmed.text, domain, args, already_length_trimmed=trimmed.length_trimmed)
            sampled_result = merge_field_results(sampled_result, cleaned)
            continue
        break
    return original_result, sampled_result


def merge_field_results(previous: FieldResult, current: FieldResult) -> FieldResult:
    action = current.action
    if previous.action and previous.action != current.action:
        action = f"{previous.action}+{current.action}"
    return FieldResult(
        current.text,
        previous.length_trimmed or current.length_trimmed,
        previous.ending_trimmed or current.ending_trimmed,
        previous.unresolved or current.unresolved,
        action,
    )


def is_valid_boundary(text: str, match: re.Match[str]) -> bool:
    end = match.end()
    token = match.group(0)
    stripped_token = token.strip()
    if stripped_token.startswith(";"):
        return True
    punctuation_index = end - len(token) + next(
        (idx for idx, char in enumerate(token) if char in ".!?\u3002\uff01\uff1f\u2026;"),
        len(token) - 1,
    )
    punctuation = text[punctuation_index] if 0 <= punctuation_index < len(text) else ""
    if punctuation != ".":
        return True
    prev_char = text[punctuation_index - 1] if punctuation_index > 0 else ""
    next_char = text[punctuation_index + 1] if punctuation_index + 1 < len(text) else ""
    if prev_char.isdigit() and next_char.isdigit():
        return False
    prefix = text[:punctuation_index]
    word_match = re.search(r"([A-Za-z]{1,6})\s*$", prefix)
    if not word_match:
        return True
    word = word_match.group(1).lower()
    common_abbreviations = {
        "no",
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "st",
        "jr",
        "sr",
        "vs",
        "etc",
        "fig",
        "eq",
        "ref",
        "refs",
        "dept",
        "inc",
        "ltd",
        "corp",
        "co",
    }
    if word in common_abbreviations:
        return False
    return True


def complete_prefix(text: str, domain: str) -> str:
    pattern = MEETING_BOUNDARY_RE if domain == "meeting_summary" else SENTENCE_BOUNDARY_RE
    matches = list(pattern.finditer(text or ""))
    if not matches:
        return ""
    return (text or "")[: matches[-1].end()].strip()


def is_complete_enough(text: str, domain: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    if has_complete_ending(stripped):
        return True
    return domain == "meeting_summary" and stripped.endswith(";")


def new_file_stats(input_path: Path, variant: str, domain: str) -> dict[str, Any]:
    return {
        "path": str(input_path),
        "variant": variant,
        "domain": domain,
        "pairs": 0,
        "length_trimmed_fields": 0,
        "ending_trimmed_fields": 0,
        "unresolved_incomplete_fields": 0,
        "empty_fields": 0,
        "changed_examples": [],
        "unresolved_examples": [],
    }


def update_stats(file_stats: dict[str, Any], result: dict[str, Any], index: int) -> None:
    file_stats["pairs"] += 1
    for field_name in ("original_stats", "sampled_stats"):
        stats = result[field_name]
        if stats["length_trimmed"]:
            file_stats["length_trimmed_fields"] += 1
            add_limited_example(file_stats["changed_examples"], index, field_name.replace("_stats", ""), stats)
        if stats["ending_trimmed"]:
            file_stats["ending_trimmed_fields"] += 1
            add_limited_example(file_stats["changed_examples"], index, field_name.replace("_stats", ""), stats)
        if stats["unresolved_incomplete"]:
            file_stats["unresolved_incomplete_fields"] += 1
            add_limited_example(file_stats["unresolved_examples"], index, field_name.replace("_stats", ""), stats)
        if stats["after_words"] == 0:
            file_stats["empty_fields"] += 1


def add_limited_example(examples: list[dict[str, Any]], index: int, field: str, stats: dict[str, Any]) -> None:
    if len(examples) >= 5:
        return
    examples.append({"index": index, "field": field, **stats})


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def ratio_band(ratio: float) -> str:
    if ratio == 0:
        return "zero"
    if ratio < 0.5:
        return "<0.5"
    if ratio < 0.65:
        return "0.5-0.65"
    if ratio < 0.8:
        return "0.65-0.8"
    if ratio <= 1.25:
        return "0.8-1.25"
    if ratio <= 1.55:
        return "1.25-1.55"
    if ratio <= 2.0:
        return "1.55-2.0"
    return ">2.0"


def print_summary(manifest: dict[str, Any]) -> None:
    summary = manifest["summary"]
    print(
        "[length-matched-benchmark] "
        f"files={summary['files']} pairs={summary['pairs']} "
        f"length_trimmed_fields={summary['length_trimmed_fields']} "
        f"ending_trimmed_fields={summary['ending_trimmed_fields']} "
        f"unresolved_incomplete_fields={summary['unresolved_incomplete_fields']} "
        f"empty_fields={summary['empty_fields']}"
    )
    print(f"[ratio-before] {summary['ratio_bands_before']}")
    print(f"[ratio-after] {summary['ratio_bands_after']}")


if __name__ == "__main__":
    main()

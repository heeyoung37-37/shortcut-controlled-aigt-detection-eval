#!/usr/bin/env python3
"""Clean narrow, audited special-case artifacts in curated shards.

This script intentionally avoids broad style edits. It handles only artifacts
that were found by manual scan after generation/editing:

- prompt-card leakage such as FACT_CARD in generated text
- leading wrapper labels such as "Summary:" in detector-facing text
- long reviewer-response separators in peer review references
- exact adjacent sentence repetition in peer review samples
- incomplete final sentence fragments that can be trimmed safely
"""

from __future__ import annotations

import argparse
import json
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

from scripts.common import read_json, read_jsonl, word_count, write_json, write_jsonl
from scripts.text_postprocess import sentence_safe_postprocess_text, surface_clean_text


LONG_SEPARATOR_RE = re.compile(r"\s*[-=]{8,}\s*")
SUMMARY_PREFIX_RE = re.compile(r"^\s*(?:summary|summary\.)\s*:\s*", re.IGNORECASE)
PROMPT_LEAK_RE = re.compile(
    r"^\s*The review comments are based on the provided FACT_CARD only\.\s*",
    re.IGNORECASE,
)
FACT_TAG_RE = re.compile(r"\s*\(Fact\s+\d+\)")
SENTENCE_RE = re.compile(r"[^.!?\u3002\uff01\uff1f\u2026]+[.!?\u3002\uff01\uff1f\u2026]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curated-root", type=Path, default=PROJECT_ROOT / "data" / "curated_shards" / "final")
    parser.add_argument("--human-export-root", type=Path, default=PROJECT_ROOT / "human_edit_input")
    parser.add_argument("--manifest-path", type=Path, default=PROJECT_ROOT / "data" / "manifests" / "special_case_postprocess_manifest.json")
    parser.add_argument("--backup-root", type=Path, default=PROJECT_ROOT / "runs" / "special_case_postprocess_backups")
    parser.add_argument("--write", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": not args.write,
        "curated_root": str(args.curated_root),
        "human_export_root": str(args.human_export_root),
        "files": [],
        "summary": Counter(),
    }

    for path in sorted(args.curated_root.rglob("*.jsonl")):
        rows = read_jsonl(path)
        changed, changes = clean_curated_rows(rows)
        if changed:
            manifest["files"].append({"path": str(path), "changes": changes})
            for change in changes:
                manifest["summary"][change["action"]] += 1
            if args.write:
                backup_file(path, args)
                write_jsonl(path, rows)

    for path in sorted(args.human_export_root.rglob("*.json")):
        payload = read_json(path)
        changed, changes = clean_fdg_payload(payload, domain=infer_domain_from_path(path, args.human_export_root))
        if changed:
            manifest["files"].append({"path": str(path), "changes": changes})
            for change in changes:
                manifest["summary"][change["action"]] += 1
            if args.write:
                backup_file(path, args)
                write_json(path, payload)

    manifest["summary"] = dict(manifest["summary"])
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))
    if args.write:
        write_json(args.manifest_path, manifest)


def backup_file(path: Path, args: argparse.Namespace) -> None:
    relative = path.relative_to(PROJECT_ROOT)
    backup_path = args.backup_root / relative
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    if not backup_path.exists():
        shutil.copy2(path, backup_path)


def clean_curated_rows(rows: list[dict[str, Any]]) -> tuple[bool, list[dict[str, Any]]]:
    changed = False
    changes: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        domain = row.get("domain", "")
        variant = row.get("variant_type", "")
        row_changes = []
        for field in ("original", "sampled"):
            old = str(row.get(field) or "")
            new, field_changes = clean_text_field(old, domain=domain, variant=variant, field=field)
            if new != old:
                row[field] = new
                changed = True
                row_changes.extend({"field": field, **item} for item in field_changes)
        if row_changes:
            metadata = row.setdefault("metadata", {})
            metadata.setdefault("special_case_postprocess", []).extend(row_changes)
            changes.extend({"index": index, "id": row.get("id"), **item} for item in row_changes)
    return changed, changes


def clean_fdg_payload(payload: dict[str, Any], *, domain: str) -> tuple[bool, list[dict[str, Any]]]:
    changed = False
    changes: list[dict[str, Any]] = []
    for field in ("original", "sampled"):
        values = payload.get(field)
        if not isinstance(values, list):
            continue
        for index, value in enumerate(values):
            old = str(value or "")
            new, field_changes = clean_text_field(old, domain=domain, variant="", field=field)
            if new != old:
                values[index] = new
                changed = True
                changes.extend({"index": index, "field": field, **item} for item in field_changes)
    return changed, changes


def infer_domain_from_path(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return ""
    return relative.parts[0] if relative.parts else ""


def clean_text_field(text: str, *, domain: str, variant: str, field: str) -> tuple[str, list[dict[str, Any]]]:
    current = surface_clean_text(text)
    changes: list[dict[str, Any]] = []

    had_prompt_leak = bool(PROMPT_LEAK_RE.search(current))
    cleaned = PROMPT_LEAK_RE.sub("", current)
    if cleaned != current:
        changes.append(change_record("remove_prompt_leak", current, cleaned))
        current = cleaned

    cleaned = FACT_TAG_RE.sub("", current)
    if (had_prompt_leak or "FACT_CARD" in current.upper()) and cleaned != current:
        changes.append(change_record("remove_fact_tags", current, cleaned))
        current = cleaned

    cleaned = SUMMARY_PREFIX_RE.sub("", current)
    if cleaned != current:
        changes.append(change_record("remove_wrapper_prefix", current, cleaned))
        current = cleaned

    if domain == "peer_review" and LONG_SEPARATOR_RE.search(current):
        cleaned = normalize_peer_review_separators(current)
        if cleaned != current:
            changes.append(change_record("normalize_long_separators", current, cleaned))
            current = cleaned

        cleaned = remove_adjacent_duplicate_sentences(current)
        if cleaned != current:
            changes.append(change_record("remove_adjacent_duplicate_sentence", current, cleaned))
            current = cleaned

    result = sentence_safe_postprocess_text(current, min_words=20, max_loss_ratio=0.65)
    if result.text != current and result.action == "trimmed":
        changes.append(change_record("trim_incomplete_final_sentence", current, result.text))
        current = result.text

    cleaned = repair_remaining_incomplete_suffix(current, domain=domain)
    if cleaned != current:
        changes.append(change_record("repair_incomplete_suffix", current, cleaned))
        current = cleaned

    return current.strip(), changes


def normalize_peer_review_separators(text: str) -> str:
    if not LONG_SEPARATOR_RE.search(text):
        return text
    cleaned = LONG_SEPARATOR_RE.sub(". ", text)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"\.{2,}", ".", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def remove_adjacent_duplicate_sentences(text: str) -> str:
    matches = list(SENTENCE_RE.finditer(text))
    if len(matches) < 2:
        return text
    output: list[str] = []
    cursor = 0
    previous_norm = ""
    changed = False
    for match in matches:
        output.append(text[cursor : match.start()])
        sentence = match.group(0)
        norm = normalize_sentence(sentence)
        if norm and norm == previous_norm:
            changed = True
        else:
            output.append(sentence)
            previous_norm = norm
        cursor = match.end()
    output.append(text[cursor:])
    if not changed:
        return text
    cleaned = "".join(output)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def repair_remaining_incomplete_suffix(text: str, *, domain: str) -> str:
    stripped = (text or "").strip()
    if not stripped:
        return stripped
    if domain == "meeting_summary" and stripped.endswith(";"):
        return stripped
    if re.search(r"[.!?\u3002\uff01\uff1f\u2026][\]\})'\"\u201d\u2019\u00bb]*$", stripped):
        return stripped

    if domain == "meeting_summary":
        semicolon = stripped.rfind(";")
        suffix = stripped[semicolon + 1 :].strip() if semicolon >= 0 else stripped
        suffix_words = suffix.split()
        last_word = suffix_words[-1].strip(".,;:!?()[]{}\"'“”‘’") if suffix_words else ""
        looks_cut = (
            suffix.lower() in {"and in", "and"}
            or bool(re.match(r"^\d+\)", suffix))
            or (last_word.isalpha() and len(last_word) <= 3)
            or last_word.lower() in {"cou", "inform", "pollutio", "provi"}
        )
        if semicolon >= 0 and looks_cut:
            return stripped[: semicolon + 1].strip()
        if 1 <= len(suffix_words) <= 12:
            return stripped + "."

    if domain == "peer_review":
        semicolon = stripped.rfind(";")
        if semicolon > 0:
            prefix = stripped[: semicolon + 1].strip()
            prefix_words = word_count(prefix)
            loss_ratio = (word_count(stripped) - prefix_words) / max(word_count(stripped), 1)
            if prefix_words >= 20 and loss_ratio <= 0.75:
                return prefix.rstrip(";").rstrip() + "."

    if domain == "blog" and word_count(stripped) >= 80:
        tail = stripped.split()[-1].strip("\"'”“‘’()[]{}")
        if tail and not tail.endswith("'s"):
            return stripped + "."

    return stripped


def normalize_sentence(sentence: str) -> str:
    return re.sub(r"\W+", " ", sentence).strip().lower()


def change_record(action: str, before: str, after: str) -> dict[str, Any]:
    return {
        "action": action,
        "before_words": word_count(before),
        "after_words": word_count(after),
    }


if __name__ == "__main__":
    main()

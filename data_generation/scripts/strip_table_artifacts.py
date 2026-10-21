#!/usr/bin/env python3
"""Remove table-like artifacts from curated benchmark shards."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import read_jsonl, write_json, write_jsonl, word_count
from scripts.text_postprocess import has_complete_ending


TABLE_MARKER_RE = re.compile(
    r"(?:-{8,}\s*)?Model\s*\|\s*RNN-equiv\s*\|\s*train\s*/\s*valid\s*/\s*test",
    re.IGNORECASE,
)
TABLE_INTRO_RE = re.compile(
    r"(?:The\s+results\s+are\s+as\s+follows?\.|"
    r"The\s+results\s+are\s+as\s+follow\.|"
    r"As\s+in\s+the\s+paper,\s+the\s+second\s+column\s+provides[^.]*\.)",
    re.IGNORECASE,
)
GENERIC_TABLE_LINE_RE = re.compile(r"(?:-{8,}|(?:\b[A-Za-z_]+\d*\b\s*\|\s*){2,})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "data" / "curated_shards" / "final")
    parser.add_argument("--backup-root", type=Path, default=PROJECT_ROOT / "runs" / "table_artifact_backups")
    parser.add_argument("--write", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = strip_table_artifacts(args.root, args.backup_root, write=args.write)
    print_summary(summary)
    if args.write:
        write_json(PROJECT_ROOT / "data" / "manifests" / "table_artifact_postprocess_manifest.json", summary)


def strip_table_artifacts(root: Path, backup_root: Path, *, write: bool) -> dict[str, Any]:
    backup_dir = backup_root / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    summary: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "dry_run": not write,
        "changed_files": 0,
        "changed_rows": 0,
        "changed_fields": 0,
        "changes": [],
    }
    for path in sorted(root.glob("*/*/*/*_shard_*.jsonl")):
        rows = read_jsonl(path)
        changed_in_file = 0
        for row in rows:
            changed_fields = []
            for field in ("original", "sampled"):
                old_text = str(row.get(field) or "")
                new_text, action = strip_table_text(old_text)
                if new_text != old_text:
                    row[field] = new_text
                    changed_fields.append(
                        {
                            "field": field,
                            "action": action,
                            "before_words": word_count(old_text),
                            "after_words": word_count(new_text),
                        }
                    )
            if changed_fields:
                changed_in_file += 1
                metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                metadata = dict(metadata)
                metadata["table_artifact_postprocess"] = {
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                    "changes": changed_fields,
                }
                row["metadata"] = metadata
                summary["changed_rows"] += 1
                summary["changed_fields"] += len(changed_fields)
                summary["changes"].append(
                    {
                        "path": str(path),
                        "id": row.get("id"),
                        "domain": row.get("domain"),
                        "model_label": row.get("model_label"),
                        "variant_type": row.get("variant_type"),
                        "fields": changed_fields,
                    }
                )
        if changed_in_file:
            summary["changed_files"] += 1
            if write:
                backup_path = backup_dir / path.relative_to(root)
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, backup_path)
                write_jsonl(path, rows)
    return summary


def strip_table_text(text: str) -> tuple[str, str]:
    if not text:
        return "", "empty"
    marker = TABLE_MARKER_RE.search(text)
    if not marker:
        return text, "unchanged"
    start = marker.start()
    intro_start = find_intro_start(text, start)
    if intro_start is not None:
        start = intro_start
    cleaned = text[:start].strip()
    cleaned = cleanup_dangling_table_intro(cleaned)
    if cleaned and not has_complete_ending(cleaned):
        cleaned = trim_to_last_complete_sentence(cleaned)
    return cleaned.strip(), "table_removed"


def find_intro_start(text: str, marker_start: int) -> int | None:
    window_start = max(0, marker_start - 600)
    prefix_window = text[window_start:marker_start]
    matches = list(TABLE_INTRO_RE.finditer(prefix_window))
    if matches:
        return window_start + matches[0].start()
    return None


def cleanup_dangling_table_intro(text: str) -> str:
    cleaned = re.sub(
        r"\s*(?:Between this and all of the reviewers[^.]*?we re-ran experiments[^.]*\.)\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    cleaned = re.sub(
        r"\s*(?:The results are as follows?|The results are as follow)\.?\s*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    return cleaned


def trim_to_last_complete_sentence(text: str) -> str:
    matches = list(re.finditer(r"[.!?\u3002\uff01\uff1f\u2026]+[\]\})'\"\u201d\u2019\u00bb]*", text))
    if not matches:
        return text.strip()
    return text[: matches[-1].end()].strip()


def print_summary(summary: dict[str, Any]) -> None:
    print(
        "[table-artifact-strip] "
        f"dry_run={summary['dry_run']} changed_files={summary['changed_files']} "
        f"changed_rows={summary['changed_rows']} changed_fields={summary['changed_fields']}"
    )
    for change in summary["changes"][:20]:
        print(f"[change] {change}")


if __name__ == "__main__":
    main()

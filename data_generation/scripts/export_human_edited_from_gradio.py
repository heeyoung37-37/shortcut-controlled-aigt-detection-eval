#!/usr/bin/env python3
"""Export completed Gradio human edits into Benchmark_length_matched/human_edited."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import read_json, safe_filename_component
from scripts.human_edit_gradio import all_input_files, load_fdg_payload, parse_domain_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=PROJECT_ROOT / "human_edited_gradio_saved")
    parser.add_argument("--reference-root", type=Path, default=PROJECT_ROOT / "human_edit_input")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "Benchmark_length_matched" / "human_edited")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Export even when progress manifests do not mark every row as saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_files = all_input_files(args.reference_root.resolve())
    editor_manifests = load_editor_manifests(args.input_root.resolve())
    errors: list[str] = []
    written: list[str] = []
    expected_outputs: set[Path] = set()

    for index, reference_path in enumerate(reference_files):
        editor_id = "1" if index % 2 == 0 else "2"
        domain, model_label = parse_domain_model(args.reference_root.resolve(), reference_path)
        output_filename = (
            f"{safe_filename_component(domain)}_"
            f"{safe_filename_component(model_label)}_human_edited.json"
        )
        editor_output = args.input_root.resolve() / f"editor_{editor_id}" / domain / output_filename
        final_output = args.output_root.resolve() / domain / output_filename
        expected_outputs.add(editor_output)
        try:
            reference_payload = load_fdg_payload(reference_path)
            edited_payload = load_fdg_payload(editor_output)
            validate_payload_pair(reference_path, editor_output, reference_payload, edited_payload)
            validate_progress(
                editor_manifests,
                editor_id=editor_id,
                reference_root=args.reference_root.resolve(),
                reference_path=reference_path,
                allow_incomplete=args.allow_incomplete,
            )
            if not args.dry_run:
                write_json_atomic(final_output, edited_payload)
            written.append(str(final_output))
        except Exception as exc:
            errors.append(str(exc))

    extra_outputs = [
        path for path in args.input_root.resolve().glob("editor_*/*/*.json")
        if path not in expected_outputs
    ]
    if extra_outputs:
        errors.extend(f"Unexpected editor output file: {path}" for path in extra_outputs)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "allow_incomplete": args.allow_incomplete,
        "reference_root": str(args.reference_root.resolve()),
        "input_root": str(args.input_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "file_count": len(written),
        "pair_count": len(written) * 10,
        "files": written,
    }
    if errors:
        print("[human-edited-export] failed")
        for error in errors[:50]:
            print(f"- {error}")
        raise SystemExit(1)
    if not args.dry_run:
        write_json_atomic(args.output_root.resolve() / "manifest.json", manifest)
    print(f"[human-edited-export] files={len(written)} pairs={len(written) * 10} -> {args.output_root}")
    if args.dry_run:
        print("[human-edited-export] dry-run only")


def load_editor_manifests(input_root: Path) -> dict[str, dict[str, Any]]:
    manifests: dict[str, dict[str, Any]] = {}
    for editor_id in ("1", "2"):
        path = input_root / f"editor_{editor_id}" / "progress_manifest.json"
        if path.exists():
            payload = read_json(path)
            if isinstance(payload, dict):
                manifests[editor_id] = payload
    return manifests


def validate_payload_pair(
    reference_path: Path,
    edited_path: Path,
    reference_payload: dict[str, list[str]],
    edited_payload: dict[str, list[str]],
) -> None:
    if len(reference_payload["original"]) != 10 or len(reference_payload["sampled"]) != 10:
        raise ValueError(f"{reference_path} must have 10 original/sample rows")
    if len(edited_payload["original"]) != 10 or len(edited_payload["sampled"]) != 10:
        raise ValueError(f"{edited_path} must have 10 original/sample rows")
    if edited_payload["original"] != reference_payload["original"]:
        raise ValueError(f"{edited_path} original differs from reference {reference_path}")
    empty = [idx for idx, text in enumerate(edited_payload["sampled"]) if not str(text).strip()]
    if empty:
        raise ValueError(f"{edited_path} has empty sampled rows: {empty}")


def validate_progress(
    manifests: dict[str, dict[str, Any]],
    *,
    editor_id: str,
    reference_root: Path,
    reference_path: Path,
    allow_incomplete: bool,
) -> None:
    if allow_incomplete:
        return
    manifest = manifests.get(editor_id)
    if not manifest:
        raise ValueError(f"Missing progress manifest for editor_{editor_id}")
    rel_input = str(reference_path.relative_to(reference_root))
    matches = [
        item for item in manifest.get("assignments", [])
        if isinstance(item, dict) and item.get("input_file") == rel_input
    ]
    if not matches:
        raise ValueError(f"Missing progress entry for editor_{editor_id}:{rel_input}")
    done = matches[0].get("done")
    if not isinstance(done, list) or len(done) != 10 or not all(bool(item) for item in done):
        saved_count = sum(bool(item) for item in done) if isinstance(done, list) else 0
        raise ValueError(f"Incomplete human edits for editor_{editor_id}:{rel_input} saved={saved_count}/10")


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_name, path)
    finally:
        tmp_path = Path(tmp_name)
        if tmp_path.exists():
            tmp_path.unlink()


if __name__ == "__main__":
    main()

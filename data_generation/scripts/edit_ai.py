#!/usr/bin/env python3
"""Create AI-edited output shards from ai_edit_input shards."""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import (
    iter_shard_files,
    output_shard_dir,
    read_jsonl,
    shard_file_name,
    validate_output_row,
    word_count,
    write_jsonl,
)
from scripts.generate_ultimate_controlled_fdg_dataset import (
    DEFAULT_DEEPSEEK_API_BASE,
    DEFAULT_QWEN_API_BASE,
    DEFAULT_QWEN_CHAT_TEMPLATE_KWARGS,
    PROVIDER_DEFAULT_MODELS,
    build_provider,
    json_object_or_none,
    model_label_for,
    normalize_model_for_provider,
)
from scripts.text_postprocess import flatten_product_review_text
from scripts.ultimate_controlled.prompts import EDIT_PROMPT_VERSION, edit_messages


WRAPPER_RE = re.compile(
    r"^\s*(?:here(?:'s| is)\s+(?:the\s+)?(?:revised|edited)\s+(?:text|version)|"
    r"revised\s+text|edited\s+text|revision)\s*:\s*",
    re.IGNORECASE,
)
MARKDOWN_FENCE_RE = re.compile(r"^\s*```(?:text|markdown)?\s*|\s*```\s*$", re.IGNORECASE)
MIN_WORD_RATIO = 0.65
MAX_WORD_RATIO = 1.35


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="final")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--provider", choices=["gemini", "claude", "qwen", "openai", "deepseek"], required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-label", default=None)
    parser.add_argument("--input-root", type=Path, default=PROJECT_ROOT / "data" / "curated_shards")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "curated_shards")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-base", default=DEFAULT_QWEN_API_BASE)
    parser.add_argument("--openai-api-base", default=None)
    parser.add_argument("--openai-reasoning-effort", default=None)
    parser.add_argument("--openai-verbosity", default=None)
    parser.add_argument("--deepseek-api-base", default=DEFAULT_DEEPSEEK_API_BASE)
    parser.add_argument("--deepseek-thinking", choices=["enabled", "disabled", "none"], default="disabled")
    parser.add_argument("--deepseek-reasoning-effort", default=None)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--chat-template-kwargs", type=json_object_or_none, default=DEFAULT_QWEN_CHAT_TEMPLATE_KWARGS)
    parser.add_argument("--skip-server-check", action="store_true")
    parser.add_argument("--gemini-thinking-budget", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument(
        "--force-row-ids",
        nargs="*",
        default=[],
        help="Re-edit only these row ids, accepting either ai_edit_input ids or ai_edited ids.",
    )
    parser.add_argument(
        "--length-guard",
        action="store_true",
        help="Add a retry instruction that strongly preserves the source AI text length and details.",
    )
    parser.add_argument("--mock", action="store_true", help="Use deterministic local edits for tests.")
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> None:
    model = normalize_model_for_provider(args.provider, args.model or PROVIDER_DEFAULT_MODELS[args.provider])
    model_label = model_label_for(args.provider, model, args.model_label)
    provider = None if args.mock else await build_provider(args, model)
    input_dir = output_shard_dir(args.input_root, args.run_id, args.domain, model_label, "ai_edit_input")
    output_dir = output_shard_dir(args.output_root, args.run_id, args.domain, model_label, "ai_edited")
    failures = []
    force_ids = normalize_force_row_ids(args.force_row_ids)
    processed_force_ids: set[str] = set()
    for shard_index, shard_path in enumerate(iter_shard_files(input_dir)):
        input_rows = read_jsonl(shard_path)
        output_path = output_dir / shard_file_name(args.domain, model_label, "ai_edited", shard_index)
        existing_rows = read_jsonl(output_path)
        if not force_ids and output_is_complete(existing_rows, input_rows):
            print(f"[ai-edit-skip] {args.domain}/{model_label}: {len(existing_rows)} rows -> {output_path}")
            continue
        existing_by_input_id = {row.get("source_input_id"): row for row in existing_rows if row.get("source_input_id")}
        edited_rows = []
        shard_start = time.monotonic()
        total_rows = len(input_rows)
        for idx, row in enumerate(input_rows, 1):
            existing = None
            try:
                existing = existing_by_input_id.get(row["id"])
                force = row["id"] in force_ids
                if existing and edited_row_is_valid(existing, row) and not force:
                    edited_rows.append(existing)
                    print_progress(args.domain, model_label, idx, total_rows, shard_start, "reuse", row.get("id"))
                    continue
                if force:
                    processed_force_ids.add(row["id"])
                edited_text = await edit_row(provider, args, row)
                edited = make_edited_row(row, edited_text, args=args, model=model)
                validate_output_row(edited)
                edited_rows.append(edited)
                print_progress(args.domain, model_label, idx, total_rows, shard_start, "force-edit" if force else "edit", row.get("id"))
            except Exception as exc:
                failures.append({"id": row.get("id"), "error": str(exc)})
                print(f"[edit-failed] {row.get('id')} {exc}", file=sys.stderr)
                if existing and edited_row_is_valid(existing, row):
                    edited_rows.append(existing)
                    print_progress(args.domain, model_label, idx, total_rows, shard_start, "keep-after-fail", row.get("id"))
        write_jsonl(output_path, edited_rows)
        print(f"[ai-edited] {args.domain}/{model_label}: {len(edited_rows)} rows -> {output_path}")
    if failures:
        write_jsonl(output_dir / "failures.jsonl", failures)
        raise SystemExit(1)
    missing_force_ids = force_ids - processed_force_ids
    if missing_force_ids:
        raise SystemExit(f"Requested force-row-ids were not found: {sorted(missing_force_ids)}")


async def edit_row(provider, args: argparse.Namespace, row: dict) -> str:
    if args.mock:
        return mock_edit_text(str(row.get("sampled", "")))
    messages = edit_messages(args.domain, row["original"], row["sampled"])
    if args.length_guard:
        messages = add_length_guard(messages, row)
    response = await provider.generate_text(
        messages,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        thinking_budget=args.gemini_thinking_budget if args.provider == "gemini" else None,
    )
    text = response.text.strip()
    if not text:
        raise ValueError("AI edit produced empty text")
    return sanitize_edited_text(args.domain, text)


def make_edited_row(source: dict, edited_text: str, *, args: argparse.Namespace, model: str) -> dict:
    edited_text = sanitize_edited_text(args.domain, edited_text)
    edited = dict(source)
    edited["id"] = source["id"].replace(":ai_edit_input:", ":ai_edited:")
    edited["variant_type"] = "ai_edited"
    edited["sampled"] = edited_text
    edited["source_input_id"] = source["id"]
    edited["metadata"] = dict(source.get("metadata", {}))
    edited["metadata"]["ai_edit"] = {
        "edited_at": datetime.now(timezone.utc).isoformat(),
        "provider": args.provider if not args.mock else "mock",
        "model": model if not args.mock else "mock",
        "source_variant": "ai_edit_input",
        "prompt_version": EDIT_PROMPT_VERSION,
        "length_guard": bool(getattr(args, "length_guard", False)),
        "review_flags": review_flags(source, edited_text, args.domain),
    }
    if edited["original"] != source["original"]:
        raise ValueError("AI edit changed original field")
    validate_edited_text(args.domain, edited_text)
    return edited


def sanitize_edited_text(domain: str, text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = MARKDOWN_FENCE_RE.sub("", cleaned).strip()
    cleaned = WRAPPER_RE.sub("", cleaned).strip()
    if domain == "product_review":
        cleaned = flatten_product_review_text(cleaned)
    return cleaned


def validate_edited_text(domain: str, text: str) -> None:
    if not text.strip():
        raise ValueError("AI edit produced empty text")
    if "```" in text:
        raise ValueError("AI edit contains markdown fence")
    if WRAPPER_RE.search(text):
        raise ValueError("AI edit contains wrapper label")
    if domain == "product_review" and "\n" in text.strip():
        raise ValueError("product_review AI edit must be one paragraph")


def review_flags(source: dict, edited_text: str, domain: str) -> list[str]:
    flags = []
    source_words = word_count(str(source.get("sampled") or ""))
    edited_words = word_count(edited_text)
    if source_words:
        ratio = edited_words / source_words
        if ratio < MIN_WORD_RATIO:
            flags.append(f"word_count_ratio_low:{ratio:.2f}")
        elif ratio > MAX_WORD_RATIO:
            flags.append(f"word_count_ratio_high:{ratio:.2f}")
    if domain == "product_review" and "\n" in edited_text.strip():
        flags.append("product_review_multi_paragraph")
    return flags


def output_is_complete(existing_rows: list[dict], input_rows: list[dict]) -> bool:
    if len(existing_rows) != len(input_rows) or not input_rows:
        return False
    existing_by_input_id = {row.get("source_input_id"): row for row in existing_rows if row.get("source_input_id")}
    if len(existing_by_input_id) != len(existing_rows):
        return False
    return all(edited_row_is_valid(existing_by_input_id.get(source.get("id"), {}), source) for source in input_rows)


def edited_row_is_valid(edited: dict, source: dict) -> bool:
    try:
        validate_output_row(edited)
        expected_id = source["id"].replace(":ai_edit_input:", ":ai_edited:")
        if edited.get("id") != expected_id:
            return False
        if edited.get("source_input_id") != source.get("id"):
            return False
        if edited.get("variant_type") != "ai_edited":
            return False
        if edited.get("original") != source.get("original"):
            return False
        if not str(edited.get("sampled") or "").strip():
            return False
        validate_edited_text(str(source.get("domain") or edited.get("domain") or ""), str(edited.get("sampled") or ""))
        metadata = edited.get("metadata") if isinstance(edited.get("metadata"), dict) else {}
        ai_edit = metadata.get("ai_edit") if isinstance(metadata.get("ai_edit"), dict) else {}
        return (
            ai_edit.get("source_variant") == "ai_edit_input"
            and bool(ai_edit.get("provider"))
            and bool(ai_edit.get("model"))
            and bool(ai_edit.get("prompt_version"))
            and bool(ai_edit.get("edited_at"))
        )
    except Exception:
        return False


def mock_edit_text(text: str) -> str:
    cleaned = " ".join((text or "").split())
    replacements = {
        "Additionally,": "Also,",
        "Furthermore,": "Also,",
        "Overall,": "In the end,",
        "It is important to note that": "Notably,",
    }
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    if not cleaned:
        raise ValueError("Mock edit received empty text")
    return cleaned


def normalize_force_row_ids(row_ids: list[str]) -> set[str]:
    normalized = set()
    for row_id in row_ids or []:
        cleaned = str(row_id).strip()
        if not cleaned:
            continue
        normalized.add(cleaned.replace(":ai_edited:", ":ai_edit_input:"))
    return normalized


def add_length_guard(messages: list[dict], row: dict) -> list[dict]:
    source_words = word_count(str(row.get("sampled") or ""))
    retry_note = (
        "\n\nLength-preserving retry instruction:\n"
        f"- The AI-written text has about {source_words} words.\n"
        "- Hard requirement: this is a copyedit, not a summary.\n"
        "- Do not summarize, compress, omit examples, merge away supporting details, or shorten lists of facts.\n"
        "- Keep the revised text close to the original AI-written text length, ideally within 90%-110% and never below 75%.\n"
        "- Preserve roughly the same number of sentences and all substantive examples/details.\n"
        "- Make local copyedits and phrasing improvements while preserving all substantive content.\n"
        "- Preserve the existing paragraph count unless the domain-specific instruction requires one paragraph.\n"
        "- Return revised text only."
    )
    guarded = [dict(message) for message in messages]
    guarded[-1]["content"] = str(guarded[-1]["content"]) + retry_note
    return guarded


def print_progress(
    domain: str,
    model_label: str,
    completed: int,
    total: int,
    start_time: float,
    status: str,
    row_id: str | None,
) -> None:
    elapsed = max(time.monotonic() - start_time, 0.001)
    rate = completed / elapsed
    remaining = max(total - completed, 0)
    eta = remaining / rate if rate else 0.0
    percent = (completed / total * 100.0) if total else 100.0
    print(
        f"[ai-edit-progress] {domain}/{model_label} {completed}/{total} "
        f"({percent:.1f}%) status={status} elapsed={format_duration(elapsed)} "
        f"eta={format_duration(eta)} id={row_id}",
        flush=True,
    )


def format_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def main() -> None:
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()

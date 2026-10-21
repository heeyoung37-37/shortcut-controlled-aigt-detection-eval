#!/usr/bin/env python3
"""Normalize downloaded raw rows into the common Our_data schema."""

from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import (
    as_native,
    clean_text,
    coerce_list,
    load_domain_specs,
    read_jsonl,
    stable_id,
    word_count,
    write_json,
    write_jsonl,
)


BLOG_CODE_RE = re.compile(
    r"""
    (^|\n)\s*(?:import|from|pip|npm|yarn|git|cd|mkdir|sudo|docker|kubectl|helm|conda|python|public|private|const|let|var|def|class|return|function)\b
    |
    ```|</?[A-Za-z][^>]*>
    |
    \$\s+\w+
    |
    [;{}]{3,}
    """,
    re.IGNORECASE | re.VERBOSE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains", nargs="+", default=None)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "datasets.yaml")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--normalized-dir", type=Path, default=PROJECT_ROOT / "data" / "normalized")
    parser.add_argument("--original-dir", type=Path, default=PROJECT_ROOT / "data" / "original")
    parser.add_argument("--manifest-dir", type=Path, default=PROJECT_ROOT / "data" / "manifests")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = load_domain_specs(args.config)
    domains = args.domains or list(specs)
    manifest = {"domains": {}}
    for domain in domains:
        if domain not in specs:
            raise ValueError(f"Unknown domain: {domain}")
        rows = normalize_domain(domain, specs[domain], args.raw_dir / f"{domain}.jsonl", limit=args.limit)
        out_path = args.normalized_dir / domain / "records.jsonl"
        write_jsonl(out_path, rows)
        write_json(args.original_dir / domain / f"{domain}.original.json", {"original": [row["human_text"] for row in rows]})
        manifest["domains"][domain] = {
            "source_dataset": specs[domain].get("source_dataset"),
            "normalized_rows": len(rows),
            "path": str(out_path),
        }
        print(f"[normalized] {domain}: {len(rows)} rows -> {out_path}")
    write_json(args.manifest_dir / "normalization_manifest.json", manifest)


def normalize_domain(domain: str, spec: dict[str, Any], raw_path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw file for {domain}: {raw_path}")
    raw_rows = read_jsonl(raw_path)
    min_words = int(spec.get("min_words") or 1)
    seen_texts: set[str] = set()
    normalized = []
    for idx, raw in enumerate(raw_rows):
        row = as_native(raw)
        human_text = clean_text(extract_text(row, spec.get("text_field")))
        if domain == "peer_review":
            human_text = normalize_peer_review_text(human_text)
        elif domain == "blog":
            human_text = normalize_blog_text(human_text)
            if not is_usable_blog_text(human_text):
                continue
        elif domain == "product_review":
            human_text = normalize_product_review_text(human_text)
            if not is_usable_product_review_text(human_text):
                continue
        elif domain == "meeting_summary":
            human_text = normalize_meeting_summary_text(human_text)
        elif domain == "translation":
            human_text = normalize_translation_text(human_text)
            if not is_usable_translation_row(row, spec, human_text):
                continue
        if word_count(human_text) < min_words:
            continue
        text_key = stable_id(human_text, length=24)
        if text_key in seen_texts:
            continue
        seen_texts.add(text_key)
        if domain == "meeting_summary":
            task_input = build_meeting_summary_task_input(row, spec)
            if not task_input:
                continue
        elif domain == "translation":
            task_input = build_translation_task_input(row, spec)
            if not task_input:
                continue
        else:
            task_input = build_task_input(row, coerce_list(spec.get("task_fields")))
        if domain == "peer_review":
            task_input = normalize_peer_review_text(task_input)
        metadata = build_metadata(row, coerce_list(spec.get("metadata_fields")))
        if domain == "meeting_summary":
            transcript = normalize_meeting_transcript(row.get(str(spec.get("transcript_field") or "transcript")))
            metadata["transcript_word_count"] = word_count(transcript)
            metadata["transcript_excerpt_word_count"] = word_count(
                truncate_words(transcript, int(spec.get("max_transcript_words") or 1800))
            )
        elif domain == "translation":
            language_config = translation_language_config(row)
            language_names = spec.get("language_names") or {}
            source_text = normalize_translation_text(row.get(str(spec.get("source_text_field") or "og_full_text")))
            metadata["language_config"] = language_config
            metadata["language_name"] = str(language_names.get(language_config, language_config))
            metadata["source_word_count"] = word_count(source_text)
            metadata["translated_word_count"] = word_count(human_text)
        source_id = f"{domain}_{stable_id(domain, spec.get('source_dataset'), idx, metadata.get('id'), human_text, length=18)}"
        fact_source_text = task_input if domain in {"meeting_summary", "translation"} else build_fact_source_text(task_input, human_text)
        normalized.append(
            {
                "source_id": source_id,
                "domain": domain,
                "source_dataset": spec.get("source_dataset"),
                "split": spec.get("split"),
                "task_input": task_input,
                "human_text": human_text,
                "fact_source_text": fact_source_text,
                "metadata": metadata,
            }
        )
        if limit and len(normalized) >= limit:
            break
    return normalized


def extract_text(row: dict[str, Any], text_field: Any) -> str:
    field = str(text_field or "")
    direct = row.get(field)
    if direct:
        if isinstance(direct, (list, tuple)):
            chunks = []
            for item in direct:
                value = item.get("comments") or item.get("text") if isinstance(item, dict) else item
                text = clean_text(value)
                if text:
                    chunks.append(text)
            return "\n\n".join(chunks)
        if isinstance(direct, dict):
            nested_comments = direct.get("comments") or direct.get("text")
            if nested_comments:
                return clean_text(nested_comments)
        return clean_text(direct)
    if field == "comments":
        comments = []
        reviews = row.get("reviews")
        review_items = list(reviews.values()) if isinstance(reviews, dict) else coerce_list(reviews)
        for review in review_items:
            if isinstance(review, dict):
                comments.append(clean_text(review.get("comments")))
        if comments:
            return "\n\n".join(comment for comment in comments if comment)
    return ""


def build_task_input(row: dict[str, Any], fields: list[Any]) -> str:
    lines = []
    for field in fields:
        key = str(field)
        value = row.get(key)
        if value in (None, "", []):
            continue
        lines.append(f"{key}: {format_value(value)}")
    return "\n".join(lines).strip()


def build_metadata(row: dict[str, Any], fields: list[Any]) -> dict[str, Any]:
    metadata = {}
    for field in fields:
        key = str(field)
        if key in row:
            metadata[key] = as_native(row.get(key))
    return metadata


def build_fact_source_text(task_input: str, human_text: str) -> str:
    if not task_input:
        return human_text
    return f"Task/context:\n{task_input}\n\nReference text:\n{human_text}"


def format_value(value: Any) -> str:
    value = as_native(value)
    if isinstance(value, list):
        return ", ".join(clean_text(item) for item in value if clean_text(item))
    if isinstance(value, dict):
        parts = [f"{key}={format_value(val)}" for key, val in value.items() if val not in (None, "", [])]
        return "; ".join(parts)
    return clean_text(value)


def normalize_peer_review_text(text: str) -> str:
    paragraphs = re.split(r"\n\s*\n", clean_text(text))
    normalized = []
    for paragraph in paragraphs:
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if not lines:
            continue
        merged_lines: list[str] = []
        for line in lines:
            if is_peer_review_standalone_line(line):
                merged_lines.append(line)
                continue
            if not merged_lines or is_peer_review_standalone_line(merged_lines[-1]):
                merged_lines.append(line)
            else:
                merged_lines[-1] = f"{merged_lines[-1].rstrip()} {line.lstrip()}"
        normalized.append("\n".join(merged_lines))
    text = "\n\n".join(normalized)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def normalize_blog_text(text: str) -> str:
    paragraphs = split_blank_paragraphs(clean_text(text))
    normalized = []
    for paragraph in paragraphs:
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if not lines:
            continue
        if is_blog_code_block(lines):
            normalized.append("\n".join(lines))
        else:
            normalized.append(" ".join(lines))
    return "\n\n".join(normalized).strip()


def normalize_product_review_text(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"<\s*br\s*/?\s*>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</\s*p\s*>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = clean_text(text)
    paragraphs = []
    for paragraph in split_blank_paragraphs(text):
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        if paragraph:
            paragraphs.append(paragraph)
    return "\n\n".join(paragraphs).strip()


def normalize_meeting_summary_text(text: str) -> str:
    text = clean_text(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_translation_text(text: Any) -> str:
    text = clean_text(text)
    text = re.sub(r"\s+\|\s+", "\n\n", text)
    paragraphs = []
    for paragraph in split_blank_paragraphs(text):
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        if paragraph:
            paragraphs.append(paragraph)
    return "\n\n".join(paragraphs).strip()


def translation_language_config(row: dict[str, Any]) -> str:
    return clean_text(row.get("language_config") or row.get("hf_config") or row.get("og_language"))


def is_usable_translation_row(row: dict[str, Any], spec: dict[str, Any], human_text: str) -> bool:
    if row.get("early_stop"):
        return False
    source_field = str(spec.get("source_text_field") or "og_full_text")
    source_text = normalize_translation_text(row.get(source_field))
    if not source_text:
        return False
    translated_words = word_count(human_text)
    if translated_words < int(spec.get("min_translated_words") or 1):
        return False
    if translated_words > int(spec.get("max_translated_words") or 10_000):
        return False
    paragraphs = split_blank_paragraphs(human_text)
    source_paragraphs = split_blank_paragraphs(source_text)
    if len(paragraphs) > int(spec.get("max_segments") or 4):
        return False
    if len(source_paragraphs) > int(spec.get("max_segments") or 4):
        return False
    short_paragraphs = sum(1 for paragraph in paragraphs if word_count(paragraph) <= 5)
    if paragraphs and short_paragraphs / len(paragraphs) > 0.25:
        return False
    if has_translation_boilerplate(source_text, human_text):
        return False
    if re.search(r"\^[\s\-–_]*\^", source_text + " " + human_text):
        return False
    noisy_patterns = [
        r"\b(cookie|cookies|privacy policy|terms of use|login|sign up|subscribe)\b",
        r"\b(required|e-mail|email address|submit button|click here)\b",
        r"https?://|www\.",
        r"</?[A-Za-z][^>]*>",
    ]
    joined = " ".join(paragraphs + source_paragraphs).lower()
    if sum(1 for pattern in noisy_patterns if re.search(pattern, joined, re.IGNORECASE)) >= 2:
        return False
    language_config = translation_language_config(row)
    if language_config not in set(str(item) for item in spec.get("hf_configs") or []):
        return False
    return True


def has_translation_boilerplate(source_text: str, human_text: str) -> bool:
    combined = f"{source_text}\n{human_text}"
    lower = combined.lower()
    if sum(combined.count(char) for char in "○●ヽﾉ･｀дｪε") >= 4:
        return True
    if re.search(r"[\(\[]?[\^￣][ _\-–]*[\^￣][\)\]]?", combined):
        return True
    if re.search(r"https?://|www\.|\b[\w.-]+\.(?:com|org|net|fr|es|jp|edu|gov)\b", lower):
        return True
    if re.search(r"\b(?:europa press|reuters|associated press|afp|efe)\s*$", lower.strip()):
        return True
    strong_patterns = [
        r"\b(return to (the )?(list|home|top)|back to (the )?(list|home|top))\b",
        r"\b(leave a reply|cancel reply|post comment|publish comment|publicar comentario|cancelar respuesta)\b",
        r"\b(please log in|log in to|sign in to|register to|subscribe to|newsletter)\b",
        r"\b(on our website|our website|our site|from our website)\b",
        r"\b(retour à la liste|connectez-vous|inscrivez-vous|abonnez-vous)\b",
        r"\b(notre site|sur notre site)\b",
        r"\b(volver a la lista|iniciar sesión|regístrate|suscríbete)\b",
        r"\b(nuestra web|nuestro sitio|en nuestro sitio)\b",
        r"\b(total or partial reproduction|reproduction of this article|prior authorization|prior consent)\b",
        r"\b(reproduction totale ou partielle|autorisation préalable|consentement préalable)\b",
        r"\b(reproducción total o parcial|autorización previa|consentimiento previo)\b",
        r"(引用をストック|引用するには|ログインしてください|ログインして|限定公開記事|再度お試しください|ストックできません)",
        r"(当サイト|本サイト)",
        r"(会員登録|続きを読む|コメントを投稿|返信をキャンセル)",
        r"\b(all rights reserved|copyright \d{4}|share this|follow us)\b",
    ]
    if any(re.search(pattern, lower, re.IGNORECASE) for pattern in strong_patterns):
        return True
    weak_patterns = [
        r"\b(home|menu|search|previous|next|tagged|category|posted by)\b",
        r"\b(email address|name required|website optional|privacy policy|terms of service)\b",
        r"(ホーム|メニュー|検索|前の記事|次の記事|カテゴリ|タグ)",
    ]
    return sum(1 for pattern in weak_patterns if re.search(pattern, lower, re.IGNORECASE)) >= 2


def build_translation_task_input(row: dict[str, Any], spec: dict[str, Any]) -> str:
    source_field = str(spec.get("source_text_field") or "og_full_text")
    source_text = normalize_translation_text(row.get(source_field))
    if not source_text:
        return ""
    language_config = translation_language_config(row)
    language_names = spec.get("language_names") or {}
    language_name = str(language_names.get(language_config, language_config))
    return (
        f"source_language: {language_name}\n"
        f"source_language_code: {language_config}\n"
        f"source_text:\n{source_text}"
    ).strip()


def normalize_meeting_transcript(text: Any) -> str:
    text = clean_text(text)
    text = re.sub(r"\[[^\]\n]{1,80}\]", " ", text)
    text = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def build_meeting_summary_task_input(row: dict[str, Any], spec: dict[str, Any]) -> str:
    uid = clean_text(row.get("uid"))
    transcript_field = str(spec.get("transcript_field") or "transcript")
    transcript = normalize_meeting_transcript(row.get(transcript_field))
    max_words = int(spec.get("max_transcript_words") or 1800)
    transcript_excerpt = truncate_words(transcript, max_words)
    if word_count(transcript_excerpt) < 80:
        return ""
    parts = []
    if uid:
        parts.append(f"meeting_uid: {uid}")
    parts.append(f"transcript_excerpt: {transcript_excerpt}")
    return "\n".join(parts).strip()


def truncate_words(text: str, max_words: int) -> str:
    if max_words <= 0:
        return ""
    words = re.findall(r"\S+", text or "")
    if len(words) <= max_words:
        return clean_text(text)
    return clean_text(" ".join(words[:max_words]))


def is_usable_blog_text(text: str) -> bool:
    paragraphs = split_blank_paragraphs(text)
    words = word_count(text)
    if not paragraphs:
        return False
    if words > 2500:
        return False
    if len(paragraphs) > 60:
        return False
    code_hits = len(BLOG_CODE_RE.findall(text))
    if code_hits > 20:
        return False
    short_paragraphs = sum(1 for paragraph in paragraphs if word_count(paragraph) <= 4)
    if short_paragraphs / max(len(paragraphs), 1) > 0.35:
        return False
    return True


def is_usable_product_review_text(text: str) -> bool:
    paragraphs = split_blank_paragraphs(text)
    if not paragraphs:
        return False
    if len(paragraphs) > 6:
        return False
    short_paragraphs = sum(1 for paragraph in paragraphs if word_count(paragraph) <= 5)
    if short_paragraphs / max(len(paragraphs), 1) > 0.35:
        return False
    noisy_patterns = [
        r"\b(click here|subscribe|privacy policy|terms of use|customer service)\b",
        r"https?://|www\.",
        r"\b(questions? and answers?|seller feedback)\b",
    ]
    lower = text.lower()
    return not any(re.search(pattern, lower, re.IGNORECASE) for pattern in noisy_patterns)


def split_blank_paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n+", text.strip()) if part.strip()]


def is_blog_code_block(lines: list[str]) -> bool:
    if len(lines) < 2:
        return False
    joined = "\n".join(lines)
    return len(BLOG_CODE_RE.findall(joined)) >= max(2, len(lines) // 3)


def is_peer_review_standalone_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if re.match(r"^[-*]\s+", stripped):
        return True
    if re.match(r"^\d+[.)]\s+", stripped):
        return True
    if stripped.lower() in {
        "pros",
        "cons",
        "strengths",
        "weaknesses",
        "summary",
        "general discussion",
        "after author response",
        "notes from ac",
    }:
        return True
    if stripped.endswith(":") and len(stripped.split()) <= 6:
        return True
    return False


if __name__ == "__main__":
    main()

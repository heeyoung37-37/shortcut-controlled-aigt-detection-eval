from __future__ import annotations

import argparse
import re
from typing import Any

from tqdm import tqdm

from .config import data_dir, load_config, split_alias
from .io_utils import append_jsonl, read_jsonl
from .llm_client import LLMClient
from .text_stats import (
    capitalized_spans,
    content_tokens,
    extract_numbers,
    extract_quote_spans,
    ngrams,
    normalize_whitespace,
    remove_quote_spans,
    split_sentences,
    word_count,
)

FORBIDDEN_FACT_KEYS = {"source_text", "original_text", "paragraph_text", "source_paragraph", "raw_article"}
NUMBER_WORD_TO_DIGIT = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}
DIGIT_TO_NUMBER_WORD = {value: key for key, value in NUMBER_WORD_TO_DIGIT.items()}


def extract_fact_card_rule_based(record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    text = record["text"]
    fact_config = config["fact_card"]
    entities = [
        {"name": span, "type": "ENTITY", "role": "mentioned entity"}
        for span in capitalized_spans(text)[: int(fact_config.get("max_entities", 15))]
    ]
    numbers = [
        {"value": number, "meaning": "number or date mentioned in the source"}
        for number in extract_numbers(text)[: int(fact_config.get("max_numbers", 8))]
    ]
    quotes = [
        {"speaker": "unknown", "quote": quote, "context": "direct quote from source"}
        for quote in extract_quote_spans(text)[: int(fact_config.get("max_quotes", 3))]
    ]
    max_facts = int(fact_config.get("max_core_facts", 12))
    core_facts = []
    for idx, sentence in enumerate(split_sentences(remove_quote_spans(text))[:max_facts], 1):
        tokens = []
        seen = set()
        for token in content_tokens(sentence):
            if token not in seen:
                tokens.append(token)
                seen.add(token)
            if len(tokens) >= 10:
                break
        if tokens:
            core_facts.append(
                {
                    "fid": idx,
                    "importance": idx,
                    "content": "Fact note: " + ", ".join(tokens),
                }
            )
    if not core_facts:
        core_facts.append({"fid": 1, "importance": 1, "content": "Fact note: news event"})
    return {
        "fact_id": record["id"],
        "split": record["split"],
        "source_word_count": word_count(text),
        "event_type": "news event",
        "topic_tags": [],
        "entities": entities,
        "core_facts": core_facts,
        "numbers": numbers,
        "quotes": quotes,
        "constraints": _default_constraints(),
        "extraction_model": "rule_based",
        "usage": {},
    }


def extract_fact_card_hybrid(record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    try:
        client = LLMClient(config)
        card, usage = client.extract_fact_card_llm(record["text"])
        enriched = _enrich_fact_card(card, record, config, usage)
        reasons = validate_fact_card(enriched, record["text"], config)
        if not reasons:
            return enriched
    except Exception:
        pass
    return extract_fact_card_rule_based(record, config)


def extract_fact_cards_for_split(config: dict[str, Any], split: str, limit: int | None = None) -> list[dict[str, Any]]:
    split = split_alias(config, split)
    pool_path = data_dir(config, "processed_dir") / "pools" / f"{split}_fact_pool.jsonl"
    records = read_jsonl(pool_path)
    effective_limit = limit or config.get("_runtime_limit")
    if effective_limit:
        records = records[: int(effective_limit)]
    out_path = data_dir(config, "processed_dir") / "fact_cards" / f"{split}.jsonl"
    failure_path = data_dir(config, "reports_dir") / f"fact_card_failures_{split}.jsonl"
    extractor = config["fact_card"].get("extractor", "llm")
    existing_records = read_jsonl(out_path)
    if config.get("_quick_test"):
        expected_model = "rule_based" if extractor == "rule_based" else config.get("llm", {}).get("fact_model")
        existing_records = [record for record in existing_records if record.get("extraction_model") == expected_model]
        from .io_utils import write_jsonl

        write_jsonl(existing_records, out_path)
    existing = {record.get("fact_id") for record in existing_records}
    client = LLMClient(config) if extractor == "llm" else None
    results: list[dict[str, Any]] = existing_records[:]

    for record in tqdm(records, desc=f"fact_cards:{split}"):
        if record["id"] in existing:
            continue
        try:
            if extractor == "rule_based":
                fact_card = extract_fact_card_rule_based(record, config)
                reasons = validate_fact_card(fact_card, record["text"], config, strict_min_facts=False)
            elif extractor == "hybrid":
                fact_card = extract_fact_card_hybrid(record, config)
                reasons = validate_fact_card(fact_card, record["text"], config, strict_min_facts=False)
            elif extractor == "llm":
                fact_card, reasons = _extract_valid_llm_fact_card(client, record, config)
            else:
                raise ValueError(f"Unknown fact_card.extractor: {extractor}")

            if reasons:
                append_jsonl(
                    {"id": record["id"], "split": split, "fail_reasons": reasons, "extractor": extractor},
                    failure_path,
                )
                continue
            append_jsonl(fact_card, out_path)
            existing.add(record["id"])
            results.append(fact_card)
        except Exception as exc:
            append_jsonl(
                {
                    "id": record.get("id"),
                    "split": split,
                    "error": str(exc),
                    "extractor": extractor,
                },
                failure_path,
            )
    return results


def validate_fact_card(
    card: dict[str, Any],
    source_text: str,
    config: dict[str, Any],
    strict_min_facts: bool = True,
) -> list[str]:
    reasons: list[str] = []
    required = {"event_type", "topic_tags", "entities", "core_facts", "numbers", "quotes", "constraints"}
    missing = sorted(required - set(card))
    if missing:
        reasons.append(f"missing_keys:{','.join(missing)}")
        return reasons

    fact_config = config["fact_card"]
    strict_mode = fact_config.get("validation_mode", "strict") == "strict"
    core_facts = card.get("core_facts") or []
    entities = card.get("entities") or []
    numbers = card.get("numbers") or []
    quotes = card.get("quotes") or []
    if not isinstance(core_facts, list) or not isinstance(entities, list) or not isinstance(numbers, list) or not isinstance(quotes, list):
        reasons.append("schema_lists_invalid")
        return reasons

    min_facts = int(fact_config.get("min_core_facts", 5))
    if not core_facts:
        reasons.append("empty_core_facts")
    elif strict_mode and strict_min_facts and word_count(source_text) >= 120 and len(core_facts) < min_facts:
        reasons.append(f"too_few_core_facts:{len(core_facts)}")
    if strict_mode and len(core_facts) > int(fact_config.get("max_core_facts", 12)):
        reasons.append("too_many_core_facts")
    if strict_mode and len(entities) > int(fact_config.get("max_entities", 15)):
        reasons.append("too_many_entities")
    if strict_mode and len(numbers) > int(fact_config.get("max_numbers", 8)):
        reasons.append("too_many_numbers")
    if strict_mode and len(quotes) > int(fact_config.get("max_quotes", 3)):
        reasons.append("too_many_quotes")

    if _contains_forbidden_keys(card):
        reasons.append("contains_forbidden_raw_text_field")

    source_quote_bank = {_quote_key(q) for q in extract_quote_spans(source_text)}
    for quote in quotes:
        quote_text = str(quote.get("quote", ""))
        if quote_text and _quote_key(quote_text) not in source_quote_bank:
            near = any(_quote_key(quote_text) in source_quote or source_quote in _quote_key(quote_text) for source_quote in source_quote_bank)
            if not near and (strict_mode or fact_config.get("hard_reject_quote_mismatch", False)):
                reasons.append("quote_not_in_source")
                break

    source_norm = _loose_text_key(source_text)
    for number in numbers:
        value = str(number.get("value", "")).strip()
        if value and not _number_value_in_source(value, source_text, source_norm):
            if not (strict_mode or fact_config.get("hard_reject_number_mismatch", False)):
                continue
            reasons.append(f"number_not_in_source:{value}")
            break

    if fact_config.get("forbid_nonquote_6gram_copy", True) and card.get("extraction_model") != "rule_based":
        source_6grams = set(ngrams([t.lower() for t in content_tokens(remove_quote_spans(source_text))], 6))
        for fact in core_facts:
            fact_text = str(fact.get("content", ""))
            fact_6grams = set(ngrams([t.lower() for t in content_tokens(fact_text)], 6))
            if fact_6grams & source_6grams:
                if not (strict_mode or fact_config.get("hard_reject_core_fact_copy", False)):
                    break
                reasons.append("core_fact_copies_source_6gram")
                break

    return reasons


def _extract_valid_llm_fact_card(
    client: LLMClient | None,
    record: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    if client is None:
        raise RuntimeError("LLM client not initialized")
    last_card: dict[str, Any] | None = None
    last_reasons: list[str] = []
    for _ in range(2):
        card, usage = client.extract_fact_card_llm(record["text"])
        enriched = _enrich_fact_card(card, record, config, usage)
        reasons = validate_fact_card(enriched, record["text"], config)
        if not reasons:
            return enriched, []
        last_card = enriched
        last_reasons = reasons
    return last_card or {}, last_reasons or ["fact_card_validation_failed"]


def _enrich_fact_card(
    card: dict[str, Any],
    record: dict[str, Any],
    config: dict[str, Any],
    usage: dict[str, Any],
) -> dict[str, Any]:
    enriched = dict(card)
    enriched["fact_id"] = record["id"]
    enriched["split"] = record["split"]
    enriched["source_word_count"] = word_count(record["text"])
    enriched["extraction_model"] = config["llm"].get("fact_model")
    enriched["usage"] = usage
    enriched.setdefault("constraints", _default_constraints())
    return _sanitize_fact_card(enriched, record["text"], config)


def _sanitize_fact_card(card: dict[str, Any], source_text: str, config: dict[str, Any]) -> dict[str, Any]:
    fact_config = config["fact_card"]
    if fact_config.get("validation_mode", "strict") == "strict":
        return card

    sanitized = dict(card)
    warnings: list[str] = list(sanitized.get("validation_warnings") or [])
    sanitized.setdefault("event_type", "news event")
    sanitized.setdefault("topic_tags", [])
    sanitized.setdefault("entities", [])
    sanitized.setdefault("core_facts", [])
    sanitized.setdefault("numbers", [])
    sanitized.setdefault("quotes", [])
    sanitized.setdefault("constraints", _default_constraints())

    sanitized["topic_tags"] = [str(item) for item in _as_list(sanitized.get("topic_tags")) if str(item).strip()]
    sanitized["entities"] = _dict_list(sanitized.get("entities"))[: int(fact_config.get("max_entities", 15))]
    sanitized["core_facts"] = _dict_list(sanitized.get("core_facts"))[: int(fact_config.get("max_core_facts", 12))]

    source_norm = _loose_text_key(source_text)
    kept_numbers = []
    for item in _dict_list(sanitized.get("numbers")):
        value = str(item.get("value", "")).strip()
        if not value:
            continue
        if not _number_value_in_source(value, source_text, source_norm):
            warnings.append(f"number_surface_not_matched:{value}")
        kept_numbers.append(item)
    sanitized["numbers"] = kept_numbers[: int(fact_config.get("max_numbers", 8))]

    source_quote_bank = {_quote_key(q) for q in extract_quote_spans(source_text)}
    kept_quotes = []
    for item in _dict_list(sanitized.get("quotes")):
        quote_text = str(item.get("quote", "")).strip()
        if not quote_text:
            continue
        quote_key = _quote_key(quote_text)
        near = quote_key in source_quote_bank or any(quote_key in source_quote or source_quote in quote_key for source_quote in source_quote_bank)
        if near:
            kept_quotes.append(item)
        else:
            warnings.append("dropped_quote_not_in_source")
    sanitized["quotes"] = kept_quotes[: int(fact_config.get("max_quotes", 3))]

    if fact_config.get("forbid_nonquote_6gram_copy", True) and _core_fact_copies_source(sanitized, source_text):
        warnings.append("core_fact_copies_source_6gram")
    if warnings:
        sanitized["validation_warnings"] = sorted(set(warnings))
    return sanitized


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _dict_list(value: Any) -> list[dict[str, Any]]:
    return [item for item in _as_list(value) if isinstance(item, dict)]


def _core_fact_copies_source(card: dict[str, Any], source_text: str) -> bool:
    source_6grams = set(ngrams([t.lower() for t in content_tokens(remove_quote_spans(source_text))], 6))
    for fact in card.get("core_facts") or []:
        fact_text = str(fact.get("content", "")) if isinstance(fact, dict) else str(fact)
        fact_6grams = set(ngrams([t.lower() for t in content_tokens(fact_text)], 6))
        if fact_6grams & source_6grams:
            return True
    return False


def _number_value_in_source(value: str, source_text: str, source_norm: str | None = None) -> bool:
    source_norm = source_norm if source_norm is not None else _loose_text_key(source_text)
    value_keys = _number_key_variants(value)
    if any(key and key in source_norm for key in value_keys):
        return True
    source_keys: set[str] = set()
    for number in extract_numbers(source_text):
        source_keys.update(_number_key_variants(number))
    return bool(value_keys & source_keys)


def _number_key_variants(value: str) -> set[str]:
    base = _loose_text_key(value).replace(",", "")
    variants = {base}
    variants.add(base.replace(" percent", "%").replace(" %", "%"))
    variants.add(base.replace("%", " percent"))
    variants.add(base.replace("£", "").replace("$", "").replace("€", "").strip())
    for number in extract_numbers(value):
        key = _loose_text_key(number).replace(",", "")
        variants.add(key)
        variants.add(key.replace(" percent", "%").replace(" %", "%"))
        variants.add(key.replace("%", " percent"))
    expanded = set(variants)
    for key in variants:
        if key in NUMBER_WORD_TO_DIGIT:
            expanded.add(NUMBER_WORD_TO_DIGIT[key])
        if key in DIGIT_TO_NUMBER_WORD:
            expanded.add(DIGIT_TO_NUMBER_WORD[key])
    return {key for key in expanded if key}


def _default_constraints() -> dict[str, bool]:
    return {
        "use_only_listed_facts": True,
        "do_not_invent_quotes": True,
        "do_not_invent_numbers": True,
        "do_not_add_unlisted_named_entities": True,
    }


def _contains_forbidden_keys(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FORBIDDEN_FACT_KEYS:
                return True
            if _contains_forbidden_keys(item):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_keys(item) for item in value)
    return False


def _quote_key(text: str) -> str:
    return normalize_whitespace(str(text).replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")).lower()


def _loose_text_key(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w%$£€.-]+", " ", str(text).lower())).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--split", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    extract_fact_cards_for_split(load_config(args.config), args.split, limit=args.limit)


if __name__ == "__main__":
    main()

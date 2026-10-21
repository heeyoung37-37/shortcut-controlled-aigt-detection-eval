#!/usr/bin/env python3
"""Generate shortcut-controlled Fast-DetectGPT-style data files for Our_data.

This runner consumes deterministic source shard inputs and produces the
original Fast-DetectGPT schema:

    {"original": [...], "sampled": [...]}

It uses structure cards and fact cards inspired by grok_ultimate_gen, but keeps
provider calls, dataset prompts, and shortcut validation local to this script.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grok_ultimate_gen.src.extract_fact_cards import (
    _enrich_fact_card,
    extract_fact_card_rule_based,
    validate_fact_card,
)
from grok_ultimate_gen.src.extract_structure_cards import extract_structure_card
from grok_ultimate_gen.src.io_utils import append_jsonl, read_jsonl, write_jsonl

from scripts.common import load_domain_specs
from scripts.text_postprocess import sentence_safe_postprocess_text
from scripts.ultimate_controlled.prompts import (
    fact_extraction_messages,
    fact_repair_messages,
    generation_messages,
)
from scripts.ultimate_controlled.providers import (
    ClaudeProviderClient,
    DeepSeekProviderClient,
    GeminiProviderClient,
    OpenAIProviderClient,
    ProviderClient,
    ProviderResponse,
    QwenProviderClient,
    parse_json_object,
)
from scripts.ultimate_controlled.shortcut_validator import (
    build_shortcut_profile,
    retry_guidance,
    validate_generation,
)


DEFAULT_DATASETS = ("essay", "arxiv_abstract", "blog", "product_review", "peer_review", "meeting_summary", "translation")
LEGACY_DATASETS = {"xsum", "squad", "wp", "pubmed"}
DEFAULT_SAMPLE_COUNT = 30
DEFAULT_MAX_CANDIDATES = 45
DEFAULT_SHARD_ROOT = Path("data/source_shards/final")
DEFAULT_OUTPUT_ROOT = Path("runs/final/generated")
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 1800
DEFAULT_FACT_MAX_TOKENS = 1600
DEFAULT_QWEN_WP_MAX_TOKENS = 1200
DEFAULT_RETRIES = 2
DEFAULT_MIN_WORDS = 55
DEFAULT_PUBMED_MIN_WORDS = 30
DEFAULT_SEED = 42
DEFAULT_TEMPLATE_IDS = ("template_01", "template_02", "template_03")
DEFAULT_QWEN_API_BASE = "http://127.0.0.1:8000/v1"
DEFAULT_QWEN_CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}
DEFAULT_DEEPSEEK_API_BASE = "https://api.deepseek.com"
SAME_SOURCE_PAIRING_DATASETS = {"meeting_summary", "product_review", "translation"}

PROVIDER_DEFAULT_MODELS = {
    "gemini": "gemini-2.5-pro",
    "claude": "claude-sonnet-4-6",
    "qwen": "Qwen/Qwen3-8B",
    "openai": "gpt-3.5-turbo",
    "deepseek": "deepseek-v4-pro",
}

PROVIDER_MODEL_LABELS = {
    "gemini": "gemini-2.5-pro",
    "claude": "claude-sonnet-4-6",
    "qwen": "qwen3-8b",
    "openai": "gpt-3.5-turbo",
    "deepseek": "deepseek-v4-pro",
}

PUBMED_SLOT_BY_PROVIDER = {
    "gemini": "model1",
    "claude": "model2",
    "qwen": "model3",
    "openai": "model1",
    "deepseek": "model3",
}

PUBMED_SLOT_BY_OPENAI_MODEL = {
    "gpt-3.5-turbo": "model1",
    "gpt-4": "model2",
    "gpt-5.5": "model3",
}

OPENAI_MODEL_ALIASES = {
    "gpt3.5": "gpt-3.5-turbo",
    "gpt3.5-turbo": "gpt-3.5-turbo",
    "gpt3.5 turbo": "gpt-3.5-turbo",
    "gpt 3.5": "gpt-3.5-turbo",
    "gpt 3.5 turbo": "gpt-3.5-turbo",
    "gpt35": "gpt-3.5-turbo",
    "gpt-35": "gpt-3.5-turbo",
    "gpt4": "gpt-4",
    "gpt 4": "gpt-4",
    "gpt5.5": "gpt-5.5",
    "gpt 5.5": "gpt-5.5",
    "gpt55": "gpt-5.5",
    "gput3.5 turbo": "gpt-3.5-turbo",
}


@dataclass(frozen=True)
class SourceRecord:
    id: str
    split: str
    dataset: str
    text: str
    source_index: int
    shard_kind: str
    model_slot: str
    task_input: str = ""
    metadata: dict[str, Any] | None = None

    @property
    def source_id(self) -> str:
        metadata = self.metadata or {}
        return str(metadata.get("_source_id") or self.id)

    def as_card_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "split": self.split,
            "dataset": self.dataset,
            "text": self.text,
            "source_index": self.source_index,
            "shard_kind": self.shard_kind,
            "model_slot": self.model_slot,
            "task_input": self.task_input,
            "metadata": self.metadata or {},
        }


@dataclass(frozen=True)
class CandidatePair:
    pair_id: str
    structure: dict[str, Any]
    fact: dict[str, Any]
    structure_source: SourceRecord
    fact_source: SourceRecord
    matching_info: dict[str, Any]


@dataclass(frozen=True)
class OutputPaths:
    output_dir: Path
    artifact_dir: Path
    raw_data: Path
    args: Path
    state: Path
    accepted: Path
    failures: Path
    metrics: Path
    profile: Path
    structure_cards: Path
    fact_cards: Path
    pairs: Path
    summary: Path


def sanitize_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "model"


def model_label_for(provider: str, model: str, explicit_label: str | None = None) -> str:
    if explicit_label:
        return sanitize_label(explicit_label)
    if provider in PROVIDER_MODEL_LABELS and model == PROVIDER_DEFAULT_MODELS[provider]:
        return PROVIDER_MODEL_LABELS[provider]
    return sanitize_label(model.split("/")[-1] if provider == "qwen" else model)


def normalize_model_for_provider(provider: str, model: str) -> str:
    if provider != "openai":
        return model
    return OPENAI_MODEL_ALIASES.get(model.strip().lower(), model)


def model_slot_for(provider: str, dataset: str, model_label: str | None = None, pubmed_model_slot: str | None = None) -> str:
    if dataset not in LEGACY_DATASETS:
        if model_label:
            return model_label
        return PROVIDER_MODEL_LABELS[provider]
    if dataset == "pubmed":
        if pubmed_model_slot:
            return pubmed_model_slot
        if provider == "openai" and model_label:
            return PUBMED_SLOT_BY_OPENAI_MODEL.get(model_label, model_label)
        return PUBMED_SLOT_BY_PROVIDER[provider]
    if provider == "deepseek":
        return PROVIDER_MODEL_LABELS["qwen"]
    if provider == "openai" and model_label:
        return model_label
    return PROVIDER_MODEL_LABELS[provider]


def output_paths(output_root: Path, model_label: str, dataset: str) -> OutputPaths:
    output_dir = output_root / model_label / dataset
    artifact_dir = output_dir / "artifacts"
    stem = f"{dataset}_{model_label}"
    return OutputPaths(
        output_dir=output_dir,
        artifact_dir=artifact_dir,
        raw_data=output_dir / f"{stem}.data.json",
        args=artifact_dir / f"{stem}.args.json",
        state=artifact_dir / f"{stem}.state.json",
        accepted=artifact_dir / f"{stem}.accepted.jsonl",
        failures=artifact_dir / f"{stem}.failures.jsonl",
        metrics=artifact_dir / f"{stem}.metrics.jsonl",
        profile=artifact_dir / f"{stem}.metrics_profile.json",
        structure_cards=artifact_dir / f"{stem}.structure_cards.jsonl",
        fact_cards=artifact_dir / f"{stem}.fact_cards.jsonl",
        pairs=artifact_dir / f"{stem}.pairs.jsonl",
        summary=artifact_dir / f"{stem}.summary.json",
    )


def load_originals(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if sorted(payload.keys()) != ["original"]:
        raise ValueError(f"{path} must contain exactly one key: original")
    originals = payload["original"]
    if not isinstance(originals, list):
        raise ValueError(f"{path} original must be a list")
    cleaned = [str(text).strip() for text in originals if str(text).strip()]
    if len(cleaned) != len(originals):
        raise ValueError(f"{path} contains empty originals")
    return cleaned


def load_shard_records(shard_root: Path, dataset: str, model_slot: str, shard_kind: str) -> list[SourceRecord]:
    path = shard_root / dataset / model_slot / f"{shard_kind}.jsonl"
    originals = load_originals(path)
    sidecar_rows = read_jsonl(path.with_name(f"{shard_kind}.sources.jsonl"))
    sidecar_by_index = {int(row.get("source_index", idx)): row for idx, row in enumerate(sidecar_rows)}
    records = []
    for idx, text in enumerate(originals):
        sidecar = sidecar_by_index.get(idx, {})
        metadata = sidecar.get("metadata") if isinstance(sidecar.get("metadata"), dict) else {}
        metadata = dict(metadata)
        if sidecar.get("source_id"):
            metadata["_source_id"] = str(sidecar["source_id"])
        records.append(
            SourceRecord(
                id=f"{dataset}_{model_slot}_{shard_kind}_{idx:06d}",
                split=f"{dataset}_{model_slot}",
                dataset=dataset,
                text=text,
                source_index=idx,
                shard_kind=shard_kind,
                model_slot=model_slot,
                task_input=str(sidecar.get("task_input") or ""),
                metadata=metadata,
            )
        )
    return records


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def build_config(args: argparse.Namespace, model: str) -> dict[str, Any]:
    return {
        "structure_card": {
            "length_boost_multiplier": 1.45,
            "max_sentences_per_paragraph": 3,
            "forbid_em_dash": True,
            "avoid_transition_words": [
                "additionally",
                "furthermore",
                "moreover",
                "ultimately",
                "overall",
                "in addition",
                "in contrast",
                "similarly",
            ],
            "avoid_paragraph_initial_words": ["additionally", "furthermore", "moreover", "despite", "as", "with", "to"],
        },
        "fact_card": {
            "extractor": args.fact_extractor,
            "validation_mode": "lenient",
            "max_core_facts": 12,
            "min_core_facts": 5,
            "max_entities": 15,
            "max_numbers": 8,
            "max_quotes": 3,
            "forbid_nonquote_6gram_copy": True,
            "hard_reject_quote_mismatch": False,
            "hard_reject_number_mismatch": False,
            "hard_reject_core_fact_copy": False,
        },
        "llm": {
            "provider": args.provider,
            "fact_model": model,
            "generation_model": model,
        },
    }


def build_structure_cards(records: list[SourceRecord], config: dict[str, Any]) -> list[dict[str, Any]]:
    cards = []
    for record in records:
        card = extract_structure_card(record.as_card_record(), config)
        card["dataset"] = record.dataset
        card["source_index"] = record.source_index
        if record.dataset == "meeting_summary":
            card["surface_style"] = classify_meeting_summary_style(record.text)
        elif record.dataset == "product_review":
            card["surface_style"] = classify_product_review_style(record.text)
        cards.append(card)
    return cards


def classify_meeting_summary_style(text: str) -> dict[str, str]:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    lower = normalized.lower()
    if lower.startswith("an ordinance") or lower.startswith("ordinance"):
        summary_kind = "ordinance_title"
        instruction = "Use ordinance/action-item wording, not narrative meeting prose."
    elif lower.startswith("a resolution") or lower.startswith("resolution"):
        summary_kind = "resolution_title"
        instruction = "Use resolution/action-item wording, not narrative meeting prose."
    elif lower.startswith("recommendation to"):
        summary_kind = "recommendation"
        instruction = "Start with a recommendation-style action phrase when possible."
    elif lower.startswith("a motion") or lower.startswith("motion"):
        summary_kind = "motion"
        instruction = "Use motion/action request wording when possible."
    elif lower.startswith("a proclamation") or lower.startswith("proclamation"):
        summary_kind = "proclamation"
        instruction = "Use proclamation-style civic recognition wording when possible."
    elif re.match(r"^(approve|adopt|authorize|accept|receive|refer|pass|grant)\b", lower):
        summary_kind = "imperative_action"
        instruction = "Use compact action-item wording with the main council action first."
    else:
        summary_kind = "agenda_item_summary"
        instruction = "Use compact agenda-item summary wording rather than a narrative recap."
    return {
        "summary_kind": summary_kind,
        "surface_register": "municipal_agenda_item",
        "style_instruction": instruction,
    }


def classify_product_review_style(text: str) -> dict[str, str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", (text or "").strip()) if part.strip()]
    normalized = re.sub(r"\s+", " ", text or "").strip()
    lower = normalized.lower()
    first_person = bool(re.search(r"\b(i|i'm|i've|my|me|we|we're|we've|our|husband|wife|daughter|son|kids)\b", lower))
    has_complaint = bool(
        re.search(r"\b(disappointed|problem|issue|wish|unfortunately|however|but|returned|broke|broken|leaked|dry|irritat)\b", lower)
    )
    has_sample_disclosure = bool(re.search(r"\b(sample|provided|received|vine|honest review|free product)\b", lower))
    if has_complaint:
        sentiment_shape = "mixed_or_critical"
    elif re.search(r"\b(love|great|excellent|perfect|favorite|recommend|works well|amazing)\b", lower):
        sentiment_shape = "positive"
    else:
        sentiment_shape = "matter_of_fact"
    paragraph_count = len(paragraphs) or 1
    instruction = (
        f"Use a {sentiment_shape.replace('_', ' ')} consumer-review register. "
        "Keep the wording personal and concrete rather than promotional, and write the review as one paragraph."
    )
    if first_person:
        instruction += " Use first-person experience wording when natural."
    if has_sample_disclosure:
        instruction += " Include a concise review-disclosure style only if supported by FACT_CARD."
    return {
        "review_register": "consumer_product_review",
        "sentiment_shape": sentiment_shape,
        "paragraph_count": str(paragraph_count),
        "first_person": "true" if first_person else "false",
        "has_complaint": "true" if has_complaint else "false",
        "has_sample_disclosure": "true" if has_sample_disclosure else "false",
        "style_instruction": instruction,
    }


async def build_fact_cards(
    provider: ProviderClient,
    records: list[SourceRecord],
    dataset: str,
    config: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if dataset == "translation":
        return build_translation_source_cards(records)
    cards = []
    for idx, record in enumerate(records, 1):
        card = await extract_fact_card(provider, dataset, record, config, args)
        cards.append(card)
        print(f"  [fact] {dataset}: {idx}/{len(records)}", flush=True)
    return cards


def build_translation_source_cards(records: list[SourceRecord]) -> list[dict[str, Any]]:
    cards = []
    for record in records:
        metadata = record.metadata or {}
        source_language = str(metadata.get("language_name") or metadata.get("language_config") or "foreign language")
        source_language_code = str(metadata.get("language_config") or metadata.get("og_language") or "")
        source_text = record.text
        if "source_text:\n" in source_text:
            source_text = source_text.split("source_text:\n", 1)[1].strip()
        cards.append(
            {
                "fact_id": f"{record.id}_source",
                "schema_version": "translation_source_card_v1",
                "dataset": record.dataset,
                "source_index": record.source_index,
                "model_slot": record.model_slot,
                "source_language": source_language,
                "source_language_code": source_language_code,
                "source_text": source_text,
                "source_word_count": int(metadata.get("translated_word_count") or len(re.findall(r"\b[\w'-]+\b", record.text))),
                "entities": [],
                "core_facts": [],
                "numbers": [],
                "quotes": [],
                "constraints": {
                    "translate_only_source_text": True,
                    "preserve_meaning": True,
                    "target_language": "English",
                },
            }
        )
    return cards


async def extract_fact_card(
    provider: ProviderClient,
    dataset: str,
    record: SourceRecord,
    config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.fact_extractor == "rule_based":
        return attach_fact_source_metadata(extract_fact_card_rule_based(record.as_card_record(), config), record)

    max_core = int(config["fact_card"]["max_core_facts"])
    max_entities = int(config["fact_card"]["max_entities"])
    max_numbers = int(config["fact_card"]["max_numbers"])
    max_quotes = int(config["fact_card"]["max_quotes"])
    first_error = ""
    raw_text = ""
    try:
        response = await provider.generate_text(
            fact_extraction_messages(
                dataset,
                record.text,
                max_core_facts=max_core,
                max_entities=max_entities,
                max_numbers=max_numbers,
                max_quotes=max_quotes,
            ),
            temperature=args.fact_temperature,
            top_p=None,
            max_tokens=args.fact_max_tokens,
            thinking_budget=args.gemini_fact_thinking_budget if args.provider == "gemini" else None,
        )
        raw_text = response.text
        card = parse_json_object(raw_text)
        enriched = _enrich_fact_card(card, record.as_card_record(), config, response.usage)
        enriched = attach_fact_source_metadata(enriched, record)
        reasons = validate_fact_card(enriched, record.text, config, strict_min_facts=False)
        if not reasons:
            return enriched
        first_error = "fact_card_validation:" + ",".join(reasons)
    except Exception as exc:
        first_error = str(exc)

    try:
        repair_response = await provider.generate_text(
            fact_repair_messages(dataset, raw_text, first_error),
            temperature=0.0,
            top_p=None,
            max_tokens=args.fact_max_tokens,
            thinking_budget=args.gemini_fact_thinking_budget if args.provider == "gemini" else None,
        )
        repaired = parse_json_object(repair_response.text)
        enriched = _enrich_fact_card(repaired, record.as_card_record(), config, repair_response.usage)
        enriched = attach_fact_source_metadata(enriched, record)
        reasons = validate_fact_card(enriched, record.text, config, strict_min_facts=False)
        if not reasons:
            enriched["repair_applied"] = True
            return enriched
        first_error = "repair_validation:" + ",".join(reasons)
    except Exception as exc:
        first_error = f"{first_error}; repair_failed:{exc}"

    if args.fact_extractor == "llm":
        raise RuntimeError(f"Fact card extraction failed for {record.id}: {first_error}")
    fallback = attach_fact_source_metadata(extract_fact_card_rule_based(record.as_card_record(), config), record)
    fallback["extraction_warning"] = first_error
    return fallback


def attach_fact_source_metadata(card: dict[str, Any], record: SourceRecord) -> dict[str, Any]:
    enriched = dict(card)
    enriched["dataset"] = record.dataset
    enriched["source_index"] = record.source_index
    enriched["model_slot"] = record.model_slot
    return enriched


def cached_or_build_structure_cards(paths: OutputPaths, records: list[SourceRecord], config: dict[str, Any]) -> list[dict[str, Any]]:
    existing = read_jsonl(paths.structure_cards)
    if len(existing) >= len(records):
        return existing[: len(records)]
    cards = build_structure_cards(records, config)
    write_jsonl(cards, paths.structure_cards)
    return cards


async def cached_or_build_fact_cards(
    paths: OutputPaths,
    provider: ProviderClient,
    records: list[SourceRecord],
    dataset: str,
    config: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    existing = read_jsonl(paths.fact_cards)
    if len(existing) >= len(records):
        return existing[: len(records)]
    cards = await build_fact_cards(provider, records, dataset, config, args)
    write_jsonl(cards, paths.fact_cards)
    return cards


def pair_cards(
    structure_cards: list[dict[str, Any]],
    fact_cards: list[dict[str, Any]],
    structure_records: list[SourceRecord],
    fact_records: list[SourceRecord],
    *,
    limit: int,
    seed: int,
) -> list[CandidatePair]:
    structures = _with_bins(structure_cards, "structure")
    facts = _with_bins(fact_cards, "fact")
    rng = random.Random(seed)
    rng.shuffle(structures)
    rng.shuffle(facts)
    source_by_structure = {card["structure_id"]: structure_records[int(card["source_index"])] for card in structures}
    source_by_fact = {card["fact_id"]: fact_records[int(card["source_index"])] for card in facts}
    available = {fact["fact_id"]: fact for fact in facts}
    by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for fact in facts:
        key = (fact["length_bin"], fact["quote_bin"], fact["number_bin"])
        by_key.setdefault(key, []).append(fact)

    pairs: list[CandidatePair] = []
    target = min(limit, len(structures), len(facts))
    for structure in structures:
        if len(pairs) >= target:
            break
        fact = _choose_fact(structure, available, by_key)
        if not fact:
            continue
        available.pop(fact["fact_id"], None)
        pair_id = f"pair_{len(pairs):06d}"
        pairs.append(
            CandidatePair(
                pair_id=pair_id,
                structure=structure,
                fact=fact,
                structure_source=source_by_structure[structure["structure_id"]],
                fact_source=source_by_fact[fact["fact_id"]],
                matching_info={
                    "length_bin": fact["length_bin"],
                    "quote_bin": fact["quote_bin"],
                    "number_bin": fact["number_bin"],
                },
            )
        )
    return pairs


def pair_same_source_cards(
    structure_cards: list[dict[str, Any]],
    fact_cards: list[dict[str, Any]],
    structure_records: list[SourceRecord],
    fact_records: list[SourceRecord],
    *,
    limit: int,
) -> list[CandidatePair]:
    fact_cards_by_index = {int(card["source_index"]): card for card in fact_cards}
    structure_records_by_index = {record.source_index: record for record in structure_records}
    fact_records_by_index = {record.source_index: record for record in fact_records}
    pairs: list[CandidatePair] = []
    for structure in sorted(structure_cards, key=lambda card: int(card.get("source_index", 0))):
        if len(pairs) >= limit:
            break
        source_index = int(structure["source_index"])
        fact_card = fact_cards_by_index.get(source_index)
        structure_record = structure_records_by_index.get(source_index)
        fact_record = fact_records_by_index.get(source_index)
        if not fact_card or not structure_record or not fact_record:
            continue
        if structure_record.source_id != fact_record.source_id:
            raise ValueError(
                "same-source pairing mismatch at index "
                f"{source_index}: structure={structure_record.source_id} fact={fact_record.source_id}"
            )
        pair_id = f"pair_{len(pairs):06d}"
        pairs.append(
            CandidatePair(
                pair_id=pair_id,
                structure=structure,
                fact=fact_card,
                structure_source=structure_record,
                fact_source=fact_record,
                matching_info={
                    "length_bin": "same_row",
                    "quote_bin": "same_row",
                    "number_bin": "same_row",
                },
            )
        )
    return pairs


def write_pairs(path: Path, pairs: list[CandidatePair]) -> None:
    write_jsonl(
        [
            {
                "pair_id": pair.pair_id,
                "structure_id": pair.structure["structure_id"],
                "fact_id": pair.fact["fact_id"],
                "structure_source_index": pair.structure_source.source_index,
                "fact_source_index": pair.fact_source.source_index,
                "structure_source_id": pair.structure_source.source_id,
                "fact_source_id": pair.fact_source.source_id,
                "matching_info": pair.matching_info,
            }
            for pair in pairs
        ],
        path,
    )


def _with_bins(cards: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    id_key = f"{kind}_id"
    sorted_cards = sorted(cards, key=lambda card: (int(card.get("source_word_count", 0)), card.get(id_key, "")))
    total = len(sorted_cards)
    output = []
    for idx, card in enumerate(sorted_cards):
        enriched = dict(card)
        q = idx / max(total - 1, 1)
        if q < 1 / 3:
            enriched["length_bin"] = "short"
        elif q < 2 / 3:
            enriched["length_bin"] = "medium"
        else:
            enriched["length_bin"] = "long"
        if kind == "structure":
            quote_count = sum(1 for paragraph in enriched.get("paragraphs", []) if paragraph.get("quote_slot"))
            number_count = sum(1 for paragraph in enriched.get("paragraphs", []) if paragraph.get("number_slot"))
        else:
            quote_count = len(enriched.get("quotes") or [])
            number_count = len(enriched.get("numbers") or [])
        enriched["quote_bin"] = _quote_bin(quote_count)
        enriched["number_bin"] = _number_bin(number_count)
        output.append(enriched)
    return output


def _choose_fact(
    structure: dict[str, Any],
    available: dict[str, dict[str, Any]],
    by_key: dict[tuple[str, str, str], list[dict[str, Any]]],
) -> dict[str, Any] | None:
    for length_bin in _nearest_length_bins(structure["length_bin"]):
        for quote_bin in [structure["quote_bin"], "no_quote", "one_quote", "multi_quote"]:
            for number_bin in [structure["number_bin"], "no_number", "one_two_numbers", "many_numbers"]:
                for fact in by_key.get((length_bin, quote_bin, number_bin), []):
                    if fact["fact_id"] in available:
                        return fact
    return next(iter(available.values()), None)


def _nearest_length_bins(bin_name: str) -> list[str]:
    order = ["short", "medium", "long"]
    if bin_name not in order:
        return order
    idx = order.index(bin_name)
    return sorted(order, key=lambda item: abs(order.index(item) - idx))


def _quote_bin(count: int) -> str:
    if count <= 0:
        return "no_quote"
    if count == 1:
        return "one_quote"
    return "multi_quote"


def _number_bin(count: int) -> str:
    if count <= 0:
        return "no_number"
    if count <= 2:
        return "one_two_numbers"
    return "many_numbers"


def load_state(path: Path, metadata: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    if not path.exists():
        return {"metadata": metadata, "accepted": [], "attempted_pair_ids": []}, True
    with path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    existing_metadata = state.get("metadata", {})
    keys = [
        "provider",
        "model",
        "model_label",
        "dataset",
        "sample_count",
        "model_slot",
        "fact_extractor",
        "seed",
    ]
    if any(existing_metadata.get(key) != metadata.get(key) for key in keys):
        return {"metadata": metadata, "accepted": [], "attempted_pair_ids": []}, True
    state.setdefault("accepted", [])
    state.setdefault("attempted_pair_ids", [])
    if existing_metadata.get("max_candidates") != metadata.get("max_candidates"):
        state["metadata"] = metadata
        state["attempted_pair_ids"] = []
    return state, False


def save_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json(path, state)


def initialize_logs(paths: OutputPaths, fresh: bool) -> None:
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    paths.artifact_dir.mkdir(parents=True, exist_ok=True)
    if fresh:
        write_jsonl([], paths.accepted)
        write_jsonl([], paths.failures)
        write_jsonl([], paths.metrics)


def clean_generated_text(text: str) -> str:
    stripped = (text or "").strip()
    stripped = re.sub(r"^```(?:text|markdown)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    stripped = re.sub(r"^\s*(Article|Output|Draft):\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\n{3,}", "\n\n", stripped)
    return stripped.strip()


def truncate_to_substring(text: str, substring: str, occurrence: int) -> str:
    idx = -1
    for _ in range(occurrence):
        idx = text.find(substring, idx + 1)
        if idx == -1:
            return text
    return text[:idx].strip()


def normalize_generated_for_validation(dataset: str, text: str, structure_card: dict[str, Any]) -> str:
    cleaned = clean_generated_text(text)
    if dataset == "pubmed":
        cleaned = truncate_to_substring(cleaned, "Question:", 2)
    expected_paragraphs = int(structure_card.get("num_paragraphs") or 0)
    if dataset != "pubmed" and expected_paragraphs == 1:
        cleaned = re.sub(r"\s*\n\s*\n\s*", " ", cleaned)
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
    if dataset == "wp":
        cleaned = cleaned.replace("—", ",").replace("–", ",")
    return cleaned.strip()


def postprocess_pair(dataset: str, original: str, sampled: str, *, min_words: int = 1) -> tuple[str, str]:
    sampled = clean_generated_text(sampled)
    if dataset == "pubmed":
        sampled = truncate_to_substring(sampled, "Question:", 2)
    original_result = sentence_safe_postprocess_text(clean_generated_text(original), min_words=min_words)
    sampled_result = sentence_safe_postprocess_text(sampled, min_words=min_words)
    return original_result.text, sampled_result.text


def validate_raw_data(raw_data: dict[str, list[str]], sample_count: int, dataset: str | None = None) -> None:
    if sorted(raw_data.keys()) != ["original", "sampled"]:
        raise ValueError("Raw data must contain only original and sampled keys")
    if len(raw_data["original"]) != sample_count:
        raise ValueError("Unexpected original count")
    if len(raw_data["sampled"]) != sample_count:
        raise ValueError("Unexpected sampled count")
    for idx, (original, sampled) in enumerate(zip(raw_data["original"], raw_data["sampled"])):
        if not original.strip() or not sampled.strip():
            raise ValueError(f"Empty text at index {idx}")
        if dataset in LEGACY_DATASETS and len(original.split()) != len(sampled.split()):
            raise ValueError(f"Word-count mismatch at index {idx}")


def min_words_for_dataset(dataset: str, args: argparse.Namespace) -> int:
    if dataset == "pubmed":
        return args.pubmed_min_words
    domain_specs = getattr(args, "domain_specs", {}) or {}
    domain_spec = domain_specs.get(dataset) or {}
    return int(domain_spec.get("min_words") or args.min_words)


def max_tokens_for_generation(dataset: str, args: argparse.Namespace) -> int:
    if args.provider == "qwen" and dataset == "wp":
        return min(args.max_tokens, args.qwen_wp_max_tokens)
    return args.max_tokens


def log_progress(args: argparse.Namespace, message: str) -> None:
    if args.verbose_progress:
        print(message, flush=True)


async def generate_for_dataset(
    provider: ProviderClient,
    *,
    dataset: str,
    model: str,
    model_label: str,
    args: argparse.Namespace,
) -> bool:
    model_slot = model_slot_for(args.provider, dataset, model_label, args.pubmed_model_slot)
    paths = output_paths(args.output_root, model_label, dataset)
    metadata = build_metadata(args, dataset, model, model_label, model_slot, paths)
    if paths.raw_data.exists():
        try:
            existing_raw_data = json.loads(paths.raw_data.read_text(encoding="utf-8"))
            validate_raw_data(existing_raw_data, args.sample_count, dataset)
            print(f"[done] {dataset}: {paths.raw_data} (existing)")
            return True
        except Exception as exc:
            print(f"[resume] {dataset}: existing raw data is not reusable: {short_error(exc)}", flush=True)
    state, fresh = load_state(paths.state, metadata)
    initialize_logs(paths, fresh)
    config = build_config(args, model)

    structure_records = load_shard_records(args.shard_root, dataset, model_slot, "structure")
    fact_records = load_shard_records(args.shard_root, dataset, model_slot, "fact")
    max_inputs = min(args.max_candidates, len(structure_records), len(fact_records))
    structure_records = structure_records[:max_inputs]
    fact_records = fact_records[:max_inputs]

    profile = build_shortcut_profile([record.text for record in structure_records])
    write_json(paths.profile, profile)
    structure_cards = cached_or_build_structure_cards(paths, structure_records, config)
    fact_cards = await cached_or_build_fact_cards(paths, provider, fact_records, dataset, config, args)
    if dataset in SAME_SOURCE_PAIRING_DATASETS:
        pairs = pair_same_source_cards(
            structure_cards,
            fact_cards,
            structure_records,
            fact_records,
            limit=args.max_candidates,
        )
    else:
        pairs = pair_cards(
            structure_cards,
            fact_cards,
            structure_records,
            fact_records,
            limit=args.max_candidates,
            seed=args.seed,
        )
    write_pairs(paths.pairs, pairs)

    accepted: list[dict[str, Any]] = list(state.get("accepted", []))
    attempted = set(state.get("attempted_pair_ids", []))
    accepted_pair_ids = {record.get("pair_id") for record in accepted}
    accepted_pair_keys = {
        (record.get("structure_id"), record.get("fact_id"))
        for record in accepted
        if record.get("structure_id") and record.get("fact_id")
    }
    print(f"[dataset] {dataset}: target={args.sample_count} accepted={len(accepted)} candidates={len(pairs)}")

    for pair in pairs:
        if len(accepted) >= args.sample_count:
            break
        pair_key = (pair.structure["structure_id"], pair.fact["fact_id"])
        if pair_key in accepted_pair_keys:
            continue
        retry_failed_pair = args.retry_failed_pairs and pair.pair_id in attempted and pair.pair_id not in accepted_pair_ids
        if pair.pair_id in attempted and not retry_failed_pair:
            continue
        log_progress(args, f"  [pair] {dataset}:{pair.pair_id} start accepted={len(accepted)}/{args.sample_count}")
        accepted_record = await try_generate_pair(provider, pair, dataset, profile, args, paths)
        attempted.add(pair.pair_id)
        if accepted_record:
            accepted.append(accepted_record)
            accepted_pair_ids.add(pair.pair_id)
            accepted_pair_keys.add(pair_key)
            append_jsonl(accepted_record, paths.accepted)
            print(f"  [accepted] {dataset}:{pair.pair_id} {len(accepted)}/{args.sample_count}", flush=True)
        else:
            log_progress(args, f"  [skipped] {dataset}:{pair.pair_id} accepted={len(accepted)}/{args.sample_count}")
        state["accepted"] = accepted
        state["attempted_pair_ids"] = sorted(attempted)
        save_state(paths.state, state)

    summary = {
        "dataset": dataset,
        "model": model,
        "model_label": model_label,
        "target": args.sample_count,
        "accepted": len(accepted),
        "attempted": len(attempted),
        "candidate_count": len(pairs),
        "data_written": len(accepted) >= args.sample_count,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(paths.summary, summary)
    if len(accepted) < args.sample_count:
        print(f"[incomplete] {dataset}: accepted={len(accepted)} target={args.sample_count}", file=sys.stderr)
        return False

    final_records = accepted[: args.sample_count]
    raw_data = {
        "original": [record["original"] for record in final_records],
        "sampled": [record["sampled"] for record in final_records],
    }
    validate_raw_data(raw_data, args.sample_count, dataset)
    write_json(paths.raw_data, raw_data)
    write_json(paths.args, metadata)
    print(f"[done] {dataset}: {paths.raw_data}")
    return True


async def try_generate_pair(
    provider: ProviderClient,
    pair: CandidatePair,
    dataset: str,
    profile: dict[str, Any],
    args: argparse.Namespace,
    paths: OutputPaths,
) -> dict[str, Any] | None:
    fail_reasons: list[str] = []
    template_ids = list(args.template_ids)
    last_result: Any = None
    last_text = ""
    for attempt in range(1, args.max_retries_per_pair + 1):
        template_id = template_ids[(attempt - 1) % len(template_ids)]
        try:
            log_progress(args, f"    [generate] {dataset}:{pair.pair_id} attempt={attempt}/{args.max_retries_per_pair}")
            response = await provider.generate_text(
                generation_messages(
                    dataset,
                    pair.structure,
                    pair.fact,
                    template_id=template_id,
                    fail_reasons=fail_reasons or None,
                ),
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=max_tokens_for_generation(dataset, args),
                thinking_budget=args.gemini_generation_thinking_budget if args.provider == "gemini" else None,
            )
            generated = normalize_generated_for_validation(dataset, response.text, pair.structure)
            result = validate_generation(
                generated,
                dataset=dataset,
                structure_card=pair.structure,
                structure_source_text=pair.structure_source.text,
                fact_source_text=pair.fact_source.text,
                profile=profile,
                min_words=min_words_for_dataset(dataset, args),
                attempt=attempt,
                max_attempts=args.max_retries_per_pair,
            )
            last_result = result
            last_text = generated
            append_jsonl(
                {
                    "pair_id": pair.pair_id,
                    "attempt": attempt,
                    "accepted": result.accepted,
                    "should_retry": result.should_retry,
                    "hard_failures": result.hard_failures,
                    "soft_flags": result.soft_flags,
                    "metrics": result.metrics,
                    "usage": response.usage,
                },
                paths.metrics,
            )
            if result.accepted:
                log_progress(
                    args,
                    (
                        f"    [validation] {dataset}:{pair.pair_id} attempt={attempt} accepted "
                        f"soft={len(result.soft_flags)} hard={len(result.hard_failures)}"
                    ),
                )
                original_out, sampled_out = postprocess_pair(
                    dataset,
                    pair.structure_source.text,
                    generated,
                    min_words=min_words_for_dataset(dataset, args),
                )
                return {
                    "pair_id": pair.pair_id,
                    "dataset": dataset,
                    "original": original_out,
                    "sampled": sampled_out,
                    "structure_id": pair.structure["structure_id"],
                    "fact_id": pair.fact["fact_id"],
                    "structure_source_index": pair.structure_source.source_index,
                    "fact_source_index": pair.fact_source.source_index,
                    "structure_source_id": pair.structure_source.source_id,
                    "fact_source_id": pair.fact_source.source_id,
                    "attempt": attempt,
                    "metrics": result.metrics,
                    "matching_info": pair.matching_info,
                }
            if result.should_retry:
                log_progress(
                    args,
                    (
                        f"    [validation] {dataset}:{pair.pair_id} attempt={attempt} retry "
                        f"soft={len(result.soft_flags)} hard={len(result.hard_failures)} "
                        f"reasons={format_reasons(result.soft_flags, result.hard_failures)}"
                    ),
                )
                fail_reasons = result.hard_failures + result.soft_flags + retry_guidance(result)
                continue
            log_progress(
                args,
                (
                    f"    [validation] {dataset}:{pair.pair_id} attempt={attempt} rejected "
                    f"soft={len(result.soft_flags)} hard={len(result.hard_failures)} "
                    f"reasons={format_reasons(result.soft_flags, result.hard_failures)}"
                ),
            )
            break
        except Exception as exc:
            log_progress(args, f"    [error] {dataset}:{pair.pair_id} attempt={attempt} {short_error(exc)}")
            append_jsonl(
                {
                    "pair_id": pair.pair_id,
                    "attempt": attempt,
                    "error": str(exc),
                },
                paths.metrics,
            )
            fail_reasons = [str(exc)]
            if attempt >= args.max_retries_per_pair:
                break
    append_jsonl(
        {
            "pair_id": pair.pair_id,
            "dataset": dataset,
            "hard_failures": getattr(last_result, "hard_failures", []),
            "soft_flags": getattr(last_result, "soft_flags", []),
            "last_text_preview": last_text[:500],
            "structure_id": pair.structure["structure_id"],
            "fact_id": pair.fact["fact_id"],
        },
        paths.failures,
    )
    return None


def short_error(exc: Exception, max_chars: int = 220) -> str:
    parts = [f"{type(exc).__name__}: {exc}"]
    cause = getattr(exc, "__cause__", None)
    context = getattr(exc, "__context__", None)
    if cause is not None:
        parts.append(f"cause={type(cause).__name__}: {cause}")
    if context is not None and context is not cause:
        parts.append(f"context={type(context).__name__}: {context}")
    text = " | ".join(parts).replace("\n", " ")
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


def format_reasons(soft_flags: list[str], hard_failures: list[str], max_items: int = 3) -> str:
    reasons = hard_failures + soft_flags
    if not reasons:
        return "none"
    shown = reasons[:max_items]
    suffix = f",+{len(reasons) - max_items}" if len(reasons) > max_items else ""
    return ";".join(shown) + suffix


def build_metadata(
    args: argparse.Namespace,
    dataset: str,
    model: str,
    model_label: str,
    model_slot: str,
    paths: OutputPaths,
) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": args.provider,
        "model": model,
        "model_label": model_label,
        "dataset": dataset,
        "model_slot": model_slot,
        "sample_count": args.sample_count,
        "max_candidates": args.max_candidates,
        "shard_root": str(args.shard_root),
        "output_data_file": str(paths.raw_data),
        "fact_extractor": args.fact_extractor,
        "openai_api_base": args.openai_api_base,
        "openai_reasoning_effort": args.openai_reasoning_effort,
        "openai_verbosity": args.openai_verbosity,
        "deepseek_api_base": args.deepseek_api_base,
        "deepseek_thinking": args.deepseek_thinking,
        "deepseek_reasoning_effort": args.deepseek_reasoning_effort,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "effective_max_tokens": max_tokens_for_generation(dataset, args),
        "qwen_wp_max_tokens": args.qwen_wp_max_tokens,
        "fact_temperature": args.fact_temperature,
        "fact_max_tokens": args.fact_max_tokens,
        "gemini_thinking_budget": args.gemini_thinking_budget,
        "gemini_fact_thinking_budget": args.gemini_fact_thinking_budget,
        "gemini_generation_thinking_budget": args.gemini_generation_thinking_budget,
        "max_retries_per_pair": args.max_retries_per_pair,
        "min_words": args.min_words,
        "effective_min_words": min_words_for_dataset(dataset, args),
        "pubmed_min_words": args.pubmed_min_words,
        "pubmed_model_slot": args.pubmed_model_slot,
        "seed": args.seed,
        "template_ids": list(args.template_ids),
        "format": "fast_detect_gpt_raw_data_original_sampled",
    }


def json_object_or_none(value: str) -> dict[str, Any] | None:
    if value.lower() in {"", "none", "null"}:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("Expected a JSON object")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["gemini", "claude", "qwen", "openai", "deepseek"], required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-label", default=None)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), choices=list(DEFAULT_DATASETS))
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument("--shard-root", type=Path, default=DEFAULT_SHARD_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--datasets-config", type=Path, default=REPO_ROOT / "configs" / "datasets.yaml")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-base", default=DEFAULT_QWEN_API_BASE)
    parser.add_argument("--openai-api-base", default=None)
    parser.add_argument("--openai-reasoning-effort", default=None)
    parser.add_argument("--openai-verbosity", default=None)
    parser.add_argument("--deepseek-api-base", default=DEFAULT_DEEPSEEK_API_BASE)
    parser.add_argument("--deepseek-thinking", choices=["enabled", "disabled", "none"], default="disabled")
    parser.add_argument("--deepseek-reasoning-effort", default=None)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument(
        "--chat-template-kwargs",
        type=json_object_or_none,
        default=DEFAULT_QWEN_CHAT_TEMPLATE_KWARGS,
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--qwen-wp-max-tokens",
        type=int,
        default=DEFAULT_QWEN_WP_MAX_TOKENS,
        help="Cap Qwen WP generation tokens to stay under the 4096-token vLLM context limit.",
    )
    parser.add_argument(
        "--gemini-thinking-budget",
        type=int,
        default=512,
        help="Default Gemini thinking budget when a task-specific Gemini budget is not set.",
    )
    parser.add_argument(
        "--gemini-fact-thinking-budget",
        type=int,
        default=512,
        help="Gemini thinking budget for fact-card extraction and repair calls.",
    )
    parser.add_argument(
        "--gemini-generation-thinking-budget",
        type=int,
        default=768,
        help="Gemini thinking budget for article/passage generation calls.",
    )
    parser.add_argument("--fact-temperature", type=float, default=0.0)
    parser.add_argument("--fact-max-tokens", type=int, default=DEFAULT_FACT_MAX_TOKENS)
    parser.add_argument("--fact-extractor", choices=["hybrid", "llm", "rule_based"], default="hybrid")
    parser.add_argument("--max-retries-per-pair", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--min-words", type=int, default=DEFAULT_MIN_WORDS)
    parser.add_argument("--pubmed-min-words", type=int, default=DEFAULT_PUBMED_MIN_WORDS)
    parser.add_argument("--pubmed-model-slot", choices=["model1", "model2", "model3"], default=None)
    parser.add_argument(
        "--retry-failed-pairs",
        action="store_true",
        help="When resuming, retry previously attempted pairs that are not already accepted.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--template-ids", nargs="+", default=list(DEFAULT_TEMPLATE_IDS))
    parser.add_argument("--skip-server-check", action="store_true")
    parser.add_argument(
        "--verbose-progress",
        action="store_true",
        help="Print per-pair generation, validation, retry, and error progress.",
    )
    return parser.parse_args(argv)


async def build_provider(args: argparse.Namespace, model: str) -> ProviderClient:
    if args.provider == "gemini":
        if args.api_key:
            raise ValueError("Gemini uses Vertex AI only. Remove --api-key and set GOOGLE_CLOUD_PROJECT.")
        return GeminiProviderClient(model, thinking_budget=args.gemini_thinking_budget)
    if args.provider == "claude":
        return ClaudeProviderClient(model, api_key=args.api_key)
    if args.provider == "qwen":
        provider = QwenProviderClient(
            model=model,
            api_base=args.api_base,
            api_key=args.api_key or os.environ.get("VLLM_API_KEY", ""),
            timeout_seconds=args.timeout_seconds,
            chat_template_kwargs=args.chat_template_kwargs,
        )
        if not args.skip_server_check:
            await provider.check_server()
        return provider
    if args.provider == "openai":
        return OpenAIProviderClient(
            model,
            api_key=args.api_key,
            base_url=args.openai_api_base,
            timeout_seconds=args.timeout_seconds,
            reasoning_effort=args.openai_reasoning_effort,
            verbosity=args.openai_verbosity,
        )
    if args.provider == "deepseek":
        return DeepSeekProviderClient(
            model,
            api_key=args.api_key,
            base_url=args.deepseek_api_base,
            timeout_seconds=args.timeout_seconds,
            thinking=None if args.deepseek_thinking == "none" else args.deepseek_thinking,
            reasoning_effort=args.deepseek_reasoning_effort,
        )
    raise ValueError(f"Unsupported provider: {args.provider}")


async def async_main(args: argparse.Namespace) -> None:
    started = time.time()
    random.seed(args.seed)
    model = normalize_model_for_provider(args.provider, args.model or PROVIDER_DEFAULT_MODELS[args.provider])
    label = model_label_for(args.provider, model, args.model_label)
    provider = await build_provider(args, model)
    results = []
    for dataset in args.datasets:
        result = await generate_for_dataset(provider, dataset=dataset, model=model, model_label=label, args=args)
        results.append(result)
    elapsed = time.time() - started
    print(f"[elapsed] {elapsed:.1f}s")
    if not all(results):
        raise SystemExit(1)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.domain_specs = load_domain_specs(args.datasets_config)
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()

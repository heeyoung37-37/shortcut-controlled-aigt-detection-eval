from __future__ import annotations

import argparse
import math
import re
from typing import Any

from tqdm import tqdm

from .config import data_dir, load_config, split_alias
from .io_utils import read_jsonl, write_jsonl
from .text_stats import (
    contains_direct_quote,
    extract_numbers,
    split_paragraphs,
    split_sentences,
    word_count,
)

ATTRIBUTION_RE = re.compile(r"\b(said|told|according|reported|claimed|stated|added)\b", re.IGNORECASE)
STRUCTURE_CARD_SCHEMA_VERSION = "structure_card_v3_boosted_targets_no_ranges"


def extract_structure_card(record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    text = record["text"]
    paragraphs = split_paragraphs(text)
    source_wc = word_count(text)
    sc_config = config["structure_card"]
    length_boost = float(sc_config.get("length_boost_multiplier", 1.45))
    max_sentences = int(sc_config.get("max_sentences_per_paragraph", 3))

    card_paragraphs: list[dict[str, Any]] = []
    num_paragraphs = len(paragraphs)
    for idx, paragraph in enumerate(paragraphs, 1):
        paragraph_wc = word_count(paragraph)
        sentences = split_sentences(paragraph)
        boosted_paragraph_wc = _boost_count(paragraph_wc, length_boost)
        sentence_word_targets = [_boost_count(word_count(sentence), length_boost) for sentence in sentences]
        quote_slot = contains_direct_quote(paragraph)
        number_slot = bool(extract_numbers(paragraph))
        card_paragraphs.append(
            {
                "pid": idx,
                "sentence_count": len(sentences),
                "target_word_count": boosted_paragraph_wc,
                "sentence_word_targets": sentence_word_targets,
                "length_bucket": _length_bucket(paragraph_wc),
                "role": _role_for_paragraph(
                    pid=idx,
                    num_paragraphs=num_paragraphs,
                    paragraph=paragraph,
                    quote_slot=quote_slot,
                    number_slot=number_slot,
                ),
                "quote_slot": quote_slot,
                "number_slot": number_slot,
            }
        )

    return {
        "structure_id": record["id"],
        "schema_version": STRUCTURE_CARD_SCHEMA_VERSION,
        "split": record["split"],
        "source_word_count": source_wc,
        "length_boost_multiplier": length_boost,
        "target_total_word_count": _boost_count(source_wc, length_boost),
        "num_paragraphs": num_paragraphs,
        "global_constraints": {
            "max_sentences_per_paragraph": max_sentences,
            "avoid_transition_words": sc_config.get("avoid_transition_words", []),
            "avoid_paragraph_initial_words": sc_config.get("avoid_paragraph_initial_words", []),
            "forbid_em_dash": bool(sc_config.get("forbid_em_dash", True)),
            "forbid_generic_forward_looking_ending": True,
            "length_matching_priority": (
                "Target word counts are intentionally generous to counter short generations. "
                "Aim for target_total_word_count and paragraph target_word_count without using word ranges."
            ),
        },
        "paragraphs": card_paragraphs,
    }


def extract_structure_cards_for_split(config: dict[str, Any], split: str, limit: int | None = None) -> list[dict[str, Any]]:
    split = split_alias(config, split)
    pool_path = data_dir(config, "processed_dir") / "pools" / f"{split}_structure_pool.jsonl"
    records = read_jsonl(pool_path)
    effective_limit = limit or config.get("_runtime_limit")
    if effective_limit:
        records = records[: int(effective_limit)]
    cards = [extract_structure_card(record, config) for record in tqdm(records, desc=f"structure:{split}")]
    out_path = data_dir(config, "processed_dir") / "structure_cards" / f"{split}.jsonl"
    write_jsonl(cards, out_path)
    return cards


def _boost_count(value: int, multiplier: float) -> int:
    return max(1, int(math.ceil(value * multiplier)))


def _length_bucket(word_count_value: int) -> str:
    if word_count_value < 12:
        return "very_short"
    if word_count_value < 20:
        return "short"
    if word_count_value < 35:
        return "medium"
    return "long"


def _role_for_paragraph(
    pid: int,
    num_paragraphs: int,
    paragraph: str,
    quote_slot: bool,
    number_slot: bool,
) -> str:
    if pid == 1:
        return "lead"
    if quote_slot:
        return "quote_or_reaction"
    if ATTRIBUTION_RE.search(paragraph):
        return "attribution_or_claim"
    if number_slot and pid <= max(2, int(num_paragraphs * 0.7)):
        return "chronology_or_numeric_detail"
    if pid > max(1, num_paragraphs - 2):
        return "closing_context"
    if pid <= max(2, int(num_paragraphs * 0.3)):
        return "main_event_detail"
    if pid <= max(2, int(num_paragraphs * 0.75)):
        return "background_context"
    return "news_detail"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--split", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    extract_structure_cards_for_split(load_config(args.config), args.split, limit=args.limit)


if __name__ == "__main__":
    main()

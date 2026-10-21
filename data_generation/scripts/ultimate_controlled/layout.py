from __future__ import annotations

from copy import deepcopy
from typing import Any


def effective_structure_card(dataset: str, structure_card: dict[str, Any]) -> dict[str, Any]:
    """Return a generation/validation layout that avoids source line-wrap traps."""
    card = deepcopy(structure_card)
    raw_paragraphs = card.get("paragraphs") or []
    raw_count = int(card.get("num_paragraphs") or len(raw_paragraphs) or 1)
    source_words = int(card.get("source_word_count") or 0)

    if dataset == "arxiv_abstract":
        target_words = _bounded_words(card, default=180, low=90, high=260)
        return _synthetic_card(card, paragraph_count=1, target_total_word_count=target_words, sentence_count=5)

    if dataset == "meeting_summary":
        target_words = min(max(_unboosted_target_words(card, default=110), 70), 260)
        if raw_count <= 3 and raw_paragraphs:
            return _resize_target_words(card, target_total_word_count=target_words)
        sentence_count = min(max(round(target_words / 45), 1), 4)
        return _synthetic_card(card, paragraph_count=1, target_total_word_count=target_words, sentence_count=sentence_count)

    if dataset == "translation":
        target_words = min(max(source_words or _unboosted_target_words(card, default=120), 60), 240)
        if raw_count <= 4:
            return _resize_target_words(card, target_total_word_count=target_words)
        paragraph_count = min(max(round(target_words / 80), 1), 4)
        return _synthetic_card(card, paragraph_count=paragraph_count, target_total_word_count=target_words, sentence_count=3)

    if dataset == "essay":
        paragraph_count = min(raw_count, 4) if raw_count > 1 else 1
        if source_words >= 130:
            paragraph_count = max(paragraph_count, 3)
        target_words = _bounded_words(card, default=180, low=90, high=280)
        return _synthetic_card(card, paragraph_count=paragraph_count, target_total_word_count=target_words, sentence_count=3)

    if dataset == "blog":
        if raw_count <= 18 and source_words <= 1600:
            return card
        paragraph_count = _blog_paragraph_count(source_words, raw_count)
        target_words = _bounded_words(card, default=720, low=360, high=1100)
        return _synthetic_card(card, paragraph_count=paragraph_count, target_total_word_count=target_words, sentence_count=2)

    if dataset == "product_review":
        target_words = min(max(_unboosted_target_words(card, default=120), 70), 320)
        sentence_count = min(max(round(target_words / 35), 1), 4)
        return _synthetic_card(
            card,
            paragraph_count=1,
            target_total_word_count=target_words,
            sentence_count=sentence_count,
        )

    return card


def effective_expected_paragraphs(dataset: str, structure_card: dict[str, Any]) -> int:
    card = effective_structure_card(dataset, structure_card)
    return int(card.get("num_paragraphs") or 0)


def _blog_paragraph_count(source_words: int, raw_count: int) -> int:
    if source_words <= 0:
        return min(max(raw_count, 4), 18)
    by_length = round(source_words / 85)
    target = max(6, by_length)
    if raw_count >= 40:
        target = max(target, 12)
    return min(target, 18)


def _bounded_words(card: dict[str, Any], *, default: int, low: int, high: int) -> int:
    value = int(card.get("target_total_word_count") or card.get("source_word_count") or default)
    return max(low, min(high, value))


def _unboosted_target_words(card: dict[str, Any], *, default: int) -> int:
    value = int(card.get("target_total_word_count") or card.get("source_word_count") or default)
    boost = float(card.get("length_boost_multiplier") or 1.0)
    if boost > 0:
        return round(value / boost)
    return value


def _resize_target_words(card: dict[str, Any], *, target_total_word_count: int) -> dict[str, Any]:
    output = deepcopy(card)
    output["length_boost_multiplier"] = 1.0
    output["target_total_word_count"] = target_total_word_count
    global_constraints = dict(output.get("global_constraints") or {})
    global_constraints["length_matching_priority"] = (
        "Match the target word count closely; translation should not be expanded beyond the source structure."
    )
    output["global_constraints"] = global_constraints

    paragraphs = output.get("paragraphs") or []
    if not paragraphs:
        return output

    old_total = sum(int(paragraph.get("target_word_count") or 0) for paragraph in paragraphs)
    if old_total <= 0:
        old_total = len(paragraphs)
    assigned = 0
    for idx, paragraph in enumerate(paragraphs):
        if idx == len(paragraphs) - 1:
            new_target = max(1, target_total_word_count - assigned)
        else:
            old_target = int(paragraph.get("target_word_count") or 1)
            new_target = max(1, round(target_total_word_count * old_target / old_total))
            assigned += new_target
        old_sentence_targets = [int(value) for value in paragraph.get("sentence_word_targets") or [] if int(value) > 0]
        paragraph["target_word_count"] = new_target
        if old_sentence_targets:
            sentence_total = sum(old_sentence_targets)
            sentence_assigned = 0
            resized = []
            for sentence_idx, sentence_target in enumerate(old_sentence_targets):
                if sentence_idx == len(old_sentence_targets) - 1:
                    resized.append(max(1, new_target - sentence_assigned))
                else:
                    value = max(1, round(new_target * sentence_target / sentence_total))
                    resized.append(value)
                    sentence_assigned += value
            paragraph["sentence_word_targets"] = resized
    return output


def _synthetic_card(
    card: dict[str, Any],
    *,
    paragraph_count: int,
    target_total_word_count: int,
    sentence_count: int,
) -> dict[str, Any]:
    paragraph_count = max(1, paragraph_count)
    paragraph_words = max(1, round(target_total_word_count / paragraph_count))
    paragraphs = []
    for idx in range(1, paragraph_count + 1):
        paragraphs.append(
            {
                "pid": idx,
                "sentence_count": sentence_count,
                "target_word_count": paragraph_words,
                "sentence_word_targets": [max(1, round(paragraph_words / sentence_count))] * sentence_count,
                "length_bucket": "medium" if paragraph_words < 70 else "long",
                "role": _paragraph_role(idx, paragraph_count),
                "quote_slot": False,
                "number_slot": False,
            }
        )
    output = dict(card)
    output["num_paragraphs"] = paragraph_count
    output["target_total_word_count"] = target_total_word_count
    output["paragraphs"] = paragraphs
    output["layout_adjustment"] = "domain_effective_layout_v1"
    return output


def _paragraph_role(idx: int, paragraph_count: int) -> str:
    if idx == 1:
        return "lead"
    if idx >= paragraph_count - 1:
        return "closing_context"
    if idx <= max(2, paragraph_count // 3):
        return "main_detail"
    return "supporting_detail"

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from analyze_aigt import (
    ALPHA_WORD_RE,
    CLICHE_PHRASES,
    DISCOURSE,
    TRANSITION_PHRASES,
    VAGUE_INTENSIFIERS,
    WORD_RE,
    count_phrase_hits,
    phrase_normalized_text,
    split_paragraphs,
    split_sentences,
)
from grok_ultimate_gen.src.text_stats import longest_common_word_sequence_len, ngram_overlap_ratio

from .layout import effective_expected_paragraphs, effective_structure_card


FOCUSED_FEATURES = [
    "transition_words_per_100w",
    "cliche_phrases_per_100w",
    "vague_intensifiers_per_100w",
    "paragraph_count",
    "avg_paragraph_len_words",
    "numbers_per_100w",
    "em_dashes_per_100w",
    "commas_per_100w",
    "sentence_length_cv",
]


@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    should_retry: bool
    hard_failures: list[str]
    soft_flags: list[str]
    metrics: dict[str, Any]


def extract_shortcut_features(text: str) -> dict[str, float]:
    words = WORD_RE.findall(text or "")
    word_count = len(words)
    lower_words = [word.lower() for word in ALPHA_WORD_RE.findall(_normalize_quotes(text))]
    text_lower = (text or "").lower()
    phrase_lower = phrase_normalized_text(text or "")
    paragraphs = split_paragraphs(text or "")
    paragraph_lengths = [len(WORD_RE.findall(paragraph)) for paragraph in paragraphs]
    sentences = split_sentences(text or "")
    sentence_lengths = [len(WORD_RE.findall(sentence)) for sentence in sentences]
    transition_words = sum(1 for word in lower_words if word in DISCOURSE)
    transition_phrases = count_phrase_hits(text_lower, TRANSITION_PHRASES)
    cliche_phrases = count_phrase_hits(phrase_lower, CLICHE_PHRASES)
    vague_intensifiers = sum(1 for word in lower_words if word in VAGUE_INTENSIFIERS)
    numbers = sum(1 for token in words if token and token[0].isdigit())
    em_dashes = (text or "").count("—") + (text or "").count("–")
    return {
        "transition_words_per_100w": _per_100(transition_words + transition_phrases, word_count),
        "cliche_phrases_per_100w": _per_100(cliche_phrases, word_count),
        "vague_intensifiers_per_100w": _per_100(vague_intensifiers, word_count),
        "paragraph_count": float(len(paragraphs)),
        "avg_paragraph_len_words": _mean(paragraph_lengths),
        "numbers_per_100w": _per_100(numbers, word_count),
        "em_dashes_per_100w": _per_100(em_dashes, word_count),
        "commas_per_100w": _per_100((text or "").count(","), word_count),
        "sentence_length_cv": _sentence_length_cv(sentence_lengths),
        "_word_count": float(word_count),
        "_em_dash_count": float(em_dashes),
    }


def build_shortcut_profile(texts: list[str]) -> dict[str, Any]:
    rows = [extract_shortcut_features(text) for text in texts if str(text).strip()]
    profile: dict[str, Any] = {"count": len(rows), "features": {}, "thresholds": {}}
    for feature in FOCUSED_FEATURES:
        values = [float(row.get(feature, 0.0)) for row in rows]
        stats = {
            "p05": _quantile(values, 0.05),
            "median": _quantile(values, 0.50),
            "p90": _quantile(values, 0.90),
            "p95": _quantile(values, 0.95),
        }
        profile["features"][feature] = stats

    feature_stats = profile["features"]
    profile["thresholds"] = {
        "transition_words_per_100w_max": min(3.0, feature_stats["transition_words_per_100w"]["p90"] + 0.75),
        "cliche_phrases_per_100w_max": min(0.8, feature_stats["cliche_phrases_per_100w"]["p90"] + 0.25),
        "vague_intensifiers_per_100w_max": min(1.5, feature_stats["vague_intensifiers_per_100w"]["p90"] + 0.50),
        "numbers_per_100w_max": feature_stats["numbers_per_100w"]["p95"] + 1.0,
        "commas_per_100w_max": feature_stats["commas_per_100w"]["p95"] + 2.0,
        "sentence_length_cv_min": max(0.0, feature_stats["sentence_length_cv"]["p05"] - 0.15),
        "sentence_length_cv_max": feature_stats["sentence_length_cv"]["p95"] + 0.15,
        "avg_paragraph_len_ratio_min": 0.55,
        "avg_paragraph_len_ratio_max": 1.75,
        "baseline_avg_paragraph_len_words": feature_stats["avg_paragraph_len_words"]["median"],
    }
    return profile


def validate_generation(
    text: str,
    *,
    dataset: str,
    structure_card: dict[str, Any],
    structure_source_text: str,
    fact_source_text: str,
    profile: dict[str, Any],
    min_words: int,
    attempt: int,
    max_attempts: int,
) -> ValidationResult:
    hard: list[str] = []
    soft: list[str] = []
    stripped = _clean_output(text)
    features = extract_shortcut_features(stripped)
    word_count = int(features["_word_count"])
    metrics: dict[str, Any] = {"features": features, "word_count": word_count}

    if not stripped:
        hard.append("empty_output")
    if word_count < min_words:
        hard.append(f"too_short:{word_count}<min:{min_words}")
    if dataset == "pubmed" and not _is_pubmed_qa(stripped):
        hard.append("pubmed_format_missing_question_or_answer")
    if dataset == "translation" and _has_translation_label(stripped):
        hard.append("translation_label_or_note_present")
    if dataset == "translation" and stripped and not _has_terminal_punctuation(stripped):
        hard.append("translation_terminal_punctuation_missing")
    if dataset == "translation":
        target_words = int(effective_structure_card(dataset, structure_card).get("target_total_word_count") or 0)
        if target_words and word_count > max(round(target_words * 1.4), target_words + 60):
            hard.append(f"translation_too_long:{word_count}>target:{target_words}")

    paragraph_count = int(features["paragraph_count"])
    expected_paragraphs = effective_expected_paragraphs(dataset, structure_card)
    metrics["expected_paragraph_count"] = expected_paragraphs
    if expected_paragraphs and abs(paragraph_count - expected_paragraphs) > 2:
        hard.append(f"paragraph_count_far:{paragraph_count}!=expected:{expected_paragraphs}")
    elif expected_paragraphs and paragraph_count != expected_paragraphs:
        soft.append(f"paragraph_count_near:{paragraph_count}!=expected:{expected_paragraphs}")

    if int(features["_em_dash_count"]) >= 2:
        hard.append(f"too_many_dashes:{int(features['_em_dash_count'])}")
    elif int(features["_em_dash_count"]) == 1:
        soft.append("contains_dash")

    structure_overlap = ngram_overlap_ratio(stripped, structure_source_text, n=5, ignore_quotes=True)
    fact_overlap = ngram_overlap_ratio(stripped, fact_source_text, n=5, ignore_quotes=True)
    fact_longest_copy = longest_common_word_sequence_len(stripped, fact_source_text, ignore_quotes=True)
    metrics.update(
        {
            "structure_source_5gram_overlap_ratio": structure_overlap,
            "fact_source_5gram_overlap_ratio": fact_overlap,
            "fact_source_longest_copy_words": fact_longest_copy,
        }
    )
    if dataset != "translation" and structure_overlap > 0.15:
        hard.append(f"structure_source_overlap:{structure_overlap:.4f}")
    if dataset != "translation" and fact_overlap > 0.22 and fact_longest_copy > 45:
        hard.append(f"fact_source_copy:{fact_overlap:.4f}/{fact_longest_copy}")

    thresholds = profile.get("thresholds", {})
    _add_max_flag(soft, features, thresholds, "transition_words_per_100w")
    _add_max_flag(soft, features, thresholds, "cliche_phrases_per_100w")
    _add_max_flag(soft, features, thresholds, "vague_intensifiers_per_100w")
    _add_max_flag(soft, features, thresholds, "numbers_per_100w")
    _add_max_flag(soft, features, thresholds, "commas_per_100w")
    cv_min = float(thresholds.get("sentence_length_cv_min", 0.0))
    cv_max = float(thresholds.get("sentence_length_cv_max", 999.0))
    cv = float(features["sentence_length_cv"])
    if cv < cv_min or cv > cv_max:
        soft.append(f"sentence_length_cv_out_of_range:{cv:.3f} not {cv_min:.3f}-{cv_max:.3f}")

    baseline_avg_para = float(thresholds.get("baseline_avg_paragraph_len_words") or 0.0)
    if baseline_avg_para > 0:
        ratio = float(features["avg_paragraph_len_words"]) / baseline_avg_para
        metrics["avg_paragraph_len_ratio"] = ratio
        ratio_min = float(thresholds.get("avg_paragraph_len_ratio_min", 0.55))
        ratio_max = float(thresholds.get("avg_paragraph_len_ratio_max", 1.75))
        if ratio < ratio_min or ratio > ratio_max:
            soft.append(f"avg_paragraph_len_ratio_out_of_range:{ratio:.3f}")

    final_attempt = attempt >= max_attempts
    accepted = not hard and len(soft) <= 2
    should_retry = not accepted and not hard and len(soft) >= 3 and not final_attempt
    metrics["thresholds"] = thresholds
    metrics["hard_failure_count"] = len(hard)
    metrics["soft_flag_count"] = len(soft)
    return ValidationResult(
        accepted=accepted,
        should_retry=should_retry,
        hard_failures=hard,
        soft_flags=soft,
        metrics=metrics,
    )


def retry_guidance(result: ValidationResult) -> list[str]:
    guidance: list[str] = []
    reasons = result.hard_failures + result.soft_flags
    joined = " ".join(reasons)
    if "too_short" in joined:
        guidance.append("Increase length while staying within the planned paragraph layout.")
    if "paragraph_count" in joined:
        guidance.append("Match the planned paragraph count more closely and use blank lines only between paragraphs.")
    if "pubmed_format" in joined:
        guidance.append("Use exactly one `Question:` marker and one `Answer:` marker.")
    if "translation_terminal_punctuation" in joined:
        guidance.append("End the translation with normal sentence punctuation.")
    if "translation_too_long" in joined:
        guidance.append("Keep the translation close to the planned word count and do not add explanatory wording.")
    if "transition_words" in joined:
        guidance.append("Reduce explicit discourse markers such as however, therefore, furthermore, overall, and similar connectors.")
    if "cliche_phrases" in joined:
        guidance.append("Remove stock phrases and broad essay-like cliches.")
    if "vague_intensifiers" in joined:
        guidance.append("Use fewer vague intensifiers such as very, really, clearly, highly, and significantly.")
    if "numbers_per_100w" in joined:
        guidance.append("Use only numbers that are clearly supported by FACT_CARD.")
    if "commas_per_100w" in joined:
        guidance.append("Shorten comma-heavy sentences and split clauses naturally.")
    if "sentence_length_cv" in joined:
        guidance.append("Use steadier sentence lengths without making every sentence identical.")
    if "dash" in joined:
        guidance.append("Do not use em dashes or en dashes.")
    if "source" in joined or "copy" in joined:
        guidance.append("Rephrase source wording while preserving the supported facts.")
    return guidance or ["Revise the text to better satisfy the layout and shortcut-control metrics."]


def _add_max_flag(
    soft: list[str],
    features: dict[str, float],
    thresholds: dict[str, Any],
    feature: str,
) -> None:
    threshold = float(thresholds.get(f"{feature}_max", 999.0))
    value = float(features.get(feature, 0.0))
    if value > threshold:
        soft.append(f"{feature}_high:{value:.3f}>{threshold:.3f}")


def _clean_output(text: str) -> str:
    stripped = (text or "").strip()
    stripped = re.sub(r"^```(?:text|markdown)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    stripped = re.sub(r"^\s*(Article|Output|Draft):\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\n{3,}", "\n\n", stripped)
    return stripped.strip()


def _is_pubmed_qa(text: str) -> bool:
    return text.count("Question:") == 1 and text.count("Answer:") == 1 and text.find("Question:") < text.find("Answer:")


def _has_translation_label(text: str) -> bool:
    first_line = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    return bool(re.match(r"^\s*(translation|english translation|english|translated text)\s*:", first_line, re.IGNORECASE))


def _has_terminal_punctuation(text: str) -> bool:
    return bool(re.search(r"[.!?][\"')\]]?\s*$", text or ""))


def _normalize_quotes(text: str) -> str:
    return (
        (text or "")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )


def _sentence_length_cv(sentence_lengths: list[int]) -> float:
    positive = [length for length in sentence_lengths if length > 0]
    if len(positive) < 2:
        return 0.0
    avg = _mean(positive)
    if avg <= 0:
        return 0.0
    variance = sum((length - avg) ** 2 for length in positive) / (len(positive) - 1)
    return math.sqrt(variance) / avg


def _per_100(count: float, word_count: int) -> float:
    return 100.0 * count / word_count if word_count else 0.0


def _mean(values: list[int] | list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    weight = pos - lo
    return float(ordered[lo] * (1 - weight) + ordered[hi] * weight)

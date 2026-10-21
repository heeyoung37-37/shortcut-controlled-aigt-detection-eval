from __future__ import annotations

from typing import Any

from .providers import Message
from .prompts import generation_messages
from .shortcut_validator import (
    ValidationResult,
    extract_shortcut_features,
    retry_guidance,
    validate_generation,
)


QWEN_V2_DATASETS = {"xsum", "squad", "wp", "pubmed"}


def generation_messages_qwen_v2(
    dataset: str,
    structure_card: dict[str, Any],
    fact_card: dict[str, Any],
    *,
    template_id: str = "template_01",
    fail_reasons: list[str] | None = None,
) -> list[Message]:
    messages = generation_messages(
        dataset,
        structure_card,
        fact_card,
        template_id=template_id,
        fail_reasons=fail_reasons,
    )
    messages[-1]["content"] += "\n\n" + _qwen_v2_control_block(dataset)
    return messages


def validate_generation_qwen_v2(
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
    base = validate_generation(
        text,
        dataset=dataset,
        structure_card=structure_card,
        structure_source_text=structure_source_text,
        fact_source_text=fact_source_text,
        profile=profile,
        min_words=min_words,
        attempt=attempt,
        max_attempts=max_attempts,
    )
    if dataset not in QWEN_V2_DATASETS:
        return base

    features = extract_shortcut_features(text)
    metrics = dict(base.metrics)
    metrics["features"] = features
    strict_thresholds = qwen_v2_thresholds(profile, dataset)
    metrics["qwen_v2_thresholds"] = strict_thresholds

    hard = list(base.hard_failures)
    soft = _layout_soft_flags(base.soft_flags)
    _add_max_flag(soft, features, strict_thresholds, "transition_words_per_100w")
    _add_max_flag(soft, features, strict_thresholds, "cliche_phrases_per_100w")
    _add_max_flag(soft, features, strict_thresholds, "vague_intensifiers_per_100w")
    _add_max_flag(soft, features, strict_thresholds, "numbers_per_100w")
    _add_max_flag(soft, features, strict_thresholds, "commas_per_100w")
    _add_cv_flags(soft, features, strict_thresholds)
    _add_avg_paragraph_ratio_flag(soft, features, base.metrics, strict_thresholds)

    soft_limit = 2.0 if dataset == "xsum" else 1.25
    soft_score = _soft_score(soft)
    final_attempt = attempt >= max_attempts
    accepted = not hard and soft_score <= soft_limit
    should_retry = (not accepted) and (not final_attempt) and _retryable_hard_failures(hard)
    metrics["hard_failure_count"] = len(hard)
    metrics["soft_flag_count"] = len(soft)
    metrics["qwen_v2_soft_score"] = soft_score
    metrics["qwen_v2_soft_limit"] = soft_limit
    return ValidationResult(
        accepted=accepted,
        should_retry=should_retry,
        hard_failures=hard,
        soft_flags=soft,
        metrics=metrics,
    )


def retry_guidance_qwen_v2(result: ValidationResult) -> list[str]:
    guidance = retry_guidance(result)
    joined = " ".join(result.soft_flags + result.hard_failures)
    if "qwen_v2_commas_per_100w" in joined:
        guidance.append("Use fewer commas; split comma-chained clauses into plain sentences.")
    if "qwen_v2_sentence_length_cv_low" in joined:
        guidance.append("Vary sentence lengths more naturally; mix short and medium sentences.")
    if "qwen_v2_numbers_per_100w" in joined:
        guidance.append("Remove numbers unless they are essential to the requested answer or event.")
    if "qwen_v2_avg_paragraph_len_ratio" in joined:
        guidance.append("Stay closer to the planned paragraph length; do not make paragraphs much shorter or longer than the plan.")
    if "qwen_v2_transition_words_per_100w" in joined:
        guidance.append("Avoid explicit connectors such as however, therefore, overall, and additionally.")
    if "qwen_v2_vague_intensifiers_per_100w" in joined:
        guidance.append("Remove vague intensifiers such as very, highly, clearly, really, and significantly.")
    return _dedupe(guidance)


def qwen_v2_thresholds(profile: dict[str, Any], dataset: str) -> dict[str, float]:
    features = profile.get("features", {})

    def stat(feature: str, key: str, default: float = 0.0) -> float:
        return float(features.get(feature, {}).get(key, default))

    thresholds = {
        "transition_words_per_100w_max": min(2.2, stat("transition_words_per_100w", "p90") + 0.25),
        "cliche_phrases_per_100w_max": min(0.45, stat("cliche_phrases_per_100w", "p90") + 0.10),
        "vague_intensifiers_per_100w_max": min(1.05, stat("vague_intensifiers_per_100w", "p90") + 0.15),
        "numbers_per_100w_max": stat("numbers_per_100w", "p90") + 0.30,
        "commas_per_100w_max": stat("commas_per_100w", "p90") + 0.50,
        "sentence_length_cv_min": max(
            stat("sentence_length_cv", "p05"),
            stat("sentence_length_cv", "median") * 0.65,
        ),
        "sentence_length_cv_max": stat("sentence_length_cv", "p95") + 0.10,
        "avg_paragraph_len_ratio_min": 0.65,
        "avg_paragraph_len_ratio_max": 1.55,
    }
    if dataset == "squad":
        thresholds["numbers_per_100w_max"] = min(
            thresholds["numbers_per_100w_max"],
            stat("numbers_per_100w", "median") + 1.50,
        )
        thresholds["sentence_length_cv_min"] = max(
            thresholds["sentence_length_cv_min"],
            stat("sentence_length_cv", "median") * 0.70,
        )
    elif dataset == "wp":
        thresholds["commas_per_100w_max"] = stat("commas_per_100w", "p90") + 0.25
        thresholds["sentence_length_cv_min"] = max(
            thresholds["sentence_length_cv_min"],
            stat("sentence_length_cv", "median") * 0.75,
        )
        thresholds["avg_paragraph_len_ratio_min"] = 0.75
        thresholds["avg_paragraph_len_ratio_max"] = 1.35
    elif dataset == "pubmed":
        thresholds["numbers_per_100w_max"] = min(
            thresholds["numbers_per_100w_max"],
            stat("numbers_per_100w", "p90") + 0.50,
        )
        thresholds["commas_per_100w_max"] = stat("commas_per_100w", "p90") + 0.25
        thresholds["sentence_length_cv_min"] = max(
            thresholds["sentence_length_cv_min"],
            stat("sentence_length_cv", "median") * 0.70,
        )
    return thresholds


def _qwen_v2_control_block(dataset: str) -> str:
    shared = """Qwen v2 writing rules. Follow these as output constraints, not optional advice:
- Use no em dashes or en dashes.
- Avoid stock transitions: however, therefore, furthermore, additionally, overall, in conclusion.
- Avoid vague intensifiers: very, really, clearly, highly, significantly, remarkably.
- Use numbers only when essential to the FACT_CARD; omit incidental dates, counts, and percentages.
- Keep punctuation plain. If a sentence has two or more commas, split it into shorter sentences.
- Mix short and medium sentences. Avoid a sequence of same-length sentences.
- Do a final silent edit before answering: remove dash punctuation, comma chains, stock transitions, and nonessential numbers.
- Return only the requested text."""
    dataset_notes = {
        "squad": (
            "For SQuAD, write a clean explanatory paragraph. Prefer definition and relation sentences. "
            "Use at most one essential number unless the FACT_CARD cannot be understood without more."
        ),
        "wp": (
            "For WP, write concrete narrative action in plain prose. Avoid lyrical lists, scene-break formatting, "
            "dialogue-heavy formatting, and comma-heavy description. Use one continuous paragraph when planned."
        ),
        "pubmed": (
            "For PubMed, keep exactly one Question and one Answer. The Answer should be concise, biomedical, "
            "and low-hedging. Avoid extra statistics unless central to the FACT_CARD."
        ),
        "xsum": "For XSum, keep the report direct, compact, and low-transition.",
    }
    return shared + "\n- " + dataset_notes.get(dataset, "")


def _layout_soft_flags(flags: list[str]) -> list[str]:
    prefixes = (
        "paragraph_count_near",
        "contains_dash",
    )
    return [flag for flag in flags if flag.startswith(prefixes)]


def _add_max_flag(
    soft: list[str],
    features: dict[str, float],
    thresholds: dict[str, float],
    feature: str,
) -> None:
    value = float(features.get(feature, 0.0))
    threshold = float(thresholds[f"{feature}_max"])
    if value > threshold:
        soft.append(f"qwen_v2_{feature}_high:{value:.3f}>{threshold:.3f}")


def _add_cv_flags(
    soft: list[str],
    features: dict[str, float],
    thresholds: dict[str, float],
) -> None:
    value = float(features.get("sentence_length_cv", 0.0))
    min_value = float(thresholds["sentence_length_cv_min"])
    max_value = float(thresholds["sentence_length_cv_max"])
    if value < min_value:
        soft.append(f"qwen_v2_sentence_length_cv_low:{value:.3f}<{min_value:.3f}")
    elif value > max_value:
        soft.append(f"qwen_v2_sentence_length_cv_high:{value:.3f}>{max_value:.3f}")


def _add_avg_paragraph_ratio_flag(
    soft: list[str],
    features: dict[str, float],
    base_metrics: dict[str, Any],
    thresholds: dict[str, float],
) -> None:
    ratio = base_metrics.get("avg_paragraph_len_ratio")
    if ratio is None:
        return
    ratio = float(ratio)
    min_ratio = float(thresholds["avg_paragraph_len_ratio_min"])
    max_ratio = float(thresholds["avg_paragraph_len_ratio_max"])
    if ratio < min_ratio or ratio > max_ratio:
        soft.append(f"qwen_v2_avg_paragraph_len_ratio_out_of_range:{ratio:.3f}")


def _soft_score(flags: list[str]) -> float:
    score = 0.0
    for flag in flags:
        if flag.startswith("qwen_v2_avg_paragraph_len_ratio"):
            score += 0.50
        elif flag.startswith("qwen_v2_sentence_length_cv"):
            score += 0.75
        elif flag.startswith("paragraph_count_near"):
            score += 0.75
        elif flag.startswith("contains_dash"):
            score += 1.00
        else:
            score += 1.00
    return score


def _retryable_hard_failures(hard: list[str]) -> bool:
    non_retryable = ("structure_source_overlap", "fact_source_copy")
    return not any(flag.startswith(non_retryable) for flag in hard)


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    output = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output

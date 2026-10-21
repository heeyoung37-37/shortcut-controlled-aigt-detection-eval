#!/usr/bin/env python3
"""Analyze human-vs-AIGT shortcuts in Fast-DetectGPT/MADE raw datasets.

The MADE files produced by this workspace use the original Fast-DetectGPT
format:

    {dataset}_{model}.raw_data.json
    {"original": [...], "sampled": [...]}

This script treats ``original`` as human text and ``sampled`` as AIGT text,
extracts shortcut-oriented style/word/content proxy features, and reports
which deltas are consistent across dataset categories and source models.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


LABEL_HUMAN = "human"
LABEL_AIGT = "aigt"
DATASET_DISPLAY = {"xsum": "XSum", "squad": "SQuAD", "writing": "WP"}

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)*")
ALPHA_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
BULLET_LINE_RE = re.compile(r"^\s*(?:[-*+]|\u2022|\d+[.)]|[A-Za-z][.)])\s+")
NUMBERED_LINE_RE = re.compile(r"^\s*(?:\d+[.)]|[A-Za-z][.)])\s+")
HEADING_LINE_RE = re.compile(r"^\s*(?:#{1,6}\s+.+|[A-Z][A-Za-z0-9 ,/&()'-]{1,70}:)\s*$")
URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
DATE_RE = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|\d{4}|"
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\b",
    re.IGNORECASE,
)
CONTRACTION_RE = re.compile(r"\b[A-Za-z]+(?:n't|'re|'ve|'ll|'d|'m|'s)\b", re.IGNORECASE)
REPEATED_PUNCT_RE = re.compile(r"([!?.,])\1+")
PROPER_PHRASE_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")

STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "with",
    "would",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}

FIRST_PERSON = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}
SECOND_PERSON = {"you", "your", "yours", "yourself", "yourselves"}
THIRD_PERSON = {
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
}
MODALS = {"can", "could", "may", "might", "must", "shall", "should", "will", "would"}
HEDGES = {
    "almost",
    "apparently",
    "around",
    "likely",
    "maybe",
    "perhaps",
    "possibly",
    "probably",
    "roughly",
    "seem",
    "seemed",
    "seeming",
    "seems",
    "somewhat",
    "suggest",
    "suggested",
    "suggests",
}
CERTAINTY = {
    "always",
    "certainly",
    "clearly",
    "definitely",
    "indeed",
    "inevitably",
    "never",
    "obviously",
    "surely",
    "undoubtedly",
}
DISCOURSE = {
    "additionally",
    "also",
    "consequently",
    "finally",
    "first",
    "furthermore",
    "hence",
    "however",
    "instead",
    "meanwhile",
    "moreover",
    "nevertheless",
    "overall",
    "second",
    "similarly",
    "therefore",
    "third",
    "thus",
}
TEMPORAL = {
    "after",
    "again",
    "already",
    "before",
    "during",
    "eventually",
    "finally",
    "later",
    "meanwhile",
    "now",
    "once",
    "previously",
    "soon",
    "then",
    "today",
    "tomorrow",
    "when",
    "while",
    "yesterday",
}
NEGATIONS = {"no", "not", "never", "none", "nobody", "nothing", "neither", "nor", "without"}
GENERIC_ADJECTIVES = {
    "different",
    "important",
    "key",
    "large",
    "major",
    "many",
    "new",
    "other",
    "several",
    "significant",
    "similar",
    "various",
}

TRANSITION_PHRASES = (
    "as a result",
    "at the same time",
    "for example",
    "for instance",
    "in addition",
    "in conclusion",
    "in fact",
    "in other words",
    "on the other hand",
    "over time",
    "the fact that",
    "this means",
)
TASK_META_PHRASES = (
    "the article",
    "the author",
    "the main character",
    "the passage",
    "the reader",
    "the story",
    "this article",
    "this passage",
    "this story",
)
AI_DISCLAIMER_PHRASES = (
    "as an ai",
    "i cannot",
    "i can't",
    "i do not have",
    "i don't have",
    "language model",
)
CLICHE_PHRASES = (
    "a new chapter",
    "a new era",
    "a step in the right direction",
    "against all odds",
    "at a crossroads",
    "at the end of the day",
    "bright future",
    "changed forever",
    "comes full circle",
    "dream come true",
    "end of an era",
    "for generations to come",
    "for years to come",
    "in today's world",
    "lasting impact",
    "left an indelible mark",
    "make a difference",
    "more than ever",
    "needless to say",
    "new chapter",
    "next chapter",
    "now more than ever",
    "one thing is certain",
    "only time will tell",
    "paving the way",
    "serves as a reminder",
    "shaping the future",
    "stands as a testament",
    "testament to",
    "the rest is history",
    "time will tell",
    "turning point",
)
VAGUE_INTENSIFIERS = {
    "absolutely",
    "basically",
    "certainly",
    "clearly",
    "deeply",
    "especially",
    "extremely",
    "fairly",
    "greatly",
    "highly",
    "incredibly",
    "largely",
    "literally",
    "particularly",
    "pretty",
    "quite",
    "rather",
    "really",
    "remarkably",
    "significantly",
    "simply",
    "somewhat",
    "truly",
    "ultimately",
    "undeniably",
    "undoubtedly",
    "very",
}
FORWARD_LOOKING_ENDING_RE = re.compile(
    r"\b(?:"
    r"only time will tell|"
    r"time will tell|"
    r"for (?:years|generations) to come|"
    r"in the (?:years|future) to come|"
    r"in the future|"
    r"going forward|"
    r"moving forward|"
    r"remains to be seen|"
    r"(?:is|are) yet to be seen|"
    r"promises? to|"
    r"(?:is|are) (?:expected|set|likely|poised) to (?:continue|become|bring|shape|transform|change|remain)|"
    r"(?:will|would|could|may|might) (?:continue to|remain|shape|determine|define|bring|change|transform|lead to|be remembered|be felt)|"
    r"next chapter|"
    r"the future (?:of|will)|"
    r"what happens next"
    r")\b",
    re.IGNORECASE,
)

FEATURE_CATEGORIES = {
    "length": {
        "char_count",
        "nonspace_char_count",
        "word_count",
        "alpha_word_count",
        "unique_word_count",
        "sentence_count",
        "paragraph_count",
        "line_count",
        "nonempty_line_count",
        "avg_word_len",
        "avg_sentence_len_words",
        "max_sentence_len_words",
        "avg_paragraph_len_words",
    },
    "format": {
        "leading_space",
        "starts_lowercase",
        "starts_quote",
        "starts_ellipsis",
        "starts_digit",
        "starts_punctuation",
        "ends_terminal_punct",
        "newline_count",
        "blank_line_count",
        "bullet_line_ratio",
        "numbered_line_ratio",
        "heading_line_ratio",
        "quote_line_ratio",
        "dialogue_line_ratio",
        "colon_line_ratio",
        "first_paragraph_short",
        "markdown_symbol_per_100w",
    },
    "punctuation": {
        "comma_per_100w",
        "semicolon_per_100w",
        "colon_per_100w",
        "exclamation_per_100w",
        "question_per_100w",
        "ellipsis_per_100w",
        "dash_per_100w",
        "parenthesis_per_100w",
        "quote_char_per_100w",
        "punctuation_char_per_100w",
        "repeated_punctuation_per_100w",
    },
    "lexical": {
        "type_token_ratio",
        "hapax_ratio",
        "stopword_ratio",
        "content_word_ratio",
        "contraction_per_100w",
        "first_person_per_100w",
        "second_person_per_100w",
        "third_person_per_100w",
        "pronoun_per_100w",
        "modal_per_100w",
        "hedge_per_100w",
        "certainty_per_100w",
        "discourse_marker_per_100w",
        "temporal_marker_per_100w",
        "negation_per_100w",
        "generic_adj_per_100w",
        "transition_phrase_per_100w",
    },
    "content_proxy": {
        "named_entity_like_per_100w",
        "proper_phrase_per_100w",
        "number_per_100w",
        "date_like_per_100w",
        "acronym_per_100w",
        "url_or_email_per_100w",
        "task_meta_phrase_per_100w",
        "ai_disclaimer_phrase_per_100w",
        "specificity_proxy_per_100w",
    },
    "structural_cliche": {
        "forward_looking_ending",
        "cliche_phrase_per_100w",
        "ending_cliche_phrase",
        "vague_intensifier_per_100w",
        "ending_vague_intensifier",
    },
    "repetition": {
        "repeated_bigram_ratio",
        "repeated_trigram_ratio",
        "duplicate_sentence_ratio",
        "most_common_token_share",
    },
    "readability": {
        "flesch_reading_ease",
        "gunning_fog",
        "complex_word_ratio",
    },
}


@dataclass
class TextRecord:
    dataset: str
    model: str
    pair_id: int
    label: str
    text: str
    source_file: str


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def per_100(count: float, word_count: int) -> float:
    return 100.0 * safe_div(count, word_count)


def mean(values: Sequence[float]) -> float:
    return safe_div(sum(values), len(values))


def median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0


def stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def cohen_d(aigt_values: Sequence[float], human_values: Sequence[float]) -> float:
    if len(aigt_values) < 2 or len(human_values) < 2:
        return 0.0
    sd_aigt = stdev(aigt_values)
    sd_human = stdev(human_values)
    pooled_num = (len(aigt_values) - 1) * sd_aigt**2 + (len(human_values) - 1) * sd_human**2
    pooled_den = len(aigt_values) + len(human_values) - 2
    pooled = math.sqrt(safe_div(pooled_num, pooled_den))
    return safe_div(mean(aigt_values) - mean(human_values), pooled)


def sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def split_sentences(text: str) -> List[str]:
    stripped = text.strip()
    if not stripped:
        return []
    parts = [part.strip() for part in SENTENCE_SPLIT_RE.split(stripped) if part.strip()]
    return parts or [stripped]


def split_paragraphs(text: str) -> List[str]:
    stripped = text.strip()
    if not stripped:
        return []
    return [part.strip() for part in re.split(r"\n\s*\n+", stripped) if part.strip()]


def normalize_text_for_words(text: str) -> str:
    return (
        text.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )


def alpha_words(text: str) -> List[str]:
    return ALPHA_WORD_RE.findall(normalize_text_for_words(text))


def normalized_words(text: str) -> List[str]:
    return [word.lower() for word in alpha_words(text)]


def collapse_whitespace(text: str) -> str:
    return " ".join(str(text).split())


def phrase_normalized_text(text: str) -> str:
    return collapse_whitespace(normalize_text_for_words(text)).lower()


def ending_region(text: str, max_words: int = 70) -> str:
    sentences = split_sentences(text)
    candidate = " ".join(sentences[-2:]) if sentences else text
    tokens = WORD_RE.findall(candidate)
    if len(tokens) <= max_words:
        return candidate
    return " ".join(tokens[-max_words:])


def final_sentence_region(text: str, max_words: int = 45) -> str:
    sentences = split_sentences(text)
    candidate = sentences[-1] if sentences else text
    tokens = WORD_RE.findall(candidate)
    if len(tokens) <= max_words:
        return candidate
    return " ".join(tokens[-max_words:])


def count_phrase_hits(text_lower: str, phrases: Iterable[str]) -> int:
    return sum(text_lower.count(phrase) for phrase in phrases)


def count_syllables(word: str) -> int:
    clean = re.sub(r"[^a-z]", "", word.lower())
    if not clean:
        return 0
    groups = re.findall(r"[aeiouy]+", clean)
    count = len(groups)
    if clean.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def repeated_ngram_ratio(tokens: Sequence[str], n: int) -> float:
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[idx : idx + n]) for idx in range(len(tokens) - n + 1)]
    counts = Counter(grams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return safe_div(repeated, len(grams))


def named_entity_like_count(sentences: Sequence[str]) -> int:
    count = 0
    for sentence in sentences:
        words = alpha_words(sentence)
        for idx, word in enumerate(words):
            lower = word.lower()
            if lower in STOPWORDS:
                continue
            if word.isupper() and len(word) > 1:
                count += 1
            elif idx > 0 and word[:1].isupper() and word[1:].islower():
                count += 1
    return count


def feature_category(feature_name: str) -> str:
    for category, names in FEATURE_CATEGORIES.items():
        if feature_name in names:
            return category
    return "other"


def extract_features(text: str) -> Dict[str, float]:
    token_text = normalize_text_for_words(text)
    tokens = WORD_RE.findall(token_text)
    words = alpha_words(text)
    lower_words = [word.lower() for word in words]
    word_count = len(tokens)
    alpha_word_count = len(words)
    unique_words = set(lower_words)
    token_counts = Counter(lower_words)
    sentences = split_sentences(text)
    sentence_lengths = [len(WORD_RE.findall(sentence)) for sentence in sentences]
    paragraphs = split_paragraphs(text)
    paragraph_lengths = [len(WORD_RE.findall(paragraph)) for paragraph in paragraphs]
    lines = text.splitlines()
    nonempty_lines = [line for line in lines if line.strip()]
    stripped = text.lstrip()
    text_lower = text.lower()
    phrase_lower = phrase_normalized_text(text)
    ending_lower = phrase_normalized_text(ending_region(text))
    final_sentence_lower = phrase_normalized_text(final_sentence_region(text))

    syllables = sum(count_syllables(word) for word in words)
    complex_words = sum(1 for word in words if count_syllables(word) >= 3)

    flesch = 0.0
    fog = 0.0
    if alpha_word_count and sentences:
        words_per_sentence = safe_div(alpha_word_count, len(sentences))
        syllables_per_word = safe_div(syllables, alpha_word_count)
        flesch = 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word
        fog = 0.4 * (words_per_sentence + 100.0 * safe_div(complex_words, alpha_word_count))

    bullet_lines = sum(1 for line in nonempty_lines if BULLET_LINE_RE.match(line))
    numbered_lines = sum(1 for line in nonempty_lines if NUMBERED_LINE_RE.match(line))
    heading_lines = sum(1 for line in nonempty_lines if HEADING_LINE_RE.match(line))
    quote_lines = sum(1 for line in nonempty_lines if line.lstrip().startswith((">", '"', "'")))
    dialogue_lines = sum(1 for line in nonempty_lines if re.match(r'^\s*(?:"|\')', line))
    colon_lines = sum(1 for line in nonempty_lines if line.rstrip().endswith(":"))
    blank_lines = sum(1 for line in lines if not line.strip())

    stopword_count = sum(1 for word in lower_words if word in STOPWORDS)
    content_word_count = sum(1 for word in lower_words if word not in STOPWORDS)
    hapax_count = sum(1 for count in token_counts.values() if count == 1)
    most_common_count = token_counts.most_common(1)[0][1] if token_counts else 0
    duplicate_sentences = 0
    if sentences:
        normalized_sentences = [re.sub(r"\s+", " ", sentence.strip().lower()) for sentence in sentences]
        duplicate_sentences = len(normalized_sentences) - len(set(normalized_sentences))

    named_entities = named_entity_like_count(sentences)
    numbers = sum(1 for token in tokens if re.fullmatch(r"\d+(?:[.,]\d+)*", token))
    dates = len(DATE_RE.findall(text))
    acronyms = sum(1 for word in words if word.isupper() and len(word) > 1)
    urls_or_emails = len(URL_RE.findall(text)) + len(EMAIL_RE.findall(text))
    proper_phrases = len(PROPER_PHRASE_RE.findall(text))
    markdown_symbols = (
        text.count("#")
        + text.count("*")
        + text.count("`")
        + text.count("_")
        + text.count("\u2022")
    )

    punctuation_chars = sum(1 for char in text if char in ".,!?;:()[]{}\"'-")
    ellipses = text.count("...") + text.count("\u2026")
    dashes = text.count("-")
    quotes = text.count('"') + text.count("'")
    parentheses = text.count("(") + text.count(")")
    repeated_punct = len(REPEATED_PUNCT_RE.findall(text))
    transition_phrases = count_phrase_hits(text_lower, TRANSITION_PHRASES)
    task_meta_phrases = count_phrase_hits(text_lower, TASK_META_PHRASES)
    ai_disclaimer_phrases = count_phrase_hits(text_lower, AI_DISCLAIMER_PHRASES)
    cliche_phrases = count_phrase_hits(phrase_lower, CLICHE_PHRASES)
    ending_cliche_phrases = count_phrase_hits(ending_lower, CLICHE_PHRASES)
    forward_looking_ending = 1 if FORWARD_LOOKING_ENDING_RE.search(final_sentence_lower) else 0
    vague_intensifiers = sum(1 for word in lower_words if word in VAGUE_INTENSIFIERS)
    ending_vague_intensifiers = sum(1 for word in normalized_words(ending_lower) if word in VAGUE_INTENSIFIERS)

    pronoun_count = sum(
        1
        for word in lower_words
        if word in FIRST_PERSON or word in SECOND_PERSON or word in THIRD_PERSON
    )
    specificity_proxy = named_entities + numbers + acronyms + proper_phrases

    features = {
        "char_count": float(len(text)),
        "nonspace_char_count": float(sum(1 for char in text if not char.isspace())),
        "word_count": float(word_count),
        "alpha_word_count": float(alpha_word_count),
        "unique_word_count": float(len(unique_words)),
        "sentence_count": float(len(sentences)),
        "paragraph_count": float(len(paragraphs)),
        "line_count": float(len(lines) if text else 0),
        "nonempty_line_count": float(len(nonempty_lines)),
        "avg_word_len": mean([len(word) for word in words]),
        "avg_sentence_len_words": mean(sentence_lengths),
        "max_sentence_len_words": float(max(sentence_lengths) if sentence_lengths else 0),
        "avg_paragraph_len_words": mean(paragraph_lengths),
        "leading_space": float(1 if text[:1].isspace() else 0),
        "starts_lowercase": float(1 if stripped[:1].islower() else 0),
        "starts_quote": float(1 if stripped.startswith(('"', "'")) else 0),
        "starts_ellipsis": float(1 if stripped.startswith(("...", "\u2026")) else 0),
        "starts_digit": float(1 if stripped[:1].isdigit() else 0),
        "starts_punctuation": float(1 if stripped[:1] and not stripped[:1].isalnum() else 0),
        "ends_terminal_punct": float(1 if text.rstrip().endswith((".", "?", "!", '"', "'")) else 0),
        "newline_count": float(text.count("\n")),
        "blank_line_count": float(blank_lines),
        "bullet_line_ratio": safe_div(bullet_lines, len(nonempty_lines)),
        "numbered_line_ratio": safe_div(numbered_lines, len(nonempty_lines)),
        "heading_line_ratio": safe_div(heading_lines, len(nonempty_lines)),
        "quote_line_ratio": safe_div(quote_lines, len(nonempty_lines)),
        "dialogue_line_ratio": safe_div(dialogue_lines, len(nonempty_lines)),
        "colon_line_ratio": safe_div(colon_lines, len(nonempty_lines)),
        "first_paragraph_short": float(1 if paragraph_lengths and paragraph_lengths[0] < 10 else 0),
        "markdown_symbol_per_100w": per_100(markdown_symbols, word_count),
        "comma_per_100w": per_100(text.count(","), word_count),
        "semicolon_per_100w": per_100(text.count(";"), word_count),
        "colon_per_100w": per_100(text.count(":"), word_count),
        "exclamation_per_100w": per_100(text.count("!"), word_count),
        "question_per_100w": per_100(text.count("?"), word_count),
        "ellipsis_per_100w": per_100(ellipses, word_count),
        "dash_per_100w": per_100(dashes, word_count),
        "parenthesis_per_100w": per_100(parentheses, word_count),
        "quote_char_per_100w": per_100(quotes, word_count),
        "punctuation_char_per_100w": per_100(punctuation_chars, word_count),
        "repeated_punctuation_per_100w": per_100(repeated_punct, word_count),
        "type_token_ratio": safe_div(len(unique_words), alpha_word_count),
        "hapax_ratio": safe_div(hapax_count, len(token_counts)),
        "stopword_ratio": safe_div(stopword_count, alpha_word_count),
        "content_word_ratio": safe_div(content_word_count, alpha_word_count),
        "contraction_per_100w": per_100(len(CONTRACTION_RE.findall(token_text)), word_count),
        "first_person_per_100w": per_100(sum(1 for word in lower_words if word in FIRST_PERSON), word_count),
        "second_person_per_100w": per_100(sum(1 for word in lower_words if word in SECOND_PERSON), word_count),
        "third_person_per_100w": per_100(sum(1 for word in lower_words if word in THIRD_PERSON), word_count),
        "pronoun_per_100w": per_100(pronoun_count, word_count),
        "modal_per_100w": per_100(sum(1 for word in lower_words if word in MODALS), word_count),
        "hedge_per_100w": per_100(sum(1 for word in lower_words if word in HEDGES), word_count),
        "certainty_per_100w": per_100(sum(1 for word in lower_words if word in CERTAINTY), word_count),
        "discourse_marker_per_100w": per_100(sum(1 for word in lower_words if word in DISCOURSE), word_count),
        "temporal_marker_per_100w": per_100(sum(1 for word in lower_words if word in TEMPORAL), word_count),
        "negation_per_100w": per_100(sum(1 for word in lower_words if word in NEGATIONS), word_count),
        "generic_adj_per_100w": per_100(sum(1 for word in lower_words if word in GENERIC_ADJECTIVES), word_count),
        "transition_phrase_per_100w": per_100(transition_phrases, word_count),
        "named_entity_like_per_100w": per_100(named_entities, word_count),
        "proper_phrase_per_100w": per_100(proper_phrases, word_count),
        "number_per_100w": per_100(numbers, word_count),
        "date_like_per_100w": per_100(dates, word_count),
        "acronym_per_100w": per_100(acronyms, word_count),
        "url_or_email_per_100w": per_100(urls_or_emails, word_count),
        "task_meta_phrase_per_100w": per_100(task_meta_phrases, word_count),
        "ai_disclaimer_phrase_per_100w": per_100(ai_disclaimer_phrases, word_count),
        "specificity_proxy_per_100w": per_100(specificity_proxy, word_count),
        "forward_looking_ending": float(forward_looking_ending),
        "cliche_phrase_per_100w": per_100(cliche_phrases, word_count),
        "ending_cliche_phrase": float(1 if ending_cliche_phrases > 0 else 0),
        "vague_intensifier_per_100w": per_100(vague_intensifiers, word_count),
        "ending_vague_intensifier": float(1 if ending_vague_intensifiers > 0 else 0),
        "repeated_bigram_ratio": repeated_ngram_ratio(lower_words, 2),
        "repeated_trigram_ratio": repeated_ngram_ratio(lower_words, 3),
        "duplicate_sentence_ratio": safe_div(duplicate_sentences, len(sentences)),
        "most_common_token_share": safe_div(most_common_count, alpha_word_count),
        "flesch_reading_ease": flesch,
        "gunning_fog": fog,
        "complex_word_ratio": safe_div(complex_words, alpha_word_count),
    }
    return features


def default_data_dir() -> Path:
    candidates = [
        Path("MADE/fast_detect_gpt_main_generation/data"),
        Path("exp_main/data"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract shortcut-style feature summaries for human vs AIGT text "
            "from Fast-DetectGPT/MADE raw_data.json files."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--out-dir", type=Path, default=Path("aigt_feature_analysis"))
    parser.add_argument("--datasets", nargs="+", default=None, help="Dataset filters, e.g. xsum squad writing.")
    parser.add_argument("--models", nargs="+", default=None, help="Source model filters.")
    parser.add_argument("--sample-limit", type=int, default=None, help="Optional per-file pair limit.")
    parser.add_argument("--min-ngram-docs", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument(
        "--whitespace-mode",
        choices=["raw", "collapse"],
        default="raw",
        help=(
            "raw keeps the raw_data.json strings as-is. collapse applies the same "
            "' '.join(text.split()) whitespace normalization to both human and AIGT "
            "texts before feature extraction."
        ),
    )
    parser.add_argument("--no-sample-features", action="store_true", help="Skip writing sample_features.csv.")
    return parser.parse_args()


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metadata_for_raw_file(path: Path) -> Tuple[str, str]:
    args_path = Path(str(path).replace(".raw_data.json", ".args.json"))
    if args_path.exists():
        args = load_json(args_path)
        if isinstance(args, dict):
            dataset = str(args.get("dataset", "")).strip()
            model = str(args.get("base_model_name", "")).strip()
            if dataset and model:
                return dataset, model

    stem = path.name.replace(".raw_data.json", "")
    dataset, _, model = stem.partition("_")
    return dataset, model


def discover_raw_files(data_dir: Path, datasets: Optional[Sequence[str]], models: Optional[Sequence[str]]) -> List[Path]:
    wanted_datasets = set(datasets or [])
    wanted_models = set(models or [])
    files = []
    for path in sorted(data_dir.glob("*.raw_data.json")):
        dataset, model = metadata_for_raw_file(path)
        if wanted_datasets and dataset not in wanted_datasets:
            continue
        if wanted_models and model not in wanted_models:
            continue
        files.append(path)
    return files


def load_records(raw_files: Sequence[Path], sample_limit: Optional[int]) -> Tuple[List[TextRecord], List[Dict[str, object]]]:
    records = []
    inventory = []
    for path in raw_files:
        dataset, model = metadata_for_raw_file(path)
        payload = load_json(path)
        if not isinstance(payload, dict) or "original" not in payload or "sampled" not in payload:
            raise ValueError(f"{path} is not a Fast-DetectGPT raw_data.json file.")
        human_texts = payload["original"]
        aigt_texts = payload["sampled"]
        if not isinstance(human_texts, list) or not isinstance(aigt_texts, list):
            raise ValueError(f"{path} must contain list-valued original and sampled fields.")
        pair_count = min(len(human_texts), len(aigt_texts))
        if sample_limit is not None:
            pair_count = min(pair_count, sample_limit)

        inventory.append(
            {
                "dataset": dataset,
                "dataset_display": DATASET_DISPLAY.get(dataset, dataset),
                "model": model,
                "pairs": pair_count,
                "human_total": len(human_texts),
                "aigt_total": len(aigt_texts),
                "source_file": str(path),
            }
        )
        for pair_id in range(pair_count):
            records.append(
                TextRecord(
                    dataset=dataset,
                    model=model,
                    pair_id=pair_id,
                    label=LABEL_HUMAN,
                    text=str(human_texts[pair_id]),
                    source_file=str(path),
                )
            )
            records.append(
                TextRecord(
                    dataset=dataset,
                    model=model,
                    pair_id=pair_id,
                    label=LABEL_AIGT,
                    text=str(aigt_texts[pair_id]),
                    source_file=str(path),
                )
            )
    return records, inventory


def row_key(row: Dict[str, object], keys: Sequence[str]) -> Tuple[object, ...]:
    return tuple(row[key] for key in keys)


def summarize_features(
    feature_rows: Sequence[Dict[str, object]],
    feature_names: Sequence[str],
    group_keys: Sequence[str],
) -> List[Dict[str, object]]:
    grouped = defaultdict(lambda: {LABEL_HUMAN: defaultdict(list), LABEL_AIGT: defaultdict(list)})
    for row in feature_rows:
        group = row_key(row, group_keys)
        label = str(row["label"])
        for feature in feature_names:
            grouped[group][label][feature].append(float(row[feature]))

    summary_rows = []
    for group, by_label in sorted(grouped.items()):
        for feature in feature_names:
            human_values = by_label[LABEL_HUMAN][feature]
            aigt_values = by_label[LABEL_AIGT][feature]
            if not human_values or not aigt_values:
                continue
            mean_human = mean(human_values)
            mean_aigt = mean(aigt_values)
            delta = mean_aigt - mean_human
            row = {
                "category": feature_category(feature),
                "feature": feature,
                "n_human": len(human_values),
                "n_aigt": len(aigt_values),
                "mean_human": mean_human,
                "mean_aigt": mean_aigt,
                "delta_aigt_minus_human": delta,
                "relative_delta_pct": 100.0 * safe_div(delta, abs(mean_human)),
                "cohen_d": cohen_d(aigt_values, human_values),
                "human_sd": stdev(human_values),
                "aigt_sd": stdev(aigt_values),
            }
            for idx, key in enumerate(group_keys):
                row[key] = group[idx]
            summary_rows.append(row)

    sort_keys = list(group_keys) + ["category", "feature"]
    return sorted(summary_rows, key=lambda row: tuple(str(row.get(key, "")) for key in sort_keys))


def make_paired_delta_rows(
    feature_rows: Sequence[Dict[str, object]],
    feature_names: Sequence[str],
) -> List[Dict[str, object]]:
    pairs = defaultdict(dict)
    for row in feature_rows:
        key = (row["dataset"], row["model"], row["pair_id"])
        pairs[key][row["label"]] = row

    delta_rows = []
    for (dataset, model, pair_id), by_label in sorted(pairs.items()):
        human = by_label.get(LABEL_HUMAN)
        aigt = by_label.get(LABEL_AIGT)
        if not human or not aigt:
            continue
        row = {
            "dataset": dataset,
            "dataset_display": DATASET_DISPLAY.get(str(dataset), str(dataset)),
            "model": model,
            "pair_id": pair_id,
        }
        for feature in feature_names:
            row[feature] = float(aigt[feature]) - float(human[feature])
        delta_rows.append(row)
    return delta_rows


def summarize_paired_deltas(
    paired_rows: Sequence[Dict[str, object]],
    feature_names: Sequence[str],
    group_keys: Sequence[str],
) -> List[Dict[str, object]]:
    grouped = defaultdict(lambda: defaultdict(list))
    for row in paired_rows:
        group = row_key(row, group_keys)
        for feature in feature_names:
            grouped[group][feature].append(float(row[feature]))

    summary_rows = []
    for group, by_feature in sorted(grouped.items()):
        for feature in feature_names:
            values = by_feature[feature]
            if not values:
                continue
            row = {
                "category": feature_category(feature),
                "feature": feature,
                "n_pairs": len(values),
                "mean_paired_delta": mean(values),
                "median_paired_delta": median(values),
                "sd_paired_delta": stdev(values),
                "pct_pairs_aigt_gt_human": 100.0 * safe_div(sum(1 for value in values if value > 0), len(values)),
                "pct_pairs_equal": 100.0 * safe_div(sum(1 for value in values if value == 0), len(values)),
            }
            for idx, key in enumerate(group_keys):
                row[key] = group[idx]
            summary_rows.append(row)
    sort_keys = list(group_keys) + ["category", "feature"]
    return sorted(summary_rows, key=lambda row: tuple(str(row.get(key, "")) for key in sort_keys))


def build_consistency_table(
    overall_rows: Sequence[Dict[str, object]],
    dataset_rows: Sequence[Dict[str, object]],
    model_rows: Sequence[Dict[str, object]],
    paired_overall_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    overall_by_feature = {str(row["feature"]): row for row in overall_rows}
    paired_by_feature = {str(row["feature"]): row for row in paired_overall_rows}
    dataset_by_feature = defaultdict(list)
    model_by_feature = defaultdict(list)
    for row in dataset_rows:
        dataset_by_feature[str(row["feature"])].append(row)
    for row in model_rows:
        model_by_feature[str(row["feature"])].append(row)

    rows = []
    for feature, overall in overall_by_feature.items():
        overall_delta = float(overall["delta_aigt_minus_human"])
        overall_sign = sign(overall_delta)
        dataset_deltas = dataset_by_feature[feature]
        model_deltas = model_by_feature[feature]
        dataset_support = sum(
            1 for row in dataset_deltas if sign(float(row["delta_aigt_minus_human"])) == overall_sign and overall_sign
        )
        model_support = sum(
            1 for row in model_deltas if sign(float(row["delta_aigt_minus_human"])) == overall_sign and overall_sign
        )
        dataset_total = len(dataset_deltas)
        model_total = len(model_deltas)
        dataset_support_ratio = safe_div(dataset_support, dataset_total)
        model_support_ratio = safe_div(model_support, model_total)
        consistency_score = abs(float(overall["cohen_d"])) * dataset_support_ratio * model_support_ratio
        paired = paired_by_feature.get(feature, {})
        rows.append(
            {
                "category": overall["category"],
                "feature": feature,
                "mean_human": overall["mean_human"],
                "mean_aigt": overall["mean_aigt"],
                "delta_aigt_minus_human": overall_delta,
                "median_paired_delta": paired.get("median_paired_delta", ""),
                "pct_pairs_aigt_gt_human": paired.get("pct_pairs_aigt_gt_human", ""),
                "cohen_d": overall["cohen_d"],
                "dataset_direction_support": f"{dataset_support}/{dataset_total}",
                "model_direction_support": f"{model_support}/{model_total}",
                "dataset_support_ratio": dataset_support_ratio,
                "model_support_ratio": model_support_ratio,
                "consistency_score": consistency_score,
            }
        )
    return sorted(rows, key=lambda row: float(row["consistency_score"]), reverse=True)


def ngram_doc_set(text: str, max_n: int = 3) -> Dict[int, set]:
    words = normalized_words(text)
    by_n = {}
    for n in range(1, max_n + 1):
        grams = set()
        if len(words) < n:
            by_n[n] = grams
            continue
        for idx in range(len(words) - n + 1):
            window = tuple(words[idx : idx + n])
            if n == 1:
                if len(window[0]) < 3 or window[0] in STOPWORDS:
                    continue
            elif not any(word not in STOPWORDS for word in window):
                continue
            grams.add(" ".join(window))
        by_n[n] = grams
    return by_n


def analyze_ngrams(
    records: Sequence[TextRecord],
    min_docs: int,
) -> List[Dict[str, object]]:
    global_df = {LABEL_HUMAN: Counter(), LABEL_AIGT: Counter()}
    dataset_df = defaultdict(lambda: {LABEL_HUMAN: Counter(), LABEL_AIGT: Counter()})
    global_doc_totals = Counter()
    dataset_doc_totals = defaultdict(Counter)
    ngram_sizes = {}

    for record in records:
        global_doc_totals[record.label] += 1
        dataset_doc_totals[record.dataset][record.label] += 1
        for n, grams in ngram_doc_set(record.text).items():
            for gram in grams:
                global_df[record.label][gram] += 1
                dataset_df[record.dataset][record.label][gram] += 1
                ngram_sizes[gram] = n

    all_grams = set(global_df[LABEL_HUMAN]) | set(global_df[LABEL_AIGT])
    rows = []
    for gram in all_grams:
        human_docs = global_df[LABEL_HUMAN][gram]
        aigt_docs = global_df[LABEL_AIGT][gram]
        if human_docs + aigt_docs < min_docs:
            continue
        human_total = global_doc_totals[LABEL_HUMAN]
        aigt_total = global_doc_totals[LABEL_AIGT]
        human_rate = safe_div(human_docs, human_total)
        aigt_rate = safe_div(aigt_docs, aigt_total)
        diff = aigt_rate - human_rate
        direction = sign(diff)
        if direction == 0:
            continue
        log_odds = math.log(safe_div(aigt_docs + 0.5, aigt_total - aigt_docs + 0.5)) - math.log(
            safe_div(human_docs + 0.5, human_total - human_docs + 0.5)
        )

        support = 0
        total_datasets = 0
        dataset_rate_bits = []
        for dataset, totals in sorted(dataset_doc_totals.items()):
            if totals[LABEL_HUMAN] == 0 or totals[LABEL_AIGT] == 0:
                continue
            total_datasets += 1
            d_human = safe_div(dataset_df[dataset][LABEL_HUMAN][gram], totals[LABEL_HUMAN])
            d_aigt = safe_div(dataset_df[dataset][LABEL_AIGT][gram], totals[LABEL_AIGT])
            if sign(d_aigt - d_human) == direction:
                support += 1
            dataset_rate_bits.append(f"{dataset}:h={d_human:.3f},a={d_aigt:.3f}")

        support_ratio = safe_div(support, total_datasets)
        odds_weight = 1.0 + min(abs(log_odds), 4.0) / 4.0
        rank_score = abs(diff) * odds_weight * support_ratio
        rows.append(
            {
                "ngram": gram,
                "n": ngram_sizes.get(gram, len(gram.split())),
                "higher_label": LABEL_AIGT if diff > 0 else LABEL_HUMAN,
                "human_doc_count": human_docs,
                "aigt_doc_count": aigt_docs,
                "human_doc_rate": human_rate,
                "aigt_doc_rate": aigt_rate,
                "diff_aigt_minus_human": diff,
                "log_odds_aigt_vs_human": log_odds,
                "dataset_direction_support": f"{support}/{total_datasets}",
                "dataset_support_ratio": support_ratio,
                "dataset_rates": "; ".join(dataset_rate_bits),
                "rank_score": rank_score,
            }
        )

    return sorted(rows, key=lambda row: float(row["rank_score"]), reverse=True)


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_float(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number) or math.isinf(number):
        return str(number)
    return f"{number:.{digits}f}"


def markdown_table(rows: Sequence[Dict[str, object]], columns: Sequence[Tuple[str, str]], limit: int) -> List[str]:
    selected = list(rows[:limit])
    if not selected:
        return ["No rows."]
    header = "| " + " | ".join(title for title, _ in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for row in selected:
        values = []
        for _, key in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                value = format_float(value)
            values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(
    path: Path,
    data_dir: Path,
    whitespace_mode: str,
    inventory: Sequence[Dict[str, object]],
    consistency_rows: Sequence[Dict[str, object]],
    ngram_rows: Sequence[Dict[str, object]],
    top_k: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total_pairs = sum(int(row["pairs"]) for row in inventory)
    datasets = sorted({str(row["dataset"]) for row in inventory})
    models = sorted({str(row["model"]) for row in inventory})
    aigt_ngrams = [row for row in ngram_rows if row["higher_label"] == LABEL_AIGT]
    human_ngrams = [row for row in ngram_rows if row["higher_label"] == LABEL_HUMAN]

    lines = [
        "# AIGT Shortcut Feature Analysis",
        "",
        f"- Data dir: `{data_dir}`",
        f"- Whitespace mode: `{whitespace_mode}`",
        f"- Dataset categories: {', '.join(DATASET_DISPLAY.get(dataset, dataset) for dataset in datasets)}",
        f"- Source models: {', '.join(models)}",
        f"- Paired examples: {total_pairs}",
        "",
        "The ranking favors features whose AIGT-human deltas point in the same direction across dataset categories and source models. This keeps the analysis focused on broad human-vs-AIGT patterns instead of one dataset's local artifacts.",
        "",
        "## Data Inventory",
    ]
    lines.extend(
        markdown_table(
            inventory,
            [
                ("Dataset", "dataset_display"),
                ("Model", "model"),
                ("Pairs", "pairs"),
            ],
            limit=len(inventory),
        )
    )

    lines.extend(
        [
            "",
            "## Most Consistent Feature Deltas",
        ]
    )
    lines.extend(
        markdown_table(
            consistency_rows,
            [
                ("Category", "category"),
                ("Feature", "feature"),
                ("Human mean", "mean_human"),
                ("AIGT mean", "mean_aigt"),
                ("Delta", "delta_aigt_minus_human"),
                ("AIGT>Human pairs %", "pct_pairs_aigt_gt_human"),
                ("d", "cohen_d"),
                ("Dataset support", "dataset_direction_support"),
                ("Model support", "model_direction_support"),
            ],
            limit=top_k,
        )
    )

    lines.extend(["", "## AIGT-Associated Ngrams"])
    lines.extend(
        markdown_table(
            aigt_ngrams,
            [
                ("Ngram", "ngram"),
                ("n", "n"),
                ("Human rate", "human_doc_rate"),
                ("AIGT rate", "aigt_doc_rate"),
                ("Log odds", "log_odds_aigt_vs_human"),
                ("Dataset support", "dataset_direction_support"),
            ],
            limit=top_k,
        )
    )

    lines.extend(["", "## Human-Associated Ngrams"])
    lines.extend(
        markdown_table(
            human_ngrams,
            [
                ("Ngram", "ngram"),
                ("n", "n"),
                ("Human rate", "human_doc_rate"),
                ("AIGT rate", "aigt_doc_rate"),
                ("Log odds", "log_odds_aigt_vs_human"),
                ("Dataset support", "dataset_direction_support"),
            ],
            limit=top_k,
        )
    )

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `sample_features.csv`: one row per text, including metadata and all extracted features.",
            "- `paired_feature_deltas.csv`: one row per original/sampled pair, with AIGT-human feature deltas.",
            "- `feature_summary_*.csv`: unpaired mean summaries overall, by dataset, by model, and by dataset-model.",
            "- `paired_summary_*.csv`: paired delta summaries overall, by dataset, by model, and by dataset-model.",
            "- `consistent_feature_deltas.csv`: features ranked by cross-dataset/model direction consistency.",
            "- `discriminative_ngrams.csv`: document-frequency ngrams ranked by label association and dataset consistency.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    raw_files = discover_raw_files(args.data_dir, args.datasets, args.models)
    if not raw_files:
        raise SystemExit(f"No *.raw_data.json files found in {args.data_dir}")

    records, inventory = load_records(raw_files, args.sample_limit)
    feature_rows = []
    feature_names = sorted(extract_features("").keys())
    for record in records:
        text = collapse_whitespace(record.text) if args.whitespace_mode == "collapse" else record.text
        features = extract_features(text)
        row = {
            "dataset": record.dataset,
            "dataset_display": DATASET_DISPLAY.get(record.dataset, record.dataset),
            "model": record.model,
            "pair_id": record.pair_id,
            "label": record.label,
            "source_file": record.source_file,
        }
        row.update(features)
        feature_rows.append(row)

    paired_delta_rows = make_paired_delta_rows(feature_rows, feature_names)

    overall_summary = summarize_features(feature_rows, feature_names, [])
    dataset_summary = summarize_features(feature_rows, feature_names, ["dataset"])
    model_summary = summarize_features(feature_rows, feature_names, ["model"])
    dataset_model_summary = summarize_features(feature_rows, feature_names, ["dataset", "model"])
    paired_overall = summarize_paired_deltas(paired_delta_rows, feature_names, [])
    paired_dataset = summarize_paired_deltas(paired_delta_rows, feature_names, ["dataset"])
    paired_model = summarize_paired_deltas(paired_delta_rows, feature_names, ["model"])
    paired_dataset_model = summarize_paired_deltas(paired_delta_rows, feature_names, ["dataset", "model"])
    consistency_rows = build_consistency_table(overall_summary, dataset_summary, model_summary, paired_overall)
    ngram_records = [
        TextRecord(
            dataset=record.dataset,
            model=record.model,
            pair_id=record.pair_id,
            label=record.label,
            text=collapse_whitespace(record.text) if args.whitespace_mode == "collapse" else record.text,
            source_file=record.source_file,
        )
        for record in records
    ]
    ngram_rows = analyze_ngrams(ngram_records, args.min_ngram_docs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata_fields = ["dataset", "dataset_display", "model", "pair_id", "label", "source_file"]
    if not args.no_sample_features:
        write_csv(args.out_dir / "sample_features.csv", feature_rows, metadata_fields + feature_names)
    write_csv(args.out_dir / "paired_feature_deltas.csv", paired_delta_rows, ["dataset", "dataset_display", "model", "pair_id"] + feature_names)
    write_csv(args.out_dir / "data_inventory.csv", inventory)
    write_csv(args.out_dir / "feature_summary_overall.csv", overall_summary)
    write_csv(args.out_dir / "feature_summary_by_dataset.csv", dataset_summary)
    write_csv(args.out_dir / "feature_summary_by_model.csv", model_summary)
    write_csv(args.out_dir / "feature_summary_by_dataset_model.csv", dataset_model_summary)
    write_csv(args.out_dir / "paired_summary_overall.csv", paired_overall)
    write_csv(args.out_dir / "paired_summary_by_dataset.csv", paired_dataset)
    write_csv(args.out_dir / "paired_summary_by_model.csv", paired_model)
    write_csv(args.out_dir / "paired_summary_by_dataset_model.csv", paired_dataset_model)
    write_csv(args.out_dir / "consistent_feature_deltas.csv", consistency_rows)
    write_csv(args.out_dir / "discriminative_ngrams.csv", ngram_rows)
    write_report(
        args.out_dir / "report.md",
        args.data_dir,
        args.whitespace_mode,
        inventory,
        consistency_rows,
        ngram_rows,
        args.top_k,
    )

    print(f"Analyzed {len(records)} texts from {len(raw_files)} files.")
    print(f"Wrote feature analysis to {args.out_dir}")
    print("Top consistent broad features:")
    for row in consistency_rows[: min(10, len(consistency_rows))]:
        print(
            "  "
            f"{row['category']}/{row['feature']}: "
            f"delta={format_float(row['delta_aigt_minus_human'])}, "
            f"d={format_float(row['cohen_d'])}, "
            f"datasets={row['dataset_direction_support']}, "
            f"models={row['model_direction_support']}"
        )


if __name__ == "__main__":
    main()

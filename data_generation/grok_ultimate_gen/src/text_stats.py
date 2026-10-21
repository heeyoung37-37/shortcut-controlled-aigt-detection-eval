from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from difflib import SequenceMatcher
from typing import Iterable

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "our",
    "she",
    "that",
    "the",
    "their",
    "them",
    "there",
    "they",
    "this",
    "to",
    "was",
    "were",
    "will",
    "with",
    "you",
    "your",
}

MONTHS = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}

NUMBER_WORDS = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
    "hundred",
    "thousand",
    "million",
    "billion",
}

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
NUMBER_RE = re.compile(
    r"""
    (?:
      [$£€]\s?\d[\d,]*(?:\.\d+)?(?:bn|m|k|billion|million|thousand)?
      |
      \d[\d,]*(?:\.\d+)?\s?(?:%|percent|bn|m|k|billion|million|thousand)?
      |
      \b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|
      thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|
      thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion)\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)
QUOTE_PATTERNS = [
    re.compile(r'"([^"\n]{2,})"'),
    re.compile(r"“([^”\n]{2,})”"),
    re.compile(r"‘([^’\n]{8,})’"),
    re.compile(r"'([^'\n]{12,})'"),
]
CAPITALIZED_SPAN_RE = re.compile(
    r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,}|of|the|and|for)){1,5}\b"
)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_preserving_paragraphs(text: str) -> str:
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n[ \t]+", "\n", normalized)
    normalized = re.sub(r"[ \t]+\n", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def split_paragraphs(text: str) -> list[str]:
    cleaned = normalize_preserving_paragraphs(text)
    if not cleaned:
        return []
    blank_split = [normalize_whitespace(p) for p in re.split(r"\n\s*\n", cleaned) if p.strip()]
    if len(blank_split) > 1:
        return blank_split
    line_split = [normalize_whitespace(p) for p in cleaned.split("\n") if p.strip()]
    return line_split if line_split else blank_split


def split_sentences(text: str) -> list[str]:
    cleaned = normalize_whitespace(text)
    if not cleaned:
        return []
    try:
        import pysbd

        segmenter = pysbd.Segmenter(language="en", clean=False)
        sentences = [s.strip() for s in segmenter.segment(cleaned) if s.strip()]
        if sentences:
            return sentences
    except Exception:
        pass
    try:
        import nltk

        sentences = [s.strip() for s in nltk.sent_tokenize(cleaned) if s.strip()]
        if sentences:
            return sentences
    except Exception:
        pass
    pattern = re.compile(r"(?<=[.!?])\s+(?=[\"'“‘]?[A-Z0-9])")
    parts = [p.strip() for p in pattern.split(cleaned) if p.strip()]
    return parts if parts else [cleaned]


def tokenize_words(text: str) -> list[str]:
    return WORD_RE.findall(text or "")


def word_count(text: str) -> int:
    return len(tokenize_words(text))


def paragraph_count(text: str) -> int:
    return len(split_paragraphs(text))


def sentence_count(text: str) -> int:
    return len(split_sentences(text))


def contains_direct_quote(text: str) -> bool:
    return bool(extract_quote_spans(text))


def extract_quote_spans(text: str) -> list[str]:
    spans: list[tuple[int, str]] = []
    for pattern in QUOTE_PATTERNS:
        for match in pattern.finditer(text or ""):
            quote = normalize_whitespace(match.group(1))
            if quote:
                spans.append((match.start(), quote))
    return [quote for _, quote in sorted(spans)]


def remove_quote_spans(text: str) -> str:
    stripped = text or ""
    for pattern in QUOTE_PATTERNS:
        stripped = pattern.sub(" ", stripped)
    return stripped


def extract_numbers(text: str) -> list[str]:
    seen: set[str] = set()
    numbers: list[str] = []
    for match in NUMBER_RE.finditer(text or ""):
        value = normalize_whitespace(match.group(0))
        key = value.lower()
        if value and key not in seen:
            numbers.append(value)
            seen.add(key)
    return numbers


def ngrams(tokens: Iterable[str], n: int) -> list[tuple[str, ...]]:
    token_list = list(tokens)
    if n <= 0 or len(token_list) < n:
        return []
    return [tuple(token_list[i : i + n]) for i in range(len(token_list) - n + 1)]


def content_tokens(text: str) -> list[str]:
    tokens = []
    for token in tokenize_words(text.lower()):
        cleaned = token.strip("-'")
        if not cleaned or cleaned in STOPWORDS:
            continue
        tokens.append(cleaned)
    return tokens


def ngram_overlap_ratio(text_a: str, text_b: str, n: int = 5, ignore_quotes: bool = False) -> float:
    if ignore_quotes:
        text_a = remove_quote_spans(text_a)
        text_b = remove_quote_spans(text_b)
    grams_a = set(ngrams(content_tokens(text_a), n))
    if not grams_a:
        return 0.0
    grams_b = set(ngrams(content_tokens(text_b), n))
    return len(grams_a & grams_b) / max(len(grams_a), 1)


def longest_common_word_sequence_len(text_a: str, text_b: str, ignore_quotes: bool = False) -> int:
    if ignore_quotes:
        text_a = remove_quote_spans(text_a)
        text_b = remove_quote_spans(text_b)
    a_tokens = [t.lower() for t in tokenize_words(text_a)]
    b_tokens = [t.lower() for t in tokenize_words(text_b)]
    if not a_tokens or not b_tokens:
        return 0
    previous = [0] * (len(b_tokens) + 1)
    best = 0
    for token_a in a_tokens:
        current = [0] * (len(b_tokens) + 1)
        for j, token_b in enumerate(b_tokens, 1):
            if token_a == token_b:
                current[j] = previous[j - 1] + 1
                best = max(best, current[j])
        previous = current
    return best


def paragraph_start_word(paragraph: str) -> str:
    match = WORD_RE.search(paragraph or "")
    return match.group(0).lower() if match else ""


def starts_with_forbidden_start(paragraph: str, forbidden_list: list[str]) -> bool:
    start = normalize_whitespace(paragraph).lower()
    for phrase in forbidden_list:
        phrase_l = phrase.lower()
        if re.match(rf"^{re.escape(phrase_l)}\b", start):
            return True
    return False


def transition_density(text: str, transition_list: list[str]) -> float:
    wc = max(word_count(text), 1)
    lower = (text or "").lower()
    count = sum(len(re.findall(rf"\b{re.escape(term.lower())}\b", lower)) for term in transition_list)
    return count / wc * 100


def sentence_length_cv(text: str) -> float:
    lengths = [word_count(sentence) for sentence in split_sentences(text)]
    return _coefficient_of_variation(lengths)


def paragraph_length_cv(text: str) -> float:
    lengths = [word_count(paragraph) for paragraph in split_paragraphs(text)]
    return _coefficient_of_variation(lengths)


def quote_word_ratio(text: str) -> float:
    total = max(word_count(text), 1)
    quote_words = sum(word_count(quote) for quote in extract_quote_spans(text))
    return quote_words / total


def number_density_per_100_words(text: str) -> float:
    return len(extract_numbers(text)) / max(word_count(text), 1) * 100


def type_token_ratio(text: str) -> float:
    tokens = [token.lower() for token in tokenize_words(text)]
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def hapax_ratio(text: str) -> float:
    tokens = [token.lower() for token in tokenize_words(text)]
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    return sum(1 for count in counts.values() if count == 1) / len(tokens)


def average_word_length(text: str) -> float:
    tokens = tokenize_words(text)
    if not tokens:
        return 0.0
    return sum(len(token) for token in tokens) / len(tokens)


def capitalized_spans(text: str) -> list[str]:
    seen: set[str] = set()
    spans: list[str] = []
    for match in CAPITALIZED_SPAN_RE.finditer(text or ""):
        span = normalize_whitespace(match.group(0))
        first = span.split()[0].lower()
        if first in MONTHS or first in {"The", "A", "An"}:
            continue
        key = span.lower()
        if key not in seen:
            spans.append(span)
            seen.add(key)
    return spans


def sequence_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_whitespace(a).lower(), normalize_whitespace(b).lower()).ratio()


def _coefficient_of_variation(values: list[int | float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_value = statistics.mean(values)
    if math.isclose(mean_value, 0.0):
        return 0.0
    return statistics.pstdev(values) / mean_value

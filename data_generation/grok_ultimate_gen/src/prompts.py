from __future__ import annotations

import json
from typing import Any


def fact_extraction_prompt(article_text: str, config: dict[str, Any]) -> list[dict[str, str]]:
    fact_config = config["fact_card"]
    system = (
        "You extract factual content from news articles for a controlled text generation dataset.\n"
        "Return valid JSON only.\n"
        "Do not invent facts.\n"
        "Preserve named entities, dates, numbers, and direct quotes only if they appear in the source.\n"
        "Do not copy non-quoted phrases of 6 or more consecutive words from the article.\n"
        "Direct quotes may be copied exactly, but only if they appear in the source article."
    )
    user = f"""Extract a FACT_CARD from the article below.

Requirements:
- Output a single JSON object.
- Use this schema:
  {{
    "event_type": string,
    "topic_tags": string[],
    "entities": [{{"name": string, "type": string, "role": string}}],
    "core_facts": [{{"fid": integer, "importance": integer, "content": string}}],
    "numbers": [{{"value": string, "meaning": string}}],
    "quotes": [{{"speaker": string, "quote": string, "context": string}}],
    "constraints": {{
      "use_only_listed_facts": true,
      "do_not_invent_quotes": true,
      "do_not_invent_numbers": true,
      "do_not_add_unlisted_named_entities": true
    }}
  }}
- core_facts must be atomic and concise.
- core_facts must not copy non-quoted source phrases of 6 or more consecutive words.
- Include at most {fact_config["max_core_facts"]} core facts.
- Include at most {fact_config["max_entities"]} entities.
- Include at most {fact_config["max_numbers"]} numbers.
- Include at most {fact_config["max_quotes"]} direct quotes.
- If the article has no direct quote, return an empty quotes list.
- If a number, date, or quote is not present in the article, do not include it.
- Return JSON only. No markdown.

ARTICLE:
{article_text}"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generation_prompt(
    structure_card: dict[str, Any],
    fact_card: dict[str, Any],
    template_id: str,
) -> list[dict[str, str]]:
    structure_card_json = json.dumps(_strip_metadata(structure_card), ensure_ascii=False, indent=2)
    fact_card_json = json.dumps(_strip_fact_metadata(fact_card), ensure_ascii=False, indent=2)
    shape = _shape_summary(structure_card, fact_card)
    return _generation_template(template_id, structure_card_json, fact_card_json, shape)


def retry_generation_prompt(
    structure_card: dict[str, Any],
    fact_card: dict[str, Any],
    previous_text: str | None,
    fail_reasons: list[str],
    template_id: str,
) -> list[dict[str, str]]:
    messages = generation_prompt(structure_card, fact_card, template_id)
    targeted_guidance = _retry_guidance(fail_reasons)
    retry_note = (
        "The previous attempt failed rule-based validation.\n"
        "Fix these issues while using only STRUCTURE_CARD and FACT_CARD:\n"
        + "\n".join(f"- {reason}" for reason in fail_reasons)
        + targeted_guidance
        + "\n\nDo not use or refer to any original source article. Return article body only."
    )
    messages[-1]["content"] = retry_note + "\n\n" + messages[-1]["content"]
    return messages


def _generation_template(
    template_id: str,
    structure_card_json: str,
    fact_card_json: str,
    shape: str,
) -> list[dict[str, str]]:
    assignment = {
        "template_01": "Write a BBC-style news article using only FACT_CARD.",
        "template_02": "Write the assigned wire-style news article using only FACT_CARD.",
        "template_03": "Convert the fact notes into layout-matched news copy using only FACT_CARD.",
    }.get(template_id, "Write a BBC-style news article using only FACT_CARD.")
    system = (
        "You write controlled, layout-matched wire-style news articles.\n"
        "Use only the supplied FACT_CARD.\n"
        "Do not invent facts, names, dates, numbers, or quotations.\n"
        "Treat STRUCTURE_CARD as the layout plan and FACT_CARD as the content source."
    )
    user = f"""{assignment}

Shape summary:
{shape}

Priority order:
1. Paragraph count is strict: write exactly STRUCTURE_CARD.num_paragraphs paragraphs.
2. Paragraph mapping is strict: one STRUCTURE_CARD.paragraphs item equals one output paragraph, in the same order.
3. Sentence count is strict: paragraph.sentence_count is the sentence count for that paragraph.
4. FACT_CARD is the only content source.
5. Target word counts are soft length goals. Never add paragraphs or unsupported facts to reach them.

Layout instructions:
- Separate paragraphs with exactly one blank line.
- Do not use headings, bullets, paragraph numbers, JSON, or explanations.
- For one-sentence paragraph slots, write one complete sentence. For two-sentence slots, write two.
- Use paragraph.target_word_count to decide sentence fullness. Avoid very short filler paragraphs unless the target is small.
- If there are many paragraph slots, keep each slot compact and stop exactly after the final planned slot.
- Do not add recap, summary, conclusion, or extra context paragraphs.

Content instructions:
- FACT_CARD.core_facts are allowed factual notes. You may reuse names, terms, numbers, and short factual wording from them.
- Compose a new article from those facts. Do not invent facts beyond FACT_CARD.
- Fit facts, actors, reactions, institutions, and background into the planned paragraph slots instead of creating extra slots.
- Reuse a fact from a new angle only when the layout has more paragraph slots than distinct facts.
- Use quotation marks only for exact quotes listed in FACT_CARD.quotes. If no quote is listed, do not use quotation marks.
- Use named entities and numerical details only when supported by FACT_CARD.entities, FACT_CARD.core_facts, FACT_CARD.numbers, or FACT_CARD.quotes.

Style instructions:
- Do not use these words or phrases anywhere: additionally, furthermore, moreover, ultimately, overall, in addition, in contrast, similarly.
- Do not begin a paragraph with: additionally, furthermore, moreover, despite, as, with, to.
- Do not use em dashes.
- End with a concrete reported fact, attribution, or listed quote, not a broad outlook or essay-like closing sentence.
- Before final output, silently count paragraphs and sentences. Fix the output if the counts do not match.

STRUCTURE_CARD:
{structure_card_json}

FACT_CARD:
{fact_card_json}"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _strip_metadata(card: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "source_word_count",
        "length_boost_multiplier",
        "target_total_word_count",
        "num_paragraphs",
        "global_constraints",
        "paragraphs",
    }
    return {key: value for key, value in card.items() if key in allowed}


def _strip_fact_metadata(card: dict[str, Any]) -> dict[str, Any]:
    allowed = {"event_type", "topic_tags", "entities", "core_facts", "numbers", "quotes", "constraints"}
    return {key: value for key, value in card.items() if key in allowed}


def _shape_summary(structure_card: dict[str, Any], fact_card: dict[str, Any]) -> str:
    quote_count = len(fact_card.get("quotes") or [])
    entity_count = len(fact_card.get("entities") or [])
    core_fact_count = len(fact_card.get("core_facts") or [])
    target_total = structure_card.get("target_total_word_count")
    paragraphs = structure_card.get("num_paragraphs")
    paragraph_plan = _paragraph_plan_summary(structure_card)
    return (
        f"- Required paragraphs: exactly {paragraphs}\n"
        f"- Target total words: about {target_total}\n"
        f"- Paragraph plan: {paragraph_plan}\n"
        f"- Core facts available: {core_fact_count}\n"
        f"- Entities available: {entity_count}\n"
        f"- Direct quotes available: {quote_count}"
    )


def _paragraph_plan_summary(structure_card: dict[str, Any]) -> str:
    parts: list[str] = []
    for paragraph in structure_card.get("paragraphs", []) or []:
        pid = paragraph.get("pid")
        sentences = paragraph.get("sentence_count", 1)
        target_words = paragraph.get("target_word_count")
        if pid is None:
            continue
        parts.append(f"P{pid}:{sentences}s/~{target_words}w")
    if len(parts) <= 45:
        return "; ".join(parts)
    head = "; ".join(parts[:25])
    tail = "; ".join(parts[-10:])
    return f"{head}; ...; {tail}"


def _retry_guidance(fail_reasons: list[str]) -> str:
    groups = {reason.split(":", 1)[0] for reason in fail_reasons}
    guidance: list[str] = []
    if "wrong_paragraph_count" in groups:
        guidance.append(
            "For wrong_paragraph_count: write exactly the STRUCTURE_CARD.num_paragraphs count, "
            "with one blank line between paragraphs. Use one output paragraph per paragraph-plan item and stop after the final item."
        )
    if any(group.startswith("paragraph_") for group in groups):
        guidance.append(
            "For paragraph sentence-count issues: if a planned paragraph says 1s, write exactly one sentence; if it says 2s, write exactly two."
        )
    if "fact_source_5gram_overlap" in groups or "fact_source_long_copy" in groups:
        guidance.append(
            "For fact-source copying: keep the same facts, but recast the article structure and surrounding prose instead of shadowing source-article sentence order."
        )
    if "invented_quote" in groups:
        guidance.append(
            "For invented_quote: remove all quotation marks unless the text inside them is an exact quote from FACT_CARD.quotes."
        )
    if "forbidden_transition" in groups:
        guidance.append(
            "For forbidden_transition: remove additionally, furthermore, moreover, ultimately, overall, in addition, in contrast, and similarly. Never use the word overall."
        )
    if "forbidden_paragraph_start" in groups:
        guidance.append(
            "For forbidden paragraph starts: rewrite any paragraph that begins with additionally, furthermore, moreover, despite, as, with, or to."
        )
    if "too_many_unknown_capitalized_entities" in groups:
        guidance.append(
            "For unknown entities: use only names and organisations supported by FACT_CARD; replace unsupported proper nouns with generic descriptions."
        )
    if not guidance:
        return ""
    return "\n\nTargeted repair instructions:\n" + "\n".join(f"- {item}" for item in guidance)

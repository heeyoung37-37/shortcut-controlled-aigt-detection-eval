from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .layout import effective_structure_card
from .providers import Message


@dataclass(frozen=True)
class DatasetPromptSpec:
    fact_source_name: str
    system_role: str
    assignment: str
    output_contract: str
    style_notes: str


@dataclass(frozen=True)
class EditPromptSpec:
    editor_role: str
    domain_instruction: str
    preserve_format: str


EDIT_PROMPT_VERSION = "domain_light_medium_no_reference_v1"


DATASET_PROMPTS: dict[str, DatasetPromptSpec] = {
    "essay": DatasetPromptSpec(
        fact_source_name="IELTS Writing Task 2 essay",
        system_role="You write controlled, layout-matched IELTS Writing Task 2 essays.",
        assignment="Write an IELTS Writing Task 2 essay using only FACT_CARD.",
        output_contract="Return the response text only. Do not include scoring notes, labels, markdown, or explanations.",
        style_notes=(
            "Use natural exam-essay prose with a clear position, topic-focused body paragraphs, and a concise conclusion. "
            "Keep the language plausible for a learner essay rather than polished professional writing."
        ),
    ),
    "arxiv_abstract": DatasetPromptSpec(
        fact_source_name="research paper abstract",
        system_role="You write controlled, layout-matched research abstracts.",
        assignment="Write a research paper abstract using only FACT_CARD.",
        output_contract="Return the abstract body only. Do not include a title, section heading, bullets, markdown, or explanations.",
        style_notes="Use concise academic prose with cautious claims, technical terms, and no promotional language.",
    ),
    "blog": DatasetPromptSpec(
        fact_source_name="blog article",
        system_role="You write controlled, layout-matched blog posts.",
        assignment="Write a blog post passage using only FACT_CARD.",
        output_contract="Return the blog post text only. Do not include metadata, bullets, markdown, or explanations.",
        style_notes="Use readable blog prose with concrete details. Avoid generic motivational wrap-ups.",
    ),
    "product_review": DatasetPromptSpec(
        fact_source_name="consumer product review",
        system_role="You write controlled, layout-matched consumer product reviews.",
        assignment="Write a product review using only FACT_CARD.",
        output_contract="Return the review text only. Do not include star ratings, bullets, markdown, labels, or explanations.",
        style_notes=(
            "Use consumer-review wording that matches STRUCTURE_CARD: first-person usage, casualness, "
            "disclosure-like phrasing, and praise/complaint balance. Keep every claim grounded in FACT_CARD."
        ),
    ),
    "peer_review": DatasetPromptSpec(
        fact_source_name="scientific peer review",
        system_role="You write controlled, layout-matched scientific peer review comments.",
        assignment="Write peer review comments using only FACT_CARD.",
        output_contract="Return the review comments only. Do not include reviewer identity, bullets, markdown, or explanations.",
        style_notes="Use measured reviewer language, balancing strengths, weaknesses, clarity, and evidence when supported.",
    ),
    "meeting_summary": DatasetPromptSpec(
        fact_source_name="municipal meeting transcript",
        system_role="You write controlled, layout-matched municipal agenda item summaries.",
        assignment="Write a same-style municipal agenda item summary using only FACT_CARD.",
        output_contract="Return the summary text only. Do not include bullets, markdown, speaker turns, transcript labels, or explanations.",
        style_notes=(
            "Match the municipal agenda-item register in STRUCTURE_CARD. Prefer action-item wording such as "
            "ordinance, resolution, recommendation, motion, proclamation, approval, adoption, or authorization "
            "when that is the source style. Avoid narrative recap openings like 'The council considered' unless "
            "STRUCTURE_CARD is already narrative."
        ),
    ),
    "translation": DatasetPromptSpec(
        fact_source_name="foreign-language source text",
        system_role="You translate controlled, layout-matched texts into English.",
        assignment="Translate the provided source text into English.",
        output_contract="Return the English translation only. Do not include labels, language names, bullets, markdown, or explanations.",
        style_notes="Use natural English while preserving the source meaning, order of information, names, numbers, and concrete details.",
    ),
    "xsum": DatasetPromptSpec(
        fact_source_name="news report",
        system_role="You write controlled, layout-matched news reports.",
        assignment="Write a news report body using only FACT_CARD.",
        output_contract="Return article body only. Do not include a headline, bullets, labels, markdown, or explanations.",
        style_notes="Use concise news prose with concrete attribution when supported.",
    ),
    "squad": DatasetPromptSpec(
        fact_source_name="knowledge passage",
        system_role="You write controlled, layout-matched explanatory passages.",
        assignment="Write an encyclopedic explanatory passage using only FACT_CARD.",
        output_contract="Return passage text only. Do not use question-answer format, headings, bullets, markdown, or explanations.",
        style_notes="Use neutral, informational prose that explains the topic without sounding like a news article.",
    ),
    "wp": DatasetPromptSpec(
        fact_source_name="fiction prompt and story passage",
        system_role="You write controlled, layout-matched narrative fiction passages.",
        assignment="Write a narrative fiction passage using only FACT_CARD.",
        output_contract="Return story text only. Do not include a title, bullets, markdown, or explanations.",
        style_notes="Use scene, action, and character details. Avoid news-style attribution unless FACT_CARD clearly requires dialogue.",
    ),
    "pubmed": DatasetPromptSpec(
        fact_source_name="biomedical question-answer entry",
        system_role="You write controlled PubMedQA-style biomedical question-answer entries.",
        assignment="Write a concise biomedical QA entry using only FACT_CARD.",
        output_contract=(
            "Return exactly one entry in this format: Question: ... Answer: ... "
            "Both markers must appear once. Do not add headings, bullets, markdown, or explanations."
        ),
        style_notes="Use cautious biomedical language and keep the answer focused on the supported finding.",
    ),
}


EDIT_PROMPTS: dict[str, EditPromptSpec] = {
    "essay": EditPromptSpec(
        editor_role="You lightly revise IELTS Writing Task 2 responses for a realistic AIGT benchmark.",
        domain_instruction=(
            "Keep the response plausible as an IELTS learner essay. Improve awkward phrasing, flow, and clarity, "
            "but do not make it sound like a polished professional native-speaker essay."
        ),
        preserve_format="Keep the essay body only, with no greeting, title, score notes, headings, bullets, or sign-off.",
    ),
    "arxiv_abstract": EditPromptSpec(
        editor_role="You lightly revise research-paper abstracts for a realistic AIGT benchmark.",
        domain_instruction=(
            "Keep concise academic abstract style. Tighten vague phrasing, cautious claims, and technical wording, "
            "but do not add results, methods, citations, or broader claims."
        ),
        preserve_format="Keep a single abstract body with no title, section heading, bullets, markdown, or commentary.",
    ),
    "blog": EditPromptSpec(
        editor_role="You lightly revise blog passages for a realistic AIGT benchmark.",
        domain_instruction=(
            "Make the prose read like a natural edited blog passage. Smooth stiff transitions and repetitive phrasing, "
            "but do not introduce code blocks, CLI commands, metadata, numbered steps, or new examples."
        ),
        preserve_format="Return article body text only, preserving the existing paragraph style unless a local sentence edit requires otherwise.",
    ),
    "product_review": EditPromptSpec(
        editor_role="You lightly revise consumer product reviews for a realistic AIGT benchmark.",
        domain_instruction=(
            "Make the review sound like a real customer edited it. Keep concrete personal experience and the praise/complaint balance, "
            "but do not turn it into ad copy, product metadata, a listing, or a star-rating summary."
        ),
        preserve_format="Keep exactly one paragraph. Do not add ratings, bullets, markdown, labels, or product-spec sections.",
    ),
    "peer_review": EditPromptSpec(
        editor_role="You lightly revise scientific peer-review comments for a realistic AIGT benchmark.",
        domain_instruction=(
            "Preserve reviewer judgment, strengths, weaknesses, and suggestions. Improve measured reviewer tone and remove repetitive phrasing, "
            "but do not add a recommendation, reviewer identity, scores, or unsupported criticism."
        ),
        preserve_format="Return reviewer comments only, without labels, bullets, markdown, or a separate recommendation line.",
    ),
    "meeting_summary": EditPromptSpec(
        editor_role="You lightly revise municipal agenda/action summaries for a realistic AIGT benchmark.",
        domain_instruction=(
            "Keep compact municipal action-item wording. Improve clarity while preserving the action, department/body, dates, amounts, and decisions. "
            "Do not convert the text into a narrative meeting recap."
        ),
        preserve_format="Return only the municipal summary/action text, without speaker turns, transcript labels, bullets, markdown, or commentary.",
    ),
    "translation": EditPromptSpec(
        editor_role="You lightly copyedit English translations for a realistic AIGT benchmark.",
        domain_instruction=(
            "Improve natural English while preserving the existing meaning, names, numbers, dates, and information order. "
            "Do not retranslate from a hidden source, summarize, localize, or add missing context."
        ),
        preserve_format="Return English text only, without translation labels, language names, markdown, translator notes, or explanations.",
    ),
}


def fact_extraction_messages(
    dataset: str,
    source_text: str,
    *,
    max_core_facts: int = 12,
    max_entities: int = 15,
    max_numbers: int = 8,
    max_quotes: int = 3,
) -> list[Message]:
    spec = _spec(dataset)
    system = (
        f"You extract factual or narrative content from a {spec.fact_source_name} "
        "for a controlled text generation dataset.\n"
        "Return valid JSON only. Do not invent details.\n"
        "Preserve named entities, dates, numbers, and direct quotes only if they appear in the source.\n"
        "Do not copy non-quoted phrases of 6 or more consecutive words from the source."
    )
    if dataset == "wp":
        source_label = "SOURCE PASSAGE"
        fact_word = "story facts"
    elif dataset == "pubmed":
        source_label = "PUBMED QA ENTRY"
        fact_word = "biomedical facts"
    elif dataset == "essay":
        source_label = "ESSAY RESPONSE"
        fact_word = "essay content facts"
    elif dataset == "arxiv_abstract":
        source_label = "ABSTRACT"
        fact_word = "research facts"
    elif dataset == "blog":
        source_label = "BLOG ARTICLE"
        fact_word = "article facts"
    elif dataset == "product_review":
        source_label = "PRODUCT REVIEW"
        fact_word = "review facts"
    elif dataset == "peer_review":
        source_label = "PEER REVIEW"
        fact_word = "review assessment facts"
    elif dataset == "meeting_summary":
        source_label = "MEETING TRANSCRIPT"
        fact_word = "meeting-summary facts"
    else:
        source_label = "SOURCE PASSAGE"
        fact_word = "facts"
    extra_fact_instruction = ""
    if dataset == "meeting_summary":
        extra_fact_instruction = (
            "- Prioritize the agenda item, proposed action, final decision, vote/result, responsible body, "
            "affected program/project, amounts, dates, and district/location; ignore procedural filler and routine speaker turns.\n"
        )
    user = f"""Extract a FACT_CARD from the {spec.fact_source_name} below.

Requirements:
- Output one JSON object only.
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
- core_facts must be atomic, concise {fact_word}.
- Include at most {max_core_facts} core facts.
- Include at most {max_entities} entities.
- Include at most {max_numbers} numbers.
- Include at most {max_quotes} direct quotes.
- If there is no direct quote, return an empty quotes list.
{extra_fact_instruction.rstrip()}
- Return JSON only. No markdown.

{source_label}:
{source_text}"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def fact_repair_messages(dataset: str, bad_text: str, error: str) -> list[Message]:
    spec = _spec(dataset)
    system = "You repair LLM JSON for a controlled generation pipeline. Return JSON only."
    user = f"""The previous FACT_CARD response for a {spec.fact_source_name} was invalid.

Error:
{error}

Bad response:
{bad_text}

Return one corrected JSON object with these keys only:
event_type, topic_tags, entities, core_facts, numbers, quotes, constraints.
Do not add source_text/original_text/raw_article fields.
Return JSON only."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generation_messages(
    dataset: str,
    structure_card: dict[str, Any],
    fact_card: dict[str, Any],
    *,
    template_id: str = "template_01",
    fail_reasons: list[str] | None = None,
) -> list[Message]:
    if dataset == "translation":
        return translation_generation_messages(
            structure_card,
            fact_card,
            template_id=template_id,
            fail_reasons=fail_reasons,
        )
    spec = _spec(dataset)
    prompt_structure_card = effective_structure_card(dataset, structure_card)
    structure_card_json = json.dumps(_strip_structure_metadata(prompt_structure_card), ensure_ascii=False, indent=2)
    fact_card_json = json.dumps(_strip_fact_metadata(fact_card), ensure_ascii=False, indent=2)
    shape = _shape_summary(prompt_structure_card, fact_card)
    assignment = _assignment_for_template(spec, template_id)
    dataset_note = _dataset_generation_note(dataset, prompt_structure_card)
    layout_contract = _layout_contract(dataset, prompt_structure_card)
    retry_note = ""
    if fail_reasons:
        retry_note = (
            "The previous attempt needs revision. Address these validation notes while still using only FACT_CARD:\n"
            + "\n".join(f"- {reason}" for reason in fail_reasons)
            + "\n\n"
        )
    system = (
        f"{spec.system_role}\n"
        "Use STRUCTURE_CARD only as the layout plan and FACT_CARD only as the content source.\n"
        "Do not invent facts, names, dates, numbers, quotations, scenes, or outcomes."
    )
    user = f"""{retry_note}{assignment}

Shape summary:
{shape}

Priority order:
1. Match the layout contract exactly.
2. Follow paragraph order and sentence-count hints from STRUCTURE_CARD.
3. Use FACT_CARD as the only content source.
4. Target word counts are soft length goals; do not pad with unsupported details.
5. Preserve the dataset format: {spec.output_contract}

Layout contract:
{layout_contract}

Layout instructions:
- Separate paragraphs with exactly one blank line unless the dataset format naturally requires one paragraph.
- {dataset_note}
- Do not output JSON.
- Do not mention STRUCTURE_CARD, FACT_CARD, or the source passage.
- Stop after the planned content; do not append summaries or explanations.

Content instructions:
- Reuse names, terms, numbers, and short factual wording only when supported by FACT_CARD.
- Use quotation marks only for exact quotes listed in FACT_CARD.quotes.
- If no quote is listed, avoid quotation marks.
- {_content_format_note(dataset)}

Style instructions:
- {spec.style_notes}
- Prefer direct prose over stock transitions, cliches, vague intensifiers, and broad essay endings.
- Keep comma use natural and avoid long comma-chained sentences.
- Avoid em dashes and en dashes.
- Numbers should appear only when supported by FACT_CARD.
- These are style controls, not permission to remove required facts.

STRUCTURE_CARD:
{structure_card_json}

FACT_CARD:
{fact_card_json}"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def translation_generation_messages(
    structure_card: dict[str, Any],
    source_card: dict[str, Any],
    *,
    template_id: str = "template_01",
    fail_reasons: list[str] | None = None,
) -> list[Message]:
    spec = _spec("translation")
    prompt_structure_card = effective_structure_card("translation", structure_card)
    structure_card_json = json.dumps(_strip_structure_metadata(prompt_structure_card), ensure_ascii=False, indent=2)
    source_language = source_card.get("source_language") or source_card.get("source_language_code") or "the source language"
    source_language_code = source_card.get("source_language_code") or ""
    source_text = source_card.get("source_text") or ""
    assignment = _assignment_for_template(spec, template_id)
    layout_contract = _layout_contract("translation", prompt_structure_card)
    retry_note = ""
    if fail_reasons:
        retry_note = (
            "The previous translation needs revision. Address these validation notes while preserving the source meaning:\n"
            + "\n".join(f"- {reason}" for reason in fail_reasons)
            + "\n\n"
        )
    system = (
        f"{spec.system_role}\n"
        "Use SOURCE_TEXT as the only semantic source and STRUCTURE_CARD only as the English layout plan.\n"
        "Do not summarize, omit, embellish, or add facts."
    )
    user = f"""{retry_note}{assignment}

Source language: {source_language} ({source_language_code})
Target language: English

Priority order:
1. Translate SOURCE_TEXT faithfully into English.
2. Match the layout contract and sentence-count hints from STRUCTURE_CARD.
3. Keep names, numbers, dates, product names, and quoted material when present.
4. Preserve the source order unless English grammar requires a local adjustment.
5. Return only the English translation.

Layout contract:
{layout_contract}

Style instructions:
- {spec.style_notes}
- Avoid translation labels such as "Translation:" or "English:".
- Avoid markdown, bullets, headings, and explanations unless the source is genuinely list-like and STRUCTURE_CARD plans separate paragraphs.
- Do not mention SOURCE_TEXT or STRUCTURE_CARD.

STRUCTURE_CARD:
{structure_card_json}

SOURCE_TEXT:
{source_text}"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def edit_messages(dataset: str, original: str, sampled: str) -> list[Message]:
    spec = _spec(dataset)
    edit_spec = _edit_spec(dataset)
    _ = original
    system = (
        f"{edit_spec.editor_role}\n"
        "You are simulating one light-to-medium AI revision pass over another AI-written text.\n"
        "Return only the revised text."
    )
    user = f"""Revise the AI-written text below so it reads like a natural human-edited version.

Requirements:
- Preserve the meaning, entities, numbers, and claims in the AI text.
- Remove overly generic AI phrasing, stiff transitions, repetitive structure, and inflated wording.
- Keep the same domain format: {spec.output_contract}
- Keep the length reasonably close to the AI text.
- Do not add facts, names, dates, numbers, citations, products, examples, outcomes, or quotes.
- Do not use or imitate any hidden human reference; edit only the AI text below.
- Keep the paragraph count unless the domain-specific instruction says otherwise.
- Return revised text only. No markdown or explanations.

Domain-specific revision:
- {edit_spec.domain_instruction}
- {edit_spec.preserve_format}

General style:
- {spec.style_notes}

AI-WRITTEN TEXT:
{sampled}"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _spec(dataset: str) -> DatasetPromptSpec:
    if dataset not in DATASET_PROMPTS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return DATASET_PROMPTS[dataset]


def _edit_spec(dataset: str) -> EditPromptSpec:
    if dataset not in EDIT_PROMPTS:
        raise ValueError(f"Unsupported edit dataset: {dataset}")
    return EDIT_PROMPTS[dataset]


def _dataset_generation_note(dataset: str, structure_card: dict[str, Any]) -> str:
    expected_paragraphs = int(structure_card.get("num_paragraphs") or 0)
    if dataset == "blog":
        return (
            "For blogs, use short readable paragraphs. If the source layout looks like code, commands, "
            "or many tiny lines, rewrite it as a compact article body instead of recreating code blocks."
        )
    if dataset == "meeting_summary":
        surface_style = structure_card.get("surface_style") or {}
        style_instruction = surface_style.get("style_instruction") or "Use compact agenda-item summary wording."
        summary_kind = surface_style.get("summary_kind") or "agenda_item_summary"
        return (
            f"For meeting summaries, match this surface style: {summary_kind}. {style_instruction} "
            "Do not convert an agenda action item into a narrative recap."
        )
    if dataset == "product_review":
        surface_style = structure_card.get("surface_style") or {}
        review_register = surface_style.get("review_register") or "consumer_product_review"
        style_instruction = surface_style.get("style_instruction") or "Use concrete consumer-review wording."
        return (
            f"For product reviews, match this surface style: {review_register}. {style_instruction} "
            "Do not turn the review into ad copy, a product listing, or a polished editorial summary."
        )
    if dataset == "translation":
        return "For translations, use English prose with paragraph breaks aligned to STRUCTURE_CARD; do not include labels or commentary."
    if dataset == "wp" and expected_paragraphs == 1:
        return "For this WP item, write one continuous paragraph only; do not use blank lines, scene breaks, or standalone dialogue lines."
    if dataset == "wp":
        return "For WP, keep paragraph breaks only where STRUCTURE_CARD explicitly plans them."
    if expected_paragraphs == 1 and dataset != "pubmed":
        return "Write one continuous paragraph only. Do not insert blank lines, section breaks, bullet-like lines, or standalone attribution lines."
    return "Keep paragraph breaks aligned with STRUCTURE_CARD."


def _content_format_note(dataset: str) -> str:
    if dataset == "pubmed":
        return "For PubMed, keep the text as one `Question: ... Answer: ...` entry."
    if dataset == "essay":
        return "For IELTS essays, write only the essay body; do not use letter greetings, sign-offs, headings, or bullet points."
    if dataset == "arxiv_abstract":
        return "For abstracts, write one abstract body without title or section labels."
    if dataset == "blog":
        return (
            "For blogs, write article body text only; do not include URL, author, date, tag metadata, "
            "code blocks, shell commands, import lists, or bullet lists."
        )
    if dataset == "product_review":
        return (
            "For product reviews, write review text only; do not prepend a star rating or product metadata. "
            "Use reviewer-style wording and keep the review as a single paragraph."
        )
    if dataset == "peer_review":
        return "For peer reviews, write reviewer comments only; do not include reviewer identity or recommendation labels."
    if dataset == "meeting_summary":
        return (
            "For meeting summaries, write only the municipal summary/action text; do not include transcript labels, "
            "speaker turns, timestamps, bullets, markdown, or explanatory commentary."
        )
    if dataset == "translation":
        return "For translations, write English translation text only; do not include source-language labels, language names, markdown, or translator notes."
    return "Preserve the requested dataset format."


def _layout_contract(dataset: str, structure_card: dict[str, Any]) -> str:
    expected_paragraphs = int(structure_card.get("num_paragraphs") or 0)
    paragraphs = structure_card.get("paragraphs", []) or []
    sentence_counts = [int(paragraph.get("sentence_count") or 1) for paragraph in paragraphs]
    if dataset == "pubmed":
        return (
            "- Output exactly one PubMedQA entry with one `Question:` marker and one `Answer:` marker.\n"
            "- Keep the question and answer in the same entry. Do not create extra sections or paragraph breaks."
        )
    if expected_paragraphs <= 1:
        target_sentences = sentence_counts[0] if sentence_counts else 1
        return (
            "- Output exactly one paragraph.\n"
            "- Do not use any blank line. A blank line creates a validation failure.\n"
            f"- Keep all content in that one paragraph with about {target_sentences} sentence(s).\n"
            "- Do not split background, details, and conclusion into separate paragraphs."
        )
    lines = [
        f"- Output exactly {expected_paragraphs} paragraphs.",
        "- Use exactly one blank line between paragraphs.",
        "- Do not add extra intro, conclusion, heading, or afterword paragraphs.",
    ]
    if dataset == "blog":
        lines.extend(
            [
                "- Treat each planned paragraph as a short blog paragraph, usually 1-2 sentences.",
                "- Do not use list markers, numbered steps, code fences, or standalone code/command lines.",
                "- If the factual source is tutorial-like, summarize the workflow in prose instead of reproducing code.",
            ]
        )
    if dataset == "product_review":
        lines.extend(
            [
                "- Keep the output in consumer-review prose, not product-description copy.",
                "- Preserve the planned paragraph count from STRUCTURE_CARD.",
                "- Do not add list markers, section labels, or metadata lines.",
            ]
        )
    if sentence_counts:
        counts = ", ".join(f"P{idx + 1}: {count}" for idx, count in enumerate(sentence_counts[:10]))
        lines.append(f"- Sentence-count plan: {counts}.")
    return "\n".join(lines)


def _assignment_for_template(spec: DatasetPromptSpec, template_id: str) -> str:
    variants = {
        "template_01": spec.assignment,
        "template_02": spec.assignment.replace("Write", "Compose", 1),
        "template_03": spec.assignment.replace("using only FACT_CARD", "from the provided FACT_CARD"),
    }
    return variants.get(template_id, spec.assignment)


def _strip_structure_metadata(card: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "source_word_count",
        "length_boost_multiplier",
        "target_total_word_count",
        "num_paragraphs",
        "global_constraints",
        "paragraphs",
        "surface_style",
    }
    return {key: value for key, value in card.items() if key in allowed}


def _strip_fact_metadata(card: dict[str, Any]) -> dict[str, Any]:
    allowed = {"event_type", "topic_tags", "entities", "core_facts", "numbers", "quotes", "constraints"}
    return {key: value for key, value in card.items() if key in allowed}


def _shape_summary(structure_card: dict[str, Any], fact_card: dict[str, Any]) -> str:
    paragraph_plan = []
    for paragraph in structure_card.get("paragraphs", []) or []:
        pid = paragraph.get("pid")
        sentence_count = paragraph.get("sentence_count", 1)
        target_words = paragraph.get("target_word_count")
        if pid is not None:
            paragraph_plan.append(f"P{pid}:{sentence_count}s/~{target_words}w")
    plan = "; ".join(paragraph_plan[:35])
    if len(paragraph_plan) > 35:
        plan += "; ..."
    return (
        f"- Required paragraphs: {structure_card.get('num_paragraphs')}\n"
        f"- Target total words: about {structure_card.get('target_total_word_count')}\n"
        f"- Paragraph plan: {plan}\n"
        f"- Core facts: {len(fact_card.get('core_facts') or [])}\n"
        f"- Entities: {len(fact_card.get('entities') or [])}\n"
        f"- Numbers: {len(fact_card.get('numbers') or [])}\n"
        f"- Quotes: {len(fact_card.get('quotes') or [])}"
    )

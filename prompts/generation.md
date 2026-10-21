# Generation prompts

Verbatim message templates and domain-specific strings extracted from `data_generation/scripts/ultimate_controlled/prompts.py`. Braced Python expressions are runtime substitutions, not text to send literally. The source module is authoritative for layout-card processing and conditional/retry instructions.

## Six non-translation domains

### System message

```text
{spec.system_role}
Use STRUCTURE_CARD only as the layout plan and FACT_CARD only as the content source.
Do not invent facts, names, dates, numbers, quotations, scenes, or outcomes.
```

### User message

```text
{retry_note}{assignment}

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
{fact_card_json}
```

## Domain-specific strings

### essay

**system_role**

```text
You write controlled, layout-matched IELTS Writing Task 2 essays.
```

**assignment**

```text
Write an IELTS Writing Task 2 essay using only FACT_CARD.
```

**output_contract**

```text
Return the response text only. Do not include scoring notes, labels, markdown, or explanations.
```

**style_notes**

```text
Use natural exam-essay prose with a clear position, topic-focused body paragraphs, and a concise conclusion. Keep the language plausible for a learner essay rather than polished professional writing.
```

### arxiv_abstract

**system_role**

```text
You write controlled, layout-matched research abstracts.
```

**assignment**

```text
Write a research paper abstract using only FACT_CARD.
```

**output_contract**

```text
Return the abstract body only. Do not include a title, section heading, bullets, markdown, or explanations.
```

**style_notes**

```text
Use concise academic prose with cautious claims, technical terms, and no promotional language.
```

### blog

**system_role**

```text
You write controlled, layout-matched blog posts.
```

**assignment**

```text
Write a blog post passage using only FACT_CARD.
```

**output_contract**

```text
Return the blog post text only. Do not include metadata, bullets, markdown, or explanations.
```

**style_notes**

```text
Use readable blog prose with concrete details. Avoid generic motivational wrap-ups.
```

### product_review

**system_role**

```text
You write controlled, layout-matched consumer product reviews.
```

**assignment**

```text
Write a product review using only FACT_CARD.
```

**output_contract**

```text
Return the review text only. Do not include star ratings, bullets, markdown, labels, or explanations.
```

**style_notes**

```text
Use consumer-review wording that matches STRUCTURE_CARD: first-person usage, casualness, disclosure-like phrasing, and praise/complaint balance. Keep every claim grounded in FACT_CARD.
```

### peer_review

**system_role**

```text
You write controlled, layout-matched scientific peer review comments.
```

**assignment**

```text
Write peer review comments using only FACT_CARD.
```

**output_contract**

```text
Return the review comments only. Do not include reviewer identity, bullets, markdown, or explanations.
```

**style_notes**

```text
Use measured reviewer language, balancing strengths, weaknesses, clarity, and evidence when supported.
```

### meeting_summary

**system_role**

```text
You write controlled, layout-matched municipal agenda item summaries.
```

**assignment**

```text
Write a same-style municipal agenda item summary using only FACT_CARD.
```

**output_contract**

```text
Return the summary text only. Do not include bullets, markdown, speaker turns, transcript labels, or explanations.
```

**style_notes**

```text
Match the municipal agenda-item register in STRUCTURE_CARD. Prefer action-item wording such as ordinance, resolution, recommendation, motion, proclamation, approval, adoption, or authorization when that is the source style. Avoid narrative recap openings like 'The council considered' unless STRUCTURE_CARD is already narrative.
```

### translation

**system_role**

```text
You translate controlled, layout-matched texts into English.
```

**assignment**

```text
Translate the provided source text into English.
```

**output_contract**

```text
Return the English translation only. Do not include labels, language names, bullets, markdown, or explanations.
```

**style_notes**

```text
Use natural English while preserving the source meaning, order of information, names, numbers, and concrete details.
```

## Translation-specific template

### System message

```text
{spec.system_role}
Use SOURCE_TEXT as the only semantic source and STRUCTURE_CARD only as the English layout plan.
Do not summarize, omit, embellish, or add facts.
```

### User message

```text
{retry_note}{assignment}

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
{source_text}
```

## Runtime fields and variants

- `STRUCTURE_CARD` supplies paragraph count, sentence counts and soft word-count targets. `FACT_CARD` supplies content for non-translation tasks.
- Translation uses `SOURCE_TEXT` as the semantic source and the English reference structure as layout.
- `template_01` uses the assignment as written; `template_02` changes the first `Write` to `Compose`; `template_03` changes `using only FACT_CARD` to `from the provided FACT_CARD`.
- When a generation fails validation, `retry_note` enumerates its validation reasons. The complete templates and conditional format rules remain in the source module.

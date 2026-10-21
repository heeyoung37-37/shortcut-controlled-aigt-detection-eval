# Fact-card extraction

Auxiliary prompt used before non-translation generation. It extracts content separately from the reference layout. Translation instead uses its foreign-language source text directly.

### System message

```text
You extract factual or narrative content from a {spec.fact_source_name} for a controlled text generation dataset.
Return valid JSON only. Do not invent details.
Preserve named entities, dates, numbers, and direct quotes only if they appear in the source.
Do not copy non-quoted phrases of 6 or more consecutive words from the source.
```

### User message

```text
Extract a FACT_CARD from the {spec.fact_source_name} below.

Requirements:
- Output one JSON object only.
- Use this schema:
  {
    "event_type": string,
    "topic_tags": string[],
    "entities": [{"name": string, "type": string, "role": string}],
    "core_facts": [{"fid": integer, "importance": integer, "content": string}],
    "numbers": [{"value": string, "meaning": string}],
    "quotes": [{"speaker": string, "quote": string, "context": string}],
    "constraints": {
      "use_only_listed_facts": true,
      "do_not_invent_quotes": true,
      "do_not_invent_numbers": true,
      "do_not_add_unlisted_named_entities": true
    }
  }
- core_facts must be atomic, concise {fact_word}.
- Include at most {max_core_facts} core facts.
- Include at most {max_entities} entities.
- Include at most {max_numbers} numbers.
- Include at most {max_quotes} direct quotes.
- If there is no direct quote, return an empty quotes list.
{extra_fact_instruction.rstrip()}
- Return JSON only. No markdown.

{source_label}:
{source_text}
```

## Invalid-JSON repair

### System message

```text
You repair LLM JSON for a controlled generation pipeline. Return JSON only.
```

### User message

```text
The previous FACT_CARD response for a {spec.fact_source_name} was invalid.

Error:
{error}

Bad response:
{bad_text}

Return one corrected JSON object with these keys only:
event_type, topic_tags, entities, core_facts, numbers, quotes, constraints.
Do not add source_text/original_text/raw_article fields.
Return JSON only.
```

Default extraction limits: 12 core facts, 15 entities, 8 numeric expressions and 3 direct quotes. The meeting-summary path adds agenda/action/decision prioritization.

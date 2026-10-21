# AI editing prompts

Version: `domain_light_medium_no_reference_v1`. The editor sees only the AI-written text. Although the source function accepts `original`, it explicitly does not insert that human reference into the prompt.

### System message

```text
{edit_spec.editor_role}
You are simulating one light-to-medium AI revision pass over another AI-written text.
Return only the revised text.
```

### User message

```text
Revise the AI-written text below so it reads like a natural human-edited version.

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
{sampled}
```

## essay

**editor_role**

```text
You lightly revise IELTS Writing Task 2 responses for a realistic AIGT benchmark.
```

**domain_instruction**

```text
Keep the response plausible as an IELTS learner essay. Improve awkward phrasing, flow, and clarity, but do not make it sound like a polished professional native-speaker essay.
```

**preserve_format**

```text
Keep the essay body only, with no greeting, title, score notes, headings, bullets, or sign-off.
```


## arxiv_abstract

**editor_role**

```text
You lightly revise research-paper abstracts for a realistic AIGT benchmark.
```

**domain_instruction**

```text
Keep concise academic abstract style. Tighten vague phrasing, cautious claims, and technical wording, but do not add results, methods, citations, or broader claims.
```

**preserve_format**

```text
Keep a single abstract body with no title, section heading, bullets, markdown, or commentary.
```


## blog

**editor_role**

```text
You lightly revise blog passages for a realistic AIGT benchmark.
```

**domain_instruction**

```text
Make the prose read like a natural edited blog passage. Smooth stiff transitions and repetitive phrasing, but do not introduce code blocks, CLI commands, metadata, numbered steps, or new examples.
```

**preserve_format**

```text
Return article body text only, preserving the existing paragraph style unless a local sentence edit requires otherwise.
```


## product_review

**editor_role**

```text
You lightly revise consumer product reviews for a realistic AIGT benchmark.
```

**domain_instruction**

```text
Make the review sound like a real customer edited it. Keep concrete personal experience and the praise/complaint balance, but do not turn it into ad copy, product metadata, a listing, or a star-rating summary.
```

**preserve_format**

```text
Keep exactly one paragraph. Do not add ratings, bullets, markdown, labels, or product-spec sections.
```


## peer_review

**editor_role**

```text
You lightly revise scientific peer-review comments for a realistic AIGT benchmark.
```

**domain_instruction**

```text
Preserve reviewer judgment, strengths, weaknesses, and suggestions. Improve measured reviewer tone and remove repetitive phrasing, but do not add a recommendation, reviewer identity, scores, or unsupported criticism.
```

**preserve_format**

```text
Return reviewer comments only, without labels, bullets, markdown, or a separate recommendation line.
```


## meeting_summary

**editor_role**

```text
You lightly revise municipal agenda/action summaries for a realistic AIGT benchmark.
```

**domain_instruction**

```text
Keep compact municipal action-item wording. Improve clarity while preserving the action, department/body, dates, amounts, and decisions. Do not convert the text into a narrative meeting recap.
```

**preserve_format**

```text
Return only the municipal summary/action text, without speaker turns, transcript labels, bullets, markdown, or commentary.
```


## translation

**editor_role**

```text
You lightly copyedit English translations for a realistic AIGT benchmark.
```

**domain_instruction**

```text
Improve natural English while preserving the existing meaning, names, numbers, dates, and information order. Do not retranslate from a hidden source, summarize, localize, or add missing context.
```

**preserve_format**

```text
Return English text only, without translation labels, language names, markdown, translator notes, or explanations.
```


## Optional length-preserving retry

Used only when `--length-guard` is enabled.

```text
Length-preserving retry instruction:
- The AI-written text has about {source_words} words.
- Hard requirement: this is a copyedit, not a summary.
- Do not summarize, compress, omit examples, merge away supporting details, or shorten lists of facts.
- Keep the revised text close to the original AI-written text length, ideally within 90%-110% and never below 75%.
- Preserve roughly the same number of sentences and all substantive examples/details.
- Make local copyedits and phrasing improvements while preserving all substantive content.
- Preserve the existing paragraph count unless the domain-specific instruction requires one paragraph.
- Return revised text only.
```

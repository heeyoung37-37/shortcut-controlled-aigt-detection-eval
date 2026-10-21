# Construction prompts

| File | Purpose |
|---|---|
| [generation.md](generation.md) | Shared generation template, seven task specifications, and translation-specific template |
| [fact_extraction.md](fact_extraction.md) | Fact-card extraction and invalid-JSON repair |
| [ai_edit.md](ai_edit.md) | Light-to-medium AI revision and optional length-preserving retry |
| [human_edit.md](human_edit.md) | Original Korean / English annotator guidelines |

These documents were extracted from the construction code. Runtime fields are marked in braces. Dynamic layout contracts, model/provider settings and retry reasons are implemented in [`data_generation/`](../data_generation/README.md).

The formatting/lexical controls reduce easy surface cues; they do not imply that every possible shortcut is eliminated. Human instructions describe a manual editing step, not an additional LLM generation step.

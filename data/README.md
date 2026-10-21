# AIGT-SignalBench data card

## Composition

**1,260 human–AI pairs = 7 domains × 6 generators × 3 conditions × 10 pairs.**

| File | Writing task | Pairs |
|---|---|---:|
| `essay.jsonl` | IELTS-style essay writing | 180 |
| `arxiv_abstract.jsonl` | Research abstract writing | 180 |
| `blog.jsonl` | Blog writing | 180 |
| `product_review.jsonl` | Product review writing | 180 |
| `peer_review.jsonl` | Scientific peer-review writing | 180 |
| `meeting_summary.jsonl` | Meeting / municipal agenda summarization | 180 |
| `translation.jsonl` | Translation into English | 180 |

Generator labels are preserved exactly as recorded: `claude-sonnet-4-6`, `deepseek-v4-pro`, `gemini-2.5-pro`, `gpt-3.5-turbo`, `gpt-4`, `gpt-5.5`.

Conditions: `raw_ai`, `ai_edited`, and `human_edited`, each with 420 pairs.

## Schema

Each UTF-8 JSONL line has exactly these five fields:

```json
{
  "original": "Human reference text in the final benchmark",
  "sampled": "AI-side generated or edited text",
  "domain_label": "essay",
  "model_label": "gpt-4",
  "variant_type": "raw_ai"
}
```

- `original` is the released human reference; `sampled` is the AI-side text. Human editing does **not** change the provenance label to human-only writing.
- The variant groups use different source instances. They are **not** three aligned edits of the same 420 original texts.
- The release contains 1,260 distinct `original` texts and 1,260 distinct `sampled` texts. No train/dev/test split is prescribed by this export.
- If creating detector splits, keep a human text and its paired AI text in the same split. Do not treat the two texts as independent splitting units.

## Preprocessing and integrity

These are the final, length-matched public-export texts, not untouched raw downloads. Construction includes task-specific formatting cleanup and boundary-aware length adjustment; either side of a pair can be shortened when needed. The human-edit export and final cleanup are already reflected in these files.

Dataset integrity:

- All seven files parse as JSONL and contain 180 pairs each.
- Every domain × model × variant cell has 10 pairs.
- No blank text fields or duplicate full texts within either role.
- All 1,260 pairs match the final benchmark export.

[`manifest.json`](manifest.json) records row counts, file sizes, and SHA-256 checksums.

## Sources and release notes

The construction configuration names the following upstream human-source datasets:

| Domain | Source |
|---|---|
| Essay | [chillies/IELTS-writing-task-2-evaluation](https://huggingface.co/datasets/chillies/IELTS-writing-task-2-evaluation) |
| Abstract | [gfissore/arxiv-abstracts-2021](https://huggingface.co/datasets/gfissore/arxiv-abstracts-2021) |
| Blog | [Kamaljp/medium_articles](https://huggingface.co/datasets/Kamaljp/medium_articles) |
| Product review | [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) |
| Peer review | [allenai/PeerRead](https://github.com/allenai/PeerRead) |
| Meeting summary | [huuuyeah/meetingbank](https://huggingface.co/datasets/huuuyeah/meetingbank) |
| Translation | [HuggingFaceFW/finetranslations](https://huggingface.co/datasets/HuggingFaceFW/finetranslations) |

Source texts remain subject to their original rights and terms. The source links do not grant additional permissions; redistribution rights have not been established for all texts in this collection. Raw source downloads and annotator identities are not included.


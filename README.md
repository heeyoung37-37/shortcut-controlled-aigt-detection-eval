<div align="center">

# AIGT-SignalBench

### Shortcut-Controlled Evaluation for AI-Generated Text Detection

**Hee-Young Jung\* · Jae-Eon Ahn\* · Seong-Jin Park · Kang-Min Kim**  
Kyung Hee University · The Catholic University of Korea  
\* Equal contribution

**[Dataset](data/README.md) · [Prompts](prompts/README.md) · [Data Generation](data_generation/README.md) · [Citation](CITATION.cff)**

</div>

> A compact release of shortcut-controlled human–AI text pairs, construction prompts, and the code used to generate and edit the benchmark.

[![Interactive benchmark explorer](assets/preview.webp)](index.html)

### Interactive explorer

Download this folder and open **[`index.html`](index.html)** in a browser. No installation, server, or API key is needed. Explore all 1,260 pairs by domain, generator, and condition; compare texts, browse the original prompts, and save selected pairs.

GitHub's README displays the preview image. The interactive page runs locally, or as a website after hosting is enabled. To refresh the offline content after editing data or prompt documents, run `python3 build_site.py`.

## At a glance

| Domains | Generators | Editing conditions | Human–AI pairs |
|:---:|:---:|:---:|:---:|
| **7** | **6** | **3** | **1,260** |

The benchmark reduces easy formatting and lexical cues while retaining realistic AI-assisted writing tasks. Each released row contains a human reference and one AI-side text. The three conditions are:

| Condition | Description | Pairs |
|---|---|---:|
| `raw_ai` | Direct AI generation | 420 |
| `ai_edited` | AI generation followed by an AI revision | 420 |
| `human_edited` | AI generation followed by human editing | 420 |

Domains: essay, research abstract, blog, product review, peer review, meeting summary, and translation into English. Every domain × generator × condition cell contains **10 pairs**.

## Use the released data

The seven domain-level JSONL files are in [`data/`](data/). No API calls are required to read them.

```python
import json
from pathlib import Path

rows = [
    json.loads(line)
    for path in sorted(Path("data").glob("*.jsonl"))
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
print(len(rows))  # 1260
```

The released files are byte-for-byte copies of the final public export. See the [data card](data/README.md) for field meanings, sources, preprocessing, and checksums.

## What is included

```text
data/              Final benchmark: 7 JSONL files + manifest + data card
prompts/           Generation, fact extraction, AI edit, and human guidelines
data_generation/   Historical preparation, generation, editing, and export code
index.html         Interactive benchmark and prompt explorer
assets/            Website styles, scripts, offline content, and preview
build_site.py      Refresh offline website content from the release files
CITATION.cff       Paper and dataset citation metadata
```

Read [`prompts/`](prompts/README.md) for the message templates and [`data_generation/`](data_generation/README.md) for the construction workflow. Detector training/evaluation experiments, result caches, model weights, API credentials, and annotator work logs are outside this release's scope.

The fixed data is the reference benchmark. Fresh API generations and new human edits will not recreate its exact bytes. See [offline checks](data_generation/README.md#offline-checks) for the validation scope.

## Citation

```bibtex
@inproceedings{jung2026shortcut,
  title = {Shortcut-Controlled Evaluation for AI-Generated Text Detection},
  author = {Jung, Hee-Young and Ahn, Jae-Eon and Park, Seong-Jin and Kim, Kang-Min},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year = {2026}
}
```

## License

Original software code in this repository is licensed under the [MIT License](LICENSE). This license does not cover third-party code, source texts, generated or edited benchmark data, or copies of dataset content in website assets. No blanket dataset license is asserted; see [source notes](data/README.md#sources-and-release-notes).

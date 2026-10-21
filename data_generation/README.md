# Data construction code

This directory preserves the code used to prepare sources, generate shortcut-controlled texts, perform AI/human editing, and assemble the benchmark. Its internal relative import layout is retained. It does not include detector training or evaluation experiments.

## Setup

```bash
cd data_generation
python -m pip install -r requirements.txt
cp api_keys.env.example api_keys.env
```

The dependency file is the historical construction snapshot, not a newly resolved lockfile. Source preparation uses Hugging Face data; generation needs your own credentials for the selected providers. Gemini uses Vertex AI and local Application Default Credentials. No real credential file or source download cache is included.

The fixed benchmark is already in [`../data/`](../data/README.md). Read it directly when using the released dataset; running this code is optional and can incur provider usage charges.

## Components

| Stage | Source files |
|---|---|
| Source download / normalization | `scripts/download_sources.py`, `normalize_sources.py`, `configs/datasets.yaml` |
| Source assignment | `scripts/build_source_shards.py` |
| Controlled generation | `scripts/generate_ultimate_controlled_fdg_dataset.py`, `ultimate_controlled/`, `grok_ultimate_gen/src/` |
| Variant partitioning | `scripts/split_variant_shards.py` |
| AI revision | `scripts/edit_ai.py`, `run_ai_edit_all.py` |
| Human editing | `scripts/human_edit_gradio.py`, `human_edit_server.py`, `export_human_edited_from_gradio.py` |
| Merge / cleanup / length matching | `scripts/merge_final_dataset.py`, `build_length_matched_benchmark.py`, task-specific postprocessing helpers |
| Validation | `scripts/validate_dataset.py`, `validate_ai_edits.py`, `tests/test_our_data_pipeline.py` |

`analyze_aigt.py` is retained because the generation validator imports its surface-feature definitions. `grok_ultimate_gen/src/` is a required helper module; its folder name does not mean Grok is one of the six released generators.

## Example commands

Run these from this directory. Commands are separate construction steps, not a claim that a fresh run will reproduce the exact released text.

```bash
# Prepare candidate human sources and model-specific source shards.
python scripts/download_sources.py --max-rows 2000
python scripts/normalize_sources.py
python scripts/build_source_shards.py --run-id final --items-per-kind 60

# Inspect the planned generation commands without calling model APIs.
python scripts/run_generation_all.py --run-id final --dry-run

# Generate using the six entries in configs/models.yaml, then partition explicitly.
python scripts/run_generation_all.py --run-id final --no-split
python scripts/split_variant_shards.py --run-id final --output-root data/curated_shards --strict

# Revise the AI-edit group.
python scripts/run_ai_edit_all.py --run-id final
```

Human editing is a manual stage. The original Gradio editor consumes domain/model FDG-format input files in `human_edit_input/` and saves to `human_edited_gradio_saved/`. These working inputs and annotation logs are intentionally not included. The simpler JSONL editor can instead work on the generated curated shards:

```bash
python scripts/human_edit_server.py \
  --run-id final --domain essay --model-label gemini-2.5-pro \
  --input-root data/curated_shards --output-root data/curated_shards
```

After the required human edits exist, the JSONL pipeline can merge them with `scripts/merge_final_dataset.py --run-id final --input-root data/curated_shards --strict`. The historical Gradio export, replacement/cleanup tools and length-matching builder are provided as construction components, rather than an unattended one-click replay of every manual curation decision.

The model APIs, runtime prompt fields, rejection/retry decisions, preprocessing and new human edits may change the generated outputs. Use the fixed seven JSONL files as the benchmark version of record.

## Offline checks

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_our_data_pipeline.py'
```

The offline suite contains 17 tests using synthetic inputs and a mock provider. It checks construction logic without making model API calls; it does not validate live provider calls, GPU execution, or dependency installation in a clean environment.

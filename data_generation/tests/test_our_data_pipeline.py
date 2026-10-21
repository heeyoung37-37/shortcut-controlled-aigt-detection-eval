import argparse
import asyncio
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import generate_ultimate_controlled_fdg_dataset as gen
from scripts.build_source_shards import build_domain_shards
from scripts.common import read_json, read_jsonl, shard_file_name, write_json, write_jsonl
from scripts.human_edit_server import HumanEditStore
from scripts.normalize_sources import normalize_domain
from scripts.text_postprocess import sentence_safe_postprocess_text
from scripts.ultimate_controlled.layout import effective_expected_paragraphs
from scripts.split_variant_shards import load_generation_records, load_structure_source_map, split_records
from scripts.edit_ai import edited_row_is_valid, make_edited_row, validate_edited_text
from scripts.ultimate_controlled.prompts import EDIT_PROMPTS, EDIT_PROMPT_VERSION, edit_messages, generation_messages
from scripts.ultimate_controlled.providers import ProviderResponse


def long_text(prefix: str, words: int = 90) -> str:
    body = " ".join(f"{prefix}{idx}" for idx in range(words))
    return f"{prefix} begins with a concrete point. {body}."


class FakeProvider:
    provider = "gemini"
    model = "gemini-2.5-pro"

    async def generate_text(self, messages, *, temperature, top_p, max_tokens, thinking_budget=None):
        joined = "\n".join(message["content"] for message in messages)
        if "Extract a FACT_CARD" in joined:
            payload = {
                "event_type": "student response",
                "topic_tags": ["education"],
                "entities": [{"name": "City Library", "type": "ORG", "role": "topic"}],
                "core_facts": [
                    {"fid": 1, "importance": 1, "content": "City Library opened study rooms for students"},
                    {"fid": 2, "importance": 2, "content": "The rooms support quiet reading and group projects"},
                ],
                "numbers": [],
                "quotes": [],
                "constraints": {
                    "use_only_listed_facts": True,
                    "do_not_invent_quotes": True,
                    "do_not_invent_numbers": True,
                    "do_not_add_unlisted_named_entities": True,
                },
            }
            return ProviderResponse(text=json.dumps(payload), usage={"total_tokens": 10})
        text = (
            "City Library opened study rooms for students this term. "
            "The rooms support quiet reading and group projects after class."
        )
        return ProviderResponse(text=text, usage={"total_tokens": 20})

    async def generate_json(self, messages, *, temperature, max_tokens, thinking_budget=None):
        response = await self.generate_text(
            messages,
            temperature=temperature,
            top_p=None,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
        )
        return json.loads(response.text), response


class OurDataPipelineTests(unittest.TestCase):
    def test_normalizers_emit_common_schema_for_all_domains(self):
        specs = {
            "essay": {"source_dataset": "essay", "split": "train", "min_words": 5, "text_field": "essay", "task_fields": ["prompt"], "metadata_fields": ["id"]},
            "arxiv_abstract": {"source_dataset": "arxiv", "split": "train", "min_words": 5, "text_field": "abstract", "task_fields": ["title", "categories"], "metadata_fields": ["id"]},
            "blog": {"source_dataset": "blog", "split": "train", "min_words": 5, "text_field": "text", "task_fields": ["title", "tags"], "metadata_fields": ["url"]},
            "product_review": {"source_dataset": "amazon", "split": "full", "min_words": 5, "text_field": "text", "task_fields": ["rating", "title"], "metadata_fields": ["asin"]},
            "peer_review": {"source_dataset": "peer", "split": "train", "min_words": 5, "text_field": "comments", "task_fields": ["title", "abstract"], "metadata_fields": ["id"]},
            "meeting_summary": {
                "source_dataset": "meetingbank",
                "split": "train",
                "min_words": 5,
                "text_field": "summary",
                "transcript_field": "transcript",
                "max_transcript_words": 120,
                "metadata_fields": ["id", "uid"],
            },
            "translation": {
                "source_dataset": "finetranslations",
                "split": "train",
                "min_words": 5,
                "text_field": "translated_text",
                "source_text_field": "og_full_text",
                "hf_configs": ["spa_Latn", "fra_Latn", "jpn_Jpan"],
                "language_names": {"spa_Latn": "Spanish", "fra_Latn": "French", "jpn_Jpan": "Japanese"},
                "metadata_fields": ["id", "og_language"],
                "min_translated_words": 5,
                "max_translated_words": 200,
                "max_segments": 4,
            },
        }
        raw_rows = {
            "essay": [{"id": "e1", "prompt": "Write about a library.", "essay": long_text("essay")}],
            "arxiv_abstract": [{"id": "a1", "title": "A Study", "categories": ["cs.CL"], "abstract": long_text("abstract")}],
            "blog": [{"url": "u1", "title": "Post", "tags": ["tech"], "text": long_text("blog")}],
            "product_review": [
                {
                    "asin": "p1",
                    "rating": 5,
                    "title": "Good",
                    "text": f"{long_text('review', 45)}<br /><br />{long_text('second', 45)}",
                }
            ],
            "peer_review": [{"id": "r1", "title": "Paper", "abstract": "Study summary", "comments": long_text("peer")}],
            "meeting_summary": [
                {
                    "id": 1,
                    "uid": "SeattleCityCouncil_06132016_Res 31669",
                    "summary": long_text("summary", 35),
                    "transcript": long_text("transcript", 180),
                }
            ],
            "translation": [
                {
                    "id": "t1",
                    "language_config": "spa_Latn",
                    "og_language": "spa_Latn",
                    "og_full_text": "La biblioteca abrió nuevas salas de estudio para estudiantes esta semana.",
                    "translated_text": long_text("translation", 40),
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for domain, rows in raw_rows.items():
                raw_path = root / f"{domain}.jsonl"
                write_jsonl(raw_path, rows)
                normalized = normalize_domain(domain, specs[domain], raw_path)
                self.assertEqual(len(normalized), 1)
                row = normalized[0]
                self.assertEqual(row["domain"], domain)
                self.assertTrue(row["source_id"].startswith(domain))
                self.assertTrue(row["human_text"])
                self.assertTrue(row["fact_source_text"])
                if domain == "meeting_summary":
                    self.assertIn("transcript_excerpt:", row["fact_source_text"])
                    self.assertNotIn("Reference text:", row["fact_source_text"])
                if domain == "translation":
                    self.assertIn("source_language: Spanish", row["fact_source_text"])
                    self.assertIn("source_text:", row["fact_source_text"])
                    self.assertNotIn("Reference text:", row["fact_source_text"])
                if domain == "product_review":
                    self.assertNotIn("<br", row["human_text"])
                    self.assertEqual(len(row["human_text"].split("\n\n")), 2)

    def test_blog_normalizer_filters_code_heavy_articles(self):
        spec = {
            "source_dataset": "blog",
            "split": "train",
            "min_words": 5,
            "text_field": "text",
            "task_fields": ["title"],
            "metadata_fields": ["url"],
        }
        code_lines = "\n\n".join(
            [
                "import dash",
                "import pandas as pd",
                "pip install dash",
                "npm install package",
                "docker build .",
                "kubectl apply -f app.yaml",
            ]
            * 12
        )
        prose = long_text("blog", 90)
        with tempfile.TemporaryDirectory() as tmp:
            raw_path = Path(tmp) / "blog.jsonl"
            write_jsonl(
                raw_path,
                [
                    {"url": "bad", "title": "Code tutorial", "text": code_lines},
                    {"url": "good", "title": "Readable post", "text": prose},
                ],
            )
            normalized = normalize_domain("blog", spec, raw_path)
        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0]["metadata"]["url"], "good")

    def test_source_shards_are_model_unique_and_ultimate_compatible(self):
        rows = [
            {
                "source_id": f"essay_{idx}",
                "domain": "essay",
                "human_text": long_text(f"s{idx}", 30),
                "fact_source_text": long_text(f"f{idx}", 30),
                "task_input": "",
                "metadata": {},
            }
            for idx in range(8)
        ]
        models = [
            {"provider": "gemini", "model": "gemini-2.5-pro", "model_label": "gemini-2.5-pro"},
            {"provider": "openai", "model": "gpt-4", "model_label": "gpt-4"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = build_domain_shards(rows, domain="essay", model_specs=models, output_root=root, items_per_kind=2, seed=7)
            left = set(manifest["models"]["gemini-2.5-pro"]["structure_source_ids"] + manifest["models"]["gemini-2.5-pro"]["fact_source_ids"])
            right = set(manifest["models"]["gpt-4"]["structure_source_ids"] + manifest["models"]["gpt-4"]["fact_source_ids"])
            self.assertFalse(left & right)
            payload = read_json(root / "essay" / "gemini-2.5-pro" / "structure.jsonl")
            self.assertEqual(sorted(payload), ["original"])
            self.assertEqual(len(payload["original"]), 2)

    def test_meeting_summary_source_shards_use_same_rows(self):
        rows = [
            {
                "source_id": f"meeting_{idx}",
                "domain": "meeting_summary",
                "human_text": long_text(f"summary{idx}", 30),
                "fact_source_text": f"transcript_excerpt:\n{long_text(f'transcript{idx}', 40)}",
                "task_input": f"meeting_title: item {idx}",
                "metadata": {"uid": f"m-{idx}"},
            }
            for idx in range(4)
        ]
        models = [
            {"provider": "gemini", "model": "gemini-2.5-pro", "model_label": "gemini-2.5-pro"},
            {"provider": "openai", "model": "gpt-4", "model_label": "gpt-4"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = build_domain_shards(
                rows,
                domain="meeting_summary",
                model_specs=models,
                output_root=root,
                items_per_kind=2,
                seed=7,
            )
            gemini = manifest["models"]["gemini-2.5-pro"]
            gpt4 = manifest["models"]["gpt-4"]
            self.assertEqual(gemini["structure_source_ids"], gemini["fact_source_ids"])
            self.assertEqual(gpt4["structure_source_ids"], gpt4["fact_source_ids"])
            self.assertFalse(set(gemini["structure_source_ids"]) & set(gpt4["structure_source_ids"]))

    def test_product_review_source_shards_use_same_rows(self):
        rows = [
            {
                "source_id": f"review_{idx}",
                "domain": "product_review",
                "human_text": long_text(f"review{idx}", 75),
                "fact_source_text": f"Task/context:\nrating: 5\n\nReference text:\n{long_text(f'review{idx}', 75)}",
                "task_input": "rating: 5",
                "metadata": {"asin": f"p-{idx}"},
            }
            for idx in range(4)
        ]
        models = [
            {"provider": "gemini", "model": "gemini-2.5-pro", "model_label": "gemini-2.5-pro"},
            {"provider": "openai", "model": "gpt-4", "model_label": "gpt-4"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = build_domain_shards(
                rows,
                domain="product_review",
                model_specs=models,
                output_root=root,
                items_per_kind=2,
                seed=7,
            )
            gemini = manifest["models"]["gemini-2.5-pro"]
            gpt4 = manifest["models"]["gpt-4"]
            self.assertEqual(gemini["structure_source_ids"], gemini["fact_source_ids"])
            self.assertEqual(gpt4["structure_source_ids"], gpt4["fact_source_ids"])
            self.assertFalse(set(gemini["structure_source_ids"]) & set(gpt4["structure_source_ids"]))

    def test_new_domain_prompts_exist_for_generation_and_editing(self):
        for domain in ("essay", "arxiv_abstract", "blog", "product_review", "peer_review", "meeting_summary", "translation"):
            messages = generation_messages(
                domain,
                {"num_paragraphs": 1, "target_total_word_count": 30, "paragraphs": [{"pid": 1, "sentence_count": 2}]},
                {
                    "core_facts": [],
                    "entities": [],
                    "numbers": [],
                    "quotes": [],
                    "source_language": "Spanish",
                    "source_language_code": "spa_Latn",
                    "source_text": "Texto de prueba.",
                },
            )
            if domain == "translation":
                self.assertIn("SOURCE_TEXT", messages[-1]["content"])
                self.assertNotIn("FACT_CARD", messages[-1]["content"])
            else:
                self.assertIn("FACT_CARD", messages[-1]["content"])
            edit = edit_messages(domain, "Original text.", "AI text.")
            self.assertIn("Revise", edit[-1]["content"])

    def test_edit_prompts_are_domain_specific_and_hide_original_reference(self):
        sentinel = "SECRET_HUMAN_REFERENCE_SHOULD_NOT_APPEAR"
        sampled = "AI text to revise with a concrete detail."
        prompt_bodies = []
        for domain in EDIT_PROMPTS:
            messages = edit_messages(domain, sentinel, sampled)
            joined = "\n".join(message["content"] for message in messages)
            prompt_bodies.append(joined)
            self.assertNotIn(sentinel, joined)
            self.assertIn(sampled, joined)
            self.assertIn(EDIT_PROMPTS[domain].domain_instruction, joined)
            self.assertIn("Return revised text only", joined)
        self.assertEqual(len(set(prompt_bodies)), len(EDIT_PROMPTS))

    def test_effective_layout_caps_linewrap_and_blog_microparagraphs(self):
        self.assertEqual(
            effective_expected_paragraphs(
                "arxiv_abstract",
                {"num_paragraphs": 18, "source_word_count": 180, "target_total_word_count": 260, "paragraphs": []},
            ),
            1,
        )
        self.assertLessEqual(
            effective_expected_paragraphs(
                "blog",
                {"num_paragraphs": 120, "source_word_count": 1800, "target_total_word_count": 2600, "paragraphs": []},
            ),
            18,
        )
        self.assertEqual(
            effective_expected_paragraphs(
                "meeting_summary",
                {"num_paragraphs": 12, "source_word_count": 40, "target_total_word_count": 50, "paragraphs": []},
            ),
            1,
        )

    def test_fake_generation_and_variant_split_create_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard_root = root / "source" / "final"
            slot_dir = shard_root / "essay" / "gemini-2.5-pro"
            slot_dir.mkdir(parents=True)
            write_json(
                slot_dir / "structure.jsonl",
                {"original": [long_text("structure0", 20), long_text("structure1", 20), long_text("structure2", 20)]},
            )
            write_json(
                slot_dir / "fact.jsonl",
                {"original": [long_text("fact0", 20), long_text("fact1", 20), long_text("fact2", 20)]},
            )
            write_jsonl(
                slot_dir / "structure.sources.jsonl",
                [{"source_index": idx, "source_id": f"essay_source_{idx}", "domain": "essay", "metadata": {}} for idx in range(3)],
            )
            args = gen.parse_args(
                [
                    "--provider",
                    "gemini",
                    "--datasets",
                    "essay",
                    "--sample-count",
                    "2",
                    "--max-candidates",
                    "3",
                    "--shard-root",
                    str(shard_root),
                    "--output-root",
                    str(root / "generated"),
                    "--min-words",
                    "5",
                ]
            )
            ok = asyncio.run(
                gen.generate_for_dataset(
                    FakeProvider(),
                    dataset="essay",
                    model="gemini-2.5-pro",
                    model_label="gemini-2.5-pro",
                    args=args,
                )
            )
            self.assertTrue(ok)
            accepted = load_generation_records(root / "generated", "essay", "gemini-2.5-pro")
            source_map = load_structure_source_map(slot_dir)
            split = split_records(
                accepted,
                domain="essay",
                model_label="gemini-2.5-pro",
                run_id="final",
                source_map=source_map,
                raw_count=1,
                ai_edit_count=1,
                human_edit_count=1,
            )
            out_root = root / "output"
            for variant, rows in split.items():
                shard_name = shard_file_name("essay", "gemini-2.5-pro", variant)
                write_jsonl(out_root / "final" / "essay" / "gemini-2.5-pro" / variant / shard_name, rows)
                self.assertTrue((out_root / "final" / "essay" / "gemini-2.5-pro" / variant / shard_name).exists())
            self.assertEqual(len(split["raw_ai"]), 1)
            self.assertEqual(len(split["ai_edit_input"]), 1)

    def test_meeting_summary_generation_pairs_same_source_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard_root = root / "source" / "final"
            slot_dir = shard_root / "meeting_summary" / "gemini-2.5-pro"
            slot_dir.mkdir(parents=True)
            summaries = [long_text(f"summary{idx}", 20) for idx in range(3)]
            transcripts = [f"transcript_excerpt:\n{long_text(f'transcript{idx}', 30)}" for idx in range(3)]
            write_json(slot_dir / "structure.jsonl", {"original": summaries})
            write_json(slot_dir / "fact.jsonl", {"original": transcripts})
            sidecars = [
                {
                    "source_index": idx,
                    "source_id": f"meeting_source_{idx}",
                    "domain": "meeting_summary",
                    "task_input": f"meeting item {idx}",
                    "metadata": {},
                }
                for idx in range(3)
            ]
            write_jsonl(slot_dir / "structure.sources.jsonl", [{**row, "shard_kind": "structure"} for row in sidecars])
            write_jsonl(slot_dir / "fact.sources.jsonl", [{**row, "shard_kind": "fact"} for row in sidecars])
            args = gen.parse_args(
                [
                    "--provider",
                    "gemini",
                    "--datasets",
                    "meeting_summary",
                    "--sample-count",
                    "2",
                    "--max-candidates",
                    "3",
                    "--shard-root",
                    str(shard_root),
                    "--output-root",
                    str(root / "generated"),
                    "--min-words",
                    "5",
                ]
            )
            ok = asyncio.run(
                gen.generate_for_dataset(
                    FakeProvider(),
                    dataset="meeting_summary",
                    model="gemini-2.5-pro",
                    model_label="gemini-2.5-pro",
                    args=args,
                )
            )
            self.assertTrue(ok)
            accepted = load_generation_records(root / "generated", "meeting_summary", "gemini-2.5-pro")
            self.assertEqual(len(accepted), 2)
            for record in accepted:
                self.assertEqual(record["structure_source_index"], record["fact_source_index"])
                self.assertEqual(record["structure_source_id"], record["fact_source_id"])

    def test_meeting_summary_postprocess_does_not_cut_sentences_to_reference_length(self):
        original = "Recommendation to adopt item."
        sampled = (
            "The council reviewed the recommendation and discussed the proposed action in detail. "
            "Members noted the operational context before moving the item forward."
        )
        original_out, sampled_out = gen.postprocess_pair("meeting_summary", original, sampled)
        self.assertEqual(original_out, original)
        self.assertEqual(sampled_out, sampled)
        gen.validate_raw_data({"original": [original_out], "sampled": [sampled_out]}, 1, "meeting_summary")

    def test_generation_postprocess_does_not_word_trim_our_data_domains(self):
        original = "This review keeps its full original wording. It has a complete ending."
        sampled = "Shorter generated review with a complete ending."
        original_out, sampled_out = gen.postprocess_pair("product_review", original, sampled, min_words=5)
        self.assertEqual(original_out, original)
        self.assertEqual(sampled_out, sampled)
        gen.validate_raw_data({"original": [original_out], "sampled": [sampled_out]}, 1, "product_review")

    def test_sentence_postprocess_keeps_short_product_review_for_regen(self):
        text = "Good product. This last unfinished part keeps going without a proper ending"
        result = sentence_safe_postprocess_text(text, min_words=70)
        self.assertEqual(result.action, "keep_below_min")
        self.assertEqual(result.text, text)

    def test_sentence_postprocess_trims_when_safe(self):
        text = (
            "The product worked well and the scent was mild. "
            "I would buy it again because the results were consistent. "
            "This trailing fragment starts but never"
        )
        result = sentence_safe_postprocess_text(text, min_words=10)
        self.assertEqual(result.action, "trimmed")
        self.assertTrue(result.text.endswith("."))
        self.assertNotIn("trailing fragment", result.text)

    def test_product_review_ai_edit_output_is_flattened_and_validated(self):
        source = {
            "id": "final:product_review:gpt-4:ai_edit_input:0000",
            "domain": "product_review",
            "model_label": "gpt-4",
            "variant_type": "ai_edit_input",
            "source_id": "review_source_1",
            "original": long_text("original", 80),
            "sampled": long_text("sampled", 80),
            "metadata": {},
        }
        args = argparse.Namespace(domain="product_review", provider="openai", mock=True)
        row = make_edited_row(source, "First revised sentence.\n\nSecond revised sentence.", args=args, model="gpt-4")
        self.assertNotIn("\n", row["sampled"])
        self.assertTrue(edited_row_is_valid(row, source))
        with self.assertRaises(ValueError):
            validate_edited_text("product_review", "First paragraph.\n\nSecond paragraph.")

    def test_mock_ai_edit_and_human_edit_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = {
                "id": "final:essay:gemini-2.5-pro:ai_edit_input:0000",
                "domain": "essay",
                "model_label": "gemini-2.5-pro",
                "variant_type": "ai_edit_input",
                "source_id": "essay_source_1",
                "original": long_text("original", 20),
                "sampled": "Additionally, the draft is clear and useful for students.",
                "metadata": {},
            }
            write_jsonl(
                root
                / "final"
                / "essay"
                / "gemini-2.5-pro"
                / "ai_edit_input"
                / shard_file_name("essay", "gemini-2.5-pro", "ai_edit_input"),
                [row],
            )
            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "edit_ai.py"),
                    "--run-id",
                    "final",
                    "--domain",
                    "essay",
                    "--provider",
                    "gemini",
                    "--model",
                    "gemini-2.5-pro",
                    "--input-root",
                    str(root),
                    "--output-root",
                    str(root),
                    "--mock",
                ],
                cwd=PROJECT_ROOT,
                check=True,
            )
            edited_rows = read_jsonl(
                root
                / "final"
                / "essay"
                / "gemini-2.5-pro"
                / "ai_edited"
                / shard_file_name("essay", "gemini-2.5-pro", "ai_edited")
            )
            self.assertEqual(edited_rows[0]["variant_type"], "ai_edited")
            self.assertEqual(edited_rows[0]["source_input_id"], row["id"])
            self.assertEqual(edited_rows[0]["metadata"]["ai_edit"]["prompt_version"], EDIT_PROMPT_VERSION)
            self.assertEqual(edited_rows[0]["metadata"]["ai_edit"]["source_variant"], "ai_edit_input")

            human_row = dict(row)
            human_row["id"] = "final:essay:gemini-2.5-pro:human_edit_input:0000"
            human_row["variant_type"] = "human_edit_input"
            write_jsonl(
                root
                / "final"
                / "essay"
                / "gemini-2.5-pro"
                / "human_edit_input"
                / shard_file_name("essay", "gemini-2.5-pro", "human_edit_input"),
                [human_row],
            )
            store = HumanEditStore(
                run_id="final",
                domain="essay",
                model_label="gemini-2.5-pro",
                input_root=root,
                output_root=root,
            )
            store.save_edit(human_row["id"], "A person revised this response with a natural tone.")
            out_path = store.export()
            self.assertTrue(out_path.exists())
            human_rows = read_jsonl(out_path)
            self.assertEqual(human_rows[0]["variant_type"], "human_edited")

    def test_merge_final_dataset_validates_variant_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_root = root / "output"
            config_dir = root / "configs"
            config_dir.mkdir()
            (config_dir / "datasets.yaml").write_text("domains:\n  essay: {}\n", encoding="utf-8")
            (config_dir / "models.yaml").write_text(
                "models:\n  - provider: gemini\n    model: gemini-2.5-pro\n    model_label: gemini-2.5-pro\n",
                encoding="utf-8",
            )
            for variant in ("raw_ai", "ai_edited", "human_edited"):
                row = {
                    "id": f"final:essay:gemini-2.5-pro:{variant}:0000",
                    "domain": "essay",
                    "model_label": "gemini-2.5-pro",
                    "variant_type": variant,
                    "source_id": f"essay_{variant}",
                    "original": long_text("original", 15),
                    "sampled": long_text("sampled", 15),
                    "metadata": {},
                }
                write_jsonl(
                    output_root / "final" / "essay" / "gemini-2.5-pro" / variant / shard_file_name("essay", "gemini-2.5-pro", variant),
                    [row],
                )
            final_path = root / "final.jsonl"
            fdg_root = root / "fdg"
            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "merge_final_dataset.py"),
                    "--run-id",
                    "final",
                    "--input-root",
                    str(output_root),
                    "--output-jsonl",
                    str(final_path),
                    "--fdg-output-root",
                    str(fdg_root),
                    "--manifest-path",
                    str(root / "manifest.json"),
                    "--datasets-config",
                    str(config_dir / "datasets.yaml"),
                    "--models-config",
                    str(config_dir / "models.yaml"),
                    "--expected-per-variant",
                    "1",
                    "--strict",
                ],
                cwd=PROJECT_ROOT,
                check=True,
            )
            self.assertEqual(len(read_jsonl(final_path)), 3)
            fdg_path = fdg_root / "raw_ai" / "essay" / "essay_gemini-2.5-pro_raw_ai.json"
            fdg_payload = read_json(fdg_path)
            self.assertEqual(sorted(fdg_payload), ["original", "sampled"])
            self.assertEqual(len(fdg_payload["original"]), 1)
            self.assertEqual(len(fdg_payload["sampled"]), 1)


if __name__ == "__main__":
    unittest.main()

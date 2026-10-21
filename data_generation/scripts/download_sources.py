#!/usr/bin/env python3
"""Download source datasets into data/raw without running generation."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Iterable

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import as_native, clean_text, load_domain_specs, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains", nargs="+", default=None, help="Domain ids to download. Defaults to all domains.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "datasets.yaml")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--max-rows", type=int, default=5000, help="Rows to keep per domain for local benchmark prep.")
    parser.add_argument("--no-streaming", action="store_true", help="Use regular load_dataset instead of streaming.")
    parser.add_argument("--peerread-data-dir", type=Path, default=None, help="Optional local PeerRead data directory.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = load_domain_specs(args.config)
    domains = args.domains or list(specs)
    for domain in domains:
        if domain not in specs:
            raise ValueError(f"Unknown domain: {domain}")
        spec = dict(specs[domain])
        if args.peerread_data_dir and spec.get("peerread_github_zip"):
            spec["peerread_data_dir"] = str(args.peerread_data_dir)
        out_path = args.output_dir / f"{domain}.jsonl"
        if out_path.exists() and not args.overwrite:
            print(f"[skip] {domain}: {out_path} exists. Use --overwrite to replace.")
            continue
        rows = list(iter_hf_rows(spec, max_rows=args.max_rows, streaming=not args.no_streaming))
        write_jsonl(out_path, rows)
        print(f"[downloaded] {domain}: {len(rows)} rows -> {out_path}")


def iter_hf_rows(spec: dict[str, Any], *, max_rows: int, streaming: bool) -> Iterable[dict[str, Any]]:
    if spec.get("parquet_files"):
        yield from iter_parquet_rows(spec, max_rows=max_rows, streaming=streaming)
        return
    if spec.get("peerread_github_zip"):
        yield from iter_peerread_rows(spec, max_rows=max_rows)
        return
    if spec.get("hf_configs"):
        yield from iter_multi_config_rows(spec, max_rows=max_rows, streaming=streaming)
        return

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets before running downloads: python -m pip install -r requirements.txt") from exc

    dataset_name = spec["hf_dataset"]
    config = spec.get("hf_config")
    split = spec.get("split") or "train"
    trust_remote_code = bool(spec.get("trust_remote_code", False))
    kwargs = {
        "split": split,
        "streaming": streaming,
        "trust_remote_code": trust_remote_code,
    }
    try:
        dataset = load_dataset(dataset_name, config, **kwargs) if config else load_dataset(dataset_name, **kwargs)
    except TypeError:
        kwargs.pop("trust_remote_code", None)
        dataset = load_dataset(dataset_name, config, **kwargs) if config else load_dataset(dataset_name, **kwargs)

    for idx, row in enumerate(dataset):
        if idx >= max_rows:
            break
        if not isinstance(row, dict):
            continue
        yield as_native(row)


def iter_multi_config_rows(spec: dict[str, Any], *, max_rows: int, streaming: bool) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets before running downloads: python -m pip install -r requirements.txt") from exc

    dataset_name = spec["hf_dataset"]
    configs = [str(config) for config in spec.get("hf_configs") or []]
    if not configs:
        return
    split = spec.get("split") or "train"
    trust_remote_code = bool(spec.get("trust_remote_code", False))
    rows_per_config = max(1, max_rows // len(configs))
    for config in configs:
        kwargs = {
            "split": split,
            "streaming": streaming,
            "trust_remote_code": trust_remote_code,
        }
        try:
            dataset = load_dataset(dataset_name, name=config, **kwargs)
        except TypeError:
            kwargs.pop("trust_remote_code", None)
            dataset = load_dataset(dataset_name, name=config, **kwargs)
        yielded = 0
        for row in dataset:
            if yielded >= rows_per_config:
                break
            if not isinstance(row, dict):
                continue
            output = as_native(row)
            output.setdefault("hf_config", config)
            output.setdefault("language_config", config)
            yield output
            yielded += 1


def iter_parquet_rows(spec: dict[str, Any], *, max_rows: int, streaming: bool) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets before running downloads: python -m pip install -r requirements.txt") from exc

    split = spec.get("split") or "train"
    files = spec["parquet_files"]
    dataset = load_dataset("parquet", data_files={split: files}, split=split, streaming=streaming)
    for idx, row in enumerate(dataset):
        if idx >= max_rows:
            break
        if isinstance(row, dict):
            yield as_native(row)


def iter_peerread_rows(spec: dict[str, Any], *, max_rows: int) -> Iterable[dict[str, Any]]:
    split = spec.get("split") or "train"
    local_dir = spec.get("peerread_data_dir")
    if local_dir:
        yielded = 0
        for path in sorted(Path(local_dir).glob(f"*/{split}/reviews/*.json")):
            if yielded >= max_rows:
                break
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            for row in peerread_review_rows(data, str(path)):
                if yielded >= max_rows:
                    break
                row = as_native(row)
                if not peerread_has_comments(row):
                    continue
                yield row
                yielded += 1
        return

    url = spec["peerread_github_zip"]
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "PeerRead-master.zip"
        urllib.request.urlretrieve(url, zip_path)
        yielded = 0
        with zipfile.ZipFile(zip_path) as archive:
            names = sorted(
                name
                for name in archive.namelist()
                if f"/{split}/reviews/" in name and name.endswith(".json")
            )
            for name in names:
                if yielded >= max_rows:
                    break
                with archive.open(name) as handle:
                    data = json.loads(handle.read().decode("utf-8", errors="replace"))
                for row in peerread_review_rows(data, name):
                    if yielded >= max_rows:
                        break
                    row = as_native(row)
                    if not peerread_has_comments(row):
                        continue
                    yield row
                    yielded += 1


def peerread_review_rows(data: dict[str, Any], archive_name: str) -> list[dict[str, Any]]:
    conference = data.get("conference") or peerread_conference_from_path(archive_name)
    metadata = data.get("metadata") or {}
    reviews = []
    raw_reviews = metadata.get("reviews") or data.get("reviews") or []
    for review in raw_reviews:
        if not isinstance(review, dict):
            continue
        get = review.get
        reviews.append(
            {
                "date": str(get("date", get("DATE", ""))),
                "title": str(get("title", get("TITLE", ""))),
                "other_keys": str(get("other_keys", get("OTHER_KEYS", ""))),
                "originality": str(get("originality", get("ORIGINALITY", ""))),
                "comments": str(get("comments", get("COMMENTS", ""))),
                "is_meta_review": str(get("is_meta_review", get("IS_META_REVIEW", ""))),
                "is_annotated": str(get("is_annotated", get("IS_ANNOTATED", ""))),
                "recommendation": str(get("recommendation", get("RECOMMENDATION", ""))),
                "replicability": str(get("replicability", get("REPLICABILITY", ""))),
                "presentation_format": str(get("presentation_format", get("PRESENTATION_FORMAT", ""))),
                "clarity": str(get("clarity", get("CLARITY", ""))),
                "meaningful_comparison": str(get("meaningful_comparison", get("MEANINGFUL_COMPARISON", ""))),
                "substance": str(get("substance", get("SUBSTANCE", ""))),
                "reviewer_confidence": str(get("reviewer_confidence", get("REVIEWER_CONFIDENCE", ""))),
                "soundness_correctness": str(get("soundness_correctness", get("SOUNDNESS_CORRECTNESS", ""))),
                "appropriateness": str(get("appropriateness", get("APPROPRIATENESS", ""))),
                "impact": str(get("impact", get("IMPACT", ""))),
            }
        )
    base = {
        "id": str(data.get("id", "")),
        "conference": str(conference),
        "subjects": str(data.get("subjects", "")),
        "version": str(data.get("version", "")),
        "date_of_submission": str(data.get("date_of_submission", "")),
        "title": str(data.get("title", "")),
        "authors": data.get("authors", []) if isinstance(data.get("authors"), list) else [data.get("authors")],
        "accepted": data.get("accepted", ""),
        "abstract": str(data.get("abstract", "")),
        "histories": data.get("histories", []),
    }
    rows: list[dict[str, Any]] = []
    if clean_text(data.get("comments")):
        rows.append({**base, "comments": str(data.get("comments", "")), "reviews": []})
    for idx, review in enumerate(reviews):
        if not clean_text(review.get("comments")):
            continue
        row = {
            **base,
            "id": f"{base['id']}_review_{idx:03d}",
            "comments": review["comments"],
            "reviews": [review],
            "review_title": review.get("title", ""),
            "recommendation": review.get("recommendation", ""),
            "originality": review.get("originality", ""),
            "clarity": review.get("clarity", ""),
            "substance": review.get("substance", ""),
            "soundness_correctness": review.get("soundness_correctness", ""),
            "impact": review.get("impact", ""),
            "reviewer_confidence": review.get("reviewer_confidence", ""),
            "is_meta_review": review.get("is_meta_review", ""),
        }
        rows.append(row)
    return rows


def peerread_has_comments(row: dict[str, Any]) -> bool:
    if clean_text(row.get("comments")):
        return True
    for review in row.get("reviews", []) or []:
        if isinstance(review, dict) and clean_text(review.get("comments")):
            return True
    return False


def peerread_conference_from_path(path: str) -> str:
    parts = Path(path).parts
    if "data" in parts:
        idx = parts.index("data")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    raw_parts = path.split("/")
    for part in raw_parts:
        if part in {"acl_2017", "conll_2016", "iclr_2017", "nips_2013-2017"} or part.startswith("arxiv.cs."):
            return part
    return ""


if __name__ == "__main__":
    main()

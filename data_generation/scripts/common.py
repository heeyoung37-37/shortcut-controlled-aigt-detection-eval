from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"

FINAL_VARIANTS = ("raw_ai", "ai_edited", "human_edited")
EDIT_INPUT_VARIANTS = ("ai_edit_input", "human_edit_input")
ALL_OUTPUT_VARIANTS = FINAL_VARIANTS + EDIT_INPUT_VARIANTS
OUTPUT_ROW_KEYS = ("id", "domain", "model_label", "variant_type", "source_id", "original", "sampled", "metadata")
WORD_RE = re.compile(r"\b[\w'-]+\b")


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML object")
    return payload


def load_domain_specs(path: Path | None = None) -> dict[str, dict[str, Any]]:
    payload = read_yaml(path or CONFIG_DIR / "datasets.yaml")
    domains = payload.get("domains")
    if not isinstance(domains, dict):
        raise ValueError("datasets.yaml must contain a domains mapping")
    return domains


def load_model_specs(path: Path | None = None) -> list[dict[str, Any]]:
    payload = read_yaml(path or CONFIG_DIR / "models.yaml")
    models = payload.get("models")
    if not isinstance(models, list):
        raise ValueError("models.yaml must contain a models list")
    normalized = []
    for item in models:
        if not isinstance(item, dict):
            raise ValueError("Each model spec must be an object")
        provider = str(item.get("provider", "")).strip()
        model = str(item.get("model", "")).strip()
        label = str(item.get("model_label", "")).strip()
        if not provider or not model or not label:
            raise ValueError(f"Invalid model spec: {item!r}")
        spec = dict(item)
        spec["provider"] = provider
        spec["model"] = model
        spec["model_label"] = label
        normalized.append(spec)
    return normalized


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            parsed = json.loads(stripped)
            if not isinstance(parsed, dict):
                raise ValueError(f"{path}:{line_no} must contain a JSON object")
            rows.append(parsed)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def stable_id(*parts: Any, length: int = 16) -> str:
    joined = "\u241f".join(clean_text(part) for part in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:length]


def as_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): as_native(v) for k, v in value.items()}
    if isinstance(value, list):
        return [as_native(v) for v in value]
    if isinstance(value, tuple):
        return [as_native(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def coerce_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def output_shard_dir(root: Path, run_id: str, domain: str, model_label: str, variant_type: str) -> Path:
    return root / run_id / domain / model_label / variant_type


def safe_filename_component(value: Any) -> str:
    text = clean_text(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "unknown"


def shard_file_name(domain: str, model_label: str, variant_type: str, shard_index: int = 0) -> str:
    domain_part = safe_filename_component(domain)
    model_part = safe_filename_component(model_label)
    variant_part = safe_filename_component(variant_type)
    return f"{domain_part}_{model_part}_{variant_part}_shard_{shard_index:03d}.jsonl"


def iter_shard_files(root: Path, pattern: str | None = None) -> list[Path]:
    patterns = [pattern] if pattern else ["*_shard_*.jsonl", "shard_*.jsonl"]
    paths: list[Path] = []
    seen: set[Path] = set()
    for item in patterns:
        for path in sorted(root.glob(item)):
            if path.is_file() and path not in seen:
                paths.append(path)
                seen.add(path)
    return paths


def validate_output_row(row: dict[str, Any]) -> None:
    missing = [key for key in OUTPUT_ROW_KEYS if key not in row]
    if missing:
        raise ValueError(f"Output row is missing keys: {missing}")
    if row["variant_type"] not in ALL_OUTPUT_VARIANTS:
        raise ValueError(f"Invalid variant_type: {row['variant_type']}")
    if not str(row["original"]).strip() or not str(row["sampled"]).strip():
        raise ValueError(f"Empty original/sample at row id={row.get('id')}")

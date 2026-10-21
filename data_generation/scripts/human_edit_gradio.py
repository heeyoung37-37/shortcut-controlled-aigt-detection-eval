#!/usr/bin/env python3
"""Gradio UI for two-person human editing of detector benchmark samples."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import gradio as gr
except Exception as exc:  # pragma: no cover - exercised only when dependency is absent.
    raise SystemExit("Missing dependency: install gradio before running this editor") from exc

from scripts.common import read_json, safe_filename_component, write_json


DOMAIN_DESCRIPTIONS = {
    "ko": {
        "essay": "AI가 작성한 IELTS/TOEFL 스타일 영어 에세이",
        "arxiv_abstract": "AI가 작성한 논문 초록",
        "blog": "AI가 작성한 블로그 포스트",
        "product_review": "AI가 작성한 상품 리뷰",
        "peer_review": "AI가 작성한 논문 peer review",
        "meeting_summary": "AI가 작성한 회의/의정 요약문",
        "translation": "AI가 외국어 원문을 영어로 번역한 문장",
    },
    "en": {
        "essay": "AI-generated IELTS/TOEFL-style English essay",
        "arxiv_abstract": "AI-generated research paper abstract",
        "blog": "AI-generated blog post",
        "product_review": "AI-generated product review",
        "peer_review": "AI-generated peer-review report",
        "meeting_summary": "AI-generated meeting or council summary",
        "translation": "AI-generated English translation of a non-English source text",
    },
}

COMMON_GUIDES = {
    "ko": [
        "어색한 표현, 부자연스러운 문장, 지나치게 AI 같은 표현을 자연스럽게 고칩니다.",
        "의미, 사실, 숫자, 고유명사는 바꾸지 않습니다.",
        "문체와 길이는 가능한 한 유지합니다.",
        "새 내용을 추가하지 않습니다.",
        "설명이나 메모를 붙이지 말고 본문만 남깁니다.",
    ],
    "en": [
        "Revise awkward, unnatural, or overly AI-like wording so that it reads naturally.",
        "Do not change the meaning, facts, numbers, or proper names.",
        "Preserve the original style and length as much as possible.",
        "Do not add new information.",
        "Leave only the revised text; do not add explanations or notes.",
    ],
}

DOMAIN_GUIDES = {
    "ko": {
        "essay": "너무 완벽한 원어민 글로 바꾸지 말고 자연스러운 학습자 에세이 느낌을 유지합니다.",
        "arxiv_abstract": "학술 초록의 압축적이고 신중한 톤을 유지합니다.",
        "blog": "읽기 쉬운 블로그 문장으로 다듬되 목록, 코드, 메타정보를 새로 추가하지 않습니다.",
        "product_review": "소비자 리뷰처럼 자연스럽게 다듬고 광고문처럼 바꾸지 않습니다.",
        "peer_review": "reviewer comment 톤을 유지하고 recommendation label을 새로 만들지 않습니다.",
        "meeting_summary": "회의 요약/action-item 형식을 유지하고 장황한 기사체로 바꾸지 않습니다.",
        "translation": "영어 번역문을 자연스럽게 다듬되 원 의미를 보존합니다.",
    },
    "en": {
        "essay": "Keep the voice of a natural learner essay; do not rewrite it as flawless native-speaker prose.",
        "arxiv_abstract": "Maintain the concise and cautious tone of an academic abstract.",
        "blog": "Improve readability as blog prose, without adding lists, code, or metadata.",
        "product_review": "Make it read like a natural consumer review, not advertising copy.",
        "peer_review": "Maintain the tone of reviewer comments and do not add a recommendation label.",
        "meeting_summary": "Preserve the meeting-summary or action-item format; do not turn it into a verbose article.",
        "translation": "Make the English translation read naturally while preserving the source meaning.",
    },
}

UI_TEXT = {
    "ko": {
        "guide_heading": "첨삭 가이드",
        "source_label": "첨삭 대상 sampled",
        "edited_label": "사람 첨삭본",
    },
    "en": {
        "guide_heading": "Editing Guidelines",
        "source_label": "Text to Edit (sampled)",
        "edited_label": "Human-Edited Text",
    },
}


@dataclass
class Assignment:
    input_path: Path
    output_path: Path
    rel_input: str
    rel_output: str
    domain: str
    model_label: str
    display_label: str
    original: list[str]
    source_sampled: list[str]
    edited_sampled: list[str]
    done: list[bool]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--editor-id", required=True, choices=["1", "2"])
    parser.add_argument("--input-root", type=Path, default=PROJECT_ROOT / "human_edit_input")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "human_edited_gradio_saved")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--language", choices=["ko", "en"], default="ko")
    parser.add_argument("--check-split", action="store_true", help="Validate assignment split and exit.")
    return parser.parse_args()


class HumanEditStore:
    def __init__(self, *, editor_id: str, input_root: Path, output_root: Path, language: str = "ko") -> None:
        self.editor_id = editor_id
        self.language = language
        self.input_root = input_root.resolve()
        self.output_root = (output_root / f"editor_{editor_id}").resolve()
        self.manifest_path = self.output_root / "progress_manifest.json"
        self.lock = threading.Lock()
        self.progress = self._load_progress()
        self.assignments = self._load_assignments()

    def _load_progress(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {}
        payload = read_json(self.manifest_path)
        if not isinstance(payload, dict):
            return {}
        return payload

    def _load_assignments(self) -> list[Assignment]:
        input_files = all_input_files(self.input_root)
        editor_offset = int(self.editor_id) - 1
        selected = [path for idx, path in enumerate(input_files) if idx % 2 == editor_offset]
        if not selected:
            raise FileNotFoundError(f"No assigned files found under {self.input_root}")
        progress_by_input = {
            item.get("input_file"): item for item in self.progress.get("assignments", [])
            if isinstance(item, dict)
        }
        assignments = []
        total = len(selected)
        for local_idx, input_path in enumerate(selected, 1):
            domain, model_label = parse_domain_model(self.input_root, input_path)
            payload = load_fdg_payload(input_path)
            output_path = self.output_path_for(input_path, domain, model_label)
            rel_input = str(input_path.relative_to(self.input_root))
            rel_output = str(output_path.relative_to(self.output_root))
            done = [False] * len(payload["sampled"])
            edited_sampled = list(payload["sampled"])
            if output_path.exists():
                saved = load_fdg_payload(output_path)
                if saved["original"] != payload["original"]:
                    raise ValueError(f"{output_path} original does not match {input_path}")
                edited_sampled = list(saved["sampled"])
            progress_item = progress_by_input.get(rel_input) or {}
            progress_done = progress_item.get("done")
            if isinstance(progress_done, list) and len(progress_done) == len(done):
                done = [bool(item) for item in progress_done]
            display_label = f"{local_idx:02d}/{total:02d} · {domain} · {model_label}"
            assignments.append(
                Assignment(
                    input_path=input_path,
                    output_path=output_path,
                    rel_input=rel_input,
                    rel_output=rel_output,
                    domain=domain,
                    model_label=model_label,
                    display_label=display_label,
                    original=list(payload["original"]),
                    source_sampled=list(payload["sampled"]),
                    edited_sampled=edited_sampled,
                    done=done,
                )
            )
        return assignments

    def output_path_for(self, input_path: Path, domain: str, model_label: str) -> Path:
        filename = f"{safe_filename_component(domain)}_{safe_filename_component(model_label)}_human_edited.json"
        return self.output_root / domain / filename

    @property
    def file_labels(self) -> list[str]:
        return [assignment.display_label for assignment in self.assignments]

    def file_index(self, file_label: str | None) -> int:
        labels = self.file_labels
        if file_label in labels:
            return labels.index(str(file_label))
        return 0

    def row_labels(self, file_index: int) -> list[str]:
        assignment = self.assignments[file_index]
        total = len(assignment.source_sampled)
        return [
            f"{idx + 1:02d}/{total:02d} · {'saved' if assignment.done[idx] else 'open'}"
            for idx in range(total)
        ]

    def row_index(self, file_index: int, row_label: str | None) -> int:
        if row_label:
            try:
                return max(0, int(str(row_label).split("/", 1)[0]) - 1)
            except Exception:
                pass
        return self.first_open_row(file_index)

    def first_open_row(self, file_index: int) -> int:
        assignment = self.assignments[file_index]
        for idx, done in enumerate(assignment.done):
            if not done:
                return idx
        return 0

    def next_position(self, file_index: int, row_index: int) -> tuple[int, int]:
        for fidx in range(file_index, len(self.assignments)):
            start = row_index + 1 if fidx == file_index else 0
            assignment = self.assignments[fidx]
            for ridx in range(start, len(assignment.source_sampled)):
                if not assignment.done[ridx]:
                    return fidx, ridx
        for fidx, assignment in enumerate(self.assignments):
            for ridx, done in enumerate(assignment.done):
                if not done:
                    return fidx, ridx
        assignment = self.assignments[file_index]
        return file_index, min(row_index + 1, len(assignment.source_sampled) - 1)

    def previous_position(self, file_index: int, row_index: int) -> tuple[int, int]:
        if row_index > 0:
            return file_index, row_index - 1
        if file_index > 0:
            prev_assignment = self.assignments[file_index - 1]
            return file_index - 1, len(prev_assignment.source_sampled) - 1
        return file_index, row_index

    def save(self, file_index: int, row_index: int, edited_text: str) -> str:
        edited = (edited_text or "").strip()
        if not edited:
            raise ValueError("Edited text cannot be empty")
        with self.lock:
            assignment = self.assignments[file_index]
            assignment.edited_sampled[row_index] = edited
            assignment.done[row_index] = True
            self._write_assignment(assignment)
            self._write_progress()
        return f"Saved row {row_index + 1}/{len(assignment.done)} at {timestamp_label()}"

    def _write_assignment(self, assignment: Assignment) -> None:
        write_json_atomic(
            assignment.output_path,
            {
                "original": assignment.original,
                "sampled": assignment.edited_sampled,
            },
        )

    def _write_progress(self) -> None:
        assignments = []
        for assignment in self.assignments:
            assignments.append(
                {
                    "input_file": assignment.rel_input,
                    "output_file": assignment.rel_output,
                    "domain": assignment.domain,
                    "model_label": assignment.model_label,
                    "done": assignment.done,
                    "saved_count": sum(assignment.done),
                    "row_count": len(assignment.done),
                }
            )
        write_json_atomic(
            self.manifest_path,
            {
                "editor_id": self.editor_id,
                "input_root": str(self.input_root),
                "output_root": str(self.output_root),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "assigned_file_count": len(self.assignments),
                "assigned_row_count": sum(len(item.done) for item in self.assignments),
                "saved_row_count": sum(sum(item.done) for item in self.assignments),
                "assignments": assignments,
            },
        )

    def progress_text(self) -> str:
        saved = sum(sum(item.done) for item in self.assignments)
        total = sum(len(item.done) for item in self.assignments)
        return f"**Progress:** {saved}/{total} rows saved · {len(self.assignments)} files assigned"

    def render(self, file_index: int, row_index: int, status: str = "") -> tuple[Any, ...]:
        assignment = self.assignments[file_index]
        row_labels = self.row_labels(file_index)
        row_label = row_labels[row_index]
        info = (
            f"### Editor {self.editor_id}\n"
            f"**Data:** {DOMAIN_DESCRIPTIONS[self.language].get(assignment.domain, assignment.domain)}  \n"
            f"**Domain:** `{assignment.domain}` · **Model:** `{assignment.model_label}`  \n"
            f"**File:** `{assignment.rel_input}`"
        )
        guide_lines = COMMON_GUIDES[self.language] + [DOMAIN_GUIDES[self.language].get(assignment.domain, "")]
        guide = f"### {UI_TEXT[self.language]['guide_heading']}\n" + "\n".join(
            f"- {line}" for line in guide_lines if line
        )
        row_status = "saved" if assignment.done[row_index] else "open"
        status_text = status or f"Row {row_index + 1}/{len(row_labels)} is {row_status}."
        return (
            gr.update(value=assignment.display_label),
            gr.update(choices=row_labels, value=row_label),
            info,
            guide,
            assignment.source_sampled[row_index],
            assignment.edited_sampled[row_index],
            status_text,
            self.progress_text(),
        )


def all_input_files(input_root: Path) -> list[Path]:
    files = sorted(path for path in input_root.rglob("*.json") if path.name != "manifest.json")
    if not files:
        raise FileNotFoundError(f"No JSON files found under {input_root}")
    return files


def parse_domain_model(input_root: Path, path: Path) -> tuple[str, str]:
    rel = path.relative_to(input_root)
    if len(rel.parts) < 2:
        raise ValueError(f"Expected {path} under a domain directory")
    domain = rel.parts[0]
    filename = path.name
    prefix = f"{domain}_"
    suffix = "_human_edit_input.json"
    if not filename.startswith(prefix) or not filename.endswith(suffix):
        raise ValueError(f"Unexpected human edit input filename: {path}")
    model_label = filename[len(prefix) : -len(suffix)]
    if not model_label:
        raise ValueError(f"Could not parse model label from {path}")
    return domain, model_label


def load_fdg_payload(path: Path) -> dict[str, list[str]]:
    payload = read_json(path)
    if not isinstance(payload, dict) or sorted(payload.keys()) != ["original", "sampled"]:
        raise ValueError(f"{path} must contain only original and sampled keys")
    original = payload.get("original")
    sampled = payload.get("sampled")
    if not isinstance(original, list) or not isinstance(sampled, list):
        raise ValueError(f"{path} original/sampled must be lists")
    if len(original) != len(sampled) or len(original) == 0:
        raise ValueError(f"{path} original/sampled length mismatch or empty")
    return {
        "original": [str(item) for item in original],
        "sampled": [str(item) for item in sampled],
    }


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_name, path)
    finally:
        tmp_path = Path(tmp_name)
        if tmp_path.exists():
            tmp_path.unlink()


def timestamp_label() -> str:
    return datetime.now().strftime("%H:%M:%S")


def build_app(store: HumanEditStore) -> gr.Blocks:
    initial_file = store.file_labels[0]
    initial_row = store.row_labels(0)[store.first_open_row(0)]
    ui_text = UI_TEXT[store.language]
    with gr.Blocks(
        title=f"Human Edit Editor {store.editor_id}",
        css="footer { display: none !important; }",
    ) as demo:
        gr.Markdown(f"# Human Edit Editor {store.editor_id}")
        with gr.Row():
            file_select = gr.Dropdown(choices=store.file_labels, value=initial_file, label="Assigned file")
            row_select = gr.Dropdown(choices=store.row_labels(0), value=initial_row, label="Row")
        progress_box = gr.Markdown(store.progress_text())
        info_box = gr.Markdown()
        guide_box = gr.Markdown()
        with gr.Row(equal_height=True):
            source_text = gr.Textbox(label=ui_text["source_label"], lines=24, interactive=False)
            edited_text = gr.Textbox(label=ui_text["edited_label"], lines=24, interactive=True)
        with gr.Row():
            prev_button = gr.Button("Prev")
            next_button = gr.Button("Next")
            save_button = gr.Button("Save", variant="primary")
            save_next_button = gr.Button("Save & Next", variant="primary")
        status_box = gr.Markdown()

        outputs = [
            file_select,
            row_select,
            info_box,
            guide_box,
            source_text,
            edited_text,
            status_box,
            progress_box,
        ]

        def current_indices(file_label: str | None, row_label: str | None) -> tuple[int, int]:
            fidx = store.file_index(file_label)
            return fidx, store.row_index(fidx, row_label)

        def render_file(file_label: str | None) -> tuple[Any, ...]:
            fidx = store.file_index(file_label)
            ridx = store.first_open_row(fidx)
            return store.render(fidx, ridx)

        def render_row(file_label: str | None, row_label: str | None) -> tuple[Any, ...]:
            fidx, ridx = current_indices(file_label, row_label)
            return store.render(fidx, ridx)

        def move_prev(file_label: str | None, row_label: str | None) -> tuple[Any, ...]:
            fidx, ridx = current_indices(file_label, row_label)
            return store.render(*store.previous_position(fidx, ridx))

        def move_next(file_label: str | None, row_label: str | None) -> tuple[Any, ...]:
            fidx, ridx = current_indices(file_label, row_label)
            return store.render(*store.next_position(fidx, ridx))

        def save_current(file_label: str | None, row_label: str | None, edited: str) -> tuple[Any, ...]:
            fidx, ridx = current_indices(file_label, row_label)
            try:
                status = store.save(fidx, ridx, edited)
            except Exception as exc:
                return store.render(fidx, ridx, f"Save failed: {exc}")
            return store.render(fidx, ridx, status)

        def save_and_next(file_label: str | None, row_label: str | None, edited: str) -> tuple[Any, ...]:
            fidx, ridx = current_indices(file_label, row_label)
            try:
                status = store.save(fidx, ridx, edited)
                nfidx, nridx = store.next_position(fidx, ridx)
                return store.render(nfidx, nridx, status)
            except Exception as exc:
                return store.render(fidx, ridx, f"Save failed: {exc}")

        file_select.change(render_file, inputs=[file_select], outputs=outputs)
        row_select.change(render_row, inputs=[file_select, row_select], outputs=outputs)
        prev_button.click(move_prev, inputs=[file_select, row_select], outputs=outputs)
        next_button.click(move_next, inputs=[file_select, row_select], outputs=outputs)
        save_button.click(save_current, inputs=[file_select, row_select, edited_text], outputs=outputs)
        save_next_button.click(save_and_next, inputs=[file_select, row_select, edited_text], outputs=outputs)
        demo.load(lambda: store.render(0, store.first_open_row(0)), outputs=outputs)
    return demo


def print_split_summary(store: HumanEditStore) -> None:
    print(f"editor_{store.editor_id}")
    print(f"files={len(store.assignments)} rows={sum(len(item.done) for item in store.assignments)}")
    for assignment in store.assignments:
        print(f"{assignment.display_label} -> {assignment.rel_output}")


def main() -> None:
    args = parse_args()
    store = HumanEditStore(
        editor_id=args.editor_id,
        input_root=args.input_root,
        output_root=args.output_root,
        language=args.language,
    )
    if args.check_split:
        print_split_summary(store)
        return
    port = args.port or (7860 + int(args.editor_id))
    app = build_app(store)
    print(f"[human-edit-gradio] editor_{args.editor_id}: http://{args.host}:{port}")
    print(f"[human-edit-gradio] saving to: {store.output_root}")
    app.launch(server_name=args.host, server_port=port, share=False)


if __name__ == "__main__":
    main()

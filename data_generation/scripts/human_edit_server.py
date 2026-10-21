#!/usr/bin/env python3
"""Small local web UI for human editing human_edit_input shards."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import iter_shard_files, output_shard_dir, read_jsonl, shard_file_name, validate_output_row, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="final")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--input-root", type=Path, default=PROJECT_ROOT / "data" / "output_shards")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "output_shards")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


class HumanEditStore:
    def __init__(self, *, run_id: str, domain: str, model_label: str, input_root: Path, output_root: Path) -> None:
        self.run_id = run_id
        self.domain = domain
        self.model_label = model_label
        self.input_dir = output_shard_dir(input_root, run_id, domain, model_label, "human_edit_input")
        self.output_dir = output_shard_dir(output_root, run_id, domain, model_label, "human_edited")
        self.input_rows = load_input_rows(self.input_dir)
        self.edits_path = self.output_dir / "working_edits.jsonl"
        self.edits = {row["source_input_id"]: row for row in read_jsonl(self.edits_path)}

    def rows_payload(self) -> list[dict]:
        payload = []
        for row in self.input_rows:
            edit = self.edits.get(row["id"])
            payload.append(
                {
                    "id": row["id"],
                    "source_id": row["source_id"],
                    "original": row["original"],
                    "sampled": row["sampled"],
                    "edited": edit["sampled"] if edit else row["sampled"],
                    "done": bool(edit),
                }
            )
        return payload

    def save_edit(self, row_id: str, edited_text: str) -> dict:
        source = next((row for row in self.input_rows if row["id"] == row_id), None)
        if not source:
            raise KeyError(f"Unknown row id: {row_id}")
        edited_text = (edited_text or "").strip()
        if not edited_text:
            raise ValueError("Edited text cannot be empty")
        output = make_human_edited_row(source, edited_text)
        self.edits[row_id] = output
        self.flush_working_edits()
        return output

    def flush_working_edits(self) -> None:
        write_jsonl(self.edits_path, self.edits.values())

    def export(self) -> Path:
        rows = [self.edits[row["id"]] for row in self.input_rows if row["id"] in self.edits]
        for row in rows:
            validate_output_row(row)
        out_path = self.output_dir / shard_file_name(self.domain, self.model_label, "human_edited")
        write_jsonl(out_path, rows)
        return out_path


def load_input_rows(input_dir: Path) -> list[dict]:
    rows = []
    for path in iter_shard_files(input_dir):
        rows.extend(read_jsonl(path))
    if not rows:
        raise FileNotFoundError(f"No human_edit_input shards found in {input_dir}")
    return rows


def make_human_edited_row(source: dict, edited_text: str) -> dict:
    row = dict(source)
    row["id"] = source["id"].replace(":human_edit_input:", ":human_edited:")
    row["variant_type"] = "human_edited"
    row["sampled"] = edited_text
    row["metadata"] = dict(source.get("metadata", {}))
    row["metadata"]["human_edit"] = {
        "edited_at": datetime.now(timezone.utc).isoformat(),
        "source_variant": "human_edit_input",
    }
    row["source_input_id"] = source["id"]
    validate_output_row(row)
    return row


def make_handler(store: HumanEditStore):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/rows":
                self.send_json(store.rows_payload())
                return
            if parsed.path == "/api/export":
                path = store.export()
                self.send_json({"ok": True, "path": str(path)})
                return
            self.send_html(render_html(store))

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/api/save":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("content-length", "0"))
                body = self.rfile.read(length).decode("utf-8")
                payload = json.loads(body)
                row = store.save_edit(str(payload.get("id", "")), str(payload.get("edited", "")))
                self.send_json({"ok": True, "row": row})
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, status=400)

        def log_message(self, format: str, *args) -> None:
            return

        def send_json(self, payload, status: int = 200) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json; charset=utf-8")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def send_html(self, html: str) -> None:
            data = html.encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "text/html; charset=utf-8")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def render_html(store: HumanEditStore) -> str:
    rows_json = json.dumps(store.rows_payload(), ensure_ascii=False)
    title = f"{store.domain} / {store.model_label}"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Human Edit - {title}</title>
  <style>
    body {{ margin: 0; font: 14px system-ui, sans-serif; color: #1d252c; background: #f7f8f4; }}
    header {{ display:flex; align-items:center; justify-content:space-between; padding: 12px 16px; background:#16302b; color:white; }}
    main {{ display:grid; grid-template-columns: 280px 1fr; min-height: calc(100vh - 52px); }}
    nav {{ border-right:1px solid #d7d9d1; overflow:auto; background:white; }}
    button {{ cursor:pointer; border:1px solid #aeb5aa; background:white; padding:8px 10px; border-radius:6px; }}
    .item {{ display:block; width:100%; text-align:left; border:0; border-bottom:1px solid #eef0ea; border-radius:0; }}
    .item.done {{ color:#176b4d; }}
    section {{ padding:16px; display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
    textarea {{ width:100%; height:42vh; box-sizing:border-box; padding:12px; font:14px ui-monospace, monospace; border:1px solid #b8bfb4; border-radius:8px; }}
    pre {{ white-space:pre-wrap; margin:0; padding:12px; background:white; border:1px solid #d7d9d1; border-radius:8px; max-height:42vh; overflow:auto; }}
    .wide {{ grid-column:1 / -1; display:flex; gap:8px; align-items:center; }}
  </style>
</head>
<body>
  <header><strong>{title}</strong><button id="export">Export</button></header>
  <main><nav id="list"></nav><section>
    <div><h3>Original</h3><pre id="original"></pre></div>
    <div><h3>AI Draft</h3><pre id="sampled"></pre></div>
    <div class="wide"><span id="rowid"></span><button id="save">Save</button><span id="status"></span></div>
    <div style="grid-column:1 / -1"><textarea id="edited"></textarea></div>
  </section></main>
  <script>
    const rows = {rows_json};
    let current = 0;
    const list = document.getElementById('list');
    function renderList() {{
      list.innerHTML = '';
      rows.forEach((row, idx) => {{
        const btn = document.createElement('button');
        btn.className = 'item' + (row.done ? ' done' : '');
        btn.textContent = `${{idx + 1}}. ${{row.done ? 'saved' : 'open'}}`;
        btn.onclick = () => {{ current = idx; renderRow(); }};
        list.appendChild(btn);
      }});
    }}
    function renderRow() {{
      const row = rows[current];
      document.getElementById('rowid').textContent = row.id;
      document.getElementById('original').textContent = row.original;
      document.getElementById('sampled').textContent = row.sampled;
      document.getElementById('edited').value = row.edited;
      document.getElementById('status').textContent = '';
    }}
    document.getElementById('save').onclick = async () => {{
      const row = rows[current];
      const res = await fetch('/api/save', {{method:'POST', body: JSON.stringify({{id: row.id, edited: document.getElementById('edited').value}})}});
      const payload = await res.json();
      if (!payload.ok) {{ document.getElementById('status').textContent = payload.error; return; }}
      row.done = true; row.edited = document.getElementById('edited').value;
      document.getElementById('status').textContent = 'saved';
      renderList();
    }};
    document.getElementById('export').onclick = async () => {{
      const payload = await (await fetch('/api/export')).json();
      document.getElementById('status').textContent = payload.path || payload.error;
    }};
    renderList(); renderRow();
  </script>
</body>
</html>"""


def main() -> None:
    args = parse_args()
    store = HumanEditStore(
        run_id=args.run_id,
        domain=args.domain,
        model_label=args.model_label,
        input_root=args.input_root,
        output_root=args.output_root,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    print(f"[human-edit] http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()

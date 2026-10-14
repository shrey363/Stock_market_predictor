"""Interactive labeller for the Phase 2 review sheets (human labels only).

    python -m stock_project.nlp.labeller --config configs/config.example.yaml

Phase 2 ends with two items that only a person can finish: reviewing the sample
and computing the classifier's metrics on it. This tool makes that review cheap —
a local page, one headline at a time, keyboard shortcuts — while keeping the
project's honesty rules intact:

* **Nothing is invented.** The tool only writes what the reviewer pressed; a skipped
  row stays blank and is counted as unreviewed, never as correct.
* **Labelling stays blind.** Only the context columns (headline, date, source) reach
  the browser. The classifier's prediction and matched keywords are never sent, even
  if the sheet was generated with ``--show-predictions``, so the reviewer cannot be
  anchored by the baseline they are meant to audit.
* **Labels stay inside the taxonomy.** Any other value is rejected with a hard error,
  so a typo cannot silently change the metrics later computed from this sheet.
* **Every write is auditable and reversible.** Each change is appended to a JSON-lines
  log next to the sheet, and ``undo`` restores the previous values.
* **Nothing is clobbered.** The sheet is rewritten atomically, preserving the original
  column order and every other column; existing labels are only replaced when the
  reviewer relabels the same row on purpose.

The server binds to loopback only: this is a single-user research tool, not a service.
Run it, label what you can, close the tab — progress is already on disk, and
``python -m stock_project.nlp.evaluate_classifier`` picks it up.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ..common.config import Config, load_config
from ..common.errors import DataNotFetchedError
from ..common.logging_utils import configure_cli_logging, get_logger
from ..common.paths import relative_to_root
from ..common.timeutils import utc_now_iso
from .annotate import REVIEW_SAMPLE_FILENAME, review_sheet_path
from .taxonomy import CATEGORY_DEFINITIONS, EVENT_CATEGORIES

logger = get_logger("stock_project.labeller")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8391
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}

LABEL_COLUMN = "event_category"
REQUIRED_COLUMNS = ("article_id", "headline", LABEL_COLUMN)
#: The only fields ever sent to the browser (blind labelling, invariant of Phase 2).
BROWSER_COLUMNS = (
    "article_id",
    "published_at",
    "headline",
    "source",
    "source_category",
    "sampling_reason",
    "notes",
)
#: Columns this tool owns; everything else in the sheet is preserved untouched.
OWNED_COLUMNS = (LABEL_COLUMN, "reviewer", "reviewed_at")


class SheetStore:
    """A review sheet held in memory, written back atomically on every change."""

    def __init__(self, path: Path | str, reviewer: str = "human") -> None:
        self.path = Path(path)
        self.reviewer = (reviewer or "human").strip() or "human"
        if not self.path.exists():
            raise DataNotFetchedError(
                f"no review sheet at {self.path}; run: python -m stock_project.nlp.annotate"
            )
        with self.path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            self.fieldnames: list[str] = list(reader.fieldnames or [])
            missing = [column for column in REQUIRED_COLUMNS if column not in self.fieldnames]
            if missing:
                raise ValueError(
                    f"review sheet {self.path} is missing required column(s) {missing}; "
                    f"found {self.fieldnames}"
                )
            self.rows: list[dict[str, str]] = [
                {key: (value if value is not None else "") for key, value in row.items()}
                for row in reader
            ]
        if not self.rows:
            raise ValueError(f"review sheet {self.path} has no rows to label")
        self._index_by_id: dict[str, int] = {}
        for position, row in enumerate(self.rows):
            article_id = row.get("article_id", "")
            if not article_id:
                raise ValueError(f"row {position + 2} of {self.path} has an empty article_id")
            if article_id in self._index_by_id:
                raise ValueError(f"duplicate article_id {article_id!r} in {self.path}")
            self._index_by_id[article_id] = position
        self._history: list[dict[str, object]] = []
        self._lock = threading.Lock()
        self._pointer = -1
        self.log_path = self.path.parent / f"{self.path.stem}_labelling_log.jsonl"

    # -- state -----------------------------------------------------------------
    @property
    def total(self) -> int:
        return len(self.rows)

    def labelled_count(self) -> int:
        return sum(1 for row in self.rows if row.get(LABEL_COLUMN, "").strip())

    def remaining_count(self) -> int:
        return self.total - self.labelled_count()

    def counts(self) -> dict[str, int]:
        tallies = {category: 0 for category in EVENT_CATEGORIES}
        for row in self.rows:
            value = row.get(LABEL_COLUMN, "").strip()
            if value:
                tallies[value] = tallies.get(value, 0) + 1
        return tallies

    def next_unlabelled(self, after: int = -1) -> int | None:
        for offset in range(1, self.total + 1):
            position = (after + offset) % self.total
            if not self.rows[position].get(LABEL_COLUMN, "").strip():
                return position
        return None

    def state(self) -> dict[str, object]:
        """The payload the browser renders — context columns only."""
        position = self.next_unlabelled(self._pointer)
        article = None
        if position is not None:
            row = self.rows[position]
            article = {column: row.get(column, "") for column in BROWSER_COLUMNS}
        return {
            "sheet": self.path.name,
            "total": self.total,
            "labelled": self.labelled_count(),
            "remaining": self.remaining_count(),
            "counts": self.counts(),
            "labels": [
                {"key": index + 1, "value": category, "definition": CATEGORY_DEFINITIONS[category]}
                for index, category in enumerate(EVENT_CATEGORIES)
            ],
            "article": article,
            "done": position is None,
            "undo_available": bool(self._history),
        }

    # -- mutations -------------------------------------------------------------
    def set_label(self, article_id: str, label: str, note: str = "") -> str:
        """Label one row. Raises ``ValueError`` for anything outside the taxonomy."""
        clean = (label or "").strip().lower()
        if clean not in EVENT_CATEGORIES:
            raise ValueError(
                f"{label!r} is not a taxonomy label; allowed values: {list(EVENT_CATEGORIES)}"
            )
        with self._lock:
            position = self._index_by_id.get(article_id)
            if position is None:
                raise KeyError(f"article_id {article_id!r} is not in {self.path.name}")
            row = self.rows[position]
            previous = {column: row.get(column, "") for column in (*OWNED_COLUMNS,)}
            row[LABEL_COLUMN] = clean
            row["reviewer"] = self.reviewer
            row["reviewed_at"] = utc_now_iso()
            if note:
                row["notes"] = note
            self._write()
            self._history.append({"index": position, "previous": previous})
            self._audit(article_id, "label", previous[LABEL_COLUMN], clean, note)
            self._pointer = position
        return clean

    def undo(self) -> bool:
        """Restore the most recent change. Returns False when there is nothing to undo."""
        with self._lock:
            if not self._history:
                return False
            entry = self._history.pop()
            row = self.rows[int(entry["index"])]
            reverted_from = row.get(LABEL_COLUMN, "")
            row.update(entry["previous"])  # type: ignore[arg-type]
            self._write()
            self._audit(row.get("article_id", ""), "undo", reverted_from, row.get(LABEL_COLUMN, ""), "")
            self._pointer = int(entry["index"])
        return True

    def skip(self) -> None:
        """Advance past the current row without writing anything."""
        with self._lock:
            position = self.next_unlabelled(self._pointer)
            self._pointer = self.total if position is None else position

    # -- persistence -----------------------------------------------------------
    def _write(self) -> None:
        temporary = self.path.with_name(self.path.name + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames, lineterminator="\r\n")
            writer.writeheader()
            writer.writerows(self.rows)
        os.replace(temporary, self.path)

    def _audit(self, article_id: str, action: str, old: str, new: str, note: str) -> None:
        record = {
            "timestamp_utc": utc_now_iso(),
            "sheet": self.path.name,
            "article_id": article_id,
            "action": action,
            "old_label": old,
            "new_label": new,
            "reviewer": self.reviewer,
            "note": note,
        }
        with self.log_path.open("a", encoding="utf-8", newline="") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\r\n")


# -- HTTP layer ----------------------------------------------------------------
PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Phase 2 review sheet</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; background: #f6f7f9; color: #16191d; }
  main { max-width: 46rem; margin: 0 auto; padding: 1.5rem 1rem 3rem; }
  h1 { font-size: 1.1rem; margin: 0 0 .25rem; }
  .meta { color: #5b6570; font-size: .85rem; margin-bottom: 1rem; }
  .bar { background: #dfe3e8; height: .5rem; border-radius: .25rem; overflow: hidden; margin-bottom: 1.25rem; }
  .bar > div { background: #2f6feb; height: 100%; transition: width .15s; }
  .card { background: #fff; border: 1px solid #d7dce2; border-radius: .5rem; padding: 1.25rem; }
  .headline { font-size: 1.3rem; line-height: 1.35; margin: 0 0 .75rem; }
  .context { color: #5b6570; font-size: .85rem; margin-bottom: 1rem; }
  .labels { display: grid; gap: .5rem; }
  button { font: inherit; text-align: left; padding: .6rem .75rem; border: 1px solid #c3cad3;
           border-radius: .35rem; background: #fff; cursor: pointer; }
  button:hover, button:focus { border-color: #2f6feb; background: #f0f5ff; }
  kbd { background: #eef1f4; border: 1px solid #c3cad3; border-radius: .25rem; padding: 0 .35rem;
        margin-right: .5rem; font-size: .8rem; }
  textarea { width: 100%; box-sizing: border-box; margin-top: 1rem; font: inherit; padding: .5rem;
             border: 1px solid #c3cad3; border-radius: .35rem; }
  .actions { margin-top: 1rem; display: flex; gap: .5rem; }
  .actions button { text-align: center; }
  .counts { margin-top: 1.5rem; font-size: .85rem; color: #5b6570; }
  .toast { position: fixed; bottom: 1rem; left: 50%; transform: translateX(-50%); background: #16191d;
           color: #fff; padding: .5rem 1rem; border-radius: .35rem; opacity: 0; transition: opacity .2s; }
  .toast.show { opacity: 1; }
  .done { text-align: center; padding: 2rem 1rem; }
</style>
</head>
<body>
<main>
  <h1>Phase 2 review sheet</h1>
  <div class="meta" id="meta"></div>
  <div class="bar"><div id="fill" style="width:0%"></div></div>
  <div id="stage"></div>
  <div class="counts" id="counts"></div>
</main>
<div class="toast" id="toast"></div>
<script>
let state = null;

function esc(text) {
  const div = document.createElement('div');
  div.textContent = text == null ? '' : String(text);
  return div.innerHTML;
}

function render() {
  if (!state) return;
  const pct = state.total ? Math.round(100 * state.labelled / state.total) : 0;
  document.getElementById('fill').style.width = pct + '%';
  document.getElementById('meta').textContent =
    state.sheet + ' — ' + state.labelled + ' / ' + state.total + ' labelled (' + pct + '%)';

  const stage = document.getElementById('stage');
  if (state.done) {
    stage.innerHTML = '<div class="card done"><h2>All ' + state.total +
      ' rows labelled</h2><p>Next step:</p><p><code>python -m stock_project.nlp.evaluate_classifier ' +
      '--config configs/config.example.yaml</code></p><p class="meta">Press <kbd>u</kbd> to undo the ' +
      'last label.</p></div>';
  } else {
    const a = state.article;
    const buttons = state.labels.map(function (item) {
      return '<button onclick="choose(\\'' + item.value + '\\')" title="' + esc(item.definition) + '">' +
        '<kbd>' + item.key + '</kbd>' + esc(item.value) + '</button>';
    }).join('');
    stage.innerHTML = '<div class="card">' +
      '<p class="headline">' + esc(a.headline) + '</p>' +
      '<div class="context">' + esc(a.published_at) + ' &middot; ' + esc(a.source) +
      ' &middot; sheet row: ' + esc(a.sampling_reason) + '</div>' +
      '<div class="labels">' + buttons + '</div>' +
      '<textarea id="note" rows="2" placeholder="optional note for this row"></textarea>' +
      '<div class="actions"><button onclick="act(\\'/api/skip\\')"><kbd>s</kbd>skip</button>' +
      '<button onclick="act(\\'/api/undo\\')"><kbd>u</kbd>undo</button></div></div>';
  }
  const entries = Object.keys(state.counts).map(function (key) { return key + ': ' + state.counts[key]; });
  document.getElementById('counts').textContent = 'labels so far — ' + entries.join(', ');
}

function toast(message) {
  const el = document.getElementById('toast');
  el.textContent = message;
  el.classList.add('show');
  setTimeout(function () { el.classList.remove('show'); }, 2500);
}

function choose(value) {
  const note = document.getElementById('note');
  act('/api/label', { article_id: state.article.article_id, label: value,
                      note: note ? note.value : '' });
}

function act(url, body) {
  return fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify(body || {}) })
    .then(function (response) { return response.json().then(function (data) {
      if (!response.ok) { toast(data.error || 'error'); return; }
      state = data; render();
    }); });
}

document.addEventListener('keydown', function (event) {
  if (event.target.tagName === 'TEXTAREA' || event.target.tagName === 'INPUT') return;
  if (!state) return;
  if (event.key >= '1' && event.key <= String(state.labels.length)) {
    const item = state.labels[Number(event.key) - 1];
    if (item && state.article) { event.preventDefault(); choose(item.value); }
  } else if (event.key === 's') { event.preventDefault(); act('/api/skip'); }
  else if (event.key === 'u') { event.preventDefault(); act('/api/undo'); }
});

fetch('/api/state').then(function (r) { return r.json(); }).then(function (data) {
  state = data; render();
});
</script>
</body>
</html>
"""


def _json_bytes(payload: object) -> bytes:
    return json.dumps(payload).encode("utf-8")


def make_handler(store: SheetStore) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "Phase2Labeller/1.0"

        def log_message(self, fmt: str, *args: object) -> None:  # keep the CLI output clean
            logger.debug("http %s", fmt % args)

        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: int, payload: object) -> None:
            self._send(status, "application/json; charset=utf-8", _json_bytes(payload))

        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            if self.path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
            elif self.path == "/api/state":
                self._send_json(200, store.state())
            else:
                self._send_json(404, {"error": f"unknown path {self.path}"})

        def do_POST(self) -> None:  # noqa: N802 - stdlib naming
            try:
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(payload, dict):
                    raise ValueError("body must be a JSON object")
                if self.path == "/api/label":
                    store.set_label(
                        str(payload.get("article_id", "")),
                        str(payload.get("label", "")),
                        str(payload.get("note", "") or ""),
                    )
                    self._send_json(200, store.state())
                elif self.path == "/api/skip":
                    store.skip()
                    self._send_json(200, store.state())
                elif self.path == "/api/undo":
                    if not store.undo():
                        self._send_json(400, {"error": "nothing to undo"})
                        return
                    self._send_json(200, store.state())
                else:
                    self._send_json(404, {"error": f"unknown path {self.path}"})
            except (ValueError, KeyError) as error:
                self._send_json(400, {"error": str(error)})
            except json.JSONDecodeError:
                self._send_json(400, {"error": "body is not valid JSON"})

    return Handler


def create_server(store: SheetStore, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    """Build (but do not start) the labelling server, so tests can drive it directly."""
    if host not in LOOPBACK_HOSTS:
        raise ValueError(
            f"refusing to bind non-loopback host {host!r}: this tool writes labels to a local "
            f"CSV with no authentication. Use one of {sorted(LOOPBACK_HOSTS)}."
        )
    server = ThreadingHTTPServer((host, port), make_handler(store))
    server.store = store  # type: ignore[attr-defined]
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Interactively label the Phase 2 review sheet (human labels only; blind by design)."
    )
    parser.add_argument("--config", type=Path, default=None, help="YAML config (default: configs/config.yaml or the example)")
    parser.add_argument("--root", type=Path, default=None, help="project root to read/write (default: this repository)")
    parser.add_argument("--sheet", type=Path, default=None, help=f"sheet to label (default: data/annotations/{REVIEW_SAMPLE_FILENAME})")
    parser.add_argument("--reviewer", default="human", help="name recorded in the reviewer column and the audit log")
    parser.add_argument("--host", default=DEFAULT_HOST, help="loopback host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port to bind (default: {DEFAULT_PORT})")
    parser.add_argument("--no-open", action="store_true", help="print the URL instead of opening a browser tab")
    parser.add_argument("--check", action="store_true", help="print progress and exit without starting the server")
    args = parser.parse_args(argv)

    configure_cli_logging()
    config: Config = load_config(args.config)
    root = args.root if args.root is not None else None
    sheet = args.sheet if args.sheet is not None else review_sheet_path(root, REVIEW_SAMPLE_FILENAME)
    store = SheetStore(sheet, reviewer=args.reviewer)

    progress = store.state()
    sheet_label = relative_to_root(store.path, root)
    print(f"sheet     : {sheet_label}")
    print(f"rows      : {progress['total']}")
    print(f"labelled  : {progress['labelled']}")
    print(f"remaining : {progress['remaining']}")
    if args.check:
        print("reviewer  :", store.reviewer)
        return 0

    server = create_server(store, host=args.host, port=args.port)
    host, port = server.server_address[0], server.server_address[1]
    url = f"http://{args.host}:{port}/"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("labelling %s at %s (labels are written to %s as you go)", sheet_label, url, sheet_label)
    print(f"\nLabel at {url} — press 1-6 for a category, s to skip, u to undo.")
    print("Close this terminal to stop; progress is saved after every label.")
    if not args.no_open:
        try:
            webbrowser.open(url)
        except Exception:  # pragma: no cover - depends on the desktop environment
            logger.debug("could not open a browser automatically; open %s by hand", url)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print("\nstopping")
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a CLI
    raise SystemExit(main())

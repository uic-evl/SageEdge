#!/usr/bin/env python3
import argparse
import json
import os
import queue
import re
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
CLEAR_LINE_RE = re.compile(r"\x1b\[2K")
TIMESTAMP_RE = re.compile(r"^\[[0-9:.,\s\-]+-->\s*[0-9:.,\s\-]+\]\s*(.*)$")
CLEAR_SPACES_RE = re.compile(r"\r+\s{20,}\r+")

clients = set()
clients_lock = threading.Lock()
history = []


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Whisper Live Transcript Display</title>
  <style>
    :root {
      color-scheme: light dark;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f5f7f8;
      color: #172026;
    }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 18px;
      border-bottom: 1px solid #d7dee2;
      background: #fff;
    }
    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 650;
    }
    .model {
      display: inline-flex;
      align-items: center;
      margin-left: 10px;
      padding: 3px 8px;
      border: 1px solid #b8c4cb;
      border-radius: 999px;
      color: #41515c;
      font-size: 12px;
      font-weight: 500;
      vertical-align: middle;
    }
    button {
      padding: 8px 12px;
      border: 1px solid #aebbc3;
      border-radius: 6px;
      background: #fff;
      color: inherit;
      font: inherit;
      cursor: pointer;
    }
    .status {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 10px 18px;
      border-bottom: 1px solid #dfe5e8;
      color: #41515c;
      font-size: 13px;
    }
    .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: #9aa8b0;
    }
    .dot.live {
      background: #0d9f76;
    }
    pre {
      margin: 0;
      overflow-y: auto;
      padding: 18px;
      white-space: pre-wrap;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      font-size: 18px;
      line-height: 1.45;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        background: #101517;
        color: #eef3f5;
      }
      header, button {
        background: #172026;
      }
      header, .status, button, .model {
        border-color: #2b373d;
      }
      .status, .model {
        color: #a8b5bc;
      }
    }
  </style>
</head>
<body>
  <header>
    <h1>Whisper Transcript Display <span class="model">base.en VAD</span></h1>
    <button id="clear">Clear</button>
  </header>
  <div class="status"><span id="dot" class="dot"></span><span id="status">Connecting</span></div>
  <pre id="output">Waiting for live base.en VAD transcript output...</pre>
  <script>
    const output = document.querySelector("#output");
    const statusEl = document.querySelector("#status");
    const dot = document.querySelector("#dot");
    const clear = document.querySelector("#clear");
    let lines = [];
    let currentLine = "";
    let started = false;

    function render() {
      output.textContent = [...lines.slice(-220), currentLine].join("\\n") || " ";
      output.scrollTop = output.scrollHeight;
    }

    function write(text) {
      if (!started) {
        started = true;
        lines = [];
        currentLine = "";
      }
      for (const ch of text) {
        if (ch === "\\r") {
          currentLine = "";
        } else if (ch === "\\n") {
          if (currentLine.trim() || lines.length) lines.push(currentLine);
          currentLine = "";
        } else {
          currentLine += ch;
        }
      }
      render();
    }

    clear.addEventListener("click", () => {
      lines = [];
      currentLine = "";
      output.textContent = "Cleared.";
    });

    const events = new EventSource("/events");
    events.addEventListener("open", () => {
      statusEl.textContent = "Connected";
      dot.classList.add("live");
    });
    events.addEventListener("message", (event) => {
      const data = JSON.parse(event.data);
      if (data.kind === "terminal") write(data.text);
      if (data.kind === "status") statusEl.textContent = data.text;
    });
    events.addEventListener("error", () => {
      statusEl.textContent = "Reconnecting";
      dot.classList.remove("live");
    });
  </script>
</body>
</html>
"""


def clean_chunk(text):
    text = CLEAR_LINE_RE.sub("\r", text)
    text = ANSI_RE.sub("", text)
    text = CLEAR_SPACES_RE.sub("\r", text)
    text = text.replace("[BLANK_AUDIO]", "")

    cleaned = []
    for part in re.split(r"[\r\n]+", text):
        line = part.strip()
        if not line:
            continue
        if line.startswith("Script started") or line.startswith("Script done"):
            continue
        if line.startswith((
            "init:",
            "whisper_",
            "whisper_print_timings:",
            "system_info:",
            "main:",
            "ggml_",
            "NvRm",
            "****",
            "ALSA ",
        )):
            continue
        if line.startswith("###"):
            continue
        if "samples per frame" in line or "required:" in line:
            continue
        match = TIMESTAMP_RE.match(line)
        if match:
            line = match.group(1).strip()
        if line and line != "[Start speaking]":
            cleaned.append(line)

    if not cleaned:
        return ""
    return "\n".join(cleaned) + "\n"


def broadcast(event):
    history.append(event)
    del history[:-100]
    with clients_lock:
        stale = []
        for client in clients:
            try:
                client.put_nowait(event)
            except queue.Full:
                stale.append(client)
        for client in stale:
            clients.discard(client)


def follow_file(path):
    broadcast({"kind": "status", "text": f"Watching {path}"})
    last_inode = None
    pos = 0
    while True:
        try:
            stat = path.stat()
            if last_inode != stat.st_ino:
                last_inode = stat.st_ino
                pos = 0
                broadcast({"kind": "status", "text": f"Reading {path}"})
            elif stat.st_size < pos:
                pos = 0
                broadcast({"kind": "status", "text": f"Reading new transcript from {path}"})
            with path.open("rb") as f:
                f.seek(pos)
                data = f.read()
                pos = f.tell()
            if data:
                text = clean_chunk(data.decode("utf-8", errors="replace"))
                if text:
                    broadcast({"kind": "terminal", "text": text})
        except FileNotFoundError:
            broadcast({"kind": "status", "text": f"Waiting for {path}"})
        time.sleep(0.03)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("content-type", "text/html; charset=utf-8")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/events":
            self.handle_events()
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def handle_events(self):
        q = queue.Queue(maxsize=100)
        with clients_lock:
            clients.add(q)
        self.send_response(HTTPStatus.OK)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "keep-alive")
        self.end_headers()
        for event in history[-30:]:
            self.write_event(event)
        try:
            while True:
                try:
                    self.write_event(q.get(timeout=15))
                except queue.Empty:
                    self.write_event({"kind": "status", "text": "Connected"})
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with clients_lock:
                clients.discard(q)

    def write_event(self, event):
        self.wfile.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))
        self.wfile.flush()


def main():
    parser = argparse.ArgumentParser(description="Display a live transcript file in a browser")
    parser.add_argument("--file", default="/tmp/whisper-transcript.txt")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()

    path = Path(args.file).expanduser().resolve()
    threading.Thread(target=follow_file, args=(path,), daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Open http://{args.host}:{args.port}")
    print(f"Mirroring {path}")
    server.serve_forever()


if __name__ == "__main__":
    main()

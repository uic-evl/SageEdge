#!/usr/bin/env python3
import argparse
import json
import os
import queue
import re
import signal
import subprocess
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
STREAM_BIN = ROOT / "build" / "bin" / "whisper-stream"
DEFAULT_MODEL = ROOT / "models" / "ggml-tiny.en.bin"

clients = set()
clients_lock = threading.Lock()
history = []
state_lock = threading.Lock()
stream_proc = None
reader_thread = None


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Live Whisper Transcript</title>
  <style>
    :root {
      color-scheme: light dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
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
      background: #ffffff;
    }
    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 650;
    }
    .controls {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    label {
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 13px;
      color: #41515c;
    }
    input, select {
      width: 72px;
      padding: 7px 8px;
      border: 1px solid #bdc8cf;
      border-radius: 6px;
      font: inherit;
      background: #ffffff;
      color: inherit;
    }
    select {
      width: 118px;
    }
    button {
      padding: 8px 12px;
      border: 1px solid #aebbc3;
      border-radius: 6px;
      background: #ffffff;
      color: #172026;
      font: inherit;
      cursor: pointer;
    }
    button.primary {
      border-color: #0d6b5f;
      background: #0d6b5f;
      color: #ffffff;
    }
    button:disabled {
      cursor: wait;
      opacity: 0.6;
    }
    main {
      display: grid;
      grid-template-rows: auto 1fr;
      min-height: 0;
    }
    .status {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 10px 18px;
      border-bottom: 1px solid #dfe5e8;
      font-size: 13px;
      color: #41515c;
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
    .transcript {
      overflow-y: auto;
      padding: 20px 18px 80px;
      font-size: 18px;
      line-height: 1.45;
      white-space: pre-wrap;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
    }
    .meta {
      color: #63727c;
      font-size: 13px;
      margin: 0 0 8px;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        background: #101517;
        color: #eef3f5;
      }
      header, input, select, button {
        background: #172026;
        color: #eef3f5;
      }
      header, .status {
        border-color: #2b373d;
      }
      input, select, button {
        border-color: #44545d;
      }
      label, .status, .meta {
        color: #a8b5bc;
      }
    }
  </style>
</head>
<body>
  <header>
    <h1>Live Whisper Transcript</h1>
    <div class="controls">
      <label>Model
        <select id="model">
          <option value="models/ggml-tiny.en.bin">tiny.en</option>
          <option value="models/ggml-base.en.bin">base.en</option>
        </select>
      </label>
      <label>Mic <input id="capture" type="number" value="0" min="0"></label>
      <label>Step ms <input id="step" type="number" value="500" min="0" step="100"></label>
      <label>Length ms <input id="length" type="number" value="5000" min="1000" step="500"></label>
      <button id="clear">Clear</button>
      <button id="stop">Stop</button>
      <button id="start" class="primary">Start</button>
    </div>
  </header>
  <main>
    <div class="status"><span id="dot" class="dot"></span><span id="status">Disconnected</span></div>
    <pre id="transcript" class="transcript" aria-live="polite">Press Start and speak into the selected microphone.</pre>
  </main>
  <script>
    const statusEl = document.querySelector("#status");
    const dot = document.querySelector("#dot");
    const transcript = document.querySelector("#transcript");
    const startBtn = document.querySelector("#start");
    const stopBtn = document.querySelector("#stop");
    const clearBtn = document.querySelector("#clear");
    const model = document.querySelector("#model");
    const capture = document.querySelector("#capture");
    const step = document.querySelector("#step");
    const length = document.querySelector("#length");
    let lines = [];
    let currentLine = "";
    let hasOutput = false;

    function setStatus(text, live) {
      statusEl.textContent = text;
      dot.classList.toggle("live", Boolean(live));
    }

    function renderTerminal() {
      const text = [...lines.slice(-160), currentLine].join("\\n");
      transcript.textContent = text || " ";
      transcript.scrollTop = transcript.scrollHeight;
    }

    function writeTerminal(text) {
      if (!text) return;
      if (!hasOutput) {
        lines = [];
        currentLine = "";
        hasOutput = true;
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
      renderTerminal();
    }

    function appendLine(text) {
      writeTerminal("\\n" + text + "\\n");
    }

    async function post(path, body) {
      startBtn.disabled = true;
      stopBtn.disabled = true;
      try {
        const res = await fetch(path, {
          method: "POST",
          headers: {"content-type": "application/json"},
          body: JSON.stringify(body || {})
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "request failed");
        return data;
      } finally {
        startBtn.disabled = false;
        stopBtn.disabled = false;
      }
    }

    startBtn.addEventListener("click", async () => {
      try {
        const data = await post("/start", {
          model: model.value,
          capture: capture.value,
          step: step.value,
          length: length.value
        });
        appendLine(data.message);
      } catch (err) {
        appendLine(err.message);
      }
    });

    stopBtn.addEventListener("click", async () => {
      try {
        const data = await post("/stop");
        appendLine(data.message);
      } catch (err) {
        appendLine(err.message);
      }
    });

    clearBtn.addEventListener("click", () => {
      lines = [];
      currentLine = "";
      hasOutput = false;
      transcript.textContent = "Transcript cleared.";
    });

    const events = new EventSource("/events");
    events.addEventListener("open", () => setStatus("Connected", false));
    events.addEventListener("message", (event) => {
      const data = JSON.parse(event.data);
      if (data.kind === "status") setStatus(data.text, data.live);
      if (data.kind === "terminal") writeTerminal(data.text);
      if (data.kind === "log") appendLine(data.text);
    });
    events.addEventListener("error", () => setStatus("Reconnecting", false));
  </script>
</body>
</html>
"""


def broadcast(event):
    history.append(event)
    del history[:-100]
    with clients_lock:
        stale = []
        for client in clients:
            try:
                client.put_nowait(event)
            except Exception:
                stale.append(client)
        for client in stale:
            clients.discard(client)


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
CLEAR_LINE_RE = re.compile(r"\x1b\[2K")


def clean_terminal_chunk(chunk):
    chunk = CLEAR_LINE_RE.sub("\r", chunk)
    chunk = ANSI_RE.sub("", chunk)
    chunk = chunk.replace("[BLANK_AUDIO]", "")
    return chunk


def read_stream(proc):
    broadcast({"kind": "status", "text": "Listening", "live": True})
    try:
        while True:
            try:
                data = os.read(proc.stdout.fileno(), 4096)
            except OSError:
                break
            if not data:
                break
            text = data.decode("utf-8", errors="replace")
            text = clean_terminal_chunk(text)
            if text:
                broadcast({"kind": "terminal", "text": text})
    finally:
        code = proc.wait()
        broadcast({"kind": "status", "text": f"Stopped ({code})", "live": False})


def start_stream(opts):
    global stream_proc, reader_thread
    with state_lock:
        if stream_proc and stream_proc.poll() is None:
            return False, "Whisper stream is already running."

        model = Path(opts.get("model") or DEFAULT_MODEL)
        if not model.is_absolute():
            model = ROOT / model
        if not STREAM_BIN.exists():
            raise RuntimeError(f"missing binary: {STREAM_BIN}")
        if not model.exists():
            raise RuntimeError(f"missing model: {model}")

        cmd = [
            "stdbuf",
            "-o0",
            "-e0",
            str(STREAM_BIN),
            "-m",
            str(model),
            "-t",
            str(opts.get("threads") or 4),
            "--step",
            str(opts.get("step") or 500),
            "--length",
            str(opts.get("length") or 5000),
            "-c",
            str(opts.get("capture") or 0),
        ]
        stream_proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        reader_thread = threading.Thread(target=read_stream, args=(stream_proc,), daemon=True)
        reader_thread.start()
        return True, "Started Whisper stream."


def stop_stream():
    global stream_proc
    with state_lock:
        proc = stream_proc
        if not proc or proc.poll() is not None:
            return False, "Whisper stream is not running."
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        return True, "Stopped Whisper stream."


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def send_json(self, data, status=HTTPStatus.OK):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

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

    def do_POST(self):
        length = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self.send_json({"error": "invalid JSON"}, HTTPStatus.BAD_REQUEST)
            return
        try:
            if self.path == "/start":
                _, message = start_stream(data)
                self.send_json({"message": message})
                return
            if self.path == "/stop":
                _, message = stop_stream()
                self.send_json({"message": message})
                return
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
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

        for event in history[-20:]:
            self.write_event(event)
        self.write_event({"kind": "status", "text": "Connected", "live": False})

        try:
            while True:
                try:
                    event = q.get(timeout=15)
                    self.write_event(event)
                except queue.Empty:
                    self.write_event({"kind": "ping", "text": str(time.time())})
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with clients_lock:
                clients.discard(q)

    def write_event(self, event):
        data = json.dumps(event)
        self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
        self.wfile.flush()


def main():
    parser = argparse.ArgumentParser(description="Browser page for whisper.cpp live transcripts")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Open http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        stop_stream()


if __name__ == "__main__":
    main()

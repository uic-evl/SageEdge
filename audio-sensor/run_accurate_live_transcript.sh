#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

OUT_FILE="${OUT_FILE:-/tmp/whisper-transcript.txt}"
MODEL="${MODEL:-models/ggml-base.en.bin}"
CAPTURE="${CAPTURE:-0}"
THREADS="${THREADS:-4}"
STEP_MS="${STEP_MS:-0}"
LENGTH_MS="${LENGTH_MS:-8000}"
VAD_THOLD="${VAD_THOLD:-0.6}"

: > "$OUT_FILE"

echo "Writing transcript display file: $OUT_FILE"
echo "Model: $MODEL"
echo "Open the display at: http://127.0.0.1:8091"
echo

cmd="./build/bin/whisper-stream -m \"$MODEL\" -t \"$THREADS\" --step \"$STEP_MS\" --length \"$LENGTH_MS\" -vth \"$VAD_THOLD\" -c \"$CAPTURE\""

exec script -q -f "$OUT_FILE" -c "$cmd"

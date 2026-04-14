# VOXORA: Real-Time Hybrid Speech-to-Text (CPU Optimized)

This project implements a low-latency, Google Keyboard-style voice typing pipeline:

- Streaming ASR (real-time): Vosk
- Final ASR (accuracy pass after stop): faster-whisper
- Bridge: Node.js WebSocket + FFmpeg
- UI: React/Next.js with live partial text and final replacement

## Architecture

1. Frontend records microphone audio with `MediaRecorder` at ~25 ms chunks.
2. Node WebSocket server receives binary chunks.
3. Node uses FFmpeg to decode and resample to `pcm_f32le`, `16 kHz`, `mono`.
4. Node slices audio into small frames (~20 ms) and streams to Python.
5. Python Vosk recognizer emits partial updates continuously.
6. Python buffers all audio in memory while streaming.
7. On stop, Python runs faster-whisper on full buffered audio.
8. Final corrected transcript replaces live text in UI.

## WebSocket Message Contract

Messages from backend to frontend:

```json
{ "type": "partial", "text": "hello this is rea" }
{ "type": "final", "text": "hello this is real time transcription" }
```

Implemented `partial` payload also includes optional stabilization fields:

```json
{ "type": "partial", "text": "hello this is rea", "stable": "hello this is", "partial": "rea" }
```

## System Requirements

- Windows, macOS, or Linux
- Node.js 18+
- Python 3.10+
- FFmpeg available on `PATH`
- CPU-only supported (Intel i5 class machine recommended with `small.en`)

Check FFmpeg:

```bash
ffmpeg -version
```

## Project Structure

- `backend/src/websocket/socket.js`: WebSocket + FFmpeg streaming bridge
- `backend/src/services/python.service.js`: framed TCP client to Python
- `backend/python/transcriber_server.py`: Vosk streaming + Whisper final pass
- `frontend/src/components/Recorder.tsx`: UI and recording flow
- `frontend/src/lib/websocket.ts`: frontend WS client

## Install

### 1) Install JavaScript dependencies

```bash
cd backend
npm install
cd ../frontend
npm install
```

### 2) Install Python dependencies

```bash
cd ../backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r python/requirements.txt
```

### 3) Download Vosk model

Windows PowerShell:

```powershell
cd backend/python
New-Item -ItemType Directory -Force models | Out-Null
Invoke-WebRequest -Uri "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip" -OutFile "models/vosk-model-small-en-us-0.15.zip"
Expand-Archive -Path "models/vosk-model-small-en-us-0.15.zip" -DestinationPath "models" -Force
```

Expected folder:

```text
backend/python/models/vosk-model-small-en-us-0.15
```

Notes:
- faster-whisper model downloads automatically on first Python startup.
- For Intel i5 CPU, keep `WHISPER_MODEL_SIZE=small.en` for responsiveness.
- If CPU headroom allows, try `WHISPER_MODEL_SIZE=medium.en` for better final accuracy.

## Configuration

### Backend env

Create `backend/.env` from `backend/.env.example` (already included defaults):

```env
PORT=5000
PYTHON_HOST=127.0.0.1
PYTHON_PORT=6000
STREAM_FRAME_MS=20

PY_SERVER_HOST=127.0.0.1
PY_SERVER_PORT=6000
STREAM_QUEUE_SIZE=1024
MAX_AUDIO_BUFFER_SECONDS=1800

VOSK_MODEL_PATH=python/models/vosk-model-small-en-us-0.15
VOSK_MIN_CONFIDENCE=0.6
PARTIAL_MIN_CONFIDENCE=0.6
PARTIAL_HISTORY_SIZE=3
SPEECH_RMS_THRESHOLD=0.01
SILENCE_THRESHOLD_MS=600
SILENCE_COMMIT_DELAY_MS=200
MIN_SEGMENT_WORDS=2
EMPTY_PARTIAL_STREAK_COMMIT=3
ENABLE_PAUSE_PUNCTUATION=true
COMMA_PAUSE_MS=700
SENTENCE_PAUSE_MS=1300
PHRASE_BIAS_TERMS=Arpan,Voxora,WebSocket,Node.js,Python,Whisper,Vosk
CALIBRATION_WINDOW_SECONDS=12
LOW_CONFIDENCE_HOLD_MS=80
ENABLE_FILLER_FILTER=true
ENABLE_PREDICTIVE_SMOOTHING=true
PERSONAL_DICT_PATH=python/personal_dictionary.json
ENABLE_SESSION_METRICS_LOG=true
SESSION_METRICS_LOG_PATH=python/session_metrics.json
MAX_SESSION_METRICS_ITEMS=400
FEEDBACK_LOOP_INTERVAL_MS=1200
CORRECTION_FEEDBACK_HIGH_RATE=0.25
LATENCY_FEEDBACK_HIGH_MS=165
SILENCE_FEEDBACK_FREQUENT_RATE=0.28
CPU_PRESSURE_RATIO=0.9
CPU_GUARD_EMIT_SHIFT_MS=18

WHISPER_MODEL_SIZE=small.en
WHISPER_COMPUTE_TYPE=int8
WHISPER_CPU_THREADS=4
```

### Frontend env

Create `frontend/.env.local`:

```env
NEXT_PUBLIC_WS_URL=ws://localhost:5000
NEXT_PUBLIC_RECORDER_CHUNK_MS=25
```

## Run

Option A (single command):

```bash
cd backend
npm run dev:all
```

Option B (3 terminals):

1. Python server:
```bash
cd backend
npm run python
```

2. Node backend:
```bash
cd backend
npm run dev
```

3. Frontend:
```bash
cd frontend
npm run dev
```

Open:

```text
http://localhost:3000
```

## Runtime Behavior

- While speaking: continuous low-latency partial transcript updates.
- Self-tuning stabilization: required confirmations auto-adapt (slow speech -> 1, normal -> 2, fast -> 3).
- Session auto-calibration (first 10-15s): adapts confidence thresholds, stability, and emit cadence to speaking style.
- Adaptive confidence gating: first appearance uses stricter confidence than repeated appearances.
- Adaptive emission cadence: faster while actively speaking, slower during pauses for smoother UX.
- Smart low-confidence hold (~80ms): reduces flash-of-wrong-word artifacts.
- Hesitation filter: suppresses isolated fillers (`uh`, `um`, `you know`, edge-case `like`) while keeping intentional phrasing.
- Lightweight predictive smoothing: high-confidence tails can render a conservative predicted continuation.
- Runtime feedback loop: correction rate, silence behavior, and latency signals continuously tune confirmation, confidence, and pause sensitivity.
- CPU guardrail: when process pressure rises, prediction is reduced and emit interval is relaxed to keep UI responsive.
- Short pauses: stabilization logic reduces flicker and can add lightweight pause punctuation.
- On stop: full-audio whisper pass produces corrected final transcript.
- Error-pattern learning: learns case/phrase corrections from partial-vs-final differences and reuses them next session.
- UI replaces live partial text with final transcript using trust-layer transitions (minor corrections merge softly; major changes highlight).

## CPU Tuning Tips (Intel i5, no GPU)

- Keep `WHISPER_COMPUTE_TYPE=int8`.
- Start with `WHISPER_MODEL_SIZE=small.en`.
- Use `WHISPER_CPU_THREADS=4` (or half your logical cores).
- Keep frontend chunks at `25 ms` and stream frames at `20 ms`.
- Partial cadence and stability now self-tune automatically.
- If words appear too conservative, lower `PARTIAL_MIN_CONFIDENCE` slightly.

## Bonus Features Included

- Lightweight punctuation/grammar post-processing
- Self-tuning partial stabilization with adaptive confidence and cadence
- Vosk confidence filtering for weak words
- Smart hesitation filtering and low-confidence holding
- Session memory + persisted personal dictionary (`PERSONAL_DICT_PATH`)
- Production session observability log (`SESSION_METRICS_LOG_PATH`) with avg WPS, correction rate, and latency
- Smooth partial-to-final transition with correction-aware highlighting
- Phrase biasing for names and product/technical terms

## Developer Debug Panel

- Toggle in the UI with `Ctrl + Shift + D` (or the hidden `D` corner button).
- Panel surfaces live calibration/audio/transcription/intelligence metrics with color health indicators.
- Designed for low overhead and can stay closed in production.

## Troubleshooting

- `Python connection failed`: ensure Python server is running on port `6000`.
- `FFmpeg error`: verify FFmpeg is installed and on `PATH`.
- Empty transcript: confirm microphone permission and Vosk model path.
- Slow final transcript: reduce whisper model size to `small.en`.

# Real-Time Hybrid Speech-to-Text (Vosk + faster-whisper)

Low-latency, CPU-friendly speech-to-text system:
- ⚡ Real-time streaming (Vosk)
- 🧠 High-accuracy final transcription (faster-whisper)
- 🎯 Smooth, stable UI updates

---

## 📡 WebSocket Messages

```json
{ "type": "partial", "stable": "hello this is", "partial": " real ti" }
{ "type": "final", "text": "Hello, this is real-time transcription." }
```

---

## ⚙️ Prerequisites

- Node.js 18+
- Python 3.10+
- FFmpeg (added to PATH)

Check FFmpeg:

```bash
ffmpeg -version
```

---

## 📦 Installation

### 1. Clone & Install

```bash
git clone <your-repo-url>
cd <your-repo-name>

cd backend && npm install
cd ../frontend && npm install
cd ../backend && python -m venv .venv && .venv\Scripts\Activate && pip install vosk faster-whisper numpy sounddevice python-dotenv
```

---

## 🎙️ Download Vosk Model

### Windows (PowerShell)

```powershell
cd backend/python

New-Item -ItemType Directory -Force models | Out-Null

Invoke-WebRequest `
  -Uri "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip" `
  -OutFile "models/vosk-model-small-en-us-0.15.zip"

Expand-Archive `
  -Path "models/vosk-model-small-en-us-0.15.zip" `
  -DestinationPath "models" `
  -Force
```

### macOS / Linux

```bash
cd backend/python

mkdir -p models
curl -L -o models/vosk-model-small-en-us-0.15.zip \
https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip

unzip models/vosk-model-small-en-us-0.15.zip -d models
```

✅ Final folder structure:

```
backend/python/models/vosk-model-small-en-us-0.15
```

---

## ▶️ Run the App

### 🚀 Single Command (Recommended)

```bash
cd backend
npm run dev:all
```

Runs:
- 🧠 Python server  
- 🔄 Backend server  
- 🌐 Frontend  

All in one terminal with unified logs. This may thorw an error.

---

### 🧪 Manual (3 Terminals)

#### Terminal A — Python Server

```bash
cd backend
npm run python
```

#### Terminal B — Backend Server

```bash
cd backend
npm run dev
```

#### Terminal C — Frontend

```bash
cd frontend
npm run dev
```

---

## 🌐 Open App

```
http://localhost:3000
```

---

## ⚡ Performance Tuning

### Python

| Variable                    | Default  | Purpose              |
|---------------------------|----------|----------------------|
| VOSK_MIN_CONFIDENCE       | 0.6      | Filter weak words    |
| SPEECH_RMS_THRESHOLD      | 0.01     | Detect speech        |
| SILENCE_THRESHOLD_MS      | 600      | Pause detection      |
| SILENCE_COMMIT_DELAY_MS   | 200      | Delay before commit  |
| MIN_SEGMENT_WORDS         | 2        | Prevent short splits |
| EMPTY_PARTIAL_STREAK_COMMIT | 3      | Force commit         |
| WHISPER_MODEL_SIZE        | small.en | Accuracy vs speed    |
| WHISPER_COMPUTE_TYPE      | int8     | CPU optimization     |
| WHISPER_CPU_THREADS       | 4        | Parallel processing  |

---

### Node

| Variable        | Default |
|----------------|--------|
| PORT           | 5000   |
| STREAM_FRAME_MS| 20     |

---

### Frontend

| Variable                      | Default             |
|-----------------------------|---------------------|
| NEXT_PUBLIC_WS_URL          | ws://localhost:5000 |
| NEXT_PUBLIC_RECORDER_CHUNK_MS | 25                |

---

## 🚀 How It Works

- 🎤 Browser captures audio (~25ms chunks)
- 🔄 Node converts → 16kHz mono float32
- ⚡ Streams ~20ms frames to Python
- 🧠 Vosk generates real-time partial text
- ⏸️ Silence finalizes stable segments
- 🎯 Whisper refines final transcript
- ✨ UI updates smoothly without flicker

---

## 🧪 Behavior

- Instant typing while speaking
- Words stabilize after short pauses
- Final transcript replaces partial text cleanly
- Processing continues after recording stops

---

## 📦 Python Requirements

```
faster-whisper>=1.1.0
numpy>=1.24.0
python-dotenv>=1.0.1
vosk>=0.3.45
```
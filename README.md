# VOXORA: Premium Real-Time Speech-to-Text

VOXORA is a high-performance, locally hosted voice dictation application built for speed, accuracy, and aesthetics. It utilizes a dual-model architecture to provide zero-latency live subtitles while you speak, instantly replacing them with a highly accurate, perfectly punctuated transcript the moment you stop.

## Core Architecture

VOXORA is built on a split-engine architecture designed specifically for CPU-only environments:

1. **Vosk (Real-Time Engine)**: As you speak, audio is streamed via WebSockets to a lightweight Vosk model. This model is optimized for maximum speed (disabling heavy word-level lattice alignment) to provide the instant "live typing" subtitle effect without taxing your CPU.
2. **faster-whisper (Final Engine)**: When you click "Stop Dictation", the backend takes the complete, uncompressed audio buffer and runs it through the high-accuracy Whisper model. This guarantees perfect punctuation, capitalization, and context-awareness.

### Recent Architectural Enhancements
- **Zero Audio Drop**: The frontend is explicitly wired to wait for the `MediaRecorder.onstop` event. This guarantees the final chunk of audio is flushed to the backend before the stop signal is sent, ensuring your very last word is never cut off.
- **Continuous Speech Guardrail**: To prevent the real-time engine from hanging or dropping frames during long dictations, the backend automatically triggers a forced memory lattice reset every 30 seconds of continuous speech. This keeps performance blazing fast forever.
- **Bypassed Jitter Filters**: The frontend renders the raw partial strings directly from Vosk, bypassing heavy debouncing logic that previously caused words to be incorrectly swallowed.

## The Interface: Ivory & Midnight

The frontend is built with React/Next.js and features a fully responsive, premium card-based layout. 

- **Dynamic Theme Engine**: Toggle between **Ivory** (a warm, editorial off-white with high-contrast typography) and **Midnight** (a deep, luxurious emerald/cyan dark mode).
- **Interactive Canvas**: The background features a highly optimized HTML5 Canvas element that tracks your mouse, emitting a subtle, theme-aware glitter effect (gold/teal for Ivory, emerald/cyan for Midnight).
- **Smooth Typography**: Uses system fonts with carefully tuned weights and contrast for maximum readability during rapid dictation.
- **Scrollable & Responsive**: The layout uses a natural document flow, ensuring that even on smaller screens or compressed windows, you can scroll through your entire transcript effortlessly.
- **Quick Actions**: The Live Transcript header features one-click "Copy" and "Clear" buttons to quickly manage your text.

## Tech Stack

- **Frontend**: Next.js, React, Framer Motion (for smooth text transitions), Tailwind CSS (customized with CSS variables), Lucide React (Icons)
- **Backend Bridge**: Node.js WebSocket Server + FFmpeg (for decoding WebM to raw PCM)
- **Transcription Server**: Python 3.10+, Vosk, faster-whisper, numpy

## Setup & Installation

### 1. Install Node Dependencies
```bash
cd backend
npm install
cd ../frontend
npm install
```

### 2. Install Python Dependencies
```bash
cd backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r python/requirements.txt
```

### 3. Download Models
Download the Vosk small model and extract it to the backend folder:
```powershell
cd backend/python
New-Item -ItemType Directory -Force models | Out-Null
Invoke-WebRequest -Uri "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip" -OutFile "models/vosk-model-small-en-us-0.15.zip"
Expand-Archive -Path "models/vosk-model-small-en-us-0.15.zip" -DestinationPath "models" -Force
```
*Note: The Whisper model (`small.en`) will download automatically on the first run.*

## Running VOXORA

You can launch the entire stack (Node Server, Python Server, and Next.js Frontend) with a single command:

```bash
cd backend
npm run dev:all
```

Navigate to `http://localhost:3000` in your browser.

## Configuration & Tuning

All major configurations are handled via `.env` files. 

For the best CPU performance on standard machines (e.g., Intel i5 without a dedicated GPU):
- Ensure `WHISPER_COMPUTE_TYPE=int8` in your `backend/.env`.
- Keep `WHISPER_MODEL_SIZE=small.en`.
- `WHISPER_CPU_THREADS=4` (adjust based on your logical cores).

## Developer Debug Panel

VOXORA includes a comprehensive internal metrics panel:
- Toggle in the UI with `Ctrl + Shift + D` (or by clicking the hidden 'D' button in the bottom right corner).
- Surfaces live WebSocket latency, server CPU pressure, audio buffering stats, and Whisper queue depth. 
- Designed to run with zero overhead when closed.

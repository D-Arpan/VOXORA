# =========================
# PHASE-1 ENHANCED (BALANCED SPEED + ACCURACY)
# =========================

import json
import socket
import struct
import threading
import queue
import numpy as np
# pyrefly: ignore [missing-import]
from vosk import Model, KaldiRecognizer
from faster_whisper import WhisperModel
import os
import time

HOST = "127.0.0.1"
PORT = 6000
SAMPLE_RATE = 16000

FRAME_AUDIO = 1
FRAME_CONTROL = 2
FRAME_EVENT = 3

HEADER = struct.Struct(">BI")

# =========================
# LOAD MODELS
# =========================
print("[python] Loading models...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "vosk-model-small-en-us-0.15")
vosk_model = Model(MODEL_PATH)

whisper_model = WhisperModel(
    "distil-medium.en",
    device="cpu",
    compute_type="int8",
    cpu_threads=os.cpu_count() or 4
)

print("[python] Models loaded")


# =========================
# SESSION
# =========================
class Session:
    def __init__(self, conn):
        self.conn = conn
        self.queue = queue.Queue()
        self.audio_buffer = bytearray()

        self.recognizer = KaldiRecognizer(vosk_model, SAMPLE_RATE)
        # SetWords(True) removed for max speed on partials

        self.running = True
        self.recording = False

        self.stable_text = ""

        self.stream_thread = threading.Thread(target=self.stream_loop, daemon=True)
        self.final_thread = threading.Thread(target=self.final_loop, daemon=True)

    def send(self, data):
        payload = json.dumps(data).encode()
        packet = HEADER.pack(FRAME_EVENT, len(payload)) + payload
        self.conn.sendall(packet)

    def start(self):
        self.stream_thread.start()
        self.final_thread.start()
        self.read_loop()

    # =========================
    # READ LOOP
    # =========================
    def read_loop(self):
        buffer = bytearray()

        try:
            while self.running:
                data = self.conn.recv(65536)
                if not data:
                    break
                buffer.extend(data)

                while len(buffer) >= 5:
                    t, size = HEADER.unpack(buffer[:5])
                    if len(buffer) < 5 + size:
                        break

                    payload = buffer[5:5+size]
                    del buffer[:5+size]

                    if t == FRAME_AUDIO:
                        if self.recording:
                            self.queue.put(payload)

                            # FIXED AUDIO BUFFER
                            float_audio = np.frombuffer(payload, dtype=np.float32)
                            pcm = (np.clip(float_audio, -1, 1) * 32767).astype(np.int16)
                            self.audio_buffer.extend(pcm.tobytes())

                    elif t == FRAME_CONTROL:
                        msg = json.loads(payload.decode())
                        self.handle_control(msg)
        except ConnectionResetError:
            pass # Gracefully handle abrupt disconnections (e.g., from wait-on)
        except Exception as e:
            print(f"[python] Socket error: {e}")
        finally:
            self.running = False

    # =========================
    # CONTROL
    # =========================
    def handle_control(self, msg):
        action = msg.get("action")

        if action == "start":
            self.recording = True

            # RESET EVERYTHING (IMPORTANT)
            self.audio_buffer.clear()
            self.stable_text = ""

            self.recognizer = KaldiRecognizer(vosk_model, SAMPLE_RATE)

            self.send({"type": "status", "state": "listening"})

        elif action == "stop":
            self.recording = False
            self.queue.put(None)

    # =========================
    # REALTIME (VOSK IMPROVED)
    # =========================
    def stream_loop(self):
        bytes_since_reset = 0
        MAX_BYTES = 32000 * 30  # 30 seconds of continuous audio without pause
        debug_log_path = os.path.join(BASE_DIR, "vosk_debug.log")

        while self.running:
            chunk = self.queue.get()
            if chunk is None:
                break

            audio = np.frombuffer(chunk, dtype=np.float32)
            audio = np.clip(audio, -1, 1)
            pcm = (audio * 32767).astype(np.int16)
            pcm_bytes = pcm.tobytes()

            if self.recognizer.AcceptWaveform(pcm_bytes):
                result = json.loads(self.recognizer.Result())
                text = result.get("text", "").strip()

                if text:
                    self.stable_text += " " + text
                    with open(debug_log_path, "a") as f:
                        f.write(f"AcceptWaveform True. Text: '{text}'. Stable now: '{self.stable_text}'\n")

                self.send({
                    "type": "partial",
                    "stable": self.stable_text.strip(),
                    "partial": ""
                })
                bytes_since_reset = 0
            else:
                bytes_since_reset += len(pcm_bytes)
                
                # FORCE RESET TO PREVENT VOSK HANGING
                if bytes_since_reset > MAX_BYTES:
                    result = json.loads(self.recognizer.PartialResult())
                    partial = result.get("partial", "").strip()
                    if partial:
                        self.stable_text += " " + partial

                    # Re-initialize to drop the huge lattice
                    self.recognizer = KaldiRecognizer(vosk_model, SAMPLE_RATE)
                    bytes_since_reset = 0
                    
                    with open(debug_log_path, "a") as f:
                        f.write(f"Forced Reset! Partial was: '{partial}'\n")

                    self.send({
                        "type": "partial",
                        "stable": self.stable_text.strip(),
                        "partial": ""
                    })
                else:
                    result = json.loads(self.recognizer.PartialResult())
                    partial = result.get("partial", "").strip()
                    
                    if partial:
                        with open(debug_log_path, "a") as f:
                            f.write(f"Partial text updated: '{partial}'\n")

                    self.send({
                        "type": "partial",
                        "stable": self.stable_text.strip(),
                        "partial": partial
                    })

    # =========================
    # FINAL (WHISPER BEST CONFIG)
    # =========================
    def final_loop(self):
        while self.running:
            time.sleep(0.1)

            if not self.recording and len(self.audio_buffer) > 0:
                audio = np.frombuffer(self.audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0

                try:
                    segments, _ = whisper_model.transcribe(
                        audio,
                        language="en",
                        beam_size=2,
                        temperature=0.0,
                        vad_filter=True,
                        condition_on_previous_text=True,
                        initial_prompt="A clear, perfectly punctuated conversation."
                    )

                    final = " ".join(s.text.strip() for s in segments)

                    # FALLBACK
                    if not final.strip():
                        final = self.stable_text

                except Exception as e:
                    print("Whisper error:", e)
                    final = self.stable_text

                self.send({
                    "type": "final",
                    "text": final.strip()
                })

                self.audio_buffer.clear()
                self.stable_text = ""


# =========================
# SERVER
# =========================
def run():
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen()

    print(f"[python] Listening on {HOST}:{PORT}")

    while True:
        conn, _ = server.accept()
        print("[python] Connected")

        session = Session(conn)
        threading.Thread(target=session.start, daemon=True).start()


if __name__ == "__main__":
    run()
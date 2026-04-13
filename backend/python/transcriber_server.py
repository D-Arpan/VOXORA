import json
import os
import queue
import re
import socket
import struct
import threading
import time
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
from faster_whisper import WhisperModel
from vosk import KaldiRecognizer, Model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(BASE_DIR)


def parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        print(f"[python] Invalid {name}={raw!r}; using default {default}")
        return default


def parse_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        print(f"[python] Invalid {name}={raw!r}; using default {default}")
        return default


def parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    print(f"[python] Invalid {name}={raw!r}; using default {default}")
    return default


def parse_str_env(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = str(raw).strip()
    return value if value else default


def resolve_path(path_value: str) -> str:
    path_value = str(path_value or "").strip()
    if not path_value:
        return ""
    if os.path.isabs(path_value):
        return os.path.normpath(path_value)

    backend_relative = os.path.normpath(os.path.join(BACKEND_DIR, path_value))
    base_relative = os.path.normpath(os.path.join(BASE_DIR, path_value))
    cwd_relative = os.path.normpath(os.path.abspath(path_value))

    for candidate in (backend_relative, base_relative, cwd_relative):
        if os.path.exists(candidate):
            return candidate

    # Prefer backend-relative location when no candidate exists yet.
    return backend_relative


try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None
else:
    for env_path in (
        os.path.normpath(os.path.join(BACKEND_DIR, ".env")),
        os.path.normpath(os.path.join(BASE_DIR, ".env")),
    ):
        try:
            if os.path.isfile(env_path):
                load_dotenv(dotenv_path=env_path, override=False)
        except Exception as exc:
            print(f"[python] Failed to load {env_path}: {exc}")

HOST = parse_str_env("PY_SERVER_HOST", "127.0.0.1")
PORT = parse_int_env("PY_SERVER_PORT", 6000)
SAMPLE_RATE = 16000

FRAME_AUDIO = 1
FRAME_CONTROL = 2
FRAME_EVENT = 3
HEADER_STRUCT = struct.Struct(">BI")
HEADER_SIZE = HEADER_STRUCT.size
STOP_TOKEN = object()
STREAM_QUEUE_SIZE = parse_int_env("STREAM_QUEUE_SIZE", 1024)
MAX_FRAME_PAYLOAD_BYTES = parse_int_env("MAX_FRAME_PAYLOAD_BYTES", 8 * 1024 * 1024)
VOSK_MIN_CONFIDENCE = parse_float_env("VOSK_MIN_CONFIDENCE", 0.6)
MAX_AUDIO_BUFFER_SECONDS = parse_int_env("MAX_AUDIO_BUFFER_SECONDS", 1800)
MAX_AUDIO_BUFFER_BYTES = SAMPLE_RATE * 2 * MAX_AUDIO_BUFFER_SECONDS
SILENCE_THRESHOLD_MS = parse_int_env("SILENCE_THRESHOLD_MS", 600)
SILENCE_COMMIT_DELAY_MS = parse_int_env("SILENCE_COMMIT_DELAY_MS", 200)
MIN_SEGMENT_WORDS = parse_int_env("MIN_SEGMENT_WORDS", 2)
SPEECH_RMS_THRESHOLD = parse_float_env("SPEECH_RMS_THRESHOLD", 0.01)
EMPTY_PARTIAL_STREAK_COMMIT = parse_int_env("EMPTY_PARTIAL_STREAK_COMMIT", 3)

DEFAULT_VOSK_MODEL_PATH = os.path.join(BASE_DIR, "models", "vosk-model-small-en-us-0.15")
VOSK_MODEL_PATH = resolve_path(os.getenv("VOSK_MODEL_PATH", DEFAULT_VOSK_MODEL_PATH))
WHISPER_MODEL_SIZE = parse_str_env("WHISPER_MODEL_SIZE", "small.en")
WHISPER_COMPUTE_TYPE = parse_str_env("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_CPU_THREADS = parse_int_env(
    "WHISPER_CPU_THREADS",
    max(1, (os.cpu_count() or 4) // 2),
)
ENABLE_LIGHT_GRAMMAR = parse_bool_env("ENABLE_LIGHT_GRAMMAR", True)

TOKEN_GRAMMAR_REPLACEMENTS = {
    "im": "I'm",
    "ive": "I've",
    "ill": "I'll",
    "id": "I'd",
    "dont": "don't",
    "cant": "can't",
    "wont": "won't",
    "isnt": "isn't",
    "arent": "aren't",
    "wasnt": "wasn't",
    "werent": "weren't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "havent": "haven't",
    "hasnt": "hasn't",
    "hadnt": "hadn't",
    "couldnt": "couldn't",
    "wouldnt": "wouldn't",
    "shouldnt": "shouldn't",
    "youre": "you're",
    "youve": "you've",
    "youll": "you'll",
    "youd": "you'd",
    "weve": "we've",
    "were": "we're",
    "theyre": "they're",
    "theyve": "they've",
    "theyll": "they'll",
    "theyd": "they'd",
    "thats": "that's",
    "theres": "there's",
    "whats": "what's",
    "lets": "let's",
    "hes": "he's",
    "shes": "she's",
    "itll": "it'll",
    "itd": "it'd",
}

VOSK_MODEL_INSTANCE: Optional[Model] = None
WHISPER_MODEL: Optional[WhisperModel] = None
WHISPER_LOCK = threading.Lock()
STARTUP_ERROR: Optional[str] = None

try:
    if not os.path.isdir(VOSK_MODEL_PATH):
        STARTUP_ERROR = (
            "Vosk model not found. Download a Vosk model and set VOSK_MODEL_PATH. "
            f"Checked: {VOSK_MODEL_PATH}"
        )
    else:
        print(f"[python] Loading Vosk model from {VOSK_MODEL_PATH}")
        VOSK_MODEL_INSTANCE = Model(VOSK_MODEL_PATH)

        print(
            "[python] Loading faster-whisper model "
            f"({WHISPER_MODEL_SIZE}, compute_type={WHISPER_COMPUTE_TYPE}, cpu_threads={WHISPER_CPU_THREADS})"
        )
        WHISPER_MODEL = WhisperModel(
            WHISPER_MODEL_SIZE,
            device="cpu",
            compute_type=WHISPER_COMPUTE_TYPE,
            cpu_threads=WHISPER_CPU_THREADS,
            num_workers=1,
        )
except Exception as exc:
    STARTUP_ERROR = f"Startup error: {exc}"

if STARTUP_ERROR:
    print(f"[python] {STARTUP_ERROR}")


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def count_words(text: str) -> int:
    cleaned = normalize_spaces(text)
    if not cleaned:
        return 0
    return len(cleaned.split(" "))


def preserve_token_case(original: str, replacement: str) -> str:
    if not original:
        return replacement
    if original.isupper():
        return replacement.upper()
    if original[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


def apply_lightweight_grammar_rules(text: str) -> str:
    corrected = normalize_spaces(text)
    if not corrected:
        return ""

    corrected = re.sub(r"\s+([,.;!?])", r"\1", corrected)
    corrected = re.sub(r"([,!?;:])([A-Za-z])", r"\1 \2", corrected)
    corrected = re.sub(r"(?<!\d)\.([A-Za-z])", r". \1", corrected)
    corrected = re.sub(r"([!?.,])\1+", r"\1", corrected)
    corrected = re.sub(r"\b([A-Za-z']+)(\s+\1\b)+", r"\1", corrected, flags=re.IGNORECASE)

    def replace_token(match: re.Match[str]) -> str:
        token = match.group(0)
        replacement = TOKEN_GRAMMAR_REPLACEMENTS.get(token.lower())
        if not replacement:
            return token
        return preserve_token_case(token, replacement)

    corrected = re.sub(r"\b[A-Za-z]+\b", replace_token, corrected)
    corrected = re.sub(r"\bi\b", "I", corrected)
    corrected = normalize_spaces(corrected)
    return corrected


def capitalize_sentences(text: str) -> str:
    if not text:
        return ""

    chars = list(text)
    for index, char in enumerate(chars):
        if char.isalpha():
            chars[index] = char.upper()
            break
    capitalized = "".join(chars)

    return re.sub(
        r"([.!?]\s+)([a-z])",
        lambda match: f"{match.group(1)}{match.group(2).upper()}",
        capitalized,
    )


def post_process_text(text: str) -> str:
    cleaned = normalize_spaces(text)
    if not cleaned:
        return ""

    if ENABLE_LIGHT_GRAMMAR:
        cleaned = apply_lightweight_grammar_rules(cleaned)
    else:
        cleaned = re.sub(r"\s+([,.;!?])", r"\1", cleaned)
        cleaned = re.sub(r"\bi\b", "I", cleaned)

    cleaned = capitalize_sentences(cleaned)

    if cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def extract_confident_vosk_text(result_json: Dict) -> str:
    words = result_json.get("result")
    if isinstance(words, list) and words:
        confident_words: List[str] = []
        for word_info in words:
            word = str(word_info.get("word", "")).strip()
            confidence = float(word_info.get("conf", 1.0))
            if word and confidence >= VOSK_MIN_CONFIDENCE:
                confident_words.append(word)
        if confident_words:
            return " ".join(confident_words)

    return str(result_json.get("text", "")).strip()


def extract_partial_text(result_json: Dict) -> str:
    words = result_json.get("partial_result")
    if isinstance(words, list) and words:
        confident_words: List[str] = []
        for word_info in words:
            word = str(word_info.get("word", "")).strip()
            confidence = float(word_info.get("conf", 1.0))
            if word and confidence >= VOSK_MIN_CONFIDENCE:
                confident_words.append(word)
        if confident_words:
            return " ".join(confident_words)

    return str(result_json.get("partial", "")).strip()


class ClientSession:
    def __init__(self, conn: socket.socket, address: Tuple[str, int]):
        self.conn = conn
        self.address = address

        self.send_lock = threading.Lock()
        self.state_lock = threading.Lock()

        self.closed = threading.Event()
        self.final_requested = threading.Event()
        self.stream_flushed = threading.Event()

        self.stream_queue: "queue.Queue[object]" = queue.Queue(maxsize=STREAM_QUEUE_SIZE)
        self.audio_buffer_bytes = bytearray()

        self.is_recording = False
        self.recognizer: Optional[KaldiRecognizer] = None
        self.stable_text = ""
        self.partial_text = ""
        self.last_emitted_stable = ""
        self.last_emitted_partial = ""
        self.last_audio_timestamp = 0.0
        self.last_speech_timestamp = 0.0
        self.last_partial_change_timestamp = 0.0
        self.empty_partial_streak = 0
        self.pause_commit_done = False

        self.stream_thread = threading.Thread(target=self.streaming_loop, daemon=True)
        self.final_thread = threading.Thread(target=self.final_loop, daemon=True)

    def run(self) -> None:
        try:
            self.stream_thread.start()
            self.final_thread.start()
            self.reader_loop()
        except Exception as exc:
            print(f"[python] Session runtime error {self.address}: {exc}")
            traceback.print_exc()
        finally:
            self.close()
            self.stream_thread.join(timeout=2)
            self.final_thread.join(timeout=2)
            print(f"[python] Connection closed: {self.address}")

    def close(self) -> None:
        if self.closed.is_set():
            return

        self.closed.set()
        self.final_requested.set()
        self.enqueue_stop_token()
        try:
            self.conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.conn.close()
        except OSError:
            pass

    def reader_loop(self) -> None:
        buffer = bytearray()
        try:
            while not self.closed.is_set():
                data = self.conn.recv(65536)
                if not data:
                    print(f"[python] Client disconnected: {self.address}")
                    break
                buffer.extend(data)

                while len(buffer) >= HEADER_SIZE:
                    frame_type, payload_size = HEADER_STRUCT.unpack(buffer[:HEADER_SIZE])
                    if payload_size > MAX_FRAME_PAYLOAD_BYTES:
                        print(
                            f"[python] Invalid payload size from {self.address}: "
                            f"{payload_size} > {MAX_FRAME_PAYLOAD_BYTES}"
                        )
                        self.send_event({"type": "error", "message": "Invalid frame size"})
                        return

                    frame_end = HEADER_SIZE + payload_size
                    if len(buffer) < frame_end:
                        break

                    payload = bytes(buffer[HEADER_SIZE:frame_end])
                    del buffer[:frame_end]
                    try:
                        self.handle_frame(frame_type, payload)
                    except Exception as exc:
                        print(f"[python] Frame handling error {self.address}: {exc}")
                        traceback.print_exc()
                        self.send_event({"type": "error", "message": "Malformed frame"})
        except (ConnectionResetError, BrokenPipeError, OSError) as exc:
            if not self.closed.is_set():
                print(f"[python] Socket read error from {self.address}: {exc}")
                traceback.print_exc()
        except Exception as exc:
            print(f"[python] Reader loop error {self.address}: {exc}")
            traceback.print_exc()

    def handle_frame(self, frame_type: int, payload: bytes) -> None:
        if frame_type == FRAME_CONTROL:
            self.handle_control(payload)
            return

        if frame_type == FRAME_AUDIO:
            self.handle_audio(payload)

    def handle_control(self, payload: bytes) -> None:
        try:
            message = json.loads(payload.decode("utf-8"))
        except Exception as exc:
            print(f"[python] Control payload error from {self.address}: {exc}")
            traceback.print_exc()
            self.send_event({"type": "error", "message": "Invalid control payload"})
            return

        action = str(message.get("action", "")).lower().strip()

        if action == "start":
            self.start_recording()
            self.send_event({"type": "status", "state": "listening"})
            return

        if action == "stop":
            self.stop_recording()
            return

        if action == "close":
            self.close()
            return

        self.send_event({"type": "error", "message": f"Unknown action: {action}"})

    def handle_audio(self, payload: bytes) -> None:
        with self.state_lock:
            if not self.is_recording:
                return

        usable_bytes = len(payload) - (len(payload) % 4)
        if usable_bytes <= 0:
            return

        float_audio = np.frombuffer(payload[:usable_bytes], dtype="<f4")
        if float_audio.size == 0:
            return

        clipped_audio = np.clip(float_audio, -1.0, 1.0)
        pcm_int16 = (clipped_audio * 32767.0).astype(np.int16)
        pcm_bytes = pcm_int16.tobytes()
        now = time.monotonic()
        rms = float(np.sqrt(np.mean(clipped_audio * clipped_audio)))

        with self.state_lock:
            self.last_audio_timestamp = now
            if rms >= SPEECH_RMS_THRESHOLD:
                self.last_speech_timestamp = now
                self.pause_commit_done = False

            self.audio_buffer_bytes.extend(pcm_bytes)
            overflow = len(self.audio_buffer_bytes) - MAX_AUDIO_BUFFER_BYTES
            if overflow > 0:
                trim_bytes = overflow + (overflow % 2)
                del self.audio_buffer_bytes[:trim_bytes]

        try:
            self.stream_queue.put_nowait(pcm_bytes)
        except queue.Full:
            # Keep latency low by dropping oldest queued frames.
            try:
                _ = self.stream_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.stream_queue.put_nowait(pcm_bytes)
            except queue.Full:
                pass

    def start_recording(self) -> None:
        if VOSK_MODEL_INSTANCE is None:
            self.send_event(
                {
                    "type": "error",
                    "message": "Vosk model is unavailable. Check Python startup logs.",
                }
            )
            return

        now = time.monotonic()
        with self.state_lock:
            self.clear_stream_queue()
            self.audio_buffer_bytes.clear()
            self.stable_text = ""
            self.partial_text = ""
            self.last_emitted_stable = ""
            self.last_emitted_partial = ""
            self.last_audio_timestamp = now
            self.last_speech_timestamp = now
            self.last_partial_change_timestamp = now
            self.empty_partial_streak = 0
            self.pause_commit_done = False
            self.final_requested.clear()
            self.stream_flushed.clear()
            self.is_recording = True

            self.recognizer = KaldiRecognizer(VOSK_MODEL_INSTANCE, SAMPLE_RATE)
            self.recognizer.SetWords(True)
            try:
                self.recognizer.SetPartialWords(True)
            except AttributeError:
                pass

    def stop_recording(self) -> None:
        with self.state_lock:
            if not self.is_recording:
                return
            self.is_recording = False

        self.enqueue_stop_token()
        self.final_requested.set()

    def enqueue_stop_token(self) -> None:
        try:
            self.stream_queue.put_nowait(STOP_TOKEN)
        except queue.Full:
            try:
                _ = self.stream_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.stream_queue.put_nowait(STOP_TOKEN)
            except queue.Full:
                pass

    def clear_stream_queue(self) -> None:
        while True:
            try:
                _ = self.stream_queue.get_nowait()
            except queue.Empty:
                break

    def append_stable_text(self, text: str) -> None:
        if not text:
            return
        with self.state_lock:
            self.stable_text = normalize_spaces(f"{self.stable_text} {text}")

    def set_partial_text(self, text: str, now: Optional[float] = None) -> bool:
        changed = False
        normalized = normalize_spaces(text)
        now_ts = now if now is not None else time.monotonic()

        with self.state_lock:
            previous = self.partial_text
            if previous != normalized:
                self.partial_text = normalized
                self.last_partial_change_timestamp = now_ts
                self.pause_commit_done = False
                changed = True

            if self.partial_text:
                self.empty_partial_streak = 0
            else:
                self.empty_partial_streak += 1

        return changed

    def commit_partial_to_stable(self, force: bool = False) -> bool:
        with self.state_lock:
            partial = normalize_spaces(self.partial_text)
            stable = self.stable_text
            if not partial:
                return False

            words = count_words(partial)
            if not force and stable and words < MIN_SEGMENT_WORDS:
                return False

            self.stable_text = normalize_spaces(f"{stable} {partial}")
            self.partial_text = ""
            self.last_partial_change_timestamp = time.monotonic()
            self.empty_partial_streak = 0
            self.pause_commit_done = True
            return True

    def maybe_commit_pause_segment(self, now: float) -> None:
        with self.state_lock:
            if not self.is_recording or not self.partial_text or self.pause_commit_done:
                return

            silence_for_ms = max(0.0, (now - self.last_speech_timestamp) * 1000.0)
            partial_idle_ms = max(0.0, (now - self.last_partial_change_timestamp) * 1000.0)
            empty_streak = self.empty_partial_streak

        if partial_idle_ms < SILENCE_COMMIT_DELAY_MS:
            return

        quiet_enough = silence_for_ms >= SILENCE_THRESHOLD_MS
        quiet_with_empty_hint = (
            empty_streak >= EMPTY_PARTIAL_STREAK_COMMIT
            and silence_for_ms >= max(300.0, SILENCE_THRESHOLD_MS * 0.6)
        )

        if not quiet_enough and not quiet_with_empty_hint:
            return

        if self.commit_partial_to_stable(force=False):
            self.emit_partial_state(force=True)

    def should_skip_partial_emit(self, stable: str, partial: str) -> bool:
        if stable == self.last_emitted_stable and partial == self.last_emitted_partial:
            return True

        # Ignore tiny backward rewinds from Vosk partial jitter.
        if stable == self.last_emitted_stable and self.last_emitted_partial and partial:
            if self.last_emitted_partial.startswith(partial):
                rewind = len(self.last_emitted_partial) - len(partial)
                if 0 < rewind <= 2:
                    return True

        return False

    def emit_partial_state(self, force: bool = False) -> None:
        with self.state_lock:
            stable = self.stable_text
            partial = self.partial_text

            if not force and not stable and not partial:
                return

            if not force and self.should_skip_partial_emit(stable, partial):
                return

            self.last_emitted_stable = stable
            self.last_emitted_partial = partial

        self.send_event({"type": "partial", "stable": stable, "partial": partial})

    def flush_vosk_final(self) -> None:
        with self.state_lock:
            recognizer = self.recognizer

        if recognizer is None:
            self.stream_flushed.set()
            return

        final_result = json.loads(recognizer.FinalResult())
        final_text = extract_confident_vosk_text(final_result)
        self.append_stable_text(final_text)
        self.set_partial_text("", now=time.monotonic())
        self.emit_partial_state(force=True)
        self.stream_flushed.set()

    def consume_audio(self) -> bytes:
        with self.state_lock:
            if not self.audio_buffer_bytes:
                return b""
            data = bytes(self.audio_buffer_bytes)
            self.audio_buffer_bytes.clear()
            return data

    def get_stable_text(self) -> str:
        with self.state_lock:
            return self.stable_text

    def reset_session_text(self) -> None:
        now = time.monotonic()
        with self.state_lock:
            self.stable_text = ""
            self.partial_text = ""
            self.last_emitted_stable = ""
            self.last_emitted_partial = ""
            self.last_audio_timestamp = now
            self.last_speech_timestamp = now
            self.last_partial_change_timestamp = now
            self.empty_partial_streak = 0
            self.pause_commit_done = False
            self.recognizer = None

    def streaming_loop(self) -> None:
        while not self.closed.is_set():
            try:
                item = self.stream_queue.get(timeout=0.1)
            except queue.Empty:
                self.maybe_commit_pause_segment(time.monotonic())
                continue

            if item is STOP_TOKEN:
                self.flush_vosk_final()
                continue

            if not isinstance(item, (bytes, bytearray)):
                continue

            with self.state_lock:
                recognizer = self.recognizer

            if recognizer is None:
                continue

            accepted = recognizer.AcceptWaveform(item)
            if accepted:
                result = json.loads(recognizer.Result())
                stable_piece = extract_confident_vosk_text(result)
                self.append_stable_text(stable_piece)
                self.set_partial_text("", now=time.monotonic())
                self.emit_partial_state(force=True)
                continue

            now = time.monotonic()
            partial_result = json.loads(recognizer.PartialResult())
            partial_text = extract_partial_text(partial_result)
            self.set_partial_text(partial_text, now=now)
            self.emit_partial_state()
            self.maybe_commit_pause_segment(now)

    def final_loop(self) -> None:
        while not self.closed.is_set():
            if not self.final_requested.wait(timeout=0.1):
                continue
            self.final_requested.clear()

            if self.closed.is_set():
                break

            self.stream_flushed.wait(timeout=1.0)
            self.stream_flushed.clear()

            audio_bytes = self.consume_audio()
            stable_text = self.get_stable_text()

            if not audio_bytes:
                self.send_event({"type": "final", "text": post_process_text(stable_text)})
                self.reset_session_text()
                continue

            int16_audio = np.frombuffer(audio_bytes, dtype=np.int16)
            if int16_audio.size == 0:
                self.send_event({"type": "final", "text": post_process_text(stable_text)})
                self.reset_session_text()
                continue

            audio_float = int16_audio.astype(np.float32) / 32768.0
            whisper_model = WHISPER_MODEL
            if whisper_model is None:
                self.send_event(
                    {
                        "type": "error",
                        "message": "Whisper model is unavailable. Returning stabilized transcript.",
                    }
                )
                self.send_event({"type": "final", "text": post_process_text(stable_text)})
                self.reset_session_text()
                continue

            try:
                with WHISPER_LOCK:
                    segments, _ = whisper_model.transcribe(
                        audio_float,
                        language="en",
                        beam_size=5,
                        vad_filter=True,
                        temperature=0.0,
                        condition_on_previous_text=False,
                    )
                final_text = " ".join(
                    segment.text.strip() for segment in segments if segment.text and segment.text.strip()
                )
                if not final_text:
                    final_text = stable_text
                final_text = post_process_text(final_text)
                self.send_event({"type": "final", "text": final_text})
            except Exception as exc:
                self.send_event({"type": "error", "message": f"Whisper failed: {exc}"})
            finally:
                self.reset_session_text()

    def send_event(self, event: Dict) -> None:
        payload = json.dumps(event, separators=(",", ":")).encode("utf-8")
        self.send_frame(FRAME_EVENT, payload)

    def send_frame(self, frame_type: int, payload: bytes) -> None:
        if self.closed.is_set():
            return

        frame_header = HEADER_STRUCT.pack(frame_type, len(payload))
        packet = frame_header + payload
        try:
            with self.send_lock:
                self.conn.sendall(packet)
        except (BrokenPipeError, OSError) as exc:
            if not self.closed.is_set():
                print(f"[python] Socket send error to {self.address}: {exc}")
            self.close()


def run_server() -> None:
    if STARTUP_ERROR:
        print(f"[python] Cannot start server: {STARTUP_ERROR}")
        return

    try:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((HOST, PORT))
        server_socket.listen()
    except Exception as exc:
        print(f"[python] Startup error: {exc}")
        traceback.print_exc()
        return

    print(f"[python] Listening on {HOST}:{PORT}")

    try:
        while True:
            try:
                conn, address = server_socket.accept()
            except OSError as exc:
                print(f"[python] Accept error: {exc}")
                traceback.print_exc()
                time.sleep(0.1)
                continue

            try:
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                print(f"[python] Connected from {address[0]}:{address[1]}")
                session = ClientSession(conn, address)
                threading.Thread(target=session.run, daemon=True).start()
            except Exception as exc:
                print(f"[python] Session setup error for {address}: {exc}")
                traceback.print_exc()
                try:
                    conn.close()
                except OSError:
                    pass
    except KeyboardInterrupt:
        print("\n[python] Shutting down.")
    finally:
        server_socket.close()


if __name__ == "__main__":
    run_server()

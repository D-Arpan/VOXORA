import json
import os
import queue
import re
import socket
import struct
import threading
import time
import traceback
from collections import deque
from difflib import SequenceMatcher
from typing import Deque
from typing import Dict, List, Optional, Set, Tuple

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


def parse_csv_env(name: str, default: List[str]) -> List[str]:
    raw = os.getenv(name)
    if raw is None:
        return list(default)
    values = [part.strip() for part in str(raw).split(",")]
    cleaned = [part for part in values if part]
    return cleaned if cleaned else list(default)


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
PARTIAL_HISTORY_SIZE = max(2, min(5, parse_int_env("PARTIAL_HISTORY_SIZE", 3)))
PARTIAL_MIN_CONFIDENCE = parse_float_env("PARTIAL_MIN_CONFIDENCE", VOSK_MIN_CONFIDENCE)
ENABLE_PAUSE_PUNCTUATION = parse_bool_env("ENABLE_PAUSE_PUNCTUATION", True)
COMMA_PAUSE_MS = max(300, parse_int_env("COMMA_PAUSE_MS", 700))
SENTENCE_PAUSE_MS = max(COMMA_PAUSE_MS, parse_int_env("SENTENCE_PAUSE_MS", 1300))

ADAPTIVE_FIRST_WORD_CONFIDENCE = 0.65
ADAPTIVE_REPEAT_WORD_CONFIDENCE = 0.5
ADAPTIVE_SPEECH_RATE_WINDOW_SECONDS = 4.0
ADAPTIVE_FAST_SPEECH_WPS = 2.8
ADAPTIVE_SLOW_SPEECH_WPS = 1.3
ADAPTIVE_EMIT_SPEAKING_MIN_MS = 80
ADAPTIVE_EMIT_SPEAKING_MAX_MS = 120
ADAPTIVE_EMIT_PAUSE_MIN_MS = 150
ADAPTIVE_EMIT_PAUSE_MAX_MS = 250
ADAPTIVE_PAUSE_START_MS = 350
ADAPTIVE_PAUSE_MAX_MS = 2200

DEFAULT_PHRASE_BIAS_TERMS = [
    "Arpan",
    "Voxora",
    "WebSocket",
    "Node.js",
    "Python",
    "Whisper",
    "Vosk",
]
PHRASE_BIAS_TERMS = parse_csv_env("PHRASE_BIAS_TERMS", DEFAULT_PHRASE_BIAS_TERMS)

CALIBRATION_WINDOW_SECONDS = parse_float_env("CALIBRATION_WINDOW_SECONDS", 12.0)
CALIBRATION_WINDOW_SECONDS = max(8.0, min(20.0, CALIBRATION_WINDOW_SECONDS))
LOW_CONFIDENCE_HOLD_MS = max(20, parse_int_env("LOW_CONFIDENCE_HOLD_MS", 80))

ENABLE_FILLER_FILTER = parse_bool_env("ENABLE_FILLER_FILTER", True)
ENABLE_PREDICTIVE_SMOOTHING = parse_bool_env("ENABLE_PREDICTIVE_SMOOTHING", True)
PERSONAL_DICT_PATH = resolve_path(parse_str_env("PERSONAL_DICT_PATH", "python/personal_dictionary.json"))
MAX_PERSONAL_CASE_MAP_ITEMS = 512
MAX_PERSONAL_PHRASE_MAP_ITEMS = 256
SESSION_METRICS_LOG_PATH = resolve_path(parse_str_env("SESSION_METRICS_LOG_PATH", "python/session_metrics.json"))
MAX_SESSION_METRICS_ITEMS = max(20, parse_int_env("MAX_SESSION_METRICS_ITEMS", 400))
ENABLE_SESSION_METRICS_LOG = parse_bool_env("ENABLE_SESSION_METRICS_LOG", True)

FEEDBACK_LOOP_INTERVAL_MS = max(400, parse_int_env("FEEDBACK_LOOP_INTERVAL_MS", 1200))
CORRECTION_FEEDBACK_HIGH_RATE = parse_float_env("CORRECTION_FEEDBACK_HIGH_RATE", 0.25)
LATENCY_FEEDBACK_HIGH_MS = max(80, parse_int_env("LATENCY_FEEDBACK_HIGH_MS", 165))
SILENCE_FEEDBACK_FREQUENT_RATE = parse_float_env("SILENCE_FEEDBACK_FREQUENT_RATE", 0.28)
CPU_PRESSURE_RATIO = parse_float_env("CPU_PRESSURE_RATIO", 0.9)
CPU_GUARD_EMIT_SHIFT_MS = max(8, parse_int_env("CPU_GUARD_EMIT_SHIFT_MS", 18))

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

DOMAIN_TERM_REPLACEMENTS = [
    (r"\barpan\b", "Arpan"),
    (r"\bvoxora\b", "Voxora"),
    (r"\bweb\s*socket\b", "WebSocket"),
    (r"\bnode\s*\.?\s*js\b", "Node.js"),
    (r"\bpython\b", "Python"),
    (r"\bwhisper\b", "Whisper"),
    (r"\bvosk\b", "Vosk"),
]

FILLER_SINGLE_WORDS: Set[str] = {"uh", "um", "erm", "hmm"}
FILLER_OPTIONAL_WORDS: Set[str] = {"like"}
FILLER_MULTI_PHRASES: List[Tuple[str, ...]] = [("you", "know")]
PREDICTIVE_CONTINUATION_MAP = {
    "build": "building",
    "record": "recording",
    "transcrib": "transcribing",
}
DOMAIN_CASE_MAP = {
    "arpan": "Arpan",
    "voxora": "Voxora",
    "websocket": "WebSocket",
    "node.js": "Node.js",
    "nodejs": "Node.js",
    "python": "Python",
    "whisper": "Whisper",
    "vosk": "Vosk",
}

PERSONAL_DICT_LOCK = threading.Lock()
PERSONAL_DICTIONARY: Dict[str, Dict[str, str]] = {
    "case_map": {},
    "phrase_map": {},
}
METRICS_LOG_LOCK = threading.Lock()

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


def words_from_text(text: str) -> List[str]:
    cleaned = normalize_spaces(text)
    if not cleaned:
        return []
    return cleaned.split(" ")


def longest_common_prefix_word_lists(word_lists: List[List[str]]) -> List[str]:
    if not word_lists:
        return []

    prefix = list(word_lists[0])
    for words in word_lists[1:]:
        max_len = min(len(prefix), len(words))
        index = 0
        while index < max_len and prefix[index] == words[index]:
            index += 1
        prefix = prefix[:index]
        if not prefix:
            break
    return prefix


def count_common_prefix_words(left: List[str], right: List[str]) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def merge_text_with_overlap(base_text: str, addition_text: str) -> str:
    base_words = words_from_text(base_text)
    addition_words = words_from_text(addition_text)

    if not addition_words:
        return normalize_spaces(base_text)
    if not base_words:
        return " ".join(addition_words)

    max_overlap = min(len(base_words), len(addition_words))
    overlap = 0
    for size in range(max_overlap, 0, -1):
        if base_words[-size:] == addition_words[:size]:
            overlap = size
            break

    merged = base_words + addition_words[overlap:]
    return " ".join(merged)


def word_diff_ratio(source_text: str, target_text: str) -> float:
    source_words = words_from_text(source_text.lower())
    target_words = words_from_text(target_text.lower())
    if not source_words and not target_words:
        return 0.0

    matcher = SequenceMatcher(a=source_words, b=target_words, autojunk=False)
    changed = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        changed += max(i2 - i1, j2 - j1)

    denominator = max(1, len(source_words), len(target_words))
    return clamp_float(changed / denominator, 0.0, 1.0)


def clamp_float(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def normalize_term_for_matching(value: str) -> str:
    lowered = value.lower()
    lowered = lowered.replace(".", " ")
    lowered = lowered.replace("-", " ")
    return normalize_spaces(lowered)


def build_phrase_bias_grammar(terms: List[str]) -> Optional[str]:
    entries: List[str] = []
    seen = set()

    for term in terms:
        normalized = normalize_spaces(term)
        if not normalized:
            continue

        variants = {
            normalized,
            normalized.lower(),
            normalize_term_for_matching(normalized),
        }

        for variant in variants:
            candidate = normalize_spaces(variant)
            if not candidate or candidate in seen:
                continue
            entries.append(candidate)
            seen.add(candidate)

    if "[unk]" not in seen:
        entries.append("[unk]")

    if not entries:
        return None

    return json.dumps(entries)


def load_personal_dictionary() -> None:
    if not PERSONAL_DICT_PATH:
        return

    if not os.path.isfile(PERSONAL_DICT_PATH):
        return

    try:
        with open(PERSONAL_DICT_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        print(f"[python] Failed to load personal dictionary: {exc}")
        return

    case_map = payload.get("case_map", {})
    phrase_map = payload.get("phrase_map", {})

    if not isinstance(case_map, dict) or not isinstance(phrase_map, dict):
        return

    with PERSONAL_DICT_LOCK:
        PERSONAL_DICTIONARY["case_map"] = {
            str(key).lower(): str(value) for key, value in case_map.items() if str(key).strip() and str(value).strip()
        }
        PERSONAL_DICTIONARY["phrase_map"] = {
            normalize_spaces(str(key).lower()): normalize_spaces(str(value))
            for key, value in phrase_map.items()
            if str(key).strip() and str(value).strip()
        }


def save_personal_dictionary() -> None:
    if not PERSONAL_DICT_PATH:
        return

    directory = os.path.dirname(PERSONAL_DICT_PATH)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with PERSONAL_DICT_LOCK:
        payload = {
            "version": 1,
            "case_map": dict(PERSONAL_DICTIONARY["case_map"]),
            "phrase_map": dict(PERSONAL_DICTIONARY["phrase_map"]),
        }

    tmp_path = f"{PERSONAL_DICT_PATH}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)
        os.replace(tmp_path, PERSONAL_DICT_PATH)
    except Exception as exc:
        print(f"[python] Failed to save personal dictionary: {exc}")
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass


def append_session_metrics(record: Dict) -> None:
    if not ENABLE_SESSION_METRICS_LOG or not SESSION_METRICS_LOG_PATH:
        return

    directory = os.path.dirname(SESSION_METRICS_LOG_PATH)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with METRICS_LOG_LOCK:
        payload: Dict[str, object] = {"version": 1, "sessions": []}
        try:
            if os.path.isfile(SESSION_METRICS_LOG_PATH):
                with open(SESSION_METRICS_LOG_PATH, "r", encoding="utf-8") as handle:
                    existing = json.load(handle)
                    if isinstance(existing, dict):
                        payload = existing
        except Exception as exc:
            print(f"[python] Failed to read session metrics log: {exc}")

        sessions_value = payload.get("sessions", [])
        sessions: List[Dict] = sessions_value if isinstance(sessions_value, list) else []
        sessions.append(record)
        if len(sessions) > MAX_SESSION_METRICS_ITEMS:
            sessions = sessions[-MAX_SESSION_METRICS_ITEMS:]
        payload["sessions"] = sessions

        tmp_path = f"{SESSION_METRICS_LOG_PATH}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=True, separators=(",", ":"), indent=2)
            os.replace(tmp_path, SESSION_METRICS_LOG_PATH)
        except Exception as exc:
            print(f"[python] Failed to write session metrics log: {exc}")
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass


PHRASE_BIAS_GRAMMAR = build_phrase_bias_grammar(PHRASE_BIAS_TERMS)
if PHRASE_BIAS_GRAMMAR:
    print(f"[python] Phrase bias terms enabled: {', '.join(PHRASE_BIAS_TERMS)}")

load_personal_dictionary()


def preserve_token_case(original: str, replacement: str) -> str:
    if not original:
        return replacement
    if original.isupper():
        return replacement.upper()
    if original[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


def apply_domain_term_casing(text: str) -> str:
    output = text
    for pattern, replacement in DOMAIN_TERM_REPLACEMENTS:
        output = re.sub(pattern, replacement, output, flags=re.IGNORECASE)
    return output


def apply_case_and_phrase_replacements(
    text: str,
    case_map: Optional[Dict[str, str]] = None,
    phrase_map: Optional[Dict[str, str]] = None,
) -> str:
    updated = normalize_spaces(text)
    if not updated:
        return ""

    if phrase_map:
        for source in sorted(phrase_map.keys(), key=len, reverse=True):
            replacement = normalize_spaces(str(phrase_map.get(source, "")))
            normalized_source = normalize_spaces(str(source).lower())
            if not normalized_source or not replacement:
                continue
            pattern = r"\b" + re.escape(normalized_source).replace(r"\ ", r"\s+") + r"\b"
            updated = re.sub(pattern, replacement, updated, flags=re.IGNORECASE)

    if case_map:
        def replace_token(match: re.Match[str]) -> str:
            token = match.group(0)
            lower = token.lower()
            replacement = case_map.get(lower)
            if replacement:
                return replacement
            domain_value = DOMAIN_CASE_MAP.get(lower)
            if domain_value:
                return domain_value
            if lower == "i":
                return "I"
            return token

        updated = re.sub(r"\b[A-Za-z0-9'.-]+\b", replace_token, updated)
    else:
        updated = re.sub(r"\bi\b", "I", updated)

    updated = apply_domain_term_casing(updated)
    return normalize_spaces(updated)


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
    cleaned = apply_domain_term_casing(cleaned)

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
        self.partial_confidences: List[float] = []
        self.last_emitted_stable = ""
        self.last_emitted_partial = ""
        self.last_emitted_partial_confidences: List[float] = []
        self.last_partial_emit_timestamp = 0.0
        self.last_audio_timestamp = 0.0
        self.last_speech_timestamp = 0.0
        self.last_partial_change_timestamp = 0.0
        self.last_audio_rms = 0.0
        self.empty_partial_streak = 0
        self.pause_commit_done = False
        self.partial_history: Deque[List[str]] = deque(maxlen=PARTIAL_HISTORY_SIZE)
        self.segment_confirmed_words: List[str] = []
        self.partial_prev_words: List[str] = []
        self.partial_prev_streaks: List[int] = []
        self.word_rate_events: Deque[Tuple[float, int]] = deque()
        self.speech_words_per_second = 0.0
        self.dynamic_required_confirmations = 2
        self.dynamic_emit_interval_ms = 100
        self.partial_prediction = ""
        self.low_conf_hold_since: Dict[Tuple[int, str], float] = {}
        self.hesitation_filter_triggered = False
        self.hesitation_filter_hits = 0
        self.prediction_hits = 0

        self.session_started_at = 0.0
        self.calibration_deadline = 0.0
        self.calibration_done = False
        self.calibration_total_confidence = 0.0
        self.calibration_confidence_samples = 0
        self.calibration_pause_count = 0
        self.calibration_partial_events = 0
        self.calibration_correction_events = 0
        self.calibration_confirmed_words = 0
        self.calibration_confirm_bias = 0
        self.calibration_emit_shift_ms = 0
        self.calibrated_first_confidence = ADAPTIVE_FIRST_WORD_CONFIDENCE
        self.calibrated_repeat_confidence = ADAPTIVE_REPEAT_WORD_CONFIDENCE
        self.feedback_last_update_at = 0.0
        self.feedback_confirm_bias = 0
        self.feedback_confidence_shift = 0.0
        self.feedback_emit_shift_ms = 0
        self.feedback_pause_shift_ms = 0
        self.feedback_partial_events = 0
        self.feedback_correction_events = 0
        self.feedback_silence_events = 0
        self.feedback_latency_sum_ms = 0.0
        self.feedback_latency_samples = 0
        self.feedback_correction_rate = 0.0
        self.feedback_avg_latency_ms = 0.0

        self.cpu_last_wall_ts = 0.0
        self.cpu_last_proc_ts = 0.0
        self.cpu_load_ratio = 0.0
        self.cpu_guard_active = False
        self.cpu_guard_emit_shift_ms = 0

        self.session_partial_emit_count = 0
        self.session_latency_sum_ms = 0.0
        self.session_latency_count = 0
        self.session_final_correction_rate = 0.0

        self.learned_case_map: Dict[str, str] = {}
        self.learned_phrase_map: Dict[str, str] = {}
        self.learned_phrase_counts: Dict[str, int] = {}
        self.personal_dict_dirty = False

        with PERSONAL_DICT_LOCK:
            self.learned_case_map.update(PERSONAL_DICTIONARY["case_map"])
            self.learned_phrase_map.update(PERSONAL_DICTIONARY["phrase_map"])

        for canonical in PHRASE_BIAS_TERMS:
            cleaned = normalize_spaces(canonical)
            if cleaned:
                self.learned_case_map.setdefault(cleaned.lower(), cleaned)
        self.learned_phrase_map.setdefault("web socket", "WebSocket")
        self.learned_phrase_map.setdefault("node js", "Node.js")
        self.learned_phrase_map.setdefault("node . js", "Node.js")

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
            self.last_audio_rms = rms
            self._update_cpu_guard_locked(now)
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
            self.partial_confidences = []
            self.last_emitted_stable = ""
            self.last_emitted_partial = ""
            self.last_emitted_partial_confidences = []
            self.last_partial_emit_timestamp = 0.0
            self.last_audio_timestamp = now
            self.last_speech_timestamp = now
            self.last_partial_change_timestamp = now
            self.last_audio_rms = 0.0
            self.empty_partial_streak = 0
            self.pause_commit_done = False
            self.clear_partial_stabilization_state()
            self.word_rate_events.clear()
            self.speech_words_per_second = 0.0
            self.dynamic_required_confirmations = 2
            self.dynamic_emit_interval_ms = ADAPTIVE_EMIT_SPEAKING_MAX_MS
            self.partial_prediction = ""
            self.hesitation_filter_triggered = False
            self.hesitation_filter_hits = 0
            self.prediction_hits = 0

            self.session_started_at = now
            self.calibration_deadline = now + CALIBRATION_WINDOW_SECONDS
            self.calibration_done = False
            self.calibration_total_confidence = 0.0
            self.calibration_confidence_samples = 0
            self.calibration_pause_count = 0
            self.calibration_partial_events = 0
            self.calibration_correction_events = 0
            self.calibration_confirmed_words = 0
            self.calibration_confirm_bias = 0
            self.calibration_emit_shift_ms = 0
            self.calibrated_first_confidence = ADAPTIVE_FIRST_WORD_CONFIDENCE
            self.calibrated_repeat_confidence = ADAPTIVE_REPEAT_WORD_CONFIDENCE
            self.feedback_last_update_at = now
            self.feedback_confirm_bias = 0
            self.feedback_confidence_shift = 0.0
            self.feedback_emit_shift_ms = 0
            self.feedback_pause_shift_ms = 0
            self.feedback_partial_events = 0
            self.feedback_correction_events = 0
            self.feedback_silence_events = 0
            self.feedback_latency_sum_ms = 0.0
            self.feedback_latency_samples = 0
            self.feedback_correction_rate = 0.0
            self.feedback_avg_latency_ms = 0.0

            self.cpu_last_wall_ts = now
            self.cpu_last_proc_ts = time.process_time()
            self.cpu_load_ratio = 0.0
            self.cpu_guard_active = False
            self.cpu_guard_emit_shift_ms = 0

            self.session_partial_emit_count = 0
            self.session_latency_sum_ms = 0.0
            self.session_latency_count = 0
            self.session_final_correction_rate = 0.0
            self.final_requested.clear()
            self.stream_flushed.clear()
            self.is_recording = True

            self.recognizer = KaldiRecognizer(VOSK_MODEL_INSTANCE, SAMPLE_RATE)
            self.recognizer.SetWords(True)
            session_grammar = self._build_session_phrase_grammar_locked()
            if session_grammar:
                try:
                    self.recognizer.SetGrammar(session_grammar)
                except AttributeError:
                    pass
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

    def clear_partial_stabilization_state(self) -> None:
        self.partial_history.clear()
        self.segment_confirmed_words = []
        self.partial_prev_words = []
        self.partial_prev_streaks = []
        self.partial_confidences = []
        self.low_conf_hold_since.clear()
        self.partial_prediction = ""

    def append_stable_text(self, text: str) -> None:
        if not text:
            return
        with self.state_lock:
            self.stable_text = merge_text_with_overlap(self.stable_text, text)

    def _update_speech_rate_locked(self, now: float) -> None:
        cutoff = now - ADAPTIVE_SPEECH_RATE_WINDOW_SECONDS
        while self.word_rate_events and self.word_rate_events[0][0] < cutoff:
            self.word_rate_events.popleft()

        if not self.word_rate_events:
            self.speech_words_per_second = 0.0
            return

        words = sum(word_count for _, word_count in self.word_rate_events)
        first_ts = self.word_rate_events[0][0]
        duration = max(0.6, now - first_ts)
        self.speech_words_per_second = words / duration

    def _register_confirmed_words_locked(self, count: int, now: float) -> None:
        if count <= 0:
            return
        self.word_rate_events.append((now, count))
        self._update_speech_rate_locked(now)

    def _compute_required_confirmations_locked(self, now: float) -> int:
        self._maybe_finalize_calibration_locked(now)
        self._update_speech_rate_locked(now)
        silence_ms = max(0.0, (now - self.last_speech_timestamp) * 1000.0)
        wps = self.speech_words_per_second

        if silence_ms >= 900:
            return 1
        required = 2
        if wps >= ADAPTIVE_FAST_SPEECH_WPS:
            required = 3
        elif wps <= ADAPTIVE_SLOW_SPEECH_WPS:
            required = 1

        required += self.calibration_confirm_bias
        required += self.feedback_confirm_bias
        return int(clamp_float(float(required), 1.0, 3.0))

    def _compute_emit_interval_locked(self, now: float) -> int:
        self._maybe_finalize_calibration_locked(now)
        silence_ms = max(0.0, (now - self.last_speech_timestamp) * 1000.0)
        base_interval = ADAPTIVE_EMIT_SPEAKING_MAX_MS
        if silence_ms < ADAPTIVE_PAUSE_START_MS:
            wps = self.speech_words_per_second
            if wps >= ADAPTIVE_FAST_SPEECH_WPS:
                base_interval = ADAPTIVE_EMIT_SPEAKING_MIN_MS
            elif wps <= ADAPTIVE_SLOW_SPEECH_WPS:
                base_interval = ADAPTIVE_EMIT_SPEAKING_MAX_MS
            else:
                span = ADAPTIVE_FAST_SPEECH_WPS - ADAPTIVE_SLOW_SPEECH_WPS
                ratio = (wps - ADAPTIVE_SLOW_SPEECH_WPS) / span if span > 0 else 0.5
                ratio = clamp_float(ratio, 0.0, 1.0)
                interpolated = ADAPTIVE_EMIT_SPEAKING_MAX_MS - ratio * (
                    ADAPTIVE_EMIT_SPEAKING_MAX_MS - ADAPTIVE_EMIT_SPEAKING_MIN_MS
                )
                base_interval = int(round(interpolated))
        else:
            pause_ratio = (silence_ms - ADAPTIVE_PAUSE_START_MS) / (ADAPTIVE_PAUSE_MAX_MS - ADAPTIVE_PAUSE_START_MS)
            pause_ratio = clamp_float(pause_ratio, 0.0, 1.0)
            interpolated = ADAPTIVE_EMIT_PAUSE_MIN_MS + pause_ratio * (
                ADAPTIVE_EMIT_PAUSE_MAX_MS - ADAPTIVE_EMIT_PAUSE_MIN_MS
            )
            base_interval = int(round(interpolated))

        interval = base_interval
        interval += self.calibration_emit_shift_ms
        interval += self.feedback_emit_shift_ms
        interval += self.cpu_guard_emit_shift_ms
        return int(clamp_float(float(interval), 70.0, 280.0))

    def _update_cpu_guard_locked(self, now: float) -> None:
        process_now = time.process_time()
        if self.cpu_last_wall_ts <= 0.0:
            self.cpu_last_wall_ts = now
            self.cpu_last_proc_ts = process_now
            return

        wall_delta = now - self.cpu_last_wall_ts
        if wall_delta < 0.35:
            return

        proc_delta = max(0.0, process_now - self.cpu_last_proc_ts)
        ratio = proc_delta / max(0.001, wall_delta)
        self.cpu_load_ratio = ratio
        self.cpu_guard_active = ratio >= CPU_PRESSURE_RATIO
        self.cpu_guard_emit_shift_ms = CPU_GUARD_EMIT_SHIFT_MS if self.cpu_guard_active else 0

        self.cpu_last_wall_ts = now
        self.cpu_last_proc_ts = process_now

    def _maybe_apply_feedback_loop_locked(self, now: float) -> None:
        if (now - self.feedback_last_update_at) * 1000.0 < FEEDBACK_LOOP_INTERVAL_MS:
            return

        partial_events = max(1, self.feedback_partial_events)
        correction_rate = self.feedback_correction_events / partial_events
        silence_rate = self.feedback_silence_events / partial_events
        avg_latency_ms = self.feedback_latency_sum_ms / max(1, self.feedback_latency_samples)

        confirm_bias = 0
        confidence_shift = 0.0
        emit_shift_ms = 0
        pause_shift_ms = 0

        if correction_rate > CORRECTION_FEEDBACK_HIGH_RATE:
            confirm_bias += 1
            confidence_shift += 0.03
        elif correction_rate < 0.11 and self.speech_words_per_second >= 2.2:
            confirm_bias -= 1

        if self.speech_words_per_second <= 1.1 and avg_latency_ms >= LATENCY_FEEDBACK_HIGH_MS:
            confirm_bias -= 1
            confidence_shift -= 0.02
            emit_shift_ms -= 10

        if silence_rate >= SILENCE_FEEDBACK_FREQUENT_RATE:
            pause_shift_ms -= 140
        elif silence_rate <= 0.08:
            pause_shift_ms += 40

        if self.cpu_guard_active:
            emit_shift_ms += CPU_GUARD_EMIT_SHIFT_MS
            confidence_shift += 0.01

        self.feedback_confirm_bias = int(clamp_float(float(confirm_bias), -1.0, 1.0))
        self.feedback_confidence_shift = clamp_float(confidence_shift, -0.06, 0.07)
        self.feedback_emit_shift_ms = int(clamp_float(float(emit_shift_ms), -20.0, 35.0))
        self.feedback_pause_shift_ms = int(clamp_float(float(pause_shift_ms), -220.0, 120.0))
        self.feedback_correction_rate = clamp_float(correction_rate, 0.0, 1.0)
        self.feedback_avg_latency_ms = max(0.0, avg_latency_ms)
        self.feedback_last_update_at = now

        # Keep a bounded rolling window without extra deque allocations.
        self.feedback_partial_events = int(self.feedback_partial_events * 0.55)
        self.feedback_correction_events = int(self.feedback_correction_events * 0.55)
        self.feedback_silence_events = int(self.feedback_silence_events * 0.55)
        self.feedback_latency_sum_ms *= 0.55
        self.feedback_latency_samples = int(self.feedback_latency_samples * 0.55)

    def _current_pause_thresholds_locked(self) -> Tuple[float, float]:
        comma_threshold = COMMA_PAUSE_MS + self.feedback_pause_shift_ms
        sentence_threshold = SENTENCE_PAUSE_MS + self.feedback_pause_shift_ms
        comma_threshold = clamp_float(float(comma_threshold), 280.0, 1900.0)
        sentence_threshold = clamp_float(float(sentence_threshold), comma_threshold + 260.0, 3200.0)
        return comma_threshold, sentence_threshold

    def _build_session_metrics_record_locked(self, final_text: str) -> Dict:
        final_words = count_words(final_text)
        session_duration_seconds = max(0.0, time.monotonic() - self.session_started_at) if self.session_started_at else 0.0
        avg_latency_ms = self.session_latency_sum_ms / max(1, self.session_latency_count)
        correction_rate = max(self.feedback_correction_rate, self.session_final_correction_rate)

        return {
            "ts": int(time.time()),
            "duration_s": round(session_duration_seconds, 2),
            "final_words": final_words,
            "avg_wps": round(self.speech_words_per_second, 3),
            "correction_rate": round(clamp_float(correction_rate, 0.0, 1.0), 4),
            "final_diff_rate": round(clamp_float(self.session_final_correction_rate, 0.0, 1.0), 4),
            "avg_latency_ms": round(max(0.0, avg_latency_ms), 2),
            "partial_emits": self.session_partial_emit_count,
            "prediction_hits": self.prediction_hits,
            "hesitation_filter_hits": self.hesitation_filter_hits,
            "cpu_load_ratio": round(max(0.0, self.cpu_load_ratio), 3),
            "cpu_guard_active": self.cpu_guard_active,
        }

    def _persist_session_metrics(self, final_text: str) -> None:
        with self.state_lock:
            if self.session_started_at <= 0.0:
                return
            record = self._build_session_metrics_record_locked(final_text)
        append_session_metrics(record)

    def _build_session_phrase_grammar_locked(self) -> Optional[str]:
        terms: List[str] = list(PHRASE_BIAS_TERMS)
        seen = {normalize_spaces(term).lower() for term in terms if normalize_spaces(term)}

        for candidate in list(self.learned_case_map.values()) + list(self.learned_phrase_map.values()):
            normalized = normalize_spaces(candidate)
            lowered = normalized.lower()
            if not normalized or lowered in seen:
                continue
            terms.append(normalized)
            seen.add(lowered)
            if len(terms) >= 192:
                break
        return build_phrase_bias_grammar(terms)

    def _maybe_finalize_calibration_locked(self, now: float) -> None:
        if self.calibration_done or self.session_started_at <= 0.0:
            return
        if now < self.calibration_deadline:
            return

        avg_confidence = (
            self.calibration_total_confidence / self.calibration_confidence_samples
            if self.calibration_confidence_samples > 0
            else PARTIAL_MIN_CONFIDENCE
        )
        correction_rate = self.calibration_correction_events / max(1, self.calibration_partial_events)
        pause_rate = self.calibration_pause_count / max(1, self.calibration_partial_events)

        confirm_bias = 0
        emit_shift_ms = 0
        if self.speech_words_per_second >= ADAPTIVE_FAST_SPEECH_WPS:
            confirm_bias += 1
            emit_shift_ms -= 15
        elif self.speech_words_per_second <= ADAPTIVE_SLOW_SPEECH_WPS:
            confirm_bias -= 1
            emit_shift_ms -= 8

        if correction_rate >= 0.22:
            confirm_bias += 1
        elif correction_rate <= 0.08:
            confirm_bias -= 1

        if pause_rate >= 0.24:
            confirm_bias -= 1

        first_threshold = ADAPTIVE_FIRST_WORD_CONFIDENCE
        if avg_confidence >= 0.84:
            first_threshold += 0.05
        elif avg_confidence >= 0.74:
            first_threshold += 0.02
        elif avg_confidence <= 0.52:
            first_threshold -= 0.08
        elif avg_confidence <= 0.62:
            first_threshold -= 0.04

        if correction_rate >= 0.26:
            first_threshold += 0.03
        elif correction_rate <= 0.07:
            first_threshold -= 0.02

        first_threshold = clamp_float(first_threshold, 0.52, 0.84)
        repeat_threshold = ADAPTIVE_REPEAT_WORD_CONFIDENCE + (first_threshold - ADAPTIVE_FIRST_WORD_CONFIDENCE) * 0.75
        repeat_threshold = clamp_float(repeat_threshold, 0.38, first_threshold - 0.08)

        self.calibration_confirm_bias = int(clamp_float(float(confirm_bias), -1.0, 1.0))
        self.calibration_emit_shift_ms = int(clamp_float(float(emit_shift_ms), -20.0, 20.0))
        self.calibrated_first_confidence = first_threshold
        self.calibrated_repeat_confidence = repeat_threshold
        self.calibration_done = True

    def _apply_learned_token_replacements_locked(
        self,
        words: List[str],
        confidences: List[float],
    ) -> Tuple[List[str], List[float]]:
        if not words:
            return [], []

        normalized_confidences = [clamp_float(conf, 0.0, 1.0) for conf in confidences]
        if len(normalized_confidences) < len(words):
            normalized_confidences.extend([PARTIAL_MIN_CONFIDENCE] * (len(words) - len(normalized_confidences)))

        phrase_entries: List[Tuple[List[str], List[str]]] = []
        for source, target in self.learned_phrase_map.items():
            source_words = words_from_text(source.lower())
            target_words = words_from_text(target)
            if len(source_words) < 2 or len(source_words) > 4:
                continue
            if not target_words:
                continue
            phrase_entries.append((source_words, target_words))
        phrase_entries.sort(key=lambda item: len(item[0]), reverse=True)

        merged_words: List[str] = []
        merged_confidences: List[float] = []
        lower_words = [word.lower() for word in words]
        index = 0
        while index < len(words):
            matched = False
            for source_words, target_words in phrase_entries:
                width = len(source_words)
                if index + width > len(words):
                    continue
                if lower_words[index : index + width] != source_words:
                    continue

                segment_confidence = max(normalized_confidences[index : index + width])
                merged_words.extend(target_words)
                merged_confidences.extend([segment_confidence] * len(target_words))
                index += width
                matched = True
                break

            if matched:
                continue

            merged_words.append(words[index])
            merged_confidences.append(normalized_confidences[index])
            index += 1

        for token_index, token in enumerate(merged_words):
            lowered = token.lower()
            replacement = self.learned_case_map.get(lowered)
            if not replacement:
                replacement = DOMAIN_CASE_MAP.get(lowered)
            if replacement:
                merged_words[token_index] = replacement
            elif lowered == "i":
                merged_words[token_index] = "I"

        return merged_words, merged_confidences

    def _filter_hesitations_locked(
        self,
        words: List[str],
        confidences: List[float],
    ) -> Tuple[List[str], List[float]]:
        if not ENABLE_FILLER_FILTER or not words:
            return words, confidences

        normalized_confidences = [clamp_float(conf, 0.0, 1.0) for conf in confidences]
        if len(normalized_confidences) < len(words):
            normalized_confidences.extend([PARTIAL_MIN_CONFIDENCE] * (len(words) - len(normalized_confidences)))

        lower_words = [word.lower() for word in words]
        if len(lower_words) == 1 and (
            lower_words[0] in FILLER_SINGLE_WORDS or lower_words[0] in FILLER_OPTIONAL_WORDS
        ):
            return [], []

        skip_indexes: Set[int] = set()

        for phrase in FILLER_MULTI_PHRASES:
            width = len(phrase)
            for index in range(0, max(0, len(lower_words) - width + 1)):
                if tuple(lower_words[index : index + width]) != phrase:
                    continue
                isolated = len(lower_words) <= 4 or index == 0 or (index + width) == len(lower_words)
                phrase_confidence = min(normalized_confidences[index : index + width])
                if isolated and phrase_confidence < 0.82:
                    for offset in range(width):
                        skip_indexes.add(index + offset)

        for index, token in enumerate(lower_words):
            if index in skip_indexes:
                continue
            confidence = normalized_confidences[index]
            at_edge = index == 0 or index == len(lower_words) - 1
            short_phrase = len(lower_words) <= 4

            if token in FILLER_SINGLE_WORDS:
                if short_phrase or (at_edge and confidence < 0.86):
                    skip_indexes.add(index)
            elif token in FILLER_OPTIONAL_WORDS:
                if at_edge and short_phrase and confidence < 0.74:
                    skip_indexes.add(index)

        filtered_words: List[str] = []
        filtered_confidences: List[float] = []
        for index, token in enumerate(words):
            if index in skip_indexes:
                continue
            filtered_words.append(token)
            filtered_confidences.append(normalized_confidences[index])

        return filtered_words, filtered_confidences

    def _apply_low_confidence_hold_locked(
        self,
        words: List[str],
        confidences: List[float],
        now: float,
        base_index: int,
    ) -> Tuple[List[str], List[float]]:
        if LOW_CONFIDENCE_HOLD_MS <= 0 or not words:
            return words, confidences

        low_confidence_threshold = max(PARTIAL_MIN_CONFIDENCE, self.calibrated_first_confidence - 0.02)
        visible_words: List[str] = []
        visible_confidences: List[float] = []
        active_keys: Set[Tuple[int, str]] = set()

        for offset, word in enumerate(words):
            confidence = confidences[offset] if offset < len(confidences) else PARTIAL_MIN_CONFIDENCE
            confidence = clamp_float(confidence, 0.0, 1.0)
            key = (base_index + offset, word.lower())

            if confidence >= low_confidence_threshold:
                if key in self.low_conf_hold_since:
                    del self.low_conf_hold_since[key]
                visible_words.append(word)
                visible_confidences.append(confidence)
                active_keys.add(key)
                continue

            seen_at = self.low_conf_hold_since.get(key)
            if seen_at is None:
                self.low_conf_hold_since[key] = now
                continue

            if (now - seen_at) * 1000.0 < LOW_CONFIDENCE_HOLD_MS:
                continue

            visible_words.append(word)
            visible_confidences.append(confidence)
            active_keys.add(key)

        stale_cutoff = now - max(0.35, (LOW_CONFIDENCE_HOLD_MS / 1000.0) * 2.0)
        for key in list(self.low_conf_hold_since.keys()):
            if key in active_keys:
                continue
            if self.low_conf_hold_since[key] <= stale_cutoff:
                del self.low_conf_hold_since[key]

        return visible_words, visible_confidences

    def _predict_partial_extension_locked(self, words: List[str], confidences: List[float]) -> str:
        if self.cpu_guard_active:
            return ""
        if not ENABLE_PREDICTIVE_SMOOTHING or not words or not confidences:
            return ""

        last_word = words[-1].lower()
        last_confidence = clamp_float(confidences[-1], 0.0, 1.0)
        high_conf_threshold = max(0.74, self.calibrated_repeat_confidence + 0.14)
        if last_confidence < high_conf_threshold:
            return ""

        for stem, prediction in PREDICTIVE_CONTINUATION_MAP.items():
            predicted_lower = prediction.lower()
            if last_word == predicted_lower:
                return ""
            if not last_word.startswith(stem):
                continue
            if predicted_lower.startswith(last_word):
                self.prediction_hits += 1
                return prediction
        return ""

    def _apply_learned_replacements_locked(self, text: str) -> str:
        return apply_case_and_phrase_replacements(
            text,
            case_map=self.learned_case_map,
            phrase_map=self.learned_phrase_map,
        )

    def _remember_case_correction_locked(self, source_word: str, target_word: str) -> None:
        source = str(source_word).strip()
        target = str(target_word).strip()
        if not source or not target:
            return
        if source.lower() != target.lower():
            return
        if source == target:
            return
        key = source.lower()
        if self.learned_case_map.get(key) == target:
            return
        self.learned_case_map[key] = target
        self.personal_dict_dirty = True

    def _remember_phrase_correction_locked(self, source_words: List[str], target_words: List[str]) -> None:
        source_phrase = normalize_spaces(" ".join(source_words).lower())
        target_phrase = normalize_spaces(" ".join(target_words))
        if not source_phrase or not target_phrase:
            return
        if source_phrase == target_phrase.lower():
            return
        if len(source_words) > 4 or len(target_words) > 4:
            return

        self.learned_phrase_counts[source_phrase] = self.learned_phrase_counts.get(source_phrase, 0) + 1

        if self.learned_phrase_map.get(source_phrase) == target_phrase:
            return

        self.learned_phrase_map[source_phrase] = target_phrase
        if len(target_words) == 1:
            self.learned_case_map[target_words[0].lower()] = target_words[0]
        self.personal_dict_dirty = True

    def _learn_from_final(self, live_text: str, final_text: str) -> None:
        live_words = words_from_text(live_text)
        final_words = words_from_text(final_text)
        if not live_words or not final_words:
            return

        final_diff_rate = word_diff_ratio(live_text, final_text)

        matcher = SequenceMatcher(
            a=[word.lower() for word in live_words],
            b=[word.lower() for word in final_words],
            autojunk=False,
        )

        with self.state_lock:
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                source_slice = live_words[i1:i2]
                target_slice = final_words[j1:j2]

                if tag == "equal":
                    for source_word, target_word in zip(source_slice, target_slice):
                        self._remember_case_correction_locked(source_word, target_word)
                    continue

                if tag == "replace" and source_slice and target_slice:
                    if len(source_slice) == len(target_slice):
                        for source_word, target_word in zip(source_slice, target_slice):
                            self._remember_case_correction_locked(source_word, target_word)
                    self._remember_phrase_correction_locked(source_slice, target_slice)
                    continue

                if tag == "insert" and target_slice and len(target_slice) == 1:
                    self._remember_case_correction_locked(target_slice[0], target_slice[0])

            while len(self.learned_case_map) > MAX_PERSONAL_CASE_MAP_ITEMS:
                self.learned_case_map.pop(next(iter(self.learned_case_map)))

            while len(self.learned_phrase_map) > MAX_PERSONAL_PHRASE_MAP_ITEMS:
                removed_key = next(iter(self.learned_phrase_map))
                self.learned_phrase_map.pop(removed_key)
                self.learned_phrase_counts.pop(removed_key, None)

            self.session_final_correction_rate = clamp_float(final_diff_rate, 0.0, 1.0)
            self.feedback_correction_rate = max(self.feedback_correction_rate, self.session_final_correction_rate)

    def _persist_personal_dictionary_if_needed(self) -> None:
        with self.state_lock:
            if not self.personal_dict_dirty:
                return
            case_payload = dict(self.learned_case_map)
            phrase_payload = dict(self.learned_phrase_map)
            self.personal_dict_dirty = False

        with PERSONAL_DICT_LOCK:
            PERSONAL_DICTIONARY["case_map"] = case_payload
            PERSONAL_DICTIONARY["phrase_map"] = phrase_payload
        save_personal_dictionary()

    def extract_adaptive_partial_tokens(self, result_json: Dict) -> List[Tuple[str, float]]:
        now = time.monotonic()
        words = result_json.get("partial_result")
        if isinstance(words, list) and words:
            with self.state_lock:
                self._update_cpu_guard_locked(now)
                self._maybe_apply_feedback_loop_locked(now)
                prev_words = list(self.partial_prev_words)
                prev_streaks = list(self.partial_prev_streaks)
                extra_conf_shift = self.feedback_confidence_shift + (0.01 if self.cpu_guard_active else 0.0)
                first_threshold = clamp_float(self.calibrated_first_confidence + extra_conf_shift, 0.48, 0.9)
                repeat_threshold = clamp_float(self.calibrated_repeat_confidence + extra_conf_shift, 0.36, 0.82)

            next_words: List[str] = []
            next_streaks: List[int] = []
            output: List[Tuple[str, float]] = []
            confidence_samples: List[float] = []
            has_correction = False

            for index, word_info in enumerate(words):
                word = str(word_info.get("word", "")).strip()
                if not word:
                    continue

                conf = clamp_float(float(word_info.get("conf", 0.0)), 0.0, 1.0)
                streak = 1
                if index < len(prev_words) and prev_words[index] == word:
                    prev_streak = prev_streaks[index] if index < len(prev_streaks) else 1
                    streak = prev_streak + 1
                elif index < len(prev_words):
                    has_correction = True

                threshold = repeat_threshold if streak >= 2 else first_threshold

                next_words.append(word)
                next_streaks.append(streak)
                confidence_samples.append(conf)

                # Keep slightly-below-threshold candidates for short holding, but
                # stop on clearly unreliable tails.
                if conf + 0.08 < threshold:
                    break

                output.append((word, conf))

            with self.state_lock:
                self.partial_prev_words = next_words
                self.partial_prev_streaks = next_streaks
                self.feedback_partial_events += 1
                if has_correction:
                    self.feedback_correction_events += 1
                if self.session_started_at > 0.0 and not self.calibration_done:
                    self.calibration_partial_events += 1
                    if has_correction:
                        self.calibration_correction_events += 1
                    self.calibration_total_confidence += sum(confidence_samples)
                    self.calibration_confidence_samples += len(confidence_samples)
                    self._maybe_finalize_calibration_locked(now)
                self._maybe_apply_feedback_loop_locked(now)

            return output

        fallback_raw = str(result_json.get("partial", ""))
        fallback_words = words_from_text(fallback_raw.strip())
        if fallback_words and not fallback_raw.endswith((" ", ".", ",", "!", "?", ";", ":")):
            fallback_words = fallback_words[:-1]
        with self.state_lock:
            self._update_cpu_guard_locked(now)
            self.feedback_partial_events += 1
            if self.session_started_at > 0.0 and not self.calibration_done:
                self.calibration_partial_events += 1
                self.calibration_total_confidence += len(fallback_words) * PARTIAL_MIN_CONFIDENCE
                self.calibration_confidence_samples += len(fallback_words)
                self._maybe_finalize_calibration_locked(now)
            self._maybe_apply_feedback_loop_locked(now)
        return [(word, PARTIAL_MIN_CONFIDENCE) for word in fallback_words]

    def stabilize_partial_words(self, tokens: List[Tuple[str, float]], now: float) -> None:
        words = [word for word, _ in tokens]
        confidences = [clamp_float(conf, 0.0, 1.0) for _, conf in tokens]
        unstable_text = ""
        unstable_confidences: List[float] = []

        with self.state_lock:
            self._maybe_finalize_calibration_locked(now)

            before_filter_count = len(words)
            words, confidences = self._filter_hesitations_locked(words, confidences)
            self.hesitation_filter_triggered = len(words) < before_filter_count
            if self.hesitation_filter_triggered:
                self.hesitation_filter_hits += 1
            words, confidences = self._apply_learned_token_replacements_locked(words, confidences)

            self.partial_history.append(words)
            history = list(self.partial_history)

            required_confirmations = self._compute_required_confirmations_locked(now)
            required_confirmations = max(1, min(required_confirmations, len(history)))
            self.dynamic_required_confirmations = required_confirmations

            consensus_words: List[str] = []
            if required_confirmations == 1:
                consensus_words = list(words)
            elif len(history) >= required_confirmations:
                consensus_words = longest_common_prefix_word_lists(history[-required_confirmations:])

            confirmed_words = self.segment_confirmed_words
            new_confirmed_words: List[str] = []
            if len(consensus_words) > len(confirmed_words) and consensus_words[: len(confirmed_words)] == confirmed_words:
                new_confirmed_words = consensus_words[len(confirmed_words) :]

            if new_confirmed_words:
                self.stable_text = merge_text_with_overlap(self.stable_text, " ".join(new_confirmed_words))
                self.segment_confirmed_words = confirmed_words + new_confirmed_words
                confirmed_words = self.segment_confirmed_words
                self._register_confirmed_words_locked(len(new_confirmed_words), now)

            unstable_start_idx = 0
            if confirmed_words and words[: len(confirmed_words)] == confirmed_words:
                unstable_start_idx = len(confirmed_words)
                unstable_words = words[unstable_start_idx:]
            else:
                shared_prefix_len = count_common_prefix_words(words, confirmed_words)
                unstable_start_idx = shared_prefix_len
                unstable_words = words[unstable_start_idx:]

            unstable_confidences = confidences[unstable_start_idx:]
            unstable_words, unstable_confidences = self._apply_low_confidence_hold_locked(
                unstable_words,
                unstable_confidences,
                now,
                unstable_start_idx,
            )
            self.partial_prediction = self._predict_partial_extension_locked(unstable_words, unstable_confidences)
            unstable_text = " ".join(unstable_words)
            self._maybe_apply_feedback_loop_locked(now)

        self.set_partial_text(unstable_text, now=now, confidences=unstable_confidences)

    def set_partial_text(
        self,
        text: str,
        now: Optional[float] = None,
        confidences: Optional[List[float]] = None,
    ) -> bool:
        changed = False
        normalized = normalize_spaces(text)
        now_ts = now if now is not None else time.monotonic()
        confidence_values = [clamp_float(float(value), 0.0, 1.0) for value in (confidences or [])]

        with self.state_lock:
            previous = self.partial_text
            confidences_changed = self.partial_confidences != confidence_values

            if previous != normalized or confidences_changed:
                self.partial_text = normalized
                self.partial_confidences = confidence_values
                self.last_partial_change_timestamp = now_ts
                self.pause_commit_done = False
                changed = True
            elif normalized:
                self.partial_confidences = confidence_values

            if self.partial_text:
                self.empty_partial_streak = 0
            else:
                self.partial_confidences = []
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

            self.stable_text = merge_text_with_overlap(stable, partial)
            self.partial_text = ""
            self.partial_confidences = []
            self.last_partial_change_timestamp = time.monotonic()
            self.empty_partial_streak = 0
            self.pause_commit_done = True
            self._register_confirmed_words_locked(words, time.monotonic())
            self.clear_partial_stabilization_state()
            return True

    def maybe_append_pause_punctuation(self, silence_for_ms: float) -> None:
        if not ENABLE_PAUSE_PUNCTUATION:
            return

        punctuation = ""
        with self.state_lock:
            comma_threshold, sentence_threshold = self._current_pause_thresholds_locked()

        if silence_for_ms >= sentence_threshold:
            punctuation = "."
        elif silence_for_ms >= comma_threshold:
            punctuation = ","

        if not punctuation:
            return

        with self.state_lock:
            stable = self.stable_text.rstrip()
            if not stable:
                return
            if stable[-1] in ",.!?":
                return
            self.stable_text = f"{stable}{punctuation}"

    def maybe_commit_pause_segment(self, now: float) -> None:
        with self.state_lock:
            if not self.is_recording or not self.partial_text or self.pause_commit_done:
                return

            silence_for_ms = max(0.0, (now - self.last_speech_timestamp) * 1000.0)
            partial_idle_ms = max(0.0, (now - self.last_partial_change_timestamp) * 1000.0)
            empty_streak = self.empty_partial_streak
            silence_threshold_ms = clamp_float(
                float(SILENCE_THRESHOLD_MS + int(self.feedback_pause_shift_ms * 0.5)),
                280.0,
                1600.0,
            )

        if partial_idle_ms < SILENCE_COMMIT_DELAY_MS:
            return

        quiet_enough = silence_for_ms >= silence_threshold_ms
        quiet_with_empty_hint = (
            empty_streak >= EMPTY_PARTIAL_STREAK_COMMIT
            and silence_for_ms >= max(300.0, silence_threshold_ms * 0.6)
        )

        if not quiet_enough and not quiet_with_empty_hint:
            return

        if self.commit_partial_to_stable(force=False):
            with self.state_lock:
                if self.session_started_at > 0.0 and not self.calibration_done:
                    self.calibration_pause_count += 1
            self.maybe_append_pause_punctuation(silence_for_ms)
            self.emit_partial_state(force=True)

    def should_skip_partial_emit(self, stable: str, partial: str, confidences: List[float]) -> bool:
        if (
            stable == self.last_emitted_stable
            and partial == self.last_emitted_partial
            and len(confidences) == len(self.last_emitted_partial_confidences)
        ):
            if not confidences:
                return True

            max_delta = 0.0
            for index, confidence in enumerate(confidences):
                delta = abs(confidence - self.last_emitted_partial_confidences[index])
                if delta > max_delta:
                    max_delta = delta
            if max_delta < 0.08:
                return True

        # Ignore tiny backward rewinds from Vosk partial jitter.
        if stable == self.last_emitted_stable and self.last_emitted_partial and partial:
            if self.last_emitted_partial.startswith(partial):
                rewind = len(self.last_emitted_partial) - len(partial)
                if 0 < rewind <= 2:
                    return True

        return False

    def emit_partial_state(self, force: bool = False) -> None:
        now = time.monotonic()
        with self.state_lock:
            self._maybe_finalize_calibration_locked(now)
            stable = self.stable_text
            partial = self.partial_text
            partial_confidences = list(self.partial_confidences)
            speech_wps = self.speech_words_per_second
            required_confirmations = self.dynamic_required_confirmations
            dynamic_emit_interval = self._compute_emit_interval_locked(now)
            self.dynamic_emit_interval_ms = dynamic_emit_interval
            audio_rms = self.last_audio_rms
            latency_ms = max(0.0, (now - self.last_audio_timestamp) * 1000.0)
            silence_ms = max(0.0, (now - self.last_speech_timestamp) * 1000.0)
            calibration_progress = 1.0
            if self.session_started_at > 0.0 and self.calibration_deadline > self.session_started_at:
                calibration_progress = clamp_float(
                    (now - self.session_started_at) / (self.calibration_deadline - self.session_started_at),
                    0.0,
                    1.0,
                )

            stable_display = self._apply_learned_replacements_locked(stable)
            partial_display = apply_case_and_phrase_replacements(
                partial,
                case_map=self.learned_case_map,
                phrase_map=None,
            )
            partial_prediction = self.partial_prediction
            thinking = self.is_recording and silence_ms >= max(350.0, SILENCE_THRESHOLD_MS * 0.55) and not partial_display
            calibrated = self.calibration_done
            confidence_baseline = self.calibrated_first_confidence
            hesitation_filtered = self.hesitation_filter_triggered
            correction_rate = self.feedback_correction_rate
            avg_latency_ms = self.feedback_avg_latency_ms
            cpu_load_ratio = self.cpu_load_ratio
            cpu_guard = self.cpu_guard_active

            if not force and not stable_display and not partial_display:
                return

            if (
                not force
                and self.last_partial_emit_timestamp > 0.0
                and (now - self.last_partial_emit_timestamp) * 1000.0 < dynamic_emit_interval
            ):
                return

            if not force and self.should_skip_partial_emit(stable_display, partial_display, partial_confidences):
                return

            self.feedback_latency_sum_ms += latency_ms
            self.feedback_latency_samples += 1
            self.session_latency_sum_ms += latency_ms
            self.session_latency_count += 1
            if silence_ms >= SILENCE_THRESHOLD_MS and not partial_display:
                self.feedback_silence_events += 1
            self.session_partial_emit_count += 1
            self._maybe_apply_feedback_loop_locked(now)

            self.last_emitted_stable = stable_display
            self.last_emitted_partial = partial_display
            self.last_emitted_partial_confidences = partial_confidences
            self.last_partial_emit_timestamp = now

        self.send_event(
            {
                "type": "partial",
                "text": normalize_spaces(f"{stable_display} {partial_display}"),
                "stable": stable_display,
                "partial": partial_display,
                "partial_confidences": [round(value, 3) for value in partial_confidences],
                "stability_level": required_confirmations,
                "speech_wps": round(speech_wps, 3),
                "emit_interval_ms": dynamic_emit_interval,
                "rms": round(audio_rms, 4),
                "silence_ms": int(round(silence_ms)),
                "thinking": thinking,
                "prediction": partial_prediction,
                "calibrated": calibrated,
                "calibration_progress": round(calibration_progress, 3),
                "confidence_baseline": round(confidence_baseline, 3),
                "hesitation_filtered": hesitation_filtered,
                "correction_rate": round(correction_rate, 3),
                "avg_latency_ms": round(avg_latency_ms, 1),
                "cpu_load_ratio": round(cpu_load_ratio, 3),
                "cpu_guard": cpu_guard,
            }
        )

    def flush_vosk_final(self) -> None:
        with self.state_lock:
            recognizer = self.recognizer

        if recognizer is None:
            self.stream_flushed.set()
            return

        final_result = json.loads(recognizer.FinalResult())
        final_text = extract_confident_vosk_text(final_result)
        with self.state_lock:
            final_text = self._apply_learned_replacements_locked(final_text)
        self.append_stable_text(final_text)
        with self.state_lock:
            self._register_confirmed_words_locked(count_words(final_text), time.monotonic())
            self.clear_partial_stabilization_state()
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
            self.partial_confidences = []
            self.last_emitted_stable = ""
            self.last_emitted_partial = ""
            self.last_emitted_partial_confidences = []
            self.last_partial_emit_timestamp = 0.0
            self.last_audio_timestamp = now
            self.last_speech_timestamp = now
            self.last_partial_change_timestamp = now
            self.last_audio_rms = 0.0
            self.empty_partial_streak = 0
            self.pause_commit_done = False
            self.clear_partial_stabilization_state()
            self.word_rate_events.clear()
            self.speech_words_per_second = 0.0
            self.dynamic_required_confirmations = 2
            self.dynamic_emit_interval_ms = ADAPTIVE_EMIT_SPEAKING_MAX_MS
            self.partial_prediction = ""
            self.hesitation_filter_triggered = False
            self.hesitation_filter_hits = 0
            self.prediction_hits = 0
            self.session_started_at = 0.0
            self.calibration_deadline = 0.0
            self.calibration_done = False
            self.calibration_total_confidence = 0.0
            self.calibration_confidence_samples = 0
            self.calibration_pause_count = 0
            self.calibration_partial_events = 0
            self.calibration_correction_events = 0
            self.calibration_confirmed_words = 0
            self.calibration_confirm_bias = 0
            self.calibration_emit_shift_ms = 0
            self.calibrated_first_confidence = ADAPTIVE_FIRST_WORD_CONFIDENCE
            self.calibrated_repeat_confidence = ADAPTIVE_REPEAT_WORD_CONFIDENCE
            self.feedback_last_update_at = 0.0
            self.feedback_confirm_bias = 0
            self.feedback_confidence_shift = 0.0
            self.feedback_emit_shift_ms = 0
            self.feedback_pause_shift_ms = 0
            self.feedback_partial_events = 0
            self.feedback_correction_events = 0
            self.feedback_silence_events = 0
            self.feedback_latency_sum_ms = 0.0
            self.feedback_latency_samples = 0
            self.feedback_correction_rate = 0.0
            self.feedback_avg_latency_ms = 0.0

            self.cpu_last_wall_ts = 0.0
            self.cpu_last_proc_ts = 0.0
            self.cpu_load_ratio = 0.0
            self.cpu_guard_active = False
            self.cpu_guard_emit_shift_ms = 0

            self.session_partial_emit_count = 0
            self.session_latency_sum_ms = 0.0
            self.session_latency_count = 0
            self.session_final_correction_rate = 0.0
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
                with self.state_lock:
                    stable_piece = self._apply_learned_replacements_locked(stable_piece)
                self.append_stable_text(stable_piece)
                with self.state_lock:
                    self._register_confirmed_words_locked(count_words(stable_piece), time.monotonic())
                    self.clear_partial_stabilization_state()
                self.set_partial_text("", now=time.monotonic())
                self.emit_partial_state(force=True)
                continue

            now = time.monotonic()
            partial_result = json.loads(recognizer.PartialResult())
            partial_tokens = self.extract_adaptive_partial_tokens(partial_result)
            self.stabilize_partial_words(partial_tokens, now=now)
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
                with self.state_lock:
                    stable_display = self._apply_learned_replacements_locked(stable_text)
                final_fallback_text = post_process_text(stable_display)
                self._learn_from_final(stable_text, final_fallback_text)
                self.send_event({"type": "final", "text": final_fallback_text})
                self._persist_personal_dictionary_if_needed()
                self._persist_session_metrics(final_fallback_text)
                self.reset_session_text()
                continue

            int16_audio = np.frombuffer(audio_bytes, dtype=np.int16)
            if int16_audio.size == 0:
                with self.state_lock:
                    stable_display = self._apply_learned_replacements_locked(stable_text)
                final_fallback_text = post_process_text(stable_display)
                self._learn_from_final(stable_text, final_fallback_text)
                self.send_event({"type": "final", "text": final_fallback_text})
                self._persist_personal_dictionary_if_needed()
                self._persist_session_metrics(final_fallback_text)
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
                with self.state_lock:
                    stable_display = self._apply_learned_replacements_locked(stable_text)
                final_fallback_text = post_process_text(stable_display)
                self._learn_from_final(stable_text, final_fallback_text)
                self.send_event({"type": "final", "text": final_fallback_text})
                self._persist_personal_dictionary_if_needed()
                self._persist_session_metrics(final_fallback_text)
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
                with self.state_lock:
                    final_text = self._apply_learned_replacements_locked(final_text)
                final_text = post_process_text(final_text)
                self._learn_from_final(stable_text, final_text)
                self.send_event({"type": "final", "text": final_text})
                self._persist_personal_dictionary_if_needed()
                self._persist_session_metrics(final_text)
            except Exception as exc:
                self.send_event({"type": "error", "message": f"Whisper failed: {exc}"})
                self._persist_session_metrics(stable_text)
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

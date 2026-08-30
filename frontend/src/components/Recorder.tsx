"use client";

import { useCallback, useEffect, useMemo, useRef, useState, type MouseEvent } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Check, Copy, Sun, Moon, Trash2 } from "lucide-react";
import MouseGlitter from "./MouseGlitter";
import { RealtimeSocket, type ServerMessage } from "@/lib/websocket";

const DEFAULT_CHUNK_MS = 100;
const MIN_CHUNK_MS = 60;
const MAX_CHUNK_MS = 120;
const UI_BATCH_DELAY_MS = 100;
const FINAL_CORRECTION_HIGHLIGHT_MS = 1200;
const WORD_STAGGER_MS = 50;
const DEBUG_HESITATION_FLASH_MS = 1200;
const FILLER_WORDS = new Set(["uh", "um", "erm", "hmm", "like"]);

const CHUNK_MS = (() => {
    const parsed = Number(process.env.NEXT_PUBLIC_RECORDER_CHUNK_MS || DEFAULT_CHUNK_MS);
    return Number.isFinite(parsed)
        ? Math.min(MAX_CHUNK_MS, Math.max(MIN_CHUNK_MS, Math.floor(parsed)))
        : DEFAULT_CHUNK_MS;
})();

type PendingUiMessages = {
    partial?: Extract<ServerMessage, { type: "partial" }>;
    final?: string;
    status?: string;
    error?: string;
};

type Ripple = {
    id: number;
    x: number;
    y: number;
};

function pickMimeType(): string {
    if (typeof MediaRecorder === "undefined") return "audio/webm";
    return MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : "audio/webm";
}

function normalizeSpaces(text: string): string {
    return text.replace(/\s+/g, " ").trim();
}

function wordsFromText(text: string): string[] {
    const cleaned = normalizeSpaces(text);
    return cleaned ? cleaned.split(" ") : [];
}

function computeChangedWordIndexes(liveText: string, finalText: string): number[] {
    const liveWords = wordsFromText(liveText);
    const finalWords = wordsFromText(finalText);
    const changed: number[] = [];

    for (let index = 0; index < finalWords.length; index += 1) {
        if (liveWords[index] !== finalWords[index]) {
            changed.push(index);
        }
    }

    return changed;
}

function shouldHighlightCorrections(liveText: string, finalText: string, changedWordIndexes: number[]): boolean {
    if (changedWordIndexes.length === 0) {
        return false;
    }
    const liveWords = wordsFromText(liveText);
    const finalWords = wordsFromText(finalText);
    const denominator = Math.max(1, liveWords.length, finalWords.length);
    const correctionRatio = changedWordIndexes.length / denominator;
    const lengthDelta = Math.abs(liveWords.length - finalWords.length);
    return changedWordIndexes.length >= 3 || correctionRatio >= 0.22 || lengthDelta >= 2;
}

function toOpacity(confidence: number | undefined): number {
    if (confidence === undefined || Number.isNaN(confidence)) {
        return 0.55;
    }
    const clamped = Math.max(0, Math.min(1, confidence));
    return 0.3 + clamped * 0.7;
}

function wordDiffRatio(sourceText: string, targetText: string): number {
    const sourceWords = wordsFromText(sourceText.toLowerCase());
    const targetWords = wordsFromText(targetText.toLowerCase());
    if (!sourceWords.length && !targetWords.length) {
        return 0;
    }

    const maxLen = Math.max(sourceWords.length, targetWords.length, 1);
    let changed = 0;
    for (let index = 0; index < maxLen; index += 1) {
        if (sourceWords[index] !== targetWords[index]) {
            changed += 1;
        }
    }
    return Math.max(0, Math.min(1, changed / maxLen));
}

function smoothPartialTail(
    previousPartial: string,
    incomingPartial: string,
    incomingConfidences: number[]
): { text: string; confidences: number[]; rewriteDetected: boolean; hesitationLikely: boolean } {
    const previousWords = wordsFromText(previousPartial);
    const incomingWords = wordsFromText(incomingPartial);
    const confidences = [...incomingConfidences];

    if (incomingWords.length === 0) {
        const hesitationLikely = previousWords.some((word) => FILLER_WORDS.has(word.toLowerCase()));
        return { text: "", confidences: [], rewriteDetected: previousWords.length > 0, hesitationLikely };
    }

    let sharedPrefix = 0;
    while (
        sharedPrefix < previousWords.length &&
        sharedPrefix < incomingWords.length &&
        previousWords[sharedPrefix] === incomingWords[sharedPrefix]
    ) {
        sharedPrefix += 1;
    }
    const rewriteDetected = sharedPrefix < Math.min(previousWords.length, incomingWords.length);

    const nextWords = [...incomingWords];
    const lastIndex = nextWords.length - 1;
    const lastWord = nextWords[lastIndex] || "";
    const lastConfidence = confidences[lastIndex] ?? 0.5;
    const previousLastWord = previousWords.length === incomingWords.length ? previousWords[lastIndex] : "";

    if (
        previousLastWord &&
        lastWord &&
        previousLastWord.length >= 5 &&
        previousLastWord.startsWith(lastWord) &&
        previousLastWord.length - lastWord.length >= 2 &&
        lastConfidence < 0.72
    ) {
        nextWords[lastIndex] = previousLastWord;
    }

    if (nextWords.length > 1) {
        const tail = nextWords[nextWords.length - 1];
        const tailConfidence = confidences[nextWords.length - 1] ?? 0.5;
        if (tail.length <= 2 && tailConfidence < 0.62) {
            nextWords.pop();
            confidences.pop();
        }
    }

    const previousLower = previousWords.map((word) => word.toLowerCase());
    const nextLower = nextWords.map((word) => word.toLowerCase());
    const hadFiller = previousLower.some((word) => FILLER_WORDS.has(word));
    const hasFillerNow = nextLower.some((word) => FILLER_WORDS.has(word));
    const hesitationLikely = hadFiller && !hasFillerNow;

    const normalizedText = normalizeSpaces(nextWords.join(" "));
    return {
        text: normalizedText,
        confidences: confidences.slice(0, nextWords.length),
        rewriteDetected,
        hesitationLikely,
    };
}

function metricTone(
    value: number,
    warningThreshold: number,
    criticalThreshold: number,
    lowerIsBetter: boolean = false
): string {
    const warningHit = lowerIsBetter ? value >= warningThreshold : value <= warningThreshold;
    const criticalHit = lowerIsBetter ? value >= criticalThreshold : value <= criticalThreshold;
    if (criticalHit) {
        return "text-red-300";
    }
    if (warningHit) {
        return "text-amber-300";
    }
    return "text-emerald-300";
}

function parseProcessingDebugStatus(status: string): {
    queueSize?: number;
    processingDelayMs?: number;
    finalFlushDurationMs?: number;
    beamSize?: number;
    activeJobs?: number;
} | null {
    if (!status.startsWith("debug:")) {
        return null;
    }
    try {
        const payload = JSON.parse(status.slice(6)) as Record<string, unknown>;
        const queueSize = Number(payload.queue_size);
        const processingDelayMs = Number(payload.processing_delay_ms);
        const finalFlushDurationMs = Number(payload.final_flush_duration_ms);
        const beamSize = Number(payload.beam_size);
        const activeJobs = Number(payload.active_jobs);
        return {
            queueSize: Number.isFinite(queueSize) ? queueSize : undefined,
            processingDelayMs: Number.isFinite(processingDelayMs) ? processingDelayMs : undefined,
            finalFlushDurationMs: Number.isFinite(finalFlushDurationMs) ? finalFlushDurationMs : undefined,
            beamSize: Number.isFinite(beamSize) ? beamSize : undefined,
            activeJobs: Number.isFinite(activeJobs) ? activeJobs : undefined,
        };
    } catch {
        return null;
    }
}

export default function Recorder() {
    const [isDarkMode, setIsDarkMode] = useState(true);

    useEffect(() => {
        if (typeof document !== 'undefined') {
            if (isDarkMode) {
                document.documentElement.classList.add("dark");
            } else {
                document.documentElement.classList.remove("dark");
            }
        }
    }, [isDarkMode]);

    const socketRef = useRef<RealtimeSocket | null>(null);
    const recorderRef = useRef<MediaRecorder | null>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const isRecordingRef = useRef(false);

    const stableRef = useRef("");
    const partialRef = useRef("");
    const finalRef = useRef("");
    const pendingRef = useRef<PendingUiMessages>({});
    const flushTimerRef = useRef<number | null>(null);
    const correctionTimerRef = useRef<number | null>(null);
    const startedAtRef = useRef<number | null>(null);
    const recorderTimerRef = useRef<number | null>(null);
    const rippleIdRef = useRef(0);
    const energyMonitorRef = useRef<number | null>(null);
    const audioContextRef = useRef<AudioContext | null>(null);
    const analyserRef = useRef<AnalyserNode | null>(null);
    const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);
    const audioDataRef = useRef<Uint8Array | null>(null);
    const hesitationTimerRef = useRef<number | null>(null);
    const lastPartialArrivalRef = useRef(0);
    const partialEventCountRef = useRef(0);
    const partialRewriteCountRef = useRef(0);
    const finalLockedRef = useRef(false);
    const processingStartedAtRef = useRef<number | null>(null);

    const [isRecording, setIsRecording] = useState(false);
    const [stableText, setStableText] = useState("");
    const [partialText, setPartialText] = useState("");
    const [partialConfidences, setPartialConfidences] = useState<number[]>([]);
    const [finalText, setFinalText] = useState("");
    const [status, setStatus] = useState("disconnected");
    const [error, setError] = useState("");
    const [copied, setCopied] = useState(false);
    const [elapsedMs, setElapsedMs] = useState(0);
    const [finalRevision, setFinalRevision] = useState(0);
    const [correctedWordIndexes, setCorrectedWordIndexes] = useState<number[]>([]);
    const [orbEnergy, setOrbEnergy] = useState(0);
    const [ripples, setRipples] = useState<Ripple[]>([]);
    const [stabilityLevel, setStabilityLevel] = useState(2);
    const [speechWps, setSpeechWps] = useState(0);
    const [emitIntervalMs, setEmitIntervalMs] = useState(100);
    const [silenceMs, setSilenceMs] = useState(0);
    const [isThinking, setIsThinking] = useState(false);
    const [prediction, setPrediction] = useState("");
    const [isCalibrated, setIsCalibrated] = useState(false);
    const [calibrationProgress, setCalibrationProgress] = useState(0);
    const [confidenceBaseline, setConfidenceBaseline] = useState(0.65);
    const [backendRms, setBackendRms] = useState(0);
    const [correctionRate, setCorrectionRate] = useState(0);
    const [perceivedLatencyMs, setPerceivedLatencyMs] = useState(0);
    const [hesitationFilterTriggered, setHesitationFilterTriggered] = useState(false);
    const [debugVisible, setDebugVisible] = useState(false);
    const [whisperQueueSize, setWhisperQueueSize] = useState(0);
    const [whisperProcessingDelayMs, setWhisperProcessingDelayMs] = useState(0);
    const [finalFlushDurationMs, setFinalFlushDurationMs] = useState(0);
    const [whisperBeamSize, setWhisperBeamSize] = useState(2);
    const [whisperActiveJobs, setWhisperActiveJobs] = useState(0);
    const [processingProgress, setProcessingProgress] = useState(0);

    useEffect(() => {
        isRecordingRef.current = isRecording;
    }, [isRecording]);

    useEffect(() => {
        const onKeyDown = (event: KeyboardEvent) => {
            if (event.ctrlKey && event.shiftKey && event.key.toLowerCase() === "d") {
                event.preventDefault();
                setDebugVisible((previous) => !previous);
            }
        };
        window.addEventListener("keydown", onKeyDown);
        return () => window.removeEventListener("keydown", onKeyDown);
    }, []);

    useEffect(() => {
        if (status !== "processing") {
            processingStartedAtRef.current = null;
            return;
        }

        if (processingStartedAtRef.current === null) {
            processingStartedAtRef.current = Date.now();
        }

        const timer = window.setInterval(() => {
            const startedAt = processingStartedAtRef.current ?? Date.now();
            const elapsedMs = Date.now() - startedAt;
            const target = Math.min(94, 12 + elapsedMs / 42);
            setProcessingProgress((previous) => (previous < target ? previous * 0.72 + target * 0.28 : previous));
        }, 90);

        return () => window.clearInterval(timer);
    }, [status]);

    const teardownEnergyMonitor = useCallback(() => {
        if (energyMonitorRef.current !== null) {
            window.clearInterval(energyMonitorRef.current);
            energyMonitorRef.current = null;
        }

        try {
            sourceNodeRef.current?.disconnect();
        } catch {
            // Ignore teardown disconnect issues.
        }

        try {
            analyserRef.current?.disconnect();
        } catch {
            // Ignore teardown disconnect issues.
        }

        if (audioContextRef.current) {
            void audioContextRef.current.close();
        }

        sourceNodeRef.current = null;
        analyserRef.current = null;
        audioContextRef.current = null;
        audioDataRef.current = null;
        setOrbEnergy(0);
    }, []);

    const setupEnergyMonitor = useCallback(
        (stream: MediaStream) => {
            teardownEnergyMonitor();

            const AudioContextCtor = window.AudioContext || (window as Window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
            if (!AudioContextCtor) {
                return;
            }

            const context = new AudioContextCtor();
            const analyser = context.createAnalyser();
            analyser.fftSize = 256;
            analyser.smoothingTimeConstant = 0.75;

            const source = context.createMediaStreamSource(stream);
            source.connect(analyser);

            audioContextRef.current = context;
            analyserRef.current = analyser;
            sourceNodeRef.current = source;
            audioDataRef.current = new Uint8Array(analyser.fftSize);

            energyMonitorRef.current = window.setInterval(() => {
                const node = analyserRef.current;
                const data = audioDataRef.current;
                if (!node || !data) {
                    return;
                }

                node.getByteTimeDomainData(data as any);
                let sum = 0;
                for (let index = 0; index < data.length; index += 1) {
                    const normalized = (data[index] - 128) / 128;
                    sum += normalized * normalized;
                }

                const rms = Math.sqrt(sum / data.length);
                const scaled = Math.max(0, Math.min(1, rms * 7));
                setOrbEnergy((previous) => previous * 0.72 + scaled * 0.28);
            }, 60);
        },
        [teardownEnergyMonitor]
    );

    const resetElapsedTimer = useCallback(() => {
        if (recorderTimerRef.current !== null) {
            window.clearInterval(recorderTimerRef.current);
            recorderTimerRef.current = null;
        }
        startedAtRef.current = null;
        setElapsedMs(0);
    }, []);

    const startElapsedTimer = useCallback(() => {
        startedAtRef.current = Date.now();
        setElapsedMs(0);

        if (recorderTimerRef.current !== null) {
            window.clearInterval(recorderTimerRef.current);
        }

        recorderTimerRef.current = window.setInterval(() => {
            if (startedAtRef.current === null) {
                return;
            }
            setElapsedMs(Date.now() - startedAtRef.current);
        }, 100);
    }, []);

    const stopTracks = useCallback(() => {
        streamRef.current?.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
    }, []);

    const stopRecorderOnly = useCallback(() => {
        if (recorderRef.current?.state !== "inactive") {
            recorderRef.current?.stop();
        }
        recorderRef.current = null;
    }, []);

    const clearUiTimers = useCallback(() => {
        if (flushTimerRef.current !== null) {
            window.clearTimeout(flushTimerRef.current);
            flushTimerRef.current = null;
        }
        if (correctionTimerRef.current !== null) {
            window.clearTimeout(correctionTimerRef.current);
            correctionTimerRef.current = null;
        }
        if (hesitationTimerRef.current !== null) {
            window.clearTimeout(hesitationTimerRef.current);
            hesitationTimerRef.current = null;
        }
    }, []);

    const clearTranscript = useCallback(() => {
        stableRef.current = "";
        partialRef.current = "";
        finalRef.current = "";
        pendingRef.current = {};
        setStableText("");
        setPartialText("");
        setPartialConfidences([]);
        setFinalText("");
        setCorrectedWordIndexes([]);
        setStabilityLevel(2);
        setSpeechWps(0);
        setEmitIntervalMs(100);
        setSilenceMs(0);
        setIsThinking(false);
        setPrediction("");
        setIsCalibrated(false);
        setCalibrationProgress(0);
        setConfidenceBaseline(0.65);
        setBackendRms(0);
        setCorrectionRate(0);
        setPerceivedLatencyMs(0);
        setHesitationFilterTriggered(false);
        setWhisperQueueSize(0);
        setWhisperProcessingDelayMs(0);
        setFinalFlushDurationMs(0);
        setWhisperBeamSize(2);
        setWhisperActiveJobs(0);
        setProcessingProgress(0);
        lastPartialArrivalRef.current = 0;
        partialEventCountRef.current = 0;
        partialRewriteCountRef.current = 0;
        finalLockedRef.current = false;
        processingStartedAtRef.current = null;
    }, []);

    const addButtonRipple = useCallback((event: MouseEvent<HTMLButtonElement>) => {
        const bounds = event.currentTarget.getBoundingClientRect();
        const ripple: Ripple = {
            id: rippleIdRef.current++,
            x: event.clientX - bounds.left,
            y: event.clientY - bounds.top,
        };
        setRipples((previous) => [...previous, ripple]);
        window.setTimeout(() => {
            setRipples((previous) => previous.filter((item) => item.id !== ripple.id));
        }, 550);
    }, []);

    const applyPendingMessages = useCallback(() => {
        flushTimerRef.current = null;
        const pending = pendingRef.current;
        pendingRef.current = {};

        if (pending.error) {
            setError(pending.error);
        }
        if (pending.status) {
            setStatus(pending.status);
        }

        if (pending.partial && !pending.final) {
            if (finalLockedRef.current && !isRecordingRef.current) {
                return;
            }
            const previousPartial = partialRef.current;
            const smoothed = smoothPartialTail(
                previousPartial,
                pending.partial.partial,
                pending.partial.partialConfidences || []
            );

            if (stableRef.current !== pending.partial.stable) {
                stableRef.current = pending.partial.stable;
                setStableText(pending.partial.stable);
            }
            const rawPartial = pending.partial.partial || "";
            if (partialRef.current !== rawPartial) {
                partialRef.current = rawPartial;
                setPartialText(rawPartial);
            }
            setPartialConfidences(pending.partial.partialConfidences || []);

            partialEventCountRef.current += 1;
            if (smoothed.rewriteDetected) {
                partialRewriteCountRef.current += 1;
            }
            const runtimeCorrectionRate = partialRewriteCountRef.current / Math.max(1, partialEventCountRef.current);
            setCorrectionRate((previous) => previous * 0.65 + runtimeCorrectionRate * 0.35);

            if (smoothed.hesitationLikely) {
                setHesitationFilterTriggered(true);
                if (hesitationTimerRef.current !== null) {
                    window.clearTimeout(hesitationTimerRef.current);
                }
                hesitationTimerRef.current = window.setTimeout(() => {
                    setHesitationFilterTriggered(false);
                }, DEBUG_HESITATION_FLASH_MS);
            }

            setStabilityLevel((previous) => pending.partial?.stabilityLevel ?? previous);
            setSpeechWps((previous) => pending.partial?.speechWps ?? previous);
            setEmitIntervalMs((previous) => pending.partial?.emitIntervalMs ?? previous);
            setSilenceMs((previous) => pending.partial?.silenceMs ?? previous);
            setIsThinking(Boolean(pending.partial.thinking));
            setPrediction(pending.partial.prediction || "");
            setIsCalibrated((previous) => pending.partial?.calibrated ?? previous);
            setCalibrationProgress((previous) => pending.partial?.calibrationProgress ?? previous);
            setConfidenceBaseline((previous) => pending.partial?.confidenceBaseline ?? previous);
            setBackendRms((previous) => {
                if (typeof pending.partial?.rms !== "number" || Number.isNaN(pending.partial.rms)) {
                    return previous;
                }
                return previous * 0.65 + pending.partial.rms * 0.35;
            });

            const now = Date.now();
            if (lastPartialArrivalRef.current > 0) {
                const delta = now - lastPartialArrivalRef.current;
                setPerceivedLatencyMs((previous) => previous * 0.7 + delta * 0.3);
            }
            lastPartialArrivalRef.current = now;

            if (finalRef.current) {
                finalRef.current = "";
                setFinalText("");
                setCorrectedWordIndexes([]);
            }

            if (isRecordingRef.current) {
                setStatus("listening");
            }
        }

        if (pending.final) {
            const nextFinalText = normalizeSpaces(pending.final);
            const liveBeforeFinal = normalizeSpaces(`${stableRef.current} ${partialRef.current}`);
            const changedIndexes = computeChangedWordIndexes(liveBeforeFinal, nextFinalText);
            const majorCorrection = shouldHighlightCorrections(liveBeforeFinal, nextFinalText, changedIndexes);
            const finalDiff = wordDiffRatio(liveBeforeFinal, nextFinalText);
            finalLockedRef.current = true;
            processingStartedAtRef.current = null;

            finalRef.current = nextFinalText;
            stableRef.current = nextFinalText;
            partialRef.current = "";

            setStableText(nextFinalText);
            setPartialText("");
            setPartialConfidences([]);
            setFinalText(nextFinalText);
            setStatus("ready");
            setFinalRevision((value) => value + 1);
            setCorrectedWordIndexes(majorCorrection ? changedIndexes : []);
            setCorrectionRate((previous) => previous * 0.55 + finalDiff * 0.45);
            setPrediction("");
            setIsThinking(false);
            setSilenceMs(0);
            setCalibrationProgress(1);
            setProcessingProgress(100);

            if (correctionTimerRef.current !== null) {
                window.clearTimeout(correctionTimerRef.current);
            }
            if (majorCorrection) {
                correctionTimerRef.current = window.setTimeout(() => {
                    setCorrectedWordIndexes([]);
                }, FINAL_CORRECTION_HIGHLIGHT_MS);
            } else {
                correctionTimerRef.current = null;
            }
        }
    }, []);

    const scheduleUiFlush = useCallback(() => {
        if (flushTimerRef.current !== null) {
            return;
        }
        flushTimerRef.current = window.setTimeout(() => {
            applyPendingMessages();
        }, UI_BATCH_DELAY_MS);
    }, [applyPendingMessages]);

    const handleServerMessage = useCallback(
        (message: ServerMessage) => {
            switch (message.type) {
                case "partial":
                    if (finalLockedRef.current && !isRecordingRef.current) {
                        return;
                    }
                    pendingRef.current.partial = message;
                    break;
                case "final":
                    pendingRef.current.final = message.text;
                    break;
                case "status":
                    pendingRef.current.status = message.state || "unknown";
                    break;
                case "processing":
                    if (typeof message.status === "string") {
                        const debugPayload = parseProcessingDebugStatus(message.status);
                        if (debugPayload) {
                            if (debugPayload.queueSize !== undefined) {
                                setWhisperQueueSize(debugPayload.queueSize);
                            }
                            if (debugPayload.processingDelayMs !== undefined) {
                                setWhisperProcessingDelayMs(debugPayload.processingDelayMs);
                            }
                            if (debugPayload.finalFlushDurationMs !== undefined) {
                                setFinalFlushDurationMs(debugPayload.finalFlushDurationMs);
                                if (debugPayload.finalFlushDurationMs > 0) {
                                    setProcessingProgress((previous) => Math.max(previous, 92));
                                }
                            }
                            if (debugPayload.beamSize !== undefined) {
                                setWhisperBeamSize(debugPayload.beamSize);
                            }
                            if (debugPayload.activeJobs !== undefined) {
                                setWhisperActiveJobs(debugPayload.activeJobs);
                            }
                            scheduleUiFlush();
                            return;
                        }
                    }
                    if (processingStartedAtRef.current === null) {
                        processingStartedAtRef.current = Date.now();
                    }
                    pendingRef.current.status = "processing";
                    break;
                case "error":
                    pendingRef.current.error = message.message;
                    break;
            }
            scheduleUiFlush();
        },
        [scheduleUiFlush]
    );

    useEffect(() => {
        const socket = new RealtimeSocket(
            {
                onMessage: handleServerMessage,
                onOpen: () => setStatus("connected"),
                onClose: () => setStatus("disconnected"),
                onError: (message) => setError(message),
            },
            process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:5000"
        );

        socketRef.current = socket;
        socket.connect();

        return () => {
            stopRecorderOnly();
            stopTracks();
            teardownEnergyMonitor();
            resetElapsedTimer();
            clearUiTimers();
            socket.close();
        };
    }, [clearUiTimers, handleServerMessage, resetElapsedTimer, stopRecorderOnly, stopTracks, teardownEnergyMonitor]);

    const startRecording = async () => {
        if (isRecording || !socketRef.current) return;

        socketRef.current.connect();
        if (!socketRef.current.isOpen()) {
            setStatus("connecting");
            const connected = await socketRef.current.waitForOpen(3000);
            if (!connected) {
                setError("WebSocket connection timed out.");
                setStatus("disconnected");
                return;
            }
        }

        setError("");
        finalLockedRef.current = false;
        processingStartedAtRef.current = null;
        setPrediction("");
        setIsThinking(false);
        setSilenceMs(0);
        setIsCalibrated(false);
        setCalibrationProgress(0);
        setBackendRms(0);
        setCorrectionRate(0);
        setPerceivedLatencyMs(0);
        setHesitationFilterTriggered(false);
        setWhisperQueueSize(0);
        setWhisperProcessingDelayMs(0);
        setFinalFlushDurationMs(0);
        setWhisperBeamSize(2);
        setWhisperActiveJobs(0);
        setProcessingProgress(0);
        partialEventCountRef.current = 0;
        partialRewriteCountRef.current = 0;
        lastPartialArrivalRef.current = 0;
        if (finalText) {
            clearTranscript();
        }

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            streamRef.current = stream;
            setupEnergyMonitor(stream);

            const mimeType = pickMimeType();
            const recorder = new MediaRecorder(stream, { mimeType });
            recorderRef.current = recorder;

            recorder.ondataavailable = async (event) => {
                if (event.data.size <= 0) {
                    return;
                }
                const buffer = await event.data.arrayBuffer();
                socketRef.current?.sendAudio(buffer);
            };

            recorder.onstop = () => {
                socketRef.current?.sendCommand({ type: "stop" });
            };

            socketRef.current.sendCommand({ type: "start", mimeType });
            recorder.start(CHUNK_MS);
            setIsRecording(true);
            setStatus("listening");
            startElapsedTimer();
        } catch (err) {
            setError(err instanceof Error ? err.message : "Microphone access denied.");
            setStatus("ready");
            teardownEnergyMonitor();
        }
    };

    const stopRecording = useCallback(() => {
        if (!isRecording) return;

        setIsRecording(false);

        // Stop recorder FIRST. This triggers the final ondataavailable and then onstop, which notifies the backend safely.
        if (recorderRef.current && recorderRef.current.state !== "inactive") {
            recorderRef.current.stop();
        }
        recorderRef.current = null;

        // Then stop tracks
        stopTracks();
        teardownEnergyMonitor();

        setStatus("processing");
        setProcessingProgress((previous) => Math.max(previous, 8));
        setIsThinking(false);
        setHesitationFilterTriggered(false);

        if (recorderTimerRef.current !== null) {
            window.clearInterval(recorderTimerRef.current);
            recorderTimerRef.current = null;
        }
    }, [isRecording, stopTracks, teardownEnergyMonitor]);

    const handlePrimaryAction = async (event: MouseEvent<HTMLButtonElement>) => {
        addButtonRipple(event);
        if (isRecording) {
            stopRecording();
            return;
        }
        await startRecording();
    };

    const copyToClipboard = () => {
        const text = normalizeSpaces(finalText || `${stableText} ${partialText}`);
        if (!text) return;
        navigator.clipboard.writeText(text);
        setCopied(true);
        window.setTimeout(() => setCopied(false), 1500);
    };

    const hasLiveContent = !!normalizeSpaces(`${stableText} ${partialText}`);
    const hasFinalContent = !!normalizeSpaces(finalText);
    const partialWords = wordsFromText(partialText);
    const finalWords = wordsFromText(finalText);
    const correctedWordIndexSet = useMemo(() => new Set(correctedWordIndexes), [correctedWordIndexes]);
    const predictedTail = useMemo(() => {
        const suggestion = normalizeSpaces(prediction);
        if (!suggestion || partialWords.length === 0) {
            return "";
        }

        const lastWord = partialWords[partialWords.length - 1];
        const suggestionLower = suggestion.toLowerCase();
        const lastLower = lastWord.toLowerCase();

        if (suggestionLower.startsWith(lastLower) && suggestionLower !== lastLower) {
            return suggestion.slice(lastWord.length);
        }
        if (suggestionLower !== lastLower) {
            return ` ${suggestion}`;
        }
        return "";
    }, [partialWords, prediction]);

    const avgPartialConfidence = partialConfidences.length > 0
        ? partialConfidences.reduce((sum, value) => sum + value, 0) / partialConfidences.length
        : confidenceBaseline;
    const stableGlow = Math.max(0.18, Math.min(0.62, avgPartialConfidence));
    const confidenceHeat = Math.max(0, Math.min(1, avgPartialConfidence));
    const elapsedTotalSeconds = Math.floor(elapsedMs / 1000);
    const formattedTime = `${String(Math.floor(elapsedTotalSeconds / 60)).padStart(2, "0")}:${String(
        elapsedTotalSeconds % 60
    ).padStart(2, "0")}`;
    const calibrationPercent = Math.round(Math.max(0, Math.min(1, calibrationProgress)) * 100);
    const thoughtPause = isRecording && (isThinking || (silenceMs > 420 && partialWords.length === 0));
    const animationTempo = Math.max(0.72, Math.min(1.34, 1.2 - speechWps * 0.17));
    const staggerSeconds = (WORD_STAGGER_MS * animationTempo) / 1000;
    const shimmerDuration = 1.15 * animationTempo;
    const partialShiftDuration = 0.15 * animationTempo;
    const cursorPulseDuration = 0.8 * animationTempo;
    const predictionActive = predictedTail.length > 0;

    const dynamicState = status === "processing"
        ? "🧠 Finalizing transcript..."
        : isRecording
            ? !isCalibrated
                ? `Calibrating voice profile... ${Math.min(99, calibrationPercent)}%`
                : thoughtPause
                    ? "Thinking..."
                    : partialWords.length > 0
                        ? "Understanding..."
                        : "Listening..."
            : "Ready";

    const orbMode: "ready" | "listening" | "processing" = status === "processing"
        ? "processing"
        : isRecording
            ? "listening"
            : "ready";

    const orbRhythm = Math.max(0, Math.min(1, speechWps / 3.2));
    const orbScale = orbMode === "listening" ? 1 + orbEnergy * 0.22 + orbRhythm * 0.05 : 1;

    return (
        <div className="min-h-screen text-[var(--text-primary)] px-4 sm:px-6 py-4 md:py-6 flex flex-col items-center justify-center font-sans relative transition-colors duration-500">
            <MouseGlitter isDarkMode={isDarkMode} />
            <div className="w-full max-w-4xl mx-auto relative z-10">
                <div className="relative glass-card rounded-[2.5rem] overflow-hidden">
                    <div className="relative p-6 md:p-8 flex flex-col items-center">
                        <div className="flex flex-col items-center text-center gap-2 mb-6 w-full relative">
                            <button
                                onClick={() => setIsDarkMode(!isDarkMode)}
                                className="absolute right-0 top-0 p-3 rounded-full hover:bg-[var(--text-primary)] hover:bg-opacity-5 transition-colors text-[var(--text-secondary)] hover:text-[var(--accent)]"
                                aria-label="Toggle Theme"
                            >
                                {isDarkMode ? <Sun size={20} /> : <Moon size={20} />}
                            </button>
                            <p className="text-xs uppercase tracking-[0.3em] text-[var(--accent)] font-medium">Voxora Intelligence</p>
                            <h1 className="text-4xl md:text-5xl font-normal tracking-tight text-[var(--text-primary)]">
                                Real-Time Dictation
                            </h1>
                            <div className="mt-4 glass-card px-6 py-2 rounded-full flex items-center gap-3">
                                <span className="relative flex h-2 w-2">
                                  {isRecording && <span className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-75" style={{ backgroundColor: "var(--accent)" }}></span>}
                                  <span className={`relative inline-flex rounded-full h-2 w-2`} style={{ backgroundColor: isRecording ? "var(--accent)" : "var(--text-tertiary)" }}></span>
                                </span>
                                <span className="text-sm font-mono tracking-widest text-[var(--text-secondary)]">{formattedTime}</span>
                            </div>
                        </div>

                        <div className="w-full flex flex-col items-center gap-8">
                            <div className="flex flex-col items-center w-full max-w-[320px]">
                                <div className="relative w-36 h-36 mx-auto flex items-center justify-center">
                                    <motion.div
                                        className="absolute inset-0 rounded-full"
                                        animate={
                                            orbMode === "processing"
                                                ? { opacity: [0.2, 0.5, 0.2], scale: [1, 1.08, 1] }
                                                : orbMode === "listening"
                                                    ? thoughtPause
                                                        ? { opacity: [0.3, 0.6, 0.3], scale: [1.02, 1.08, 1.02] }
                                                        : { opacity: 0.6 + orbEnergy * 0.4, scale: 1.05 + orbEnergy * 0.25 }
                                                    : { opacity: [0.1, 0.3, 0.1], scale: [1, 1.05, 1] }
                                        }
                                        transition={
                                            orbMode === "listening"
                                                ? { duration: thoughtPause ? 1.2 : Math.max(0.1, 0.2 - orbRhythm * 0.1), ease: "easeOut" }
                                                : { duration: 3, repeat: Infinity, ease: "easeInOut" }
                                        }
                                        style={{ backgroundColor: "var(--bg-orb-pulse)", filter: "blur(20px)" }}
                                    />
                                    
                                    <motion.div
                                        className="absolute inset-[15px] rounded-full border"
                                        style={{ borderColor: "var(--border-color)", backgroundColor: "var(--bg-card)" }}
                                        animate={{ scale: orbMode === "listening" ? 1 + orbEnergy * 0.05 : 1 }}
                                        transition={{ duration: 0.1 }}
                                    />

                                    <motion.div
                                        className="absolute inset-[30px] rounded-full shadow-lg"
                                        style={{ 
                                            background: orbMode === "listening" ? "var(--accent)" : "var(--bg-base)",
                                            border: `1px solid var(--border-color)`
                                        }}
                                        animate={{ 
                                            scale: orbMode === "listening" ? 1 + orbEnergy * 0.1 : 1,
                                            opacity: orbMode === "processing" ? [0.5, 1, 0.5] : 1
                                        }}
                                        transition={{ 
                                            duration: orbMode === "processing" ? 1.5 : 0.1,
                                            repeat: orbMode === "processing" ? Infinity : 0
                                        }}
                                    />
                                </div>

                                <div className="mt-6 text-center">
                                    <p className="text-xs uppercase tracking-[0.2em] font-medium" style={{ color: "var(--accent)" }}>State</p>
                                    <p className="text-lg mt-2 text-[var(--text-primary)]">{dynamicState}</p>
                                    <p className="text-xs mt-2 text-[var(--text-secondary)]">
                                        Stability {stabilityLevel} | {speechWps.toFixed(2)} w/s | {emitIntervalMs} ms
                                    </p>
                                    <p className="text-[11px] mt-1 text-[var(--text-tertiary)]">
                                        {isCalibrated ? "Calibrated profile active" : `Calibrating ${Math.min(99, calibrationPercent)}%`} | Base conf {confidenceBaseline.toFixed(2)}
                                    </p>
                                </div>

                                <motion.button
                                    onClick={handlePrimaryAction}
                                    whileHover={{ scale: 1.015 }}
                                    whileTap={{ scale: 0.985 }}
                                    className={`relative mt-4 w-full overflow-hidden rounded-2xl px-5 py-3 font-medium tracking-wide transition ${isRecording
                                        ? "border border-red-400/30 text-red-500 bg-red-500/10"
                                        : "text-white"
                                        }`}
                                    style={!isRecording ? { backgroundColor: "var(--accent)" } : undefined}
                                >
                                    {ripples.map((ripple) => (
                                        <span
                                            key={ripple.id}
                                            className="pointer-events-none absolute h-24 w-24 -translate-x-1/2 -translate-y-1/2 rounded-full animate-ping"
                                            style={{ left: ripple.x, top: ripple.y, backgroundColor: "var(--bg-card)", opacity: 0.4 }}
                                        />
                                    ))}
                                    <span className="relative z-10">{isRecording ? "Stop Dictation" : "Start Dictation"}</span>
                                </motion.button>

                                {error ? <p className="mt-3 text-sm text-red-500">{error}</p> : null}
                            </div>

                            <div className="w-full glass-card rounded-3xl min-h-[300px] flex flex-col relative overflow-hidden mt-6">
                                <div className="absolute inset-0" style={{ backgroundImage: "linear-gradient(to bottom, var(--border-color), transparent)", opacity: 0.5 }} />
                                <div className="relative flex items-center justify-between px-8 py-5 border-b" style={{ borderColor: "var(--border-color)" }}>
                                    <div>
                                        <p className="text-xs uppercase tracking-[0.2em] text-[var(--accent)] font-medium">Live Transcript</p>
                                        <p className="text-sm text-[var(--text-tertiary)] mt-1">
                                            Confirmed text stays stable. Live hypothesis shimmers until confirmed.
                                        </p>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <button
                                            onClick={clearTranscript}
                                            disabled={!hasLiveContent && !hasFinalContent}
                                            className="flex items-center gap-2 px-4 py-2 rounded-xl transition disabled:opacity-40 hover:opacity-80 hover:text-red-500"
                                            style={{ backgroundColor: "var(--border-color)", color: "var(--text-primary)" }}
                                        >
                                            <Trash2 size={16} />
                                            Clear
                                        </button>
                                        <button
                                            onClick={copyToClipboard}
                                            disabled={!hasLiveContent && !hasFinalContent}
                                            className="flex items-center gap-2 px-4 py-2 rounded-xl transition disabled:opacity-40 hover:opacity-80"
                                            style={{ backgroundColor: "var(--border-color)", color: "var(--text-primary)" }}
                                        >
                                            {copied ? <Check size={16} /> : <Copy size={16} />}
                                            Copy
                                        </button>
                                    </div>
                                </div>

                                <div className="flex-1 px-6 py-5 overflow-y-auto">
                                    <AnimatePresence mode="wait" initial={false}>
                                        {!hasLiveContent && !hasFinalContent ? (
                                            <motion.div
                                                key="idle"
                                                initial={{ opacity: 0 }}
                                                animate={{ opacity: 0.8 }}
                                                exit={{ opacity: 0 }}
                                                className="flex h-full items-center justify-center text-[var(--text-tertiary)] text-lg md:text-xl font-normal tracking-wide text-center pt-20"
                                            >
                                                Tap the orb to begin speaking...
                                            </motion.div>
                                        ) : hasFinalContent ? (
                                            <motion.div
                                                key={`final-${finalRevision}`}
                                                initial={{ opacity: 0, filter: "blur(8px)", y: 12 }}
                                                animate={{ opacity: 1, filter: "blur(0px)", y: 0 }}
                                                exit={{ opacity: 0, filter: "blur(4px)", y: -10 }}
                                                transition={{ duration: 0.5, ease: "easeOut" }}
                                                className="text-[1.2rem] md:text-[1.4rem] leading-relaxed font-normal text-center text-[var(--text-primary)]"
                                            >
                                                {finalWords.map((word, index) => {
                                                    const corrected = correctedWordIndexSet.has(index);
                                                    return (
                                                        <motion.span
                                                            key={`final-word-${index}-${word}`}
                                                            className={corrected ? "rounded px-1" : ""}
                                                            style={corrected ? { color: "var(--accent)" } : undefined}
                                                            initial={corrected ? { backgroundColor: "var(--border-color)" } : false}
                                                            animate={corrected ? { backgroundColor: "transparent" } : undefined}
                                                            transition={corrected ? { duration: 0.5 } : undefined}
                                                        >
                                                            {word}{" "}
                                                        </motion.span>
                                                    );
                                                })}
                                            </motion.div>
                                        ) : (
                                            <motion.div
                                                key="live"
                                                initial={{ opacity: 0, filter: "blur(8px)", y: 8 }}
                                                animate={{ opacity: 1, filter: "blur(0px)", y: 0 }}
                                                exit={{ opacity: 0, filter: "blur(4px)", y: -8 }}
                                                transition={{ duration: 0.4, ease: "easeOut" }}
                                                className="text-[1.2rem] md:text-[1.4rem] leading-relaxed font-normal text-center text-[var(--text-primary)]"
                                            >
                                                <span
                                                    className="transition-[text-shadow] duration-150"
                                                    style={{
                                                        textShadow: isDarkMode ? `0 0 18px rgba(52,211,153,${(stableGlow * 0.3).toFixed(3)})` : 'none',
                                                        opacity: 0.78 + confidenceHeat * 0.22,
                                                    }}
                                                >
                                                    {stableText}
                                                </span>
                                                {stableText && partialWords.length > 0 ? " " : null}
                                                {partialWords.map((word, index) => (
                                                    <motion.span
                                                        key={`partial-${index}-${word}`}
                                                        initial={{ opacity: 0, y: 6 }}
                                                        animate={{
                                                            opacity: toOpacity(partialConfidences[index]),
                                                            y: 0,
                                                            backgroundPosition: ["0% 50%", "100% 50%", "0% 50%"],
                                                        }}
                                                        transition={{
                                                            opacity: { duration: Math.max(0.08, partialShiftDuration) },
                                                            y: { duration: Math.max(0.1, partialShiftDuration + 0.02), delay: index * staggerSeconds },
                                                            backgroundPosition: { duration: shimmerDuration, repeat: Infinity, ease: "linear" },
                                                        }}
                                                        className="inline-block bg-clip-text text-transparent"
                                                        style={{ backgroundImage: "linear-gradient(to right, var(--text-secondary), var(--text-primary), var(--text-secondary))", backgroundSize: "220% 100%" }}
                                                    >
                                                        {word}{" "}
                                                    </motion.span>
                                                ))}
                                                {predictedTail ? (
                                                    <motion.span
                                                        initial={{ opacity: 0, y: 4 }}
                                                        animate={{ opacity: 0.34, y: 0 }}
                                                        transition={{ duration: 0.18 }}
                                                        className="text-[var(--text-tertiary)]"
                                                    >
                                                        {predictedTail}
                                                    </motion.span>
                                                ) : null}
                                                {isRecording ? (
                                                    <motion.span
                                                        animate={{ opacity: [0.2, 1, 0.2] }}
                                                        transition={{ repeat: Infinity, duration: cursorPulseDuration }}
                                                        className="text-[var(--accent)]"
                                                    >
                                                        |
                                                    </motion.span>
                                                ) : null}
                                                {thoughtPause ? (
                                                    <motion.span
                                                        initial={{ opacity: 0 }}
                                                        animate={{ opacity: [0.2, 0.9, 0.2] }}
                                                        transition={{ repeat: Infinity, duration: 1.1 }}
                                                        className="ml-2 text-[var(--accent)] text-sm align-middle"
                                                    >
                                                        ...
                                                    </motion.span>
                                                ) : null}
                                                {status === "processing" ? (
                                                    <span className="ml-3 inline-flex flex-col gap-1 align-middle min-w-[220px]">
                                                        <motion.span
                                                            initial={{ opacity: 0 }}
                                                            animate={{ opacity: 1 }}
                                                            className="inline-flex items-center gap-2 text-[var(--accent)] text-sm"
                                                        >
                                                            <motion.span
                                                                animate={{ rotate: 360 }}
                                                                transition={{ repeat: Infinity, duration: 0.9, ease: "linear" }}
                                                                className="inline-block"
                                                            >
                                                                ◌
                                                            </motion.span>
                                                            Finalizing transcript... {Math.min(99, Math.round(processingProgress))}%
                                                        </motion.span>
                                                        <span className="h-1.5 w-full rounded-full overflow-hidden" style={{ backgroundColor: "var(--border-color)" }}>
                                                            <motion.span
                                                                className="h-full block rounded-full"
                                                                style={{ backgroundColor: "var(--accent)" }}
                                                                animate={{ width: `${Math.max(8, Math.min(100, processingProgress))}%` }}
                                                                transition={{ duration: 0.18, ease: "easeOut" }}
                                                            />
                                                        </span>
                                                    </span>
                                                ) : null}
                                            </motion.div>
                                        )}
                                    </AnimatePresence>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <button
                type="button"
                onClick={() => setDebugVisible((previous) => !previous)}
                className="fixed right-16 top-4 z-40 h-10 w-10 flex items-center justify-center rounded-full border transition hover:opacity-75"
                style={{ borderColor: "var(--border-color)", backgroundColor: "var(--bg-card)", color: "var(--text-tertiary)" }}
                aria-label="Toggle debug panel (Ctrl+Shift+D)"
                title="Toggle debug panel (Ctrl+Shift+D)"
            >
                D
            </button>

            <AnimatePresence>
                {debugVisible ? (
                    <motion.aside
                        initial={{ opacity: 0, y: 12, scale: 0.98 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: 8, scale: 0.98 }}
                        transition={{ duration: 0.2 }}
                        className="fixed right-4 bottom-4 z-50 w-[330px] rounded-2xl border p-4 backdrop-blur-xl glass-card text-[var(--text-primary)]"
                        style={{ borderColor: "var(--border-color)", fontFamily: "'IBM Plex Mono', 'Consolas', monospace" }}
                    >
                        <div className="flex items-center justify-between text-[11px] uppercase tracking-[0.14em] text-cyan-200/80">
                            <span>Debug Metrics</span>
                            <span className="text-white/55">Ctrl+Shift+D</span>
                        </div>

                        <div className="mt-3 space-y-3 text-[12px] leading-relaxed">
                            <div className="rounded-lg border border-white/10 bg-black/20 p-2.5">
                                <p className="text-white/50 uppercase tracking-[0.12em] text-[10px] mb-1">Calibration</p>
                                <p className={metricTone(speechWps, 1.1, 0.7)}>WPS: {speechWps.toFixed(2)}</p>
                                <p className={stabilityLevel >= 3 ? "text-amber-300" : "text-emerald-300"}>Stability: {stabilityLevel}</p>
                                <p className={metricTone(confidenceBaseline, 0.58, 0.5)}>Confidence Baseline: {confidenceBaseline.toFixed(2)}</p>
                            </div>

                            <div className="rounded-lg border border-white/10 bg-black/20 p-2.5">
                                <p className="text-white/50 uppercase tracking-[0.12em] text-[10px] mb-1">Audio</p>
                                <p className={metricTone(backendRms, 0.008, 0.004)}>RMS: {backendRms.toFixed(4)}</p>
                                <p className={metricTone(silenceMs, 700, 1300, true)}>Silence: {Math.round(silenceMs)} ms</p>
                                <p className={metricTone(orbEnergy, 0.08, 0.03)}>Energy: {orbEnergy.toFixed(3)}</p>
                            </div>

                            <div className="rounded-lg border border-white/10 bg-black/20 p-2.5">
                                <p className="text-white/50 uppercase tracking-[0.12em] text-[10px] mb-1">Transcription</p>
                                <p className={metricTone(avgPartialConfidence, 0.62, 0.5)}>Partial Avg Conf: {(avgPartialConfidence * 100).toFixed(1)}%</p>
                                <p className={metricTone(correctionRate, 0.18, 0.25, true)}>Correction Rate: {(correctionRate * 100).toFixed(1)}%</p>
                                <p className={metricTone(emitIntervalMs, 140, 190, true)}>Emit Interval: {emitIntervalMs} ms</p>
                                <p className={metricTone(perceivedLatencyMs || emitIntervalMs + UI_BATCH_DELAY_MS, 150, 230, true)}>
                                    Perceived Latency: {(perceivedLatencyMs || emitIntervalMs + UI_BATCH_DELAY_MS).toFixed(1)} ms
                                </p>
                            </div>

                            <div className="rounded-lg border border-white/10 bg-black/20 p-2.5">
                                <p className="text-white/50 uppercase tracking-[0.12em] text-[10px] mb-1">Whisper</p>
                                <p className={metricTone(whisperQueueSize, 2, 3, true)}>Queue Size: {whisperQueueSize}</p>
                                <p className={metricTone(whisperProcessingDelayMs, 900, 1500, true)}>
                                    Processing Delay: {whisperProcessingDelayMs.toFixed(1)} ms
                                </p>
                                <p className={metricTone(finalFlushDurationMs || processingProgress * 20, 1500, 2500, true)}>
                                    Final Flush: {(finalFlushDurationMs || 0).toFixed(1)} ms
                                </p>
                                <p className={whisperBeamSize <= 1 ? "text-amber-300" : "text-emerald-300"}>
                                    Beam Size: {whisperBeamSize} | Active Jobs: {whisperActiveJobs}
                                </p>
                            </div>

                            <div className="rounded-lg border border-white/10 bg-black/20 p-2.5">
                                <p className="text-white/50 uppercase tracking-[0.12em] text-[10px] mb-1">Intelligence</p>
                                <p className={predictionActive ? "text-emerald-300" : "text-white/65"}>Prediction Active: {predictionActive ? "true" : "false"}</p>
                                <p className={hesitationFilterTriggered ? "text-amber-300" : "text-emerald-300"}>
                                    Hesitation Filter: {hesitationFilterTriggered ? "yes" : "no"}
                                </p>
                                <p className="text-white/65">Calibrated: {isCalibrated ? "true" : "false"}</p>
                            </div>
                        </div>
                    </motion.aside>
                ) : null}
            </AnimatePresence>
        </div>
    );
}

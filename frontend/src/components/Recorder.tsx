"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Mic, Square, Copy, Check } from "lucide-react";
import { RealtimeSocket, type ServerMessage } from "@/lib/websocket";

/* ====== SAME LOGIC (UNCHANGED) ====== */
const DEFAULT_CHUNK_MS = 25;
const MIN_CHUNK_MS = 20;
const MAX_CHUNK_MS = 30;

const CHUNK_MS = (() => {
    const parsed = Number(process.env.NEXT_PUBLIC_RECORDER_CHUNK_MS || DEFAULT_CHUNK_MS);
    return isFinite(parsed)
        ? Math.min(MAX_CHUNK_MS, Math.max(MIN_CHUNK_MS, Math.floor(parsed)))
        : DEFAULT_CHUNK_MS;
})();

function pickMimeType(): string {
    if (typeof MediaRecorder === "undefined") return "audio/webm";
    return MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : "audio/webm";
}

export default function PerfectPremiumRecorder() {
    const socketRef = useRef<RealtimeSocket | null>(null);
    const recorderRef = useRef<MediaRecorder | null>(null);
    const streamRef = useRef<MediaStream | null>(null);

    const [isRecording, setIsRecording] = useState(false);
    const [stableText, setStableText] = useState("");
    const [partialText, setPartialText] = useState("");
    const [finalText, setFinalText] = useState("");
    const [status, setStatus] = useState("disconnected");
    const [error, setError] = useState("");
    const [copied, setCopied] = useState(false);

    const stableRef = useRef("");
    const partialRef = useRef("");
    const finalRef = useRef("");

    const copyToClipboard = () => {
        const text = finalText || `${stableText} ${partialText}`;
        if (!text.trim()) return;
        navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    const clearTranscript = () => {
        setStableText("");
        setPartialText("");
        setFinalText("");
        stableRef.current = "";
        partialRef.current = "";
        finalRef.current = "";
    };

    const handleServerMessage = useCallback((message: ServerMessage) => {
        switch (message.type) {
            case "partial":
                stableRef.current = message.stable;
                partialRef.current = message.partial;
                setStableText(message.stable);
                setPartialText(message.partial);
                if (finalRef.current) {
                    finalRef.current = "";
                    setFinalText("");
                }
                break;
            case "final":
                finalRef.current = message.text;
                setFinalText(message.text);
                setStableText(message.text);
                setPartialText("");
                setStatus("ready");
                break;
            case "status":
                setStatus(message.state || "unknown");
                break;
            case "error":
                setError(message.message);
                break;
        }
    }, []);

    const stopTracks = useCallback(() => {
        streamRef.current?.getTracks().forEach(t => t.stop());
        streamRef.current = null;
    }, []);

    const stopRecorderOnly = useCallback(() => {
        if (recorderRef.current?.state !== "inactive") recorderRef.current?.stop();
        recorderRef.current = null;
    }, []);

    useEffect(() => {
        const socket = new RealtimeSocket(
            {
                onMessage: handleServerMessage,
                onOpen: () => setStatus("connected"),
                onClose: () => setStatus("disconnected"),
                onError: (msg) => setError(msg),
            },
            process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:5000"
        );
        socketRef.current = socket;
        socket.connect();

        return () => {
            stopRecorderOnly();
            stopTracks();
            socket.close();
        };
    }, [handleServerMessage, stopRecorderOnly, stopTracks]);

    const startRecording = async () => {
        if (isRecording || !socketRef.current) return;

        socketRef.current.connect();
        if (!socketRef.current.isOpen()) {
            setStatus("connecting");
            setError("Establishing secure connection...");
            return;
        }

        setError("");
        if (finalText) clearTranscript();

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            streamRef.current = stream;

            const mimeType = pickMimeType();
            const recorder = new MediaRecorder(stream, { mimeType });
            recorderRef.current = recorder;

            recorder.ondataavailable = async (e) => {
                if (e.data.size > 0) {
                    const buffer = await e.data.arrayBuffer();
                    socketRef.current?.sendAudio(buffer);
                }
            };

            socketRef.current.sendCommand({ type: "start", mimeType });
            recorder.start(CHUNK_MS);

            setIsRecording(true);
            setStatus("listening");
        } catch (err) {
            setError(err instanceof Error ? err.message : "Microphone access denied.");
        }
    };

    const stopRecording = () => {
        if (!isRecording) return;
        stopRecorderOnly();
        stopTracks();
        socketRef.current?.sendCommand({ type: "stop" });
        setIsRecording(false);
        setStatus("processing");
    };

    const hasContent = !!(finalText || stableText || partialText);

    /* ====== UI REDESIGN STARTS HERE ====== */

    return (
        <div className="min-h-screen bg-black flex items-center justify-center p-6 text-white">

            <div className="w-full max-w-6xl h-[520px] rounded-3xl border border-white/10 bg-[#0b0f14]/80 backdrop-blur-xl shadow-2xl flex overflow-hidden">

                {/* LEFT PANEL */}
                <div className="w-1/2 flex flex-col items-center justify-center border-r border-white/10 relative">

                    <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/10 via-transparent to-indigo-500/10 blur-2xl" />

                    <div className="relative z-10 flex flex-col items-center">

                        <p className="text-xs text-zinc-500 tracking-widest mb-2">AI CORE</p>
                        <h2 className="text-xl font-semibold mb-6">Voice Intelligence</h2>

                        {/* Wave Circle */}
                        <motion.div
                            animate={isRecording ? { scale: [1, 1.15, 1] } : { scale: 1 }}
                            transition={
                                isRecording
                                    ? { repeat: Infinity, duration: 1.5 }
                                    : { duration: 0 }
                            }
                            className="w-40 h-40 rounded-full border border-cyan-400/30 flex items-center justify-center mb-6"
                        >
                            <div className="flex gap-[2px]">
                                {Array.from({ length: 25 }).map((_, i) => (
                                    <motion.div
                                        key={i}
                                        animate={isRecording ? { height: [4, 22, 6] } : { height: 4 }}
                                        transition={
                                            isRecording
                                                ? { repeat: Infinity, duration: 1, delay: i * 0.04 }
                                                : { duration: 0 }
                                        }
                                        className="w-[2px] bg-cyan-400"
                                    />
                                ))}
                            </div>
                        </motion.div>

                        <h1 className="text-4xl font-light mb-2">00:00</h1>
                        <p className="text-xs text-zinc-500 uppercase tracking-wider mb-6">
                            {isRecording
                                ? "RECORDING"
                                : status === "processing"
                                    ? "PROCESSING"
                                    : status === "connecting"
                                        ? "CONNECTING"
                                        : "READY"}
                        </p>

                        <button
                            onClick={isRecording ? stopRecording : startRecording}
                            className={`px-10 py-3 rounded-xl font-medium tracking-wide transition-all ${isRecording
                                    ? "border border-red-500 text-red-400 hover:bg-red-500/10"
                                    : "bg-gradient-to-r from-cyan-500 to-indigo-500 text-black"
                                }`}
                        >
                            {isRecording ? "STOP SESSION" : "START SESSION"}
                        </button>

                        {error && (
                            <p className="mt-4 text-red-400 text-sm">{error}</p>
                        )}
                    </div>
                </div>

                {/* RIGHT PANEL */}
                <div className="w-1/2 flex flex-col">

                    {/* TRANSCRIPT */}
                    <div className="flex-1 p-6 overflow-y-auto">
                        <p className="text-xs text-zinc-500 tracking-widest mb-4">TRANSCRIPT</p>

                        <AnimatePresence>
                            {!hasContent ? (
                                <motion.p
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    className="text-zinc-600"
                                >
                                    Start speaking...
                                </motion.p>
                            ) : (
                                <motion.div className="text-lg leading-relaxed">
                                    <span className="text-white">{stableText} </span>
                                    <span className="text-cyan-300/60">{partialText}</span>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>

                    {/* FOOTER */}
                    <div className="border-t border-white/10 p-4 flex items-center justify-between">

                        <div>
                            <p className="text-xs text-cyan-400 tracking-widest">LIVE INFERENCE</p>
                            <p className="text-xs text-zinc-500">
                                {isRecording
                                    ? "Listening..."
                                    : status === "processing"
                                        ? "Processing..."
                                        : status === "connecting"
                                            ? "Connecting..."
                                            : "Ready"}
                            </p>
                        </div>

                        <button
                            onClick={copyToClipboard}
                            disabled={!hasContent}
                            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white/5 hover:bg-white/10 transition disabled:opacity-30"
                        >
                            {copied ? <Check size={16} /> : <Copy size={16} />}
                            Copy
                        </button>

                    </div>
                </div>
            </div>
        </div>
    );
}
"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Mic, Square, Copy, Check, RotateCcw, Sparkles } from "lucide-react";
import { RealtimeSocket, type ServerMessage } from "@/lib/websocket";

const DEFAULT_CHUNK_MS = 25;
const MIN_CHUNK_MS = 20;
const MAX_CHUNK_MS = 30;
const CHUNK_MS = (() => {
    const parsed = Number(process.env.NEXT_PUBLIC_RECORDER_CHUNK_MS || DEFAULT_CHUNK_MS);
    return isFinite(parsed) ? Math.min(MAX_CHUNK_MS, Math.max(MIN_CHUNK_MS, Math.floor(parsed))) : DEFAULT_CHUNK_MS;
})();

function pickMimeType(): string {
    if (typeof MediaRecorder === "undefined") return "audio/webm";
    return MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? "audio/webm;codecs=opus" : "audio/webm";
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
        if (finalText) clearTranscript(); // Auto-clear if starting fresh

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

    return (
        <div className="min-h-screen bg-[#0a0a0a] text-zinc-100 flex flex-col items-center justify-center p-4 sm:p-8 font-sans selection:bg-indigo-500/30">

            <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
                className="w-full max-w-3xl flex flex-col h-[80vh] max-h-[800px] relative"
            >
                {/* Background Ambient Glow */}
                <div className="absolute inset-0 bg-gradient-to-b from-indigo-500/5 to-transparent rounded-3xl pointer-events-none blur-3xl" />

                <div className="flex-1 flex flex-col bg-[#111111]/80 backdrop-blur-2xl border border-white/[0.05] rounded-[2rem] shadow-2xl overflow-hidden relative z-10">

                    {/* Header & AI Core */}
                    <div className="flex flex-col items-center justify-center pt-10 pb-6 px-8 relative z-20">
                        <div className="relative flex items-center justify-center w-16 h-16 mb-4">
                            {/* Pulsing Orb */}
                            <motion.div
                                animate={{
                                    scale: isRecording ? [1, 1.5, 1] : [1, 1.05, 1],
                                    opacity: isRecording ? [0.4, 0.8, 0.4] : [0.1, 0.2, 0.1]
                                }}
                                transition={{
                                    duration: isRecording ? 1.5 : 4,
                                    repeat: Infinity,
                                    ease: "easeInOut"
                                }}
                                className={`absolute inset-0 rounded-full blur-xl transition-colors duration-700 ${isRecording ? 'bg-indigo-500' : 'bg-zinc-500'}`}
                            />
                            <div className={`relative z-10 w-4 h-4 rounded-full transition-colors duration-700 shadow-inner ${isRecording ? 'bg-indigo-400 shadow-indigo-200' : 'bg-zinc-600'}`} />
                        </div>

                        <h2 className="text-sm font-medium text-zinc-400 tracking-wide uppercase flex items-center gap-2">
                            {isRecording ? "Listening..." : "Voice Intelligence"}
                        </h2>

                        {error && (
                            <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} className="mt-4 px-4 py-2 bg-red-500/10 border border-red-500/20 text-red-400 text-sm rounded-full">
                                {error}
                            </motion.div>
                        )}
                    </div>

                    {/* Transcript Area */}
                    <div className="flex-1 overflow-y-auto px-8 sm:px-12 pb-24 scrollbar-thin scrollbar-thumb-white/10 scrollbar-track-transparent">
                        <AnimatePresence mode="wait">
                            {!hasContent ? (
                                <motion.div
                                    key="empty"
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    exit={{ opacity: 0 }}
                                    className="h-full flex flex-col items-center justify-center text-center space-y-4"
                                >
                                    <Sparkles className="w-8 h-8 text-zinc-700" />
                                    <p className="text-xl sm:text-2xl text-zinc-500 font-light tracking-tight">
                                        What is on your mind?
                                    </p>
                                </motion.div>
                            ) : (
                                <motion.div
                                    key="content"
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    className="text-2xl sm:text-3xl md:text-4xl leading-[1.4] font-light tracking-tight"
                                >
                                    <span className="text-zinc-100">{stableText} </span>
                                    <motion.span
                                        layout
                                        className="text-indigo-300/70 blur-[0.5px]"
                                    >
                                        {partialText}
                                    </motion.span>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>

                    {/* Action Dock (Bottom Overlay) */}
                    <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-4 p-2 bg-zinc-900/80 backdrop-blur-xl border border-white/10 rounded-full shadow-2xl z-30">

                        {/* Clear Button */}
                        <button
                            onClick={clearTranscript}
                            disabled={!hasContent || isRecording}
                            className="w-12 h-12 flex items-center justify-center text-zinc-400 hover:text-zinc-100 disabled:opacity-30 disabled:cursor-not-allowed transition-colors rounded-full hover:bg-white/5"
                            title="Clear Transcript"
                        >
                            <RotateCcw className="w-5 h-5" />
                        </button>

                        {/* Main Record Button */}
                        <motion.button
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                            onClick={isRecording ? stopRecording : startRecording}
                            className={`w-16 h-16 flex items-center justify-center rounded-full shadow-lg transition-all duration-300 ${isRecording
                                    ? "bg-zinc-800 text-red-400 border border-red-500/30 hover:bg-zinc-700"
                                    : "bg-white text-black hover:bg-zinc-200"
                                }`}
                        >
                            <AnimatePresence mode="wait">
                                {isRecording ? (
                                    <motion.div key="stop" initial={{ scale: 0 }} animate={{ scale: 1 }} exit={{ scale: 0 }}>
                                        <Square className="w-6 h-6 fill-current" />
                                    </motion.div>
                                ) : (
                                    <motion.div key="mic" initial={{ scale: 0 }} animate={{ scale: 1 }} exit={{ scale: 0 }}>
                                        <Mic className="w-6 h-6 fill-current" />
                                    </motion.div>
                                )}
                            </AnimatePresence>
                        </motion.button>

                        {/* Copy Button */}
                        <button
                            onClick={copyToClipboard}
                            disabled={!hasContent}
                            className="w-12 h-12 flex items-center justify-center text-zinc-400 hover:text-zinc-100 disabled:opacity-30 disabled:cursor-not-allowed transition-colors rounded-full hover:bg-white/5"
                            title="Copy to Clipboard"
                        >
                            {copied ? <Check className="w-5 h-5 text-emerald-400" /> : <Copy className="w-5 h-5" />}
                        </button>

                    </div>
                </div>
            </motion.div>
        </div>
    );
}
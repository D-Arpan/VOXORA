const WebSocket = require("ws");
const ffmpeg = require("fluent-ffmpeg");
const { PassThrough } = require("stream");

const { createPythonSession } = require("../services/python.service");

const STREAM_FRAME_MS = Math.min(30, Math.max(20, Number(process.env.STREAM_FRAME_MS || 20)));
const SAMPLE_RATE = 16000;
const FLOAT32_BYTES_PER_SAMPLE = 4;
const SAMPLES_PER_FRAME = Math.floor((SAMPLE_RATE * STREAM_FRAME_MS) / 1000);
const FLOAT32_FRAME_BYTES = SAMPLES_PER_FRAME * FLOAT32_BYTES_PER_SAMPLE;

function safeSend(ws, payload) {
    if (ws.readyState !== WebSocket.OPEN) {
        return;
    }
    ws.send(JSON.stringify(payload));
}

function detectInputFormat(mimeType = "") {
    if (typeof mimeType !== "string") {
        return "webm";
    }
    if (mimeType.includes("ogg")) {
        return "ogg";
    }
    return "webm";
}

function createFloat32FrameSlicer(onFrame) {
    let remainder = Buffer.alloc(0);

    const push = (chunk) => {
        if (!chunk || chunk.length === 0) {
            return;
        }

        const merged = remainder.length > 0 ? Buffer.concat([remainder, chunk]) : chunk;
        const fullBytes = merged.length - (merged.length % FLOAT32_FRAME_BYTES);

        let offset = 0;
        while (offset < fullBytes) {
            onFrame(merged.subarray(offset, offset + FLOAT32_FRAME_BYTES));
            offset += FLOAT32_FRAME_BYTES;
        }

        remainder = fullBytes < merged.length ? Buffer.from(merged.subarray(fullBytes)) : Buffer.alloc(0);
    };

    const flush = () => {
        if (remainder.length === 0) {
            return;
        }
        const padded = Buffer.alloc(FLOAT32_FRAME_BYTES);
        remainder.copy(padded, 0, 0, remainder.length);
        remainder = Buffer.alloc(0);
        onFrame(padded);
    };

    return { push, flush };
}

function createFfmpegPipeline({ inputFormat, onAudioFrame, onEnd, onError }) {
    const inputStream = new PassThrough();
    const slicer = createFloat32FrameSlicer(onAudioFrame);

    const command = ffmpeg()
        .input(inputStream)
        .inputFormat(inputFormat)
        .inputOptions(["-fflags", "+nobuffer", "-flags", "low_delay"])
        .noVideo()
        .audioChannels(1)
        .audioFrequency(SAMPLE_RATE)
        .audioCodec("pcm_f32le")
        .format("f32le")
        .audioFilters(`aresample=${SAMPLE_RATE}`)
        .outputOptions(["-fflags", "+nobuffer", "-flags", "low_delay", "-flush_packets", "1"])
        .on("error", (error) => onError(error));

    const outputStream = command.pipe();
    outputStream.on("data", (chunk) => slicer.push(chunk));
    outputStream.on("end", () => {
        slicer.flush();
        onEnd();
    });
    outputStream.on("error", (error) => onError(error));

    return {
        writeChunk(chunk) {
            inputStream.write(chunk);
        },
        endInput() {
            if (!inputStream.destroyed) {
                inputStream.end();
            }
        },
        destroy() {
            try {
                inputStream.destroy();
            } catch {
                // Ignore cleanup errors.
            }

            try {
                outputStream.destroy();
            } catch {
                // Ignore cleanup errors.
            }

            const proc = command.ffmpegProc;
            if (proc && !proc.killed) {
                try {
                    proc.kill("SIGKILL");
                } catch {
                    // Ignore cleanup errors.
                }
            }
        },
    };
}

function initWebSocket(server) {
    const wss = new WebSocket.Server({
        server,
        perMessageDeflate: false,
    });

    wss.on("connection", (ws) => {
        if (ws._socket && typeof ws._socket.setNoDelay === "function") {
            ws._socket.setNoDelay(true);
        }

        const pythonSession = createPythonSession();
        let pythonReady = false;
        let recording = false;
        let stopPending = false;
        let stopFallbackTimer = null;
        let ffmpegPipeline = null;
        let closed = false;

        const pythonReadyPromise = pythonSession
            .connect()
            .then(() => {
                pythonReady = true;
                safeSend(ws, { type: "status", state: "ready" });
            })
            .catch((error) => {
                safeSend(ws, { type: "error", message: `Python connection failed: ${error.message}` });
                cleanup();
            });

        const clearStopFallbackTimer = () => {
            if (stopFallbackTimer) {
                clearTimeout(stopFallbackTimer);
                stopFallbackTimer = null;
            }
        };

        const finalizeStop = () => {
            if (!stopPending) {
                return;
            }
            stopPending = false;
            clearStopFallbackTimer();
            pythonSession.stop();
        };

        const teardownFfmpeg = () => {
            if (!ffmpegPipeline) {
                return;
            }
            ffmpegPipeline.destroy();
            ffmpegPipeline = null;
        };

        const startRecording = ({ mimeType }) => {
            if (recording || closed) {
                return;
            }
            if (!pythonReady) {
                safeSend(ws, {
                    type: "error",
                    message: "Python server is not ready yet. Try again in a moment.",
                });
                return;
            }

            const inputFormat = detectInputFormat(mimeType);
            stopPending = false;
            clearStopFallbackTimer();

            ffmpegPipeline = createFfmpegPipeline({
                inputFormat,
                onAudioFrame: (frame) => {
                    pythonSession.sendAudio(frame);
                },
                onEnd: () => {
                    finalizeStop();
                    teardownFfmpeg();
                },
                onError: (error) => {
                    safeSend(ws, { type: "error", message: `FFmpeg error: ${error.message}` });
                    finalizeStop();
                    teardownFfmpeg();
                },
            });

            pythonSession.start();
            recording = true;
            safeSend(ws, { type: "status", state: "listening" });
        };

        const stopRecording = () => {
            if (!recording) {
                return;
            }

            recording = false;
            stopPending = true;

            if (ffmpegPipeline) {
                ffmpegPipeline.endInput();
                stopFallbackTimer = setTimeout(() => {
                    finalizeStop();
                    teardownFfmpeg();
                }, 250);
            } else {
                finalizeStop();
            }
        };

        const cleanup = () => {
            if (closed) {
                return;
            }
            closed = true;
            recording = false;
            stopPending = false;
            clearStopFallbackTimer();
            teardownFfmpeg();
            pythonSession.close();
            try {
                if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
                    ws.close();
                }
            } catch {
                // Ignore close failures.
            }
        };

        pythonSession.on("event", (event) => {
            if (event.type === "partial") {
                const hasStructuredPartial =
                    Object.prototype.hasOwnProperty.call(event, "stable") ||
                    Object.prototype.hasOwnProperty.call(event, "partial");

                if (hasStructuredPartial) {
                    const stable = String(event.stable || "");
                    const partial = String(event.partial || "");
                    const combined = `${stable} ${partial}`.trim();
                    const partialConfidences = Array.isArray(event.partial_confidences)
                        ? event.partial_confidences
                            .map((value) => Number(value))
                            .filter((value) => Number.isFinite(value))
                        : [];
                    safeSend(ws, {
                        type: "partial",
                        text: combined,
                        stable,
                        partial,
                        partialConfidences,
                        stabilityLevel: Number.isFinite(Number(event.stability_level))
                            ? Number(event.stability_level)
                            : undefined,
                        speechWps: Number.isFinite(Number(event.speech_wps))
                            ? Number(event.speech_wps)
                            : undefined,
                        emitIntervalMs: Number.isFinite(Number(event.emit_interval_ms))
                            ? Number(event.emit_interval_ms)
                            : undefined,
                        rms: Number.isFinite(Number(event.rms))
                            ? Number(event.rms)
                            : undefined,
                        silenceMs: Number.isFinite(Number(event.silence_ms))
                            ? Number(event.silence_ms)
                            : undefined,
                        thinking: typeof event.thinking === "boolean"
                            ? event.thinking
                            : undefined,
                        prediction: typeof event.prediction === "string"
                            ? event.prediction
                            : undefined,
                        calibrated: typeof event.calibrated === "boolean"
                            ? event.calibrated
                            : undefined,
                        calibrationProgress: Number.isFinite(Number(event.calibration_progress))
                            ? Number(event.calibration_progress)
                            : undefined,
                        confidenceBaseline: Number.isFinite(Number(event.confidence_baseline))
                            ? Number(event.confidence_baseline)
                            : undefined,
                    });
                    return;
                }

                const text = String(event.text || "");
                safeSend(ws, {
                    type: "partial",
                    text,
                    stable: text,
                    partial: "",
                    partialConfidences: [],
                });
                return;
            }

            if (event.type === "final") {
                safeSend(ws, {
                    type: "final",
                    text: String(event.text || ""),
                });
                return;
            }

            if (event.type === "error") {
                safeSend(ws, {
                    type: "error",
                    message: String(event.message || "Python transcription error"),
                });
                return;
            }

            if (event.type === "status") {
                safeSend(ws, {
                    type: "status",
                    state: String(event.state || "unknown"),
                });
            }
        });

        pythonSession.on("pythonError", (error) => {
            safeSend(ws, { type: "error", message: `Python socket error: ${error.message}` });
        });

        pythonSession.on("closed", () => {
            if (!closed) {
                safeSend(ws, { type: "error", message: "Python session disconnected" });
                cleanup();
            }
        });

        ws.on("message", (message, isBinary) => {
            if (closed) {
                return;
            }

            if (isBinary) {
                if (!recording || !ffmpegPipeline) {
                    return;
                }
                const audioChunk = Buffer.isBuffer(message) ? message : Buffer.from(message);
                ffmpegPipeline.writeChunk(audioChunk);
                return;
            }

            let parsed;
            try {
                parsed = JSON.parse(message.toString("utf8"));
            } catch {
                safeSend(ws, { type: "error", message: "Invalid websocket JSON payload" });
                return;
            }

            const command = String(parsed.type || "").toLowerCase().trim();
            if (command === "start") {
                pythonReadyPromise
                    .then(() => startRecording(parsed))
                    .catch(() => {
                        // Error already forwarded when pythonReadyPromise failed.
                    });
                return;
            }

            if (command === "stop") {
                stopRecording();
                return;
            }

            if (command === "close") {
                cleanup();
            }
        });

        ws.on("close", () => {
            cleanup();
        });

        ws.on("error", () => {
            cleanup();
        });
    });
}

module.exports = { initWebSocket };

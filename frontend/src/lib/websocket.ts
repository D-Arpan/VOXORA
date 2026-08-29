export type PartialMessage = {
    type: "partial";
    text: string;
    stable: string;
    partial: string;
    partialConfidences: number[];
    stabilityLevel?: number;
    speechWps?: number;
    emitIntervalMs?: number;
    rms?: number;
    silenceMs?: number;
    thinking?: boolean;
    prediction?: string;
    calibrated?: boolean;
    calibrationProgress?: number;
    confidenceBaseline?: number;
};

export type FinalMessage = {
    type: "final";
    text: string;
};

export type StatusMessage = {
    type: "status";
    state: string;
};

export type ErrorMessage = {
    type: "error";
    message: string;
};

export type ServerMessage = PartialMessage | FinalMessage | StatusMessage | ErrorMessage;

type EventHandlers = {
    onMessage: (message: ServerMessage) => void;
    onOpen?: () => void;
    onClose?: () => void;
    onError?: (message: string) => void;
};

type StartPayload = {
    type: "start";
    mimeType: string;
};

type StopPayload = {
    type: "stop";
};

type ClosePayload = {
    type: "close";
};

type ClientPayload = StartPayload | StopPayload | ClosePayload;

const DEFAULT_WS_URL = "ws://localhost:5000";

function parseServerMessage(raw: string): ServerMessage | null {
    let parsed: unknown;
    try {
        parsed = JSON.parse(raw);
    } catch {
        return null;
    }

    if (!parsed || typeof parsed !== "object") {
        return null;
    }

    const candidate = parsed as Record<string, unknown>;
    const type = String(candidate.type || "");

    if (type === "partial") {
        const text = String(candidate.text || "");
        const stable = String(candidate.stable || "");
        const partial = String(candidate.partial || "");
        const resolvedText = text || `${stable} ${partial}`.trim();
        const partialConfidences = Array.isArray(candidate.partialConfidences)
            ? candidate.partialConfidences
                .map((value) => Number(value))
                .filter((value) => Number.isFinite(value))
            : [];
        const stabilityLevel = Number(candidate.stabilityLevel);
        const speechWps = Number(candidate.speechWps);
        const emitIntervalMs = Number(candidate.emitIntervalMs);
        const rms = Number(candidate.rms);
        const silenceMs = Number(candidate.silenceMs);
        const calibrationProgress = Number(candidate.calibrationProgress);
        const confidenceBaseline = Number(candidate.confidenceBaseline);

        return {
            type: "partial",
            text: resolvedText,
            stable: stable || resolvedText,
            partial,
            partialConfidences,
            stabilityLevel: Number.isFinite(stabilityLevel) ? stabilityLevel : undefined,
            speechWps: Number.isFinite(speechWps) ? speechWps : undefined,
            emitIntervalMs: Number.isFinite(emitIntervalMs) ? emitIntervalMs : undefined,
            rms: Number.isFinite(rms) ? rms : undefined,
            silenceMs: Number.isFinite(silenceMs) ? silenceMs : undefined,
            thinking: typeof candidate.thinking === "boolean" ? candidate.thinking : undefined,
            prediction: typeof candidate.prediction === "string" ? candidate.prediction : undefined,
            calibrated: typeof candidate.calibrated === "boolean" ? candidate.calibrated : undefined,
            calibrationProgress: Number.isFinite(calibrationProgress) ? calibrationProgress : undefined,
            confidenceBaseline: Number.isFinite(confidenceBaseline) ? confidenceBaseline : undefined,
        };
    }

    if (type === "final") {
        return {
            type: "final",
            text: String(candidate.text || ""),
        };
    }

    if (type === "status") {
        return {
            type: "status",
            state: String(candidate.state || ""),
        };
    }

    if (type === "error") {
        return {
            type: "error",
            message: String(candidate.message || "WebSocket error"),
        };
    }

    return null;
}

export class RealtimeSocket {
    private readonly wsUrl: string;
    private readonly handlers: EventHandlers;
    private socket: WebSocket | null = null;

    constructor(handlers: EventHandlers, wsUrl: string = process.env.NEXT_PUBLIC_WS_URL || DEFAULT_WS_URL) {
        this.handlers = handlers;
        this.wsUrl = wsUrl;
    }

    connect(): void {
        if (this.socket && (this.socket.readyState === WebSocket.OPEN || this.socket.readyState === WebSocket.CONNECTING)) {
            return;
        }

        const ws = new WebSocket(this.wsUrl);
        ws.binaryType = "arraybuffer";

        ws.onopen = () => {
            this.handlers.onOpen?.();
        };

        ws.onclose = () => {
            this.handlers.onClose?.();
        };

        ws.onerror = () => {
            this.handlers.onError?.("WebSocket connection error");
        };

        ws.onmessage = (event) => {
            if (typeof event.data !== "string") {
                return;
            }
            const message = parseServerMessage(event.data);
            if (!message) {
                return;
            }
            this.handlers.onMessage(message);
        };

        this.socket = ws;
    }

    async waitForOpen(timeoutMs: number = 2500, pollMs: number = 25): Promise<boolean> {
        if (this.isOpen()) {
            return true;
        }

        const startedAt = Date.now();
        return new Promise((resolve) => {
            const interval = window.setInterval(() => {
                if (this.isOpen()) {
                    window.clearInterval(interval);
                    resolve(true);
                    return;
                }

                if (Date.now() - startedAt >= timeoutMs) {
                    window.clearInterval(interval);
                    resolve(false);
                }
            }, pollMs);
        });
    }

    isOpen(): boolean {
        return this.socket?.readyState === WebSocket.OPEN;
    }

    sendCommand(payload: ClientPayload): void {
        if (!this.isOpen() || !this.socket) {
            return;
        }
        this.socket.send(JSON.stringify(payload));
    }

    sendAudio(arrayBuffer: ArrayBuffer): void {
        if (!this.isOpen() || !this.socket || arrayBuffer.byteLength === 0) {
            return;
        }
        this.socket.send(arrayBuffer);
    }

    close(): void {
        if (!this.socket) {
            return;
        }

        if (this.socket.readyState === WebSocket.OPEN) {
            this.sendCommand({ type: "close" });
        }
        this.socket.close();
        this.socket = null;
    }
}

export type PartialMessage = {
    type: "partial";
    stable: string;
    partial: string;
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
        return {
            type: "partial",
            stable: String(candidate.stable || candidate.text || ""),
            partial: String(candidate.partial || ""),
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


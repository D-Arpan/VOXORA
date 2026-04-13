const net = require("net");
const { EventEmitter } = require("events");

const FRAME_AUDIO = 1;
const FRAME_CONTROL = 2;
const FRAME_EVENT = 3;
const HEADER_SIZE = 5;

const PYTHON_HOST = process.env.PYTHON_HOST || "127.0.0.1";
const PYTHON_PORT = Number(process.env.PYTHON_PORT || 6000);

function buildFrame(type, payloadBuffer) {
    const frame = Buffer.allocUnsafe(HEADER_SIZE + payloadBuffer.length);
    frame.writeUInt8(type, 0);
    frame.writeUInt32BE(payloadBuffer.length, 1);
    payloadBuffer.copy(frame, HEADER_SIZE);
    return frame;
}

class PythonSession extends EventEmitter {
    constructor({ host = PYTHON_HOST, port = PYTHON_PORT } = {}) {
        super();
        this.host = host;
        this.port = port;
        this.socket = null;
        this.connected = false;
        this.buffer = Buffer.alloc(0);
        this.destroyed = false;
    }

    connect(timeoutMs = 5000) {
        if (this.connected && this.socket) {
            return Promise.resolve();
        }

        return new Promise((resolve, reject) => {
            const socket = new net.Socket();
            let settled = false;

            const rejectOnce = (error) => {
                if (settled) {
                    return;
                }
                settled = true;
                reject(error);
            };

            const timeout = setTimeout(() => {
                socket.destroy();
                rejectOnce(new Error("Timed out connecting to Python server"));
            }, timeoutMs);

            socket.once("connect", () => {
                clearTimeout(timeout);
                settled = true;

                this.socket = socket;
                this.connected = true;
                this.destroyed = false;
                this.buffer = Buffer.alloc(0);
                socket.setNoDelay(true);

                socket.on("data", (chunk) => this.onData(chunk));
                socket.on("error", (error) => {
                    this.emit("pythonError", error);
                });
                socket.on("close", () => {
                    this.connected = false;
                    this.emit("closed");
                });

                resolve();
            });

            socket.once("error", (error) => {
                clearTimeout(timeout);
                this.connected = false;
                rejectOnce(error);
            });

            socket.connect(this.port, this.host);
        });
    }

    onData(chunk) {
        this.buffer = Buffer.concat([this.buffer, chunk]);

        while (this.buffer.length >= HEADER_SIZE) {
            const type = this.buffer.readUInt8(0);
            const payloadLength = this.buffer.readUInt32BE(1);
            const frameLength = HEADER_SIZE + payloadLength;

            if (this.buffer.length < frameLength) {
                return;
            }

            const payload = this.buffer.subarray(HEADER_SIZE, frameLength);
            this.buffer = this.buffer.subarray(frameLength);
            this.handleFrame(type, payload);
        }
    }

    handleFrame(type, payload) {
        if (type !== FRAME_EVENT && type !== FRAME_CONTROL) {
            return;
        }

        try {
            const event = JSON.parse(payload.toString("utf8"));
            this.emit("event", event);
        } catch (error) {
            this.emit("pythonError", new Error(`Invalid JSON from Python: ${error.message}`));
        }
    }

    sendControl(control) {
        const payload = Buffer.from(JSON.stringify(control), "utf8");
        return this.sendFrame(FRAME_CONTROL, payload);
    }

    sendAudio(audioBuffer) {
        return this.sendFrame(FRAME_AUDIO, audioBuffer);
    }

    sendFrame(type, payload) {
        if (!this.socket || !this.connected || this.destroyed) {
            return false;
        }

        const frame = buildFrame(type, payload);
        return this.socket.write(frame);
    }

    start() {
        this.sendControl({ action: "start" });
    }

    stop() {
        this.sendControl({ action: "stop" });
    }

    close() {
        if (this.destroyed) {
            return;
        }

        if (this.socket) {
            try {
                this.sendControl({ action: "close" });
            } catch {
                // Ignore close-send failures.
            }

            try {
                this.socket.end();
            } catch {
                // Ignore close failures.
            }

            this.socket.destroy();
            this.socket = null;
        }

        this.destroyed = true;
        this.connected = false;
    }
}

function createPythonSession(options) {
    return new PythonSession(options);
}

module.exports = {
    createPythonSession,
};

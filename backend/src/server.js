require("dotenv").config();

const express = require("express");
const cors = require("cors");
const http = require("http");

const { initWebSocket } = require("./websocket/socket");
const healthRoute = require("./routes/health.route");

const app = express();
const server = http.createServer(app);

// Middleware
app.use(cors());
app.use(express.json());

// Routes
app.use("/api/health", healthRoute);

// WebSocket init
initWebSocket(server);

// Start server
const PORT = process.env.PORT || 5000;

server.listen(PORT, () => {
    console.log(`🚀 Server running on http://localhost:${PORT}`);
});
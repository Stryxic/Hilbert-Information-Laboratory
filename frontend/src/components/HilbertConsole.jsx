// =============================================================================
// HilbertConsole.jsx — Reactive Pipeline Console (content-only version)
// =============================================================================
//
// Displays a live backend log feed. Designed for use *inside* a dashboard panel
// (no recursive borders or wrappers). Polls FastAPI endpoints periodically.
//
// Integration:
//   - Included as a plugin component in hilbert_plugins.jsx
// =============================================================================

import React, { useEffect, useState, useRef } from "react";
import axios from "axios";

const API_BASE = "http://127.0.0.1:8000/api/v1";

export default function HilbertConsole() {
  const [logs, setLogs] = useState([
    { time: new Date().toLocaleTimeString(), text: "Hilbert Console initialized.", type: "system" },
  ]);
  const [autoScroll, setAutoScroll] = useState(true);
  const scrollRef = useRef(null);
  const [refreshInterval, setRefreshInterval] = useState(4000);
  const [loading, setLoading] = useState(false);

  // ---------------------------------------------------------------------------
  // Append a formatted log line
  // ---------------------------------------------------------------------------
  const appendLog = (msg, type = "info") => {
    const line = {
      time: new Date().toLocaleTimeString(),
      text: msg,
      type,
    };
    setLogs((prev) => [...prev.slice(-300), line]);
  };

  // ---------------------------------------------------------------------------
  // Poll backend for updates (mocked via get_results)
  // ---------------------------------------------------------------------------
  const pollBackend = async () => {
    try {
      setLoading(true);
      const res = await axios.get(`${API_BASE}/get_results?latest=true`);
      const data = res.data || {};

      const keys = Object.keys(data);
      appendLog(`Fetched results snapshot: ${keys.length} items.`, "system");

      if (keys.includes("hilbert_elements.csv")) appendLog("Element signatures updated.", "success");
      if (keys.includes("signal_stability.csv")) appendLog("Signal stability refreshed.", "success");
      if (keys.includes("molecule_network.json")) appendLog("Network map updated.", "success");
      if (keys.includes("hilbert_summary.pdf")) appendLog("Summary PDF ready.", "system");
    } catch (err) {
      appendLog(`Backend fetch error: ${err.message}`, "error");
    } finally {
      setLoading(false);
    }
  };

  // ---------------------------------------------------------------------------
  // Interval polling
  // ---------------------------------------------------------------------------
  useEffect(() => {
    pollBackend();
    const timer = setInterval(pollBackend, refreshInterval);
    return () => clearInterval(timer);
  }, [refreshInterval]);

  // ---------------------------------------------------------------------------
  // Auto-scroll to bottom
  // ---------------------------------------------------------------------------
  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs, autoScroll]);

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------
  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        display: "flex",
        flexDirection: "column",
        fontFamily: 'ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace',
        fontSize: 12,
        color: "#c9d1d9",
      }}
    >
      {/* Header */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 4,
        }}
      >
        <div style={{ color: "#58a6ff", fontWeight: 600 }}>System Console</div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <label
            style={{
              color: "#8b949e",
              fontSize: 11,
              userSelect: "none",
              cursor: "pointer",
            }}
          >
            <input
              type="checkbox"
              checked={autoScroll}
              onChange={(e) => setAutoScroll(e.target.checked)}
              style={{ marginRight: 4 }}
            />
            Auto-scroll
          </label>
          <button
            onClick={pollBackend}
            disabled={loading}
            style={{
              background: "#161b22",
              color: "#10b981",
              border: "1px solid #30363d",
              borderRadius: 6,
              padding: "3px 8px",
              cursor: loading ? "wait" : "pointer",
              fontSize: 11,
            }}
          >
            {loading ? "Loading…" : "Refresh"}
          </button>
        </div>
      </div>

      {/* Log Output */}
      <div
        ref={scrollRef}
        style={{
          flex: 1,
          overflowY: "auto",
          padding: "0.4rem 0.5rem",
          lineHeight: 1.4,
          background: "#0d1117",
          border: "1px solid #30363d",
          borderRadius: 8,
          marginBottom: 4,
        }}
      >
        {logs.map((l, i) => {
          let color = "#9ba7b1";
          if (l.type === "error") color = "#ef4444";
          else if (l.type === "success") color = "#10b981";
          else if (l.type === "system") color = "#58a6ff";
          return (
            <div key={i} style={{ marginBottom: 2 }}>
              <span style={{ color: "#8b949e", marginRight: 8 }}>
                [{l.time}]
              </span>
              <span style={{ color }}>{l.text}</span>
            </div>
          );
        })}
      </div>

      {/* Footer Controls */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          color: "#8b949e",
          fontSize: 11,
        }}
      >
        <span>Logs: {logs.length}</span>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <button
            onClick={() => setLogs([])}
            style={{
              background: "transparent",
              border: "1px solid #30363d",
              color: "#8b949e",
              borderRadius: 6,
              padding: "2px 6px",
              cursor: "pointer",
              fontSize: 11,
            }}
          >
            Clear
          </button>
          <select
            value={refreshInterval}
            onChange={(e) => setRefreshInterval(parseInt(e.target.value))}
            style={{
              background: "#0d1117",
              border: "1px solid #30363d",
              borderRadius: 6,
              color: "#e6edf3",
              padding: "2px 4px",
              fontSize: 11,
            }}
          >
            <option value={2000}>2s</option>
            <option value={4000}>4s</option>
            <option value={8000}>8s</option>
          </select>
        </div>
      </div>
    </div>
  );
}

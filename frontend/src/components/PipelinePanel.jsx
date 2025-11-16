// ============================================================================
// PipelinePanel.jsx — Hilbert Pipeline Orchestrator UI
// ============================================================================
//
// Visualises the modular pipeline exposed by hilbert_orchestrator:
//
//   - Step list with status per stage
//   - Live log console (from ctx.emit("log", ...))
//   - Artifact list (from ctx.emit("artifact", { path }))
//
// Uses:
//   - GET /api/v1/get_pipeline_plan
//   - WS  /ws/pipeline
//   - (optional) /api/v1/list_artifacts + /api/v1/get_artifact
// ============================================================================

import React, { useEffect, useState, useMemo, useRef } from "react";
import { getPipelinePlan, connectPipelineStream } from "./hilbert_pipeline_api";

const colors = {
  bg: "#020617",
  panelBg: "#0f172a",
  border: "#1e293b",
  text: "#e2e8f0",
  muted: "#94a3b8",
  accent: "#38bdf8",
  ok: "#22c55e",
  warn: "#eab308",
  err: "#ef4444",
};

// Map backend event types to a local status state
const STATUS_ICON = {
  idle: "○",
  running: "●",
  complete: "✔",
  error: "✖",
};

function statusColor(status) {
  switch (status) {
    case "running":
      return colors.accent;
    case "complete":
      return colors.ok;
    case "error":
      return colors.err;
    default:
      return colors.muted;
  }
}

export default function PipelinePanel() {
  const [plan, setPlan] = useState([]); // from get_pipeline_plan
  const [stepStatus, setStepStatus] = useState({}); // id -> { status, duration }
  const [logLines, setLogLines] = useState([]);
  const [artifacts, setArtifacts] = useState([]); // { path, ts }
  const [connected, setConnected] = useState(false);

  const logEndRef = useRef(null);

  // Auto-scroll log
  useEffect(() => {
    if (logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: "smooth", block: "end" });
    }
  }, [logLines]);

  // Initial load: fetch pipeline plan
  useEffect(() => {
    let cancelled = false;

    async function loadPlan() {
      try {
        const steps = await getPipelinePlan();
        if (cancelled) return;

        setPlan(steps);

        // Reset statuses
        const initial = {};
        for (const s of steps) {
          initial[s.id] = { status: "idle", duration: null };
        }
        setStepStatus(initial);
      } catch (err) {
        console.error("[PipelinePanel] Failed to load pipeline plan:", err);
      }
    }

    loadPlan();
    return () => {
      cancelled = true;
    };
  }, []);

  // Connect to WebSocket stream
  useEffect(() => {
    const unsubscribe = connectPipelineStream(handleEvent);
    setConnected(true);

    return () => {
      unsubscribe();
      setConnected(false);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Process incoming events from WebSocket
  const handleEvent = (evt) => {
    if (!evt || typeof evt !== "object") return;
    const { type, step, data, timestamp } = evt;

    // Step-oriented events
    if (type === "step_start") {
      const id = (data && data.id) || step;
      if (!id) return;
      setStepStatus((prev) => ({
        ...prev,
        [id]: {
          status: "running",
          duration: null,
        },
      }));
    } else if (type === "step_complete") {
      const id = (data && data.id) || step;
      if (!id) return;
      setStepStatus((prev) => ({
        ...prev,
        [id]: {
          status: "complete",
          duration: data && typeof data.duration === "number" ? data.duration : null,
        },
      }));
    } else if (type === "step_error") {
      const id = (data && data.id) || step;
      setStepStatus((prev) => ({
        ...prev,
        [id]: {
          status: "error",
          duration: data && typeof data.duration === "number" ? data.duration : null,
          error: data && data.error,
        },
      }));
      if (data && data.error) {
        appendLogLine(
          `[ERROR:${id}] ${data.error}`,
          timestamp || Date.now()
        );
      }
    } else if (type === "log") {
      if (data && data.message) {
        appendLogLine(
          data.message,
          timestamp || Date.now(),
          step || null
        );
      }
    } else if (type === "artifact") {
      if (data && data.path) {
        setArtifacts((prev) => {
          const exists = prev.some((a) => a.path === data.path);
          if (exists) return prev;
          return [
            ...prev,
            { path: data.path, ts: timestamp || Date.now() },
          ];
        });
      }
    }
  };

  const appendLogLine = (message, ts, stepId = null) => {
    setLogLines((prev) => [
      ...prev,
      {
        ts,
        stepId,
        message,
      },
    ]);
  };

  // Derived: plan with status merged in
  const stepsWithStatus = useMemo(() => {
    return plan.map((s) => {
      const st = stepStatus[s.id] || { status: "idle", duration: null };
      return {
        ...s,
        status: st.status,
        duration: st.duration,
        error: st.error,
      };
    });
  }, [plan, stepStatus]);

  return (
    <div
      className="h-full flex flex-col"
      style={{
        background: colors.bg,
        color: colors.text,
        fontSize: 12,
        borderRadius: 12,
        border: `1px solid ${colors.border}`,
        overflow: "hidden",
      }}
    >
      {/* Header */}
      <div
        style={{
          padding: "8px 10px",
          borderBottom: `1px solid ${colors.border}`,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 8,
        }}
      >
        <div>
          <div style={{ fontSize: 13, fontWeight: 600 }}>
            Pipeline Orchestrator
          </div>
          <div style={{ fontSize: 11, color: colors.muted }}>
            Live view of LSA, elements, stability, graph, molecules, condensation, and exports.
          </div>
        </div>

        <div style={{ fontSize: 11, color: colors.muted }}>
          <span
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 6,
            }}
          >
            <span
              style={{
                width: 8,
                height: 8,
                borderRadius: 999,
                background: connected ? colors.ok : colors.err,
                display: "inline-block",
              }}
            />
            {connected ? "stream connected" : "stream offline"}
          </span>
        </div>
      </div>

      {/* Body: steps + right side logs */}
      <div
        style={{
          flex: 1,
          display: "grid",
          gridTemplateColumns: "260px minmax(0, 2fr)",
          minHeight: 0,
        }}
      >
        {/* Left: step list */}
        <div
          style={{
            borderRight: `1px solid ${colors.border}`,
            background: colors.panelBg,
            padding: 8,
            overflowY: "auto",
          }}
        >
          <div style={{ fontSize: 11, fontWeight: 600, marginBottom: 6 }}>
            Pipeline Steps
          </div>
          {stepsWithStatus.length === 0 && (
            <div style={{ fontSize: 11, color: colors.muted }}>
              No steps loaded. Check /api/v1/get_pipeline_plan.
            </div>
          )}
          {stepsWithStatus.map((step) => {
            const col = statusColor(step.status);
            const icon = STATUS_ICON[step.status] || STATUS_ICON.idle;
            return (
              <div
                key={step.id}
                style={{
                  padding: "6px 8px",
                  borderRadius: 8,
                  border: `1px solid ${colors.border}`,
                  marginBottom: 6,
                  background:
                    step.status === "running" ? "#020617" : "transparent",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    gap: 8,
                  }}
                >
                  <div style={{ fontWeight: 600, fontSize: 11 }}>
                    <span style={{ color: col, marginRight: 6 }}>{icon}</span>
                    {step.title}
                  </div>
                  {typeof step.duration === "number" && (
                    <div
                      style={{
                        fontSize: 10,
                        color: colors.muted,
                        whiteSpace: "nowrap",
                      }}
                    >
                      {step.duration.toFixed(2)}s
                    </div>
                  )}
                </div>
                <div
                  style={{
                    fontSize: 10,
                    color: colors.muted,
                    marginTop: 2,
                  }}
                >
                  {step.description}
                </div>
                {step.error && (
                  <div
                    style={{
                      fontSize: 10,
                      color: colors.err,
                      marginTop: 4,
                    }}
                  >
                    {step.error}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Right: logs + artifacts */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            background: "#020617",
            minHeight: 0,
          }}
        >
          {/* Logs */}
          <div
            style={{
              flex: 1,
              minHeight: 0,
              borderBottom: `1px solid ${colors.border}`,
              display: "flex",
              flexDirection: "column",
            }}
          >
            <div
              style={{
                padding: "6px 8px",
                borderBottom: `1px solid ${colors.border}`,
                fontSize: 11,
                fontWeight: 600,
              }}
            >
              Log
            </div>
            <div
              style={{
                flex: 1,
                overflowY: "auto",
                fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace",
                fontSize: 11,
                padding: 8,
              }}
            >
              {logLines.length === 0 && (
                <div style={{ color: colors.muted }}>
                  Waiting for pipeline output...
                </div>
              )}
              {logLines.map((line, idx) => (
                <div key={idx} style={{ whiteSpace: "pre-wrap" }}>
                  <span style={{ color: colors.muted, marginRight: 6 }}>
                    [{new Date(line.ts).toLocaleTimeString()}]
                  </span>
                  {line.stepId && (
                    <span style={{ color: colors.accent, marginRight: 4 }}>
                      ({line.stepId})
                    </span>
                  )}
                  <span>{line.message}</span>
                </div>
              ))}
              <div ref={logEndRef} />
            </div>
          </div>

          {/* Artifacts */}
          <div
            style={{
              height: 120,
              padding: 8,
              background: colors.panelBg,
            }}
          >
            <div
              style={{
                fontSize: 11,
                fontWeight: 600,
                marginBottom: 4,
              }}
            >
              Artifacts
            </div>
            <div
              style={{
                fontSize: 10,
                color: colors.muted,
                marginBottom: 4,
              }}
            >
              Generated files from the latest run (paths emitted via ctx.emit("artifact", ...)).
            </div>

            <div
              style={{
                display: "flex",
                flexWrap: "wrap",
                gap: 6,
                maxHeight: 70,
                overflowY: "auto",
              }}
            >
              {artifacts.length === 0 && (
                <div style={{ fontSize: 10, color: colors.muted }}>
                  No artifacts reported yet.
                </div>
              )}
              {artifacts.map((a) => (
                <a
                  key={a.path}
                  href={a.path}
                  target="_blank"
                  rel="noreferrer"
                  style={{
                    fontSize: 10,
                    padding: "3px 6px",
                    borderRadius: 999,
                    border: `1px solid ${colors.border}`,
                    textDecoration: "none",
                    color: colors.accent,
                    background: "#020617",
                  }}
                >
                  {shortenPath(a.path)}
                </a>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function shortenPath(p) {
  if (!p) return "";
  const parts = p.split(/[\\/]/);
  if (parts.length <= 2) return p;
  return `${parts[parts.length - 2]}/${parts[parts.length - 1]}`;
}

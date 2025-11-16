// ============================================================================
// PipelineStepViewer.jsx — Detailed view for a single pipeline step
// ============================================================================

import React, { useMemo } from "react";

export default function PipelineStepViewer({ events, stepId }) {
  const logs = useMemo(
    () =>
      events
        .filter((e) => e.step === stepId && e.type === "log")
        .map((e) => e.data.message),
    [events, stepId]
  );

  const meta = useMemo(() => {
    const m = { started: null, completed: null, error: null, duration: null };
    for (const e of events) {
      if (e.step !== stepId) continue;
      if (e.type === "step_start") m.started = e.timestamp;
      if (e.type === "step_complete") {
        m.completed = e.timestamp;
        m.duration = e.data?.duration;
      }
      if (e.type === "step_error") {
        m.error = e.data?.error || "Unknown error";
        m.duration = e.data?.duration;
      }
    }
    return m;
  }, [events, stepId]);

  return (
    <div className="h-full flex flex-col bg-[#0d1117] border border-[#30363d] rounded-xl">
      {/* HEADER */}
      <div className="px-3 py-2 border-b border-[#30363d] bg-[#161b22] flex justify-between items-center">
        <div className="text-sm font-semibold">{stepId}</div>
        {meta.duration && (
          <div className="text-xs text-[#8b949e]">
            Duration: {meta.duration.toFixed(2)}s
          </div>
        )}
      </div>

      {/* BODY */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2 text-xs">
        {meta.error && (
          <div className="text-red-400 font-medium">Error: {meta.error}</div>
        )}

        {!logs.length && <div className="text-[#8b949e]">No logs recorded.</div>}

        {logs.map((l, i) => (
          <div key={i} className="whitespace-pre-wrap">
            • {l}
          </div>
        ))}
      </div>
    </div>
  );
}

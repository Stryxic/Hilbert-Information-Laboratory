// ============================================================================
// PipelineProgressPanel.jsx — Visual step-by-step pipeline progress
// ============================================================================

import React, { useMemo } from "react";

export default function PipelineProgressPanel({ events }) {
  const stepMap = useMemo(() => {
    const map = {};
    for (const evt of events) {
      const id = evt.step;
      if (!id) continue;

      if (!map[id]) {
        map[id] = {
          id,
          started: null,
          completed: null,
          error: null,
          duration: null,
          logs: [],
        };
      }

      if (evt.type === "step_start") {
        map[id].started = evt.timestamp;
      } else if (evt.type === "step_complete") {
        map[id].completed = evt.timestamp;
        map[id].duration = evt.data?.duration || null;
      } else if (evt.type === "step_error") {
        map[id].error = evt.data?.error || "Unknown error";
        map[id].duration = evt.data?.duration || null;
      } else if (evt.type === "log") {
        map[id].logs.push(evt.data?.message || "");
      }
    }
    return map;
  }, [events]);

  const ordered = useMemo(() => {
    return events
      .filter((e) => e.type === "step_start")
      .map((e) => stepMap[e.step])
      .filter(Boolean);
  }, [events, stepMap]);

  return (
    <div className="flex flex-col bg-[#0d1117] text-[#e6edf3] border border-[#30363d] rounded-xl p-3 gap-3">
      <div className="text-sm font-semibold text-[#8b949e]">Pipeline Progress</div>

      <div className="flex flex-col gap-2">
        {ordered.map((step) => {
          const color = step.error
            ? "#ef4444" // error red
            : step.completed
            ? "#10b981" // success green
            : step.started
            ? "#58a6ff" // running blue
            : "#8b949e"; // pending

          return (
            <div
              key={step.id}
              className="p-2 rounded-lg border border-[#30363d] flex flex-col"
              style={{ background: "#11151f" }}
            >
              <div className="flex justify-between items-center">
                <div className="font-semibold">{step.id}</div>
                <div style={{ color }}>{step.error ? "Error" : step.completed ? "Done" : "Running…"}</div>
              </div>

              {step.duration && (
                <div className="text-xs text-[#8b949e]">
                  Duration: {step.duration.toFixed(2)}s
                </div>
              )}

              {step.error && (
                <div className="text-xs text-red-400 mt-1">{step.error}</div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

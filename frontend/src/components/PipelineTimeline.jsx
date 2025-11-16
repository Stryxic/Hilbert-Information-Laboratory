// ============================================================================
// PipelineTimeline.jsx — Horizontal timeline for pipeline steps
// ============================================================================

import React, { useMemo } from "react";

export default function PipelineTimeline({ events }) {
  const stepData = useMemo(() => {
    const data = {};
    for (const e of events) {
      const id = e.step;
      if (!id) continue;
      data[id] ||= { start: null, end: null };

      if (e.type === "step_start") data[id].start = e.timestamp;
      if (e.type === "step_complete" || e.type === "step_error")
        data[id].end = e.timestamp;
    }
    return data;
  }, [events]);

  const items = Object.entries(stepData).map(([id, info]) => ({
    id,
    start: info.start,
    end: info.end || info.start + 1,
  }));

  const t0 = Math.min(...items.map((i) => i.start));
  const t1 = Math.max(...items.map((i) => i.end));
  const span = t1 - t0;

  return (
    <div className="bg-[#0d1117] border border-[#30363d] rounded-xl p-3 text-xs">
      <div className="font-semibold text-[#8b949e] mb-2">
        Pipeline Timeline
      </div>

      <div className="relative h-40 bg-[#11151f] rounded">
        {items.map((i) => {
          const left = ((i.start - t0) / span) * 100;
          const width = ((i.end - i.start) / span) * 100;

          return (
            <div
              key={i.id}
              title={i.id}
              style={{
                position: "absolute",
                left: `${left}%`,
                width: `${width}%`,
                top: `${Math.random() * 70 + 10}px`,
                height: "10px",
                background: "#58a6ff",
                borderRadius: "6px",
              }}
            >
              <span className="absolute left-0 -top-5 text-[#8b949e]">
                {i.id}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

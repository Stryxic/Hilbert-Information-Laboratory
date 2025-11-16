// ============================================================================
// PipelineEventStream.jsx — Chronological event timeline
// ============================================================================

import React from "react";

export default function PipelineEventStream({ events }) {
  return (
    <div className="h-full flex flex-col bg-[#0d1117] border border-[#30363d] rounded-xl">
      <div className="px-3 py-2 border-b border-[#30363d] bg-[#161b22]">
        <div className="text-sm font-semibold">Pipeline Event Stream</div>
      </div>

      <div className="flex-1 overflow-y-auto p-3 text-xs space-y-2">
        {events.map((evt, i) => (
          <div key={i} className="flex gap-2">
            <span className="text-[#8b949e] w-24">
              {new Date(evt.timestamp * 1000).toLocaleTimeString()}
            </span>
            <span className="font-semibold">{evt.type}</span>
            <span className="text-[#8b949e]">{evt.step || ""}</span>
            {evt.data?.message && <span>{evt.data.message}</span>}
          </div>
        ))}
      </div>
    </div>
  );
}

// ============================================================================
// PipelineRunView.jsx — Full Pipeline UI
// ============================================================================

import React, { useState } from "react";
import PipelineProgressPanel from "./PipelineProgressPanel";
import PipelineStepViewer from "./PipelineStepViewer";
import PipelineEventStream from "./PipelineEventStream";
import { usePipelineRun } from "./usePipelineRun";

export default function PipelineRunView() {
  const { runState, runPipeline, events, log, errors } = usePipelineRun();
  const [selectedStep, setSelectedStep] = useState(null);

  const steps = Array.from(
    new Set(events.filter((e) => e.step).map((e) => e.step))
  );

  const start = () => {
    runPipeline("uploaded_corpus", "results/hilbert_run");
  };

  return (
    <div className="h-full w-full grid grid-cols-12 gap-3 p-3 bg-[#010409] text-[#e6edf3]">

      {/* LEFT SIDEBAR: PROGRESS */}
      <div className="col-span-3">
        <PipelineProgressPanel events={events} />
        <button
          onClick={start}
          className="mt-3 w-full py-2 bg-[#238636] hover:bg-[#2ea043] rounded-lg text-sm font-semibold"
        >
          Run Pipeline
        </button>
        {errors.length > 0 && (
          <div className="mt-3 text-red-400 text-xs">
            {errors.map((e, i) => (
              <div key={i}>{e}</div>
            ))}
          </div>
        )}
      </div>

      {/* MIDDLE: STEP VIEWER */}
      <div className="col-span-5 flex flex-col gap-3">
        <div className="p-2 bg-[#0d1117] border border-[#30363d] rounded-xl">
          <div className="text-xs mb-2">Select Step</div>
          <select
            className="w-full bg-[#161b22] border border-[#30363d] rounded p-1 text-sm"
            onChange={(e) => setSelectedStep(e.target.value)}
          >
            <option value="">Choose…</option>
            {steps.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </div>

        {selectedStep ? (
          <PipelineStepViewer events={events} stepId={selectedStep} />
        ) : (
          <div className="flex-1 bg-[#0d1117] border border-[#30363d] rounded-xl flex items-center justify-center text-[#8b949e]">
            Select a step to view details.
          </div>
        )}
      </div>

      {/* RIGHT: EVENT STREAM */}
      <div className="col-span-4">
        <PipelineEventStream events={events} />
      </div>
    </div>
  );
}

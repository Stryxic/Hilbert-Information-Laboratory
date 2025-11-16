// ============================================================================
// StepInspector.jsx — Shows inputs, outputs, and generated artifacts per step
// ============================================================================

import React from "react";

export default function StepInspector({ events, outputDir }) {
  const stepFiles = {}; // step → generated files
  const stepLogs = {}; // step → logs

  for (const evt of events) {
    if (!evt.step) continue;
    if (evt.type === "log") {
      stepLogs[evt.step] ||= [];
      stepLogs[evt.step].push(evt.data.message);
    }
    if (evt.type === "artifact") {
      stepFiles[evt.step] ||= [];
      stepFiles[evt.step].push(evt.data.path);
    }
  }

  return (
    <div className="bg-[#0d1117] border border-[#30363d] rounded-xl p-3 text-xs">
      <div className="font-semibold text-[#8b949e] mb-3">Step Inspector</div>

      {Object.keys(stepFiles).map((step) => (
        <div key={step} className="mb-4">
          <div className="font-semibold mb-1">{step}</div>

          {stepFiles[step].map((f) => (
            <div key={f} className="ml-2 text-[#8b949e]">
              → {f}
            </div>
          ))}

          {stepLogs[step] && (
            <details className="ml-2 mt-1">
              <summary className="cursor-pointer text-[#58a6ff]">Logs</summary>
              <pre className="whitespace-pre-wrap p-2">
                {stepLogs[step].join("\n")}
              </pre>
            </details>
          )}
        </div>
      ))}
    </div>
  );
}

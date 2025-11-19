import React from "react";
import DocumentsPanel from "../DocumentsPanel";

export default function InputPanel({
  results,
  appendLog,
  onRunPipeline,
  onSelectElement,
}) {
  return (
    <div className="h-full border border-[#30363d] rounded-xl overflow-hidden">
      <DocumentsPanel
        results={results}
        onRunHilbert={() => onRunPipeline("uploaded_corpus", "results/hilbert_run")}
        appendLog={appendLog}
        onSelectElement={onSelectElement}
      />
    </div>
  );
}

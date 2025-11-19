import React, { useState } from "react";

export default function FunctionPanel({
  results,
  onRunPipeline,
  appendLog,
}) {
  const [corpusDir, setCorpusDir] = useState("uploaded_corpus");
  const [outputDir, setOutputDir] = useState("results/hilbert_run");

  return (
    <div className="h-full bg-[#0d1117] border border-[#30363d] rounded-xl p-3 flex flex-col text-xs">
      {/* Header */}
      <div className="font-semibold mb-2">Functions</div>

      {/* Corpus + output directories */}
      <div className="space-y-1 mb-3">
        <input
          value={corpusDir}
          onChange={(e) => setCorpusDir(e.target.value)}
          className="w-full bg-[#161b22] border border-[#30363d] rounded p-1"
        />
        <input
          value={outputDir}
          onChange={(e) => setOutputDir(e.target.value)}
          className="w-full bg-[#161b22] border border-[#30363d] rounded p-1"
        />
      </div>

      {/* Actions */}
      <button
        className="bg-[#238636] p-2 rounded text-xs mb-2 hover:bg-[#2ea043]"
        onClick={() => onRunPipeline(corpusDir, outputDir)}
      >
        Run Full Hilbert Pipeline
      </button>

      <div className="mt-3 font-semibold">Individual Stages</div>
      <div className="text-[10px] text-[#8b949e]">Coming soon: stage-level execution</div>

      {/* Status */}
      <div className="mt-auto text-[10px] opacity-60">
        {results ? "Results loaded" : "No results yet"}
      </div>
    </div>
  );
}

import React, { useState } from "react";

import InputPanel from "./components/InputPanel";
import FunctionPanel from "./components/FunctionPanel";
import OutputPanel from "./components/OutputPanel";

export default function HilbertApp() {
  const [results, setResults] = useState(null);
  const [activeElement, setActiveElement] = useState(null);
  const [log, setLog] = useState([]);

  const appendLog = (msg) =>
    setLog((L) => [...L, `[${new Date().toLocaleTimeString()}] ${msg}`]);

  const handlePipelineRun = async (corpusDir, outputDir) => {
    appendLog("Running Hilbert pipeline...");
    const res = await fetch("http://127.0.0.1:8000/api/v1/analyze_corpus", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        corpus_dir: corpusDir,
        output_dir: outputDir,
      }),
    }).then((r) => r.json());

    setResults(res);
    appendLog("Pipeline completed.");
    return res;
  };

  return (
    <div className="h-screen w-screen grid grid-cols-12 gap-2 bg-[#0d1117] p-2 text-[#e6edf3]">
      {/* Input panel */}
      <div className="col-span-3 h-full">
        <InputPanel
          results={results}
          onSelectElement={setActiveElement}
          onRunPipeline={handlePipelineRun}
          appendLog={appendLog}
        />
      </div>

      {/* Function panel */}
      <div className="col-span-2 h-full">
        <FunctionPanel
          results={results}
          appendLog={appendLog}
          onRunPipeline={handlePipelineRun}
        />
      </div>

      {/* Output panel */}
      <div className="col-span-7 h-full">
        <OutputPanel
          results={results}
          activeElement={activeElement}
          onSelectElement={setActiveElement}
          appendLog={appendLog}
          logs={log}
        />
      </div>
    </div>
  );
}

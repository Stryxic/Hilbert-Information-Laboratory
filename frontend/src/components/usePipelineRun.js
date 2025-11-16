// ============================================================================
// usePipelineRun.js — Hook for running Hilbert pipeline + receiving events
// ============================================================================

import { useState, useEffect, useCallback } from "react";

export function usePipelineRun() {
  const [runState, setRunState] = useState("idle"); // idle | running | complete | error
  const [events, setEvents] = useState([]);
  const [log, setLog] = useState([]);
  const [errors, setErrors] = useState([]);
  const [result, setResult] = useState(null);

  const runPipeline = useCallback(async (corpusDir, outputDir) => {
    setRunState("running");
    setEvents([]);
    setLog([]);
    setErrors([]);

    try {
      const res = await fetch("/api/v1/run_pipeline", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ corpus_dir: corpusDir, output_dir: outputDir }),
      });

      const data = await res.json();
      setRunState(data.success ? "complete" : "error");
      setEvents(data.events || []);
      setLog(data.log || []);
      setErrors(data.errors || []);
      setResult(data);

    } catch (err) {
      setRunState("error");
      setErrors([String(err)]);
    }
  }, []);

  return {
    runState,
    events,
    log,
    errors,
    result,
    runPipeline,
  };
}

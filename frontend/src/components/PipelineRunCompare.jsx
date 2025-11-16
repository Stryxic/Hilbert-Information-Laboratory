// ============================================================================
// PipelineRunCompare.jsx — Compare two pipeline runs
// ============================================================================

import React, { useEffect, useState } from "react";

export default function PipelineRunCompare() {
  const [runs, setRuns] = useState([]);
  const [runA, setRunA] = useState(null);
  const [runB, setRunB] = useState(null);

  useEffect(() => {
    fetch("/api/v1/list_runs")
      .then((r) => r.json())
      .then((d) => setRuns(d.runs || []));
  }, []);

  const [eventsA, setEventsA] = useState([]);
  const [eventsB, setEventsB] = useState([]);

  const loadRun = async (run, setEvents) => {
    const res = await fetch(`/api/v1/get_run?run=${run}`);
    const data = await res.json();
    setEvents(data.events || []);
  };

  return (
    <div className="h-full p-3 bg-[#010409] text-[#e6edf3] grid grid-cols-2 gap-3">
      {/* SELECTORS */}
      <div>
        <select
          className="w-full bg-[#161b22] border border-[#30363d] rounded p-1 text-sm"
          onChange={(e) => {
            setRunA(e.target.value);
            loadRun(e.target.value, setEventsA);
          }}
        >
          <option>Select run A</option>
          {runs.map((r) => (
            <option key={r} value={r}>
              {r}
            </option>
          ))}
        </select>
      </div>

      <div>
        <select
          className="w-full bg-[#161b22] border border-[#30363d] rounded p-1 text-sm"
          onChange={(e) => {
            setRunB(e.target.value);
            loadRun(e.target.value, setEventsB);
          }}
        >
          <option>Select run B</option>
          {runs.map((r) => (
            <option key={r} value={r}>
              {r}
            </option>
          ))}
        </select>
      </div>

      {/* TWO TIMELINES SIDE BY SIDE */}
      <div className="border border-[#30363d] rounded-xl">
        <PipelineTimeline events={eventsA} />
      </div>

      <div className="border border-[#30363d] rounded-xl">
        <PipelineTimeline events={eventsB} />
      </div>
    </div>
  );
}

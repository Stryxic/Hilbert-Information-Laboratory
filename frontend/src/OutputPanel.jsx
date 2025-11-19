import React, { useState } from "react";

import PeriodicTablePanel from "./components/PeriodicTablePanel";
import GraphSelectorPanel from "./components/GraphSelectorPanel";
import TimelinePanel from "./components/TimelinePanel";
import ThesisAdvisorPanel from "./components/ThesisAdvisorPanel";
import ReportPanel from "./components/ReportPanel";

export default function OutputPanel({
  results,
  activeElement,
  onSelectElement,
  appendLog,
  logs,
}) {
  const [tab, setTab] = useState("periodic");

  const tabs = [
    ["periodic", "Periodic Table"],
    ["graph", "Graph"],
    ["timeline", "Timeline"],
    ["thesis", "Thesis Advisor"],
    ["report", "Report"],
    ["logs", "Logs"],
  ];

  return (
    <div className="h-full flex flex-col border border-[#30363d] rounded-xl bg-[#0d1117]">
      {/* Tab Strip */}
      <div className="flex-shrink-0 flex gap-2 px-3 py-2 border-b border-[#30363d] text-xs">
        {tabs.map(([id, label]) => (
          <button
            key={id}
            onClick={() => setTab(id)}
            className={`px-2 py-1 rounded ${
              tab === id
                ? "bg-[#161b22] text-[#58a6ff]"
                : "text-[#8b949e]"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Main content */}
      <div className="flex-1 overflow-hidden">
        {tab === "periodic" && (
          <PeriodicTablePanel
            results={results}
            activeElement={activeElement}
            onSelectElement={onSelectElement}
            appendLog={appendLog}
          />
        )}

        {tab === "graph" && (
          <GraphSelectorPanel
            results={results}
            activeElement={activeElement}
            onSelectElement={onSelectElement}
            appendLog={appendLog}
          />
        )}

        {tab === "timeline" && (
          <TimelinePanel results={results} />
        )}

        {tab === "thesis" && (
          <ThesisAdvisorPanel
            results={results}
            activeElement={activeElement}
            appendLog={appendLog}
          />
        )}

        {tab === "report" && (
          <ReportPanel results={results} />
        )}

        {tab === "logs" && (
          <div className="p-3 text-[10px] overflow-auto">
            {logs.map((l, i) => (
              <div key={i}>{l}</div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

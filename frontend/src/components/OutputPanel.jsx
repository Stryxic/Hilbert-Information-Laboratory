// ============================================================================
// OutputPanel.jsx - Unified Hilbert Output Multiplexer
// ----------------------------------------------------------------------------
// Combines multiple analytic views into one panel with an internal nav:
//   - Graph and Relations
//   - Molecule Viewer
//   - Periodic Table
//   - Timeline
//   - Chronology
//   - Persistence
//   - Thesis Advisor
//   - Report
//   - Console
// ============================================================================

import React, { useMemo, useState } from "react";

import GraphSelectorPanel from "./GraphSelectorPanel";
import MoleculeViewer from "./MoleculeViewer";
import PeriodicTablePanel from "./PeriodicTablePanel";
import TimelinePanel from "./TimelinePanel";
import PersistencePanel from "./PersistencePanel";
import ReportPanel from "./ReportPanel";
import ThesisAdvisorPanel from "./ThesisAdvisorPanel";
import HilbertConsole from "./HilbertConsole";
import ChronologyView from "./ChronologyView";

const palette = {
  bg: "#0d1117",
  panelBg: "#161b22",
  border: "#30363d",
  text: "#e6edf3",
  muted: "#8b949e",
  accent: "#58a6ff",
};

const VIEWS = [
  {
    id: "graph",
    label: "Graph",
    description: "Hilbert informational graph layout.",
  },
  {
    id: "molecules",
    label: "Molecules",
    description: "Compound and molecule viewer.",
  },
  {
    id: "periodic",
    label: "Periodic Table",
    description: "Informational periodic table.",
  },
  {
    id: "timeline",
    label: "Timeline",
    description: "Scenario timeline and alignment.",
  },
  {
    id: "chronology",
    label: "Chronology",
    description: "Element prominence over time by centrality and core score.",
  },
  {
    id: "persistence",
    label: "Persistence",
    description: "Signal stability and persistence fields.",
  },
  {
    id: "report",
    label: "Report",
    description: "Auto-generated run report and figures.",
  },
  {
    id: "advisor",
    label: "Thesis Advisor",
    description: "Advisor hints and commentary.",
  },
  {
    id: "console",
    label: "Console",
    description: "Low-level console and logs.",
  },
];

// ============================================================================
// MAIN
// ============================================================================

export default function OutputPanel({
  // From Dashboard sharedProps
  results,
  timeline: timelineProp, // Dashboard currently passes `timeline`
  timelineEvents: timelineEventsProp, // Optional explicit override
  timelineCursor: timelineCursorProp, // Global cursor from Dashboard (optional)
  onTimelineChange, // Global setter from Dashboard (optional)
  currentTimelineEntry, // Optional explicit override

  activeElement,
  onSelectElement,
  onRunHilbert,
  appendLog,
}) {
  const [activeView, setActiveView] = useState("graph");

  // ---------------------------------------------------------------------------
  // Timeline normalisation shared by periodic, timeline and chronology
  // ---------------------------------------------------------------------------
  const normalizedTimeline = useMemo(() => {
    // Prefer explicit timelineEventsProp if present
    if (Array.isArray(timelineEventsProp) && timelineEventsProp.length) {
      return timelineEventsProp;
    }
    // Then fall back to `timeline` prop from Dashboard
    if (Array.isArray(timelineProp) && timelineProp.length) {
      return timelineProp;
    }
    // Finally, if a panel is mounted directly with results only, use those
    const t = results?.timeline?.timeline || results?.timeline || [];
    return Array.isArray(t) ? t : [];
  }, [timelineEventsProp, timelineProp, results]);

  // Local cursor only used when parent does not control it
  const [internalCursor, setInternalCursor] = useState(0);

  const timelineDates = useMemo(
    () =>
      normalizedTimeline
        .map((ev) => ev.date || ev.timestamp || null)
        .filter(Boolean),
    [normalizedTimeline]
  );

  const maxIndex = timelineDates.length ? timelineDates.length - 1 : 0;

  const hasExternalCursor = typeof timelineCursorProp === "number";

  const effectiveCursorRaw = hasExternalCursor
    ? timelineCursorProp
    : internalCursor;

  const effectiveCursor =
    maxIndex > 0
      ? Math.min(Math.max(effectiveCursorRaw, 0), maxIndex)
      : 0;

  const handleTimelineChange = (idx) => {
    if (onTimelineChange) {
      onTimelineChange(idx);
    }
    if (!hasExternalCursor) {
      setInternalCursor(idx);
    }
  };

  const derivedTimelineEntry =
    normalizedTimeline.length > 0
      ? normalizedTimeline[Math.min(effectiveCursor, normalizedTimeline.length - 1)]
      : null;

  const effectiveTimelineEntry = currentTimelineEntry || derivedTimelineEntry;

  // -------------------------------------------------------------------------
  // Render current view
  // -------------------------------------------------------------------------
  const renderView = () => {
    switch (activeView) {
      case "graph":
        return (
          <GraphSelectorPanel
            results={results}
            activeElement={activeElement}
            onSelectElement={onSelectElement}
            appendLog={appendLog}
          />
        );

      case "molecules":
        return (
          <MoleculeViewer
            results={results}
            activeElement={activeElement}
            onSelectElement={onSelectElement}
            appendLog={appendLog}
          />
        );

      case "periodic":
        return (
          <PeriodicTablePanel
            results={results}
            activeElement={activeElement}
            onSelectElement={onSelectElement}
            appendLog={appendLog}
            timeline={normalizedTimeline}
            timelineDates={timelineDates}
            timelineCursor={effectiveCursor}
            onTimelineChange={handleTimelineChange}
          />
        );

      case "timeline":
        return (
          <TimelinePanel
            results={results}
            timelineEvents={normalizedTimeline}
            currentTimelineEntry={effectiveTimelineEntry}
            appendLog={appendLog}
          />
        );

      case "chronology":
        return (
          <ChronologyView
            results={results}
            timelineEvents={normalizedTimeline}
            onSelectElement={onSelectElement}
            appendLog={appendLog}
          />
        );

      case "persistence":
        return <PersistencePanel results={results} />;

      case "report":
        return <ReportPanel results={results} onRunHilbert={onRunHilbert} />;

      case "advisor":
        return <ThesisAdvisorPanel results={results} />;

      case "console":
        return <HilbertConsole />;

      default:
        return (
          <div
            style={{
              padding: 16,
              fontSize: 12,
              color: palette.muted,
            }}
          >
            No view selected.
          </div>
        );
    }
  };

  const activeMeta = VIEWS.find((v) => v.id === activeView) || VIEWS[0];

  // ========================================================================
  // RENDER
  // ========================================================================
  return (
    <div
      style={{
        display: "flex",
        height: "100%",
        background: palette.bg,
        color: palette.text,
        borderRadius: 12,
        border: `1px solid ${palette.border}`,
        overflow: "hidden",
        fontSize: 12,
      }}
    >
      {/* LEFT NAV */}
      <div
        style={{
          width: 170,
          background: "#050814",
          borderRight: `1px solid ${palette.border}`,
          display: "flex",
          flexDirection: "column",
          padding: 8,
          gap: 4,
          flexShrink: 0,
        }}
      >
        <div
          style={{
            fontSize: 11,
            fontWeight: 600,
            color: palette.accent,
            marginBottom: 4,
          }}
        >
          Outputs
        </div>

        {VIEWS.map((view) => {
          const selected = activeView === view.id;
          return (
            <button
              key={view.id}
              onClick={() => setActiveView(view.id)}
              style={{
                textAlign: "left",
                padding: "6px 8px",
                borderRadius: 8,
                border: `1px solid ${
                  selected ? palette.accent : palette.border
                }`,
                background: selected ? "#0d1117" : "transparent",
                color: selected ? palette.accent : palette.muted,
                fontSize: 11,
                cursor: "pointer",
              }}
            >
              {view.label}
            </button>
          );
        })}
      </div>

      {/* MAIN CONTENT */}
      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          background: palette.panelBg,
        }}
      >
        {/* Header strip for current subview */}
        <div
          style={{
            padding: "8px 10px",
            borderBottom: `1px solid ${palette.border}`,
            display: "flex",
            justifyContent: "space-between",
            alignItems: "baseline",
            gap: 8,
          }}
        >
          <div>
            <div
              style={{
                fontSize: 13,
                fontWeight: 600,
                color: palette.text,
              }}
            >
              {activeMeta.label}
            </div>
            <div
              style={{
                fontSize: 10,
                color: palette.muted,
              }}
            >
              {activeMeta.description}
            </div>
          </div>

          {results?.meta && (
            <div
              style={{
                fontSize: 10,
                color: palette.muted,
                textAlign: "right",
              }}
            >
              <div>{results.meta.run}</div>
              <div>{results.meta.generated_at}</div>
            </div>
          )}
        </div>

        {/* Body: actual child panel */}
        <div
          style={{
            flex: 1,
            minHeight: 0,
          }}
        >
          {renderView()}
        </div>
      </div>
    </div>
  );
}

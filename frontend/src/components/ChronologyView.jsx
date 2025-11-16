// ============================================================================
// ChronologyView.jsx - Element prominence over time (slider view)
// ----------------------------------------------------------------------------
// Uses timeline.json + Hilbert element stats to show which informational
// elements are most central on each date. A slider scrubs through dates.
// Selecting an element notifies Dashboard so Periodic Table highlights it.
// ============================================================================

import React, { useMemo, useState } from "react";

const palette = {
  bg: "#0d1117",
  panelBg: "#161b22",
  border: "#30363d",
  text: "#e6edf3",
  muted: "#8b949e",
  accent: "#58a6ff",
  info: "#10b981",
  mis: "#f59e0b",
  dis: "#ef4444",
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function coreScore(row) {
  if (!row) return 0;
  const tf = Number(row.tf ?? 0);
  const df = Number(row.df ?? 0);
  return Math.log1p(tf) + 0.5 * Math.log1p(df);
}

function normaliseClassification(raw) {
  const c = String(raw || "").toLowerCase();

  if (c.startsWith("dis")) return "Disinformation";
  if (c.startsWith("mis")) return "Misinformation";
  if (c.startsWith("pri")) return "Information (Primary source)";
  if (c.startsWith("cor")) return "Information (Corrective)";
  if (c.startsWith("inf")) return "Information";

  return "Information";
}

function classificationColor(cls) {
  const c = String(cls || "").toLowerCase();
  if (c.startsWith("dis")) return palette.dis;
  if (c.startsWith("mis")) return palette.mis;
  return palette.info;
}

// ---------------------------------------------------------------------------
// Reusable "no data" block
// ---------------------------------------------------------------------------
function noData(msg) {
  return (
    <div
      style={{
        padding: 16,
        fontSize: 12,
        color: palette.muted,
        height: "100%",
        background: palette.panelBg,
      }}
    >
      {msg}
    </div>
  );
}

// ============================================================================
// Component
// ============================================================================

export default function ChronologyView({
  results,
  timelineEvents,
  activeElement,
  onSelectElement,
  appendLog,
}) {
  // -------------------------------------------------------------------------
  // 1) Raw inputs
  // -------------------------------------------------------------------------
  const elements = results?.elements?.elements || [];

  const rawTimeline = useMemo(() => {
    if (Array.isArray(timelineEvents) && timelineEvents.length) {
      return timelineEvents;
    }
    const t = results?.timeline?.timeline || results?.timeline || [];
    return Array.isArray(t) ? t : [];
  }, [timelineEvents, results]);

  const hasElements = elements.length > 0;
  const hasTimeline = rawTimeline.length > 0;

  // -------------------------------------------------------------------------
  // 2) Build reverse lookup: document -> elements in that document
  // -------------------------------------------------------------------------
  const elementsByDoc = useMemo(() => {
    const map = new Map();
    for (const e of elements) {
      const doc = e.doc || e.filename || e.source;
      if (!doc) continue;
      if (!map.has(doc)) map.set(doc, []);
      map.get(doc).push(e);
    }
    return map;
  }, [elements]);

  // -------------------------------------------------------------------------
  // 3) Build global centrality ranking
  // -------------------------------------------------------------------------
  const { globalRankById } = useMemo(() => {
    const rankings = [];

    elements.forEach((row, i) => {
      const baseId = row.element || row.token || `E${i + 1}`;
      const id = String(baseId);
      const key = id.toLowerCase();
      const label = row.label || row.token || row.element || id;

      const cent = Number(row.centrality ?? 0);
      const cs = coreScore(row);

      rankings.push({
        id,
        key,
        label,
        centrality: cent,
        coreScore: cs,
      });
    });

    rankings.sort(
      (a, b) => b.centrality - a.centrality || b.coreScore - a.coreScore
    );

    const rankMap = new Map();
    rankings.forEach((r, idx) => rankMap.set(r.key, idx + 1));

    return { globalRankById: rankMap };
  }, [elements]);

  // -------------------------------------------------------------------------
  // 4) Build chronology by date
  // -------------------------------------------------------------------------
  const chronology = useMemo(() => {
    if (!hasElements || !hasTimeline) return [];

    const byDate = new Map();

    for (const ev of rawTimeline) {
      const date = String(ev.date || ev.timestamp || "").slice(0, 10);
      const doc = ev.doc || ev.document || ev.source || null;

      if (!date || !doc) continue;

      const docElements = elementsByDoc.get(doc);
      if (!docElements) continue;

      const enriched = docElements.map((e) => {
        const baseId = e.element || e.token || e.label || "";
        const id = String(baseId);
        const key = id.toLowerCase();
        const label = e.label || e.token || e.element || id;

        const classification = normaliseClassification(
          ev.classification || e.classification
        );

        return {
          date,
          elementId: key,
          label,
          classification,
          centrality: Number(e.centrality ?? 0),
          coreScore: Number(e.core ?? coreScore(e)),
          globalRank: globalRankById.get(key),
        };
      });

      enriched.sort(
        (a, b) => b.centrality - a.centrality || b.coreScore - a.coreScore
      );

      if (!byDate.has(date)) byDate.set(date, []);
      byDate.get(date).push(...enriched);
    }

    const dates = Array.from(byDate.keys()).sort();

    return dates.map((date) => {
      const entries = byDate.get(date) || [];
      const summary = entries
        .slice(0, 3)
        .map(
          (e) =>
            `${e.label} (${e.classification}, rank ${e.globalRank ?? "–"})`
        )
        .join("; ");

      return {
        date,
        entries,
        summary,
      };
    });
  }, [hasElements, hasTimeline, rawTimeline, elementsByDoc, globalRankById]);

  if (!hasElements) {
    return noData("Chronology requires Hilbert element statistics.");
  }

  if (!hasTimeline) {
    return noData("No timeline annotations available.");
  }

  if (!chronology.length) {
    return noData("No timeline dates match Hilbert elements.");
  }

  // -------------------------------------------------------------------------
  // 5) Slider logic (local to this tab)
  // -------------------------------------------------------------------------
  const [cursor, setCursor] = useState(0);
  const clampedIndex = Math.min(
    Math.max(0, cursor),
    chronology.length - 1
  );
  const active = chronology[clampedIndex];

  // narrative export
  const narrative = useMemo(
    () =>
      chronology
        .map(
          (d) =>
            `On ${d.date}, the most central elements are: ${
              d.summary || "none"
            }.`
        )
        .join("\n"),
    [chronology]
  );

  // highlight rows based on globally active element (if any)
  const isActiveRow = (e) => {
    if (!activeElement) return false;
    const currentId = String(
      activeElement.element ||
        activeElement.token ||
        activeElement.label ||
        ""
    ).toLowerCase();
    if (!currentId) return false;
    return (
      currentId === e.elementId ||
      currentId === (e.label || "").toLowerCase()
    );
  };

  // notify selection
  const handleClick = (entry) => {
    const payload = {
      element: entry.elementId,
      label: entry.label,
      date: entry.date,
      source: "chronology",
    };

    onSelectElement?.(payload);
    appendLog?.(`[Chronology] Selected ${entry.label} (${entry.date})`);
  };

  // -------------------------------------------------------------------------
  // RENDER
  // -------------------------------------------------------------------------
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        background: palette.panelBg,
        color: palette.text,
        fontSize: 12,
      }}
    >
      {/* Header with local slider */}
      <div
        style={{
          padding: "8px 10px",
          borderBottom: `1px solid ${palette.border}`,
          display: "flex",
          flexDirection: "column",
          gap: 6,
        }}
      >
        <div style={{ fontSize: 11, fontWeight: 600, color: palette.muted }}>
          Temporal Element Chronology
        </div>

        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          <span style={{ width: 90, fontSize: 10, color: palette.muted }}>
            Current date:
          </span>
          <span style={{ fontFamily: "monospace" }}>{active.date}</span>
        </div>

        <input
          type="range"
          min={0}
          max={chronology.length - 1}
          value={clampedIndex}
          onChange={(e) => setCursor(Number(e.target.value))}
        />
      </div>

      <div style={{ display: "flex", flex: 1, minHeight: 0 }}>
        {/* Table */}
        <div style={{ flex: 3, padding: 10, overflowY: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ borderBottom: `1px solid ${palette.border}` }}>
                <th>#</th>
                <th>Element</th>
                <th>Class</th>
                <th style={{ textAlign: "right" }}>Centrality</th>
                <th style={{ textAlign: "right" }}>Core</th>
                <th style={{ textAlign: "right" }}>Rank</th>
              </tr>
            </thead>
            <tbody>
              {active.entries.map((e, i) => (
                <tr
                  key={e.elementId + i}
                  onClick={() => handleClick(e)}
                  style={{
                    cursor: "pointer",
                    background: isActiveRow(e) ? "#111827" : "transparent",
                  }}
                >
                  <td>{i + 1}</td>
                  <td style={{ color: isActiveRow(e) ? palette.accent : "" }}>
                    {e.label}
                  </td>
                  <td>
                    <span
                      style={{
                        padding: "1px 6px",
                        borderRadius: 8,
                        border: `1px solid ${classificationColor(
                          e.classification
                        )}`,
                        color: classificationColor(e.classification),
                      }}
                    >
                      {e.classification}
                    </span>
                  </td>
                  <td style={{ textAlign: "right" }}>
                    {e.centrality.toFixed(3)}
                  </td>
                  <td style={{ textAlign: "right" }}>
                    {e.coreScore.toFixed(2)}
                  </td>
                  <td style={{ textAlign: "right" }}>
                    {e.globalRank ?? "–"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Narrative export */}
        <div
          style={{
            flex: 2,
            borderLeft: `1px solid ${palette.border}`,
            padding: 10,
            display: "flex",
            flexDirection: "column",
          }}
        >
          <div style={{ fontSize: 11, fontWeight: 600, marginBottom: 4 }}>
            Chronology export
          </div>
          <textarea
            readOnly
            value={narrative}
            style={{
              flex: 1,
              fontFamily: "monospace",
              fontSize: 10,
              background: "#020617",
              color: palette.text,
              border: `1px solid ${palette.border}`,
              borderRadius: 6,
              padding: 8,
              resize: "none",
            }}
          />
        </div>
      </div>
    </div>
  );
}

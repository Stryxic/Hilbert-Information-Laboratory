// ============================================================================
// PeriodicTablePanel.jsx — Informational Periodic Table (Root-aware, Timeline-synced)
// Final production-stable version for Hilbert Information Chemistry Lab UI
// ============================================================================

import React, { useMemo, useState } from "react";

const palette = {
  bg: "#0d1117",
  cardBg: "#0d1117",
  cardBgActive: "#111827",
  border: "#30363d",
  borderSoft: "#21262d",
  text: "#e6edf3",
  muted: "#8b949e",
  accent: "#58a6ff",
  info: "#10b981",
  mis: "#f97316",
  dis: "#ef4444",
};

// ---------------------------------------------------------------------------
// Classification → colour mapping
// ---------------------------------------------------------------------------
function colorForClassification(cls) {
  if (!cls) return palette.accent;
  const c = String(cls).toLowerCase();
  if (c.startsWith("mis")) return palette.mis;
  if (c.startsWith("dis")) return palette.dis;
  if (c.startsWith("info")) return palette.info;
  return palette.accent;
}

// ---------------------------------------------------------------------------
// Timeline-based colouring
// ---------------------------------------------------------------------------
function colorForElementAtDate(el, timelineEvents, currentDate) {
  if (!timelineEvents?.length || !currentDate) return palette.accent;

  const id = String(el.id).toLowerCase();

  const seen = timelineEvents.filter(
    (t) =>
      String(t.element).toLowerCase() === id &&
      new Date(t.date) <= new Date(currentDate)
  );

  if (!seen.length) return palette.accent;
  return colorForClassification(seen[seen.length - 1].classification);
}

// ============================================================================
// MAIN COMPONENT
// ============================================================================
export default function PeriodicTablePanel({
  results,
  activeElement,
  onSelectElement,
  appendLog,
  elementClusters = {},

  // timeline props:
  timeline = [],
  timelineDates = [],
  timelineCursor = 0,
  onTimelineChange,
}) {
  const [mode, setMode] = useState("top120");
  const [query, setQuery] = useState("");

  // -------------------------------------------------------------------------
  // 1) Extract elements from results according to new schema
  // -------------------------------------------------------------------------
  const elements = useMemo(() => {
    const rows = results?.elements?.elements || [];
    return rows.map((row, i) => {
      const id =
        row.element ||
        row.token ||
        row.id ||
        row.code ||
        `E${i}`;

      const label =
        row.label ||
        row.token ||
        row.element ||
        row.id ||
        `E${i}`;

      return {
        id: String(id),
        label: String(label),
        tf: Number(row.tf) || 0,
        df: Number(row.df) || 0,
        entropy: Number(row.entropy) || 0,
        coherence: Number(row.coherence) || 0,
        raw: row,
      };
    });
  }, [results]);

  // -------------------------------------------------------------------------
  // 2) Ranking (core score)
  // -------------------------------------------------------------------------
  const ranked = useMemo(() => {
    return [...elements].sort((a, b) => {
      const sa = Math.log1p(a.tf) + 0.5 * Math.log1p(a.df);
      const sb = Math.log1p(b.tf) + 0.5 * Math.log1p(b.df);
      return sb - sa;
    });
  }, [elements]);

  // -------------------------------------------------------------------------
  // 3) Top filtering + search filter
  // -------------------------------------------------------------------------
  const filtered = useMemo(() => {
    let list = ranked;

    if (mode === "top60") list = list.slice(0, 60);
    else if (mode === "top120") list = list.slice(0, 120);
    // mode === "all": leave intact

    if (query.trim()) {
      const q = query.toLowerCase();
      list = list.filter(
        (el) =>
          el.id.toLowerCase().includes(q) ||
          el.label.toLowerCase().includes(q)
      );
    }

    return list;
  }, [ranked, mode, query]);

  // -------------------------------------------------------------------------
  // 4) Timeline date
  // -------------------------------------------------------------------------
  const currentDate =
    timelineDates.length > 0
      ? timelineDates[timelineCursor]
      : null;

  // -------------------------------------------------------------------------
  // 5) Selection handler
  // -------------------------------------------------------------------------
  const handleSelect = (el) => {
    onSelectElement?.({
      element: el.id,
      label: el.label,
      token: el.label,
      source: "periodic",
    });
    appendLog?.(`[Periodic] Selected ${el.id} (${el.label})`);
  };

  const activeId =
    activeElement &&
    (activeElement.element ||
      activeElement.id ||
      activeElement.token);

  // ==========================================================================
  // RENDER
  // ==========================================================================
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        overflow: "hidden",
        background: palette.bg,
        color: palette.text,
        fontSize: 10,
      }}
    >
      {/* ---------------- HEADER ---------------- */}
      <div
        style={{
          padding: "6px 8px",
          borderBottom: `1px solid ${palette.border}`,
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: 10,
          flexShrink: 0,
        }}
      >
        <div>
          <div style={{ fontWeight: 600, fontSize: 12 }}>
            Informational Periodic Table
          </div>
          <div style={{ fontSize: 9, color: palette.muted }}>
            Temporal evolution of informational elements by date.
          </div>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <div style={{ display: "flex", gap: 4 }}>
            {["top60", "top120", "all"].map((opt) => (
              <button
                key={opt}
                onClick={() => setMode(opt)}
                style={{
                  padding: "2px 6px",
                  fontSize: 8,
                  borderRadius: 999,
                  border:
                    mode === opt
                      ? `1px solid ${palette.accent}`
                      : `1px solid ${palette.borderSoft}`,
                  background:
                    mode === opt ? "#161b22" : "transparent",
                  color: mode === opt ? palette.accent : palette.muted,
                  cursor: "pointer",
                }}
              >
                {opt.replace("top", "Top ")}
              </button>
            ))}
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{ fontSize: 8, color: palette.muted }}>
              ● <span style={{ color: palette.info }}>Info</span>{" "}
              ● <span style={{ color: palette.mis }}>Mis</span>{" "}
              ● <span style={{ color: palette.dis }}>Dis</span>
            </span>
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Filter…"
              style={{
                fontSize: 8,
                padding: "2px 6px",
                borderRadius: 999,
                border: `1px solid ${palette.borderSoft}`,
                background: "#020817",
                color: palette.text,
                minWidth: 70,
              }}
            />
          </div>
        </div>
      </div>

      {/* ---------------- TIMELINE ---------------- */}
      {timelineDates.length > 1 && (
        <div
          style={{
            padding: "6px 10px",
            borderBottom: `1px solid ${palette.border}`,
            display: "flex",
            alignItems: "center",
            gap: 10,
            flexShrink: 0,
            color: palette.muted,
          }}
        >
          <div style={{ width: 60 }}>Date</div>
          <input
            type="range"
            min={0}
            max={timelineDates.length - 1}
            value={timelineCursor}
            onChange={(e) =>
              onTimelineChange(Number(e.target.value))
            }
            style={{ flex: 1 }}
          />
          <div style={{ width: 100, textAlign: "right", fontSize: 9 }}>
            {currentDate || ""}
          </div>
        </div>
      )}

      {/* ---------------- GRID ---------------- */}
      <div
        style={{
          flex: 1,
          overflowY: "auto",
          padding: 6,
          display: "grid",
          gridTemplateColumns:
            "repeat(auto-fill, minmax(90px, 1fr))",
          gap: 4,
          minHeight: 0, // IMPORTANT: fixes the "big blue box" issue
        }}
      >
        {filtered.map((el) => {
          const isActive =
            activeId &&
            String(activeId).toLowerCase() ===
              String(el.id).toLowerCase();

          const color = colorForElementAtDate(
            el,
            timeline,
            currentDate
          );

          return (
            <button
              key={el.id}
              onClick={() => handleSelect(el)}
              style={{
                padding: "4px 4px 3px",
                borderRadius: 6,
                textAlign: "left",
                background: isActive
                  ? palette.cardBgActive
                  : palette.cardBg,
                border: `1px solid ${
                  isActive
                    ? palette.accent
                    : palette.borderSoft
                }`,
                color: palette.text,
                cursor: "pointer",
              }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "baseline",
                }}
              >
                <span
                  style={{ fontSize: 7, color: palette.muted }}
                >
                  {el.id}
                </span>
                <span
                  style={{
                    width: 6,
                    height: 6,
                    borderRadius: "50%",
                    background: color,
                  }}
                />
              </div>

              <div
                style={{
                  fontSize: 10,
                  fontWeight: 600,
                  overflow: "hidden",
                  whiteSpace: "nowrap",
                  textOverflow: "ellipsis",
                }}
              >
                {el.label}
              </div>

              <div style={{ fontSize: 7, color: palette.muted }}>
                core {Math.log1p(el.tf).toFixed(2)} · df {el.df}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

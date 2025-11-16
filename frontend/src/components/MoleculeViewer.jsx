// ============================================================================
// MoleculeViewer.jsx — Molecular Field & Compound Map (Timeline + Decomposition)
// ============================================================================

import React, { useMemo, useState } from "react";
import { runHilbertSubsetForElements } from "./hilbert_api";

const colors = {
  bg: "#0d1117",
  border: "#30363d",
  text: "#e6edf3",
  muted: "#8b949e",
  stable: "#10b981",
  hot: "#ef4444",
  accent: "#58a6ff",
  mis: "#facc15",
  dis: "#ef4444",
  info: "#10b981",
};

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

// Assign color based on stability score
function stabilityColor(s) {
  if (!Number.isFinite(s)) return colors.accent;
  if (s >= 0.66) return colors.stable;
  if (s <= 0.33) return colors.hot;
  return colors.accent;
}

// Normalize compound structures from results
function extractCompounds(results) {
  if (!results) return [];

  const raw =
    results.compounds?.compounds ||
    results.informational_compounds ||
    results["informational_compounds.json"] ||
    [];

  const arr = Array.isArray(raw) ? raw : Object.values(raw || {});

  return arr.map((c, i) => {
    const elemList =
      Array.isArray(c.elements) || Array.isArray(c.element_ids)
        ? (c.elements || c.element_ids).map((x) => String(x).toLowerCase())
        : typeof c.elements === "string"
        ? c.elements
            .split(/[,\s]+/)
            .filter(Boolean)
            .map((x) => String(x).toLowerCase())
        : [];

    return {
      id: String(c.compound_id || c.id || `C${i + 1}`),
      elements: elemList,
      stability: Number(c.stability || c.compound_stability || 0),
      temperature: Number(c.mean_temperature || c.temperature || 0.5),
      num_elements: elemList.length,
      num_bonds: Number(c.num_bonds || 0),
      info: Number(c.info || 0),
      mis: Number(c.mis || 0),
      dis: Number(c.dis || 0),
    };
  });
}

// Count classification hits for a compound’s elements up to a date
function countClassesForElements(elements, enrichedTimeline, dateCutoff) {
  if (!Array.isArray(enrichedTimeline) || !enrichedTimeline.length) {
    return { info: 0, mis: 0, dis: 0, seen: false };
  }

  if (!dateCutoff) return { info: 0, mis: 0, dis: 0, seen: false };

  const elementSet = new Set(elements.map((e) => String(e).toLowerCase()));
  let info = 0,
    mis = 0,
    dis = 0,
    seen = false;

  for (const evt of enrichedTimeline) {
    const d = evt.date || evt.timestamp;
    if (!d) continue;
    if (d > dateCutoff) break;

    if (!Array.isArray(evt.elements)) continue;

    for (const e of evt.elements) {
      const id =
        e.element ||
        e.label ||
        e.token ||
        e.id ||
        e.key ||
        e.raw?.element ||
        "";
      if (!id) continue;

      const lowered = String(id).toLowerCase();
      if (!elementSet.has(lowered)) continue;

      seen = true;
      const cls = String(e.classification || "").toLowerCase();

      if (cls.startsWith("dis")) dis++;
      else if (cls.startsWith("mis")) mis++;
      else info++;
    }
  }

  return { info, mis, dis, seen };
}

// ============================================================================
// MoleculeViewer Component
// ============================================================================

export default function MoleculeViewer({
  results,
  activeElement,
  onSelectElement,
  appendLog,

  // Timeline props from OutputPanel
  timeline,
  timelineDates,
  timelineCursor,
  onTimelineChange,
}) {
  // Extract compounds from main results
  const compounds = useMemo(() => extractCompounds(results), [results]);

  // Timeline context
  const currentDate =
    timelineDates && timelineDates.length
      ? timelineDates[Math.min(timelineCursor, timelineDates.length - 1)]
      : null;

  const enrichedTimeline = Array.isArray(timeline) ? timeline : [];
  const timelineEnabled =
    Array.isArray(timelineDates) && timelineDates.length > 1;

  // Mode + filters
  const [mode, setMode] = useState("stability");
  const [onlyVisibleAtTime, setOnlyVisibleAtTime] = useState(false);

  // For highlighting a selected compound
  const activeCompoundId =
    activeElement?.type === "compound"
      ? activeElement.id || activeElement.compound_id
      : null;

  // Decomposition state
  const [expandedId, setExpandedId] = useState(null);
  const [subsets, setSubsets] = useState({}); // id -> { loading, error, compounds }

  // Time aware enrichment
  const enriched = useMemo(() => {
    return compounds.map((c) => {
      const cls = countClassesForElements(
        c.elements,
        enrichedTimeline,
        currentDate
      );
      return { ...c, _cls: cls };
    });
  }, [compounds, enrichedTimeline, currentDate]);

  // Filter by visibility at current time
  const filtered = useMemo(() => {
    return onlyVisibleAtTime ? enriched.filter((c) => c._cls.seen) : enriched;
  }, [enriched, onlyVisibleAtTime]);

  // Sort by current mode
  const sorted = useMemo(() => {
    const list = [...filtered];
    if (mode === "stability") {
      list.sort((a, b) => b.stability - a.stability);
    } else {
      list.sort((a, b) => b.temperature - a.temperature);
    }
    return list;
  }, [filtered, mode]);

  // Select compound in global graph view
  const handleSelectCompound = (c) => {
    onSelectElement?.({
      type: "compound",
      compound_id: c.id,
      id: c.id,
      elements: c.elements,
      source: "molecule-viewer",
    });
    appendLog?.(`[MoleculeViewer] Selected compound ${c.id}`);
  };

  // Break a compound into sub molecules by running a subset pipeline
  const handleToggleDecompose = async (compound, evt) => {
    evt?.stopPropagation();

    if (expandedId === compound.id) {
      // Collapse if already open
      setExpandedId(null);
      return;
    }

    setExpandedId(compound.id);

    // If already loaded or currently loading, do not re-run
    const existing = subsets[compound.id];
    if (existing && (existing.loading || existing.compounds?.length)) {
      return;
    }

    setSubsets((prev) => ({
      ...prev,
      [compound.id]: { loading: true, error: null, compounds: [] },
    }));

    try {
      appendLog?.(
        `[MoleculeViewer] Decomposing ${compound.id} using subset pipeline`
      );

      const subsetResults = await runHilbertSubsetForElements({
        compound_id: compound.id,
        elements: compound.elements,
      });

      const subCompounds = extractCompounds(subsetResults);

      setSubsets((prev) => ({
        ...prev,
        [compound.id]: {
          loading: false,
          error: null,
          compounds: subCompounds,
        },
      }));

      appendLog?.(
        `[MoleculeViewer] ${compound.id} decomposed into ${subCompounds.length} sub-molecules`
      );
    } catch (err) {
      console.error("Subset decomposition failed:", err);
      setSubsets((prev) => ({
        ...prev,
        [compound.id]: {
          loading: false,
          error: err.message || String(err),
          compounds: [],
        },
      }));
      appendLog?.(
        `[MoleculeViewer] Decomposition failed for ${compound.id}: ${
          err.message || String(err)
        }`
      );
    }
  };

  // Select a sub molecule
  const handleSelectSubcompound = (parent, sub) => {
    onSelectElement?.({
      type: "compound",
      compound_id: sub.id,
      id: sub.id,
      elements: sub.elements,
      parent_compound_id: parent.id,
      source: "molecule-viewer-subset",
    });
    appendLog?.(
      `[MoleculeViewer] Selected sub-molecule ${sub.id} of ${parent.id}`
    );
  };

  // ========================================================================
  // RENDER
  // ========================================================================
  return (
    <div
      className="h-full flex flex-col"
      style={{
        background: colors.bg,
        border: `1px solid ${colors.border}`,
        borderRadius: 12,
        color: colors.text,
        fontSize: 12,
      }}
    >
      {/* HEADER */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          padding: "8px 10px",
          borderBottom: `1px solid ${colors.border}`,
        }}
      >
        <div>
          <div style={{ fontWeight: 600, fontSize: 12 }}>
            Molecular Stability Map
          </div>
          <div style={{ fontSize: 10, color: colors.muted }}>
            Explore compounds by stability or temperature. Click to focus graph
            or break down into sub molecules.
          </div>
        </div>

        {/* Mode Switch */}
        <div style={{ display: "flex", gap: 6 }}>
          <button
            onClick={() => setMode("stability")}
            style={{
              padding: "4px 8px",
              borderRadius: 8,
              border: `1px solid ${colors.border}`,
              background: mode === "stability" ? "#0f1720" : "#0d1117",
              color: colors.muted,
              fontSize: 10,
              cursor: "pointer",
            }}
          >
            Stability
          </button>

          <button
            onClick={() => setMode("temperature")}
            style={{
              padding: "4px 8px",
              borderRadius: 8,
              border: `1px solid ${colors.border}`,
              background: mode === "temperature" ? "#0f1720" : "#0d1117",
              color: colors.muted,
              fontSize: 10,
              cursor: "pointer",
            }}
          >
            Temperature
          </button>
        </div>
      </div>

      {/* TIMELINE CONTROLS */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          padding: "8px 10px",
          borderBottom: `1px solid ${colors.border}`,
          fontSize: 11,
          color: colors.muted,
        }}
      >
        {timelineEnabled ? (
          <>
            <div style={{ minWidth: 70 }}>Timeline</div>

            <input
              type="range"
              min={0}
              max={timelineDates.length - 1}
              value={timelineCursor}
              onChange={(e) => onTimelineChange(Number(e.target.value))}
              style={{ flex: 1 }}
            />

            <div style={{ width: 110, textAlign: "right" }}>
              {currentDate}
            </div>

            <label style={{ display: "flex", gap: 6, alignItems: "center" }}>
              <input
                type="checkbox"
                checked={onlyVisibleAtTime}
                onChange={(e) => setOnlyVisibleAtTime(e.target.checked)}
              />
              Only current-time compounds
            </label>
          </>
        ) : (
          <div style={{ fontStyle: "italic" }}>
            Timeline not available – showing all compounds.
          </div>
        )}
      </div>

      {/* COMPOUND LIST */}
      <div style={{ flex: 1, overflowY: "auto", padding: "8px 10px" }}>
        {sorted.length === 0 ? (
          <div style={{ fontSize: 11, color: colors.muted }}>
            No compounds match the current filter.
          </div>
        ) : (
          sorted.map((c) => {
            const isActive =
              activeCompoundId &&
              String(activeCompoundId) === String(c.id);

            const col = stabilityColor(
              mode === "stability" ? c.stability : 1 - c.temperature
            );

            const subsetState = subsets[c.id] || {
              loading: false,
              error: null,
              compounds: [],
            };
            const isExpanded = expandedId === c.id;

            return (
              <div
                key={c.id}
                onClick={() => handleSelectCompound(c)}
                style={{
                  border: `1px solid ${
                    isActive ? colors.accent : colors.border
                  }`,
                  background: isActive ? "#111827" : "#0d1117",
                  borderRadius: 10,
                  padding: "8px 10px",
                  cursor: "pointer",
                  marginBottom: 6,
                }}
              >
                {/* Header */}
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    gap: 8,
                    alignItems: "baseline",
                  }}
                >
                  <div style={{ fontWeight: 600, fontSize: 12 }}>
                    {c.id}{" "}
                    <span style={{ color: colors.muted }}>
                      ({c.num_elements} elements)
                    </span>
                  </div>

                  <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                    <Badge color={colors.info} label="I" value={c._cls.info} />
                    <Badge color={colors.mis} label="M" value={c._cls.mis} />
                    <Badge color={colors.dis} label="D" value={c._cls.dis} />

                    <span
                      style={{
                        width: 8,
                        height: 8,
                        borderRadius: 999,
                        background: col,
                        display: "inline-block",
                      }}
                    />

                    <span style={{ color: colors.muted, fontSize: 10 }}>
                      S {c.stability.toFixed(2)} · T{" "}
                      {c.temperature.toFixed(2)}
                    </span>

                    <button
                      onClick={(e) => handleToggleDecompose(c, e)}
                      style={{
                        padding: "2px 8px",
                        borderRadius: 8,
                        border: `1px solid ${colors.border}`,
                        background: isExpanded ? "#111827" : "#0d1117",
                        color: colors.muted,
                        fontSize: 10,
                        cursor: "pointer",
                      }}
                    >
                      {isExpanded ? "Hide sub molecules" : "Break down"}
                    </button>
                  </div>
                </div>

                {/* Element list */}
                {c.elements.length > 0 && (
                  <div
                    style={{
                      marginTop: 4,
                      color: colors.muted,
                      fontSize: 11,
                      lineHeight: 1.4,
                    }}
                  >
                    {c.elements.join(", ")}
                  </div>
                )}

                {/* Collapsible sub molecules */}
                {isExpanded && (
                  <div
                    style={{
                      marginTop: 8,
                      paddingTop: 6,
                      borderTop: `1px dashed ${colors.border}`,
                      fontSize: 11,
                    }}
                  >
                    {subsetState.loading && (
                      <div style={{ color: colors.muted }}>
                        Decomposing compound into sub molecules...
                      </div>
                    )}

                    {!subsetState.loading && subsetState.error && (
                      <div style={{ color: colors.hot }}>
                        Decomposition failed: {subsetState.error}
                      </div>
                    )}

                    {!subsetState.loading &&
                      !subsetState.error &&
                      (!subsetState.compounds ||
                        subsetState.compounds.length === 0) && (
                        <div style={{ color: colors.muted }}>
                          No sub molecules were found for this compound.
                        </div>
                      )}

                    {!subsetState.loading &&
                      !subsetState.error &&
                      subsetState.compounds &&
                      subsetState.compounds.length > 0 && (
                        <div>
                          <div
                            style={{
                              marginBottom: 4,
                              color: colors.muted,
                            }}
                          >
                            Sub molecules:
                          </div>
                          <ul
                            style={{
                              listStyle: "none",
                              paddingLeft: 0,
                              margin: 0,
                            }}
                          >
                            {subsetState.compounds.map((sub) => (
                              <li
                                key={sub.id}
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleSelectSubcompound(c, sub);
                                }}
                                style={{
                                  padding: "4px 6px",
                                  borderRadius: 6,
                                  border: `1px solid ${colors.border}`,
                                  marginBottom: 4,
                                  cursor: "pointer",
                                }}
                              >
                                <span style={{ fontWeight: 500 }}>
                                  {sub.id}
                                </span>{" "}
                                <span style={{ color: colors.muted }}>
                                  ({sub.num_elements} elements)
                                </span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Badge Component
// ---------------------------------------------------------------------------
function Badge({ color, label, value }) {
  if (!value) return null;
  return (
    <span
      title={label}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 6,
        fontSize: 10,
        color: "#c9d1d9",
      }}
    >
      <span
        style={{
          width: 8,
          height: 8,
          borderRadius: 999,
          background: color,
          display: "inline-block",
        }}
      />
      {label}:{value}
    </span>
  );
}

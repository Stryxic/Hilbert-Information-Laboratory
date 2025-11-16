// ======================================================================
// TimelinePanel.jsx - Temporal Diffusion Visualizer (prop-based)
// ======================================================================
//
// Uses the timeline already loaded by Dashboard (including timeline.json)
// to show how information / misinfo / disinfo evolve over time.
//
// Props expected (all optional):
//   - results: full Hilbert results object
//   - timelineEvents: enriched timeline array from Dashboard
//   - appendLog: optional logger
// ======================================================================

import React, { useMemo } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { Card } from "./ui/card";

const palette = {
  bg: "#0d1117",
  panelBg: "#161b22",
  border: "#30363d",
  text: "#e6edf3",
  muted: "#8b949e",
  info: "#10b981",
  mis: "#facc15",
  dis: "#ef4444",
};

// Basic classification normalisation
function normaliseClassification(raw) {
  const c = String(raw || "").toLowerCase().trim();
  if (!c) return "Unclassified";
  if (c.startsWith("dis")) return "Disinformation";
  if (c.startsWith("mis")) return "Misinformation";
  if (c.startsWith("pri")) return "Information";
  if (c.startsWith("cor")) return "Information";
  if (c.startsWith("info")) return "Information";
  return "Unclassified";
}

export default function TimelinePanel({
  results,
  timelineEvents,
  appendLog,
}) {
  // -------------------------------------------------------------------
  // 1) Normalise timeline source
  // -------------------------------------------------------------------
  const timeline = useMemo(() => {
    // Prefer explicit prop from Dashboard (already based on timeline.json)
    if (Array.isArray(timelineEvents) && timelineEvents.length) {
      return timelineEvents;
    }
    // Fall back to whatever is embedded in results
    const t = results?.timeline?.timeline || results?.timeline || [];
    return Array.isArray(t) ? t : [];
  }, [timelineEvents, results]);

  // -------------------------------------------------------------------
  // 2) Aggregate into chartData by "time step"
  // -------------------------------------------------------------------
  const chartData = useMemo(() => {
    if (!timeline.length) return [];

    const grouped = {};

    timeline.forEach((ev, idx) => {
      // Use t_index if present, otherwise the index position
      const tIndex =
        typeof ev.t_index === "number" ? ev.t_index : idx;

      const cls = normaliseClassification(
        ev.classification || ev.class || ev.bucket
      );

      if (!grouped[tIndex]) {
        grouped[tIndex] = {
          t_index: tIndex,
          Information: 0,
          Misinformation: 0,
          Disinformation: 0,
          Unclassified: 0,
        };
      }

      if (cls === "Information") grouped[tIndex].Information += 1;
      else if (cls === "Misinformation")
        grouped[tIndex].Misinformation += 1;
      else if (cls === "Disinformation")
        grouped[tIndex].Disinformation += 1;
      else grouped[tIndex].Unclassified += 1;
    });

    const sorted = Object.values(grouped).sort(
      (a, b) => a.t_index - b.t_index
    );

    if (!sorted.length && appendLog) {
      appendLog(
        "[Timeline] Normalisation produced no rows. Check timeline.json alignment."
      );
    }

    return sorted;
  }, [timeline, appendLog]);

  const loading = false;

  // -------------------------------------------------------------------
  // 3) Render
  // -------------------------------------------------------------------
  return (
    <Card className="h-full flex flex-col bg-[#0d1117] border border-[#30363d] text-xs">
      <div className="flex items-center justify-between px-3 py-2 border-b border-[#30363d]">
        <div>
          <div className="font-semibold text-[#e6edf3]">
            Timeline of Information Diffusion
          </div>
          <div className="text-[9px] text-[#8b949e]">
            Temporal evolution of information, misinformation, and disinformation
          </div>
        </div>
      </div>

      <div className="flex-1 p-3 overflow-auto">
        {loading ? (
          <div className="text-[#8b949e] text-[10px] p-2">
            Loading timeline data...
          </div>
        ) : chartData.length === 0 ? (
          <div className="text-[#8b949e] text-[10px] p-2">
            No annotated data available.
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={250}>
            <AreaChart
              data={chartData}
              margin={{ top: 10, right: 20, left: 0, bottom: 0 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
              <XAxis
                dataKey="t_index"
                tick={{ fill: "#8b949e", fontSize: 10 }}
              />
              <YAxis tick={{ fill: "#8b949e", fontSize: 10 }} />
              <Tooltip
                contentStyle={{
                  background: "#161b22",
                  border: "1px solid #30363d",
                  color: "#e6edf3",
                  fontSize: 10,
                }}
              />
              <Legend
                verticalAlign="top"
                height={20}
                wrapperStyle={{
                  fontSize: 10,
                  color: "#8b949e",
                }}
              />
              <Area
                type="monotone"
                dataKey="Information"
                stackId="1"
                stroke={palette.info}
                fill={palette.info + "80"}
              />
              <Area
                type="monotone"
                dataKey="Misinformation"
                stackId="1"
                stroke={palette.mis}
                fill={palette.mis + "80"}
              />
              <Area
                type="monotone"
                dataKey="Disinformation"
                stackId="1"
                stroke={palette.dis}
                fill={palette.dis + "80"}
              />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>
    </Card>
  );
}

// ============================================================================
// PersistencePanel.jsx — Stable version with figure fallback
// ============================================================================

import React, { useMemo } from "react";
import { Card } from "./ui/card";

const colors = {
  bg: "#0d1117",
  border: "#30363d",
  text: "#e6edf3",
  muted: "#8b949e",
  accent: "#58a6ff",
};

export default function PersistencePanel({ results }) {
  // Unified figure access
  const figures =
    results?.figures ||
    results?.results?.figures ||
    results?.results?.results?.figures ||
    {};

  const stability =
    results?.field?.spans   ||
    results?.stability ||
    results?.results?.field?.spans   ||
    [];

  const persistenceField =
    figures["persistence_field.png"] ||
    results?.persistence_field_url ||
    results?.results?.persistence_field_url;

  const scatterUrl =
    figures["stability_scatter.png"] ||
    results?.stability_scatter_url ||
    results?.results?.stability_scatter_url;

  const volatiles = useMemo(() => {
    if (!Array.isArray(stability) || stability.length === 0) return [];
    return [...stability]
      .filter((s) => s.element && s.stability != null)
      .sort((a, b) => a.stability - b.stability)
      .slice(0, 5);
  }, [stability]);

  return (
    <Card className="h-full flex flex-col bg-[#0d1117] border border-[#30363d] text-xs">
      <div className="flex items-center justify-between px-3 py-2 border-b border-[#30363d]">
        <div>
          <div className="font-semibold text-[#e6edf3]">
            Persistence & Stability
          </div>
          <div className="text-[9px] text-[#8b949e]">
            How stable are signals across spans?
          </div>
        </div>
      </div>

      <div className="flex-1 flex flex-col gap-2 p-3 overflow-auto">
        <div className="text-[9px] text-[#8b949e]">
          The line chart shows span-wise signal persistence. The scatter
          visualizes entropy–coherence balance per element.
        </div>

        {persistenceField && (
          <div>
            <div className="text-[8px] text-[#8b949e] mb-1">
              Span-wise signal stability
            </div>
            <img
              src={persistenceField}
              alt="Persistence field"
              className="w-full rounded border border-[#30363d]"
            />
          </div>
        )}

        {scatterUrl && (
          <div>
            <div className="text-[8px] text-[#8b949e] mb-1">
              Element entropy vs coherence
            </div>
            <img
              src={scatterUrl}
              alt="Stability scatter"
              className="w-full rounded border border-[#30363d]"
            />
          </div>
        )}

        {volatiles.length > 0 && (
          <div className="mt-1">
            <div className="text-[8px] text-[#8b949e] mb-1">
              Most volatile elements (lowest stability)
            </div>
            <table className="w-full text-[8px]">
              <thead className="text-[#8b949e]">
                <tr>
                  <th className="text-left">Element</th>
                  <th className="text-right">Stability</th>
                </tr>
              </thead>
              <tbody>
                {volatiles.map((v) => (
                  <tr key={v.element}>
                    <td className="text-[#e6edf3]">{v.element}</td>
                    <td className="text-right text-[#f97316]">
                      {Number(v.stability).toFixed(3)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </Card>
  );
}

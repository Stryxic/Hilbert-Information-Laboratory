// ============================================================================
// GraphSelectorPanel.jsx — Clean Force-Directed Layout (No fisheye, clustered)
// ============================================================================

import React, { useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";
import * as d3 from "d3";

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

// Regime colouring
function colorForRegime(row) {
  const info = Number(row.info_score ?? row.info ?? 0);
  const mis  = Number(row.misinfo_score ?? row.mis ?? 0);
  const dis  = Number(row.disinfo_score ?? row.dis ?? 0);
  if (dis > info && dis > mis) return palette.dis;
  if (mis > info && mis > dis) return palette.mis;
  return palette.info;
}

function coreScore(r) {
  const tf = Number(r.tf ?? 0);
  const df = Number(r.df ?? 0);
  return Math.log1p(tf) + 0.5 * Math.log1p(df);
}

export default function GraphSelectorPanel({
  results,
  activeElement,
  onSelectElement,
  appendLog,
}) {
  const fgRef = useRef();
  const containerRef = useRef();

  const [hoverNode, setHoverNode] = useState(null);
  const [hoverLink, setHoverLink] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);

  const [visiblePercent, setVisiblePercent] = useState(10);
  const [visibleTypes, setVisibleTypes] = useState({
    semantic: true,
    cooccur: true,
    compound: true,
  });

  // Track panel size
  const [graphSize, setGraphSize] = useState({ width: 800, height: 600 });
  useEffect(() => {
    const el = containerRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver((entries) => {
      const r = entries[0].contentRect;
      setGraphSize({ width: r.width, height: r.height });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Normalised schema
  const elementRows  = results?.elements?.elements ?? [];
  const edgeRows     = results?.edges?.edges ?? [];
  const compounds    = results?.compounds?.compounds ?? [];

  const compoundMembers = useMemo(() => {
    const out = {};
    compounds.forEach((c, i) => {
      const id = String(c.compound_id ?? c.id ?? `C${i+1}`);
      let list = c.elements || c.element_ids || [];
      if (typeof list === "string") list = list.split(/[,\s]+/);
      out[id] = new Set(list.map(String));
    });
    return out;
  }, [compounds]);

  const focusedCompoundId =
    activeElement?.compound_id ?? activeElement?.compound ?? null;

  // Build graph with top-% filtering
  const fullGraph = useMemo(() => {
    if (!elementRows.length) return { nodes: [], links: [], totalNodes: 0 };

    const scored = elementRows
      .map((r, i) => ({
        id    : String(r.element ?? r.token ?? r.id ?? `E${i+1}`),
        label : String(r.label ?? r.token ?? r.element ?? `E${i+1}`),
        row   : r,
        score : coreScore(r),
        color : colorForRegime(r),
        centrality : Number(r.centrality ?? 0),
      }))
      .sort((a, b) => b.score - a.score);

    const pct     = Math.max(5, Math.min(visiblePercent, 100));
    const cutoff  = Math.max(1, Math.round((pct/100) * scored.length));
    const visible = scored.slice(0, cutoff);
    const idset   = new Set(visible.map(n => n.id));

    // Build node list
    const nodes = visible.map(n => ({
      ...n,
      x: (Math.random() - 0.5) * 200,
      y: (Math.random() - 0.5) * 200
    }));

    // Infer edge type
    const inferType = (e) => {
      if (e.type) return e.type;
      if (e.edge_type) return e.edge_type;
      if (e.relation) return e.relation;
      if (e.compound_id) return "compound";
      return Number(e.weight ?? 0.1) < 0.15 ? "semantic" : "cooccur";
    };

    // Restrict links to visible nodes
    const links = edgeRows
      .map(e => ({
        source: String(e.source ?? e.a ?? e.element_a ?? ""),
        target: String(e.target ?? e.b ?? e.element_b ?? ""),
        value: Number(e.weight ?? 0.1),
        type : inferType(e)
      }))
      .filter(l => idset.has(l.source) && idset.has(l.target));

    return { nodes, links, totalNodes: scored.length };
  }, [elementRows, edgeRows, visiblePercent]);

  // Link-type filtering
  const filteredGraph = useMemo(() => {
    const links = fullGraph.links.filter(l => visibleTypes[l.type]);
    return { nodes: fullGraph.nodes, links, totalNodes: fullGraph.totalNodes };
  }, [fullGraph, visibleTypes]);

  // Fit graph to viewport
  useEffect(() => {
    const fg = fgRef.current;
    if (!fg || !filteredGraph.nodes.length) return;
    const t = setTimeout(() => {
      try { fg.zoomToFit(800, 40); } catch {}
    }, 200);
    return () => clearTimeout(t);
  }, [filteredGraph, graphSize]);

  // Node style
  const nodeColor = (n) => {
    if (selectedNode && selectedNode.id === n.id) return palette.accent;
    if (hoverNode && hoverNode.id === n.id) return "#9cc4ff";
    if (focusedCompoundId && compoundMembers[focusedCompoundId]?.has(n.id))
      return palette.info;
    return n.color;
  };

  // Node opacity when a node is selected
  const nodeOpacity = (n) => {
    if (!selectedNode) return 1;
    const sel = selectedNode.id;
    const connected = filteredGraph.links.some(
      l =>
        (l.source.id ?? l.source) === sel ||
        (l.target.id ?? l.target) === sel
    ) && filteredGraph.links.some(
      l =>
        (l.source.id ?? l.source) === n.id ||
        (l.target.id ?? l.target) === n.id
    );
    return connected || n.id === sel ? 1 : 0.25;
  };

  const linkColor = (l) => {
    if (hoverLink === l) return palette.accent;
    if (l.type === "compound") return "rgba(16,185,129,0.35)";
    if (l.type === "cooccur")  return "rgba(166,88,255,0.35)";
    return "rgba(88,166,255,0.35)";
  };

  // Clean node render
  const nodeCanvasObject = (node, ctx, scale) => {
    const r = 5 + Math.sqrt(node.centrality + 1);

    ctx.globalAlpha = nodeOpacity(node);
    ctx.beginPath();
    ctx.arc(node.x, node.y, r, 0, Math.PI * 2);
    ctx.fillStyle = nodeColor(node);
    ctx.fill();
    ctx.globalAlpha = 1;

    ctx.fillStyle = palette.text;
    ctx.font = `${Math.max(9, 14/scale)}px Inter, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(node.label, node.x, node.y + r + 1);
  };

  const nodePointerAreaPaint = (node, color, ctx) => {
    ctx.beginPath();
    ctx.arc(node.x, node.y, 14, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
  };

  // Click handling
  const handleNodeClick = (node) => {
    setSelectedNode(node);
    onSelectElement?.({
      element: node.id,
      label: node.label,
      source: "graph",
    });
    appendLog?.(`[Graph] Selected ${node.id}`);
  };

  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      height: "100%",
      background: palette.bg,
      border: `1px solid ${palette.border}`,
      borderRadius: 12,
      overflow: "hidden"
    }}>
      {/* HEADER */}
      <div
        style={{
          padding: "6px 10px",
          borderBottom: `1px solid ${palette.border}`,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between"
        }}
      >
        <div>
          <div style={{ fontWeight: 600, fontSize: 13, color: palette.accent }}>
            Graph & Molecule Selector
          </div>
          <div style={{ fontSize: 10, color: palette.muted }}>
            {filteredGraph.nodes.length} nodes • {filteredGraph.links.length} edges
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 10 }}>
          {["semantic", "cooccur", "compound"].map((t) => (
            <button
              key={t}
              onClick={() =>
                setVisibleTypes((prev) => ({ ...prev, [t]: !prev[t] }))
              }
              style={{
                padding: "2px 6px",
                borderRadius: 6,
                border: `1px solid ${palette.border}`,
                background: visibleTypes[t] ? "#0d1117" : "#111318",
                color: palette.text,
                fontSize: 10,
                cursor: "pointer",
              }}
            >
              {t}
            </button>
          ))}

          <span style={{ color: palette.muted }}>
            {filteredGraph.nodes.length} / {fullGraph.totalNodes}
          </span>

          <input
            type="range"
            min={5}
            max={100}
            step={5}
            value={visiblePercent}
            onChange={(e) => setVisiblePercent(parseInt(e.target.value, 10))}
          />
        </div>
      </div>

      {/* GRAPH */}
      <div ref={containerRef} style={{ flex: 1, background: palette.panelBg }}>
        <ForceGraph2D
          ref={fgRef}
          width={graphSize.width}
          height={graphSize.height}
          graphData={filteredGraph}
          backgroundColor={palette.panelBg}
          nodeCanvasObject={nodeCanvasObject}
          nodePointerAreaPaint={nodePointerAreaPaint}
          onNodeHover={setHoverNode}
          onLinkHover={setHoverLink}
          onNodeClick={handleNodeClick}
          onBackgroundClick={() => setSelectedNode(null)}
          
          // --- physics forces: mild repulsion + central force ---
          d3AlphaDecay={0.02}
          d3VelocityDecay={0.4}
          d3Force={(fg) => {
            fg
              .force("charge", d3.forceManyBody().strength(-40))
              .force("centerX", d3.forceX(0).strength(0.05))
              .force("centerY", d3.forceY(0).strength(0.05))
              .force("link", d3.forceLink(filteredGraph.links).id(d => d.id).distance(60).strength(0.6));
          }}

          // --- link appearance
          linkColor={linkColor}
          linkWidth={(l) => 0.7 + (l.value || 0.1)}
          linkDirectionalParticles={hoverLink ? 2 : 0}
          linkDirectionalParticleWidth={(l) => (hoverLink === l ? 2 : 0)}

          // --- remove edge labels entirely
          linkCanvasObjectMode={() => null}
        />
      </div>
    </div>
  );
}

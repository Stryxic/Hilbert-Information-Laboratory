// ======================================================================
// Dashboard.jsx — Hilbert Information Chemistry Lab UI (Safe + Robust)
// ======================================================================
//
// Features:
//   ✓ Working "Run Hilbert" button (POST /api/v1/analyze_corpus)
//   ✓ Full diagnostics in console + UI
//   ✓ Strict endpoint handling
//   ✓ Graceful fallback when results are missing
//   ✓ Normalisation compatible with current backend schema
//   ✓ Clear error source printed to UI
//   ✓ Proper wiring of timeline.json -> Timeline / Chronology views
//   ✓ Draggable panels via react-grid-layout
//
// ======================================================================

import React, {
  useState,
  useEffect,
  useCallback,
  useMemo,
  useRef,
} from "react";
import GridLayout from "react-grid-layout";
import "react-grid-layout/css/styles.css";
import "react-resizable/css/styles.css";

import {
  getLatestResults,
  getTimelineAnnotations,
  uploadCorpus,
  runHilbertPipeline,
} from "./hilbert_api";

import { PLUGINS } from "./hilbert_plugins";

// ----------------------------------------------------------------------
// Layout constants
// ----------------------------------------------------------------------
const DEFAULT_COLS = 12;
const DEFAULT_ROW_HEIGHT = 36;

function defaultLayoutForIndex(idx) {
  const w = 4;
  const h = 11;
  const x = (idx * w) % DEFAULT_COLS;
  const y = Math.floor((idx * w) / DEFAULT_COLS) * h;
  return { x, y, w, h, i: `panel-${idx}` };
}

// ======================================================================
// DASHBOARD ROOT COMPONENT
// ======================================================================
export default function Dashboard() {
  // -------------------------------------------------------------------
  // State
  // -------------------------------------------------------------------
  const [results, setResults] = useState(null);
  const [timelineEvents, setTimelineEvents] = useState([]);
  const [timelineCursor, setTimelineCursor] = useState(0);

  const [isLoadingResults, setIsLoadingResults] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [uploading, setUploading] = useState(false);

  const [errorMsg, setErrorMsg] = useState(null);

  const [selectedElement, setSelectedElement] = useState(null);
  const [selectedDoc, setSelectedDoc] = useState(null);
  const [selectedCompound, setSelectedCompound] = useState(null);

  // Default panels at startup
  const [panels, setPanels] = useState(() => [
    { key: "documents", pluginId: "documents-panel" },
    { key: "output-suite", pluginId: "output-suite" },
  ]);

  const [layout, setLayout] = useState(() =>
    panels.map((p, i) => ({ ...defaultLayoutForIndex(i), i: p.key }))
  );

  const [panelToAdd, setPanelToAdd] = useState("documents-panel");

  const gridContainerRef = useRef(null);
  const [gridWidth, setGridWidth] = useState(1300);

  // -------------------------------------------------------------------
  // Resize observer
  // -------------------------------------------------------------------
  useEffect(() => {
    function updateWidth() {
      if (gridContainerRef.current) {
        setGridWidth(gridContainerRef.current.offsetWidth || 1200);
      }
    }
    updateWidth();
    window.addEventListener("resize", updateWidth);
    return () => window.removeEventListener("resize", updateWidth);
  }, []);

  // ====================================================================
  // NORMALISE RESULTS
  // ====================================================================
  function normaliseResults(raw) {
    try {
      if (!raw) throw new Error("Backend returned empty result");

      // Backend: /api/v1/get_results returns:
      // { status, meta, field, elements, edges, compounds, documents, timeline, figures }
      const root = raw.results || raw;

      // Elements layer
      const elementRows =
        root.elements?.elements ||
        root.hilbert_elements || // older schema compatibility
        root.elements ||
        [];

      const elements = elementRows.map((e, i) => ({
        element: e.element || e.token || `E${i}`,
        token: e.token || e.element || `E${i}`,
        label: e.label || e.token || e.element || `E${i}`,
        tf: Number(e.tf ?? e.count ?? 0),
        df: Number(e.df ?? e.doc_freq ?? 0),
        centrality: Number(e.centrality ?? 0),
        core: Number(e.core ?? 0),
        date: e.date || e.timestamp || null,
        doc: e.doc || e.filename || e.source || null,
        raw: e,
      }));

      // Compounds layer
      const compounds =
        root.compounds?.compounds ||
        root.informational_compounds ||
        root.compounds ||
        [];

      // Edges layer
      const edges = root.edges?.edges || root.edges || [];

      // Timeline (backend: { timeline: [...], error })
      const timeline =
        root.timeline?.timeline ||
        root.timeline?.events ||
        root.timeline ||
        [];

      const timelineDates = (timeline || [])
        .map((t) => t.date || t.timestamp)
        .filter(Boolean);

      // Keep original field (spans + global H_bar/C_global)
      const field = root.field || {};

      // Stats
      const numElements =
        root.elements?.stats?.n_elements != null
          ? root.elements.stats.n_elements
          : elements.length;

      const numCompounds =
        root.compounds?.stats?.n_compounds != null
          ? root.compounds.stats.n_compounds
          : compounds.length;

      const edgesCount =
        root.edges?.stats?.n_edges != null
          ? root.edges.stats.n_edges
          : edges.length;

      return {
        ...root,
        field,
        elements: { elements, stats: root.elements?.stats || {} },
        compounds: { compounds, stats: root.compounds?.stats || {} },
        edges: { edges, stats: root.edges?.stats || {} },
        timeline,
        timelineDates,
        figures: root.figures || {},
        num_elements: numElements,
        num_compounds: numCompounds,
        edges_count: edgesCount,
      };
    } catch (err) {
      console.error("normaliseResults() FAILED:", err);
      setErrorMsg(`Result normalisation failed: ${err.message || String(err)}`);
      return null;
    }
  }

  // ====================================================================
  // FETCH RESULTS
  // ====================================================================
  const fetchResults = useCallback(async () => {
    try {
      setIsLoadingResults(true);
      setErrorMsg(null);

      const raw = await getLatestResults();
      if (!raw) throw new Error("Empty response from /get_results");

      const normalized = normaliseResults(raw);
      if (!normalized) throw new Error("Failed to normalise results");

      setResults(normalized);

      console.log(
        `[Dashboard] Loaded: ${normalized.num_elements} elements, ${normalized.num_compounds} compounds`
      );
    } catch (err) {
      console.error("fetchResults() FAILED:", err);
      setResults(null);
      setErrorMsg("fetchResults failed: " + (err.message || String(err)));
    } finally {
      setIsLoadingResults(false);
    }
  }, []);

  // ====================================================================
  // FETCH TIMELINE (timeline.json → /api/v1/get_timeline_annotations)
  // ====================================================================
  const fetchTimeline = useCallback(async () => {
    try {
      const res = await getTimelineAnnotations();
      const events = res.timeline || res.events || [];
      setTimelineEvents(events);
      console.log(`[Dashboard] Loaded ${events.length} timeline events.`);
    } catch (err) {
      console.error("fetchTimeline FAILED:", err);
    }
  }, []);

  // -------------------------------------------------------------------
  // Initial load: get latest run + timeline
  // -------------------------------------------------------------------
  useEffect(() => {
    fetchResults();
    fetchTimeline();
  }, [fetchResults, fetchTimeline]);

  // -------------------------------------------------------------------
  // Enrich timeline events with element centrality/core from elements
  // -------------------------------------------------------------------
  useEffect(() => {
    if (!results) return;

    const rawTimeline =
      timelineEvents.length
        ? timelineEvents
        : results.timeline?.timeline || results.timeline || [];

    if (!rawTimeline || !rawTimeline.length) return;

    const elems = results.elements?.elements || [];

    const enriched = rawTimeline.map((ev) => {
      const date = (ev.date || ev.timestamp || "").slice(0, 10);

      const matched = elems.filter((e) => {
        const d = (e.date || e.timestamp || "").slice(0, 10);
        return d === date;
      });

      return {
        ...ev,
        elements: matched.map((e) => ({
          element: e.element || e.token,
          centrality: Number(e.centrality ?? 0),
          core: Number(e.core ?? 0),
          classification: ev.classification || "info",
        })),
      };
    });

    setTimelineEvents((prev) => {
      const same = JSON.stringify(prev) === JSON.stringify(enriched);
      return same ? prev : enriched;
    });
  }, [results]); // do not include timelineEvents → avoids loops

  // ====================================================================
  // RUN PIPELINE (Run Hilbert button)
  // ====================================================================
  const handleRunHilbert = useCallback(
    async () => {
      try {
        setIsRunning(true);
        setErrorMsg(null);

        console.log("[Dashboard] Beginning pipeline run…");

        // POST /api/v1/analyze_corpus
        const res = await runHilbertPipeline();
        console.log("[Dashboard] Pipeline call result:", res);

        // When backend returns, pipeline is complete; reload artifacts
        await fetchResults();
        await fetchTimeline();
      } catch (err) {
        console.error("Pipeline FAILED:", err);
        setErrorMsg("Pipeline run failed: " + (err.message || String(err)));
      } finally {
        setIsRunning(false);
      }
    },
    [fetchResults, fetchTimeline]
  );

  // ====================================================================
  // UPLOAD CORPUS (triggers pipeline)
  // ====================================================================
  const handleUpload = useCallback(
    async (event) => {
      const files = Array.from(event.target.files || []);
      if (!files.length) return;

      try {
        setUploading(true);
        setErrorMsg(null);

        await uploadCorpus(files);
        console.log("[Dashboard] Corpus uploaded; starting pipeline…");

        await handleRunHilbert();
      } catch (err) {
        console.error("Upload FAILED:", err);
        setErrorMsg("Upload failed: " + (err.message || String(err)));
      } finally {
        setUploading(false);
        event.target.value = "";
      }
    },
    [handleRunHilbert]
  );

  // -------------------------------------------------------------------
  // TIMELINE cursor
  // -------------------------------------------------------------------
  const currentTimeline = useMemo(() => {
    if (!timelineEvents.length) return null;
    return timelineEvents[Math.min(timelineCursor, timelineEvents.length - 1)];
  }, [timelineEvents, timelineCursor]);

  // -------------------------------------------------------------------
  // PANELS
  // -------------------------------------------------------------------
  const addPanel = () => {
    const pluginId = panelToAdd;
    const key = `${pluginId}-${Date.now()}`;
    setPanels((prev) => [...prev, { key, pluginId }]);
    setLayout((prev) => [
      ...prev,
      { ...defaultLayoutForIndex(prev.length), i: key },
    ]);
  };

  const removePanel = (key) => {
    setPanels((prev) => prev.filter((p) => p.key !== key));
    setLayout((prev) => prev.filter((l) => l.i !== key));
  };

  // Shared props to every panel (global selection state etc.)
  const sharedProps = {
    results,

    // timeline props
    timelineEvents,
    timeline: timelineEvents,
    timelineDates: results?.timelineDates || [],
    timelineCursor,
    onTimelineChange: setTimelineCursor,

    // element selection - global & consistent
    activeElement: selectedElement,
    selectedElement,
    onSelectElement: setSelectedElement,

    // document selection
    selectedDoc,
    onSelectDoc: setSelectedDoc,

    // compound selection
    selectedCompound,
    onSelectCompound: setSelectedCompound,

    // pipeline control
    onRunHilbert: handleRunHilbert,
  };

  function renderPanel(panel) {
    const plugin = PLUGINS.find((p) => p.id === panel.pluginId);
    if (!plugin || !plugin.component) return null;

    const Component = plugin.component;

    return (
      <div key={panel.key} data-grid={layout.find((l) => l.i === panel.key)}>
        <div className="h-full flex flex-col bg-[#0d1117] border border-[#30363d] rounded-xl overflow-hidden">
          {/* Header is the drag handle */}
          <div className="react-grid-drag-handle flex items-center justify-between px-3 py-2 border-b border-[#30363d] bg-[#161b22]">
            <div className="text-sm font-semibold text-[#e6edf3]">
              {plugin.title}
            </div>
            <button
              className="text-xs px-2 py-1 rounded bg-[#21262d] hover:bg-[#30363d]"
              onClick={() => removePanel(panel.key)}
            >
              ✕
            </button>
          </div>

          <div className="flex-1 min-h-0 overflow-hidden">
            <Component {...sharedProps} />
          </div>
        </div>
      </div>
    );
  }

  // ====================================================================
  // RENDER
  // ====================================================================
  return (
    <div className="h-screen w-screen bg-[#010409] flex flex-col text-[#e6edf3]">
      {/* TOP BAR */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-[#30363d] bg-[#020817]">
        <div className="flex items-center gap-2">
          <button
            className="px-3 py-1 rounded-lg bg-[#238636] hover:bg-[#2ea043] text-sm font-semibold"
            onClick={addPanel}
          >
            + Add Panel
          </button>

          <select
            className="bg-[#0d1117] border border-[#30363d] rounded-lg text-sm px-2 py-1"
            value={panelToAdd}
            onChange={(e) => setPanelToAdd(e.target.value)}
          >
            {PLUGINS.map((p) => (
              <option key={p.id} value={p.id}>
                {p.title}
              </option>
            ))}
          </select>

          <button
            className="px-3 py-1 rounded-lg bg-[#21262d] hover:bg-[#30363d] text-sm"
            onClick={fetchResults}
          >
            Refresh
          </button>

          <label className="px-3 py-1 rounded-lg bg-[#21262d] hover:bg-[#30363d] text-sm cursor-pointer">
            Upload Corpus
            <input
              type="file"
              multiple
              className="hidden"
              onChange={handleUpload}
            />
          </label>

          <button
            className="px-3 py-1 rounded-lg bg-[#1f6feb] hover:bg-[#388bfd] text-sm font-semibold"
            disabled={isRunning || uploading}
            onClick={handleRunHilbert}
          >
            {isRunning ? "Running…" : "Run Hilbert"}
          </button>
        </div>

        <div className="flex items-center gap-4 text-xs">
          {currentTimeline && (
            <div className="flex flex-col items-end">
              <span className="text-[#8b949e]">Timeline</span>
              <span className="font-mono">
                {currentTimeline.date || currentTimeline.timestamp}
              </span>
            </div>
          )}

          {timelineEvents.length > 1 && (
            <input
              type="range"
              min={0}
              max={timelineEvents.length - 1}
              value={timelineCursor}
              onChange={(e) => setTimelineCursor(Number(e.target.value))}
            />
          )}

          {results && (
            <>
              <span className="text-[#8b949e]">
                Elements: {results.num_elements}
              </span>
              <span className="text-[#8b949e]">
                Compounds: {results.num_compounds}
              </span>
            </>
          )}
        </div>
      </div>

      {/* STATUS BAR */}
      <div className="px-4 py-1 text-xs border-b border-[#30363d] bg-[#02040c] flex justify-between">
        <div>
          Status:
          {isLoadingResults && <span> Loading results…</span>}
          {isRunning && <span> Running pipeline…</span>}
          {uploading && <span> Uploading corpus…</span>}
          {!isLoadingResults && !isRunning && !uploading && <span> Idle</span>}
        </div>
        {errorMsg && <div className="text-red-400">{errorMsg}</div>}
      </div>

      {/* PANEL GRID */}
      <div
        ref={gridContainerRef}
        className="flex-1 overflow-hidden px-3 pb-3 pt-2"
      >
        <GridLayout
          className="layout"
          cols={DEFAULT_COLS}
          rowHeight={DEFAULT_ROW_HEIGHT}
          width={gridWidth}
          margin={[8, 8]}
          compactType="vertical"
          draggableHandle=".react-grid-drag-handle"
          layout={layout}
          onLayoutChange={setLayout}
        >
          {panels.map(renderPanel)}
        </GridLayout>
      </div>
    </div>
  );
}

// ============================================================================
// DocumentsPanel.jsx — Fully fixed for new Hilbert unified schema
// ============================================================================

import React, { useEffect, useMemo, useState } from "react";
import axios from "axios";
import DocumentViewer from "./DocumentViewer";

const API_BASE = "http://127.0.0.1:8000/api/v1";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function basename(p) {
  if (!p) return "";
  const parts = String(p).split(/[/\\]/);
  return parts[parts.length - 1];
}

function rowDocId(row) {
  return basename(
    row.doc ||
      row.document ||
      row.source ||
      row.file ||
      row.filename ||
      row.raw_doc ||
      ""
  );
}

function bucket(cls) {
  const c = String(cls || "").toLowerCase();
  if (c.startsWith("dis")) return "dis";
  if (c.startsWith("mis")) return "mis";
  return "info";
}

// ---------------------------------------------------------------------------
// MAIN COMPONENT
// ---------------------------------------------------------------------------

export default function DocumentsPanel({
  results,
  timeline,
  onRunHilbert,
  appendLog,
  onSelectElement,
}) {
  const [docs, setDocs] = useState([]);
  const [selectedDoc, setSelectedDoc] = useState(null);
  const [loading, setLoading] = useState(false);
  const [runStatus, setRunStatus] = useState("");
  const [error, setError] = useState("");

  // ---- Correct schema mapping ----
  const elements = results?.elements?.elements || [];
  const compounds = results?.compounds?.compounds || [];
  const spans = results?.field?.spans?.length ?? "—";

  // -----------------------------------------------------------------------
  // Load document list
  // -----------------------------------------------------------------------
  useEffect(() => {
    let cancelled = false;

    async function loadDocs() {
      setLoading(true);
      try {
        const res = await axios.get(`${API_BASE}/get_document_list`);
        if (!cancelled) setDocs(res.data?.documents || []);
      } catch (err) {
        console.error("[DocumentsPanel] Failed doc list", err);
        if (!cancelled) setError("Failed to load document list.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    loadDocs();
    return () => {
      cancelled = true;
    };
  }, [results?.meta?.generated_at]);

  // -----------------------------------------------------------------------
  // Build per-document element frequencies
  // -----------------------------------------------------------------------
  const docElements = useMemo(() => {
    if (!elements.length) return {};

    const perDoc = {};

    for (const row of elements) {
      const docId = rowDocId(row);
      const token = row.element || row.token;
      if (!docId || !token) continue;

      if (!perDoc[docId]) perDoc[docId] = {};
      perDoc[docId][token] = (perDoc[docId][token] || 0) + 1;
    }

    const top = {};
    for (const [docId, freq] of Object.entries(perDoc)) {
      const sorted = Object.entries(freq)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5)
        .map(([el, count]) => ({ el, count }));
      top[docId] = sorted;
    }

    return top;
  }, [elements]);

  // -----------------------------------------------------------------------
  // Index docs by basename
  // -----------------------------------------------------------------------
  const docsByBase = useMemo(() => {
    const m = new Map();
    for (const d of docs) m.set(basename(d.name), d);
    return m;
  }, [docs]);

  // -----------------------------------------------------------------------
  // Timeline rows (Connor Reed annotated)
  // -----------------------------------------------------------------------
  const timelineRows = useMemo(() => {
    if (!timeline || !timeline.length) return [];

    return timeline.map((ev, idx) => {
      const mappedFile = basename(ev.filename || ev.file || ev.doc || "");
      const matchingDoc = mappedFile ? docsByBase.get(mappedFile) || null : null;

      return {
        id: `T-${idx}`,
        kind: "timeline",
        event: ev,
        doc: matchingDoc,
      };
    });
  }, [timeline, docsByBase]);

  // -----------------------------------------------------------------------
  // Extra docs not in timeline
  // -----------------------------------------------------------------------
  const extraDocRows = useMemo(() => {
    if (!docs.length) return [];

    const linkedNames = new Set(
      timelineRows.filter((r) => r.doc).map((r) => r.doc.name)
    );

    return docs
      .filter((d) => !linkedNames.has(d.name))
      .map((doc, idx) => ({
        id: `D-${idx}`,
        kind: "doc-only",
        doc,
      }));
  }, [timelineRows, docs]);

  // Combined list
  const allRows = [...timelineRows, ...extraDocRows];

  // -----------------------------------------------------------------------
  // Run pipeline
  // -----------------------------------------------------------------------
  const handleRun = async () => {
    try {
      setRunStatus("Running…");
      appendLog?.("[DocumentsPanel] Running pipeline...");
      await onRunHilbert();
      setRunStatus("Complete.");
    } catch (err) {
      console.error(err);
      setRunStatus("Failed.");
    }
  };

  // -----------------------------------------------------------------------
  // If selected doc → DocumentViewer
  // -----------------------------------------------------------------------
  if (selectedDoc) {
    const docName = selectedDoc.name;
    const elementsForDoc = elements.filter(
      (row) => rowDocId(row) === basename(docName)
    );

    return (
      <DocumentViewer
        docName={docName}
        elementsForDoc={elementsForDoc}
        compounds={compounds}
        onSelectElement={onSelectElement}
        onClose={() => setSelectedDoc(null)}
        appendLog={appendLog}
      />
    );
  }

  // -----------------------------------------------------------------------
  // MAIN LIST VIEW
  // -----------------------------------------------------------------------
  return (
    <div className="flex h-full flex-col bg-[#0d1117] border border-[#30363d] rounded-xl overflow-hidden text-[#e6edf3]">
      {/* HEADER */}
      <div className="px-3 py-2 border-b border-[#30363d] bg-[#161b22]">
        <div className="font-semibold text-sm">Documents</div>
        <div className="text-xs text-[#8b949e]">
          Timeline-linked documents appear first.
        </div>
        {runStatus && (
          <div className="text-[10px] text-[#6e7681] mt-1">{runStatus}</div>
        )}
        <button
          onClick={handleRun}
          className="mt-2 text-xs px-3 py-1 rounded bg-[#238636] hover:bg-[#2ea043]"
        >
          Run Hilbert
        </button>
      </div>

      {/* METRICS */}
      <div className="grid grid-cols-3 gap-2 px-3 py-2 border-b border-[#161b22] text-xs">
        <Metric label="Spans" value={spans} />
        <Metric label="Elements" value={results?.num_elements ?? "—"} />
        <Metric label="Compounds" value={results?.num_compounds ?? "—"} />
      </div>

      {/* LIST */}
      <div className="flex-1 overflow-y-auto px-2 py-2 text-xs">
        {loading && <div className="text-[#8b949e]">Loading…</div>}
        {error && <div className="text-red-400">{error}</div>}
        {!error &&
          !loading &&
          allRows.map((row) =>
            row.kind === "timeline" ? (
              <TimelineRow
                key={row.id}
                row={row}
                docElements={docElements}
                onOpenDoc={setSelectedDoc}
              />
            ) : (
              <DocRow
                key={row.id}
                row={row}
                docElements={docElements}
                onOpenDoc={setSelectedDoc}
              />
            )
          )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// SUBCOMPONENTS
// ---------------------------------------------------------------------------

function TimelineRow({ row, docElements, onOpenDoc }) {
  const { event, doc } = row;

  const classification = bucket(event.classification);
  const color =
    classification === "dis"
      ? "#ef4444"
      : classification === "mis"
      ? "#f97316"
      : "#10b981";

  const docBase = doc ? basename(doc.name) : "";
  const top = doc ? docElements[docBase] || [] : [];

  return (
    <div
      className="px-2 py-2 border-b border-[#161b22]"
      style={{
        cursor: doc ? "pointer" : "default",
        opacity: doc ? 1 : 0.6,
      }}
      onClick={() => doc && onOpenDoc(doc)}
    >
      <div className="flex justify-between">
        <div className="font-semibold text-sm truncate w-2/3">
          {doc ? doc.name : event.headline}
        </div>
        <div className="text-right">
          <div className="text-[#58a6ff]">{event.date || event.timestamp}</div>
          <span className="inline-flex items-center gap-1 text-xs">
            <span
              className="w-2 h-2 rounded-full inline-block"
              style={{ background: color }}
            />
            {classification}
          </span>
        </div>
      </div>

      <div className="text-[#c9d1d9] mt-1">
        {event.headline || ""}
      </div>

      {doc && (
        <div className="mt-1 text-[#8b949e]">
          {top.length
            ? top.map((t) => (
                <span key={t.el} className="mr-2 text-[#58a6ff]">
                  {t.el}
                  {t.count > 1 ? `×${t.count}` : ""}
                </span>
              ))
            : "No indexed elements."}
        </div>
      )}
    </div>
  );
}

function DocRow({ row, docElements, onOpenDoc }) {
  const doc = row.doc;
  const docBase = basename(doc.name);
  const top = docElements[docBase] || [];

  return (
    <div
      className="px-2 py-2 border-b border-[#161b22] cursor-pointer"
      onClick={() => onOpenDoc(doc)}
    >
      <div className="flex justify-between">
        <div className="font-semibold text-sm truncate w-2/3">
          {doc.name}
        </div>
        <div className="text-right text-[#8b949e] text-[10px]">
          No timeline
          <br />
          {doc.size_kb} KB
        </div>
      </div>
      <div className="mt-1 text-[#8b949e]">
        {top.length
          ? top.map((t) => (
              <span key={t.el} className="mr-2 text-[#58a6ff]">
                {t.el}
                {t.count > 1 ? `×${t.count}` : ""}
              </span>
            ))
          : "No indexed elements yet."}
      </div>
    </div>
  );
}

function Metric({ label, value }) {
  return (
    <div className="bg-[#151b23] border border-[#30363d] rounded-lg p-2 text-center">
      <div className="text-[10px] uppercase text-[#8b949e]">{label}</div>
      <div className="text-sm font-semibold">{value}</div>
    </div>
  );
}

// ============================================================================
// hilbert_api.js — Hilbert Information Chemistry Lab
// Frontend API Adapter (Final Production-Version Matching FastAPI Backend)
// ============================================================================

import axios from "axios";

// ---------------------------------------------------------------------------
// Base URLs
// ---------------------------------------------------------------------------
export const API_HOST = "http://127.0.0.1:8000";
export const API_BASE = `${API_HOST}/api/v1`;

// ===========================================================================
// Figure URL normalisation
// ===========================================================================
function fixFigureUrls(figs) {
  const out = {};
  if (!figs || typeof figs !== "object") return out;

  for (const [k, v] of Object.entries(figs)) {
    if (typeof v === "string" && v.startsWith("/results/")) {
      out[k] = `${API_HOST}${v}`;
    } else {
      out[k] = v;
    }
  }
  return out;
}

// ===========================================================================
// normalizeResults — matches EXACT backend shape in app.py:get_results
// ===========================================================================
export function normalizeResults(payload) {
  if (!payload || typeof payload !== "object") return {};

  // This matches the unified schema from app.py get_results()
  const meta = payload.meta || {};

  // ---------------- FIELD ----------------
  const field = payload.field || {};

  // ---------------- ELEMENTS ----------------
  const elemLayer = payload.elements || {};
  const elemArray = elemLayer.elements || [];
  const elemStats = elemLayer.stats || {};

  // ---------------- EDGES ----------------
  const edgeLayer = payload.edges || {};
  const edgeArray = edgeLayer.edges || [];
  const edgeStats = edgeLayer.stats || {};

  // ---------------- COMPOUNDS ----------------
  const compLayer = payload.compounds || {};
  const compArray = compLayer.compounds || [];
  const compStats = compLayer.stats || {};

  // ---------------- DOCUMENTS ----------------
  const docsLayer = payload.documents || {};
  const docs = docsLayer.documents || [];

  // ---------------- TIMELINE ----------------
  const tlLayer = payload.timeline || {};
  const tlArray = tlLayer.timeline || [];

  // ---------------- FIGURES ----------------
  const figures = fixFigureUrls(payload.figures || {});

  return {
    raw: payload,

    meta,
    field,

    elements: elemArray,
    element_stats: elemStats,

    edges: edgeArray,
    edge_stats: edgeStats,

    compounds: compArray,
    compound_stats: compStats,

    documents: docs,

    timeline: tlArray,

    figures,
    num_elements: elemArray.length,
    num_compounds: compArray.length,
    edges_count: edgeArray.length,

    persistence_field_url: figures["persistence_field.png"] || null,
    stability_scatter_url: figures["stability_scatter.png"] || null,
    summary_pdf_url: figures["hilbert_summary.pdf"] || null,

    run_folder: meta.dir || null,
  };
}

// ===========================================================================
// API CALLS
// ===========================================================================

// Upload one or more corpus documents
export async function uploadCorpus(files) {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));

  const res = await axios.post(`${API_BASE}/upload_corpus`, form, {
    headers: { "Content-Type": "multipart/form-data" },
  });

  return res.data;
}

// Run full pipeline
export async function runHilbertPipeline() {
  const res = await axios.post(`${API_BASE}/analyze_corpus`);
  return res.data;
}

// Latest results (schema aligned)
export async function getLatestResults() {
  const res = await axios.get(`${API_BASE}/get_results`, {
    params: { latest: true },
  });
  return normalizeResults(res.data || {});
}

// Timeline annotations (Connor Reed timeline)
export async function getTimelineAnnotations() {
  const res = await axios.get(`${API_BASE}/get_timeline_annotations`);
  const data = res.data || {};

  // Backend returns { timeline: [ ... ], error }
  return {
    events: Array.isArray(data.timeline) ? data.timeline : [],
    error: data.error || null,
    raw: data,
  };
}

// List of uploaded documents
export async function getDocumentList() {
  const res = await axios.get(`${API_BASE}/get_document_list`);
  return res.data?.documents || [];
}

// Get full text of a document
export async function getDocumentText(name) {
  const res = await axios.get(`${API_BASE}/get_document_text`, {
    params: { name },
  });
  return {
    name: res.data?.name || name,
    text: res.data?.text || "",
  };
}

// Compound context lookup
export async function getCompoundContext(element) {
  const res = await axios.get(`${API_BASE}/get_compound_context`, {
    params: { element },
  });
  return {
    element: res.data?.element || element,
    compounds: res.data?.compounds || [],
  };
}

// Element metadata (description, stats)
export async function getElementMap() {
  const res = await axios.get(`${API_BASE}/get_element_map`);
  const data = res.data || {};
  return (
    (Array.isArray(data.elements) && data.elements) ||
    (Array.isArray(data) && data) ||
    []
  );
}

export async function runHilbertSubsetForElements(payload) {
  // payload: { compound_id: string, elements: string[] }
  const res = await fetch("/run_hilbert_subset_for_elements", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(
      `Subset pipeline failed (${res.status}): ${text || res.statusText}`
    );
  }

  return res.json();
}

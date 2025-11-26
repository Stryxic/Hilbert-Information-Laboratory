// src/api/hilbert_api.js
//
// Centralized Hilbert API client for the frontend.
// Fully updated to match the corrected backend app.py.
//
// Covers:
//   - Corpus upload + pipeline
//   - Runs + artifacts
//   - Graph snapshots
//   - Elements / molecules (paginated)
//   - Stability: table, compounds, persistence field
//   - Search
//

// ---------------------------------------------------------------------------
// Base URL (Vite-compatible)
// ---------------------------------------------------------------------------

const DEFAULT_BASE_URL =
  (import.meta.env && import.meta.env.VITE_API_BASE) ||
  window.HILBERT_API_BASE ||
  "http://127.0.0.1:8000";

const DEFAULT_API_PREFIX = "/api/v1";

function buildUrl(
  path,
  { baseUrl = DEFAULT_BASE_URL, prefix = DEFAULT_API_PREFIX } = {}
) {
  const root = baseUrl.replace(/\/+$/, "");
  const pre = prefix.replace(/^\/?/, "/").replace(/\/+$/, "");
  const p = path.startsWith("/") ? path : `/${path}`;
  return `${root}${pre}${p}`;
}

// ---------------------------------------------------------------------------
// Low-level unified fetch wrapper
// ---------------------------------------------------------------------------

async function apiFetch(path, options = {}) {
  const url = buildUrl(path, options);

  const response = await fetch(url, {
    credentials: "same-origin",
    ...options,
    headers: {
      Accept: "application/json",
      ...(options.headers || {}),
    },
  });

  const contentType = response.headers.get("content-type") || "";
  let body;

  if (contentType.includes("application/json")) {
    body = await response.json();
  } else if (contentType.includes("text/")) {
    body = await response.text();
  } else {
    body = await response.blob().catch(() => null);
  }

  if (!response.ok) {
    const err = new Error(
      `Hilbert API error ${response.status}: ${
        response.statusText || "Unknown error"
      }`
    );
    err.status = response.status;
    err.body = body;
    throw err;
  }

  return body;
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

export async function getHealth() {
  return apiFetch("/health", { method: "GET" });
}

// ---------------------------------------------------------------------------
// Corpus Upload & Pipeline
// ---------------------------------------------------------------------------

export async function uploadCorpus(files) {
  if (!files || files.length === 0) {
    throw new Error("uploadCorpus: no files provided");
  }

  const form = new FormData();
  for (const f of files) form.append("files", f);

  return apiFetch("/upload_corpus", {
    method: "POST",
    body: form,
    headers: {},
  });
}

export async function runFull({ useNative = true } = {}) {
  const qs = new URLSearchParams({ use_native: useNative ? "true" : "false" });
  return apiFetch(`/run_full?${qs.toString()}`, { method: "POST" });
}

export async function analyzeCorpus({ maxDocs = null, useNative = true } = {}) {
  const params = new URLSearchParams();
  if (maxDocs != null) params.set("max_docs", maxDocs);
  params.set("use_native", useNative ? "true" : "false");
  const qs = params.toString();
  return apiFetch(`/analyze_corpus?${qs}`, { method: "POST" });
}

// ---------------------------------------------------------------------------
// Runs + Artifacts
// ---------------------------------------------------------------------------

export async function listRuns({ corpusId = null } = {}) {
  const params = new URLSearchParams();
  if (corpusId) params.set("corpus_id", corpusId);
  const qs = params.toString();
  const data = await apiFetch(qs ? `/runs?${qs}` : "/runs", {
    method: "GET",
  });
  return data.runs || [];
}

export async function getRun(runId) {
  return apiFetch(`/runs/${encodeURIComponent(runId)}`, { method: "GET" });
}

export async function listArtifacts(runId) {
  const data = await apiFetch(
    `/runs/${encodeURIComponent(runId)}/artifacts`,
    { method: "GET" }
  );
  return data.artifacts || [];
}

// ---------------------------------------------------------------------------
// Graphs
// ---------------------------------------------------------------------------

export async function listGraphs(runId) {
  const data = await apiFetch(
    `/graphs/${encodeURIComponent(runId)}/available`,
    { method: "GET" }
  );
  return data.available || [];
}

export async function getGraphSnapshot(runId, depth = null) {
  const params = new URLSearchParams();
  if (depth) params.set("depth", depth);
  const qs = params.toString();

  return apiFetch(
    qs
      ? `/graphs/${encodeURIComponent(runId)}/snapshot?${qs}`
      : `/graphs/${encodeURIComponent(runId)}/snapshot`,
    { method: "GET" }
  );
}

// ---------------------------------------------------------------------------
// Elements (pagination)
// ---------------------------------------------------------------------------

export async function listElements(
  runId,
  { page = 1, pageSize = 200 } = {}
) {
  const params = new URLSearchParams({
    page: String(page),
    page_size: String(pageSize),
  });

  const data = await apiFetch(
    `/runs/${encodeURIComponent(runId)}/elements?${params.toString()}`,
    { method: "GET" }
  );

  return {
    runId: data.run_id,
    page: data.page,
    pageSize: data.page_size,
    total: data.total,
    items: data.items || [],
  };
}

export async function getElementDetail(runId, elementId) {
  return apiFetch(
    `/runs/${encodeURIComponent(runId)}/elements/${encodeURIComponent(
      elementId
    )}`,
    { method: "GET" }
  );
}

// ---------------------------------------------------------------------------
// Molecules (pagination)
// ---------------------------------------------------------------------------

export async function listMolecules(
  runId,
  { page = 1, pageSize = 200 } = {}
) {
  const params = new URLSearchParams({
    page: String(page),
    page_size: String(pageSize),
  });

  const data = await apiFetch(
    `/runs/${encodeURIComponent(runId)}/molecules?${params.toString()}`,
    { method: "GET" }
  );

  return {
    runId: data.run_id,
    page: data.page,
    pageSize: data.page_size,
    total: data.total,
    items: data.items || [],
  };
}

export async function getMoleculeDetail(runId, moleculeId) {
  return apiFetch(
    `/runs/${encodeURIComponent(runId)}/molecules/${encodeURIComponent(
      moleculeId
    )}`,
    { method: "GET" }
  );
}

// ---------------------------------------------------------------------------
// Stability API (matches backend app.py)
// ---------------------------------------------------------------------------

export async function getStabilityTable(runId) {
  return apiFetch(
    `/runs/${encodeURIComponent(runId)}/stability/table`,
    { method: "GET" }
  );
}

export async function getStabilityCompounds(runId) {
  return apiFetch(
    `/runs/${encodeURIComponent(runId)}/stability/compounds`,
    { method: "GET" }
  );
}

export async function getPersistenceField(runId) {
  return apiFetch(
    `/runs/${encodeURIComponent(runId)}/persistence_field`,
    { method: "GET" }
  );
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

export async function searchAll({ runId = null, query, limit = 50 } = {}) {
  if (!query)
    return { query: "", runId, elements: [], molecules: [] };

  const params = new URLSearchParams({ query, limit });
  if (runId) params.set("run_id", runId);

  const data = await apiFetch(`/search?${params.toString()}`, {
    method: "GET",
  });

  return {
    query: data.query,
    runId: data.run_id ?? runId,
    elements: data.elements || [],
    molecules: data.molecules || [],
  };
}

export async function searchElements(runId, query, { limit = 50 } = {}) {
  const d = await searchAll({ runId, query, limit });
  return d.elements;
}

export async function searchMolecules(runId, query, { limit = 50 } = {}) {
  const d = await searchAll({ runId, query, limit });
  return d.molecules;
}

// ---------------------------------------------------------------------------
// Unified export
// ---------------------------------------------------------------------------

const HilbertAPI = {
  getHealth,

  uploadCorpus,
  analyzeCorpus,
  runFull,

  listRuns,
  getRun,
  listArtifacts,

  listGraphs,
  getGraphSnapshot,

  listElements,
  getElementDetail,

  listMolecules,
  getMoleculeDetail,

  getStabilityTable,
  getStabilityCompounds,
  getPersistenceField,

  searchAll,
  searchElements,
  searchMolecules,
};

export default HilbertAPI;

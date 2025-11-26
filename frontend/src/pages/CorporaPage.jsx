// src/pages/CorporaPage.jsx
//
// Fully wired Hilbert Dashboard Corpora/Run page with DB reset
//

import React, {
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";

import { useHilbertDB } from "../context/HilbertDBProvider";
import {
  uploadCorpus,
  runFull,
  listGraphs,
  getGraphSnapshot,
  listElements,
  listMolecules,
  getStabilityTable,
  getStabilityCompounds,
  getPersistenceField,
} from "../api/hilbert_api";

import "../styles/corpora.css";

export default function CorporaPage() {
  const {
    runs,
    selectedRun,
    runMetadata,
    selectRun,
    refreshRuns,
  } = useHilbertDB();

  // Upload state
  const [uploadFiles, setUploadFiles] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [uploadStatus, setUploadStatus] = useState("");

  // Right pane view state
  const [activeView, setActiveView] = useState("overview");
  const [viewLoading, setViewLoading] = useState(false);
  const [viewError, setViewError] = useState(null);

  // Fetched view data
  const [graphInfo, setGraphInfo] = useState(null);
  const [elementSlice, setElementSlice] = useState([]);
  const [moleculeSlice, setMoleculeSlice] = useState([]);
  const [stabilityTable, setStabilityTable] = useState(null);
  const [stabilityCompounds, setStabilityCompounds] = useState(null);
  const [persistenceField, setPersistenceField] = useState(null);

  // DB reset
  const [resetStatus, setResetStatus] = useState("");

  const hasRuns = runs.length > 0;
  const hasSelectedRun = !!selectedRun;

  // Metadata of selected run
  const selectedRunMeta = useMemo(() => {
    if (!runMetadata || !selectedRun) return null;
    return String(runMetadata.run_id) === String(selectedRun)
      ? runMetadata
      : null;
  }, [runMetadata, selectedRun]);

  const overviewCorpusLabel =
    selectedRunMeta?.corpus_name || "Unknown corpus";
  const overviewCreatedLabel =
    selectedRunMeta?.created_ts != null
      ? new Date(selectedRunMeta.created_ts * 1000).toLocaleString()
      : "Unknown";
  const overviewDocsLabel =
    selectedRunMeta?.settings?.max_docs
      ? `${selectedRunMeta.settings.max_docs} docs`
      : "full corpus";

  // Reset view when run is changed
  useEffect(() => {
    setActiveView("overview");
    setViewLoading(false);
    setViewError(null);
    setGraphInfo(null);
    setElementSlice([]);
    setMoleculeSlice([]);
    setStabilityTable(null);
    setStabilityCompounds(null);
    setPersistenceField(null);
  }, [selectedRun]);

  // ------------------------------------------------------------
  // Upload + run pipeline
  // ------------------------------------------------------------

  const handleFileChange = useCallback((e) => {
    const files = Array.from(e.target.files || []);
    setUploadFiles(files);
    setUploadStatus(files.length ? `${files.length} file(s) selected` : "");
  }, []);

  const handleProcessCorpus = useCallback(async () => {
    if (uploadFiles.length === 0) {
      setUploadStatus("Select files first.");
      return;
    }

    setIsProcessing(true);
    setUploadStatus("Uploading + running pipeline…");

    try {
      await uploadCorpus(uploadFiles);
      await runFull({ useNative: true });
      await refreshRuns();
      setUploadStatus("Pipeline finished. Runs updated.");
    } catch (err) {
      setUploadStatus(
        err?.body?.detail ||
          err?.message ||
          "Upload / pipeline failed."
      );
    } finally {
      setIsProcessing(false);
    }
  }, [uploadFiles, refreshRuns]);

  // ------------------------------------------------------------
  // Run selection
  // ------------------------------------------------------------

  const handleRunClick = useCallback(
    async (runId) => selectRun(runId),
    [selectRun]
  );

  const actionDisabled = !hasSelectedRun || viewLoading;

  // ------------------------------------------------------------
  // Right pane actions
  // ------------------------------------------------------------

  const handleOpenGraph = useCallback(async () => {
    if (!selectedRun) return;

    setActiveView("graph");
    setViewLoading(true);
    setViewError(null);

    try {
      const depths = await listGraphs(selectedRun);
      const depth = depths.includes("full") ? "full" : depths[0] || null;

      const snap = await getGraphSnapshot(selectedRun, depth);
      setGraphInfo({ depths, snapshot: snap });
    } catch (err) {
      setViewError("Failed to load graph.");
    } finally {
      setViewLoading(false);
    }
  }, [selectedRun]);

  const handleViewElements = useCallback(async () => {
    if (!selectedRun) return;

    setActiveView("elements");
    setViewLoading(true);
    setViewError(null);

    try {
      const page = await listElements(selectedRun);
      setElementSlice((page.items || []).slice(0, 20));
    } catch {
      setViewError("Failed to load elements.");
    } finally {
      setViewLoading(false);
    }
  }, [selectedRun]);

  const handleExploreMolecules = useCallback(async () => {
    if (!selectedRun) return;

    setActiveView("molecules");
    setViewLoading(true);
    setViewError(null);

    try {
      const page = await listMolecules(selectedRun);
      setMoleculeSlice((page.items || []).slice(0, 20));
    } catch {
      setViewError("Failed to load molecules.");
    } finally {
      setViewLoading(false);
    }
  }, [selectedRun]);

  const handleDocumentStats = useCallback(async () => {
    if (!selectedRun) return;

    setActiveView("docs");
    setViewLoading(true);
    setViewError(null);

    try {
      const [table, compounds, field] = await Promise.all([
        getStabilityTable(selectedRun),
        getStabilityCompounds(selectedRun),
        getPersistenceField(selectedRun),
      ]);

      setStabilityTable(table);
      setStabilityCompounds(compounds);
      setPersistenceField(field);
    } catch {
      setViewError("Failed to load document statistics.");
    } finally {
      setViewLoading(false);
    }
  }, [selectedRun]);

  // ------------------------------------------------------------
  // DB RESET button
  // ------------------------------------------------------------

  const handleResetDatabase = async () => {
    try {
      setResetStatus("Resetting database…");

      await fetch("http://127.0.0.1:8000/api/v1/admin/reset_db", {
        method: "POST",
      });

      setResetStatus("Database reset. Reloading runs…");
      await refreshRuns();
      setResetStatus("Database reset complete.");
    } catch (err) {
      setResetStatus("Reset failed: " + err.message);
    }
  };

  // ------------------------------------------------------------
  // Right pane renderer
  // ------------------------------------------------------------

  function renderActiveView() {
    if (!selectedRun)
      return <div className="empty-note">Select a run.</div>;

    if (viewLoading)
      return <div className="empty-note">Loading…</div>;

    if (viewError)
      return (
        <div className="empty-note" style={{ color: "#f66" }}>
          {viewError}
        </div>
      );

    // --- GRAPH ---
    if (activeView === "graph" && graphInfo) {
      const { snapshot } = graphInfo;
      return (
        <div className="view-block">
          <div><strong>Depth:</strong> {snapshot.depth}</div>
          <div><strong>Nodes:</strong> {snapshot.nodes.length} • <strong>Edges:</strong> {snapshot.edges.length}</div>
        </div>
      );
    }

    // --- ELEMENTS ---
    if (activeView === "elements") {
      return (
        <ul className="view-list">
          {elementSlice.map((el) => (
            <li key={el.element_id}>
              <strong>{el.element_id}</strong>
            </li>
          ))}
        </ul>
      );
    }

    // --- MOLECULES ---
    if (activeView === "molecules") {
      return (
        <ul className="view-list">
          {moleculeSlice.map((m) => (
            <li key={m.molecule_id || m.id}>
              <strong>{m.label || m.molecule_id || m.id}</strong>
            </li>
          ))}
        </ul>
      );
    }

    // --- DOC STATS ---
    if (activeView === "docs") {
      const rows = stabilityTable?.points || [];
      const compounds = stabilityCompounds?.compounds || [];
      const field = persistenceField;

      return (
        <div className="view-block">
          <div><strong>Rows:</strong> {rows.length}</div>
          <div><strong>Compounds:</strong> {compounds.length}</div>
          {field?.field_shape && (
            <div>
              <strong>Field:</strong> {field.field_shape[0]} × {field.field_shape[1]}
            </div>
          )}
        </div>
      );
    }

    return null;
  }

  // ------------------------------------------------------------
  // Render
  // ------------------------------------------------------------

  return (
    <div className="corpora-page">

      {/* LEFT PANE */}
      <div className="corpora-left">
        <div className="corpora-header">
          <h1>Corpora</h1>
          <p>Select or upload a corpus.</p>
        </div>

        <button
          className="hil-btn hil-btn-danger"
          style={{ marginBottom: 12 }}
          onClick={handleResetDatabase}
        >
          Reset Database
        </button>

        {resetStatus && (
          <div className="upload-status" style={{ marginBottom: 12 }}>
            {resetStatus}
          </div>
        )}

        <div className="upload-box">
          <div className="upload-label">Upload Corpus</div>
          <input type="file" multiple onChange={handleFileChange} />
          <button
            className="hil-btn hil-btn-primary"
            disabled={isProcessing}
            onClick={handleProcessCorpus}
          >
            {isProcessing ? "Processing…" : "Process corpus"}
          </button>
          {uploadStatus && <div className="upload-status">{uploadStatus}</div>}
        </div>

        <h2 className="section-heading">Available corpora</h2>
        <div className="empty-note">No corpora yet.</div>
      </div>

      {/* MIDDLE PANE */}
      <div className="corpora-middle">
        <h2>Runs</h2>
        <div className="middle-subtitle">Select a run.</div>

        {!hasRuns && (
          <div className="empty-note">No runs yet.</div>
        )}

        <div className="runs-list">
          {runs.map((r) => (
            <div
              key={r.run_id}
              className={
                String(r.run_id) === String(selectedRun)
                  ? "run-item selected"
                  : "run-item"
              }
              onClick={() => handleRunClick(r.run_id)}
            >
              <div className="run-title">
                Run {r.run_id}
              </div>
              <div className="run-meta">
                {r.created_ts
                  ? new Date(r.created_ts * 1000).toLocaleString()
                  : "Unknown"}
                {" • "}
                {r.settings?.max_docs
                  ? `${r.settings.max_docs} docs`
                  : "full corpus"}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* RIGHT PANE */}
      <div className="corpora-right">
        <h2>Run Overview</h2>

        <div className="section-card">
          <div className="section-title">Run ID</div>
          <div className="section-subtitle">
            {selectedRun || "No run selected"}
          </div>

          <div className="section-title" style={{ marginTop: 12 }}>
            Actions
          </div>

          <div className="actions-row">
            <button className="hil-btn hil-btn-primary" disabled={actionDisabled} onClick={handleOpenGraph}>
              Open Graph
            </button>
            <button className="hil-btn hil-btn-primary" disabled={actionDisabled} onClick={handleViewElements}>
              View Elements
            </button>
            <button className="hil-btn hil-btn-primary" disabled={actionDisabled} onClick={handleExploreMolecules}>
              Explore Molecules
            </button>
            <button className="hil-btn hil-btn-primary" disabled={actionDisabled} onClick={handleDocumentStats}>
              Document Statistics
            </button>
          </div>

          {selectedRun && (
            <div className="run-meta-grid">
              <div>
                <div className="meta-label">Corpus</div>
                <div className="meta-value">{overviewCorpusLabel}</div>
              </div>
              <div>
                <div className="meta-label">Created</div>
                <div className="meta-value">{overviewCreatedLabel}</div>
              </div>
              <div>
                <div className="meta-label">Docs</div>
                <div className="meta-value">{overviewDocsLabel}</div>
              </div>
            </div>
          )}

          {renderActiveView()}
        </div>
      </div>

    </div>
  );
}

// src/context/HilbertDBProvider.jsx
//
// Central global state provider for the Hilbert Dashboard.
// Ensures correct exports:  useHilbertDB  AND  HilbertDBProvider.
//

import React, {
  createContext,
  useContext,
  useEffect,
  useState,
  useCallback,
} from "react";

import {
  listRuns,
  getRun,
  listArtifacts,
} from "../api/hilbert_api";

// ------------------------------------------------------------
// Context
// ------------------------------------------------------------

const HilbertDBContext = createContext(null);

// ------------------------------------------------------------
// Hook Export  (<<< THIS MUST BE EXPORTED OR VITE WILL ERROR)
// ------------------------------------------------------------
export function useHilbertDB() {
  const ctx = useContext(HilbertDBContext);
  if (!ctx) {
    throw new Error("useHilbertDB must be used inside <HilbertDBProvider>");
  }
  return ctx;
}

// ------------------------------------------------------------
// Provider
// ------------------------------------------------------------
export function HilbertDBProvider({ children }) {
  const [corpora, setCorpora] = useState([]);
  const [runs, setRuns] = useState([]);
  const [selectedRun, setSelectedRun] = useState(null);

  const [runMetadata, setRunMetadata] = useState(null);
  const [artifacts, setArtifacts] = useState([]);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // ------------------------------------------------------------
  // Refresh run list
  // ------------------------------------------------------------
  const refreshRuns = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const list = await listRuns(); // GET /api/v1/runs
      setRuns(list || []);
    } catch (err) {
      console.error("Failed to load runs", err);
      setError(err);
    } finally {
      setLoading(false);
    }
  }, []);

  // ------------------------------------------------------------
  // Refresh individual run metadata
  // ------------------------------------------------------------
  const refreshRunMetadata = useCallback(async (runId) => {
    if (!runId) return;

    setLoading(true);
    setError(null);

    try {
      const meta = await getRun(runId);           // GET /api/v1/runs/{id}
      const arts = await listArtifacts(runId);    // GET /api/v1/runs/{id}/artifacts
      setRunMetadata(meta);
      setArtifacts(arts || []);
    } catch (err) {
      console.error("Failed to load run metadata", err);
      setError(err);
    } finally {
      setLoading(false);
    }
  }, []);

  // ------------------------------------------------------------
  // Select run
  // ------------------------------------------------------------
  const selectRun = useCallback(
    async (runId) => {
      setSelectedRun(runId);
      await refreshRunMetadata(runId);
    },
    [refreshRunMetadata]
  );

  // ------------------------------------------------------------
  // Placeholder until corpus API is exposed
  // ------------------------------------------------------------
  const refreshCorpora = useCallback(async () => {
    setCorpora([]);
  }, []);

  // ------------------------------------------------------------
  // Initial load
  // ------------------------------------------------------------
  useEffect(() => {
    refreshRuns();
  }, [refreshRuns]);

  // ------------------------------------------------------------
  // Provided global context value
  // ------------------------------------------------------------
  const value = {
    corpora,
    runs,

    selectedRun,
    runMetadata,
    artifacts,

    loading,
    error,

    refreshRuns,
    refreshCorpora,
    refreshRunMetadata,
    selectRun,
  };

  return (
    <HilbertDBContext.Provider value={value}>
      {children}
    </HilbertDBContext.Provider>
  );
}

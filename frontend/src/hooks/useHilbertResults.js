// src/hooks/useHilbertResults.js
//
// Central hook for talking to the Hilbert backend.
// - Fetches latest results and timeline
// - Derives element / molecule / compound counts
// - Exposes pipeline controls (run full, run batched, upload corpus)
// - Tracks loading / error state for the UI

import { useState, useEffect, useCallback } from "react";
import {
  getResults,
  getTimelineAnnotations,
  getElements,
  getMolecules,
  runHilbertPipelineFull,
  runHilbertPipelineBatched,
  uploadCorpus,
} from "../api/hilbert_api";

export default function useHilbertResults() {
  const [results, setResults] = useState(null);

  const [isLoadingResults, setIsLoadingResults] = useState(false);
  const [isRunningFull, setIsRunningFull] = useState(false);
  const [isRunningBatched, setIsRunningBatched] = useState(false);
  const [isUploading, setIsUploading] = useState(false);

  const [timelineEvents, setTimelineEvents] = useState([]);
  const [timelineCursor, setTimelineCursor] = useState(0);

  const [elementCount, setElementCount] = useState(null);
  const [moleculeCount, setMoleculeCount] = useState(null);
  const [compoundCount, setCompoundCount] = useState(null);
  const [corpusName, setCorpusName] = useState(null);

  const [error, setError] = useState(null);

  // ---------------------------------------------
  // Helpers to fetch counts & corpus name
  // ---------------------------------------------

  const fetchCounts = useCallback(async () => {
    try {
      const [elemRes, molRes] = await Promise.all([
        getElements({ limit: 1 }),
        getMolecules({ limit: 1 }),
      ]);

      setElementCount(elemRes?.n_elements ?? null);
      setMoleculeCount(molRes?.n_molecules ?? null);
    } catch (err) {
      console.warn("[useHilbertResults] fetchCounts failed:", err);
    }

    // Compound count - simple CSV line count via /results mount.
    try {
      const csvRes = await fetch("/results/hilbert_run/compound_stability.csv");
      if (csvRes.ok) {
        const txt = await csvRes.text();
        const lines = txt.split(/\r?\n/).filter((ln) => ln.trim().length > 0);
        // subtract header if present
        const count = Math.max(lines.length - 1, 0);
        setCompoundCount(count);
      } else {
        setCompoundCount(null);
      }
    } catch (err) {
      console.warn("[useHilbertResults] fetch compound count failed:", err);
      setCompoundCount(null);
    }
  }, []);

  // ---------------------------------------------
  // Fetch latest results
  // ---------------------------------------------
  const fetchResults = useCallback(async () => {
    try {
      setIsLoadingResults(true);
      setError(null);

      const data = await getResults();
      setResults(data || null);

      // derive corpus name from run_summary if available
      const corpus =
        data?.run_summary?.corpus_dir ||
        data?.run_summary?.corpus_root ||
        "unnamed corpus";
      setCorpusName(corpus);

      await fetchCounts();
    } catch (err) {
      console.error("[useHilbertResults] fetchResults failed:", err);
      setResults(null);
      setError(err.message || String(err));
    } finally {
      setIsLoadingResults(false);
    }
  }, [fetchCounts]);

  // ---------------------------------------------
  // Fetch timeline (non-fatal if missing)
  // ---------------------------------------------
  const fetchTimeline = useCallback(async () => {
    try {
      const res = await getTimelineAnnotations();
      const events = res?.timeline || res?.annotations || [];
      setTimelineEvents(events);
      if (events.length) {
        setTimelineCursor(events.length - 1);
      } else {
        setTimelineCursor(0);
      }
    } catch (err) {
      console.warn("[useHilbertResults] fetchTimeline failed:", err);
      // timeline is optional
    }
  }, []);

  // Initial load
  useEffect(() => {
    fetchResults();
    fetchTimeline();
  }, [fetchResults, fetchTimeline]);

  // ---------------------------------------------
  // Pipeline controls
  // ---------------------------------------------
  const handleRunFull = useCallback(async () => {
    try {
      setIsRunningFull(true);
      setError(null);

      await runHilbertPipelineFull();
      await fetchResults();
      await fetchTimeline();
    } catch (err) {
      console.error("[useHilbertResults] run full failed:", err);
      setError(err.message || String(err));
    } finally {
      setIsRunningFull(false);
    }
  }, [fetchResults, fetchTimeline]);

  const handleRunBatched = useCallback(
    async (batchSize = 5) => {
      try {
        setIsRunningBatched(true);
        setError(null);

        await runHilbertPipelineBatched({ batchSize });
        await fetchResults();
        await fetchTimeline();
      } catch (err) {
        console.error("[useHilbertResults] run batched failed:", err);
        setError(err.message || String(err));
      } finally {
        setIsRunningBatched(false);
      }
    },
    [fetchResults, fetchTimeline]
  );

  const handleUploadCorpus = useCallback(
    async (files) => {
      if (!files || !files.length) return;
      try {
        setIsUploading(true);
        setError(null);

        await uploadCorpus(files);
        // After upload, run the full pipeline
        await handleRunFull();
      } catch (err) {
        console.error("[useHilbertResults] upload failed:", err);
        setError(err.message || String(err));
      } finally {
        setIsUploading(false);
      }
    },
    [handleRunFull]
  );

  // Derived status string for status card / toolbar
  let statusText = "Idle";
  if (isUploading) statusText = "Uploading corpus…";
  else if (isRunningFull) statusText = "Running full pipeline…";
  else if (isRunningBatched) statusText = "Running batched pipeline…";
  else if (isLoadingResults) statusText = "Loading results…";

  return {
    // data
    results,
    timelineEvents,
    timelineCursor,
    elementCount,
    moleculeCount,
    compoundCount,
    corpusName,

    // state
    isLoadingResults,
    isRunningFull,
    isRunningBatched,
    isUploading,
    error,
    statusText,

    // controls
    fetchResults,
    fetchTimeline,
    handleRunFull,
    handleRunBatched,
    handleUploadCorpus,
    setTimelineCursor,
  };
}

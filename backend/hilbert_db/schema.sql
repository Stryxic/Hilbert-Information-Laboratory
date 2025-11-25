-- ============================================================
-- Hilbert DB Canonical Schema (v1)
-- ============================================================

-- ------------------------------------------------------------
-- Schema version table (for future migrations)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS schema_version (
    version      INTEGER NOT NULL,
    applied_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ------------------------------------------------------------
-- Corpus Table
--
-- A corpus is uniquely identified by its fingerprint (content hash).
-- corpus_id is a stable text identifier (usually identical to fingerprint).
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS corpus (
    corpus_id    TEXT PRIMARY KEY,                     -- stable ID (fingerprint)
    name         TEXT NOT NULL,
    fingerprint  TEXT UNIQUE NOT NULL,
    source_uri   TEXT,
    notes        TEXT,
    status       TEXT DEFAULT 'active',                -- optional future use
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_corpus_fingerprint
    ON corpus (fingerprint);

-- ------------------------------------------------------------
-- Runs Table
--
-- A run belongs to a corpus.
-- run_id is TEXT (timestamp, UUID, snowflake, etc.).
-- settings_json holds orchestrator settings including "__signature__".
-- status ∈ {pending, running, ok, failed, canceled}.
-- export_key stores object store path to deterministic export.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS runs (
    run_id                TEXT PRIMARY KEY,
    corpus_id             TEXT NOT NULL,
    orchestrator_version  TEXT,
    settings_json         TEXT,                  -- JSON blob
    status                TEXT DEFAULT 'pending',
    created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    finished_at           TIMESTAMP,
    export_key            TEXT,                  -- object-store path
    FOREIGN KEY (corpus_id) REFERENCES corpus(corpus_id)
);

CREATE INDEX IF NOT EXISTS idx_runs_corpus
    ON runs (corpus_id);

-- ------------------------------------------------------------
-- Artifacts Table
--
-- Each artifact is a file or structured output of a run:
-- CSVs, JSON files, PNGs, graph snapshots, export zips, etc.
--
-- artifact_id: TEXT so the orchestrator can generate deterministic IDs
-- key: object-store logical key (e.g. "runs/123/graphs/1pct.json")
-- kind: small label grouping artifacts ("export", "elements", "graph", ...)
-- meta_json: arbitrary metadata (json.dumps dict)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id   TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL,
    name          TEXT NOT NULL,
    kind          TEXT,
    key           TEXT NOT NULL,
    meta_json     TEXT,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_artifacts_run
    ON artifacts (run_id);

-- ------------------------------------------------------------
-- Initial schema version marker
-- ------------------------------------------------------------
INSERT INTO schema_version (version)
VALUES (1);

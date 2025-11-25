# ======================================================================
# hilbert_db - Distributed Storage and Retrieval Layer (Functional Stub)
# ======================================================================

from __future__ import annotations

import os
import json
import sqlite3
import hashlib
import shutil
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

@dataclass
class HilbertDBConfig:
    postgres_url: str
    object_store_bucket: str
    object_store_base_path: str
    cache_enabled: bool = True
    cache_dir: Optional[str] = None


# ----------------------------------------------------------------------
# Core Client Implementations (sqlite + local FS)
# ----------------------------------------------------------------------

class DBConnection:
    """Thin wrapper around sqlite3 connection."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._cursor: Optional[sqlite3.Cursor] = None

    def execute(self, query: str, params: Tuple[Any, ...] = ()) -> sqlite3.Cursor:
        self._cursor = self._conn.execute(query, params)
        return self._cursor

    def fetchone(self):
        if self._cursor is None:
            return None
        return self._cursor.fetchone()

    def fetchall(self):
        if self._cursor is None:
            return []
        return self._cursor.fetchall()

    def commit(self) -> None:
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


class DBPool:
    """Simple, non concurrent pool for sqlite."""

    def __init__(self, config: HilbertDBConfig):
        self._db_path = self._parse_sqlite_path(config.postgres_url)
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row

    @staticmethod
    def _parse_sqlite_path(url: str) -> str:
        # Expect format sqlite:///path/to/file.db
        if not url.startswith("sqlite:///"):
            raise ValueError(
                f"Functional stub only supports sqlite:/// URLs, got: {url}"
            )
        return url[len("sqlite:///") :]

    def get(self) -> DBConnection:
        return DBConnection(self._conn)

    def release(self, conn: DBConnection) -> None:
        # Single shared connection in this stub - nothing to do
        pass


class ObjectStoreClient:
    """
    Local filesystem object storage.

    All keys are stored under:
      base_path / bucket / remote_key
    """

    def __init__(self, base_path: str, bucket: str):
        self.base_path = os.path.abspath(base_path)
        self.bucket = bucket
        os.makedirs(os.path.join(self.base_path, self.bucket), exist_ok=True)

    def _full_path(self, remote_key: str) -> str:
        return os.path.join(self.base_path, self.bucket, remote_key)

    def upload_file(self, local_path: str, remote_key: str) -> None:
        dest = self._full_path(remote_key)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(local_path, dest)

    def download_file(self, remote_key: str, local_path: str) -> None:
        src = self._full_path(remote_key)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        shutil.copy2(src, local_path)

    def exists(self, remote_key: str) -> bool:
        return os.path.exists(self._full_path(remote_key))

    def delete(self, remote_key: str) -> None:
        path = self._full_path(remote_key)
        if os.path.exists(path):
            os.remove(path)


# ----------------------------------------------------------------------
# Corpus Registry
# ----------------------------------------------------------------------

@dataclass
class CorpusRecord:
    id: str
    content_hash: str
    original_name: str
    created_at: str
    num_files: int
    total_bytes: int
    metadata: Dict[str, Any]


class CorpusRegistry:
    def __init__(self, db: DBPool, store: ObjectStoreClient):
        self._db = db
        self._store = store

    def create_corpus(self, path: str, metadata: Dict[str, Any]) -> CorpusRecord:
        conn = self._db.get()
        cur = conn.execute("SELECT datetime('now')")
        created_at = cur.fetchone()[0]

        paths = []
        num_files = 0
        total_bytes = 0
        for root, _, files in os.walk(path):
            for f in files:
                full = os.path.join(root, f)
                paths.append(full)
                num_files += 1
                try:
                    total_bytes += os.path.getsize(full)
                except OSError:
                    pass

        content_hash = compute_content_hash(paths)
        original_name = os.path.basename(os.path.abspath(path))

        # Check if corpus already exists
        cur = conn.execute(
            "SELECT id, content_hash, original_name, created_at, "
            "num_files, total_bytes, metadata FROM corpora WHERE content_hash=?",
            (content_hash,),
        )
        row = cur.fetchone()
        if row:
            return CorpusRecord(
                id=row["id"],
                content_hash=row["content_hash"],
                original_name=row["original_name"],
                created_at=row["created_at"],
                num_files=row["num_files"],
                total_bytes=row["total_bytes"],
                metadata=json.loads(row["metadata"] or "{}"),
            )

        corpus_id = content_hash  # simple stub
        conn.execute(
            """
            INSERT INTO corpora(id, content_hash, original_name, created_at,
                                num_files, total_bytes, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                corpus_id,
                content_hash,
                original_name,
                created_at,
                num_files,
                total_bytes,
                json.dumps(metadata or {}),
            ),
        )
        conn.commit()

        # Store files under object store for future re use
        for full in paths:
            rel = os.path.relpath(full, path)
            remote_key = os.path.join("corpora", corpus_id, rel).replace("\\", "/")
            self._store.upload_file(full, remote_key)

        return CorpusRecord(
            id=corpus_id,
            content_hash=content_hash,
            original_name=original_name,
            created_at=created_at,
            num_files=num_files,
            total_bytes=total_bytes,
            metadata=metadata or {},
        )

    def get_corpus(self, corpus_id: str) -> Optional[CorpusRecord]:
        conn = self._db.get()
        cur = conn.execute(
            "SELECT id, content_hash, original_name, created_at, "
            "num_files, total_bytes, metadata FROM corpora WHERE id=?",
            (corpus_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return CorpusRecord(
            id=row["id"],
            content_hash=row["content_hash"],
            original_name=row["original_name"],
            created_at=row["created_at"],
            num_files=row["num_files"],
            total_bytes=row["total_bytes"],
            metadata=json.loads(row["metadata"] or "{}"),
        )

    def list_corpora(self, limit: int, offset: int) -> List[CorpusRecord]:
        conn = self._db.get()
        cur = conn.execute(
            """
            SELECT id, content_hash, original_name, created_at,
                   num_files, total_bytes, metadata
            FROM corpora
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )
        rows = cur.fetchall()
        out: List[CorpusRecord] = []
        for row in rows:
            out.append(
                CorpusRecord(
                    id=row["id"],
                    content_hash=row["content_hash"],
                    original_name=row["original_name"],
                    created_at=row["created_at"],
                    num_files=row["num_files"],
                    total_bytes=row["total_bytes"],
                    metadata=json.loads(row["metadata"] or "{}"),
                )
            )
        return out

    def find_by_hash(self, content_hash: str) -> Optional[CorpusRecord]:
        conn = self._db.get()
        cur = conn.execute(
            """
            SELECT id, content_hash, original_name, created_at,
                   num_files, total_bytes, metadata
            FROM corpora
            WHERE content_hash=?
            """,
            (content_hash,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return CorpusRecord(
            id=row["id"],
            content_hash=row["content_hash"],
            original_name=row["original_name"],
            created_at=row["created_at"],
            num_files=row["num_files"],
            total_bytes=row["total_bytes"],
            metadata=json.loads(row["metadata"] or "{}"),
        )

    def attach_corpus_files(self, corpus_id: str, local_paths: List[str]) -> None:
        for full in local_paths:
            rel = os.path.basename(full)
            remote_key = os.path.join("corpora", corpus_id, rel).replace("\\", "/")
            self._store.upload_file(full, remote_key)

    def list_corpus_files(self, corpus_id: str) -> List[str]:
        # Stub: just list keys under corpora/<id>/ in object store
        root = os.path.join(self._store.base_path, self._store.bucket, "corpora", corpus_id)
        out: List[str] = []
        if not os.path.exists(root):
            return out
        for dirpath, _, files in os.walk(root):
            for f in files:
                full = os.path.join(dirpath, f)
                rel = os.path.relpath(full, root).replace("\\", "/")
                out.append(rel)
        return out


# ----------------------------------------------------------------------
# Run Registry
# ----------------------------------------------------------------------

@dataclass
class RunRecord:
    id: str
    corpus_id: str
    created_at: str
    finished_at: Optional[str]
    status: str
    settings: Dict[str, Any]
    summary: Dict[str, Any]


class RunRegistry:
    def __init__(self, db: DBPool, store: ObjectStoreClient):
        self._db = db
        self._store = store

    def create_run(self, corpus_id: str, settings: Dict[str, Any]) -> RunRecord:
        conn = self._db.get()
        cur = conn.execute("SELECT datetime('now')")
        created_at = cur.fetchone()[0]
        run_id = hashlib.sha1(f"{corpus_id}-{created_at}".encode("utf-8")).hexdigest()

        conn.execute(
            """
            INSERT INTO runs(id, corpus_id, created_at, finished_at, status,
                             settings, summary)
            VALUES (?, ?, ?, NULL, ?, ?, ?)
            """,
            (
                run_id,
                corpus_id,
                created_at,
                "created",
                json.dumps(settings or {}),
                json.dumps({}),
            ),
        )
        conn.commit()

        return RunRecord(
            id=run_id,
            corpus_id=corpus_id,
            created_at=created_at,
            finished_at=None,
            status="created",
            settings=settings or {},
            summary={},
        )

    def update_run_status(self, run_id: str, status: str) -> None:
        conn = self._db.get()
        if status in ("finished", "failed"):
            cur = conn.execute("SELECT datetime('now')")
            finished_at = cur.fetchone()[0]
        else:
            finished_at = None
        conn.execute(
            """
            UPDATE runs
            SET status = ?, finished_at = COALESCE(?, finished_at)
            WHERE id = ?
            """,
            (status, finished_at, run_id),
        )
        conn.commit()

    def update_run_summary(self, run_id: str, summary: Dict[str, Any]) -> None:
        conn = self._db.get()
        conn.execute(
            "UPDATE runs SET summary = ? WHERE id = ?",
            (json.dumps(summary or {}), run_id),
        )
        conn.commit()

    def get_run(self, run_id: str) -> Optional[RunRecord]:
        conn = self._db.get()
        cur = conn.execute(
            "SELECT id, corpus_id, created_at, finished_at, status, settings, summary "
            "FROM runs WHERE id = ?",
            (run_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return RunRecord(
            id=row["id"],
            corpus_id=row["corpus_id"],
            created_at=row["created_at"],
            finished_at=row["finished_at"],
            status=row["status"],
            settings=json.loads(row["settings"] or "{}"),
            summary=json.loads(row["summary"] or "{}"),
        )

    def list_runs_for_corpus(self, corpus_id: str) -> List[RunRecord]:
        conn = self._db.get()
        cur = conn.execute(
            """
            SELECT id, corpus_id, created_at, finished_at, status, settings, summary
            FROM runs
            WHERE corpus_id = ?
            ORDER BY created_at DESC
            """,
            (corpus_id,),
        )
        out: List[RunRecord] = []
        for row in cur.fetchall():
            out.append(
                RunRecord(
                    id=row["id"],
                    corpus_id=row["corpus_id"],
                    created_at=row["created_at"],
                    finished_at=row["finished_at"],
                    status=row["status"],
                    settings=json.loads(row["settings"] or "{}"),
                    summary=json.loads(row["summary"] or "{}"),
                )
            )
        return out

    def attach_export_zip(self, run_id: str, path: str) -> None:
        remote_key = f"runs/{run_id}/export.zip"
        self._store.upload_file(path, remote_key)

        conn = self._db.get()
        conn.execute(
            """
            INSERT INTO artifacts(run_id, name, kind, uri, metadata)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(run_id, name) DO UPDATE SET
              kind=excluded.kind,
              uri=excluded.uri,
              metadata=excluded.metadata
            """,
            (run_id, "export.zip", "run-export", remote_key, json.dumps({})),
        )
        conn.commit()

    def get_export_zip_path(self, run_id: str) -> Optional[str]:
        conn = self._db.get()
        cur = conn.execute(
            """
            SELECT uri FROM artifacts
            WHERE run_id = ? AND name = ?
            """,
            (run_id, "export.zip"),
        )
        row = cur.fetchone()
        if not row:
            return None
        return row["uri"]


# ----------------------------------------------------------------------
# Artifact Registry
# ----------------------------------------------------------------------

@dataclass
class ArtifactRecord:
    run_id: str
    name: str
    kind: str
    uri: str
    metadata: Dict[str, Any]


class ArtifactRegistry:
    def __init__(self, db: DBPool):
        self._db = db

    def add_artifact(self, rec: ArtifactRecord) -> None:
        conn = self._db.get()
        conn.execute(
            """
            INSERT INTO artifacts(run_id, name, kind, uri, metadata)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(run_id, name) DO UPDATE SET
              kind=excluded.kind,
              uri=excluded.uri,
              metadata=excluded.metadata
            """,
            (
                rec.run_id,
                rec.name,
                rec.kind,
                rec.uri,
                json.dumps(rec.metadata or {}),
            ),
        )
        conn.commit()

    def list_artifacts(self, run_id: str) -> List[ArtifactRecord]:
        conn = self._db.get()
        cur = conn.execute(
            "SELECT run_id, name, kind, uri, metadata FROM artifacts WHERE run_id=?",
            (run_id,),
        )
        out: List[ArtifactRecord] = []
        for row in cur.fetchall():
            out.append(
                ArtifactRecord(
                    run_id=row["run_id"],
                    name=row["name"],
                    kind=row["kind"],
                    uri=row["uri"],
                    metadata=json.loads(row["metadata"] or "{}"),
                )
            )
        return out

    def get_artifact(self, run_id: str, name: str) -> Optional[ArtifactRecord]:
        conn = self._db.get()
        cur = conn.execute(
            "SELECT run_id, name, kind, uri, metadata "
            "FROM artifacts WHERE run_id=? AND name=?",
            (run_id, name),
        )
        row = cur.fetchone()
        if not row:
            return None
        return ArtifactRecord(
            run_id=row["run_id"],
            name=row["name"],
            kind=row["kind"],
            uri=row["uri"],
            metadata=json.loads(row["metadata"] or "{}"),
        )


# ----------------------------------------------------------------------
# Graph / Element / Molecule / Stability APIs
# ----------------------------------------------------------------------

class RunImporter:
    """
    Rehydrate an exported run ZIP into a temporary directory.
    """

    def __init__(self, store: ObjectStoreClient, cache_dir: Optional[str] = None):
        self._store = store
        self._cache_dir = cache_dir

    def import_run(self, run_id: str, export_zip_uri: str, tmp_dir: Optional[str] = None) -> str:
        import zipfile

        if tmp_dir is None:
            tmp_dir = ensure_temp_dir(self._cache_dir)

        local_zip = os.path.join(tmp_dir, f"{run_id}.zip")
        self._store.download_file(export_zip_uri, local_zip)

        out_dir = os.path.join(tmp_dir, run_id)
        os.makedirs(out_dir, exist_ok=True)

        with zipfile.ZipFile(local_zip, "r") as zf:
            zf.extractall(out_dir)

        # Many exports pack everything under a top level directory; detect that
        entries = os.listdir(out_dir)
        if len(entries) == 1:
            candidate = os.path.join(out_dir, entries[0])
            if os.path.isdir(candidate):
                return candidate
        return out_dir


class GraphAPI:
    def __init__(self, db: DBPool, store: ObjectStoreClient, importer: RunImporter):
        self._db = db
        self._store = store
        self._importer = importer

    def _ensure_local_run_dir(self, run_id: str) -> Optional[str]:
        conn = self._db.get()
        cur = conn.execute(
            "SELECT uri FROM artifacts WHERE run_id=? AND name=?",
            (run_id, "export.zip"),
        )
        row = cur.fetchone()
        if not row:
            return None
        uri = row["uri"]
        return self._importer.import_run(run_id, uri)

    def get_graph_metadata(self, run_id: str) -> Dict[str, Any]:
        run_dir = self._ensure_local_run_dir(run_id)
        if not run_dir:
            return {}
        path = os.path.join(run_dir, "graph_metadata.json")
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_snapshots(self, run_id: str) -> List[str]:
        run_dir = self._ensure_local_run_dir(run_id)
        if not run_dir:
            return []
        idx_path = os.path.join(run_dir, "graph_snapshots_index.json")
        if not os.path.exists(idx_path):
            # Fallback: list graph_*.png in root
            snaps: List[str] = []
            for f in os.listdir(run_dir):
                if f.startswith("graph_") and f.endswith(".png"):
                    snaps.append(f)
            snaps.sort()
            return snaps
        with open(idx_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Expect list of depth labels
        return list(data.get("snapshots", []))

    def get_snapshot_image(self, run_id: str, depth: str) -> bytes:
        run_dir = self._ensure_local_run_dir(run_id)
        if not run_dir:
            return b""
        fname = f"graph_{depth}.png"
        path = os.path.join(run_dir, fname)
        if not os.path.exists(path):
            return b""
        with open(path, "rb") as f:
            return f.read()

    def get_snapshot_json(self, run_id: str, depth: str) -> Dict[str, Any]:
        run_dir = self._ensure_local_run_dir(run_id)
        if not run_dir:
            return {}
        fname = f"graph_{depth}.json"
        path = os.path.join(run_dir, fname)
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


class ElementsAPI:
    def __init__(self, db: DBPool, store: ObjectStoreClient, importer: RunImporter):
        self._db = db
        self._store = store
        self._importer = importer

    def _elements_path(self, run_id: str) -> Optional[str]:
        run_dir = self._ensure_local_run_dir(run_id)
        if not run_dir:
            return None
        path = os.path.join(run_dir, "hilbert_elements.csv")
        return path if os.path.exists(path) else None

    def _ensure_local_run_dir(self, run_id: str) -> Optional[str]:
        conn = self._db.get()
        cur = conn.execute(
            "SELECT uri FROM artifacts WHERE run_id=? AND name=?",
            (run_id, "export.zip"),
        )
        row = cur.fetchone()
        if not row:
            return None
        uri = row["uri"]
        return self._importer.import_run(run_id, uri)

    def list_elements(self, run_id: str, limit: int, offset: int):
        import csv
        path = self._elements_path(run_id)
        if not path:
            return []

        rows = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows[offset : offset + limit]

    def get_element(self, run_id: str, element: str):
        import csv
        path = self._elements_path(run_id)
        if not path:
            return None
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("element") == element:
                    return row
        return None

    def search_elements(self, run_id: str, query: str):
        import csv
        path = self._elements_path(run_id)
        if not path:
            return []
        q = query.lower()
        out = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                el = (row.get("element") or "").lower()
                if q in el:
                    out.append(row)
        return out


class MoleculesAPI:
    def __init__(self, db: DBPool, store: ObjectStoreClient, importer: RunImporter):
        self._db = db
        self._store = store
        self._importer = importer

    def _run_dir(self, run_id: str) -> Optional[str]:
        conn = self._db.get()
        cur = conn.execute(
            "SELECT uri FROM artifacts WHERE run_id=? AND name=?",
            (run_id, "export.zip"),
        )
        row = cur.fetchone()
        if not row:
            return None
        uri = row["uri"]
        return self._importer.import_run(run_id, uri)

    def list_molecules(self, run_id: str, limit: int, offset: int):
        import csv
        run_dir = self._run_dir(run_id)
        if not run_dir:
            return []
        path = os.path.join(run_dir, "molecules.csv")
        if not os.path.exists(path):
            return []
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows[offset : offset + limit]

    def get_molecule(self, run_id: str, molecule_id: str):
        import csv
        run_dir = self._run_dir(run_id)
        if not run_dir:
            return None
        path = os.path.join(run_dir, "molecules.csv")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("molecule_id") == molecule_id:
                    return row
        return None

    def get_compound(self, run_id: str, compound_id: str):
        run_dir = self._run_dir(run_id)
        if not run_dir:
            return None
        path = os.path.join(run_dir, "informational_compounds.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get(str(compound_id))


class StabilityAPI:
    def __init__(self, db: DBPool, store: ObjectStoreClient, importer: RunImporter):
        self._db = db
        self._store = store
        self._importer = importer

    def _run_dir(self, run_id: str) -> Optional[str]:
        conn = self._db.get()
        cur = conn.execute(
            "SELECT uri FROM artifacts WHERE run_id=? AND name=?",
            (run_id, "export.zip"),
        )
        row = cur.fetchone()
        if not row:
            return None
        uri = row["uri"]
        return self._importer.import_run(run_id, uri)

    def list_stability(self, run_id: str, limit: int, offset: int):
        import csv
        run_dir = self._run_dir(run_id)
        if not run_dir:
            return []
        path = os.path.join(run_dir, "signal_stability.csv")
        if not os.path.exists(path):
            return []
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows[offset : offset + limit]

    def _sorted_by_stability(self, run_id: str, ascending: bool) -> List[Dict[str, Any]]:
        import csv
        run_dir = self._run_dir(run_id)
        if not run_dir:
            return []
        path = os.path.join(run_dir, "signal_stability.csv")
        if not os.path.exists(path):
            return []
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    row["_stab"] = float(row.get("stability", row.get("signal_stability", "0")) or 0.0)
                except Exception:
                    row["_stab"] = 0.0
                rows.append(row)
        rows.sort(key=lambda r: r["_stab"], reverse=not ascending)
        return rows

    def top_unstable(self, run_id: str, k: int):
        rows = self._sorted_by_stability(run_id, ascending=True)
        return rows[:k]

    def top_stable(self, run_id: str, k: int):
        rows = self._sorted_by_stability(run_id, ascending=False)
        return rows[:k]


# ----------------------------------------------------------------------
# High Level DB Facade
# ----------------------------------------------------------------------

class HilbertDB:
    def __init__(self, config: HilbertDBConfig):
        self._config = config
        self._db = DBPool(config)
        self._store = ObjectStoreClient(
            base_path=config.object_store_base_path,
            bucket=config.object_store_bucket,
        )
        self._init_schema()

        cache_dir = config.cache_dir
        self._importer = RunImporter(self._store, cache_dir=cache_dir)

        self._corpus_registry = CorpusRegistry(self._db, self._store)
        self._run_registry = RunRegistry(self._db, self._store)
        self._artifact_registry = ArtifactRegistry(self._db)
        self._graph_api = GraphAPI(self._db, self._store, self._importer)
        self._elements_api = ElementsAPI(self._db, self._store, self._importer)
        self._molecules_api = MoleculesAPI(self._db, self._store, self._importer)
        self._stability_api = StabilityAPI(self._db, self._store, self._importer)

    # Schema setup
    def _init_schema(self) -> None:
        conn = self._db.get()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS corpora (
                id TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                original_name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                num_files INTEGER NOT NULL,
                total_bytes INTEGER NOT NULL,
                metadata TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                corpus_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                finished_at TEXT,
                status TEXT NOT NULL,
                settings TEXT NOT NULL,
                summary TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
                run_id TEXT NOT NULL,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                uri TEXT NOT NULL,
                metadata TEXT NOT NULL,
                PRIMARY KEY (run_id, name)
            )
            """
        )
        conn.commit()

    # Exposed registries / APIs
    def corpus(self) -> CorpusRegistry:
        return self._corpus_registry

    def runs(self) -> RunRegistry:
        return self._run_registry

    def artifacts(self) -> ArtifactRegistry:
        return self._artifact_registry

    def graph(self) -> GraphAPI:
        return self._graph_api

    def elements(self) -> ElementsAPI:
        return self._elements_api

    def molecules(self) -> MoleculesAPI:
        return self._molecules_api

    def stability(self) -> StabilityAPI:
        return self._stability_api

    def importer(self) -> RunImporter:
        return self._importer


# ----------------------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------------------

def compute_content_hash(paths: List[str]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.encode("utf-8"))
        try:
            with open(p, "rb") as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    h.update(chunk)
        except OSError:
            continue
    return h.hexdigest()


def ensure_temp_dir(root: Optional[str] = None) -> str:
    if root:
        os.makedirs(root, exist_ok=True)
        return tempfile.mkdtemp(dir=root)
    return tempfile.mkdtemp()


def cleanup_temp_dir(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)


def serialize_summary_for_db(summary: Dict[str, Any]) -> Dict[str, Any]:
    # Stub passthrough; hook for future normalisation
    return summary or {}


def extract_metadata_from_export(results_dir: str) -> Dict[str, Any]:
    # Stub: try to load hilbert_run.json if present
    path = os.path.join(results_dir, "hilbert_run.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# ----------------------------------------------------------------------
# Exports
# ----------------------------------------------------------------------

__all__ = [
    "HilbertDBConfig",
    "DBPool",
    "DBConnection",
    "ObjectStoreClient",
    "CorpusRecord",
    "RunRecord",
    "ArtifactRecord",
    "CorpusRegistry",
    "RunRegistry",
    "ArtifactRegistry",
    "GraphAPI",
    "ElementsAPI",
    "MoleculesAPI",
    "StabilityAPI",
    "RunImporter",
    "HilbertDB",
    "compute_content_hash",
    "ensure_temp_dir",
    "cleanup_temp_dir",
    "serialize_summary_for_db",
    "extract_metadata_from_export",
]

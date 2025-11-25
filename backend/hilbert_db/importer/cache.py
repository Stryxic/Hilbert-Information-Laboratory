from __future__ import annotations

"""
Local cache management for imported Hilbert runs.

This module implements a filesystem-backed cache used by HilbertDB to
store rehydrated deterministic exports. It ensures that previously
imported runs can be reopened instantly without reprocessing the corpus.

This cache is *purely local* and can be safely deleted at any time:
HilbertDB will reconstruct it from the object store when needed.
"""

import os
import shutil
from dataclasses import dataclass
from typing import Optional


@dataclass
class CacheManager:
    """
    Filesystem-backed cache manager for imported Hilbert runs.

    Parameters
    ----------
    root_dir : Optional[str]
        Base directory for cache storage.
        If None, default path is ~/.hilbert_db/cache.
    """

    root_dir: Optional[str] = None

    # ------------------------------------------------------------------
    # Core directory helpers
    # ------------------------------------------------------------------

    def get_root(self) -> str:
        """
        Return the absolute cache root directory, creating it if needed.

        The default is ~/.hilbert_db/cache.
        """
        if self.root_dir is None:
            home = os.path.expanduser("~")
            self.root_dir = os.path.join(home, ".hilbert_db", "cache")

        root = os.path.abspath(self.root_dir)
        os.makedirs(root, exist_ok=True)
        return root

    def get_run_cache_dir(self, run_id: str) -> str:
        """
        Return the absolute path to the cache directory for the given run.

        This directory is created if it does not already exist.
        """
        root = self.get_root()
        run_dir = os.path.join(root, "runs", run_id)
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def has_cached_run(self, run_id: str) -> bool:
        """
        Return True if the run has a non-empty cache directory.

        A run is considered cached if:
            - a directory runs/<run_id>/ exists, and
            - it contains at least one entry.

        This does not validate the contents; it only checks presence.
        """
        run_dir = os.path.join(self.get_root(), "runs", run_id)
        if not os.path.isdir(run_dir):
            return False

        try:
            return any(os.scandir(run_dir))
        except OSError:
            return False

    # ------------------------------------------------------------------
    # Cache management utilities
    # ------------------------------------------------------------------

    def clear_run_cache(self, run_id: str) -> None:
        """
        Remove the cache directory and all cached contents for a run.

        Safe to call even if the directory does not exist.
        """
        run_dir = os.path.join(self.get_root(), "runs", run_id)
        shutil.rmtree(run_dir, ignore_errors=True)

    def clear_all(self) -> None:
        """
        Remove the entire cache tree.

        WARNING:
            This deletes all cached exports across all runs.
            All imported runs will need to be rehydrated from the
            object store afterwards.
        """
        root = self.get_root()
        shutil.rmtree(root, ignore_errors=True)
        os.makedirs(root, exist_ok=True)

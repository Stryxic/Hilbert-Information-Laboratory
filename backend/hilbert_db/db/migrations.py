"""
Database schema migration manager.

This component does NOT impose a specific migration system.
Instead, it provides a tiny abstraction so that HilbertDB can
eventually support:

    - versioned schema upgrades
    - reversible migrations
    - cross-backend consistency (SQLite, Postgres)

It is intentionally conservative and minimal.
"""

from __future__ import annotations
from typing import Callable, Dict, Any


class MigrationManager:
    """
    Schema migration registry and executor.

    Usage pattern:

        mgr = MigrationManager()
        mgr.register(
            version=1,
            upgrade=lambda conn: conn.execute("CREATE TABLE ..."),
            downgrade=lambda conn: conn.execute("DROP TABLE ..."),
        )
        mgr.apply_migrations(conn)

    This is a placeholder for future production-grade migrations.
    """

    def __init__(self):
        # Mapping:
        #     version -> {"upgrade": fn, "downgrade": fn}
        self.migrations: Dict[int, Dict[str, Callable]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        version: int,
        upgrade: Callable[[Any], None],
        downgrade: Callable[[Any], None] = None,
    ):
        """
        Register a migration step.

        Parameters
        ----------
        version:
            Integer schema version identifier.

        upgrade:
            Callable taking a DBConnection (or raw backend conn) that
            applies the upgrade.

        downgrade:
            Optional callable for reversing the migration. Not all
            production migrations are reversible, but we support the
            pattern for local dev.
        """
        self.migrations[version] = {
            "upgrade": upgrade,
            "downgrade": downgrade,
        }

    def get_latest_version(self) -> int:
        """
        Return the highest registered migration number, or 0 if none exist.
        """
        return max(self.migrations.keys(), default=0)

    # ------------------------------------------------------------------
    # Execution (placeholder)
    # ------------------------------------------------------------------

    def apply_migrations(self, conn) -> None:
        """
        Apply pending upgrades.

        This is intentionally a no-op for now. Future behaviour:

            1. Ensure schema_version table exists.
            2. Fetch the current database schema version.
            3. For each migration version > current version, call its
               upgrade() function.
            4. Update schema_version.

        Developers operating with SQLite can continue to rely on
        init_schema(), while production Postgres deployments will
        adopt a proper migration flow later.
        """
        # Future: implement real migration execution flow.
        return None

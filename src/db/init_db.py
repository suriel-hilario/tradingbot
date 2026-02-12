"""Create all database tables and seed default config.

Idempotent — safe to run on every container start.
  - create_all is a no-op for tables that already exist
  - seed_defaults only inserts keys that are missing

Retries the DB connection with backoff so it works even when
PostgreSQL is still accepting connections after Docker healthcheck passes.

Usage:
    python -m src.db.init_db
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time

from src.db.models import Base, Config, async_session, engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [init_db] %(levelname)s %(message)s",
)
log = logging.getLogger("init_db")

DEFAULTS: dict[str, str] = {
    "paper_trading": "true",
    "cycle_interval_sec": "300",
    "auto_trade_max_pct": "0.05",
    "max_daily_trades": "5",
}

MAX_RETRIES = 10
RETRY_DELAY = 2  # seconds


async def create_tables() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("Tables ready:")
    for table in Base.metadata.sorted_tables:
        log.info("  - %s", table.name)


async def seed_defaults() -> None:
    inserted = 0
    async with async_session() as session:
        for key, value in DEFAULTS.items():
            existing = await session.get(Config, key)
            if existing is None:
                session.add(Config(key=key, value=value))
                inserted += 1
        await session.commit()
    if inserted:
        log.info("Seeded %d new config entries", inserted)
    else:
        log.info("Config already seeded — no changes")


async def init() -> None:
    """Create tables + seed config with retry loop."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            await create_tables()
            await seed_defaults()
            await engine.dispose()
            log.info("Database initialization complete")
            return
        except Exception as exc:
            if attempt == MAX_RETRIES:
                log.error("Failed after %d attempts: %s", MAX_RETRIES, exc)
                sys.exit(1)
            log.warning(
                "Attempt %d/%d failed (%s) — retrying in %ds",
                attempt, MAX_RETRIES, exc, RETRY_DELAY,
            )
            await asyncio.sleep(RETRY_DELAY)


def main() -> None:
    asyncio.run(init())


if __name__ == "__main__":
    main()

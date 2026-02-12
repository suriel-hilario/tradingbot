"""SQLAlchemy async models for the crypto agent."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

load_dotenv()

_raw_url = os.getenv("DB_URL", "postgresql+asyncpg://agent:changeme@db:5432/cryptoagent")
DB_URL = _raw_url.replace("postgresql://", "postgresql+asyncpg://", 1) if _raw_url.startswith("postgresql://") else _raw_url

engine = create_async_engine(DB_URL, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Portfolio — periodic snapshots of total portfolio value
# ---------------------------------------------------------------------------

class Portfolio(Base):
    __tablename__ = "portfolios"

    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        index=True,
    )
    total_value_usd: Mapped[float]
    stablecoin_value_usd: Mapped[float] = mapped_column(default=0.0)
    stablecoin_pct: Mapped[float] = mapped_column(default=0.0)
    drawdown_pct: Mapped[float] = mapped_column(default=0.0)
    holdings: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    positions: Mapped[list[Position]] = relationship(
        back_populates="portfolio", cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Portfolio id={self.id} ${self.total_value_usd:.2f} @ {self.timestamp}>"


# ---------------------------------------------------------------------------
# Position — individual coin holdings within a snapshot
# ---------------------------------------------------------------------------

class Position(Base):
    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id"), index=True)
    symbol: Mapped[str] = mapped_column(String(20))
    amount: Mapped[float]
    entry_price: Mapped[float]
    current_price: Mapped[float]
    pnl_usd: Mapped[float] = mapped_column(default=0.0)
    pnl_pct: Mapped[float] = mapped_column(default=0.0)
    portfolio_pct: Mapped[float] = mapped_column(default=0.0)

    portfolio: Mapped[Portfolio] = relationship(back_populates="positions")

    def __repr__(self) -> str:
        return f"<Position {self.symbol} qty={self.amount} pnl={self.pnl_pct:+.2f}%>"


# ---------------------------------------------------------------------------
# Trade — every executed order
# ---------------------------------------------------------------------------

class Trade(Base):
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        index=True,
    )
    pair: Mapped[str] = mapped_column(String(20))
    side: Mapped[str] = mapped_column(String(4))           # buy / sell
    order_type: Mapped[str] = mapped_column(String(10))    # limit / market
    amount: Mapped[float]
    price: Mapped[float | None] = mapped_column(nullable=True)
    fill_price: Mapped[float | None] = mapped_column(nullable=True)
    fee: Mapped[float] = mapped_column(default=0.0)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    exchange_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    decision_id: Mapped[int | None] = mapped_column(
        ForeignKey("agent_decisions.id"), nullable=True,
    )

    decision: Mapped[AgentDecision | None] = relationship(back_populates="trades")

    def __repr__(self) -> str:
        return f"<Trade {self.side} {self.amount} {self.pair} @ {self.fill_price}>"


# ---------------------------------------------------------------------------
# Signal — research signals from any source
# ---------------------------------------------------------------------------

class Signal(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        index=True,
    )
    source: Mapped[str] = mapped_column(String(50))        # messari / glassnode / coindesk / onchain
    signal_type: Mapped[str] = mapped_column(String(30))   # bullish / bearish / neutral
    symbol: Mapped[str | None] = mapped_column(String(20), nullable=True)
    confidence: Mapped[str] = mapped_column(String(10))    # HIGH / MEDIUM / LOW
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_data: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    def __repr__(self) -> str:
        return f"<Signal {self.source} {self.signal_type} {self.confidence}>"


# ---------------------------------------------------------------------------
# AgentDecision — full log of every Claude response + actions
# ---------------------------------------------------------------------------

class AgentDecision(Base):
    __tablename__ = "agent_decisions"

    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        index=True,
    )
    model: Mapped[str] = mapped_column(String(60), default="claude-sonnet-4-5-20250929")
    prompt_tokens: Mapped[int] = mapped_column(default=0)
    completion_tokens: Mapped[int] = mapped_column(default=0)
    claude_response: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    actions_taken: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    trades: Mapped[list[Trade]] = relationship(back_populates="decision")

    def __repr__(self) -> str:
        return f"<AgentDecision id={self.id} @ {self.timestamp}>"


# ---------------------------------------------------------------------------
# Config — runtime key-value settings
# ---------------------------------------------------------------------------

class Config(Base):
    __tablename__ = "config"

    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    def __repr__(self) -> str:
        return f"<Config {self.key}={self.value!r}>"


# ---------------------------------------------------------------------------
# Helpers used by the orchestrator
# ---------------------------------------------------------------------------

async def log_to_db(response: Any, actions: list[dict]) -> None:
    """Persist a Claude response and its resulting actions."""
    async with async_session() as session:
        decision = AgentDecision(
            model=response.model,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            claude_response={"content": [b.model_dump() for b in response.content]},
            actions_taken=actions,
        )
        session.add(decision)
        await session.commit()

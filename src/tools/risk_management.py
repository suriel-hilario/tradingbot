"""Risk management engine — every trade must pass through here.

Implements all checks from skills/risk_management.md.
Limits are loaded from environment variables with sensible defaults
that match agent.md.  A future version can override them from the
db Config table at runtime.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

# ── Stablecoins recognized by the system ──────────────────────────────────
STABLECOINS = frozenset(os.getenv("STABLECOINS", "USDT,USDC,BUSD,DAI,TUSD").split(","))

MEMECOIN_WHITELIST = frozenset(
    t.strip() for t in os.getenv("MEMECOIN_WHITELIST", "").split(",") if t.strip()
)

# Known memecoins — expand as needed.  Anything here is blocked unless
# it also appears in MEMECOIN_WHITELIST.
KNOWN_MEMECOINS = frozenset(
    os.getenv(
        "KNOWN_MEMECOINS",
        "DOGE,SHIB,PEPE,FLOKI,BONK,WIF,BRETT,TURBO,BABYDOGE,MEME",
    ).split(",")
)


# ── Configurable limits ──────────────────────────────────────────────────

@dataclass(frozen=True)
class RiskLimits:
    """All risk thresholds.  Each can be overridden via env or db."""

    max_position_pct: float = float(os.getenv("MAX_POSITION_PCT", "0.10"))
    stop_loss_pct: float = float(os.getenv("STOP_LOSS_PCT", "-0.08"))
    max_drawdown_pct: float = float(os.getenv("MAX_DRAWDOWN_PCT", "-0.20"))
    min_stablecoin_pct: float = float(os.getenv("MIN_STABLECOIN_PCT", "0.30"))
    min_market_cap: float = float(os.getenv("MIN_MARKET_CAP", "50000000"))
    max_daily_trades: int = int(os.getenv("MAX_DAILY_TRADES", "5"))
    min_trade_interval_sec: int = int(os.getenv("MIN_TRADE_INTERVAL_SEC", "3600"))
    min_signal_sources: int = int(os.getenv("MIN_SIGNAL_SOURCES", "2"))
    auto_trade_max_pct: float = float(os.getenv("AUTO_TRADE_MAX_PCT", "0.05"))


DEFAULT_LIMITS = RiskLimits()


# ── Input / output data structures ───────────────────────────────────────

@dataclass
class TradeProposal:
    pair: str                               # e.g. "BTC/USDT"
    side: str                               # "buy" | "sell"
    amount: float                           # quantity of base asset
    price: float                            # expected fill price (USD)
    order_type: str = "limit"               # "limit" | "market"
    stop_loss_pct: float | None = None      # e.g. -0.08
    leverage: float = 1.0                   # must be 1.0
    market_cap: float | None = None         # USD market cap of token
    is_memecoin: bool = False
    signal_sources: int = 0                 # how many independent sources


@dataclass
class PortfolioState:
    total_value_usd: float
    stablecoin_value_usd: float
    positions: list[PositionInfo] = field(default_factory=list)
    trades_today: int = 0
    last_trade_times: dict[str, datetime] = field(default_factory=dict)
    drawdown_pct: float = 0.0              # current drawdown from peak


@dataclass
class PositionInfo:
    symbol: str
    value_usd: float
    pnl_pct: float = 0.0


@dataclass
class ValidationResult:
    approved: bool
    violations: list[str] = field(default_factory=list)


# ── Individual check functions ───────────────────────────────────────────
# Each returns None on pass or a human-readable violation string.

def check_position_size(
    proposal: TradeProposal,
    portfolio: PortfolioState,
    limits: RiskLimits,
) -> str | None:
    if portfolio.total_value_usd <= 0:
        return "Portfolio value is zero — cannot evaluate position size"
    trade_value = proposal.amount * proposal.price
    pct = trade_value / portfolio.total_value_usd
    if pct > limits.max_position_pct:
        return (
            f"Position size {pct:.1%} exceeds max {limits.max_position_pct:.0%} "
            f"(${trade_value:,.2f} of ${portfolio.total_value_usd:,.2f})"
        )
    return None


def check_stop_loss(
    proposal: TradeProposal,
    limits: RiskLimits,
) -> str | None:
    if proposal.side == "sell":
        return None
    if proposal.stop_loss_pct is None:
        return "Buy order must include a stop-loss"
    if proposal.stop_loss_pct > 0:
        return f"Stop-loss must be negative, got {proposal.stop_loss_pct}"
    if proposal.stop_loss_pct < limits.stop_loss_pct:
        return (
            f"Stop-loss {proposal.stop_loss_pct:.1%} is looser than limit "
            f"{limits.stop_loss_pct:.1%}"
        )
    return None


def check_stablecoin_reserve(
    proposal: TradeProposal,
    portfolio: PortfolioState,
    limits: RiskLimits,
) -> str | None:
    if proposal.side == "sell":
        return None
    if portfolio.total_value_usd <= 0:
        return "Portfolio value is zero — cannot evaluate stablecoin reserve"
    trade_cost = proposal.amount * proposal.price
    stables_after = portfolio.stablecoin_value_usd - trade_cost
    pct_after = stables_after / portfolio.total_value_usd
    if pct_after < limits.min_stablecoin_pct:
        return (
            f"Stablecoin reserve would drop to {pct_after:.1%} "
            f"(min {limits.min_stablecoin_pct:.0%}). "
            f"Need ${limits.min_stablecoin_pct * portfolio.total_value_usd:,.2f} reserved, "
            f"would have ${stables_after:,.2f}"
        )
    return None


def check_drawdown(
    portfolio: PortfolioState,
    limits: RiskLimits,
) -> str | None:
    if portfolio.drawdown_pct <= limits.max_drawdown_pct:
        return (
            f"Portfolio drawdown {portfolio.drawdown_pct:.1%} exceeds max "
            f"{limits.max_drawdown_pct:.0%} — all new buys blocked"
        )
    return None


def check_daily_trade_limit(
    portfolio: PortfolioState,
    limits: RiskLimits,
) -> str | None:
    if portfolio.trades_today >= limits.max_daily_trades:
        return (
            f"Daily trade limit reached ({portfolio.trades_today}/"
            f"{limits.max_daily_trades})"
        )
    return None


def check_trade_interval(
    proposal: TradeProposal,
    portfolio: PortfolioState,
    limits: RiskLimits,
) -> str | None:
    pair = proposal.pair
    last_time = portfolio.last_trade_times.get(pair)
    if last_time is None:
        return None
    now = datetime.now(timezone.utc)
    elapsed = (now - last_time).total_seconds()
    if elapsed < limits.min_trade_interval_sec:
        remaining = limits.min_trade_interval_sec - elapsed
        return (
            f"Too soon to trade {pair} again — "
            f"{remaining:.0f}s remaining of {limits.min_trade_interval_sec}s cooldown"
        )
    return None


def check_market_cap(
    proposal: TradeProposal,
    limits: RiskLimits,
) -> str | None:
    if proposal.side == "sell":
        return None
    if proposal.market_cap is None:
        return "Market cap data missing — cannot validate"
    if proposal.market_cap < limits.min_market_cap:
        return (
            f"Market cap ${proposal.market_cap:,.0f} below minimum "
            f"${limits.min_market_cap:,.0f}"
        )
    return None


def check_memecoin(proposal: TradeProposal) -> str | None:
    if proposal.side == "sell":
        return None
    base = proposal.pair.split("/")[0]
    if proposal.is_memecoin or base in KNOWN_MEMECOINS:
        if base not in MEMECOIN_WHITELIST:
            return f"{base} is a memecoin and not whitelisted"
    return None


def check_leverage(proposal: TradeProposal) -> str | None:
    if proposal.leverage != 1.0:
        return f"Leverage ({proposal.leverage}x) is forbidden"
    return None


def check_signal_sources(
    proposal: TradeProposal,
    limits: RiskLimits,
) -> str | None:
    if proposal.side == "sell":
        return None
    if proposal.signal_sources < limits.min_signal_sources:
        return (
            f"Only {proposal.signal_sources} signal source(s), "
            f"need at least {limits.min_signal_sources}"
        )
    return None


# ── Main validator ───────────────────────────────────────────────────────

def validate_trade(
    proposal: TradeProposal,
    portfolio: PortfolioState,
    limits: RiskLimits = DEFAULT_LIMITS,
) -> ValidationResult:
    """Run every risk check and return a combined result."""
    violations: list[str] = []

    checks: list[str | None] = [
        check_leverage(proposal),
        check_position_size(proposal, portfolio, limits),
        check_stop_loss(proposal, limits),
        check_stablecoin_reserve(proposal, portfolio, limits),
        check_drawdown(portfolio, limits),
        check_daily_trade_limit(portfolio, limits),
        check_trade_interval(proposal, portfolio, limits),
        check_market_cap(proposal, limits),
        check_memecoin(proposal),
        check_signal_sources(proposal, limits),
    ]

    for result in checks:
        if result is not None:
            violations.append(result)

    return ValidationResult(approved=len(violations) == 0, violations=violations)


# ── Stop-loss scanner (called every cycle) ───────────────────────────────

def scan_stop_losses(
    positions: list[PositionInfo],
    limits: RiskLimits = DEFAULT_LIMITS,
) -> list[PositionInfo]:
    """Return positions that have breached the stop-loss threshold."""
    triggered: list[PositionInfo] = []
    for pos in positions:
        if pos.pnl_pct <= limits.stop_loss_pct:
            triggered.append(pos)
    return triggered


# ── Approval check ───────────────────────────────────────────────────────

def requires_approval(
    proposal: TradeProposal,
    portfolio: PortfolioState,
    limits: RiskLimits = DEFAULT_LIMITS,
) -> bool:
    """True if the trade is large enough to need human confirmation."""
    if portfolio.total_value_usd <= 0:
        return True
    trade_value = proposal.amount * proposal.price
    pct = trade_value / portfolio.total_value_usd
    return pct > limits.auto_trade_max_pct

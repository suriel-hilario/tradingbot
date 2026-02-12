"""Portfolio state tracking — delegates risk checks to risk_management.py.

Keeps the `passes_risk_checks` / `requires_approval` signatures the
orchestrator already imports, but routes through the full validator.
"""

from __future__ import annotations

from typing import Any

from src.tools.risk_management import (
    DEFAULT_LIMITS,
    PortfolioState,
    PositionInfo,
    RiskLimits,
    TradeProposal,
    ValidationResult,
    scan_stop_losses,
    validate_trade,
    requires_approval as _requires_approval,
)


def _build_proposal(trade: dict[str, Any]) -> TradeProposal:
    """Translate the raw dict from Claude tool_use into a TradeProposal."""
    return TradeProposal(
        pair=trade.get("pair", ""),
        side=trade.get("side", "buy"),
        amount=float(trade.get("amount", 0)),
        price=float(trade.get("price", 0)),
        order_type=trade.get("order_type", "limit"),
        stop_loss_pct=trade.get("stop_loss_pct"),
        leverage=float(trade.get("leverage", 1.0)),
        market_cap=trade.get("market_cap"),
        is_memecoin=trade.get("is_memecoin", False),
        signal_sources=int(trade.get("signal_sources", 0)),
    )


def _build_portfolio(state: dict[str, Any] | None = None) -> PortfolioState:
    """Build a PortfolioState from a raw dict (or empty defaults)."""
    if state is None:
        state = {}
    positions = [
        PositionInfo(
            symbol=p.get("symbol", ""),
            value_usd=float(p.get("value_usd", 0)),
            pnl_pct=float(p.get("pnl_pct", 0)),
        )
        for p in state.get("positions", [])
    ]
    return PortfolioState(
        total_value_usd=float(state.get("total_value_usd", 0)),
        stablecoin_value_usd=float(state.get("stablecoin_value_usd", 0)),
        positions=positions,
        trades_today=int(state.get("trades_today", 0)),
        drawdown_pct=float(state.get("drawdown_pct", 0)),
    )


def passes_risk_checks(trade: dict[str, Any], portfolio_state: dict[str, Any] | None = None) -> bool:
    """Validate a proposed trade.  Returns True if all checks pass."""
    proposal = _build_proposal(trade)
    portfolio = _build_portfolio(portfolio_state)
    result = validate_trade(proposal, portfolio)
    return result.approved


def requires_approval(trade: dict[str, Any], portfolio_state: dict[str, Any] | None = None) -> bool:
    """True when the trade needs manual Telegram approval."""
    proposal = _build_proposal(trade)
    portfolio = _build_portfolio(portfolio_state)
    return _requires_approval(proposal, portfolio)


def check_stop_losses(positions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return positions that have hit their stop-loss threshold."""
    infos = [
        PositionInfo(
            symbol=p.get("symbol", ""),
            value_usd=float(p.get("value_usd", 0)),
            pnl_pct=float(p.get("unrealized_pnl_pct", 0)),
        )
        for p in positions
    ]
    triggered = scan_stop_losses(infos)
    return [
        {"symbol": t.symbol, "value_usd": t.value_usd, "pnl_pct": t.pnl_pct}
        for t in triggered
    ]

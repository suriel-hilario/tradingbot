"""Unit tests for risk_management.py — edge cases matter here.

Run:  python -m pytest tests/test_risk_management.py -v
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.tools.risk_management import (
    DEFAULT_LIMITS,
    PortfolioState,
    PositionInfo,
    RiskLimits,
    TradeProposal,
    ValidationResult,
    check_daily_trade_limit,
    check_drawdown,
    check_leverage,
    check_market_cap,
    check_memecoin,
    check_position_size,
    check_signal_sources,
    check_stablecoin_reserve,
    check_stop_loss,
    check_trade_interval,
    requires_approval,
    scan_stop_losses,
    validate_trade,
)

# ── Fixtures ─────────────────────────────────────────────────────────────

LIMITS = RiskLimits(
    max_position_pct=0.10,
    stop_loss_pct=-0.08,
    max_drawdown_pct=-0.20,
    min_stablecoin_pct=0.30,
    min_market_cap=50_000_000,
    max_daily_trades=5,
    min_trade_interval_sec=3600,
    min_signal_sources=2,
    auto_trade_max_pct=0.03,
)


def _portfolio(
    total: float = 100_000,
    stables: float = 50_000,
    trades_today: int = 0,
    drawdown: float = 0.0,
    last_trade_times: dict | None = None,
) -> PortfolioState:
    return PortfolioState(
        total_value_usd=total,
        stablecoin_value_usd=stables,
        trades_today=trades_today,
        drawdown_pct=drawdown,
        last_trade_times=last_trade_times or {},
    )


def _proposal(
    pair: str = "BTC/USDT",
    side: str = "buy",
    amount: float = 0.01,
    price: float = 50_000.0,
    stop_loss_pct: float | None = -0.08,
    leverage: float = 1.0,
    market_cap: float | None = 500_000_000_000,
    is_memecoin: bool = False,
    signal_sources: int = 3,
) -> TradeProposal:
    return TradeProposal(
        pair=pair,
        side=side,
        amount=amount,
        price=price,
        stop_loss_pct=stop_loss_pct,
        leverage=leverage,
        market_cap=market_cap,
        is_memecoin=is_memecoin,
        signal_sources=signal_sources,
    )


# =====================================================================
#  1. Position size
# =====================================================================

class TestPositionSize:
    def test_within_limit(self):
        # 0.01 * 50k = $500 = 0.5% of $100k → ok
        assert check_position_size(_proposal(), _portfolio(), LIMITS) is None

    def test_exactly_at_limit(self):
        # $10k = 10% of $100k → ok
        p = _proposal(amount=0.2, price=50_000)
        assert check_position_size(p, _portfolio(), LIMITS) is None

    def test_exceeds_limit(self):
        # $11k = 11% → fail
        p = _proposal(amount=0.22, price=50_000)
        result = check_position_size(p, _portfolio(), LIMITS)
        assert result is not None
        assert "exceeds max" in result

    def test_zero_portfolio(self):
        result = check_position_size(_proposal(), _portfolio(total=0), LIMITS)
        assert result is not None
        assert "zero" in result.lower()

    def test_large_position_small_portfolio(self):
        # $500 trade vs $1000 portfolio = 50%
        p = _proposal(amount=0.01, price=50_000)
        result = check_position_size(p, _portfolio(total=1_000), LIMITS)
        assert result is not None


# =====================================================================
#  2. Stop-loss
# =====================================================================

class TestStopLoss:
    def test_valid_stop_loss(self):
        assert check_stop_loss(_proposal(stop_loss_pct=-0.05), LIMITS) is None

    def test_exactly_at_limit(self):
        # -8% is the limit itself — allowed (not looser)
        assert check_stop_loss(_proposal(stop_loss_pct=-0.08), LIMITS) is None

    def test_missing_stop_loss(self):
        result = check_stop_loss(_proposal(stop_loss_pct=None), LIMITS)
        assert result is not None
        assert "must include" in result.lower()

    def test_positive_stop_loss(self):
        result = check_stop_loss(_proposal(stop_loss_pct=0.05), LIMITS)
        assert result is not None
        assert "negative" in result.lower()

    def test_too_loose_stop_loss(self):
        # -10% is looser than -8% (further from zero)
        result = check_stop_loss(_proposal(stop_loss_pct=-0.10), LIMITS)
        assert result is not None
        assert "looser" in result.lower()

    def test_sell_order_skips_check(self):
        # Sells don't need a stop-loss
        assert check_stop_loss(_proposal(side="sell", stop_loss_pct=None), LIMITS) is None


# =====================================================================
#  3. Stablecoin reserve
# =====================================================================

class TestStablecoinReserve:
    def test_enough_reserve(self):
        # $50k stables - $500 trade = $49.5k, still 49.5% of $100k
        assert check_stablecoin_reserve(_proposal(), _portfolio(), LIMITS) is None

    def test_reserve_drops_below_min(self):
        # $35k stables - $10k trade = $25k = 25% < 30%
        p = _proposal(amount=0.2, price=50_000)
        port = _portfolio(stables=35_000)
        result = check_stablecoin_reserve(p, port, LIMITS)
        assert result is not None
        assert "reserve" in result.lower()

    def test_exactly_at_min(self):
        # stables after = exactly 30% → passes (not below)
        # $30k - 0 = 30% of $100k
        p = _proposal(amount=0, price=50_000)
        port = _portfolio(stables=30_000)
        assert check_stablecoin_reserve(p, port, LIMITS) is None

    def test_sell_order_skips_check(self):
        port = _portfolio(stables=0)
        assert check_stablecoin_reserve(
            _proposal(side="sell"), port, LIMITS
        ) is None

    def test_zero_portfolio(self):
        result = check_stablecoin_reserve(
            _proposal(), _portfolio(total=0), LIMITS
        )
        assert result is not None


# =====================================================================
#  4. Drawdown
# =====================================================================

class TestDrawdown:
    def test_no_drawdown(self):
        assert check_drawdown(_portfolio(drawdown=0.0), LIMITS) is None

    def test_mild_drawdown(self):
        assert check_drawdown(_portfolio(drawdown=-0.10), LIMITS) is None

    def test_at_limit(self):
        # -20% exactly → fail (breached)
        result = check_drawdown(_portfolio(drawdown=-0.20), LIMITS)
        assert result is not None
        assert "drawdown" in result.lower()

    def test_severe_drawdown(self):
        result = check_drawdown(_portfolio(drawdown=-0.35), LIMITS)
        assert result is not None

    def test_positive_drawdown_passes(self):
        # Unusual but shouldn't fail
        assert check_drawdown(_portfolio(drawdown=0.05), LIMITS) is None


# =====================================================================
#  5. Daily trade limit
# =====================================================================

class TestDailyTradeLimit:
    def test_under_limit(self):
        assert check_daily_trade_limit(_portfolio(trades_today=0), LIMITS) is None
        assert check_daily_trade_limit(_portfolio(trades_today=4), LIMITS) is None

    def test_at_limit(self):
        result = check_daily_trade_limit(_portfolio(trades_today=5), LIMITS)
        assert result is not None
        assert "limit reached" in result.lower()

    def test_over_limit(self):
        result = check_daily_trade_limit(_portfolio(trades_today=10), LIMITS)
        assert result is not None


# =====================================================================
#  6. Trade interval
# =====================================================================

class TestTradeInterval:
    def test_no_previous_trade(self):
        assert check_trade_interval(_proposal(), _portfolio(), LIMITS) is None

    def test_enough_time_passed(self):
        old = datetime.now(timezone.utc) - timedelta(hours=2)
        port = _portfolio(last_trade_times={"BTC/USDT": old})
        assert check_trade_interval(_proposal(), port, LIMITS) is None

    def test_too_soon(self):
        recent = datetime.now(timezone.utc) - timedelta(minutes=30)
        port = _portfolio(last_trade_times={"BTC/USDT": recent})
        result = check_trade_interval(_proposal(), port, LIMITS)
        assert result is not None
        assert "too soon" in result.lower()

    def test_different_pair_ok(self):
        recent = datetime.now(timezone.utc) - timedelta(minutes=5)
        port = _portfolio(last_trade_times={"ETH/USDT": recent})
        assert check_trade_interval(_proposal(pair="BTC/USDT"), port, LIMITS) is None

    def test_exactly_at_boundary(self):
        boundary = datetime.now(timezone.utc) - timedelta(seconds=3600)
        port = _portfolio(last_trade_times={"BTC/USDT": boundary})
        # At exactly the boundary — elapsed >= interval, should pass
        assert check_trade_interval(_proposal(), port, LIMITS) is None


# =====================================================================
#  7. Market cap
# =====================================================================

class TestMarketCap:
    def test_high_market_cap(self):
        assert check_market_cap(_proposal(market_cap=500e9), LIMITS) is None

    def test_exactly_at_min(self):
        assert check_market_cap(_proposal(market_cap=50e6), LIMITS) is None

    def test_below_min(self):
        result = check_market_cap(_proposal(market_cap=10e6), LIMITS)
        assert result is not None
        assert "below minimum" in result.lower()

    def test_missing_market_cap(self):
        result = check_market_cap(_proposal(market_cap=None), LIMITS)
        assert result is not None
        assert "missing" in result.lower()

    def test_sell_skips_check(self):
        assert check_market_cap(_proposal(side="sell", market_cap=None), LIMITS) is None


# =====================================================================
#  8. Memecoin
# =====================================================================

class TestMemecoin:
    def test_normal_coin(self):
        assert check_memecoin(_proposal(pair="BTC/USDT")) is None

    def test_flagged_memecoin(self):
        result = check_memecoin(_proposal(pair="DOGE/USDT", is_memecoin=True))
        assert result is not None
        assert "memecoin" in result.lower()

    def test_known_memecoin_not_flagged(self):
        # SHIB is in KNOWN_MEMECOINS, detected even without is_memecoin=True
        result = check_memecoin(_proposal(pair="SHIB/USDT", is_memecoin=False))
        assert result is not None

    def test_sell_memecoin_ok(self):
        # You can always sell memecoins (exit position)
        assert check_memecoin(_proposal(pair="DOGE/USDT", side="sell", is_memecoin=True)) is None


# =====================================================================
#  9. Leverage
# =====================================================================

class TestLeverage:
    def test_no_leverage(self):
        assert check_leverage(_proposal(leverage=1.0)) is None

    def test_with_leverage(self):
        result = check_leverage(_proposal(leverage=2.0))
        assert result is not None
        assert "forbidden" in result.lower()

    def test_fractional_leverage(self):
        # Even 1.5x is blocked
        result = check_leverage(_proposal(leverage=1.5))
        assert result is not None


# =====================================================================
# 10. Signal sources
# =====================================================================

class TestSignalSources:
    def test_enough_sources(self):
        assert check_signal_sources(_proposal(signal_sources=3), LIMITS) is None

    def test_exactly_min(self):
        assert check_signal_sources(_proposal(signal_sources=2), LIMITS) is None

    def test_too_few(self):
        result = check_signal_sources(_proposal(signal_sources=1), LIMITS)
        assert result is not None
        assert "signal source" in result.lower()

    def test_zero_sources(self):
        result = check_signal_sources(_proposal(signal_sources=0), LIMITS)
        assert result is not None

    def test_sell_skips_check(self):
        assert check_signal_sources(
            _proposal(side="sell", signal_sources=0), LIMITS
        ) is None


# =====================================================================
# 11. Full validate_trade integration
# =====================================================================

class TestValidateTrade:
    def test_clean_trade_passes(self):
        result = validate_trade(_proposal(), _portfolio(), LIMITS)
        assert result.approved is True
        assert result.violations == []

    def test_multiple_violations(self):
        # Overleveraged + memecoin + no stop-loss + too few sources
        p = _proposal(
            pair="DOGE/USDT",
            leverage=3.0,
            stop_loss_pct=None,
            is_memecoin=True,
            signal_sources=0,
        )
        result = validate_trade(p, _portfolio(), LIMITS)
        assert result.approved is False
        assert len(result.violations) >= 3

    def test_sell_is_more_lenient(self):
        # Sell bypasses stop-loss, market-cap, memecoin, signal checks
        p = _proposal(
            side="sell",
            stop_loss_pct=None,
            market_cap=None,
            is_memecoin=True,
            signal_sources=0,
        )
        result = validate_trade(p, _portfolio(), LIMITS)
        assert result.approved is True

    def test_drawdown_blocks_everything(self):
        port = _portfolio(drawdown=-0.25)
        result = validate_trade(_proposal(), port, LIMITS)
        assert result.approved is False
        assert any("drawdown" in v.lower() for v in result.violations)

    def test_custom_limits(self):
        # Relax position limit to 50%
        relaxed = RiskLimits(max_position_pct=0.50)
        big = _proposal(amount=1.0, price=50_000)  # 50% of $100k
        result = validate_trade(big, _portfolio(), relaxed)
        # Position size passes, but other checks still apply
        assert not any("exceeds max" in v for v in result.violations)


# =====================================================================
# 12. Stop-loss scanner
# =====================================================================

class TestScanStopLosses:
    def test_no_positions(self):
        assert scan_stop_losses([], LIMITS) == []

    def test_healthy_positions(self):
        pos = [PositionInfo("BTC", 5000, pnl_pct=-0.03)]
        assert scan_stop_losses(pos, LIMITS) == []

    def test_triggered(self):
        pos = [
            PositionInfo("BTC", 5000, pnl_pct=-0.03),
            PositionInfo("ETH", 3000, pnl_pct=-0.09),
        ]
        result = scan_stop_losses(pos, LIMITS)
        assert len(result) == 1
        assert result[0].symbol == "ETH"

    def test_exactly_at_threshold(self):
        # -8% exactly should trigger
        pos = [PositionInfo("SOL", 2000, pnl_pct=-0.08)]
        assert len(scan_stop_losses(pos, LIMITS)) == 1

    def test_multiple_triggered(self):
        pos = [
            PositionInfo("A", 1000, pnl_pct=-0.10),
            PositionInfo("B", 2000, pnl_pct=-0.15),
            PositionInfo("C", 3000, pnl_pct=-0.01),
        ]
        result = scan_stop_losses(pos, LIMITS)
        assert len(result) == 2
        symbols = {p.symbol for p in result}
        assert symbols == {"A", "B"}


# =====================================================================
# 13. Requires approval
# =====================================================================

class TestRequiresApproval:
    def test_small_trade_auto(self):
        # $500 / $100k = 0.5% < 3% → auto-approved
        assert requires_approval(_proposal(), _portfolio(), LIMITS) is False

    def test_large_trade_needs_approval(self):
        # $5000 / $100k = 5% > 3%
        p = _proposal(amount=0.1, price=50_000)
        assert requires_approval(p, _portfolio(), LIMITS) is True

    def test_exactly_at_threshold(self):
        # 3% exactly → not above → auto
        p = _proposal(amount=0.06, price=50_000)  # $3000 / $100k = 3%
        assert requires_approval(p, _portfolio(), LIMITS) is False

    def test_zero_portfolio_always_needs_approval(self):
        assert requires_approval(_proposal(), _portfolio(total=0), LIMITS) is True

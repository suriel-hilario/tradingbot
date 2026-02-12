"""Crypto agent orchestrator.

Loads agent.md + skills/*.md as the system prompt, then runs a 5-minute
loop that gathers market context, calls Claude with tool definitions,
executes tool_use responses, and logs everything to the database.

A lightweight /health endpoint runs alongside so Docker can probe liveness.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import uuid

import anthropic
from aiohttp import web
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv

from src.tools.exchange import (
    PaperExchangeClient,
    create_exchange_client,
)
from src.tools.risk_management import (
    PortfolioState,
    PositionInfo,
    TradeProposal,
    scan_stop_losses,
    validate_trade,
    requires_approval as rm_requires_approval,
)
from src.tools.notifications import send_alert, send_telegram_message
from src.tools.research import gather_signals, run_research_pipeline

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────

CYCLE_INTERVAL = int(os.getenv("CYCLE_INTERVAL", "300"))
MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")
HEALTH_PORT = int(os.getenv("HEALTH_PORT", "8080"))
PAPER_TRADING = os.getenv("PAPER_TRADING", "true").lower() == "true"
DAILY_SUMMARY_HOUR = int(os.getenv("DAILY_SUMMARY_HOUR", "9"))
MAX_RESTART_DELAY = int(os.getenv("MAX_RESTART_DELAY", "60"))

TOP_COINS = os.getenv(
    "WATCHLIST",
    "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT,"
    "ADA/USDT,AVAX/USDT,DOGE/USDT,DOT/USDT,LINK/USDT",
).split(",")

log = logging.getLogger("orchestrator")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)

# ── System prompt ─────────────────────────────────────────────────────────

def load_md(path: str) -> str:
    return Path(path).read_text()


def build_system_prompt() -> str:
    agent = load_md("agent.md")
    skills_dir = Path("skills")
    parts = [agent, ""]
    for md in sorted(skills_dir.glob("*.md")):
        parts.append(f"## Skill: {md.stem}\n{md.read_text()}")
    mode = "PAPER TRADING" if PAPER_TRADING else "LIVE TRADING"
    parts.append(f"\n## Current Mode\n{mode} — all orders are simulated.")
    return "\n\n---\n\n".join(parts)


# ── Anthropic tool definitions ────────────────────────────────────────────

TOOLS: list[dict[str, Any]] = [
    {
        "name": "check_portfolio",
        "description": (
            "Return the current portfolio state: total value in USD, "
            "USDT balance, and every open position with quantity and "
            "current market value."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_market_data",
        "description": (
            "Fetch current price, 24h change, bid/ask, high/low, and "
            "volume for one or more trading pairs. "
            "If symbols is omitted the top-10 watchlist is returned."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        'List of trading pairs, e.g. ["BTC/USDT","ETH/USDT"]. '
                        "Omit for the full watchlist."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "execute_trade",
        "description": (
            "Place a buy or sell order.  The order goes through risk "
            "management checks first.  Returns fill details or a list "
            "of violations if the trade is rejected."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pair": {
                    "type": "string",
                    "description": 'Trading pair, e.g. "BTC/USDT"',
                },
                "side": {
                    "type": "string",
                    "enum": ["buy", "sell"],
                },
                "amount": {
                    "type": "number",
                    "description": "Quantity of the base asset",
                },
                "order_type": {
                    "type": "string",
                    "enum": ["market", "limit"],
                    "description": "Order type (default: market)",
                },
                "price": {
                    "type": "number",
                    "description": "Limit price — required for limit orders",
                },
                "stop_loss_pct": {
                    "type": "number",
                    "description": "Stop-loss as negative fraction, e.g. -0.08",
                },
                "rationale": {
                    "type": "string",
                    "description": "Why this trade should be executed",
                },
            },
            "required": ["pair", "side", "amount"],
        },
    },
]


# ── Tool execution ────────────────────────────────────────────────────────

class ToolExecutor:
    """Stateful handler that holds the exchange client across cycles."""

    def __init__(self) -> None:
        self.exchange = create_exchange_client()
        self.paused: bool = False
        self._trades_today: int = 0
        self._last_trade_times: dict[str, datetime] = {}
        self._cycle_count: int = 0
        self._last_cycle: float = 0.0

    async def close(self) -> None:
        await self.exchange.close()

    # -- context builders ---------------------------------------------------

    async def get_portfolio_context(self) -> dict[str, Any]:
        portfolio = await self.exchange.get_portfolio()
        portfolio["paper_trading"] = isinstance(self.exchange, PaperExchangeClient)
        portfolio["trades_today"] = self._trades_today
        return portfolio

    async def get_market_context(self, symbols: list[str] | None = None) -> dict[str, Any]:
        pairs = symbols or TOP_COINS
        result: dict[str, Any] = {}
        for pair in pairs:
            try:
                t = await self.exchange.get_ticker(pair)
                result[pair] = t.to_dict()
            except Exception as exc:
                log.warning("Ticker fetch failed for %s: %s", pair, exc)
        return result

    # -- portfolio state for risk checks ------------------------------------

    async def _build_risk_portfolio(self) -> PortfolioState:
        pf = await self.exchange.get_portfolio()
        positions = [
            PositionInfo(
                symbol=p["symbol"],
                value_usd=p.get("value_usd") or 0,
                pnl_pct=p.get("pnl_pct", 0),
            )
            for p in pf.get("positions", [])
        ]
        return PortfolioState(
            total_value_usd=pf.get("total_value_usd", 0),
            stablecoin_value_usd=pf.get("usdt_balance", 0),
            positions=positions,
            trades_today=self._trades_today,
            last_trade_times=dict(self._last_trade_times),
            drawdown_pct=pf.get("drawdown_pct", 0),
        )

    # -- trade execution with risk checks -----------------------------------

    async def execute_trade(self, params: dict[str, Any]) -> dict[str, Any]:
        pair = params["pair"]
        side = params["side"]
        amount = float(params["amount"])
        order_type = params.get("order_type", "market")
        price_param = params.get("price")

        # Fetch current price for risk evaluation
        ticker = await self.exchange.get_ticker(pair)
        price = float(price_param) if price_param else ticker.last

        proposal = TradeProposal(
            pair=pair,
            side=side,
            amount=amount,
            price=price,
            order_type=order_type,
            stop_loss_pct=params.get("stop_loss_pct"),
            market_cap=params.get("market_cap"),
            signal_sources=int(params.get("signal_sources", 2)),
        )
        risk_portfolio = await self._build_risk_portfolio()
        result = validate_trade(proposal, risk_portfolio)

        if not result.approved:
            log.warning("Trade REJECTED: %s", result.violations)
            return {
                "status": "rejected",
                "violations": result.violations,
            }

        needs_approval = rm_requires_approval(proposal, risk_portfolio)
        if needs_approval:
            from src.telegram_bot import notify_pending_trade
            trade_id = uuid.uuid4().hex[:8]
            trade_value = amount * price
            pf_pct = (trade_value / risk_portfolio.total_value_usd * 100
                       if risk_portfolio.total_value_usd > 0 else 100)
            trade_data = {
                "pair": pair,
                "side": side,
                "amount": amount,
                "order_type": order_type,
                "price": price,
                "rationale": params.get("rationale", ""),
                "trade_value_usd": trade_value,
                "portfolio_pct": round(pf_pct, 1),
            }
            await notify_pending_trade(trade_id, trade_data)
            return {
                "status": "pending_approval",
                "trade_id": trade_id,
                "message": f"Trade ({pf_pct:.1f}% of portfolio) sent to Telegram for approval",
            }

        # Execute
        try:
            order = await self.exchange.place_order(
                symbol=pair,
                side=side,
                amount=amount,
                order_type=order_type,
                price=price if order_type == "limit" else None,
            )
        except Exception as exc:
            log.error("Order failed: %s", exc)
            return {"status": "error", "message": str(exc)}

        self._trades_today += 1
        self._last_trade_times[pair] = datetime.now(timezone.utc)

        log.info(
            "TRADE %s %s %s @ $%s (fee $%.4f)",
            order.side, order.amount, order.symbol, order.fill_price, order.fee,
        )
        # Notify Telegram
        from src.telegram_bot import notify_trade_executed
        await notify_trade_executed(order.to_dict())

        return {
            "status": "filled",
            "order": order.to_dict(),
        }

    # -- stop-loss scan -----------------------------------------------------

    async def check_stop_losses(self) -> list[dict[str, Any]]:
        risk_pf = await self._build_risk_portfolio()
        triggered = scan_stop_losses(risk_pf.positions)
        results: list[dict[str, Any]] = []
        for pos in triggered:
            log.warning("STOP-LOSS triggered for %s (%.1f%%)", pos.symbol, pos.pnl_pct * 100)
            try:
                order = await self.exchange.place_order(
                    symbol=f"{pos.symbol}/USDT",
                    side="sell",
                    amount=pos.value_usd,  # approximate; real impl would use qty
                    order_type="market",
                )
                results.append({"symbol": pos.symbol, "order": order.to_dict()})
                await send_alert(
                    "Stop-Loss Executed",
                    f"Sold {pos.symbol} at market — PnL was {pos.pnl_pct:.1%}",
                )
            except Exception as exc:
                log.error("Stop-loss sell failed for %s: %s", pos.symbol, exc)
        return results

    # -- tool dispatch ------------------------------------------------------

    async def dispatch(self, name: str, input_data: dict[str, Any]) -> Any:
        if name == "check_portfolio":
            return await self.get_portfolio_context()
        if name == "get_market_data":
            return await self.get_market_context(input_data.get("symbols"))
        if name == "execute_trade":
            return await self.execute_trade(input_data)
        return {"error": f"Unknown tool: {name}"}


# ── Claude agentic loop (multi-turn tool use) ─────────────────────────────

async def run_agent_cycle(
    client: anthropic.AsyncAnthropic,
    system: str,
    executor: ToolExecutor,
) -> dict[str, Any]:
    """Run one full cycle: build context → Claude → tool loop → done."""

    # Gather context
    portfolio = await executor.get_portfolio_context()
    market = await executor.get_market_context()

    # Research pipeline — graceful fallback if feeds/API are unavailable
    try:
        signals = await gather_signals()
    except Exception as exc:
        log.warning("Research pipeline failed: %s", exc)
        signals = []

    signal_section = "No signals this cycle."
    if signals:
        signal_section = json.dumps(signals, indent=2)

    user_message = (
        f"## Current Portfolio\n```json\n{json.dumps(portfolio, indent=2)}\n```\n\n"
        f"## Market Data (Top 10)\n```json\n{json.dumps(market, indent=2, default=str)}\n```\n\n"
        f"## Research Signals ({len(signals)} articles classified)\n"
        f"```json\n{signal_section}\n```\n\n"
        "Analyze the data above.  Use the tools to inspect positions or "
        "execute trades if you see an opportunity.  If no action is needed, "
        "reply with a brief market summary."
    )

    messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
    actions_taken: list[dict[str, Any]] = []

    # Multi-turn tool-use loop
    for turn in range(10):  # safety cap
        log.info("Claude call #%d", turn + 1)
        response = await client.messages.create(
            model=MODEL,
            system=system,
            messages=messages,
            tools=TOOLS,
            max_tokens=4096,
        )

        # Collect text + tool_use blocks
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        text_blocks = [b.text for b in response.content if b.type == "text"]

        if text_blocks:
            log.info("Claude says: %s", text_blocks[0][:200])

        # If no tool calls, we're done
        if not tool_uses:
            break

        # Build assistant message with full content, then tool results
        messages.append({"role": "assistant", "content": response.content})

        tool_results: list[dict[str, Any]] = []
        for tu in tool_uses:
            log.info("Tool call: %s(%s)", tu.name, json.dumps(tu.input)[:200])
            try:
                result = await executor.dispatch(tu.name, tu.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": json.dumps(result, default=str),
                })
                actions_taken.append({
                    "tool": tu.name,
                    "input": tu.input,
                    "result_summary": _summarize(result),
                })
            except Exception as exc:
                log.error("Tool %s failed: %s", tu.name, exc)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": json.dumps({"error": str(exc)}),
                    "is_error": True,
                })
                actions_taken.append({
                    "tool": tu.name,
                    "input": tu.input,
                    "error": str(exc),
                })

        messages.append({"role": "user", "content": tool_results})

        if response.stop_reason == "end_turn":
            break

    return {
        "model": response.model,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
        "response_content": [
            b.text for b in response.content if b.type == "text"
        ],
        "actions": actions_taken,
    }


def _summarize(result: Any) -> str:
    """Short string for logging; avoids dumping huge dicts."""
    s = json.dumps(result, default=str)
    return s[:300] + "..." if len(s) > 300 else s


# ── DB logging (graceful — won't crash if DB is unavailable) ──────────────

async def log_decision(cycle_result: dict[str, Any]) -> None:
    try:
        from src.db.models import AgentDecision, async_session

        async with async_session() as session:
            decision = AgentDecision(
                model=cycle_result.get("model", MODEL),
                prompt_tokens=cycle_result.get("usage", {}).get("input_tokens", 0),
                completion_tokens=cycle_result.get("usage", {}).get("output_tokens", 0),
                claude_response={"text": cycle_result.get("response_content", [])},
                actions_taken=cycle_result.get("actions", []),
            )
            session.add(decision)
            await session.commit()
            log.info("Logged decision id=%s", decision.id)
    except Exception as exc:
        log.warning("DB logging skipped: %s", exc)


# ── Health endpoint ───────────────────────────────────────────────────────

_start_time = time.time()
_executor_ref: ToolExecutor | None = None


async def health_handler(request: web.Request) -> web.Response:
    uptime = time.time() - _start_time
    data: dict[str, Any] = {
        "status": "ok",
        "uptime_seconds": round(uptime, 1),
        "paper_trading": PAPER_TRADING,
        "model": MODEL,
    }
    if _executor_ref:
        data["cycles_completed"] = _executor_ref._cycle_count
        data["trades_today"] = _executor_ref._trades_today
    return web.json_response(data)


async def start_health_server() -> web.AppRunner:
    app = web.Application()
    app.router.add_get("/health", health_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", HEALTH_PORT)
    await site.start()
    log.info("Health endpoint listening on :%d/health", HEALTH_PORT)
    return runner


# ── Main loop ─────────────────────────────────────────────────────────────

async def agent_loop() -> None:
    global _executor_ref

    # Ensure DB tables exist before anything else
    try:
        from src.db.init_db import init as init_db
        await init_db()
    except Exception as exc:
        log.warning("DB init skipped: %s", exc)

    claude = anthropic.AsyncAnthropic()
    system = build_system_prompt()
    executor = ToolExecutor()
    _executor_ref = executor

    health_runner = await start_health_server()

    # Start Telegram bot as a concurrent async task
    bot_app = None
    try:
        from src.telegram_bot import start_bot, stop_bot
        bot_app = await start_bot(executor)
    except Exception as exc:
        log.warning("Telegram bot failed to start: %s", exc)

    # APScheduler — daily summary at configured hour (default 9am UTC)
    scheduler = AsyncIOScheduler(timezone="UTC")
    try:
        from src.telegram_bot import send_daily_summary
        scheduler.add_job(
            send_daily_summary,
            CronTrigger(hour=DAILY_SUMMARY_HOUR, minute=0, timezone="UTC"),
            id="daily_summary",
            name="Daily portfolio summary to Telegram",
            replace_existing=True,
        )
        scheduler.start()
        log.info("APScheduler started — daily summary at %02d:00 UTC", DAILY_SUMMARY_HOUR)
    except Exception as exc:
        log.warning("APScheduler failed to start: %s", exc)

    log.info(
        "Agent starting — model=%s  cycle=%ds  paper=%s  watchlist=%s",
        MODEL, CYCLE_INTERVAL, PAPER_TRADING, ",".join(TOP_COINS),
    )

    consecutive_failures = 0
    MAX_BACKOFF = 600  # cap at 10 minutes

    try:
        while True:
            # Check paused flag (toggled by /stop and /resume Telegram commands)
            if executor.paused:
                log.info("Agent paused — skipping cycle")
                await asyncio.sleep(CYCLE_INTERVAL)
                continue

            cycle_start = time.time()
            executor._cycle_count += 1
            log.info("═══ Cycle %d ═══", executor._cycle_count)

            try:
                # Stop-loss scan first (automatic, no approval needed)
                stop_loss_results = await executor.check_stop_losses()
                if stop_loss_results:
                    log.warning("Stop-losses triggered: %d", len(stop_loss_results))

                # Main agent cycle
                result = await run_agent_cycle(claude, system, executor)
                await log_decision(result)

                elapsed = time.time() - cycle_start
                log.info(
                    "Cycle %d done in %.1fs — %d actions, %d input / %d output tokens",
                    executor._cycle_count,
                    elapsed,
                    len(result.get("actions", [])),
                    result.get("usage", {}).get("input_tokens", 0),
                    result.get("usage", {}).get("output_tokens", 0),
                )
                consecutive_failures = 0  # reset on success

            except anthropic.APIError as exc:
                consecutive_failures += 1
                log.error("Claude API error (failure #%d): %s", consecutive_failures, exc)
            except Exception as exc:
                consecutive_failures += 1
                log.exception("Cycle %d failed (failure #%d): %s",
                              executor._cycle_count, consecutive_failures, exc)

            executor._last_cycle = time.time()

            # Reset daily counter at midnight UTC
            now = datetime.now(timezone.utc)
            if now.hour == 0 and now.minute < (CYCLE_INTERVAL // 60):
                executor._trades_today = 0
                log.info("Daily trade counter reset")

            # Exponential backoff on consecutive failures
            if consecutive_failures > 0:
                backoff = min(CYCLE_INTERVAL * (2 ** (consecutive_failures - 1)), MAX_BACKOFF)
                log.warning("Backing off %ds after %d consecutive failures", backoff, consecutive_failures)
                await asyncio.sleep(backoff)
            else:
                await asyncio.sleep(CYCLE_INTERVAL)

    finally:
        if scheduler.running:
            scheduler.shutdown(wait=False)
        if bot_app:
            await stop_bot(bot_app)
        await executor.close()
        await health_runner.cleanup()


# ── Entrypoint with auto-restart ──────────────────────────────────────────

def main() -> None:
    """Run the agent loop with automatic restart on crash.

    Uses exponential backoff between restarts, capped at MAX_RESTART_DELAY
    seconds.  KeyboardInterrupt exits cleanly.
    """
    restart_count = 0

    while True:
        try:
            log.info("Starting agent loop (restart #%d)", restart_count)
            asyncio.run(agent_loop())
            break  # clean exit
        except KeyboardInterrupt:
            log.info("Interrupted — shutting down")
            break
        except Exception as exc:
            restart_count += 1
            delay = min(5 * (2 ** (restart_count - 1)), MAX_RESTART_DELAY)
            log.critical(
                "Agent loop crashed (restart #%d): %s — restarting in %ds",
                restart_count, exc, delay,
            )
            try:
                import time as _time
                _time.sleep(delay)
            except KeyboardInterrupt:
                log.info("Interrupted during restart delay — shutting down")
                break


if __name__ == "__main__":
    main()

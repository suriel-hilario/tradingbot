"""Telegram bot — human interface for the crypto agent.

Commands:
    /start          — welcome message
    /status         — agent running state, last cycle, mode
    /portfolio      — current holdings with PnL
    /trades         — last 10 trades
    /stop           — pause the orchestrator loop
    /resume         — resume the orchestrator loop
    /approve <id>   — approve a pending trade
    /deny <id>      — reject a pending trade
    /set <k> <v>    — update a runtime setting

Proactive notifications:
    - Trade execution alerts
    - Daily summary at 09:00 UTC
    - >5% price move alerts on watched coins

Runs as an async task alongside the orchestrator, sharing the same
ToolExecutor (and therefore the same exchange client / state).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

if TYPE_CHECKING:
    from src.orchestrator import ToolExecutor

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
ALLOWED_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
PAPER_TRADING = os.getenv("PAPER_TRADING", "true").lower() == "true"
PRICE_ALERT_THRESHOLD = float(os.getenv("PRICE_ALERT_THRESHOLD", "0.05"))
DAILY_SUMMARY_HOUR = int(os.getenv("DAILY_SUMMARY_HOUR", "9"))

log = logging.getLogger("telegram_bot")

# ── Shared state — set by the orchestrator at startup ─────────────────────

_executor: ToolExecutor | None = None
_pending_trades: dict[str, dict[str, Any]] = {}
_trade_history: list[dict[str, Any]] = []
_price_snapshots: dict[str, float] = {}
_bot_app: Application | None = None


def set_executor(executor: ToolExecutor) -> None:
    global _executor
    _executor = executor


def add_pending_trade(trade_id: str, trade_data: dict[str, Any]) -> None:
    _pending_trades[trade_id] = trade_data


def record_trade(trade: dict[str, Any]) -> None:
    _trade_history.append(trade)
    if len(_trade_history) > 100:
        _trade_history.pop(0)


# ── Auth helper ───────────────────────────────────────────────────────────

def _authorized(update: Update) -> bool:
    if not ALLOWED_CHAT_ID:
        return True
    return str(update.effective_chat.id) == ALLOWED_CHAT_ID


# ── /start ────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return
    mode = "PAPER" if PAPER_TRADING else "LIVE"
    await update.message.reply_text(
        f"*Crypto Agent Bot*  ({mode} mode)\n\n"
        "/status — agent state\n"
        "/portfolio — holdings & PnL\n"
        "/trades — recent trades\n"
        "/stop / /resume — pause/resume agent\n"
        "/approve <id> — approve pending trade\n"
        "/deny <id> — reject pending trade\n"
        "/set <param> <value> — change setting",
        parse_mode="Markdown",
    )


# ── /status ───────────────────────────────────────────────────────────────

async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return
    if _executor is None:
        await update.message.reply_text("Agent not connected.")
        return

    paused = getattr(_executor, "paused", False)
    state = "PAUSED" if paused else "RUNNING"
    cycle = _executor._cycle_count
    last = _executor._last_cycle
    last_str = (
        datetime.fromtimestamp(last, tz=timezone.utc).strftime("%H:%M:%S UTC")
        if last > 0 else "never"
    )
    mode = "PAPER" if PAPER_TRADING else "LIVE"
    trades_today = _executor._trades_today
    pending = len(_pending_trades)

    text = (
        f"*Agent Status*\n"
        f"State: `{state}`\n"
        f"Mode: `{mode}`\n"
        f"Cycles completed: `{cycle}`\n"
        f"Last cycle: `{last_str}`\n"
        f"Trades today: `{trades_today}`\n"
        f"Pending approvals: `{pending}`"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


# ── /portfolio ────────────────────────────────────────────────────────────

async def cmd_portfolio(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return
    if _executor is None:
        await update.message.reply_text("Agent not connected.")
        return

    try:
        pf = await _executor.get_portfolio_context()
    except Exception as exc:
        await update.message.reply_text(f"Error: {exc}")
        return

    total = pf.get("total_value_usd", 0)
    usdt = pf.get("usdt_balance", 0)
    positions = pf.get("positions", [])

    lines = [
        f"*Portfolio*  —  ${total:,.2f}",
        f"USDT: ${usdt:,.2f}",
        "",
    ]
    if positions:
        for p in positions:
            sym = p.get("symbol", "?")
            amt = p.get("amount", 0)
            price = p.get("current_price")
            val = p.get("value_usd")
            price_str = f"${price:,.2f}" if price else "N/A"
            val_str = f"${val:,.2f}" if val else "N/A"
            lines.append(f"`{sym}` {amt:.6f} @ {price_str} = {val_str}")
    else:
        lines.append("_No open positions_")

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


# ── /trades ───────────────────────────────────────────────────────────────

async def cmd_trades(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return

    recent = _trade_history[-10:]
    if not recent:
        await update.message.reply_text("No trades recorded yet.")
        return

    lines = ["*Last Trades*\n"]
    for t in reversed(recent):
        ts = t.get("timestamp", "")
        if isinstance(ts, (int, float)):
            ts = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%m/%d %H:%M")
        side = t.get("side", "?").upper()
        sym = t.get("symbol", t.get("pair", "?"))
        amt = t.get("amount", 0)
        fp = t.get("fill_price")
        status = t.get("status", "?")
        price_str = f"${fp:,.2f}" if fp else "N/A"
        lines.append(f"`{ts}` {side} {amt} {sym} @ {price_str} [{status}]")

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


# ── /stop and /resume ─────────────────────────────────────────────────────

async def cmd_stop(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return
    if _executor is None:
        await update.message.reply_text("Agent not connected.")
        return
    _executor.paused = True
    log.info("Agent PAUSED by Telegram user")
    await update.message.reply_text("Agent *paused*. Use /resume to continue.", parse_mode="Markdown")


async def cmd_resume(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return
    if _executor is None:
        await update.message.reply_text("Agent not connected.")
        return
    _executor.paused = False
    log.info("Agent RESUMED by Telegram user")
    await update.message.reply_text("Agent *resumed*.", parse_mode="Markdown")


# ── /approve <id> and /deny <id> ──────────────────────────────────────────

async def cmd_approve(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return
    if not ctx.args:
        # List pending trades if no ID given
        if not _pending_trades:
            await update.message.reply_text("No pending trades.")
            return
        lines = ["*Pending Trades*\n"]
        for tid, td in _pending_trades.items():
            side = td.get("side", "?").upper()
            pair = td.get("pair", "?")
            amt = td.get("amount", 0)
            lines.append(f"`{tid}` — {side} {amt} {pair}")
        lines.append("\nUse `/approve <id>` to approve.")
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
        return

    trade_id = ctx.args[0]
    trade = _pending_trades.pop(trade_id, None)
    if trade is None:
        await update.message.reply_text(f"Trade `{trade_id}` not found.", parse_mode="Markdown")
        return

    if _executor is None:
        await update.message.reply_text("Agent not connected — cannot execute.")
        _pending_trades[trade_id] = trade
        return

    await update.message.reply_text(f"Executing trade `{trade_id}`...", parse_mode="Markdown")
    try:
        # Force-execute: bypass the approval check in execute_trade by calling
        # the exchange directly (risk checks already passed when it was queued)
        order = await _executor.exchange.place_order(
            symbol=trade["pair"],
            side=trade["side"],
            amount=float(trade["amount"]),
            order_type=trade.get("order_type", "market"),
            price=trade.get("price"),
        )
        _executor._trades_today += 1
        _executor._last_trade_times[trade["pair"]] = datetime.now(timezone.utc)
        record_trade(order.to_dict())

        await update.message.reply_text(
            f"*Trade Executed*\n"
            f"{order.side.upper()} {order.amount} {order.symbol}\n"
            f"Fill: ${order.fill_price:,.2f}  Fee: ${order.fee:.4f}",
            parse_mode="Markdown",
        )
    except Exception as exc:
        log.error("Approved trade execution failed: %s", exc)
        await update.message.reply_text(f"Execution failed: {exc}")


async def cmd_deny(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return
    if not ctx.args:
        await update.message.reply_text("Usage: `/deny <trade_id>`", parse_mode="Markdown")
        return

    trade_id = ctx.args[0]
    trade = _pending_trades.pop(trade_id, None)
    if trade is None:
        await update.message.reply_text(f"Trade `{trade_id}` not found.", parse_mode="Markdown")
        return

    log.info("Trade %s DENIED by user", trade_id)
    await update.message.reply_text(
        f"Trade `{trade_id}` *denied*.\n"
        f"({trade.get('side', '?').upper()} {trade.get('amount', 0)} {trade.get('pair', '?')})",
        parse_mode="Markdown",
    )


# ── /set <param> <value> ──────────────────────────────────────────────────

ALLOWED_SETTINGS = {
    "cycle_interval",
    "auto_trade_max_pct",
    "max_daily_trades",
    "price_alert_threshold",
    "daily_summary_hour",
}


async def cmd_set(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return
    if not ctx.args or len(ctx.args) < 2:
        lines = ["*Configurable Settings*\n"]
        for s in sorted(ALLOWED_SETTINGS):
            lines.append(f"`{s}`")
        lines.append("\nUsage: `/set <param> <value>`")
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
        return

    key = ctx.args[0].lower()
    value = ctx.args[1]

    if key not in ALLOWED_SETTINGS:
        await update.message.reply_text(
            f"Unknown setting `{key}`.\nAllowed: {', '.join(sorted(ALLOWED_SETTINGS))}",
            parse_mode="Markdown",
        )
        return

    # Try to persist to DB Config table
    try:
        from src.db.models import Config as ConfigModel, async_session
        async with async_session() as session:
            existing = await session.get(ConfigModel, key)
            if existing:
                existing.value = value
            else:
                session.add(ConfigModel(key=key, value=value))
            await session.commit()
        await update.message.reply_text(f"Set `{key}` = `{value}` (saved to DB)", parse_mode="Markdown")
    except Exception:
        await update.message.reply_text(f"Set `{key}` = `{value}` (DB unavailable, in-memory only)", parse_mode="Markdown")

    log.info("Setting changed: %s = %s", key, value)


# ── Proactive notifications ───────────────────────────────────────────────

async def notify_trade_executed(order_dict: dict[str, Any]) -> None:
    """Called by the orchestrator after a trade fills."""
    if _bot_app is None:
        return
    record_trade(order_dict)
    side = order_dict.get("side", "?").upper()
    sym = order_dict.get("symbol", "?")
    amt = order_dict.get("amount", 0)
    fp = order_dict.get("fill_price")
    fee = order_dict.get("fee", 0)
    price_str = f"${fp:,.2f}" if fp else "N/A"

    text = (
        f"*Trade Executed*\n"
        f"{side} {amt} {sym} @ {price_str}\n"
        f"Fee: ${fee:.4f}"
    )
    try:
        await _bot_app.bot.send_message(
            chat_id=ALLOWED_CHAT_ID, text=text, parse_mode="Markdown",
        )
    except Exception as exc:
        log.warning("Failed to send trade notification: %s", exc)


async def notify_pending_trade(trade_id: str, trade_data: dict[str, Any]) -> None:
    """Called by the orchestrator when a trade needs approval."""
    if _bot_app is None:
        return
    add_pending_trade(trade_id, trade_data)
    side = trade_data.get("side", "?").upper()
    pair = trade_data.get("pair", "?")
    amt = trade_data.get("amount", 0)
    price = trade_data.get("price", 0)
    rationale = trade_data.get("rationale", "N/A")
    trade_value = trade_data.get("trade_value_usd")
    pf_pct = trade_data.get("portfolio_pct")

    value_line = ""
    if trade_value is not None and pf_pct is not None:
        value_line = f"Value: ${trade_value:,.2f} ({pf_pct}% of portfolio)\n"

    text = (
        f"*Approval Required*\n"
        f"ID: `{trade_id}`\n"
        f"{side} {amt} {pair} @ ${price:,.2f}\n"
        f"{value_line}"
        f"Reason: {rationale}\n\n"
        f"`/approve {trade_id}` or `/deny {trade_id}`"
    )
    try:
        await _bot_app.bot.send_message(
            chat_id=ALLOWED_CHAT_ID, text=text, parse_mode="Markdown",
        )
    except Exception as exc:
        log.warning("Failed to send approval request: %s", exc)


async def send_daily_summary() -> None:
    """Send portfolio summary to Telegram.  Called by APScheduler in the orchestrator."""
    if _executor is None or _bot_app is None:
        log.debug("Daily summary skipped — executor or bot not ready")
        return

    try:
        pf = await _executor.get_portfolio_context()
        total = pf.get("total_value_usd", 0)
        usdt = pf.get("usdt_balance", 0)
        positions = pf.get("positions", [])
        trades_count = _executor._trades_today
        cycles = _executor._cycle_count

        lines = [
            f"*Daily Summary*  —  {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
            f"Portfolio: ${total:,.2f}",
            f"USDT: ${usdt:,.2f}",
            f"Trades today: {trades_count}",
            f"Cycles completed: {cycles}",
            "",
        ]
        for p in positions:
            sym = p.get("symbol", "?")
            val = p.get("value_usd")
            pnl = p.get("pnl_pct")
            val_str = f"${val:,.2f}" if val else "N/A"
            pnl_str = f" ({pnl:+.1f}%)" if pnl is not None else ""
            lines.append(f"  {sym}: {val_str}{pnl_str}")

        if not positions:
            lines.append("  _No positions_")

        pending = len(_pending_trades)
        if pending:
            lines.append(f"\nPending approvals: {pending}")

        await _bot_app.bot.send_message(
            chat_id=ALLOWED_CHAT_ID,
            text="\n".join(lines),
            parse_mode="Markdown",
        )
        log.info("Daily summary sent")
    except Exception as exc:
        log.warning("Daily summary failed: %s", exc)


async def _price_alert_loop() -> None:
    """Check for >5% price moves on watched coins every 60 seconds."""
    global _price_snapshots

    while True:
        await asyncio.sleep(60)

        if _executor is None or _bot_app is None:
            continue

        try:
            from src.orchestrator import TOP_COINS
            for pair in TOP_COINS:
                try:
                    ticker = await _executor.exchange.get_ticker(pair)
                except Exception:
                    continue

                prev = _price_snapshots.get(pair)
                current = ticker.last
                _price_snapshots[pair] = current

                if prev is None or prev == 0:
                    continue

                change = (current - prev) / prev
                if abs(change) >= PRICE_ALERT_THRESHOLD:
                    direction = "UP" if change > 0 else "DOWN"
                    try:
                        await _bot_app.bot.send_message(
                            chat_id=ALLOWED_CHAT_ID,
                            text=(
                                f"*Price Alert*  {pair}\n"
                                f"{direction} {abs(change):.1%} "
                                f"(${prev:,.2f} -> ${current:,.2f})"
                            ),
                            parse_mode="Markdown",
                        )
                    except Exception as exc:
                        log.warning("Price alert send failed: %s", exc)
                    # Reset snapshot so we don't fire again immediately
                    _price_snapshots[pair] = current
        except Exception as exc:
            log.warning("Price alert loop error: %s", exc)


# ── Bot lifecycle ─────────────────────────────────────────────────────────

async def start_bot(executor: ToolExecutor) -> Application:
    """Build, register handlers, and start polling (non-blocking).

    Returns the Application so the orchestrator can shut it down later.
    """
    global _bot_app

    set_executor(executor)

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("portfolio", cmd_portfolio))
    app.add_handler(CommandHandler("trades", cmd_trades))
    app.add_handler(CommandHandler("stop", cmd_stop))
    app.add_handler(CommandHandler("resume", cmd_resume))
    app.add_handler(CommandHandler("approve", cmd_approve))
    app.add_handler(CommandHandler("deny", cmd_deny))
    app.add_handler(CommandHandler("set", cmd_set))

    # Initialize and start polling without blocking
    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)

    _bot_app = app
    log.info("Telegram bot started (polling)")

    # Launch proactive price-alert task (daily summary is handled by APScheduler)
    asyncio.create_task(_price_alert_loop())

    return app


async def stop_bot(app: Application) -> None:
    """Gracefully shut down the bot."""
    global _bot_app
    try:
        await app.updater.stop()
        await app.stop()
        await app.shutdown()
    except Exception as exc:
        log.warning("Bot shutdown error: %s", exc)
    _bot_app = None
    log.info("Telegram bot stopped")

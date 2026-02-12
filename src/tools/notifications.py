"""Telegram notifications and webhook helpers."""

import os
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"


async def send_telegram_message(text: str) -> None:
    """Send a message to the configured Telegram chat."""
    async with httpx.AsyncClient() as client:
        await client.post(
            f"{TELEGRAM_API}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"},
        )


async def request_telegram_approval(tool_call: Any) -> None:
    """Send a trade proposal to Telegram for manual approval."""
    msg = (
        f"*Trade Approval Requested*\n"
        f"Tool: `{tool_call.name}`\n"
        f"Input: `{tool_call.input}`\n"
        f"\nReply /approve or /reject"
    )
    await send_telegram_message(msg)


async def send_alert(title: str, body: str) -> None:
    """Send an alert notification (price moves, stop-loss triggers)."""
    await send_telegram_message(f"*{title}*\n{body}")

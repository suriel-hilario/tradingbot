"""Exchange client — ccxt wrapper with paper-trading simulator.

When PAPER_TRADING=true the client uses an in-memory simulator that fetches
real market prices via ccxt but tracks balances and orders locally.
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import ccxt.async_support as ccxt
from dotenv import load_dotenv

load_dotenv()

PAPER_TRADING = os.getenv("PAPER_TRADING", "true").lower() == "true"
EXCHANGE_ID = os.getenv("EXCHANGE", "binance")
PAPER_STARTING_USDT = float(os.getenv("PAPER_STARTING_USDT", "10000"))


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Ticker:
    symbol: str
    last: float
    bid: float
    ask: float
    high: float
    low: float
    volume: float
    percentage: float | None = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "last": self.last,
            "bid": self.bid,
            "ask": self.ask,
            "high": self.high,
            "low": self.low,
            "volume": self.volume,
            "percentage": self.percentage,
            "timestamp": self.timestamp,
        }


@dataclass
class Order:
    id: str
    symbol: str
    side: str          # "buy" | "sell"
    order_type: str    # "market" | "limit"
    amount: float
    price: float | None
    fill_price: float | None = None
    filled: float = 0.0
    remaining: float = 0.0
    status: str = "open"  # open | closed | canceled
    fee: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side,
            "type": self.order_type,
            "amount": self.amount,
            "price": self.price,
            "fill_price": self.fill_price,
            "filled": self.filled,
            "remaining": self.remaining,
            "status": self.status,
            "fee": self.fee,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Live exchange client (real ccxt)
# ---------------------------------------------------------------------------

class LiveExchangeClient:
    """Thin async wrapper around a ccxt exchange instance."""

    def __init__(self) -> None:
        exchange_cls = getattr(ccxt, EXCHANGE_ID)
        self._exchange: ccxt.Exchange = exchange_cls({
            "apiKey": os.getenv("BINANCE_API_KEY", ""),
            "secret": os.getenv("BINANCE_SECRET", ""),
            "enableRateLimit": True,
        })

    async def close(self) -> None:
        await self._exchange.close()

    async def get_balance(self) -> dict[str, Any]:
        bal = await self._exchange.fetch_balance()
        return {"total": bal.get("total", {}), "free": bal.get("free", {})}

    async def get_ticker(self, symbol: str) -> Ticker:
        t = await self._exchange.fetch_ticker(symbol)
        return Ticker(
            symbol=t["symbol"],
            last=t["last"],
            bid=t["bid"],
            ask=t["ask"],
            high=t["high"],
            low=t["low"],
            volume=t.get("baseVolume", 0),
            percentage=t.get("percentage"),
        )

    async def place_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        order_type: str = "limit",
        price: float | None = None,
    ) -> Order:
        if order_type == "market":
            raw = await self._exchange.create_order(symbol, "market", side, amount)
        else:
            if price is None:
                raise ValueError("price is required for limit orders")
            raw = await self._exchange.create_order(symbol, "limit", side, amount, price)
        return Order(
            id=str(raw["id"]),
            symbol=raw["symbol"],
            side=raw["side"],
            order_type=raw["type"],
            amount=raw["amount"],
            price=raw.get("price"),
            fill_price=raw.get("average"),
            filled=raw.get("filled", 0),
            remaining=raw.get("remaining", amount),
            status=raw.get("status", "open"),
            fee=raw.get("fee", {}).get("cost", 0) if raw.get("fee") else 0,
        )

    async def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        raw_list = await self._exchange.fetch_open_orders(symbol)
        return [
            Order(
                id=str(o["id"]),
                symbol=o["symbol"],
                side=o["side"],
                order_type=o["type"],
                amount=o["amount"],
                price=o.get("price"),
                filled=o.get("filled", 0),
                remaining=o.get("remaining", 0),
                status=o.get("status", "open"),
            )
            for o in raw_list
        ]

    async def cancel_order(self, order_id: str, symbol: str) -> Order:
        raw = await self._exchange.cancel_order(order_id, symbol)
        return Order(
            id=str(raw["id"]),
            symbol=raw.get("symbol", symbol),
            side=raw.get("side", ""),
            order_type=raw.get("type", ""),
            amount=raw.get("amount", 0),
            price=raw.get("price"),
            status="canceled",
        )

    async def get_portfolio(self) -> dict[str, Any]:
        bal = await self._exchange.fetch_balance()
        total = bal.get("total", {})
        positions: list[dict[str, Any]] = []
        for coin, qty in total.items():
            if qty and qty > 0 and coin != "USDT":
                try:
                    t = await self._exchange.fetch_ticker(f"{coin}/USDT")
                    positions.append({
                        "symbol": coin,
                        "amount": qty,
                        "current_price": t["last"],
                        "value_usd": qty * t["last"],
                    })
                except Exception:
                    positions.append({
                        "symbol": coin,
                        "amount": qty,
                        "current_price": None,
                        "value_usd": None,
                    })
        usdt = total.get("USDT", 0) or 0
        total_usd = usdt + sum(p["value_usd"] for p in positions if p["value_usd"])
        return {
            "total_value_usd": total_usd,
            "usdt_balance": usdt,
            "positions": positions,
        }


# ---------------------------------------------------------------------------
# Paper-trading simulator
# ---------------------------------------------------------------------------

class PaperExchangeClient:
    """In-memory simulator.  Fetches real prices via ccxt but keeps balances
    and order state locally.  Simulates immediate fills for market orders and
    fill-on-create for limit orders at the requested price.
    """

    FEE_RATE = 0.001  # 0.1% taker fee

    def __init__(self, starting_usdt: float = PAPER_STARTING_USDT) -> None:
        self._balances: dict[str, float] = {"USDT": starting_usdt}
        self._orders: dict[str, Order] = {}
        # lightweight ccxt instance for price lookups only (no keys needed)
        exchange_cls = getattr(ccxt, EXCHANGE_ID)
        self._price_feed: ccxt.Exchange = exchange_cls({"enableRateLimit": True})

    async def close(self) -> None:
        await self._price_feed.close()

    # -- helpers -------------------------------------------------------------

    def _coin(self, symbol: str) -> tuple[str, str]:
        """'BTC/USDT' -> ('BTC', 'USDT')"""
        base, quote = symbol.split("/")
        return base, quote

    def _adjust_balance(self, coin: str, delta: float) -> None:
        self._balances[coin] = self._balances.get(coin, 0.0) + delta

    async def _fetch_price(self, symbol: str) -> float:
        t = await self._price_feed.fetch_ticker(symbol)
        return float(t["last"])

    def _fill_order(self, order: Order, fill_price: float) -> Order:
        base, quote = self._coin(order.symbol)
        cost = fill_price * order.amount
        fee = cost * self.FEE_RATE

        if order.side == "buy":
            self._adjust_balance(quote, -(cost + fee))
            self._adjust_balance(base, order.amount)
        else:
            self._adjust_balance(base, -order.amount)
            self._adjust_balance(quote, cost - fee)

        order.fill_price = fill_price
        order.filled = order.amount
        order.remaining = 0.0
        order.fee = fee
        order.status = "closed"
        return order

    # -- public API ----------------------------------------------------------

    async def get_balance(self) -> dict[str, Any]:
        # filter out zero balances
        non_zero = {k: v for k, v in self._balances.items() if v > 0}
        return {"total": dict(non_zero), "free": dict(non_zero)}

    async def get_ticker(self, symbol: str) -> Ticker:
        t = await self._price_feed.fetch_ticker(symbol)
        return Ticker(
            symbol=t["symbol"],
            last=t["last"],
            bid=t["bid"],
            ask=t["ask"],
            high=t["high"],
            low=t["low"],
            volume=t.get("baseVolume", 0),
            percentage=t.get("percentage"),
        )

    async def place_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        order_type: str = "limit",
        price: float | None = None,
    ) -> Order:
        base, quote = self._coin(symbol)

        # determine fill price
        if order_type == "market":
            fill_price = await self._fetch_price(symbol)
        else:
            if price is None:
                raise ValueError("price is required for limit orders")
            fill_price = price

        # pre-validate funds
        cost = fill_price * amount
        fee = cost * self.FEE_RATE
        if side == "buy":
            available = self._balances.get(quote, 0.0)
            if available < cost + fee:
                raise ValueError(
                    f"Insufficient {quote}: need {cost + fee:.2f}, have {available:.2f}"
                )
        else:
            available = self._balances.get(base, 0.0)
            if available < amount:
                raise ValueError(
                    f"Insufficient {base}: need {amount}, have {available}"
                )

        order = Order(
            id=uuid.uuid4().hex[:12],
            symbol=symbol,
            side=side,
            order_type=order_type,
            amount=amount,
            price=price,
            remaining=amount,
        )

        # market orders fill immediately; limit orders also fill immediately
        # in the simulator (simplification — real limits would sit on book)
        self._fill_order(order, fill_price)
        self._orders[order.id] = order
        return order

    async def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        return [
            o for o in self._orders.values()
            if o.status == "open" and (symbol is None or o.symbol == symbol)
        ]

    async def cancel_order(self, order_id: str, symbol: str = "") -> Order:
        order = self._orders.get(order_id)
        if order is None:
            raise ValueError(f"Order {order_id} not found")
        if order.status != "open":
            raise ValueError(f"Order {order_id} is already {order.status}")
        order.status = "canceled"
        return order

    async def get_portfolio(self) -> dict[str, Any]:
        positions: list[dict[str, Any]] = []
        for coin, qty in self._balances.items():
            if qty > 0 and coin != "USDT":
                try:
                    price = await self._fetch_price(f"{coin}/USDT")
                    positions.append({
                        "symbol": coin,
                        "amount": qty,
                        "current_price": price,
                        "value_usd": qty * price,
                    })
                except Exception:
                    positions.append({
                        "symbol": coin,
                        "amount": qty,
                        "current_price": None,
                        "value_usd": None,
                    })
        usdt = self._balances.get("USDT", 0.0)
        total_usd = usdt + sum(p["value_usd"] for p in positions if p["value_usd"])
        return {
            "total_value_usd": total_usd,
            "usdt_balance": usdt,
            "positions": positions,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_exchange_client() -> LiveExchangeClient | PaperExchangeClient:
    """Return the right client based on PAPER_TRADING env var."""
    if PAPER_TRADING:
        return PaperExchangeClient()
    return LiveExchangeClient()


# ---------------------------------------------------------------------------
# Module-level helpers (used by the orchestrator)
# ---------------------------------------------------------------------------

async def get_portfolio_state() -> dict[str, Any]:
    client = create_exchange_client()
    try:
        return await client.get_balance()
    finally:
        await client.close()


async def get_market_overview() -> dict[str, Any]:
    client = create_exchange_client()
    try:
        result: dict[str, Any] = {}
        for symbol in ["BTC/USDT", "ETH/USDT"]:
            t = await client.get_ticker(symbol)
            result[symbol] = {"last": t.last, "change": t.percentage}
        return result
    finally:
        await client.close()

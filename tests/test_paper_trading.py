"""Smoke test for the paper-trading simulator.

Run:  python -m tests.test_paper_trading
"""

import asyncio
import json
import sys
import os

# ensure project root is on path when running as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# force paper trading regardless of .env
os.environ["PAPER_TRADING"] = "true"

from src.tools.exchange import PaperExchangeClient


def pp(label: str, obj: object) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    if hasattr(obj, "to_dict"):
        print(json.dumps(obj.to_dict(), indent=2))
    else:
        print(json.dumps(obj, indent=2, default=str))


async def main() -> None:
    client = PaperExchangeClient(starting_usdt=10_000.0)

    try:
        # 1. Starting balance
        balance = await client.get_balance()
        pp("Starting balance", balance)

        # 2. Fetch BTC price
        ticker = await client.get_ticker("BTC/USDT")
        pp("BTC/USDT ticker", ticker)

        # 3. Market buy 0.01 BTC
        print("\n>> Placing market buy: 0.01 BTC/USDT ...")
        buy_order = await client.place_order(
            symbol="BTC/USDT",
            side="buy",
            amount=0.01,
            order_type="market",
        )
        pp("Buy order result", buy_order)

        # 4. Balance after buy
        balance = await client.get_balance()
        pp("Balance after buy", balance)

        # 5. Portfolio view
        portfolio = await client.get_portfolio()
        pp("Portfolio", portfolio)

        # 6. Open orders (should be empty — market fills instantly)
        open_orders = await client.get_open_orders()
        pp("Open orders", open_orders)

        # 7. Limit sell the BTC we just bought
        sell_price = round(ticker.last * 1.05, 2)  # 5% above current
        print(f"\n>> Placing limit sell: 0.01 BTC @ ${sell_price:,.2f} ...")
        sell_order = await client.place_order(
            symbol="BTC/USDT",
            side="sell",
            amount=0.01,
            order_type="limit",
            price=sell_price,
        )
        pp("Sell order result", sell_order)

        # 8. Final balance
        balance = await client.get_balance()
        pp("Final balance", balance)

        # 9. Final portfolio (should have no BTC left)
        portfolio = await client.get_portfolio()
        pp("Final portfolio", portfolio)

        # 10. Verify accounting
        print(f"\n{'='*60}")
        print(f"  Summary")
        print(f"{'='*60}")
        print(f"  Started with:   $10,000.00 USDT")
        print(f"  Bought 0.01 BTC @ ${buy_order.fill_price:,.2f} (fee: ${buy_order.fee:.2f})")
        print(f"  Sold   0.01 BTC @ ${sell_order.fill_price:,.2f} (fee: ${sell_order.fee:.2f})")
        final_usdt = balance["total"].get("USDT", 0)
        net = final_usdt - 10_000.0
        print(f"  Final USDT:     ${final_usdt:,.2f}")
        print(f"  Net P&L:        ${net:+,.2f}")
        print(f"  (profit from 5% markup minus 0.1% fees on each leg)")

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())

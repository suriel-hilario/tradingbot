# Trading Skill
## Exchange: Binance (via ccxt)
## Order Types
- Limit orders preferred (0.3% better than market avg)
- Market orders only for stop-loss execution
- DCA: split entries into 3 tranches over 24h for positions >5% portfolio

## Tool: execute_trade
Parameters: pair, side, amount, order_type, price_limit
Returns: order_id, fill_price, status

## Tool: check_portfolio
Returns: all positions, PnL, available balance
```

**`skills/alerts.md`** — Communication rules for Telegram/dashboard

---

## 3. Tech Stack Recommendation

| Component | Tech | Why |
|-----------|------|-----|
| Orchestrator | **Python** | ccxt, async, mature crypto libs |
| Agent brain | **Claude API** (Sonnet for routine, Opus for big decisions) | Cost balance |
| DB | **PostgreSQL** | Trades, portfolio history, signals log |
| Telegram | **python-telegram-bot** | Mature, async |
| Dashboard | **Next.js + Recharts** or **Streamlit** (faster MVP) | |
| Scheduler | **APScheduler** or **Celery + Redis** | Periodic research runs |
| Exchange | **ccxt** library | Unified API for 100+ exchanges |
| Deployment | **Docker Compose** on DO droplet | Easy to manage |

## 4. Project Structure
```
crypto-agent/
├── agent.md
├── skills/
│   ├── research.md
│   ├── trading.md
│   ├── alerts.md
│   └── risk_management.md
├── src/
│   ├── orchestrator.py        # Main loop: reads agent.md, calls Claude, executes tools
│   ├── tools/
│   │   ├── exchange.py        # ccxt wrapper
│   │   ├── research.py        # Source scrapers/API clients
│   │   ├── portfolio.py       # Portfolio state & risk checks
│   │   └── notifications.py   # Telegram + webhook for dashboard
│   ├── telegram_bot.py        # Command interface (/status, /approve, /override)
│   ├── dashboard/             # Next.js or Streamlit app
│   └── db/
│       ├── models.py          # SQLAlchemy models
│       └── migrations/
├── docker-compose.yml
├── .env                       # API keys (NEVER in repo)
└── CLAUDE.md                  # Instructions for Claude Code to work on this project
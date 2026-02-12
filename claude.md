# Project: Crypto Investment Bot Agent

## Architecture
See agent.md for agent behavior. See skills/ for capabilities.

## Dev Guidelines
- Python 3.12+, async throughout
- Use ccxt for all exchange interactions
- All trades MUST pass risk_management checks before execution
- Never hardcode API keys — use .env
- Every trade and decision gets logged to PostgreSQL
- Telegram bot is the primary human interface
- Dashboard is read-only visualization

## Testing
- Use ccxt sandbox/testnet mode for development
- Paper trading mode must be togglable via .env flag
- Unit tests for risk management are critical — never skip

## Key Commands
- `docker-compose up` to run everything
- `PAPER_TRADING=true` for dry runs
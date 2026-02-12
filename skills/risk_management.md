# Risk Management Skill

## Hard Limits (never override)
- Max single position: 10% of portfolio
- Stop-loss: -8% per position (market sell)
- Max portfolio drawdown: -20%
- Minimum stablecoin reserve: 30%
- No leverage — ever

## Soft Limits (can be adjusted via /override)
- Max daily trade count: 5
- Min time between trades on same pair: 1h
- Max portfolio concentration in single sector: 25%

## Pre-Trade Checklist
1. Position size within 10% limit?
2. Enough stablecoin reserve after trade?
3. Total portfolio exposure still under limits?
4. Token market cap >= $50M?
5. Not a memecoin (unless whitelisted)?
6. At least 2 independent signal sources confirm?

## Tool: validate_trade
Parameters: pair, side, amount, portfolio_state
Returns: { approved: bool, violations: list[str] }

## Stop-Loss Monitoring
- Check positions every cycle (5 min)
- If position hits -8%: execute market sell immediately
- No approval needed for stop-loss — it is automatic

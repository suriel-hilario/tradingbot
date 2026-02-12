# Alerts Skill

## Channels
- Telegram (primary): all alerts go here
- Dashboard (secondary): read-only log

## Alert Types

### Immediate Alerts
- Trade executed (with fill details)
- Stop-loss triggered
- Price move >5% in 1h on held position
- Portfolio drawdown warning at -15%

### Scheduled Reports
- Daily portfolio summary (08:00 UTC)
- Weekly performance report (Sunday 20:00 UTC)

## Tool: send_alert
Parameters: alert_type, title, body, urgency (low/medium/high)
High urgency alerts repeat every 5 minutes until acknowledged.

## Formatting
- Use Markdown for Telegram messages
- Include relevant numbers: price, PnL %, portfolio impact
- Always include timestamp in UTC

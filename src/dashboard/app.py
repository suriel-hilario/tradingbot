"""Streamlit dashboard — read-only portfolio visualization.

Pages:
    1. Portfolio  — line chart of value over time, holdings pie chart, PnL
    2. Trades     — filterable table of all trades
    3. Signals    — latest research signals with sentiment
    4. Agent Log  — decision history with Claude's reasoning

Connects to the same PostgreSQL database as the orchestrator.
Protected by a simple password gate (no external deps).

Run with:
    streamlit run src/dashboard/app.py --server.port 8501
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

# -- DB connection (sync driver for Streamlit) -----------------------------

_raw_url = os.getenv("DB_URL", "postgresql://agent:changeme@db:5432/cryptoagent")
# Ensure we use the sync driver, not asyncpg
DB_URL = _raw_url.replace("postgresql+asyncpg://", "postgresql://", 1)

DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "")


@st.cache_resource
def get_engine():
    eng = create_engine(DB_URL, pool_pre_ping=True)
    # Ensure tables exist (idempotent — no-op if they already exist)
    try:
        from src.db.models import Base
        Base.metadata.create_all(eng)
    except Exception:
        pass  # DB might not be reachable yet; queries will show errors
    return eng


def run_query(query: str, params: dict | None = None) -> pd.DataFrame:
    """Execute a read-only SQL query and return a DataFrame."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            return pd.read_sql(text(query), conn, params=params or {})
    except Exception as exc:
        st.error(f"Database query failed: {exc}")
        return pd.DataFrame()


# -- Auth gate (optional — skipped when DASHBOARD_PASSWORD is not set) -----

def check_password() -> bool:
    """Return True if the user is allowed in.

    When DASHBOARD_PASSWORD is empty/unset, authentication is skipped
    entirely so local development works without friction.
    """
    if not DASHBOARD_PASSWORD:
        return True

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.markdown("### Login")
    password = st.text_input("Password", type="password", key="login_pw")
    if st.button("Enter"):
        expected = hashlib.sha256(DASHBOARD_PASSWORD.encode()).hexdigest()
        entered = hashlib.sha256(password.encode()).hexdigest()
        if entered == expected:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password")
    return False


# -- Page config -----------------------------------------------------------

st.set_page_config(
    page_title="Crypto Agent Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

if not check_password():
    st.stop()

# -- Sidebar navigation ---------------------------------------------------

page = st.sidebar.radio(
    "Navigation",
    ["Portfolio", "Trades", "Signals", "Agent Log"],
)

st.sidebar.markdown("---")
st.sidebar.caption("Read-only dashboard")

# Auto-refresh toggle
auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
if auto_refresh:
    st.sidebar.caption("Page will refresh automatically")
    import time
    # Streamlit reruns the script; we use st.empty + sleep for auto-refresh
    _placeholder = st.sidebar.empty()


# ==========================================================================
# PAGE 1: Portfolio
# ==========================================================================

def page_portfolio() -> None:
    st.title("Portfolio Overview")

    # -- Portfolio value over time -----------------------------------------
    st.subheader("Portfolio Value Over Time")

    time_range = st.selectbox(
        "Time range",
        ["24h", "7d", "30d", "All"],
        index=1,
        key="pf_range",
    )

    range_filter = {
        "24h": "interval '1 day'",
        "7d": "interval '7 days'",
        "30d": "interval '30 days'",
        "All": "interval '100 years'",
    }[time_range]

    df_pf = run_query(f"""
        SELECT timestamp, total_value_usd, stablecoin_value_usd,
               stablecoin_pct, drawdown_pct
        FROM portfolios
        WHERE timestamp >= now() - {range_filter}
        ORDER BY timestamp
    """)

    if df_pf.empty:
        st.info("No portfolio snapshots yet. The agent records these each cycle.")
    else:
        fig = px.line(
            df_pf,
            x="timestamp",
            y="total_value_usd",
            title="Total Portfolio Value (USD)",
            labels={"total_value_usd": "Value ($)", "timestamp": ""},
        )
        fig.update_layout(hovermode="x unified", height=400)
        st.plotly_chart(fig, use_container_width=True)

        # KPI row
        latest = df_pf.iloc[-1]
        earliest = df_pf.iloc[0]
        pnl = latest["total_value_usd"] - earliest["total_value_usd"]
        pnl_pct = (pnl / earliest["total_value_usd"] * 100) if earliest["total_value_usd"] else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Value", f"${latest['total_value_usd']:,.2f}")
        c2.metric(f"PnL ({time_range})", f"${pnl:,.2f}", f"{pnl_pct:+.2f}%")
        c3.metric("Stablecoin %", f"{latest['stablecoin_pct']:.1f}%")
        c4.metric("Drawdown", f"{latest['drawdown_pct']:.1f}%")

    # -- Current holdings pie chart ----------------------------------------
    st.subheader("Current Holdings")

    df_pos = run_query("""
        SELECT p.symbol, p.amount, p.current_price,
               p.pnl_usd, p.pnl_pct, p.portfolio_pct
        FROM positions p
        INNER JOIN (
            SELECT id FROM portfolios ORDER BY timestamp DESC LIMIT 1
        ) latest ON p.portfolio_id = latest.id
        ORDER BY p.portfolio_pct DESC
    """)

    if df_pos.empty:
        st.info("No position data available.")
    else:
        df_pos["value_usd"] = df_pos["amount"] * df_pos["current_price"]

        col_pie, col_table = st.columns([1, 1])

        with col_pie:
            fig_pie = px.pie(
                df_pos,
                values="value_usd",
                names="symbol",
                title="Holdings Allocation",
                hole=0.4,
            )
            fig_pie.update_traces(textinfo="label+percent")
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_table:
            display_df = df_pos[["symbol", "amount", "current_price", "value_usd", "pnl_usd", "pnl_pct"]].copy()
            display_df.columns = ["Symbol", "Qty", "Price", "Value ($)", "PnL ($)", "PnL (%)"]
            display_df["PnL (%)"] = display_df["PnL (%)"].apply(lambda x: f"{x:+.2f}%")
            display_df["Price"] = display_df["Price"].apply(lambda x: f"${x:,.2f}")
            display_df["Value ($)"] = display_df["Value ($)"].apply(lambda x: f"${x:,.2f}")
            display_df["PnL ($)"] = display_df["PnL ($)"].apply(lambda x: f"${x:+,.2f}")
            st.dataframe(display_df, use_container_width=True, hide_index=True)


# ==========================================================================
# PAGE 2: Trades
# ==========================================================================

def page_trades() -> None:
    st.title("Trade History")

    # Filters
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        pair_filter = st.text_input("Pair filter", placeholder="e.g. BTC/USDT")
    with col2:
        side_filter = st.selectbox("Side", ["All", "buy", "sell"], key="trade_side")
    with col3:
        status_filter = st.selectbox("Status", ["All", "filled", "pending", "rejected"], key="trade_status")
    with col4:
        days_back = st.number_input("Days back", min_value=1, max_value=365, value=30)

    # Build query
    conditions = [f"timestamp >= now() - interval '{days_back} days'"]
    params: dict = {}

    if pair_filter:
        conditions.append("pair ILIKE :pair")
        params["pair"] = f"%{pair_filter}%"
    if side_filter != "All":
        conditions.append("side = :side")
        params["side"] = side_filter
    if status_filter != "All":
        conditions.append("status = :status")
        params["status"] = status_filter

    where = " AND ".join(conditions)

    df_trades = run_query(f"""
        SELECT id, timestamp, pair, side, order_type, amount,
               price, fill_price, fee, status, rationale
        FROM trades
        WHERE {where}
        ORDER BY timestamp DESC
        LIMIT 500
    """, params)

    if df_trades.empty:
        st.info("No trades match your filters.")
        return

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Trades", len(df_trades))

    buys = df_trades[df_trades["side"] == "buy"]
    sells = df_trades[df_trades["side"] == "sell"]
    c2.metric("Buys", len(buys))
    c3.metric("Sells", len(sells))

    total_fees = df_trades["fee"].sum()
    c4.metric("Total Fees", f"${total_fees:,.2f}")

    # Trade table
    st.dataframe(
        df_trades,
        use_container_width=True,
        hide_index=True,
        column_config={
            "timestamp": st.column_config.DatetimeColumn("Time", format="YYYY-MM-DD HH:mm"),
            "pair": "Pair",
            "side": "Side",
            "order_type": "Type",
            "amount": st.column_config.NumberColumn("Amount", format="%.6f"),
            "price": st.column_config.NumberColumn("Price", format="$%.2f"),
            "fill_price": st.column_config.NumberColumn("Fill Price", format="$%.2f"),
            "fee": st.column_config.NumberColumn("Fee", format="$%.4f"),
            "status": "Status",
            "rationale": "Rationale",
        },
    )

    # Volume by pair chart
    st.subheader("Trade Volume by Pair")
    if "fill_price" in df_trades.columns:
        df_vol = df_trades.copy()
        df_vol["trade_value"] = df_vol["amount"] * df_vol["fill_price"].fillna(df_vol["price"].fillna(0))
        vol_by_pair = df_vol.groupby("pair")["trade_value"].sum().reset_index()
        vol_by_pair.columns = ["Pair", "Volume ($)"]
        fig_vol = px.bar(vol_by_pair, x="Pair", y="Volume ($)", title="Trade Volume by Pair")
        st.plotly_chart(fig_vol, use_container_width=True)


# ==========================================================================
# PAGE 3: Signals
# ==========================================================================

def page_signals() -> None:
    st.title("Research Signals")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        source_filter = st.selectbox("Source", ["All", "coindesk", "theblock", "decrypt"], key="sig_source")
    with col2:
        sentiment_filter = st.selectbox("Sentiment", ["All", "bullish", "bearish", "neutral"], key="sig_sent")
    with col3:
        confidence_filter = st.selectbox("Confidence", ["All", "HIGH", "MEDIUM", "LOW"], key="sig_conf")

    conditions = ["1=1"]
    params: dict = {}

    if source_filter != "All":
        conditions.append("source = :source")
        params["source"] = source_filter
    if sentiment_filter != "All":
        conditions.append("signal_type = :signal_type")
        params["signal_type"] = sentiment_filter
    if confidence_filter != "All":
        conditions.append("confidence = :confidence")
        params["confidence"] = confidence_filter

    where = " AND ".join(conditions)

    df_signals = run_query(f"""
        SELECT id, timestamp, source, signal_type, symbol,
               confidence, summary, raw_data
        FROM signals
        WHERE {where}
        ORDER BY timestamp DESC
        LIMIT 200
    """, params)

    if df_signals.empty:
        st.info("No signals recorded yet.")
        return

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Signals", len(df_signals))

    bullish = len(df_signals[df_signals["signal_type"] == "bullish"])
    bearish = len(df_signals[df_signals["signal_type"] == "bearish"])
    neutral = len(df_signals[df_signals["signal_type"] == "neutral"])

    c2.metric("Bullish", bullish)
    c3.metric("Bearish", bearish)
    c4.metric("Neutral", neutral)

    # Sentiment distribution chart
    col_chart, col_timeline = st.columns(2)

    with col_chart:
        sentiment_counts = df_signals["signal_type"].value_counts().reset_index()
        sentiment_counts.columns = ["Sentiment", "Count"]
        color_map = {"bullish": "#2ecc71", "bearish": "#e74c3c", "neutral": "#95a5a6"}
        fig_sent = px.pie(
            sentiment_counts,
            values="Count",
            names="Sentiment",
            title="Sentiment Distribution",
            color="Sentiment",
            color_discrete_map=color_map,
            hole=0.4,
        )
        st.plotly_chart(fig_sent, use_container_width=True)

    with col_timeline:
        # Signals over time
        df_time = df_signals.copy()
        df_time["date"] = pd.to_datetime(df_time["timestamp"]).dt.date
        timeline = df_time.groupby(["date", "signal_type"]).size().reset_index(name="count")
        fig_tl = px.bar(
            timeline,
            x="date",
            y="count",
            color="signal_type",
            title="Signals Over Time",
            color_discrete_map=color_map,
            barmode="stack",
        )
        st.plotly_chart(fig_tl, use_container_width=True)

    # Signal table
    st.subheader("Signal Details")

    display_df = df_signals[["timestamp", "source", "signal_type", "symbol", "confidence", "summary"]].copy()
    display_df.columns = ["Time", "Source", "Sentiment", "Symbol", "Confidence", "Summary"]

    def color_sentiment(val: str) -> str:
        colors = {"bullish": "background-color: #d4edda", "bearish": "background-color: #f8d7da", "neutral": ""}
        return colors.get(val, "")

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Time": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
        },
    )

    # Expandable raw data
    st.subheader("Raw Signal Data")
    for _, row in df_signals.head(10).iterrows():
        raw = row.get("raw_data")
        label = f"[{row['source']}] {row.get('summary', 'N/A')[:80]}"
        with st.expander(label):
            if raw:
                st.json(raw if isinstance(raw, dict) else json.loads(raw))
            else:
                st.write("No raw data available")


# ==========================================================================
# PAGE 4: Agent Log
# ==========================================================================

def page_agent_log() -> None:
    st.title("Agent Decision Log")

    days_back = st.slider("Days back", 1, 30, 7, key="log_days")

    df_decisions = run_query(f"""
        SELECT id, timestamp, model, prompt_tokens, completion_tokens,
               claude_response, actions_taken, error
        FROM agent_decisions
        WHERE timestamp >= now() - interval '{days_back} days'
        ORDER BY timestamp DESC
        LIMIT 100
    """)

    if df_decisions.empty:
        st.info("No agent decisions recorded yet.")
        return

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Decisions", len(df_decisions))
    c2.metric("Total Input Tokens", f"{df_decisions['prompt_tokens'].sum():,}")
    c3.metric("Total Output Tokens", f"{df_decisions['completion_tokens'].sum():,}")

    errors = df_decisions["error"].notna().sum()
    c4.metric("Errors", errors)

    # Token usage over time
    st.subheader("Token Usage Over Time")
    df_tokens = df_decisions[["timestamp", "prompt_tokens", "completion_tokens"]].copy()
    df_tokens["total_tokens"] = df_tokens["prompt_tokens"] + df_tokens["completion_tokens"]
    fig_tokens = px.bar(
        df_tokens,
        x="timestamp",
        y=["prompt_tokens", "completion_tokens"],
        title="Token Usage per Decision",
        labels={"value": "Tokens", "timestamp": ""},
        barmode="stack",
    )
    fig_tokens.update_layout(height=300)
    st.plotly_chart(fig_tokens, use_container_width=True)

    # Decision details
    st.subheader("Decision Details")

    for _, row in df_decisions.iterrows():
        ts = row["timestamp"]
        if isinstance(ts, str):
            ts_str = ts
        else:
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S UTC")

        tokens = f"{row['prompt_tokens']:,} in / {row['completion_tokens']:,} out"
        header = f"#{row['id']} — {ts_str} — {tokens}"

        if row.get("error"):
            header += " [ERROR]"

        with st.expander(header):
            # Claude's response text
            response = row.get("claude_response")
            if response:
                if isinstance(response, str):
                    response = json.loads(response)

                texts = response.get("text", response.get("content", []))
                if isinstance(texts, list):
                    for t in texts:
                        if isinstance(t, str):
                            st.markdown(f"> {t}")
                        elif isinstance(t, dict) and t.get("type") == "text":
                            st.markdown(f"> {t.get('text', '')}")
                elif isinstance(texts, str):
                    st.markdown(f"> {texts}")

            # Actions taken
            actions = row.get("actions_taken")
            if actions:
                if isinstance(actions, str):
                    actions = json.loads(actions)

                st.markdown("**Actions:**")
                if isinstance(actions, list):
                    for action in actions:
                        tool = action.get("tool", "?")
                        inp = action.get("input", {})
                        result = action.get("result_summary", action.get("error", ""))
                        st.code(f"Tool: {tool}\nInput: {json.dumps(inp, indent=2)}\nResult: {result}", language="json")
                else:
                    st.json(actions)

            # Error
            if row.get("error"):
                st.error(f"Error: {row['error']}")

            st.caption(f"Model: {row['model']}")


# ==========================================================================
# Router
# ==========================================================================

if page == "Portfolio":
    page_portfolio()
elif page == "Trades":
    page_trades()
elif page == "Signals":
    page_signals()
elif page == "Agent Log":
    page_agent_log()


# -- Auto-refresh ----------------------------------------------------------

if auto_refresh:
    import time as _time
    _time.sleep(30)
    st.rerun()

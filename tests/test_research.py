"""Tests for src/tools/research.py — RSS parsing, classification, rate limiting."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.research import (
    Article,
    ClassifiedSignal,
    CLASSIFY_PROMPT,
    RATE_LIMIT_SECONDS,
    _last_fetch,
    _mark_fetched,
    _rate_limited,
    classify_article,
    classify_batch,
    fetch_all_feeds,
    fetch_rss,
    gather_signals,
    store_signals,
)


# ── Fixtures ─────────────────────────────────────────────────────────────


SAMPLE_RSS_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
<channel>
  <title>CoinDesk</title>
  <item>
    <title>Bitcoin Surges Past $100K on ETF Inflows</title>
    <link>https://coindesk.com/article1</link>
    <description>Bitcoin hit a new ATH driven by massive ETF inflows.</description>
    <pubDate>Mon, 10 Feb 2026 12:00:00 GMT</pubDate>
  </item>
  <item>
    <title>Ethereum Layer 2 TVL Hits Record</title>
    <link>https://coindesk.com/article2</link>
    <description>&lt;p&gt;Ethereum L2 TVL reached $50B.&lt;/p&gt;</description>
    <pubDate>Mon, 10 Feb 2026 10:00:00 GMT</pubDate>
  </item>
  <item>
    <title>Fed Signals Rate Pause</title>
    <link>https://coindesk.com/article3</link>
    <description>The Federal Reserve paused rate hikes, boosting risk assets.</description>
    <pubDate>Mon, 10 Feb 2026 08:00:00 GMT</pubDate>
  </item>
</channel>
</rss>
"""


@pytest.fixture(autouse=True)
def clear_rate_limit():
    """Reset rate limiter between tests."""
    _last_fetch.clear()
    yield
    _last_fetch.clear()


def make_article(title: str = "Test", source: str = "coindesk") -> Article:
    return Article(
        source=source,
        title=title,
        summary="Some crypto news summary.",
        url="https://example.com/article",
        published="Mon, 10 Feb 2026 12:00:00 GMT",
    )


def make_haiku_response(sentiment: str = "bullish", coins: list[str] | None = None,
                         confidence: str = "HIGH") -> MagicMock:
    """Mock an Anthropic messages.create response."""
    coins = coins or ["BTC"]
    payload = json.dumps({
        "sentiment": sentiment,
        "coins_mentioned": coins,
        "confidence": confidence,
        "one_line_summary": f"Test article classified as {sentiment}",
    })
    block = MagicMock()
    block.text = payload
    block.type = "text"
    resp = MagicMock()
    resp.content = [block]
    return resp


# ── Rate Limiter Tests ───────────────────────────────────────────────────


class TestRateLimiter:
    def test_not_rate_limited_first_time(self):
        assert not _rate_limited("coindesk")

    def test_rate_limited_after_mark(self):
        _mark_fetched("coindesk")
        assert _rate_limited("coindesk")

    def test_not_rate_limited_after_expiry(self):
        _last_fetch["coindesk"] = time.time() - RATE_LIMIT_SECONDS - 1
        assert not _rate_limited("coindesk")

    def test_different_sources_independent(self):
        _mark_fetched("coindesk")
        assert not _rate_limited("theblock")

    def test_mark_updates_timestamp(self):
        _mark_fetched("decrypt")
        first = _last_fetch["decrypt"]
        time.sleep(0.01)
        _mark_fetched("decrypt")
        assert _last_fetch["decrypt"] > first


# ── RSS Fetching Tests ───────────────────────────────────────────────────


class TestFetchRss:
    @pytest.mark.asyncio
    async def test_fetches_and_parses_articles(self):
        mock_resp = MagicMock()
        mock_resp.text = SAMPLE_RSS_XML
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.tools.research.httpx.AsyncClient", return_value=mock_client):
            articles = await fetch_rss("coindesk", "https://example.com/rss")

        assert len(articles) == 3
        assert articles[0].title == "Bitcoin Surges Past $100K on ETF Inflows"
        assert articles[0].source == "coindesk"
        assert articles[0].url == "https://coindesk.com/article1"

    @pytest.mark.asyncio
    async def test_strips_html_from_summary(self):
        mock_resp = MagicMock()
        mock_resp.text = SAMPLE_RSS_XML
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.tools.research.httpx.AsyncClient", return_value=mock_client):
            articles = await fetch_rss("coindesk", "https://example.com/rss")

        # Second article has <p> tags that should be stripped
        assert "<" not in articles[1].summary

    @pytest.mark.asyncio
    async def test_respects_max_articles(self):
        mock_resp = MagicMock()
        mock_resp.text = SAMPLE_RSS_XML
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.tools.research.httpx.AsyncClient", return_value=mock_client):
            articles = await fetch_rss("coindesk", "https://example.com/rss", max_articles=1)

        assert len(articles) == 1

    @pytest.mark.asyncio
    async def test_returns_empty_on_http_error(self):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("Connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.tools.research.httpx.AsyncClient", return_value=mock_client):
            articles = await fetch_rss("coindesk", "https://example.com/rss")

        assert articles == []

    @pytest.mark.asyncio
    async def test_skips_when_rate_limited(self):
        _mark_fetched("coindesk")

        articles = await fetch_rss("coindesk", "https://example.com/rss")
        assert articles == []

    @pytest.mark.asyncio
    async def test_truncates_long_summaries(self):
        long_summary = "A" * 600
        rss = f"""\
<?xml version="1.0"?>
<rss version="2.0"><channel><item>
  <title>Test</title>
  <description>{long_summary}</description>
</item></channel></rss>"""

        mock_resp = MagicMock()
        mock_resp.text = rss
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.tools.research.httpx.AsyncClient", return_value=mock_client):
            articles = await fetch_rss("test", "https://example.com/rss")

        assert len(articles[0].summary) == 503  # 500 + "..."


class TestFetchAllFeeds:
    @pytest.mark.asyncio
    async def test_aggregates_from_all_sources(self):
        async def mock_fetch(source, url, max_articles=5):
            return [make_article(f"Article from {source}", source)]

        with patch("src.tools.research.fetch_rss", side_effect=mock_fetch):
            articles = await fetch_all_feeds()

        assert len(articles) == 3  # one from each of 3 sources
        sources = {a.source for a in articles}
        assert sources == {"coindesk", "theblock", "decrypt"}

    @pytest.mark.asyncio
    async def test_handles_partial_failures(self):
        call_count = 0

        async def mock_fetch(source, url, max_articles=5):
            nonlocal call_count
            call_count += 1
            if source == "theblock":
                raise Exception("Feed down")
            return [make_article(source=source)]

        with patch("src.tools.research.fetch_rss", side_effect=mock_fetch):
            articles = await fetch_all_feeds()

        assert len(articles) == 2  # theblock failed, other 2 succeeded


# ── Classification Tests ─────────────────────────────────────────────────


class TestClassifyArticle:
    @pytest.mark.asyncio
    async def test_classifies_bullish(self):
        client = AsyncMock()
        client.messages.create = AsyncMock(return_value=make_haiku_response("bullish", ["BTC"]))

        article = make_article("Bitcoin Surges")
        result = await classify_article(client, article)

        assert result is not None
        assert result.signal_type == "bullish"
        assert result.symbols == ["BTC"]
        assert result.confidence == "HIGH"
        assert result.source == "coindesk"

    @pytest.mark.asyncio
    async def test_classifies_bearish(self):
        client = AsyncMock()
        client.messages.create = AsyncMock(
            return_value=make_haiku_response("bearish", ["ETH"], "MEDIUM")
        )

        result = await classify_article(client, make_article("ETH Drops"))
        assert result is not None
        assert result.signal_type == "bearish"
        assert result.confidence == "MEDIUM"

    @pytest.mark.asyncio
    async def test_handles_markdown_wrapped_json(self):
        block = MagicMock()
        block.text = '```json\n{"sentiment":"neutral","coins_mentioned":["MARKET"],"confidence":"LOW","one_line_summary":"test"}\n```'
        resp = MagicMock()
        resp.content = [block]

        client = AsyncMock()
        client.messages.create = AsyncMock(return_value=resp)

        result = await classify_article(client, make_article())
        assert result is not None
        assert result.signal_type == "neutral"

    @pytest.mark.asyncio
    async def test_returns_none_on_invalid_json(self):
        block = MagicMock()
        block.text = "This is not JSON at all"
        resp = MagicMock()
        resp.content = [block]

        client = AsyncMock()
        client.messages.create = AsyncMock(return_value=resp)

        result = await classify_article(client, make_article())
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_api_error(self):
        client = AsyncMock()
        client.messages.create = AsyncMock(side_effect=Exception("API overloaded"))

        result = await classify_article(client, make_article())
        assert result is None

    @pytest.mark.asyncio
    async def test_preserves_article_metadata_in_raw_data(self):
        client = AsyncMock()
        client.messages.create = AsyncMock(return_value=make_haiku_response())

        article = make_article("Important News")
        result = await classify_article(client, article)

        assert result is not None
        assert result.raw_data["title"] == "Important News"
        assert "classification" in result.raw_data
        assert result.article_url == article.url


class TestClassifyBatch:
    @pytest.mark.asyncio
    async def test_classifies_multiple_articles(self):
        articles = [make_article(f"Article {i}") for i in range(5)]

        with patch("src.tools.research.anthropic.AsyncAnthropic") as mock_cls:
            client = AsyncMock()
            client.messages.create = AsyncMock(return_value=make_haiku_response())
            mock_cls.return_value = client

            signals = await classify_batch(articles)

        assert len(signals) == 5
        assert client.messages.create.call_count == 5

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(self):
        signals = await classify_batch([])
        assert signals == []

    @pytest.mark.asyncio
    async def test_filters_out_failed_classifications(self):
        call_count = 0

        async def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("API error")
            return make_haiku_response()

        with patch("src.tools.research.anthropic.AsyncAnthropic") as mock_cls:
            client = AsyncMock()
            client.messages.create = AsyncMock(side_effect=mock_create)
            mock_cls.return_value = client

            articles = [make_article(f"A{i}") for i in range(3)]
            signals = await classify_batch(articles)

        assert len(signals) == 2  # one failed


# ── Storage Tests ────────────────────────────────────────────────────────


class TestStoreSignals:
    @pytest.mark.asyncio
    async def test_returns_zero_for_empty_list(self):
        count = await store_signals([])
        assert count == 0

    @pytest.mark.asyncio
    async def test_stores_signals_creates_rows_per_symbol(self):
        signal = ClassifiedSignal(
            source="coindesk",
            signal_type="bullish",
            symbols=["BTC", "ETH"],
            confidence="HIGH",
            summary="Test",
            article_url="https://example.com",
        )

        mock_session = AsyncMock()
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_signal_cls = MagicMock()
        mock_models = MagicMock()
        mock_models.Signal = mock_signal_cls
        mock_models.async_session = MagicMock(return_value=mock_session)

        with patch.dict("sys.modules", {"src.db.models": mock_models}):
            count = await store_signals([signal])

        # 2 rows: one per symbol (BTC, ETH)
        assert mock_session.add.call_count == 2
        assert count == 2

    @pytest.mark.asyncio
    async def test_graceful_on_db_unavailable(self):
        signal = ClassifiedSignal(
            source="coindesk",
            signal_type="bearish",
            symbols=["SOL"],
            confidence="MEDIUM",
            summary="SOL drops",
            article_url="https://example.com",
        )
        # Should not raise, just return 0
        count = await store_signals([signal])
        assert count == 0


# ── Integration: gather_signals ──────────────────────────────────────────


class TestGatherSignals:
    @pytest.mark.asyncio
    async def test_returns_empty_when_no_articles(self):
        with patch("src.tools.research.fetch_all_feeds", return_value=[]):
            result = await gather_signals()
        assert result == []

    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        articles = [make_article("BTC up"), make_article("ETH down")]

        mock_signals = [
            ClassifiedSignal(
                source="coindesk",
                signal_type="bullish",
                symbols=["BTC"],
                confidence="HIGH",
                summary="BTC surging",
                article_url="https://example.com/1",
            ),
            ClassifiedSignal(
                source="coindesk",
                signal_type="bearish",
                symbols=["ETH"],
                confidence="MEDIUM",
                summary="ETH declining",
                article_url="https://example.com/2",
            ),
        ]

        with patch("src.tools.research.fetch_all_feeds", return_value=articles):
            with patch("src.tools.research.classify_batch", return_value=mock_signals):
                with patch("src.tools.research.store_signals", return_value=2):
                    result = await gather_signals()

        assert len(result) == 2
        assert result[0]["signal_type"] == "bullish"
        assert result[0]["symbols"] == ["BTC"]
        assert result[0]["confidence"] == "HIGH"
        assert result[1]["signal_type"] == "bearish"
        assert result[1]["symbols"] == ["ETH"]

    @pytest.mark.asyncio
    async def test_output_format(self):
        articles = [make_article()]
        mock_signal = ClassifiedSignal(
            source="decrypt",
            signal_type="neutral",
            symbols=["MARKET"],
            confidence="LOW",
            summary="General market update",
            article_url="https://decrypt.co/test",
        )

        with patch("src.tools.research.fetch_all_feeds", return_value=articles):
            with patch("src.tools.research.classify_batch", return_value=[mock_signal]):
                with patch("src.tools.research.store_signals", return_value=1):
                    result = await gather_signals()

        assert len(result) == 1
        entry = result[0]
        assert set(entry.keys()) == {"source", "signal_type", "symbols", "confidence", "summary", "url"}
        assert entry["url"] == "https://decrypt.co/test"

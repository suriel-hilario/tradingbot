"""RSS feed parser + Claude Haiku classifier for crypto market signals.

Fetches articles from CoinDesk, The Block, and Decrypt, classifies each
as bullish/bearish/neutral using Claude Haiku, and stores structured
signals in the Signal table.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import anthropic
import feedparser
import httpx
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("research")

HAIKU_MODEL = os.getenv("CLASSIFY_MODEL", "claude-haiku-4-5-20251001")
MAX_ARTICLES_PER_SOURCE = int(os.getenv("MAX_ARTICLES_PER_SOURCE", "5"))
RATE_LIMIT_SECONDS = int(os.getenv("RESEARCH_RATE_LIMIT", "300"))  # 5 min default


# -- RSS feed sources ------------------------------------------------------

RSS_FEEDS: dict[str, str] = {
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "theblock": "https://www.theblock.co/rss.xml",
    "decrypt": "https://decrypt.co/feed",
}


# -- Rate limiter ----------------------------------------------------------

_last_fetch: dict[str, float] = {}


def _rate_limited(source: str) -> bool:
    """Return True if source was fetched too recently."""
    last = _last_fetch.get(source, 0)
    return (time.time() - last) < RATE_LIMIT_SECONDS


def _mark_fetched(source: str) -> None:
    _last_fetch[source] = time.time()


# -- Data classes ----------------------------------------------------------


@dataclass
class Article:
    source: str
    title: str
    summary: str
    url: str
    published: str


@dataclass
class ClassifiedSignal:
    source: str
    signal_type: str       # bullish / bearish / neutral
    symbols: list[str]
    confidence: str        # HIGH / MEDIUM / LOW
    summary: str
    article_url: str
    raw_data: dict[str, Any] = field(default_factory=dict)


# -- RSS fetching ----------------------------------------------------------


async def fetch_rss(
    source: str,
    url: str,
    max_articles: int = MAX_ARTICLES_PER_SOURCE,
) -> list[Article]:
    """Fetch and parse an RSS feed, return latest N articles."""
    if _rate_limited(source):
        log.debug("Rate-limited: %s (skipping)", source)
        return []

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, follow_redirects=True)
            resp.raise_for_status()
    except Exception as exc:
        log.warning("RSS fetch failed for %s: %s", source, exc)
        return []

    _mark_fetched(source)
    feed = feedparser.parse(resp.text)
    articles: list[Article] = []

    for entry in feed.entries[:max_articles]:
        title = entry.get("title", "")
        summary = entry.get("summary", entry.get("description", ""))
        # Strip HTML tags from summary
        if "<" in summary:
            summary = re.sub(r"<[^>]+>", "", summary).strip()
        # Truncate long summaries
        if len(summary) > 500:
            summary = summary[:500] + "..."

        articles.append(Article(
            source=source,
            title=title,
            summary=summary,
            url=entry.get("link", ""),
            published=entry.get("published", ""),
        ))

    log.info("Fetched %d articles from %s", len(articles), source)
    return articles


async def fetch_all_feeds() -> list[Article]:
    """Fetch all configured RSS feeds concurrently."""
    tasks = [
        fetch_rss(source, url)
        for source, url in RSS_FEEDS.items()
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    articles: list[Article] = []
    for result in results:
        if isinstance(result, list):
            articles.extend(result)
        elif isinstance(result, Exception):
            log.warning("Feed fetch error: %s", result)
    return articles


# -- Claude Haiku classification -------------------------------------------

CLASSIFY_PROMPT = """\
You are a crypto market analyst. Classify this article.

Title: {title}
Summary: {summary}
Source: {source}

Respond with ONLY a JSON object (no markdown, no explanation):
{{
  "sentiment": "bullish" | "bearish" | "neutral",
  "coins_mentioned": ["BTC", "ETH", ...],
  "confidence": "HIGH" | "MEDIUM" | "LOW",
  "one_line_summary": "brief summary of the signal"
}}

Rules:
- confidence HIGH = clear, unambiguous signal with data support
- confidence MEDIUM = some signal but mixed or uncertain
- confidence LOW = vague or speculative
- coins_mentioned should use ticker symbols (BTC, ETH, SOL, etc.)
- If no specific coin is mentioned, use ["MARKET"] for general market signals
"""


async def classify_article(
    client: anthropic.AsyncAnthropic,
    article: Article,
) -> ClassifiedSignal | None:
    """Use Claude Haiku to classify a single article."""
    try:
        response = await client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=256,
            messages=[{
                "role": "user",
                "content": CLASSIFY_PROMPT.format(
                    title=article.title,
                    summary=article.summary,
                    source=article.source,
                ),
            }],
        )

        text = response.content[0].text.strip()
        # Handle potential markdown wrapping
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        data = json.loads(text)

        return ClassifiedSignal(
            source=article.source,
            signal_type=data.get("sentiment", "neutral"),
            symbols=data.get("coins_mentioned", ["MARKET"]),
            confidence=data.get("confidence", "LOW"),
            summary=data.get("one_line_summary", article.title),
            article_url=article.url,
            raw_data={
                "title": article.title,
                "published": article.published,
                "classification": data,
            },
        )
    except json.JSONDecodeError as exc:
        log.warning("Haiku returned invalid JSON for '%s': %s", article.title, exc)
        return None
    except Exception as exc:
        log.warning("Classification failed for '%s': %s", article.title, exc)
        return None


async def classify_batch(articles: list[Article]) -> list[ClassifiedSignal]:
    """Classify all articles with concurrency limit to avoid API rate limits."""
    if not articles:
        return []

    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(3)  # max 3 concurrent Haiku calls

    async def _classify(article: Article) -> ClassifiedSignal | None:
        async with semaphore:
            return await classify_article(client, article)

    results = await asyncio.gather(
        *[_classify(a) for a in articles],
        return_exceptions=True,
    )

    signals: list[ClassifiedSignal] = []
    for r in results:
        if isinstance(r, ClassifiedSignal):
            signals.append(r)
        elif isinstance(r, Exception):
            log.warning("Classification error: %s", r)
    return signals


# -- DB persistence --------------------------------------------------------


async def store_signals(signals: list[ClassifiedSignal]) -> int:
    """Persist signals to the Signal table. Returns count stored."""
    if not signals:
        return 0

    try:
        from src.db.models import Signal as SignalModel, async_session

        stored = 0
        async with async_session() as session:
            for sig in signals:
                for symbol in sig.symbols:
                    row = SignalModel(
                        source=sig.source,
                        signal_type=sig.signal_type,
                        symbol=symbol,
                        confidence=sig.confidence,
                        summary=sig.summary,
                        raw_data=sig.raw_data,
                    )
                    session.add(row)
                    stored += 1
            await session.commit()
        log.info("Stored %d signal rows in DB", stored)
        return stored
    except Exception as exc:
        log.warning("Signal DB storage skipped: %s", exc)
        return 0


# -- Public API ------------------------------------------------------------


async def gather_signals() -> list[dict[str, Any]]:
    """Full pipeline: fetch RSS -> classify with Haiku -> store -> return."""
    articles = await fetch_all_feeds()
    if not articles:
        log.info("No new articles to classify")
        return []

    signals = await classify_batch(articles)
    await store_signals(signals)

    # Convert to dicts for the orchestrator
    return [
        {
            "source": s.source,
            "signal_type": s.signal_type,
            "symbols": s.symbols,
            "confidence": s.confidence,
            "summary": s.summary,
            "url": s.article_url,
        }
        for s in signals
    ]


# -- Backward-compatible wrappers ------------------------------------------


async def fetch_analyst_reports() -> list[dict[str, Any]]:
    """Fetch and classify RSS articles (backward-compatible wrapper)."""
    return await gather_signals()


async def check_onchain_metrics() -> dict[str, Any]:
    """On-chain metrics placeholder (requires paid API keys)."""
    # TODO: integrate Glassnode / Dune Analytics when API keys available
    return {}


async def run_research_pipeline() -> list[dict[str, Any]]:
    """Run all research sources and aggregate signals."""
    signals = await gather_signals()
    onchain = await check_onchain_metrics()
    return [
        {"source": "rss_signals", "data": signals},
        {"source": "onchain_metrics", "data": onchain},
    ]

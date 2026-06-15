"""
SwingLab News Service — Alpaca News API + LLM Sentiment
Fetches news for stocks and analyzes sentiment via LLM.
"""

import httpx
from datetime import datetime, timedelta
from app.config import settings
from app.db.mongodb import get_db
from app.services.llm_service import llm_ask, llm_available

ALPACA_NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
HEADERS = {
    "APCA-API-KEY-ID": settings.ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": settings.ALPACA_SECRET_KEY,
}


async def fetch_news(symbol, limit=5):
    """Fetch latest news for a symbol from Alpaca."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                ALPACA_NEWS_URL,
                headers=HEADERS,
                params={"symbols": symbol, "limit": limit, "sort": "desc"},
            )
            if r.status_code != 200:
                return []
            data = r.json()
            news = []
            for article in data.get("news", []):
                news.append({
                    "headline": article.get("headline", ""),
                    "summary": article.get("summary", "")[:200],
                    "source": article.get("source", ""),
                    "created_at": article.get("created_at", ""),
                    "url": article.get("url", ""),
                })
            return news
    except Exception as e:
        print(f"  News fetch error {symbol}: {e}")
        return []


async def analyze_news_sentiment(symbol, news_items):
    """Use LLM to analyze sentiment from news headlines."""
    if not news_items or not llm_available():
        return None

    headlines = "\n".join([f"- {n['headline']} ({n['source']})" for n in news_items[:5]])

    try:
        result = llm_ask(
            system_prompt=(
                "Sei un analista finanziario. Analizza queste news per il ticker indicato. "
                "Rispondi in italiano con ESATTAMENTE questo formato:\n"
                "SENTIMENT: POSITIVO/NEGATIVO/NEUTRO\n"
                "SINTESI: [una frase che riassume il sentiment delle news]\n"
                "Sii diretto, no disclaimers."
            ),
            user_prompt=f"Ticker: {symbol}\nNews recenti:\n{headlines}",
            max_tokens=100,
            temperature=0.2,
        )
        return result
    except Exception as e:
        print(f"  News sentiment error {symbol}: {e}")
        return None


async def get_stock_news_with_sentiment(symbol):
    """Complete pipeline: fetch news + analyze sentiment."""
    news = await fetch_news(symbol, limit=5)
    if not news:
        return {"news": [], "sentiment": None, "ticker": symbol}

    sentiment = await analyze_news_sentiment(symbol, news)

    return {
        "ticker": symbol,
        "news": news,
        "sentiment": sentiment,
        "news_count": len(news),
        "last_news_at": news[0]["created_at"] if news else None,
    }


async def get_batch_news_sentiment(symbols, limit_per_stock=3):
    """Fetch news and sentiment for multiple stocks."""
    results = {}
    for symbol in symbols[:10]:  # Max 10 per risparmiare quota
        data = await get_stock_news_with_sentiment(symbol)
        if data["news"]:
            results[symbol] = data
    return results

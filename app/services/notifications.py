import httpx
from datetime import datetime
from app.db.mongodb import get_db
from app.config import settings


async def send_telegram(message):
    """Send a message via Telegram Bot"""
    if not settings.TELEGRAM_BOT_TOKEN or not settings.TELEGRAM_CHAT_ID:
        print("Telegram not configured")
        return False
    url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": settings.TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(url, json=payload)
            if r.status_code == 200:
                print(f"Telegram sent OK")
                return True
            else:
                print(f"Telegram error: {r.status_code} {r.text}")
                return False
    except Exception as e:
        print(f"Telegram error: {e}")
        return False


async def check_and_notify():
    """Check for alerts and send Telegram notifications"""
    db = get_db()

    # Get all assets
    assets = await db.assets.find().to_list(300)
    sectors = await db.sectors.find().to_list(20)
    regime = await db.market_regime.find_one({"symbol": "SPY"})

    if not assets:
        return {"sent": 0, "reason": "no data"}

    # Sort sectors by composite_score
    sectors.sort(key=lambda x: x.get("composite_score", 0), reverse=True)
    sector_ranks = {s["code"]: i + 1 for i, s in enumerate(sectors)}

    # Calculate confluence for each asset
    alerts = []
    bottoms = []

    for a in assets:
        confluence = 0

        # POC proximity
        if a.get("poc_price") and a.get("price"):
            poc_dist = abs((a["price"] - a["poc_price"]) / a["price"] * 100)
            if poc_dist <= 2:
                confluence += 2

        # Bullish patterns
        bull_patterns = [p for p in (a.get("candlestick_patterns") or []) if p.get("type") == "bullish"]
        if bull_patterns:
            confluence += 1.5

        # RSI sweet spot
        rsi = a.get("rsi", 50)
        if 40 <= rsi <= 60:
            confluence += 1

        # MACD bullish
        macd = a.get("macd", {})
        if macd.get("histogram", 0) > 0:
            confluence += 1

        # EMA uptrend
        price = a.get("price", 0)
        ema10 = a.get("ema10", 0)
        ema20 = a.get("ema20", 0)
        ema50 = a.get("ema50", 0)
        if price > ema10 > ema20 > ema50:
            confluence += 1.5
        elif price > ema20 > ema50:
            confluence += 0.75

        # Volume
        if a.get("relative_volume", 0) >= 1.5:
            confluence += 1

        # Sector rank
        rank = sector_ranks.get(a.get("sector_code", ""), 11)
        if rank <= 5:
            confluence += 1

        # Near 52W high
        if a.get("pct_from_high") and a["pct_from_high"] >= -10:
            confluence += 0.5

        # Momentum
        if 0 < a.get("change_pct", 0) <= 5:
            confluence += 0.5

        # Bearish override
        bear_patterns = [p for p in (a.get("candlestick_patterns") or []) if p.get("type") == "bearish" and p.get("strength") == "strong"]
        if bear_patterns:
            confluence = max(0, confluence - 2)
        if rsi > 75:
            confluence = max(0, confluence - 1.5)

        confluence = round(confluence, 1)

        if confluence >= 6:
            alerts.append({
                "ticker": a["ticker"],
                "price": a.get("price", 0),
                "confluence": confluence,
                "setup": a.get("setup_type", "neutral"),
                "rsi": rsi,
                "change": a.get("change_pct", 0),
                "poc": a.get("poc_price"),
                "sector": a.get("sector_code", ""),
                "patterns": [p["name"] for p in bull_patterns],
            })

        # Bottom detection
        bottom_score = 0
        if rsi <= 30:
            bottom_score += 3
        elif rsi <= 40:
            bottom_score += 2
        if a.get("value_area_low") and a.get("price"):
            va_dist = ((a["price"] - a["value_area_low"]) / a["price"] * 100)
            if -5 <= va_dist <= 2:
                bottom_score += 2
        if a.get("low_52w") and a.get("price"):
            low_dist = ((a["price"] - a["low_52w"]) / a["low_52w"] * 100)
            if low_dist <= 15:
                bottom_score += 1.5

        if bottom_score >= 5:
            bottoms.append({
                "ticker": a["ticker"],
                "price": a.get("price", 0),
                "score": round(bottom_score, 1),
                "rsi": rsi,
            })

    alerts.sort(key=lambda x: x["confluence"], reverse=True)
    bottoms.sort(key=lambda x: x["score"], reverse=True)

    # Build message
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"🔬 <b>SwingLab Alert</b> - {now}\n"]

    # Market regime
    if regime:
        spy_price = regime.get("price", 0)
        spy_rsi = regime.get("rsi", 50)
        spy_ema50 = regime.get("ema50", 0)
        if spy_price > spy_ema50 and spy_rsi > 50:
            lines.append(f"🟢 Market: BULL | SPY ${spy_price} | RSI {spy_rsi}")
        elif spy_price > spy_ema50:
            lines.append(f"🟡 Market: NEUTRAL | SPY ${spy_price} | RSI {spy_rsi}")
        elif spy_rsi > 35:
            lines.append(f"🔴 Market: BEAR | SPY ${spy_price} | RSI {spy_rsi}")
        else:
            lines.append(f"⚫ Market: CRASH | SPY ${spy_price} | RSI {spy_rsi}")
        lines.append("")

    # Elite/Strong alerts
    elite = [a for a in alerts if a["confluence"] >= 8]
    strong = [a for a in alerts if 6 <= a["confluence"] < 8]

    if elite:
        lines.append("🔥🔥🔥 <b>ELITE SETUPS</b>")
        for a in elite[:5]:
            pat = f" | {', '.join(a['patterns'])}" if a['patterns'] else ""
            lines.append(f"  <b>{a['ticker']}</b> ${a['price']:.2f} | {a['confluence']}/10 | {a['setup']} | RSI {a['rsi']:.0f}{pat}")
        lines.append("")

    if strong:
        lines.append("🔥🔥 <b>STRONG BUY</b>")
        for a in strong[:5]:
            lines.append(f"  <b>{a['ticker']}</b> ${a['price']:.2f} | {a['confluence']}/10 | {a['setup']} | RSI {a['rsi']:.0f}")
        lines.append("")

    # Bottoms
    if bottoms:
        lines.append("🔴 <b>BOTTOM SIGNALS</b>")
        for b in bottoms[:5]:
            lines.append(f"  <b>{b['ticker']}</b> ${b['price']:.2f} | Bottom {b['score']}/10 | RSI {b['rsi']:.0f}")
        lines.append("")

    # Sector summary
    if sectors:
        top3 = sectors[:3]
        bot3 = sectors[-3:]
        lines.append("📊 <b>SECTORS</b>")
        lines.append(f"  Top: {', '.join([f\"{s['code']} ({s.get('composite_score',0):.0f})\" for s in top3])}")
        lines.append(f"  Bottom: {', '.join([f\"{s['code']} ({s.get('composite_score',0):.0f})\" for s in bot3])}")
        lines.append("")

    # Summary
    lines.append(f"📈 {len(assets)} stocks | {len(alerts)} alerts | {len(bottoms)} bottoms")
    lines.append(f"🌐 https://swinglab-frontend-git-main-giotosi94s-projects.vercel.app")

    # Only send if there are alerts or bottoms
    if alerts or bottoms:
        message = "\n".join(lines)
        sent = await send_telegram(message)
        return {"sent": 1 if sent else 0, "alerts": len(alerts), "bottoms": len(bottoms)}
    else:
        # Send minimal update
        message = "\n".join(lines)
        sent = await send_telegram(message)
        return {"sent": 1 if sent else 0, "alerts": 0, "bottoms": 0}

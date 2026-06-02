import httpx
from app.config import settings
from app.db.mongodb import get_db


async def send_telegram(message):
    if not settings.TELEGRAM_BOT_TOKEN or not settings.TELEGRAM_CHAT_ID:
        print("Telegram not configured")
        return False
    url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": settings.TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
    }
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(url, json=payload)
            if r.status_code == 200:
                print("Telegram message sent")
                return True
            else:
                print(f"Telegram error: {r.status_code} - {r.text}")
                return False
    except Exception as e:
        print(f"Telegram error: {e}")
        return False


async def send_daily_briefing():
    db = get_db()
    sectors = await db.sectors.find().sort("composite_score", -1).to_list(20)
    assets = await db.assets.find().to_list(200)

    if not assets:
        await send_telegram("SwingLab: No data available.")
        return None

    top_sector = sectors[0] if sectors else None

    strong_buys = []
    buys = []
    bottoms = []
    overbought = []

    for a in assets:
        score = a.get("setup_score", 0)
        stype = a.get("setup_type", "")
        rsi = a.get("rsi", 50)

        if score >= 65 and stype in ("breakout", "pullback_to_poc"):
            strong_buys.append(a)
        elif score >= 55 and stype in ("pullback_to_poc", "ema_bounce", "breakout"):
            buys.append(a)

        if rsi <= 35:
            bottoms.append(a)
        if rsi >= 70:
            overbought.append(a)

    msg = "<b>SwingLab Daily Briefing</b>\n\n"

    if top_sector:
        code = top_sector.get("code", "")
        name = top_sector.get("name", "")
        comp = top_sector.get("composite_score", 0)
        msg += "<b>Top Sector:</b> {} ({}) Score: {:.1f}\n\n".format(code, name, comp)

    msg += "<b>Signals:</b>\n"
    msg += "  Strong Buy: {}\n".format(len(strong_buys))
    msg += "  Buy: {}\n".format(len(buys))
    msg += "  Bottoming: {}\n".format(len(bottoms))
    msg += "  Overbought: {}\n\n".format(len(overbought))

    if strong_buys:
        msg += "<b>STRONG BUY:</b>\n"
        sorted_sb = sorted(strong_buys, key=lambda x: x.get("setup_score", 0), reverse=True)
        for a in sorted_sb[:5]:
            ticker = a.get("ticker", "")
            price = a.get("price", 0)
            score = a.get("setup_score", 0)
            stype = a.get("setup_type", "")
            msg += "  {} ${:.2f} Score:{} [{}]\n".format(ticker, price, score, stype)
        msg += "\n"

    if buys:
        msg += "<b>BUY:</b>\n"
        sorted_b = sorted(buys, key=lambda x: x.get("setup_score", 0), reverse=True)
        for a in sorted_b[:5]:
            ticker = a.get("ticker", "")
            price = a.get("price", 0)
            score = a.get("setup_score", 0)
            msg += "  {} ${:.2f} Score:{}\n".format(ticker, price, score)
        msg += "\n"

    if bottoms:
        msg += "<b>BOTTOMING:</b>\n"
        sorted_bot = sorted(bottoms, key=lambda x: x.get("rsi", 50))
        for a in sorted_bot[:5]:
            ticker = a.get("ticker", "")
            rsi = a.get("rsi", 0)
            msg += "  {} RSI:{:.1f}\n".format(ticker, rsi)
        msg += "\n"

    if overbought:
        msg += "<b>OVERBOUGHT:</b>\n"
        sorted_ob = sorted(overbought, key=lambda x: x.get("rsi", 50), reverse=True)
        for a in sorted_ob[:5]:
            ticker = a.get("ticker", "")
            rsi = a.get("rsi", 0)
            msg += "  {} RSI:{:.1f}\n".format(ticker, rsi)
        msg += "\n"

    msg += "Stocks: {} | Sectors: {}".format(len(assets), len(sectors))

    await send_telegram(msg)
    return msg

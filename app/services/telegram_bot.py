import httpx
from datetime import datetime
from app.config import settings
from app.db.mongodb import get_db
from app.services.stock_names import get_stock_name


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
                return True
            else:
                print(f"Telegram error: {r.status_code} - {r.text[:100]}")
                return False
    except Exception as e:
        print(f"Telegram error: {e}")
        return False


async def send_daily_briefing():
    """Morning briefing con stato mercato, top picks, e portfolio."""
    db = get_db()
    sectors = await db.sectors.find().sort("composite_score", -1).to_list(20)
    assets = await db.assets.find(
        {}, {"ticker":1,"setup_score":1,"setup_type":1,"rsi":1,"price":1,"sector_code":1}
    ).to_list(300)
    market_ctx = await db.market_context.find_one({"_id": "latest"})
    alpaca_state = await db.auto_trader.find_one({"_id": "alpaca_state"})

    if not assets:
        await send_telegram("SwingLab: No data available.")
        return

    regime = market_ctx.get("market_regime", "UNKNOWN") if market_ctx else "UNKNOWN"
    confidence = market_ctx.get("regime_confidence", 0) if market_ctx else 0

    sorted_assets = sorted(assets, key=lambda x: x.get("setup_score", 0), reverse=True)
    strong_buys = [a for a in sorted_assets if a.get("setup_score", 0) >= 65]
    buys = [a for a in sorted_assets if 50 <= a.get("setup_score", 0) < 65]
    oversold = [a for a in assets if a.get("rsi", 50) <= 30]
    overbought = [a for a in assets if a.get("rsi", 50) >= 70]

    equity = alpaca_state.get("equity", 0) if alpaca_state else 0
    positions = alpaca_state.get("positions", 0) if alpaca_state else 0

    regime_emoji = {"BULL":"\U0001F7E2","NEUTRAL":"\U0001F7E1","BEAR":"\U0001F7E0","CRASH":"\U0001F534"}.get(regime, "\u26AA")

    msg = "<b>\U0001F305 SwingLab Morning Briefing</b>\n\n"
    msg += f"<b>Market:</b> {regime_emoji} {regime} ({confidence:.0f}%)\n"
    if sectors:
        top = sectors[0]
        msg += f"<b>Top Sector:</b> {top.get('code','')} ({top.get('name','')}) Score: {top.get('composite_score',0):.1f}\n"
    msg += f"\n<b>Portfolio:</b> ${equity:,.0f} | {positions} positions\n"
    msg += f"\n<b>Signals:</b>\n"
    msg += f"  \U0001F525 Strong Buy: {len(strong_buys)}\n"
    msg += f"  \U0001F4C8 Buy: {len(buys)}\n"
    msg += f"  \U0001F4C9 Oversold: {len(oversold)}\n"
    msg += f"  \u26A0 Overbought: {len(overbought)}\n"

    if strong_buys:
        msg += f"\n<b>\U0001F525 TOP PICKS:</b>\n"
        for a in strong_buys[:5]:
            t = a.get("ticker", "")
            msg += f"  <b>{t}</b> ({get_stock_name(t)}) ${a.get('price',0):.2f} Score:{a.get('setup_score',0)} [{a.get('setup_type','')}]\n"

    await send_telegram(msg)
    print("\U0001F4F1 Daily briefing sent")


async def send_evening_report():
    """Evening report con P&L del giorno."""
    db = get_db()
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    trades = await db.trade_history.find({"date": {"$gte": today}}).to_list(50)
    buys_today = [t for t in trades if t.get("side") == "buy"]
    sells_today = [t for t in trades if t.get("side") == "sell"]

    alpaca_state = await db.auto_trader.find_one({"_id": "alpaca_state"})
    equity = alpaca_state.get("equity", 0) if alpaca_state else 0

    msg = "<b>\U0001F319 SwingLab Evening Report</b>\n\n"
    msg += f"<b>Equity:</b> ${equity:,.2f}\n"
    msg += f"<b>Today:</b> {len(buys_today)} buys, {len(sells_today)} sells\n"

    if sells_today:
        total_pnl = sum(t.get("pnl_pct", 0) for t in sells_today)
        wins = sum(1 for t in sells_today if t.get("pnl_pct", 0) > 0)
        msg += f"\n<b>Closed:</b>\n"
        for t in sells_today:
            emoji = "\U0001F7E2" if t.get("pnl_pct", 0) > 0 else "\U0001F534"
            ticker = t.get("ticker", "")
            msg += f"  {emoji} {ticker} ({get_stock_name(ticker)}) {t.get('pnl_pct',0):+.1f}% [{t.get('reason','')}]\n"
        msg += f"\n<b>Day P&L:</b> {total_pnl:+.1f}% | {wins}/{len(sells_today)} wins\n"

    if buys_today:
        msg += f"\n<b>New Positions:</b>\n"
        for t in buys_today:
            ticker = t.get("ticker", "")
            msg += f"  \U0001F7E1 {ticker} ({get_stock_name(ticker)}) {t.get('shares',0)} shares @ ${t.get('entry_price',0):.2f}\n"

    await send_telegram(msg)
    print("\U0001F4F1 Evening report sent")

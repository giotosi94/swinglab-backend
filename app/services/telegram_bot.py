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


async def build_max_outcome_section(compact=True):
    """
    Report outcome Max Strategy (ARMED / TRIGGERED / performance).
    Usa shadow + plans + signals già in Mongo. Non ricalcola la strategia.
    """
    db = get_db()

    shadow = await db.agent_state.find_one({"_id": "max_strategy_shadow"}) or {}
    action_counts = shadow.get("action_counts") or {}
    status_counts = shadow.get("status_counts") or {}
    regime = shadow.get("regime") or "—"
    version = shadow.get("strategy_version") or "max"
    live = bool(shadow.get("live_execution_enabled"))

    # Piani lifecycle
    state_rows = await db.max_strategy_plans.aggregate([
        {"$group": {"_id": "$lifecycle_state", "count": {"$sum": 1}}},
    ]).to_list(30)
    plan_states = {row["_id"] or "UNKNOWN": row["count"] for row in state_rows}

    total_plans = await db.max_strategy_plans.count_documents({})
    active_plans = await db.max_strategy_plans.count_documents({"is_active": True})
    triggered_plans = await db.max_strategy_plans.count_documents({"triggered_at": {"$ne": None}})

    # Outcome su piani triggered (se presenti)
    triggered_docs = await db.max_strategy_plans.find(
        {"triggered_at": {"$ne": None}},
        {"outcomes": 1, "ticker": 1, "lifecycle_state": 1},
    ).to_list(500)

    def _avg(values):
        values = [v for v in values if isinstance(v, (int, float))]
        return round(sum(values) / len(values), 2) if values else None

    mfe = [(p.get("outcomes") or {}).get("mfe_pct") for p in triggered_docs]
    mae = [(p.get("outcomes") or {}).get("mae_pct") for p in triggered_docs]
    r5 = [(p.get("outcomes") or {}).get("return_5d_pct") for p in triggered_docs]
    r10 = [(p.get("outcomes") or {}).get("return_10d_pct") for p in triggered_docs]

    avg_mfe = _avg(mfe)
    avg_mae = _avg(mae)
    avg_r5 = _avg(r5)
    avg_r10 = _avg(r10)
    win5 = [v for v in r5 if isinstance(v, (int, float))]
    win_rate_5d = round(sum(1 for v in win5 if v > 0) / len(win5) * 100, 1) if win5 else None

    # Top ARMED dal shadow (se disponibili)
    candidates = shadow.get("candidates") or []
    armed = [
        c for c in candidates
        if c.get("shadow_action") == "WOULD_ARM" or c.get("plan_status") in ("ARMED", "TRIGGERED")
    ][:5]

    lines = []
    lines.append("<b>📐 Max Strategy</b>")
    lines.append(f"Regime: <b>{regime}</b> · live: {'ON' if live else 'SHADOW'}")
    lines.append(f"Ver: {version}")
    lines.append("")
    lines.append("<b>Shadow azioni</b>")
    lines.append(
        f"ARM {action_counts.get('WOULD_ARM', 0)} · "
        f"BUY {action_counts.get('WOULD_BUY', 0)} · "
        f"WAIT {action_counts.get('WOULD_WAIT', 0)} · "
        f"REJECT {action_counts.get('WOULD_REJECT', 0)}"
    )
    if status_counts:
        lines.append(
            f"Piani shadow: ARMED {status_counts.get('ARMED', 0)} · "
            f"TRIGGERED {status_counts.get('TRIGGERED', 0)} · "
            f"WAIT {status_counts.get('WAIT_RETEST', 0)}"
        )
    lines.append("")
    lines.append("<b>Lifecycle piani</b>")
    lines.append(f"Totali {total_plans} · attivi {active_plans} · triggered {triggered_plans}")
    if plan_states:
        top_states = sorted(plan_states.items(), key=lambda x: -x[1])[:6]
        lines.append(" · ".join(f"{k} {v}" for k, v in top_states))

    if triggered_plans:
        lines.append("")
        lines.append("<b>Outcome (piani triggered)</b>")
        if avg_mfe is not None:
            lines.append(f"MFE medio: {avg_mfe:+.2f}%")
        if avg_mae is not None:
            lines.append(f"MAE medio: {avg_mae:+.2f}%")
        if avg_r5 is not None:
            wr = f" · WR {win_rate_5d}%" if win_rate_5d is not None else ""
            lines.append(f"Return 5g: {avg_r5:+.2f}%{wr}")
        if avg_r10 is not None:
            lines.append(f"Return 10g: {avg_r10:+.2f}%")

    if armed and not compact:
        lines.append("")
        lines.append("<b>Top ARMED ora</b>")
        for c in armed:
            t = c.get("ticker", "?")
            score = c.get("max_score")
            score_s = f" score {score}" if score is not None else ""
            lines.append(f"  • {t}{score_s} · {c.get('plan_status', c.get('shadow_action', ''))}")

    return "\n".join(lines)


async def send_max_outcome_report():
    """Report Max standalone (manuale o cron)."""
    try:
        body = await build_max_outcome_section(compact=False)
        msg = f"<b>📊 Max Outcome Report</b>\n\n{body}\n\n<i>{datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC</i>"
        ok = await send_telegram(msg)
        print("📱 Max outcome report sent" if ok else "📱 Max outcome report failed")
        return ok
    except Exception as e:
        print(f"Max outcome report error: {e}")
        try:
            await send_telegram(f"⚠️ Max outcome report error: {e}")
        except Exception:
            pass
        return False


async def send_daily_briefing():
    """Morning briefing con stato mercato, top picks, portfolio + Max snapshot."""
    db = get_db()
    sectors = await db.sectors.find().sort("composite_score", -1).to_list(20)
    assets = await db.assets.find(
        {}, {"ticker": 1, "setup_score": 1, "setup_type": 1, "rsi": 1, "price": 1, "sector_code": 1}
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

    regime_emoji = {"BULL": "🟢", "NEUTRAL": "🟡", "BEAR": "🟠", "CRASH": "🔴"}.get(regime, "⚪")

    msg = "<b>🌅 SwingLab Morning Briefing</b>\n\n"
    msg += f"<b>Market:</b> {regime_emoji} {regime} ({confidence:.0f}%)\n"
    if sectors:
        top = sectors[0]
        msg += f"<b>Top Sector:</b> {top.get('code', '')} ({top.get('name', '')}) Score: {top.get('composite_score', 0):.1f}\n"
    msg += f"\n<b>Portfolio:</b> ${equity:,.0f} | {positions} positions\n"
    msg += f"\n<b>Signals:</b>\n"
    msg += f"  🔥 Strong Buy: {len(strong_buys)}\n"
    msg += f"  📈 Buy: {len(buys)}\n"
    msg += f"  📉 Oversold: {len(oversold)}\n"
    msg += f"  ⚠️ Overbought: {len(overbought)}\n"

    if strong_buys:
        msg += f"\n<b>🔥 TOP PICKS:</b>\n"
        for a in strong_buys[:5]:
            t = a.get("ticker", "")
            msg += f"  <b>{t}</b> ({get_stock_name(t)}) ${a.get('price', 0):.2f} Score:{a.get('setup_score', 0)} [{a.get('setup_type', '')}]\n"

    # Max snapshot (compatto) — non interrompe il briefing se fallisce
    try:
        max_section = await build_max_outcome_section(compact=True)
        msg += f"\n{max_section}\n"
    except Exception as e:
        print(f"Morning Max section skipped: {e}")

    await send_telegram(msg)
    print("📱 Daily briefing sent")


async def send_evening_report():
    """Evening report con P&L del giorno + Max outcome."""
    db = get_db()
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    trades = await db.trade_history.find({"date": {"$gte": today}}).to_list(50)
    buys_today = [t for t in trades if t.get("side") == "buy"]
    sells_today = [t for t in trades if t.get("side") == "sell"]

    alpaca_state = await db.auto_trader.find_one({"_id": "alpaca_state"})
    equity = alpaca_state.get("equity", 0) if alpaca_state else 0

    msg = "<b>🌙 SwingLab Evening Report</b>\n\n"
    msg += f"<b>Equity:</b> ${equity:,.2f}\n"
    msg += f"<b>Today:</b> {len(buys_today)} buys, {len(sells_today)} sells\n"

    if sells_today:
        total_pnl = sum(t.get("pnl_pct", 0) for t in sells_today)
        wins = sum(1 for t in sells_today if t.get("pnl_pct", 0) > 0)
        msg += f"\n<b>Closed:</b>\n"
        for t in sells_today:
            emoji = "🟢" if t.get("pnl_pct", 0) > 0 else "🔴"
            ticker = t.get("ticker", "")
            msg += f"  {emoji} {ticker} ({get_stock_name(ticker)}) {t.get('pnl_pct', 0):+.1f}% [{t.get('reason', '')}]\n"
        msg += f"\n<b>Day P&L:</b> {total_pnl:+.1f}% | {wins}/{len(sells_today)} wins\n"

    if buys_today:
        msg += f"\n<b>New Positions:</b>\n"
        for t in buys_today:
            ticker = t.get("ticker", "")
            msg += f"  🟡 {ticker} ({get_stock_name(ticker)}) {t.get('shares', 0)} shares @ ${t.get('entry_price', 0):.2f}\n"

    # Max outcome insieme agli altri report
    try:
        max_section = await build_max_outcome_section(compact=True)
        msg += f"\n{max_section}\n"
    except Exception as e:
        print(f"Evening Max section skipped: {e}")
        msg += f"\n⚠️ Max section error: {e}\n"

    await send_telegram(msg)
    print("📱 Evening report sent")

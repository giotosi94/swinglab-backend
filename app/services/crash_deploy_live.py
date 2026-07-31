"""
Crash Deploy LIVE — Progetto Alpha (VALIDATO +14.5% bear 2022).
ISOLATO dagli agenti. Doppia sicurezza: flag master + dry_run.
Gate: agisce SOLO in regime BEAR/CRASH. Compra SPY a fette nei crash.
"""

from datetime import datetime
from app.db.mongodb import get_db

# Fette progressive (soglie drawdown SPY, frazione del cash disponibile)
FETTE = [
    {"dd": -8,  "id": 1, "frac": 0.30},
    {"dd": -15, "id": 2, "frac": 0.40},
    {"dd": -25, "id": 3, "frac": 0.30},
]


async def get_crash_deploy_state():
    """Stato persistente del crash deploy (fette fatte, posizione SPY, flag)."""
    db = get_db()
    st = await db.crash_deploy_state.find_one({"_id": "state"})
    if not st:
        st = {
            "_id": "state",
            "enabled": False,       # 🔴 MASTER FLAG — OFF di default
            "dry_run": True,        # 🧪 simula senza comprare
            "fette_done": [],
            "spy_shares": 0.0,
            "spy_invested": 0.0,
            "peak_at_entry": 0.0,
            "trimmed": False,
            "events": [],
        }
        await db.crash_deploy_state.update_one({"_id": "state"}, {"$set": st}, upsert=True)
    return st


async def set_crash_deploy_flags(enabled: bool = None, dry_run: bool = None):
    """Toggle master flag e dry-run."""
    db = get_db()
    upd = {}
    if enabled is not None:
        upd["enabled"] = enabled
    if dry_run is not None:
        upd["dry_run"] = dry_run
    if upd:
        await db.crash_deploy_state.update_one({"_id": "state"}, {"$set": upd}, upsert=True)
    return await get_crash_deploy_state()


async def check_and_deploy():
    """
    Chiamato dalla pipeline. Valuta il Crash Radar e, se in BEAR/CRASH,
    esegue (o simula) il deploy a fette su SPY.
    """
    db = get_db()
    state = await get_crash_deploy_state()

    # 1. FLAG MASTER
    if not state.get("enabled", False):
        return {"status": "disabled", "note": "master flag OFF"}

    # 2. Leggi Crash Radar dal market_context
    ctx = await db.market_context.find_one({"_id": "latest"})
    if not ctx:
        return {"status": "no_context"}
    cr = ctx.get("crash_radar", {})
    regime = ctx.get("market_regime", "NEUTRAL")
    spy_dd = cr.get("spy_drawdown_pct", 0)

    # 3. GATE REGIME: agisci solo in BEAR/CRASH
    if regime not in ("BEAR", "CRASH"):
        return {"status": "gate_closed", "regime": regime, "spy_dd": spy_dd,
                "note": "Crash deploy attivo solo in BEAR/CRASH"}

    # 4. Prezzo SPY corrente
    spy = await db.market_regime.find_one({"symbol": "SPY"})
    spy_price = float(spy.get("price", 0)) if spy else 0
    if spy_price <= 0:
        return {"status": "no_spy_price"}

    # 5. Account (cash disponibile)
    from app.services.alpaca_trader import get_account
    account = await get_account()
    cash = float(account.get("cash", 0)) if account else 0

    dry = state.get("dry_run", True)
    actions = []
    fette_done = set(state.get("fette_done", []))

    # 6. Valuta le fette
    for f in FETTE:
        if spy_dd <= f["dd"] and f["id"] not in fette_done:
            deploy_cash = cash * f["frac"]
            if deploy_cash < 100:
                continue

            if dry:
                actions.append({
                    "action": f"[DRY] DEPLOY_FETTA_{f['id']}",
                    "spy_dd": spy_dd, "spy_price": round(spy_price, 2),
                    "would_buy_usd": round(deploy_cash, 2),
                })
            else:
                # 🔴 ESECUZIONE REALE: compra SPY notional
                from app.services.alpaca_trader import place_notional_buy
                res = await place_notional_buy("SPY", round(deploy_cash, 2))
                if res:
                    sh = deploy_cash / spy_price
                    state["spy_shares"] = state.get("spy_shares", 0) + sh
                    state["spy_invested"] = state.get("spy_invested", 0) + deploy_cash
                    fette_done.add(f["id"])
                    actions.append({
                        "action": f"DEPLOY_FETTA_{f['id']}",
                        "spy_dd": spy_dd, "spy_price": round(spy_price, 2),
                        "bought_usd": round(deploy_cash, 2),
                    })
                    # Telegram alert
                    try:
                        from app.services.telegram_bot import send_telegram
                        await send_telegram(
                            f"🔴 <b>CRASH DEPLOY FETTA {f['id']}</b>\n"
                            f"SPY drawdown {spy_dd:.1f}% · comprato ${deploy_cash:.0f} SPY @ ${spy_price:.2f}"
                        )
                    except Exception:
                        pass

    # Salva stato
    state["fette_done"] = list(fette_done)
    if actions:
        state["events"] = (state.get("events", []) + actions)[-50:]
    await db.crash_deploy_state.update_one({"_id": "state"}, {"$set": state}, upsert=True)

    return {
        "status": "ok",
        "regime": regime,
        "spy_dd": spy_dd,
        "dry_run": dry,
        "actions": actions,
        "fette_done": list(fette_done),
    }

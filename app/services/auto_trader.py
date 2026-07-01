"""
auto_trader.py — Wrapper retrocompatibile.
Ora usa il Multi-Agent Orchestrator internamente.
Le route /api/data/autotrader/* continuano a funzionare come prima.

v2.0 — Reset completo e pulito per ripartenza da zero.
"""
from datetime import datetime
from app.db.mongodb import get_db
from app.agents.orchestrator import Orchestrator


# Singleton orchestrator
_orchestrator = None

def _get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator


async def run_auto_trader():
    """Esegue il pipeline multi-agent (sostituisce il vecchio auto_trader)."""
    orch = _get_orchestrator()
    return await orch.run()


async def get_auto_trader_state():
    """Ritorna lo stato del pipeline (backward compat)."""
    db = get_db()
    state = await db.auto_trader.find_one({"_id": "alpaca_state"})
    if state:
        state["_id"] = str(state["_id"])
    return state


async def reset_auto_trader(initial_capital: float = None):
    """
    🔄 RESET COMPLETO v2.1
    
    ⚠️ IMPORTANTE: initial_capital è opzionale.
    Se non fornito, viene letto automaticamente da Alpaca (equity attuale).
    Alpaca è la SINGLE SOURCE OF TRUTH per il capitale.
    
    Pulisce TUTTO lo stato operativo per ripartire da zero, MANTENENDO:
    - stock_bars (cache prezzi storici)
    - assets (universo configurato + fractionable cache)
    - sectors (definizione settori)
    - market_regime (dati macro storici)
    - watchlist
    
    PULISCE:
    - Ordini aperti su Alpaca + posizioni aperte su Alpaca
    - trade_history (storico trade)
    - trailing_stops (gestione stop dinamici)
    - shared_brain (stato condiviso agenti)
    - alpaca_orders (log ordini interno)
    - auto_trader.alpaca_state (stato pipeline)
    - market_context (contesto macro corrente)
    - Brain, decisioni e performance di TUTTI gli agenti
    - Collection legacy: agent_brain, agent_decisions
    """
    from app.services.alpaca_trader import close_all_positions, cancel_all_orders, get_account
    
    db = get_db()
    report = {
        "alpaca": {},
        "cleaned": {},
        "updated": {},
        "errors": [],
        "started_at": datetime.utcnow().isoformat(),
    }
    
    # 🆕 Se initial_capital non fornito, prendi da Alpaca
    if initial_capital is None:
        try:
            account = await get_account()
            if account:
                initial_capital = float(account.get("equity", 100000))
            else:
                initial_capital = 100000
        except Exception as e:
            report["errors"].append(f"get_account: {str(e)}")
            initial_capital = 100000
    
    print("=" * 60)
    print("🔄 SWINGLAB RESET v2.1 — Starting...")
    print(f"   Target capital (from Alpaca): ${initial_capital:,.0f}")
    print("=" * 60)
    
    # ============================================
    # FASE 1: ALPACA — Cancella ordini e chiudi posizioni
    # ============================================
    print("\n📡 [1/4] Cleaning Alpaca account...")
    try:
        cancel_result = await cancel_all_orders()
        report["alpaca"]["orders_cancelled"] = "ok" if cancel_result is not None else "skipped"
        print(f"  ✅ Orders cancelled: {report['alpaca']['orders_cancelled']}")
    except Exception as e:
        report["errors"].append(f"cancel_all_orders: {str(e)}")
        report["alpaca"]["orders_cancelled"] = f"error: {str(e)}"
        print(f"  ❌ Cancel orders error: {e}")
    
    try:
        close_result = await close_all_positions()
        report["alpaca"]["positions_closed"] = "ok" if close_result is not None else "skipped"
        print(f"  ✅ Positions closed: {report['alpaca']['positions_closed']}")
    except Exception as e:
        report["errors"].append(f"close_all_positions: {str(e)}")
        report["alpaca"]["positions_closed"] = f"error: {str(e)}"
        print(f"  ❌ Close positions error: {e}")
    
    # ============================================
    # FASE 2: TRADE HISTORY + TRAILING STOPS + ORDERS LOG
    # ============================================
    print("\n🗑️  [2/4] Cleaning trade data...")
    
    collections_trade = [
        "trade_history",
        "trailing_stops",
        "alpaca_orders",
        "auto_trader",
        "market_context",
        "shared_brain",
    ]
    
    for coll_name in collections_trade:
        try:
            result = await db[coll_name].delete_many({})
            report["cleaned"][coll_name] = result.deleted_count
            print(f"  ✅ {coll_name}: {result.deleted_count} docs deleted")
        except Exception as e:
            report["errors"].append(f"{coll_name}: {str(e)}")
            report["cleaned"][coll_name] = f"error: {str(e)}"
            print(f"  ❌ {coll_name} error: {e}")
    
    # ============================================
    # FASE 3: AGENT BRAINS, DECISIONS, PERFORMANCE
    # ============================================
    print("\n🧠 [3/4] Cleaning agent brains...")
    
    agent_names = ["macro_analyst", "alpha_strategist", "risk_manager", "executor"]
    for agent_name in agent_names:
        for prefix in ["agent_memory", "agent_decisions", "agent_performance"]:
            coll_name = f"{prefix}_{agent_name}"
            try:
                result = await db[coll_name].delete_many({})
                report["cleaned"][coll_name] = result.deleted_count
                print(f"  ✅ {coll_name}: {result.deleted_count} docs")
            except Exception as e:
                report["errors"].append(f"{coll_name}: {str(e)}")
                print(f"  ❌ {coll_name} error: {e}")
    
    # Collection legacy
    print("\n🗑️  Cleaning legacy collections...")
    legacy_collections = ["agent_brain", "agent_decisions"]
    for coll_name in legacy_collections:
        try:
            result = await db[coll_name].delete_many({})
            report["cleaned"][f"legacy_{coll_name}"] = result.deleted_count
            print(f"  ✅ {coll_name} (legacy): {result.deleted_count} docs")
        except Exception as e:
            report["errors"].append(f"legacy_{coll_name}: {str(e)}")
            print(f"  ❌ legacy {coll_name} error: {e}")
    
    # ============================================
    # FASE 4: NOTE — NON aggiorniamo più starting_capital
    # ============================================
    # 🆕 v2.1: starting_capital NON viene più salvato in app_settings.
    # Fonte di verità = Alpaca (endpoint /api/data/starting-capital)
    print(f"\n⚙️  [4/4] Skipping starting_capital update (now from Alpaca)")
    
    # ✅ Rimuovi vecchio starting_capital residuo dai settings
    try:
        settings_update = await db.app_settings.update_one(
            {"_id": "risk_params"},
            {"$unset": {"starting_capital": ""}},
            upsert=False,
        )
        if settings_update.modified_count > 0:
            print(f"  🧹 Removed old starting_capital from app_settings")
    except Exception as e:
        print(f"  ⚠️ Cleanup old starting_capital: {e}")
    
    # ============================================
    # FINAL REPORT
    # ============================================
    report["finished_at"] = datetime.utcnow().isoformat()
    report["initial_capital"] = float(initial_capital)
    report["capital_source"] = "alpaca"
    report["status"] = "success" if not report["errors"] else "partial"
    
    total_deleted = sum(v for v in report["cleaned"].values() if isinstance(v, int))
    
    print("\n" + "=" * 60)
    print(f"🏁 RESET COMPLETE")
    print(f"   Total docs deleted: {total_deleted}")
    print(f"   Errors: {len(report['errors'])}")
    print(f"   Capital (from Alpaca): ${initial_capital:,.0f}")
    if report["errors"]:
        print(f"   ⚠️  Errors: {report['errors']}")
    print("=" * 60)
    
    report["message"] = (
        f"Reset complete: {total_deleted} docs deleted, "
        f"capital from Alpaca: ${initial_capital:,.0f}"
    )
    
    return report

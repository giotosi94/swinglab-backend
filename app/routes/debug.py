"""
🚨 EMERGENCY DEBUG ENDPOINTS
Endpoint critici per ripristinare SL/TP su posizioni esistenti
quando il bug _cancel_stale_orders li ha eliminati.

Uso temporaneo - da rimuovere dopo i fix permanenti.
"""
from fastapi import APIRouter, Query
from datetime import datetime

from app.db.mongodb import get_db
from app.services.alpaca_trader import (
    get_positions,
    get_orders,
    place_order,
)

router = APIRouter()


@router.get("/positions-status")
async def positions_status():
    """
    Diagnostica dettagliata: per ogni posizione mostra
    stato SL/TP atteso (DB) vs attuale (Alpaca).
    
    Non modifica nulla, solo lettura.
    """
    db = get_db()
    positions = await get_positions() or []
    open_orders = await get_orders(status="open", limit=100) or []

    valid_statuses = ("new", "accepted", "pending_new", "held", "partially_filled")

    open_stops = {
        o.get("symbol"): o for o in open_orders
        if o.get("side") == "sell"
        and o.get("type") in ("stop", "stop_limit")
        and o.get("status") in valid_statuses
    }
    open_limits = {
        o.get("symbol"): o for o in open_orders
        if o.get("side") == "sell"
        and o.get("type") == "limit"
        and o.get("status") in valid_statuses
    }

    rows = []
    for p in positions:
        ticker = p.get("symbol")
        try:
            current_price = float(p.get("current_price", 0))
            avg_entry = float(p.get("avg_entry_price", 0))
            qty = int(float(p.get("qty", 0)))

            buy_trade = await db.trade_history.find_one(
                {"ticker": ticker, "side": "buy", "sell_linked": {"$ne": True}},
                sort=[("date", -1)]
            )
            trailing = await db.trailing_stops.find_one({"ticker": ticker})

            stored_sl = float(buy_trade.get("stop_loss", 0)) if buy_trade else 0
            stored_tp = float(buy_trade.get("target", 0)) if buy_trade else 0
            trailing_sl = float(trailing.get("stop_price", 0)) if trailing else 0
            effective_sl = max(stored_sl, trailing_sl)

            active_stop = open_stops.get(ticker)
            active_limit = open_limits.get(ticker)

            sl_status = "NO_SL_CONFIG"
            if effective_sl > 0:
                if current_price <= effective_sl:
                    sl_status = "VIOLATED"
                elif not active_stop:
                    sl_status = "MISSING_ON_ALPACA"
                else:
                    sl_status = "ACTIVE"

            tp_status = "NO_TP_CONFIG"
            if stored_tp > 0:
                if current_price >= stored_tp:
                    tp_status = "REACHED"
                elif not active_limit:
                    tp_status = "MISSING_ON_ALPACA"
                else:
                    tp_status = "ACTIVE"

            rows.append({
                "ticker": ticker,
                "qty": qty,
                "current_price": current_price,
                "entry_price": avg_entry,
                "pnl_pct": round(float(p.get("unrealized_plpc", 0)) * 100, 2),
                "stored_sl": stored_sl,
                "stored_tp": stored_tp,
                "trailing_sl": trailing_sl,
                "effective_sl": effective_sl,
                "active_stop_order": {
                    "id": active_stop.get("id") if active_stop else None,
                    "stop_price": float(active_stop.get("stop_price", 0)) if active_stop else None,
                },
                "active_limit_order": {
                    "id": active_limit.get("id") if active_limit else None,
                    "limit_price": float(active_limit.get("limit_price", 0)) if active_limit else None,
                },
                "sl_status": sl_status,
                "tp_status": tp_status,
            })
        except Exception as e:
            rows.append({
                "ticker": ticker,
                "error": str(e),
            })

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "total_positions": len(rows),
        "positions": rows,
    }
    

@router.post("/restore-stops")
async def restore_stops(dry_run: bool = Query(default=True)):
    """
    🛡️ RIPRISTINA SL/TP PER POSIZIONI APERTE.
    
    Per ogni posizione su Alpaca:
    1. Recupera il BUY originale dal DB (campi stop_loss e target)
    2. Verifica se esiste già un ordine SELL stop attivo per quel ticker
    3. Verifica se esiste già un ordine SELL limit (TP) attivo per quel ticker
    4. Se mancano → li ricrea con time_in_force="gtc"
    
    Considera trailing_stops aggiornati nel DB (priorità su stored_sl).
    
    Args:
        dry_run: se True (default), simula soltanto.
                 Se False, esegue realmente gli ordini su Alpaca.
    
    Returns:
        Report dettagliato di tutte le azioni effettuate / da effettuare.
    """
    db = get_db()
    report = {
        "dry_run": dry_run,
        "timestamp": datetime.utcnow().isoformat(),
        "positions_checked": 0,
        "sl_restored": [],
        "tp_restored": [],
        "sl_already_active": [],
        "tp_already_active": [],
        "skipped": [],
        "errors": [],
    }

    positions = await get_positions() or []
    open_orders = await get_orders(status="open", limit=100) or []
    report["positions_checked"] = len(positions)

    if not positions:
        report["message"] = "Nessuna posizione aperta su Alpaca."
        return report

    valid_statuses = ("new", "accepted", "pending_new", "held", "partially_filled")

    open_sell_stops = {
        o.get("symbol"): o for o in open_orders
        if o.get("side") == "sell"
        and o.get("type") in ("stop", "stop_limit")
        and o.get("status") in valid_statuses
    }
    open_sell_limits = {
        o.get("symbol"): o for o in open_orders
        if o.get("side") == "sell"
        and o.get("type") == "limit"
        and o.get("status") in valid_statuses
    }

    for pos in positions:
        ticker = pos.get("symbol")
        try:
            qty = int(float(pos.get("qty", 0)))
            current_price = float(pos.get("current_price", 0))
            avg_entry = float(pos.get("avg_entry_price", 0))

            if qty <= 0:
                report["skipped"].append({"ticker": ticker, "reason": "qty=0"})
                continue

            buy_trade = await db.trade_history.find_one(
                {"ticker": ticker, "side": "buy", "sell_linked": {"$ne": True}},
                sort=[("date", -1)]
            )
            if not buy_trade:
                report["skipped"].append({
                    "ticker": ticker,
                    "reason": "No BUY trade found in DB (manual position?)"
                })
                continue

            stored_sl = float(buy_trade.get("stop_loss", 0) or 0)
            stored_tp = float(buy_trade.get("target", 0) or 0)

            trailing = await db.trailing_stops.find_one({"ticker": ticker})
            trailing_sl = float(trailing.get("stop_price", 0)) if trailing else 0
            effective_sl = max(stored_sl, trailing_sl)

            # ========== RESTORE STOP LOSS ==========
            if effective_sl > 0:
                if ticker in open_sell_stops:
                    existing_sp = float(open_sell_stops[ticker].get("stop_price", 0) or 0)
                    report["sl_already_active"].append({
                        "ticker": ticker,
                        "active_stop_price": existing_sp,
                        "expected": effective_sl,
                    })
                else:
                    if effective_sl >= current_price:
                        report["errors"].append({
                            "ticker": ticker,
                            "type": "SL_ALREADY_VIOLATED",
                            "current_price": current_price,
                            "stop_loss": effective_sl,
                            "action": "MANUAL_REVIEW_REQUIRED",
                            "suggestion": "Prezzo già sotto SL: chiudere manualmente.",
                        })
                    else:
                        action = {
                            "ticker": ticker,
                            "qty": qty,
                            "stop_price": round(effective_sl, 2),
                            "current_price": current_price,
                            "entry_price": avg_entry,
                            "type": "stop",
                            "time_in_force": "gtc",
                        }
                        if not dry_run:
                            result = await place_order(
                                symbol=ticker,
                                qty=qty,
                                side="sell",
                                order_type="stop",
                                time_in_force="gtc",
                                stop_price=round(effective_sl, 2),
                            )
                            if result:
                                action["order_id"] = result.get("id", "")
                                action["status"] = "PLACED"
                            else:
                                action["status"] = "FAILED"
                                report["errors"].append({
                                    "ticker": ticker,
                                    "type": "SL_PLACE_FAILED",
                                    "details": "Alpaca returned None"
                                })
                                continue
                        else:
                            action["status"] = "WOULD_PLACE"
                        report["sl_restored"].append(action)

            # ========== RESTORE TAKE PROFIT ==========
            if stored_tp > 0:
                if ticker in open_sell_limits:
                    existing_lp = float(open_sell_limits[ticker].get("limit_price", 0) or 0)
                    report["tp_already_active"].append({
                        "ticker": ticker,
                        "active_limit_price": existing_lp,
                        "expected": stored_tp,
                    })
                else:
                    if stored_tp <= current_price:
                        report["errors"].append({
                            "ticker": ticker,
                            "type": "TP_ALREADY_REACHED",
                            "current_price": current_price,
                            "target": stored_tp,
                            "action": "MANUAL_REVIEW_REQUIRED",
                            "suggestion": "Prezzo già sopra target: vendere manualmente.",
                        })
                    else:
                        action = {
                            "ticker": ticker,
                            "qty": qty,
                            "limit_price": round(stored_tp, 2),
                            "current_price": current_price,
                            "entry_price": avg_entry,
                            "type": "limit",
                            "time_in_force": "gtc",
                        }
                        if not dry_run:
                            result = await place_order(
                                symbol=ticker,
                                qty=qty,
                                side="sell",
                                order_type="limit",
                                time_in_force="gtc",
                                limit_price=round(stored_tp, 2),
                            )
                            if result:
                                action["order_id"] = result.get("id", "")
                                action["status"] = "PLACED"
                            else:
                                action["status"] = "FAILED"
                                report["errors"].append({
                                    "ticker": ticker,
                                    "type": "TP_PLACE_FAILED",
                                    "details": "Alpaca returned None"
                                })
                                continue
                        else:
                            action["status"] = "WOULD_PLACE"
                        report["tp_restored"].append(action)

        except Exception as e:
            report["errors"].append({
                "ticker": ticker,
                "type": "EXCEPTION",
                "details": str(e),
            })

    report["summary"] = {
        "sl_to_place" if dry_run else "sl_placed": len(report["sl_restored"]),
        "tp_to_place" if dry_run else "tp_placed": len(report["tp_restored"]),
        "sl_already_ok": len(report["sl_already_active"]),
        "tp_already_ok": len(report["tp_already_active"]),
        "errors": len(report["errors"]),
        "skipped": len(report["skipped"]),
    }
    return report

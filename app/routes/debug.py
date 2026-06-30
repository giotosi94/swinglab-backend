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
    place_oco_order,
    cancel_order,
    close_position,
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
    

@router.post("/cancel-orphan-stops")
async def cancel_orphan_stops(dry_run: bool = Query(default=True)):
    """
    🧹 Cancella tutti gli ordini SELL stop "orfani" (cioè non parte di OCO/bracket).
    
    Serve a ripulire prima di piazzare nuovi OCO orders.
    
    Args:
        dry_run: se True simula, se False cancella veramente.
    """
    report = {
        "dry_run": dry_run,
        "timestamp": datetime.utcnow().isoformat(),
        "cancelled": [],
        "skipped": [],
        "errors": [],
    }

    open_orders = await get_orders(status="open", limit=100) or []
    valid_statuses = ("new", "accepted", "pending_new", "held", "partially_filled")
    
    for o in open_orders:
        if o.get("side") != "sell":
            continue
        if o.get("type") not in ("stop", "stop_limit", "limit"):
            continue
        if o.get("status") not in valid_statuses:
            continue
        
        order_class = o.get("order_class", "")
        # SKIP se fa parte di OCO o bracket (non vogliamo rompere quelli buoni)
        if order_class in ("oco", "bracket", "oto"):
            report["skipped"].append({
                "ticker": o.get("symbol"),
                "order_id": o.get("id"),
                "reason": f"Belongs to {order_class}",
            })
            continue
        
        action = {
            "ticker": o.get("symbol"),
            "order_id": o.get("id"),
            "type": o.get("type"),
            "stop_price": o.get("stop_price"),
            "limit_price": o.get("limit_price"),
        }
        
        if dry_run:
            action["status"] = "WOULD_CANCEL"
        else:
            try:
                result = await cancel_order(o.get("id"))
                action["status"] = "CANCELLED" if result is not None else "FAILED"
                if result is None:
                    report["errors"].append({
                        "ticker": o.get("symbol"),
                        "order_id": o.get("id"),
                        "error": "cancel_order returned None"
                    })
            except Exception as e:
                action["status"] = "EXCEPTION"
                report["errors"].append({
                    "ticker": o.get("symbol"),
                    "error": str(e)
                })
        
        report["cancelled"].append(action)

    report["summary"] = {
        "cancelled": len([c for c in report["cancelled"] if c.get("status") in ("CANCELLED", "WOULD_CANCEL")]),
        "skipped_oco_bracket": len(report["skipped"]),
        "errors": len(report["errors"]),
    }
    return report


@router.post("/restore-stops-oco")
async def restore_stops_oco(dry_run: bool = Query(default=True)):
    """
    🎯 Ripristina SL+TP usando OCO orders (One-Cancels-Other).
    
    Per ogni posizione su Alpaca:
    1. Recupera stop_loss e target dal DB (trade_history)
    2. Verifica se esiste già un OCO attivo
    3. Se mancano entrambi → piazza un OCO con SL+TP linkati
    
    PRE-REQUISITO: devi prima cancellare gli SL/TP orfani con cancel-orphan-stops.
    """
    db = get_db()
    report = {
        "dry_run": dry_run,
        "timestamp": datetime.utcnow().isoformat(),
        "positions_checked": 0,
        "oco_placed": [],
        "already_protected": [],
        "skipped": [],
        "errors": [],
    }

    positions = await get_positions() or []
    open_orders = await get_orders(status="open", limit=100) or []
    report["positions_checked"] = len(positions)

    if not positions:
        report["message"] = "Nessuna posizione aperta."
        return report

    valid_statuses = ("new", "accepted", "pending_new", "held", "partially_filled")
    
    # Indicizza OCO/bracket attivi per ticker
    protected_tickers = set()
    for o in open_orders:
        if o.get("status") not in valid_statuses:
            continue
        if o.get("side") != "sell":
            continue
        if o.get("order_class") in ("oco", "bracket"):
            protected_tickers.add(o.get("symbol"))

    for pos in positions:
        ticker = pos.get("symbol")
        try:
            qty = int(float(pos.get("qty", 0)))
            current_price = float(pos.get("current_price", 0))
            avg_entry = float(pos.get("avg_entry_price", 0))

            if qty <= 0:
                report["skipped"].append({"ticker": ticker, "reason": "qty=0"})
                continue

            # Se già protetto da OCO/bracket → skip
            if ticker in protected_tickers:
                report["already_protected"].append({
                    "ticker": ticker,
                    "reason": "OCO/bracket already active"
                })
                continue

            buy_trade = await db.trade_history.find_one(
                {"ticker": ticker, "side": "buy", "sell_linked": {"$ne": True}},
                sort=[("date", -1)]
            )
            if not buy_trade:
                report["skipped"].append({
                    "ticker": ticker,
                    "reason": "No BUY trade in DB"
                })
                continue

            stored_sl = float(buy_trade.get("stop_loss", 0) or 0)
            stored_tp = float(buy_trade.get("target", 0) or 0)

            trailing = await db.trailing_stops.find_one({"ticker": ticker})
            trailing_sl = float(trailing.get("stop_price", 0)) if trailing else 0
            effective_sl = max(stored_sl, trailing_sl)

            # OCO richiede SIA SL SIA TP > 0
            if effective_sl <= 0 or stored_tp <= 0:
                report["skipped"].append({
                    "ticker": ticker,
                    "reason": f"Missing SL or TP (SL={effective_sl}, TP={stored_tp})"
                })
                continue

            # Validità prezzi: SL < current < TP
            if effective_sl >= current_price:
                report["errors"].append({
                    "ticker": ticker,
                    "type": "SL_ALREADY_VIOLATED",
                    "current_price": current_price,
                    "stop_loss": effective_sl,
                    "action": "MANUAL_REVIEW_REQUIRED",
                })
                continue
            
            if stored_tp <= current_price:
                report["errors"].append({
                    "ticker": ticker,
                    "type": "TP_ALREADY_REACHED",
                    "current_price": current_price,
                    "target": stored_tp,
                    "action": "MANUAL_REVIEW_REQUIRED",
                })
                continue

            action = {
                "ticker": ticker,
                "qty": qty,
                "stop_loss": round(effective_sl, 2),
                "take_profit": round(stored_tp, 2),
                "current_price": current_price,
                "entry_price": avg_entry,
            }
            
            if dry_run:
                action["status"] = "WOULD_PLACE_OCO"
            else:
                result = await place_oco_order(
                    symbol=ticker,
                    qty=qty,
                    take_profit_price=stored_tp,
                    stop_loss_price=effective_sl,
                    time_in_force="gtc",
                )
                if result:
                    action["order_id"] = result.get("id", "")
                    action["status"] = "PLACED"
                else:
                    action["status"] = "FAILED"
                    report["errors"].append({
                        "ticker": ticker,
                        "type": "OCO_PLACE_FAILED",
                        "details": "Alpaca returned None"
                    })
                    continue
            
            report["oco_placed"].append(action)

        except Exception as e:
            report["errors"].append({
                "ticker": ticker,
                "type": "EXCEPTION",
                "details": str(e),
            })

    report["summary"] = {
        "oco_to_place" if dry_run else "oco_placed": len(report["oco_placed"]),
        "already_protected": len(report["already_protected"]),
        "errors": len(report["errors"]),
        "skipped": len(report["skipped"]),
    }
    return report
    

@router.post("/close-position/{ticker}")
async def close_position_endpoint(ticker: str, dry_run: bool = Query(default=True)):
    """
    🚨 Chiude immediatamente una posizione a mercato.
    
    1. Cancella prima eventuali ordini sell aperti (SL/TP/OCO) per il ticker
    2. Chiama close_position di Alpaca (vende tutto a mercato)
    3. Marca il BUY originale come sell_linked=True nel DB
    
    Args:
        ticker: simbolo della posizione (es. "BMY")
        dry_run: se True simula, se False chiude veramente.
    """
    db = get_db()
    ticker = ticker.upper()
    report = {
        "ticker": ticker,
        "dry_run": dry_run,
        "timestamp": datetime.utcnow().isoformat(),
        "actions": [],
        "errors": [],
    }

    positions = await get_positions() or []
    target_pos = next((p for p in positions if p.get("symbol") == ticker), None)
    if not target_pos:
        report["errors"].append(f"No open position for {ticker}")
        return report

    qty = int(float(target_pos.get("qty", 0)))
    current_price = float(target_pos.get("current_price", 0))
    entry_price = float(target_pos.get("avg_entry_price", 0))
    pnl_pct = float(target_pos.get("unrealized_plpc", 0)) * 100
    
    report["position"] = {
        "qty": qty,
        "current_price": current_price,
        "entry_price": entry_price,
        "pnl_pct": round(pnl_pct, 2),
    }

    open_orders = await get_orders(status="open", limit=100) or []
    sell_orders = [
        o for o in open_orders
        if o.get("symbol") == ticker and o.get("side") == "sell"
    ]
    
    for o in sell_orders:
        action = {
            "type": "CANCEL_ORDER",
            "order_id": o.get("id"),
            "order_type": o.get("type"),
            "order_class": o.get("order_class", ""),
        }
        if dry_run:
            action["status"] = "WOULD_CANCEL"
        else:
            try:
                result = await cancel_order(o.get("id"))
                action["status"] = "CANCELLED" if result is not None else "FAILED"
            except Exception as e:
                action["status"] = "EXCEPTION"
                report["errors"].append(f"Cancel error: {e}")
        report["actions"].append(action)

    close_action = {
        "type": "CLOSE_POSITION",
        "ticker": ticker,
        "qty": qty,
        "current_price": current_price,
        "estimated_pnl_pct": round(pnl_pct, 2),
    }
    
    if dry_run:
        close_action["status"] = "WOULD_CLOSE"
    else:
        try:
            result = await close_position(ticker)
            if result is not None:
                close_action["status"] = "CLOSED"
                close_action["order_id"] = result.get("id", "")
                
                update_result = await db.trade_history.update_one(
                    {"ticker": ticker, "side": "buy", "sell_linked": {"$ne": True}},
                    {"$set": {"sell_linked": True, "sell_linked_at": datetime.utcnow()}},
                    upsert=False
                )
                close_action["db_updated"] = update_result.modified_count > 0
            else:
                close_action["status"] = "FAILED"
                report["errors"].append("close_position returned None")
        except Exception as e:
            close_action["status"] = "EXCEPTION"
            report["errors"].append(f"Close error: {e}")
    
    report["actions"].append(close_action)
    return report


# ============================================
# 🆕 v2.1 — POPULATE FRACTIONABLE FLAG
# ============================================

@router.post("/populate-fractionable")
async def populate_fractionable():
    """
    🔧 One-shot admin endpoint.
    Per ogni asset in db.assets, controlla su Alpaca se è fractionable
    e salva il flag in DB.
    
    Da chiamare UNA volta dopo il deploy del Fix #3.
    Poi il RiskManager userà sempre la cache DB senza chiamare Alpaca.
    
    Returns:
        report con totali, errori, e dettagli
    """
    from datetime import datetime
    from app.services.alpaca_trader import is_fractionable
    from app.db.mongodb import get_db
    
    db = get_db()
    
    # Carica tutti i ticker
    assets = await db.assets.find({}, {"ticker": 1}).to_list(500)
    if not assets:
        return {"error": "No assets in db", "checked": 0}
    
    tickers = [a["ticker"] for a in assets if a.get("ticker")]
    
    report = {
        "started_at": datetime.utcnow().isoformat(),
        "total_assets": len(tickers),
        "fractionable": [],
        "not_fractionable": [],
        "errors": [],
    }
    
    print(f"🔍 Populating fractionable flag for {len(tickers)} assets...")
    
    for ticker in tickers:
        try:
            is_frac = await is_fractionable(ticker)
            
            await db.assets.update_one(
                {"ticker": ticker},
                {"$set": {
                    "fractionable": bool(is_frac),
                    "fractionable_checked_at": datetime.utcnow(),
                }}
            )
            
            if is_frac:
                report["fractionable"].append(ticker)
            else:
                report["not_fractionable"].append(ticker)
            
            print(f"  {'✅' if is_frac else '❌'} {ticker}: fractionable={is_frac}")
        
        except Exception as e:
            report["errors"].append({"ticker": ticker, "error": str(e)})
            print(f"  ⚠️ {ticker} error: {e}")
    
    report["finished_at"] = datetime.utcnow().isoformat()
    report["summary"] = {
        "fractionable_count": len(report["fractionable"]),
        "not_fractionable_count": len(report["not_fractionable"]),
        "errors_count": len(report["errors"]),
    }
    
    print(f"\n🏁 DONE: {report['summary']['fractionable_count']} fractionable, "
          f"{report['summary']['not_fractionable_count']} not, "
          f"{report['summary']['errors_count']} errors")
    
    return report

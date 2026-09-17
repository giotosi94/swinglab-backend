"""
EMERGENCY DEBUG ENDPOINTS
Ripristino SL/TP su posizioni esistenti quando mancano su Alpaca.

v2 — Fix critico: qty frazionaria non più troncata con int().
     Le posizioni notional (es. 45.6789 shares) venivano protette
     solo per la parte intera, e quelle < 1 share saltate del tutto.
"""
from fastapi import APIRouter, Query
from datetime import datetime
from app.db.mongodb import get_db
from app.services.alpaca_trader import (
    get_positions,
    get_orders,
    place_order,
    cancel_order,
    close_position,
)

router = APIRouter()

VALID_ORDER_STATUSES = ("new", "accepted", "pending_new", "held", "partially_filled")
MIN_QTY = 0.0001


def _qty_of(pos) -> float:
    """Quantita' reale della posizione, frazionaria. MAI int()."""
    try:
        return round(float(pos.get("qty", 0) or 0), 4)
    except (TypeError, ValueError):
        return 0.0


@router.get("/positions-status")
async def positions_status():
    """
    Diagnostica: per ogni posizione mostra SL/TP attesi (DB) vs attivi (Alpaca).
    Sola lettura, non modifica nulla.
    """
    db = get_db()
    positions = await get_positions() or []
    open_orders = await get_orders(status="open", limit=100) or []

    open_stops = {
        o.get("symbol"): o for o in open_orders
        if o.get("side") == "sell"
        and o.get("type") in ("stop", "stop_limit")
        and o.get("status") in VALID_ORDER_STATUSES
    }
    open_limits = {
        o.get("symbol"): o for o in open_orders
        if o.get("side") == "sell"
        and o.get("type") == "limit"
        and o.get("status") in VALID_ORDER_STATUSES
    }

    rows = []
    unprotected = 0

    for p in positions:
        ticker = p.get("symbol")
        try:
            current_price = float(p.get("current_price", 0))
            avg_entry = float(p.get("avg_entry_price", 0))
            qty = _qty_of(p)

            buy_trade = await db.trade_history.find_one(
                {"ticker": ticker, "side": "buy", "sell_linked": {"$ne": True}},
                sort=[("date", -1)]
            )
            trailing = await db.trailing_stops.find_one({"ticker": ticker})

            stored_sl = float(buy_trade.get("stop_loss", 0) or 0) if buy_trade else 0
            stored_tp = float(buy_trade.get("target", 0) or 0) if buy_trade else 0
            trailing_sl = float(trailing.get("stop_price", 0) or 0) if trailing else 0
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

            if sl_status in ("NO_SL_CONFIG", "MISSING_ON_ALPACA", "VIOLATED"):
                unprotected += 1

            rows.append({
                "ticker": ticker,
                "qty": qty,
                "is_fractional": qty != int(qty),
                "current_price": current_price,
                "entry_price": avg_entry,
                "pnl_pct": round(float(p.get("unrealized_plpc", 0)) * 100, 2),
                "stored_sl": stored_sl,
                "stored_tp": stored_tp,
                "trailing_sl": trailing_sl,
                "effective_sl": effective_sl,
                "has_buy_trade": bool(buy_trade),
                "apm_managed": bool(trailing and trailing.get("apm_managed")),
                "last_target_hit": buy_trade.get("last_target_hit", 0) if buy_trade else None,
                "active_stop_order": {
                    "id": active_stop.get("id") if active_stop else None,
                    "stop_price": float(active_stop.get("stop_price", 0)) if active_stop else None,
                    "qty": active_stop.get("qty") if active_stop else None,
                },
                "active_limit_order": {
                    "id": active_limit.get("id") if active_limit else None,
                    "limit_price": float(active_limit.get("limit_price", 0)) if active_limit else None,
                    "qty": active_limit.get("qty") if active_limit else None,
                },
                "sl_status": sl_status,
                "tp_status": tp_status,
            })
        except Exception as e:
            rows.append({"ticker": ticker, "error": str(e)})

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "total_positions": len(rows),
        "unprotected": unprotected,
        "positions": rows,
    }


@router.post("/restore-stops")
async def restore_stops(dry_run: bool = Query(default=True)):
    """
    Ripristina SL/TP mancanti sulle posizioni aperte.

    v2 — qty frazionaria preservata (round 4 decimali, niente int()).

    Per ogni posizione:
    1. Legge stop_loss e target dal BUY in trade_history
    2. Applica il trailing_stop del DB se piu' alto
    3. Se manca l'ordine su Alpaca lo ricrea (GTC)

    dry_run=True (default) simula soltanto.
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

    open_sell_stops = {
        o.get("symbol"): o for o in open_orders
        if o.get("side") == "sell"
        and o.get("type") in ("stop", "stop_limit")
        and o.get("status") in VALID_ORDER_STATUSES
    }
    open_sell_limits = {
        o.get("symbol"): o for o in open_orders
        if o.get("side") == "sell"
        and o.get("type") == "limit"
        and o.get("status") in VALID_ORDER_STATUSES
    }

    for pos in positions:
        ticker = pos.get("symbol")
        try:
            qty = _qty_of(pos)
            current_price = float(pos.get("current_price", 0))
            avg_entry = float(pos.get("avg_entry_price", 0))

            if qty < MIN_QTY:
                report["skipped"].append({"ticker": ticker, "reason": f"qty troppo piccola ({qty})"})
                continue

            buy_trade = await db.trade_history.find_one(
                {"ticker": ticker, "side": "buy", "sell_linked": {"$ne": True}},
                sort=[("date", -1)]
            )

            if not buy_trade:
                report["skipped"].append({
                    "ticker": ticker,
                    "reason": "Nessun BUY nel DB (posizione manuale?) — usa /fix-orphaned-positions",
                })
                continue

            stored_sl = float(buy_trade.get("stop_loss", 0) or 0)
            stored_tp = float(buy_trade.get("target", 0) or 0)

            trailing = await db.trailing_stops.find_one({"ticker": ticker})
            trailing_sl = float(trailing.get("stop_price", 0) or 0) if trailing else 0
            effective_sl = max(stored_sl, trailing_sl)

            # ========== STOP LOSS ==========
            if effective_sl > 0:
                if ticker in open_sell_stops:
                    report["sl_already_active"].append({
                        "ticker": ticker,
                        "active_stop_price": float(open_sell_stops[ticker].get("stop_price", 0) or 0),
                        "expected": effective_sl,
                    })
                elif effective_sl >= current_price:
                    report["errors"].append({
                        "ticker": ticker,
                        "type": "SL_ALREADY_VIOLATED",
                        "current_price": current_price,
                        "stop_loss": effective_sl,
                        "action": "MANUAL_REVIEW_REQUIRED",
                        "suggestion": "Prezzo gia' sotto SL: valutare chiusura manuale.",
                    })
                else:
                    action = {
                        "ticker": ticker,
                        "qty": qty,
                        "stop_price": round(effective_sl, 2),
                        "current_price": current_price,
                        "entry_price": avg_entry,
                        "source": "trailing" if trailing_sl > stored_sl else "buy_trade",
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
                                "details": "Alpaca returned None",
                            })
                            continue
                    else:
                        action["status"] = "WOULD_PLACE"
                    report["sl_restored"].append(action)
            else:
                report["errors"].append({
                    "ticker": ticker,
                    "type": "NO_SL_IN_DB",
                    "action": "MANUAL_REVIEW_REQUIRED",
                    "suggestion": "stop_loss assente nel BUY: usa /fix-position-targets",
                })

            # ========== TAKE PROFIT ==========
            if stored_tp > 0:
                if ticker in open_sell_limits:
                    report["tp_already_active"].append({
                        "ticker": ticker,
                        "active_limit_price": float(open_sell_limits[ticker].get("limit_price", 0) or 0),
                        "expected": stored_tp,
                    })
                elif stored_tp <= current_price:
                    report["errors"].append({
                        "ticker": ticker,
                        "type": "TP_ALREADY_REACHED",
                        "current_price": current_price,
                        "target": stored_tp,
                        "action": "MANUAL_REVIEW_REQUIRED",
                    })
                else:
                    action = {
                        "ticker": ticker,
                        "qty": qty,
                        "limit_price": round(stored_tp, 2),
                        "current_price": current_price,
                        "entry_price": avg_entry,
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
                                "details": "Alpaca returned None",
                            })
                            continue
                    else:
                        action["status"] = "WOULD_PLACE"
                    report["tp_restored"].append(action)

        except Exception as e:
            report["errors"].append({"ticker": ticker, "type": "EXCEPTION", "details": str(e)})

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
    Cancella gli ordini SELL orfani: quelli su ticker che NON hanno piu'
    una posizione aperta. Non tocca gli ordini di posizioni vive.
    """
    report = {
        "dry_run": dry_run,
        "timestamp": datetime.utcnow().isoformat(),
        "cancelled": [],
        "skipped": [],
        "errors": [],
    }

    positions = await get_positions() or []
    open_tickers = {p.get("symbol") for p in positions}
    open_orders = await get_orders(status="open", limit=100) or []

    for o in open_orders:
        if o.get("side") != "sell":
            continue
        if o.get("type") not in ("stop", "stop_limit", "limit", "trailing_stop"):
            continue
        if o.get("status") not in VALID_ORDER_STATUSES:
            continue

        symbol = o.get("symbol")

        if symbol in open_tickers:
            report["skipped"].append({
                "ticker": symbol,
                "order_id": o.get("id"),
                "reason": "Posizione ancora aperta: ordine legittimo",
            })
            continue

        action = {
            "ticker": symbol,
            "order_id": o.get("id"),
            "type": o.get("type"),
            "stop_price": o.get("stop_price"),
            "limit_price": o.get("limit_price"),
            "reason": "Nessuna posizione aperta per questo ticker",
        }

        if dry_run:
            action["status"] = "WOULD_CANCEL"
        else:
            try:
                result = await cancel_order(o.get("id"))
                action["status"] = "CANCELLED" if result is not None else "FAILED"
                if result is None:
                    report["errors"].append({
                        "ticker": symbol,
                        "order_id": o.get("id"),
                        "error": "cancel_order returned None",
                    })
            except Exception as e:
                action["status"] = "EXCEPTION"
                report["errors"].append({"ticker": symbol, "error": str(e)})

        report["cancelled"].append(action)

    report["summary"] = {
        "orphans_found": len(report["cancelled"]),
        "skipped_active": len(report["skipped"]),
        "errors": len(report["errors"]),
    }
    return report


@router.post("/close-position/{ticker}")
async def close_position_endpoint(ticker: str, dry_run: bool = Query(default=True)):
    """
    Chiude una posizione a mercato.
    1. Cancella gli ordini sell aperti sul ticker
    2. close_position su Alpaca
    3. Marca il BUY come sell_linked nel DB
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
        report["errors"].append(f"Nessuna posizione aperta per {ticker}")
        return report

    qty = _qty_of(target_pos)
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


@router.post("/populate-fractionable")
async def populate_fractionable():
    """
    One-shot: salva il flag fractionable su ogni asset del DB.
    """
    from app.services.alpaca_trader import is_fractionable

    db = get_db()
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
        except Exception as e:
            report["errors"].append({"ticker": ticker, "error": str(e)})

    report["finished_at"] = datetime.utcnow().isoformat()
    report["summary"] = {
        "fractionable_count": len(report["fractionable"]),
        "not_fractionable_count": len(report["not_fractionable"]),
        "errors_count": len(report["errors"]),
    }
    return report


@router.post("/fix-position-targets")
async def fix_position_targets():
    """
    Ricalcola stop_loss e target delle posizioni aperte sul filled price reale.
    Usa da eseguire quando SL/TP nel DB sono incoerenti con l'entry effettiva.
    """
    db = get_db()
    positions = await get_positions() or []

    if not positions:
        return {"message": "No open positions", "fixed": 0}

    report = {
        "started_at": datetime.utcnow().isoformat(),
        "checked": 0,
        "fixed": [],
        "skipped": [],
        "errors": [],
    }

    for pos in positions:
        symbol = pos.get("symbol")
        entry_price = float(pos.get("avg_entry_price", 0))

        if not symbol or entry_price <= 0:
            report["errors"].append({"ticker": symbol, "error": "Invalid position data"})
            continue

        report["checked"] += 1

        buy_trade = await db.trade_history.find_one(
            {"ticker": symbol, "side": "buy", "sell_linked": {"$ne": True}},
            sort=[("date", -1)]
        )

        if not buy_trade:
            report["skipped"].append({"ticker": symbol, "reason": "No buy_trade found"})
            continue

        old_stop = float(buy_trade.get("stop_loss", 0) or 0)
        old_target = float(buy_trade.get("target", 0) or 0)

        needs_fix = False
        new_stop = old_stop
        new_target = old_target

        if old_stop <= 0:
            new_stop = round(entry_price * 0.96, 2)
            needs_fix = True
        elif old_stop >= entry_price:
            new_stop = round(entry_price * 0.96, 2)
            needs_fix = True
        elif old_stop < entry_price * 0.85:
            new_stop = round(entry_price * 0.96, 2)
            needs_fix = True
        elif old_stop > entry_price * 0.99:
            new_stop = round(entry_price * 0.96, 2)
            needs_fix = True

        if old_target <= 0:
            new_target = round(entry_price * 1.08, 2)
            needs_fix = True
        elif old_target <= entry_price:
            new_target = round(entry_price * 1.08, 2)
            needs_fix = True
        elif old_target < entry_price * 1.02:
            new_target = round(entry_price * 1.08, 2)
            needs_fix = True

        if not needs_fix:
            report["skipped"].append({
                "ticker": symbol,
                "reason": "SL/TP already correct",
                "entry": entry_price,
                "stop_loss": old_stop,
                "target": old_target,
            })
            continue

        try:
            await db.trade_history.update_one(
                {"_id": buy_trade["_id"]},
                {"$set": {
                    "stop_loss": new_stop,
                    "target": new_target,
                    "original_stop_loss": old_stop,
                    "original_target": old_target,
                    "target_fixed_at": datetime.utcnow(),
                    "target_fix_reason": "post_fill_recalc",
                }}
            )
            report["fixed"].append({
                "ticker": symbol,
                "entry_price": entry_price,
                "old_stop": old_stop,
                "new_stop": new_stop,
                "old_target": old_target,
                "new_target": new_target,
            })
        except Exception as e:
            report["errors"].append({"ticker": symbol, "error": str(e)})

    report["finished_at"] = datetime.utcnow().isoformat()
    report["summary"] = {
        "checked": report["checked"],
        "fixed_count": len(report["fixed"]),
        "skipped_count": len(report["skipped"]),
        "errors_count": len(report["errors"]),
    }
    return report


@router.get("/positions-detail")
async def positions_detail():
    """
    Posizioni aperte arricchite con SL/TP, setup e giorni di holding dal DB.
    Usata dal frontend (PositionsUnified).
    """
    db = get_db()
    positions = await get_positions() or []

    if not positions:
        return {"positions": [], "count": 0}

    detailed = []

    for pos in positions:
        symbol = pos.get("symbol")
        current_price = float(pos.get("current_price", 0))
        entry_price = float(pos.get("avg_entry_price", 0))
        qty = _qty_of(pos)
        market_value = float(pos.get("market_value", 0))
        pnl = float(pos.get("unrealized_pl", 0))
        pnl_pct = float(pos.get("unrealized_plpc", 0)) * 100

        buy_trade = await db.trade_history.find_one(
            {"ticker": symbol, "side": "buy", "sell_linked": {"$ne": True}},
            sort=[("date", -1)]
        )

        stop_loss = 0
        target = 0
        setup_type = "unknown"
        sector = "unknown"
        confluence = 0
        days_held = 0
        buy_date = None

        if buy_trade:
            stop_loss = float(buy_trade.get("stop_loss", 0) or 0)
            target = float(buy_trade.get("target", 0) or 0)
            setup_type = buy_trade.get("setup_type", "unknown")
            sector = buy_trade.get("sector", "unknown")
            confluence = buy_trade.get("confluence", 0)
            buy_date = buy_trade.get("date")
            if buy_date:
                days_held = max(1, (datetime.utcnow() - buy_date).days)

        trailing = await db.trailing_stops.find_one({"ticker": symbol})
        trailing_stop = float(trailing.get("stop_price", 0) or 0) if trailing else None

        effective_stop = stop_loss
        if trailing_stop and trailing_stop > stop_loss:
            effective_stop = trailing_stop

        stop_distance_pct = 0
        target_distance_pct = 0
        if current_price > 0:
            stop_distance_pct = round(((effective_stop - current_price) / current_price) * 100, 2) if effective_stop > 0 else 0
            target_distance_pct = round(((target - current_price) / current_price) * 100, 2) if target > 0 else 0

        risk = abs(current_price - effective_stop) if effective_stop > 0 else 0
        reward = abs(target - current_price) if target > 0 else 0
        rr = round(reward / risk, 2) if risk > 0 else 0

        detailed.append({
            "ticker": symbol,
            "qty": qty,
            "entry_price": round(entry_price, 2),
            "current_price": round(current_price, 2),
            "market_value": round(market_value, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "stop_loss": round(effective_stop, 2),
            "stop_loss_initial": round(stop_loss, 2),
            "trailing_stop": round(trailing_stop, 2) if trailing_stop else None,
            "apm_managed": bool(trailing and trailing.get("apm_managed")),
            "target": round(target, 2),
            "stop_distance_pct": stop_distance_pct,
            "target_distance_pct": target_distance_pct,
            "risk_reward": rr,
            "setup_type": setup_type,
            "sector": sector,
            "confluence": confluence,
            "days_held": days_held,
            "buy_date": buy_date.isoformat() if buy_date else None,
        })

    return {
        "positions": detailed,
        "count": len(detailed),
        "calculated_at": datetime.utcnow().isoformat(),
    }


@router.post("/fix-orphaned-positions")
async def fix_orphaned_positions():
    """
    Ripara le posizioni orfane: presenti su Alpaca ma senza un BUY attivo nel DB
    (tipicamente per un sell_linked applicato per errore, o per un buy manuale).

    v2 — confronto qty con tolleranza 5%, frazionarie gestite correttamente.
    """
    db = get_db()
    positions = await get_positions() or []

    if not positions:
        return {"message": "No open positions", "fixed": 0}

    report = {
        "started_at": datetime.utcnow().isoformat(),
        "checked": 0,
        "fixed_buys": [],
        "created_buys": [],
        "already_ok": [],
        "errors": [],
    }

    for pos in positions:
        symbol = pos.get("symbol")
        qty = _qty_of(pos)
        entry_price = float(pos.get("avg_entry_price", 0))

        if not symbol or qty < MIN_QTY:
            continue

        report["checked"] += 1

        active_buy = await db.trade_history.find_one(
            {"ticker": symbol, "side": "buy", "sell_linked": {"$ne": True}},
            sort=[("date", -1)]
        )

        if active_buy:
            sl = float(active_buy.get("stop_loss", 0) or 0)
            tp = float(active_buy.get("target", 0) or 0)
            if sl > 0 and tp > 0:
                report["already_ok"].append({
                    "ticker": symbol,
                    "stop_loss": sl,
                    "target": tp,
                })
                continue

        buy_trade = await db.trade_history.find_one(
            {"ticker": symbol, "side": "buy"},
            sort=[("date", -1)]
        )

        if not buy_trade:
            # Nessun BUY: posizione creata fuori dal sistema (buy manuale).
            # Crea un record minimo con SL/TP di sicurezza.
            new_stop = round(entry_price * 0.96, 2)
            new_target = round(entry_price * 1.08, 2)
            await db.trade_history.insert_one({
                "ticker": symbol,
                "side": "buy",
                "sizing_mode": "notional",
                "entry_price": entry_price,
                "shares": qty,
                "stop_loss": new_stop,
                "target": new_target,
                "confluence": 0,
                "setup_type": "manual",
                "sector": "unknown",
                "market_regime": "UNKNOWN",
                "agent": "orphan_recovery",
                "order_id": f"orphan_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "date": datetime.utcnow(),
                "sell_linked": False,
                "recovered": True,
            })
            report["created_buys"].append({
                "ticker": symbol,
                "entry": entry_price,
                "qty": qty,
                "stop_loss": new_stop,
                "target": new_target,
                "note": "BUY ricostruito: posizione creata fuori dal sistema",
            })
            continue

        buy_shares = float(buy_trade.get("shares", 0) or 0)
        shares_diff_pct = abs(buy_shares - qty) / qty if qty > 0 else 1.0

        if shares_diff_pct > 0.05:
            report["errors"].append({
                "ticker": symbol,
                "error": f"Buy shares {buy_shares:.4f} != posizione {qty:.4f}",
                "diff_pct": round(shares_diff_pct * 100, 1),
                "action": "MANUAL_REVIEW_REQUIRED",
            })
            continue

        await db.trade_history.update_one(
            {"_id": buy_trade["_id"]},
            {"$unset": {"sell_linked": "", "sell_order_id": ""}}
        )

        old_sl = float(buy_trade.get("stop_loss", 0) or 0)
        old_tp = float(buy_trade.get("target", 0) or 0)

        if old_sl <= 0 or old_tp <= 0:
            new_stop = round(entry_price * 0.96, 2)
            new_target = round(entry_price * 1.08, 2)
            await db.trade_history.update_one(
                {"_id": buy_trade["_id"]},
                {"$set": {
                    "stop_loss": new_stop,
                    "target": new_target,
                    "recalculated_at": datetime.utcnow(),
                    "recalc_reason": "orphan_position_fix",
                }}
            )
            report["fixed_buys"].append({
                "ticker": symbol,
                "action": "unlinked + recalc SL/TP",
                "entry": entry_price,
                "new_stop": new_stop,
                "new_target": new_target,
            })
        else:
            report["fixed_buys"].append({
                "ticker": symbol,
                "action": "unlinked only",
                "stop_loss": old_sl,
                "target": old_tp,
            })

    report["finished_at"] = datetime.utcnow().isoformat()
    report["summary"] = {
        "positions_checked": report["checked"],
        "fixed_count": len(report["fixed_buys"]),
        "created_count": len(report["created_buys"]),
        "already_ok": len(report["already_ok"]),
        "errors": len(report["errors"]),
    }
    return report

import asyncio
from datetime import datetime

import pandas as pd

from app.db.mongodb import get_db
from app.services.max_strategy import analyze_max_strategy

SCAN_VERSION = "max_structure_v1_5_2"
TRACKED_STATUSES = ("ARMED", "TRIGGERED", "CONFIRMED_4H", "CONFIRMED_DAILY", "WAIT_RETEST")
REFERENCE_TICKERS = {"SPY", "XLK", "XLF", "XLV", "XLI", "XLY", "XLP", "XLE", "XLU", "XLB", "XLRE", "XLC"}
MIN_HISTORY_BARS = 160
PAUSE_SECONDS = 0.05
BUSY_WAIT_SECONDS = 20

_SCAN_STATE = {"running": False, "stop": False}


def _num(value):
    try:
        number = float(value)
        return number if number == number else None
    except (TypeError, ValueError):
        return None


def _bars_to_df(bars):
    df = pd.DataFrame(bars)
    df = df.rename(columns={"date": "datetime", "o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).reset_index(drop=True)


def _weekly_checkpoints(df, scan_days):
    dates = df["datetime"]
    start_index = max(MIN_HISTORY_BARS - 1, len(df) - scan_days)
    checkpoints = []
    for index in range(start_index, len(df)):
        is_last = index == len(df) - 1
        next_week = None if is_last else dates.iloc[index + 1].isocalendar()[1]
        if is_last or next_week != dates.iloc[index].isocalendar()[1]:
            checkpoints.append(index)
    return checkpoints


def _qualified(result, plan, price):
    trigger = _num(plan.get("trigger_price"))
    maximum_entry = _num(plan.get("maximum_entry_price"))
    invalidation = _num(plan.get("invalidation_price"))
    levels_valid = bool(
        price and trigger and maximum_entry
        and trigger > 0
        and maximum_entry > trigger
        and (invalidation is None or 0 < invalidation < trigger)
    )
    return bool(
        result.get("data_eligible")
        and result.get("strategy_eligible")
        and plan.get("weekly_plan_qualified")
        and not plan.get("blocking_phase")
        and levels_valid
        and not (invalidation and price <= invalidation)
    )


def _compact_signal(ticker, date_text, result):
    plan = result.get("entry_plan") or {}
    status = plan.get("status")
    if status not in TRACKED_STATUSES:
        return None
    price = _num(result.get("price"))
    qualified = _qualified(result, plan, price)
    if not qualified:
        return None
    return {
        "_id": f"{SCAN_VERSION}:{ticker}:{date_text}",
        "version": SCAN_VERSION,
        "ticker": ticker,
        "date": date_text,
        "status": status,
        "order_action": plan.get("order_action"),
        "trade_ready": bool(result.get("trade_ready")),
        "qualified": True,
        "price": price,
        "trigger_price": _num(plan.get("trigger_price")),
        "maximum_entry_price": _num(plan.get("maximum_entry_price")),
        "invalidation_price": _num(plan.get("invalidation_price")),
        "atr14": _num(result.get("atr14")),
        "trigger_source": plan.get("trigger_source"),
        "strategy_type": result.get("strategy_type"),
        "phase": (result.get("market_phase") or {}).get("phase"),
        "max_score": result.get("max_score"),
    }


async def _wait_if_busy(is_busy):
    waited = 0
    while is_busy and is_busy():
        if _SCAN_STATE["stop"]:
            return
        await asyncio.sleep(BUSY_WAIT_SECONDS)
        waited += BUSY_WAIT_SECONDS
    return waited


async def run_max_history_scan(scan_days=760, restart=False, is_busy=None):
    db = get_db()
    job_id = f"max_scan_{SCAN_VERSION}"
    if _SCAN_STATE["running"]:
        return
    _SCAN_STATE["running"] = True
    _SCAN_STATE["stop"] = False
    try:
        tickers = sorted({
            doc.get("ticker") async for doc in db.assets.find({}, {"ticker": 1})
            if doc.get("ticker") and doc.get("ticker") not in REFERENCE_TICKERS
        })
        job = await db.max_scan_jobs.find_one({"_id": job_id}) or {}
        done = set() if restart else set(job.get("done_tickers") or [])
        if restart:
            await db.max_backtest_signals.delete_many({"version": SCAN_VERSION})
        await db.max_scan_jobs.update_one(
            {"_id": job_id},
            {"$set": {
                "status": "running",
                "version": SCAN_VERSION,
                "scan_days": scan_days,
                "tickers_total": len(tickers),
                "done_tickers": sorted(done),
                "started_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "error": None,
            }},
            upsert=True,
        )
        analyses = int(job.get("analyses", 0)) if not restart else 0
        signals_saved = int(job.get("signals_saved", 0)) if not restart else 0
        for ticker in tickers:
            if _SCAN_STATE["stop"]:
                break
            if ticker in done:
                continue
            await _wait_if_busy(is_busy)
            doc = await db.stock_bars.find_one({"ticker": ticker}, {"bars": 1})
            bars = (doc or {}).get("bars") or []
            ticker_signals = []
            if len(bars) >= MIN_HISTORY_BARS:
                df = _bars_to_df(bars)
                del bars, doc
                for index in _weekly_checkpoints(df, scan_days):
                    if _SCAN_STATE["stop"]:
                        break
                    if is_busy and is_busy():
                        await _wait_if_busy(is_busy)
                    window = df.iloc[:index + 1]
                    as_of = window["datetime"].iloc[-1]
                    try:
                        result = await asyncio.to_thread(analyze_max_strategy, window, None, as_of)
                    except Exception as e:
                        print(f"[MAX SCAN] {ticker} {as_of.date()} error: {e}")
                        result = None
                    analyses += 1
                    if result:
                        signal = _compact_signal(ticker, as_of.strftime("%Y-%m-%d"), result)
                        if signal:
                            ticker_signals.append(signal)
                    await asyncio.sleep(PAUSE_SECONDS)
                del df
            if _SCAN_STATE["stop"]:
                break
            for signal in ticker_signals:
                await db.max_backtest_signals.replace_one({"_id": signal["_id"]}, signal, upsert=True)
            signals_saved += len(ticker_signals)
            done.add(ticker)
            await db.max_scan_jobs.update_one(
                {"_id": job_id},
                {"$set": {
                    "done_tickers": sorted(done),
                    "tickers_done": len(done),
                    "analyses": analyses,
                    "signals_saved": signals_saved,
                    "last_ticker": ticker,
                    "updated_at": datetime.utcnow(),
                }},
            )
            print(f"[MAX SCAN] {ticker}: {len(ticker_signals)} segnali ({len(done)}/{len(tickers)})")
        final_status = "stopped" if _SCAN_STATE["stop"] else "done"
        await db.max_scan_jobs.update_one(
            {"_id": job_id},
            {"$set": {"status": final_status, "finished_at": datetime.utcnow(), "updated_at": datetime.utcnow()}},
        )
        print(f"[MAX SCAN] {final_status}: {len(done)}/{len(tickers)} ticker, {signals_saved} segnali")
    except Exception as e:
        await db.max_scan_jobs.update_one(
            {"_id": job_id},
            {"$set": {"status": "error", "error": str(e), "updated_at": datetime.utcnow()}},
            upsert=True,
        )
        print(f"[MAX SCAN] error: {e}")
    finally:
        _SCAN_STATE["running"] = False
        _SCAN_STATE["stop"] = False


def request_stop():
    if _SCAN_STATE["running"]:
        _SCAN_STATE["stop"] = True
        return True
    return False


def is_running():
    return _SCAN_STATE["running"]


async def get_scan_status():
    db = get_db()
    job_id = f"max_scan_{SCAN_VERSION}"
    job = await db.max_scan_jobs.find_one({"_id": job_id}, {"done_tickers": 0}) or {"_id": job_id, "status": "not_started"}
    job["_id"] = str(job["_id"])
    job["running_now"] = _SCAN_STATE["running"]
    job["signals_in_db"] = await db.max_backtest_signals.count_documents({"version": SCAN_VERSION})
    for key in ("started_at", "updated_at", "finished_at"):
        if job.get(key) and hasattr(job[key], "isoformat"):
            job[key] = job[key].isoformat()
    total = job.get("tickers_total") or 0
    job["progress_pct"] = round((job.get("tickers_done") or 0) / total * 100, 1) if total else 0
    return job

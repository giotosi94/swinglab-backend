from fastapi import APIRouter, Query

from app.db.mongodb import get_db

router = APIRouter()

SECTOR_CODES = ["XLK", "XLF", "XLV", "XLI", "XLY", "XLP", "XLE", "XLU", "XLB", "XLRE", "XLC"]

# Margine sulle barre richieste: serve un po' di storico in piu' della
# finestra selezionata per gestire festivita' e disallineamenti fra ticker.
BARS_BUFFER = 12


# L'analisi gira SEMPRE, in qualunque momento della giornata. Non esiste un
# orario che la spegne: cambia solo la qualita' dichiarata del dato.
#
# Il criterio non e' l'orologio ma il TIMESTAMP dell'ultimo scambio. Se un
# ETF non ha ancora scambiato oggi lo dichiariamo, invece di inventare un
# movimento dello zero per cento che sembrerebbe un dato reale.


async def _live_quotes(symbols):
    """
    Prezzi correnti per SPY e i settori, con informazione sulla freschezza.

    get_live_prices restituisce anche traded_today e age_minutes: servono a
    capire quali settori stanno davvero scambiando in questo momento.
    """
    try:
        from app.services.alpaca_trader import get_live_prices, market_session_now

        quotes = await get_live_prices(list(symbols))
        return quotes or {}, market_session_now()
    except Exception as e:
        print(f"  Sector live quotes non disponibili: {e}")
        return {}, {"session": "UNKNOWN", "today_et": None, "is_regular": False}


@router.get("/")
async def get_sectors(sort_by: str = "composite_score"):
    db = get_db()
    sectors = await db.sectors.find().to_list(100)

    for sector in sectors:
        sector["_id"] = str(sector["_id"])

    sectors.sort(key=lambda item: item.get(sort_by, 0), reverse=True)
    return sectors


@router.get("/relative-strength")
async def get_sector_relative_strength(
    days: int = Query(60, ge=5, le=750),
    include_today: bool = Query(True),
):
    """
    Forza relativa dei settori contro SPY, indicizzata a 100.

    Quando i prezzi correnti sono disponibili viene aggiunto un punto
    PROIETTATO per la seduta in corso. E' marcato come provvisorio perche'
    cambia fino alla chiusura, e viene disegnato solo per i settori che
    hanno effettivamente scambiato oggi.
    """
    db = get_db()
    requested = ["SPY"] + SECTOR_CODES

    # Solo le barre necessarie: questa route caricava l'intero storico di
    # 12 ticker anche per un grafico a 21 sedute.
    bars_needed = days + BARS_BUFFER

    docs = await db.stock_bars.find(
        {"ticker": {"$in": requested}},
        {"ticker": 1, "bars": {"$slice": -bars_needed}},
    ).to_list(20)

    bars_by_ticker = {doc.get("ticker"): doc.get("bars", []) for doc in docs}
    spy_bars = bars_by_ticker.get("SPY", [])

    if not spy_bars:
        return {"error": "SPY history unavailable"}

    spy_by_date = {
        bar.get("date"): float(bar.get("c", 0) or 0)
        for bar in spy_bars
        if bar.get("date") and bar.get("c")
    }

    common_dates = sorted(spy_by_date.keys())[-days:]

    if not common_dates:
        return {"error": "Nessuna seduta disponibile"}

    last_close_date = common_dates[-1]

    # ---- Punto proiettato: sempre tentato, mai forzato ----
    #
    # Nessun gate orario. Proviamo sempre a leggere i prezzi correnti: se
    # SPY ha scambiato oggi il punto ha senso, altrimenti no. Lo decide il
    # dato, non l'orologio.
    live_quotes = {}
    session = {"session": "UNKNOWN", "today_et": None, "is_regular": False}
    spy_live = None
    spy_traded_today = False

    if include_today:
        live_quotes, session = await _live_quotes(requested)

        spy_quote = live_quotes.get("SPY") or {}
        spy_live = spy_quote.get("price")
        spy_traded_today = bool(spy_quote.get("traded_today"))

        if spy_live:
            spy_live = float(spy_live)

    today_et = session.get("today_et")

    projection_enabled = bool(
        include_today
        and spy_live
        and spy_traded_today
        and today_et
        and today_et != last_close_date
    )

    # Contatori di qualita': quanti settori stanno davvero scambiando.
    sectors_traded_today = 0
    sectors_no_trade = []

    series = []

    sector_docs = await db.sectors.find({"code": {"$in": SECTOR_CODES}}).to_list(20)
    sector_names = {doc.get("code"): doc.get("name", doc.get("code")) for doc in sector_docs}

    for code in SECTOR_CODES:
        sector_by_date = {
            bar.get("date"): float(bar.get("c", 0) or 0)
            for bar in bars_by_ticker.get(code, [])
            if bar.get("date") and bar.get("c")
        }

        dates = [
            date for date in common_dates
            if date in sector_by_date and spy_by_date.get(date, 0) > 0
        ]

        if len(dates) < 2:
            continue

        first_ratio = sector_by_date[dates[0]] / spy_by_date[dates[0]]

        if first_ratio <= 0:
            continue

        points = []
        for date in dates:
            ratio = sector_by_date[date] / spy_by_date[date]
            points.append({
                "date": date,
                "value": round(ratio / first_ratio * 100, 3),
                "projected": False,
            })

        last_close_value = points[-1]["value"]

        # ---- Proiezione di oggi, settore per settore ----
        projected_value = None
        intraday_move = None

        sector_quote = live_quotes.get(code) or {}
        sector_traded = bool(sector_quote.get("traded_today"))
        sector_live_price = sector_quote.get("price")

        if include_today and sector_quote:
            if sector_traded:
                sectors_traded_today += 1
            else:
                sectors_no_trade.append(code)

        # Il punto viene disegnato solo se QUESTO settore ha scambiato oggi.
        # Un ETF fermo non deve produrre una linea piatta che sembra un dato.
        if projection_enabled and sector_traded and sector_live_price and spy_live:
            sector_live = float(sector_live_price)
            live_ratio = sector_live / spy_live
            projected_value = round(live_ratio / first_ratio * 100, 3)

            previous_close = sector_by_date.get(dates[-1], 0)
            if previous_close > 0:
                intraday_move = round(
                    (sector_live - previous_close) / previous_close * 100, 2
                )

            points.append({
                "date": today_et,
                "value": projected_value,
                "projected": True,
            })

        # L'accelerazione resta calcolata sulle sole barre chiuse: un punto
        # provvisorio non deve alterare una metrica di tendenza.
        closed_points = [p for p in points if not p["projected"]]
        lookback_index = max(0, len(closed_points) - 21)
        acceleration = closed_points[-1]["value"] - closed_points[lookback_index]["value"]

        series.append({
            "code": code,
            "name": sector_names.get(code, code),
            "points": points,
            "relative_return_pct": round(last_close_value - 100, 2),
            "projected_return_pct": round(projected_value - 100, 2) if projected_value else None,
            "intraday_move_pct": intraday_move,
            "acceleration_20d": round(acceleration, 2),
            "last_value": last_close_value,
            "projected_value": projected_value,
            "has_projection": projected_value is not None,
            "traded_today": sector_traded,
            "live_price": round(float(sector_live_price), 2) if sector_live_price else None,
            "quote_age_minutes": sector_quote.get("age_minutes"),
        })

    # L'ordinamento usa la chiusura: la classifica non deve ballare durante
    # la seduta per movimenti di pochi decimi.
    series.sort(key=lambda item: item["relative_return_pct"], reverse=True)

    assets = await db.assets.find(
        {"sector_code": {"$in": SECTOR_CODES}},
        {
            "ticker": 1, "sector_code": 1, "setup_score": 1, "setup_type": 1,
            "confluence": 1, "rsi": 1, "price": 1, "mtf": 1, "poc_shift": 1,
            "alpha_snapshot": 1,
        },
    ).to_list(400)

    best_stocks = {}

    for code in SECTOR_CODES:
        candidates = [asset for asset in assets if asset.get("sector_code") == code]
        candidates.sort(
            key=lambda asset: float(
                (asset.get("alpha_snapshot") or {}).get(
                    "confluence",
                    asset.get("confluence", asset.get("setup_score", 0)),
                ) or 0
            ),
            reverse=True,
        )

        best_stocks[code] = []

        for asset in candidates[:5]:
            alpha = asset.get("alpha_snapshot") or {}
            final_confluence = float(
                alpha.get("confluence", asset.get("confluence", asset.get("setup_score", 0))) or 0
            )
            threshold = float(alpha.get("min_confluence", 48) or 48)
            status = alpha.get("status") or (
                "CANDIDATE" if final_confluence >= threshold else "NO_ALPHA_SNAPSHOT"
            )

            best_stocks[code].append({
                "ticker": asset.get("ticker"),
                "score": round(float(asset.get("setup_score", 0) or 0), 1),
                "alpha_confluence": round(final_confluence, 1),
                "confluence_before_sector": round(
                    float(alpha.get("confluence_before_sector", final_confluence) or 0), 1
                ),
                "sector_adjustment": round(float(alpha.get("sector_adjustment", 0) or 0), 1),
                "sector_rank": alpha.get("sector_rank"),
                "sector_reason": alpha.get(
                    "sector_intelligence_reason", "Snapshot Alpha non ancora disponibile"
                ),
                "risk_reward": round(float(alpha.get("risk_reward", 0) or 0), 2),
                "status": status,
                "threshold": threshold,
                "setup_type": alpha.get("setup_type", asset.get("setup_type", "neutral")),
                "rsi": round(float(alpha.get("rsi", asset.get("rsi", 0)) or 0), 1),
                "price": round(float(asset.get("price", 0) or 0), 2),
                "weekly_trend": alpha.get(
                    "weekly_trend", (asset.get("mtf") or {}).get("weekly_trend", "UNKNOWN")
                ),
                "poc_shift": bool((asset.get("poc_shift") or {}).get("shifted_bull", False)),
                "alpha_updated_at": alpha.get("updated_at"),
            })

    projected_ranking = sorted(
        [item for item in series if item["has_projection"]],
        key=lambda item: item["projected_return_pct"],
        reverse=True,
    )

    return {
        "benchmark": "SPY",
        "requested_days": days,
        "executed_days": len(common_dates),
        "start_date": common_dates[0],
        "end_date": last_close_date,
        "baseline": 100,

        # ---- Qualita' del dato corrente ----
        #
        # L'analisi e' sempre attiva: questi campi dicono al frontend quanto
        # fidarsi del punto di oggi, non se mostrarlo o meno.
        "session": session.get("session"),
        "market_open": session.get("is_regular", False),
        "today": today_et,
        "last_close_date": last_close_date,

        "projection_available": projection_enabled and any(i["has_projection"] for i in series),
        "spy_traded_today": spy_traded_today,

        "sectors_traded_today": sectors_traded_today,
        "sectors_total": len(series),
        "sectors_no_trade": sectors_no_trade,

        "data_quality": (
            "FULL" if session.get("is_regular") and sectors_traded_today >= len(series) * 0.8
            else "PARTIAL" if sectors_traded_today > 0
            else "CLOSED_BARS_ONLY"
        ),

        "spy_intraday_move_pct": (
            round((spy_live - spy_by_date[last_close_date]) / spy_by_date[last_close_date] * 100, 2)
            if projection_enabled and spy_live and spy_by_date.get(last_close_date)
            else None
        ),

        "series": series,
        "ranking": [
            {
                "rank": index + 1,
                "code": item["code"],
                "name": item["name"],
                "relative_return_pct": item["relative_return_pct"],
                "projected_return_pct": item["projected_return_pct"],
                "intraday_move_pct": item["intraday_move_pct"],
                "acceleration_20d": item["acceleration_20d"],
                "traded_today": item["traded_today"],
            }
            for index, item in enumerate(series)
        ],
        "projected_leaders": [item["code"] for item in projected_ranking[:3]],
        "best_stocks": best_stocks,
    }




def _sma(values, end_index, period):
    start = end_index - period + 1
    if start < 0:
        return None
    window = values[start:end_index + 1]
    if len(window) != period:
        return None
    return sum(window) / period


def _percentile_rank(history, value):
    valid = [item for item in history if item is not None]
    if not valid:
        return 50.0
    return sum(1 for item in valid if item <= value) / len(valid) * 100


def _change(points, key, lookback):
    if len(points) <= lookback:
        return 0.0
    current = points[-1].get(key)
    previous = points[-1 - lookback].get(key)
    if current is None or previous is None:
        return 0.0
    return current - previous


def _bottom_classification(points):
    if not points:
        return {
            "state": "NO_DATA",
            "score": 0,
            "reason": "Storico insufficiente",
        }
    latest = points[-1]
    b20 = float(latest.get("above_sma20_pct", 0) or 0)
    b50 = float(latest.get("above_sma50_pct", 0) or 0)
    b200 = float(latest.get("above_sma200_pct", 0) or 0)
    new_low20 = float(latest.get("new_low20_pct", 0) or 0)
    b20_change_5d = _change(points, "above_sma20_pct", 5)
    b50_change_10d = _change(points, "above_sma50_pct", 10)
    b200_change_20d = _change(points, "above_sma200_pct", 20)
    new_low_change_5d = _change(points, "new_low20_pct", 5)
    percentile = _percentile_rank(
        [point.get("above_sma200_pct") for point in points[-252:]],
        b200,
    )
    washout_score = max(0.0, min(30.0, (35.0 - percentile) / 35.0 * 30.0))
    recovery_score = 0.0
    recovery_score += max(0.0, min(10.0, b20_change_5d * 0.8))
    recovery_score += max(0.0, min(10.0, b50_change_10d * 0.6))
    recovery_score += max(0.0, min(5.0, b200_change_20d * 0.5))
    recovery_score += 5.0 if b20 > b50 > b200 else 0.0
    stabilization_score = 0.0
    stabilization_score += max(0.0, min(12.0, -new_low_change_5d * 0.8))
    stabilization_score += 8.0 if new_low20 <= 15 else 4.0 if new_low20 <= 25 else 0.0
    participation_score = 0.0
    participation_score += 8.0 if b20 >= 50 else 4.0 if b20 >= 35 else 0.0
    participation_score += 6.0 if b50_change_10d > 0 else 0.0
    participation_score += 6.0 if b200_change_20d > 0 else 0.0
    score = round(min(100.0, washout_score + recovery_score + stabilization_score + participation_score), 1)
    washout = b200 <= 20 or (b200 <= 40 and percentile <= 20)
    recovering = b20_change_5d >= 8 and b50_change_10d > 0 and new_low_change_5d < 0
    broad_recovery = b20 >= 45 and b20 > b50 and b50_change_10d >= 8
    confirmed = washout and b50 >= 50 and b200_change_20d > 0 and new_low20 <= 10
    correction_recovery = bool(
        not washout
        and b20_change_5d >= 15
        and b50_change_10d > 0
        and new_low_change_5d < 0
    )
    if confirmed and score >= 65:
        state = "RECLAIM_CONFERMATO"
        reason = "Dopo un washout, la breadth di medio e lungo periodo torna a espandersi con pochi nuovi minimi."
    elif washout and broad_recovery and score >= 55:
        state = "RECUPERO_DIFFUSO"
        reason = "Il recupero successivo al washout coinvolge una quota crescente dei componenti."
    elif washout and recovering and score >= 40:
        state = "BOTTOM_IN_FORMAZIONE"
        reason = "Breadth assoluta e storica depresse, con recupero iniziale di SMA20/SMA50 e minori nuovi minimi."
    elif correction_recovery:
        state = "RECUPERO_DA_CORREZIONE"
        reason = "La breadth di breve periodo rimbalza, ma la partecipazione SMA200 non indica un washout strutturale."
    elif washout:
        state = "WASHOUT"
        reason = "Partecipazione assoluta e storica estremamente debole, senza inversione confermata."
    else:
        state = "NESSUN_BOTTOM"
        reason = "Non risultano contemporaneamente washout assoluto e recupero diffuso della partecipazione."
    return {
        "state": state,
        "score": score,
        "reason": reason,
        "breadth_200_percentile_1y": round(percentile, 1),
        "above_sma20_pct": round(b20, 1),
        "above_sma50_pct": round(b50, 1),
        "above_sma200_pct": round(b200, 1),
        "new_low20_pct": round(new_low20, 1),
        "breadth20_change_5d": round(b20_change_5d, 1),
        "breadth50_change_10d": round(b50_change_10d, 1),
        "breadth200_change_20d": round(b200_change_20d, 1),
        "new_low20_change_5d": round(new_low_change_5d, 1),
        "components": {
            "washout": round(washout_score, 1),
            "recovery": round(recovery_score, 1),
            "stabilization": round(stabilization_score, 1),
            "participation": round(participation_score, 1),
        },
    }


@router.get("/breadth-bottom")
async def get_sector_breadth_bottom(days: int = Query(252, ge=21, le=252)):
    db = get_db()
    cached = await db.sector_bottom_breadth.find_one({"_id": "latest"})
    if cached:
        cached.pop("_id", None)
        return cached

    assets = await db.assets.find(
        {"sector_code": {"$in": SECTOR_CODES}, "data_eligible": {"$ne": False}},
        {"ticker": 1, "sector_code": 1},
    ).to_list(400)
    ticker_sector = {
        asset.get("ticker"): asset.get("sector_code")
        for asset in assets
        if asset.get("ticker") and asset.get("sector_code") in SECTOR_CODES
    }
    tickers = list(ticker_sector)
    docs = await db.stock_bars.find(
        {"ticker": {"$in": tickers}},
        {"ticker": 1, "bars": {"$slice": -472}},
    ).to_list(400)

    sector_dates = {code: {} for code in SECTOR_CODES}
    sector_members = {code: 0 for code in SECTOR_CODES}
    all_dates = set()

    for doc in docs:
        ticker = doc.get("ticker")
        code = ticker_sector.get(ticker)
        if not code:
            continue
        rows = []
        for bar in doc.get("bars", []):
            date = bar.get("date")
            close = float(bar.get("c", 0) or 0)
            low = float(bar.get("l", 0) or 0)
            if date and close > 0 and low > 0:
                rows.append((date, close, low))
        rows.sort(key=lambda item: item[0])
        if len(rows) < 200:
            continue
        sector_members[code] += 1
        closes = [row[1] for row in rows]
        lows = [row[2] for row in rows]
        prefix = [0.0]
        for close in closes:
            prefix.append(prefix[-1] + close)
        for index in range(199, len(rows)):
            date, close, low = rows[index]
            all_dates.add(date)
            metrics = sector_dates[code].setdefault(date, {
                "above20": 0, "eligible20": 0,
                "above50": 0, "eligible50": 0,
                "above200": 0, "eligible200": 0,
                "new_low20": 0, "new_low_eligible": 0,
                "distances200": [],
            })
            sma20 = (prefix[index + 1] - prefix[index - 19]) / 20
            sma50 = (prefix[index + 1] - prefix[index - 49]) / 50
            sma200 = (prefix[index + 1] - prefix[index - 199]) / 200
            metrics["eligible20"] += 1
            metrics["eligible50"] += 1
            metrics["eligible200"] += 1
            metrics["above20"] += int(close > sma20)
            metrics["above50"] += int(close > sma50)
            metrics["above200"] += int(close > sma200)
            metrics["distances200"].append((close - sma200) / sma200 * 100)
            metrics["new_low_eligible"] += 1
            metrics["new_low20"] += int(low <= min(lows[index - 19:index + 1]))

    requested_dates = sorted(all_dates)[-days:]
    sector_docs = await db.sectors.find(
        {"code": {"$in": SECTOR_CODES}}, {"code": 1, "name": 1}
    ).to_list(20)
    sector_names = {doc.get("code"): doc.get("name", doc.get("code")) for doc in sector_docs}
    series = []

    for code in SECTOR_CODES:
        points = []
        for date in requested_dates:
            metrics = sector_dates[code].get(date)
            if not metrics or not metrics["eligible200"]:
                continue
            distances = sorted(metrics["distances200"])
            middle = len(distances) // 2
            median = distances[middle] if len(distances) % 2 else (distances[middle - 1] + distances[middle]) / 2
            above200 = metrics["above200"] / metrics["eligible200"] * 100
            points.append({
                "date": date,
                "above_sma20_pct": round(metrics["above20"] / metrics["eligible20"] * 100, 1),
                "above_sma50_pct": round(metrics["above50"] / metrics["eligible50"] * 100, 1),
                "above_sma200_pct": round(above200, 1),
                "below_sma200_pct": round(100 - above200, 1),
                "median_distance_sma200_pct": round(median, 2),
                "new_low20_pct": round(metrics["new_low20"] / metrics["new_low_eligible"] * 100, 1),
                "eligible_20": metrics["eligible20"],
                "eligible_50": metrics["eligible50"],
                "eligible_200": metrics["eligible200"],
            })
        if points:
            series.append({
                "code": code,
                "name": sector_names.get(code, code),
                "members_total": sector_members[code],
                "points": points,
                "bottom": _bottom_classification(points),
            })

    series.sort(key=lambda item: item["bottom"]["score"], reverse=True)
    result = {
        "mode": "SECTOR_BOTTOM_BREADTH",
        "visual_only": True,
        "requested_days": days,
        "executed_days": len(requested_dates),
        "start_date": requested_dates[0] if requested_dates else None,
        "end_date": requested_dates[-1] if requested_dates else None,
        "series": series,
        "ranking": [
            {"rank": index + 1, "code": item["code"], "name": item["name"],
             "members_total": item["members_total"], **item["bottom"]}
            for index, item in enumerate(series)
        ],
        "methodology": {
            "title": "Sector Bottom Breadth Score",
            "summary": "Misura washout e recupero equal-weighted dei componenti del settore. Non e' un segnale operativo.",
            "breadth": "Percentuale di aziende sopra la propria SMA20, SMA50 e SMA200.",
            "washout": "Breadth SMA200 nel tratto piu' debole della propria distribuzione a un anno.",
            "recovery": "Aumento della breadth SMA20 in 5 sedute e SMA50 in 10 sedute.",
            "stabilization": "Riduzione della percentuale di aziende su nuovi minimi a 20 sedute.",
        },
        "cached": False,
    }
    await db.sector_bottom_breadth.replace_one(
        {"_id": "latest"}, {"_id": "latest", **result}, upsert=True
    )
    return result

@router.get("/{code}")
async def get_sector(code: str):
    db = get_db()
    sector = await db.sectors.find_one({"code": code.upper()})

    if sector:
        sector["_id"] = str(sector["_id"])

    return sector or {"error": "Sector not found"}

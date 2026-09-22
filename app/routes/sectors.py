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


@router.get("/{code}")
async def get_sector(code: str):
    db = get_db()
    sector = await db.sectors.find_one({"code": code.upper()})

    if sector:
        sector["_id"] = str(sector["_id"])

    return sector or {"error": "Sector not found"}

import httpx
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime
from app.db.mongodb import get_db
from app.config import settings
import traceback

SECTOR_MAP = {
    "XLK": "Technology", "XLF": "Financials", "XLV": "Health Care",
    "XLI": "Industrials", "XLY": "Consumer Discretionary", "XLP": "Consumer Staples",
    "XLE": "Energy", "XLU": "Utilities", "XLB": "Materials",
    "XLRE": "Real Estate", "XLC": "Communication Services",
}

SECTOR_STOCKS = {
    "XLK": ["AAPL","MSFT","NVDA","AVGO","AMD","CRM","ADBE","INTC","CSCO","ORCL",
            "PLTR","NOW","SNOW","CRWD","PANW","MNDY","SHOP","SQ","UBER","DDOG"],
    "XLF": ["JPM","BAC","WFC","GS","MS","BLK","SCHW","AXP","C","USB",
            "V","MA","PYPL","COF","ICE","SPGI","MCO","MMC","AON","TFC"],
    "XLV": ["UNH","JNJ","PFE","ABBV","MRK","TMO","ABT","LLY","BMY","AMGN",
            "ISRG","DXCM","VRTX","REGN","ZTS","HCA","CI","ELV","HUM","SYK"],
    "XLI": ["CAT","DE","UNP","HON","BA","RTX","LMT","GE","MMM","FDX",
            "UPS","WM","ETN","ITW","EMR","NSC","CSX","PCAR","ROK","IR"],
    "XLY": ["AMZN","TSLA","HD","MCD","NKE","SBUX","LOW","TJX","BKNG","CMG",
            "LULU","ROST","DHI","LEN","ABNB","DASH","EBAY","MAR","HLT","YUM"],
    "XLP": ["PG","KO","PEP","COST","WMT","PM","MO","CL","MDLZ","KHC",
            "STZ","SYY","HSY","GIS","ADM","MNST","KDP","CHD","CLX","SJM"],
    "XLE": ["XOM","CVX","COP","SLB","EOG","MPC","PSX","VLO","OXY","HAL",
            "DVN","FANG","WMB","KMI","TRGP","BKR","CTRA","MRO","APA","AR"],
    "XLU": ["NEE","DUK","SO","D","AEP","SRE","EXC","XEL","ED","WEC",
            "AWK","ES","ATO","CMS","PNW","PPL","FE","DTE","AES","ETR"],
    "XLB": ["LIN","APD","SHW","FCX","NEM","ECL","DOW","NUE","VMC","MLM",
            "CF","MOS","BALL","PKG","IFF","EMN","CE","RPM","SEE","AVY"],
    "XLRE": ["PLD","AMT","CCI","EQIX","SPG","PSA","O","WELL","DLR","AVB",
             "VICI","MAA","EXR","ARE","UDR","ESS","REG","HST","KIM","CPT"],
    "XLC": ["META","GOOGL","GOOG","NFLX","DIS","CMCSA","T","VZ","TMUS","EA",
            "SPOT","RBLX","TTWO","WBD","PARA","MTCH","ZM","PINS","SNAP","LYV"],
}

ALPACA_HEADERS = {
    "APCA-API-KEY-ID": settings.ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": settings.ALPACA_SECRET_KEY,
}

async def fetch_bars(client, symbol):
    url = "https://data.alpaca.markets/v2/stocks/{}/bars".format(symbol)
    params = {"timeframe": "1Day", "limit": 252, "feed": "iex"}
    try:
        r = await client.get(url, headers=ALPACA_HEADERS, params=params)
        if r.status_code != 200:
            print("  {}: HTTP {}".format(symbol, r.status_code))
            return None
        data = r.json()
        bars = data.get("bars", [])
        if not bars:
            print("  {}: no bars".format(symbol))
            return None
        df = pd.DataFrame(bars)
        df = df.rename(columns={"t": "datetime", "o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
        return df
    except Exception as e:
        print("  {} error: {}".format(symbol, e))
        return None

def detect_fvg(df):
    if df is None or len(df) < 3:
        return []
    fvgs = []
    high = df["High"].values
    low = df["Low"].values
    for i in range(2, len(df)):
        if low[i] > high[i-2]:
            fvg_top = float(low[i])
            fvg_bottom = float(high[i-2])
            fvg_size = fvg_top - fvg_bottom
            mid_price = (fvg_top + fvg_bottom) / 2
            filled = False
            for j in range(i+1, len(df)):
                if low[j] <= fvg_bottom:
                    filled = True
                    break
            if not filled and fvg_size > 0:
                fvgs.append({"type":"bullish","top":round(fvg_top,2),"bottom":round(fvg_bottom,2),"mid":round(mid_price,2),"size_pct":round((fvg_size/mid_price)*100,2),"age_days":len(df)-i,"filled":False})
        if high[i] < low[i-2]:
            fvg_top = float(low[i-2])
            fvg_bottom = float(high[i])
            fvg_size = fvg_top - fvg_bottom
            mid_price = (fvg_top + fvg_bottom) / 2
            filled = False
            for j in range(i+1, len(df)):
                if high[j] >= fvg_top:
                    filled = True
                    break
            if not filled and fvg_size > 0:
                fvgs.append({"type":"bearish","top":round(fvg_top,2),"bottom":round(fvg_bottom,2),"mid":round(mid_price,2),"size_pct":round((fvg_size/mid_price)*100,2),"age_days":len(df)-i,"filled":False})
    return sorted(fvgs, key=lambda x: x["age_days"])[:5]

def detect_wyckoff_phase(df):
    if df is None or len(df) < 30:
        return {"phase": "unknown", "description": "Not enough data", "signal": "neutral", "confidence": 0, "metrics": {}}
    close = df["Close"].values
    volume = df["Volume"].values
    high = df["High"].values
    low = df["Low"].values
    n = len(df)
    current_price = float(close[-1])
    last_20_close = close[-20:]
    last_20_vol = volume[-20:]
    last_10_close = close[-10:]
    prev_20_close = close[-40:-20] if n >= 40 else close[:20]
    prev_20_vol = volume[-40:-20] if n >= 40 else volume[:20]
    avg_vol_recent = float(np.mean(last_20_vol))
    avg_vol_prev = float(np.mean(prev_20_vol)) if len(prev_20_vol) > 0 else avg_vol_recent
    vol_change = (avg_vol_recent - avg_vol_prev) / avg_vol_prev * 100 if avg_vol_prev > 0 else 0
    price_range_20 = (float(max(last_20_close)) - float(min(last_20_close))) / float(np.mean(last_20_close)) * 100
    price_change_20 = (float(last_20_close[-1]) - float(last_20_close[0])) / float(last_20_close[0]) * 100
    price_change_10 = (float(last_10_close[-1]) - float(last_10_close[0])) / float(last_10_close[0]) * 100
    ema20 = float(pd.Series(close).ewm(span=20).mean().iloc[-1])
    ema50 = float(pd.Series(close).ewm(span=50).mean().iloc[-1])
    lows_5 = [float(min(low[max(0,i-5):i+1])) for i in range(n-20, n, 5)]
    higher_lows = all(lows_5[i] >= lows_5[i-1] for i in range(1, len(lows_5))) if len(lows_5) >= 2 else False
    highs_5 = [float(max(high[max(0,i-5):i+1])) for i in range(n-20, n, 5)]
    lower_highs = all(highs_5[i] <= highs_5[i-1] for i in range(1, len(highs_5))) if len(highs_5) >= 2 else False
    phase = "transition"
    confidence = 30
    description = "No clear phase."
    signal = "neutral"
    if price_range_20 < 10 and current_price < ema50 and higher_lows:
        phase = "accumulation"
        confidence = 70 + (15 if vol_change < -10 else 0) + (10 if price_change_10 > 0 else 0)
        description = "Accumulating near lows."
        signal = "bullish_soon"
    elif price_change_20 > 5 and current_price > ema20 > ema50:
        phase = "markup"
        confidence = 75 + (10 if vol_change > 10 else 0)
        description = "Strong uptrend."
        signal = "bullish"
    elif price_range_20 < 10 and current_price > ema50 and lower_highs:
        phase = "distribution"
        confidence = 65 + (15 if vol_change > 15 else 0)
        description = "Distributing near highs."
        signal = "bearish_soon"
    elif price_change_20 < -5 and current_price < ema20 and current_price < ema50:
        phase = "markdown"
        confidence = 75
        description = "Downtrend."
        signal = "bearish"
    elif n >= 10:
        recent_low = float(min(low[-10:]))
        prev_low = float(min(low[-30:-10])) if n >= 30 else float(min(low[:20]))
        if recent_low < prev_low and price_change_10 > 3:
            phase = "spring"
            confidence = 60
            description = "Potential spring."
            signal = "strong_bullish"
    return {"phase":phase,"confidence":min(confidence,95),"description":description,"signal":signal,"metrics":{"price_change_20d":round(price_change_20,2),"price_change_10d":round(price_change_10,2),"higher_lows":higher_lows,"lower_highs":lower_highs}}

def calc_accumulation_score(df, poc_price, va_low, va_high):
    if df is None or len(df) < 20:
        return {"score": 0, "level": "unknown", "factors": []}
    close = df["Close"].values
    volume = df["Volume"].values
    low = df["Low"].values
    current_price = float(close[-1])
    factors = []
    score = 0
    if poc_price and current_price < poc_price:
        dist = abs(current_price - poc_price) / poc_price * 100
        if dist <= 5: score += 25; factors.append({"name":"Below POC","score":25,"detail":"{}% below".format(round(dist,1)),"pass":True})
        elif dist <= 15: score += 15; factors.append({"name":"Below POC","score":15,"detail":"{}% below".format(round(dist,1)),"pass":True})
        else: factors.append({"name":"Below POC","score":0,"detail":"too far","pass":False})
    else: factors.append({"name":"Below POC","score":0,"detail":"above POC","pass":False})
    if va_low and current_price <= va_low * 1.02: score += 20; factors.append({"name":"Near VA Low","score":20,"detail":"near/below","pass":True})
    else: factors.append({"name":"Near VA Low","score":0,"detail":"above","pass":False})
    if len(volume) >= 20:
        v1 = float(np.mean(volume[-20:-10]))
        v2 = float(np.mean(volume[-10:]))
        if v1 > 0:
            vd = (v2-v1)/v1*100
            if vd < -15: score += 15; factors.append({"name":"Vol Decreasing","score":15,"detail":"down {}%".format(round(vd)),"pass":True})
            else: factors.append({"name":"Vol Decreasing","score":0,"detail":"{}%".format(round(vd)),"pass":False})
    score = min(score, 100)
    level = "strong" if score >= 70 else "moderate" if score >= 40 else "weak" if score >= 20 else "none"
    return {"score": score, "level": level, "factors": factors}

def detect_candlestick_patterns(df):
    if df is None or len(df) < 3: return []
    patterns = []
    o,h,l,c = df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values
    i = len(df)-1; i1 = i-1; i2 = i-2
    body = abs(c[i]-o[i]); range_total = h[i]-l[i]
    if range_total == 0: return []
    upper_shadow = h[i]-max(o[i],c[i]); lower_shadow = min(o[i],c[i])-l[i]
    is_bullish = c[i]>o[i]; is_bearish = c[i]<o[i]
    body1 = abs(c[i1]-o[i1]); is_bullish1 = c[i1]>o[i1]; is_bearish1 = c[i1]<o[i1]
    body2 = abs(c[i2]-o[i2]); body_pct = body/range_total
    if body_pct<0.35 and lower_shadow>=body*2 and upper_shadow<body*0.5:
        patterns.append({"name":"Hammer","type":"bullish","strength":"strong","description":"Bullish reversal"})
    if body_pct<0.35 and upper_shadow>=body*2 and lower_shadow<body*0.5:
        patterns.append({"name":"Inverted Hammer","type":"bullish","strength":"moderate","description":"Potential reversal"})
    if is_bearish1 and is_bullish and c[i]>o[i1] and o[i]<c[i1] and body>body1:
        patterns.append({"name":"Bullish Engulfing","type":"bullish","strength":"strong","description":"Strong reversal"})
    if is_bullish1 and is_bearish and o[i]>c[i1] and c[i]<o[i1] and body>body1:
        patterns.append({"name":"Bearish Engulfing","type":"bearish","strength":"strong","description":"Strong reversal"})
    if body_pct<0.1: patterns.append({"name":"Doji","type":"neutral","strength":"moderate","description":"Indecision"})
    day1_mid = (o[i2]+c[i2])/2
    if c[i2]<o[i2] and body2>range_total*0.15 and body1<body2*0.4 and is_bullish and c[i]>day1_mid:
        patterns.append({"name":"Morning Star","type":"bullish","strength":"strong","description":"3-candle reversal"})
    day1_mid2 = (o[i2]+c[i2])/2
    if c[i2]>o[i2] and body2>range_total*0.15 and body1<body2*0.4 and is_bearish and c[i]<day1_mid2:
        patterns.append({"name":"Evening Star","type":"bearish","strength":"strong","description":"3-candle reversal"})
    if (c[i2]>o[i2]) and (c[i1]>o[i1]) and (c[i]>o[i]) and c[i1]>c[i2] and c[i]>c[i1]:
        patterns.append({"name":"Three White Soldiers","type":"bullish","strength":"strong","description":"Strong buying"})
    if (c[i2]<o[i2]) and (c[i1]<o[i1]) and (c[i]<o[i]) and c[i1]<c[i2] and c[i]<c[i1]:
        patterns.append({"name":"Three Black Crows","type":"bearish","strength":"strong","description":"Strong selling"})
    if body_pct<0.3 and upper_shadow>=body*2 and lower_shadow<body*0.3 and is_bearish:
        patterns.append({"name":"Shooting Star","type":"bearish","strength":"moderate","description":"Bearish reversal"})
    return patterns

def get_pattern_score_bonus(patterns):
    bonus = 0
    for p in patterns:
        if p["type"]=="bullish" and p["strength"]=="strong": bonus += 8
        elif p["type"]=="bullish": bonus += 4
        elif p["type"]=="bearish" and p["strength"]=="strong": bonus -= 6
        elif p["type"]=="bearish": bonus -= 3
    return bonus

def calc_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta>0,0).rolling(window=period).mean()
    loss = (-delta.where(delta<0,0)).rolling(window=period).mean()
    rs = gain/loss; rsi = 100-(100/(1+rs))
    val = rsi.iloc[-1] if not rsi.empty else 50
    return 50 if pd.isna(val) else val

def calc_ema(prices, period):
    ema = prices.ewm(span=period, adjust=False).mean()
    val = ema.iloc[-1] if not ema.empty else 0
    return 0 if pd.isna(val) else val

def calc_macd(prices):
    e12 = prices.ewm(span=12,adjust=False).mean()
    e26 = prices.ewm(span=26,adjust=False).mean()
    ml = e12-e26; sig = ml.ewm(span=9,adjust=False).mean(); hist = ml-sig
    return {"macd":round(float(ml.iloc[-1]),4),"signal":round(float(sig.iloc[-1]),4),"histogram":round(float(hist.iloc[-1]),4)}

def calc_volume_profile(highs, lows, volumes, bins=30):
    try:
        pmin = float(lows.min()); pmax = float(highs.max())
        if pmax <= pmin: return None,None,None,[]
        edges = np.linspace(pmin,pmax,bins+1); vpl = np.zeros(bins)
        for idx in range(len(highs)):
            rl,rh,rv = float(lows.iloc[idx]),float(highs.iloc[idx]),float(volumes.iloc[idx])
            sp = max(1,int((rh-rl)/((pmax-pmin)/bins)))
            for i in range(bins):
                if rl<=edges[i+1] and rh>=edges[i]: vpl[i]+=rv/sp
        pi = int(np.argmax(vpl)); poc = round((edges[pi]+edges[pi+1])/2,2)
        tv = vpl.sum(); tgt = tv*0.70; si = np.argsort(vpl)[::-1]; cum = 0; vai = []
        for idx in si: cum+=vpl[idx]; vai.append(idx); 
        if cum >= tgt: pass
        va_low = round(float(edges[min(vai)]),2); va_high = round(float(edges[max(vai)+1]),2)
        mx = float(vpl.max()) if vpl.max()>0 else 1
        dist = [{"price":round((edges[i]+edges[i+1])/2,2),"volume_pct":round(float(vpl[i]/mx*100),1),"is_poc":i==pi,"in_value_area":i in vai} for i in range(bins)]
        return poc,va_high,va_low,dist
    except: return None,None,None,[]

def calc_setup_score(data):
    score = 0; rsi = data.get("rsi",50)
    if 40<=rsi<=60: score+=15
    elif 30<=rsi<=70: score+=10
    else: score+=5
    if data.get("macd_histogram",0)>0: score+=15
    elif data.get("macd_histogram",0)>-0.5: score+=8
    p,e10,e20,e50 = data.get("price",0),data.get("ema10",0),data.get("ema20",0),data.get("ema50",0)
    if p>e10>e20>e50: score+=15
    elif p>e20>e50: score+=10
    elif p>e50: score+=5
    rv = data.get("relative_volume",1)
    if rv>=2: score+=15
    elif rv>=1.5: score+=12
    elif rv>=1: score+=8
    poc = data.get("poc_price",0)
    if poc and p: d=abs(p-poc)/p*100; score+=(15 if d<=2 else 10 if d<=5 else 5 if d<=10 else 0)
    score += int(data.get("sector_strength",50)/100*15)
    ch = data.get("change_pct",0)
    if 0.5<=ch<=5: score+=10
    elif 0<ch<=0.5: score+=6
    score += data.get("pattern_bonus",0)
    return max(0,min(score,100))

def detect_setup_type(data):
    p,poc,vah,e20,rsi,rv = data.get("price",0),data.get("poc_price",0),data.get("va_high",0),data.get("ema20",0),data.get("rsi",50),data.get("relative_volume",1)
    if vah and p>vah and rv>=1.5: return "breakout"
    if poc and p and abs(p-poc)/p*100<=2: return "pullback_to_poc"
    if e20 and p and abs(p-e20)/p*100<=1.5: return "ema_bounce"
    if rsi<=30: return "oversold_reversal"
    if rsi>=70: return "overbought_warning"
    return "neutral"

async def fetch_and_analyze_sectors():
    db = get_db()
    print("=" * 50)
    print("STARTING SECTOR REFRESH (Alpaca)")
    print("=" * 50)
    async with httpx.AsyncClient(timeout=30) as client:
        spy_df = await fetch_bars(client, "SPY")
        spy_return = 0
        if spy_df is not None and len(spy_df) >= 20:
            spy_return = ((float(spy_df["Close"].iloc[-1]) / float(spy_df["Close"].iloc[-20])) - 1) * 100
            print("SPY 20d return: {:.2f}%".format(spy_return))
            spy_close = spy_df["Close"]
            spy_ema20 = float(spy_close.ewm(span=20).mean().iloc[-1])
            spy_ema50 = float(spy_close.ewm(span=50).mean().iloc[-1])
            spy_rsi_d = spy_close.diff()
            spy_g = spy_rsi_d.where(spy_rsi_d>0,0).rolling(14).mean()
            spy_l = (-spy_rsi_d.where(spy_rsi_d<0,0)).rolling(14).mean()
            spy_rsi_val = float((100-(100/(1+spy_g/spy_l))).iloc[-1])
            spy_price = float(spy_close.iloc[-1])
            spy_change = float(((spy_close.iloc[-1]/spy_close.iloc[-2])-1)*100)
            await db.market_regime.update_one({"symbol":"SPY"},{"$set":{"symbol":"SPY","price":round(spy_price,2),"change_pct":round(spy_change,2),"ema20":round(spy_ema20,2),"ema50":round(spy_ema50,2),"rsi":round(spy_rsi_val,1),"return_20d":round(spy_return,2),"updated_at":datetime.utcnow()}},upsert=True)
        # Fetch major indices
        for idx_sym in ["QQQ", "IWM", "DIA"]:
            await asyncio.sleep(0.3)
            try:
                idx_df = await fetch_bars(client, idx_sym)
                if idx_df is not None and len(idx_df) >= 2:
                    ip = float(idx_df["Close"].iloc[-1])
                    ipp = float(idx_df["Close"].iloc[-2])
                    ic = round(((ip - ipp) / ipp) * 100, 2)
                    ir20 = round(((float(idx_df["Close"].iloc[-1]) / float(idx_df["Close"].iloc[-20])) - 1) * 100, 2) if len(idx_df) >= 20 else 0
                    await db.market_regime.update_one({"symbol": idx_sym}, {"$set": {"symbol": idx_sym, "price": round(ip, 2), "change_pct": ic, "return_20d": ir20, "updated_at": datetime.utcnow()}}, upsert=True)
                    print("  {} saved: ${:.2f} ({:+.2f}%)".format(idx_sym, ip, ic))
            except Exception as e:
                print("  {} error: {}".format(idx_sym, e))

        # Fetch crypto
        for crypto in ["BTC/USD", "ETH/USD"]:
            await asyncio.sleep(0.3)
            try:
                cr = await client.get("https://data.alpaca.markets/v1beta3/crypto/us/bars", headers=ALPACA_HEADERS, params={"symbols": crypto, "timeframe": "1Day", "limit": 5})
                if cr.status_code == 200:
                    cbars = cr.json().get("bars", {}).get(crypto, [])
                    if cbars and len(cbars) >= 2:
                        cp = float(cbars[-1].get("c", 0))
                        cpp = float(cbars[-2].get("c", cp))
                        cc = round(((cp - cpp) / cpp) * 100, 2) if cpp > 0 else 0
                        await db.market_regime.update_one({"symbol": crypto}, {"$set": {"symbol": crypto, "price": round(cp, 2), "change_pct": cc, "updated_at": datetime.utcnow()}}, upsert=True)
                        print("  {} saved: ${:.2f} ({:+.2f}%)".format(crypto, cp, cc))
            except Exception as e:
                print("  {} error: {}".format(crypto, e))

        # Fetch EUR/USD proxy
        for fx in ["FXE", "UUP"]:
            await asyncio.sleep(0.3)
            try:
                fx_df = await fetch_bars(client, fx)
                if fx_df is not None and len(fx_df) >= 2:
                    fp = float(fx_df["Close"].iloc[-1])
                    fpp = float(fx_df["Close"].iloc[-2])
                    fc = round(((fp - fpp) / fpp) * 100, 2)
                    await db.market_regime.update_one({"symbol": fx}, {"$set": {"symbol": fx, "price": round(fp, 2), "change_pct": fc, "updated_at": datetime.utcnow()}}, upsert=True)
                    print("  {} saved: ${:.2f} ({:+.2f}%)".format(fx, fp, fc))
            except Exception as e:
                print("  {} error: {}".format(fx, e))
        results = []
        for etf, name in SECTOR_MAP.items():
            try:
                await asyncio.sleep(0.3)
                df = await fetch_bars(client, etf)
                if df is None or len(df) < 20: print("  SKIP {}".format(etf)); continue
                close = df["Close"]; volume = df["Volume"]
                ret_20d = ((float(close.iloc[-1])/float(close.iloc[-20]))-1)*100
                strength = round(float(ret_20d-spy_return),2)
                rsi = round(float(calc_rsi(close)),2)
                ema10 = float(calc_ema(close,10)); ema20_val = float(calc_ema(close,20)); ema50 = float(calc_ema(close,50))
                price = float(close.iloc[-1])
                avg_vol = float(volume.rolling(20).mean().iloc[-1]); curr_vol = float(volume.iloc[-1])
                rel_vol = round(curr_vol/avg_vol,2) if avg_vol>0 else 1
                trend = 90 if price>ema10>ema20_val>ema50 else (70 if price>ema20_val>ema50 else (50 if price>ema50 else 30))
                composite = round((strength*2+trend+rsi)/4,2)
                history = []
                rs_s = pd.Series(dtype=float); d=close.diff(); g=d.where(d>0,0).rolling(14).mean(); lo=(-d.where(d<0,0)).rolling(14).mean(); rs_s=100-(100/(1+g/lo))
                e10s=close.ewm(span=10).mean(); e20s=close.ewm(span=20).mean(); e50s=close.ewm(span=50).mean()
                for idx in range(max(20,len(df)-90), len(df)):
                    dc=float(close.iloc[idx]); dr=float(rs_s.iloc[idx]) if not pd.isna(rs_s.iloc[idx]) else 50
                    zone = "oversold" if dr<=30 else ("weak" if dr<=40 else ("overbought" if dr>=70 else ("strong" if dr>=60 else "neutral")))
                    history.append({"date":df["datetime"].iloc[idx].strftime("%Y-%m-%d") if "datetime" in df.columns else "d{}".format(idx),"close":round(dc,2),"rsi":round(dr,1),"ema10":round(float(e10s.iloc[idx]),2),"ema20":round(float(e20s.iloc[idx]),2),"ema50":round(float(e50s.iloc[idx]),2),"zone":zone})
                sector_doc = {"code":etf,"name":name,"etf_ticker":etf,"price":round(price,2),"return_20d":round(float(ret_20d),2),"strength_score":strength,"trend_score":trend,"volume_score":round(rel_vol*30,2),"rsi":rsi,"composite_score":composite,"history":history,"updated_at":datetime.utcnow()}
                await db.sectors.update_one({"code":etf},{"$set":sector_doc},upsert=True)
                results.append(sector_doc)
                print("  OK {}: ${:.2f} score={:.2f}".format(etf,price,composite))
            except Exception as e: print("  ERROR {}: {}".format(etf,e)); traceback.print_exc()
    print("\nSECTORS DONE: {}/11".format(len(results)))
    return results

async def fetch_and_analyze_stocks():
    db = get_db()
    print("=" * 50)
    print("STARTING STOCKS REFRESH (Alpaca)")
    print("=" * 50)
    sector_scores = {}
    async for s in db.sectors.find(): sector_scores[s["code"]] = s.get("composite_score",50)
    results = []
    async with httpx.AsyncClient(timeout=30) as client:
        for sector_code, tickers in SECTOR_STOCKS.items():
            print("\n--- {} ---".format(sector_code))
            for ticker in tickers:
                try:
                    await asyncio.sleep(0.3)
                    df = await fetch_bars(client, ticker)
                    if df is None or len(df) < 20: print("    SKIP {}".format(ticker)); continue
                    close=df["Close"]; volume=df["Volume"]; high=df["High"]; low=df["Low"]
                    price=float(close.iloc[-1]); prev_close=float(close.iloc[-2])
                    change_pct=round(((price-prev_close)/prev_close)*100,2)
                    avg_vol=float(volume.rolling(20).mean().iloc[-1]); curr_vol=float(volume.iloc[-1])
                    rel_vol=round(curr_vol/avg_vol,2) if avg_vol>0 else 1
                    rsi=round(float(calc_rsi(close)),2); macd=calc_macd(close)
                    ema10=round(float(calc_ema(close,10)),2); ema20=round(float(calc_ema(close,20)),2); ema50=round(float(calc_ema(close,50)),2)
                    poc_result=calc_volume_profile(high,low,volume)
                    high_52w=round(float(high.max()),2); low_52w=round(float(low.min()),2)
                    pct_from_high=round(((price-high_52w)/high_52w)*100,2) if high_52w>0 else 0
                    pct_from_low=round(((price-low_52w)/low_52w)*100,2) if low_52w>0 else 0
                    range_position=round(((price-low_52w)/(high_52w-low_52w))*100,1) if (high_52w-low_52w)>0 else 50
                    poc=poc_result[0]; va_high=poc_result[1]; va_low=poc_result[2]; vp_distribution=poc_result[3]
                    patterns=detect_candlestick_patterns(df)
                    fvgs=detect_fvg(df); wyckoff=detect_wyckoff_phase(df); accumulation=calc_accumulation_score(df,poc,va_low,va_high)
                    rs_s=pd.Series(dtype=float); ds=close.diff(); gs=ds.where(ds>0,0).rolling(14).mean(); ls=(-ds.where(ds<0,0)).rolling(14).mean(); rs_s=100-(100/(1+gs/ls))
                    e10s=close.ewm(span=10).mean(); e20s=close.ewm(span=20).mean(); e50s=close.ewm(span=50).mean()
                    price_history=[]
                    start_idx=max(20,len(df)-90)
                    for idx in range(start_idx, len(df)):
                        dr=float(rs_s.iloc[idx]) if not pd.isna(rs_s.iloc[idx]) else 50
                        price_history.append({"date":df["datetime"].iloc[idx].strftime("%Y-%m-%d") if "datetime" in df.columns else "d{}".format(idx),"close":round(float(close.iloc[idx]),2),"high":round(float(high.iloc[idx]),2),"low":round(float(low.iloc[idx]),2),"volume":int(volume.iloc[idx]),"rsi":round(dr,1),"ema10":round(float(e10s.iloc[idx]),2),"ema20":round(float(e20s.iloc[idx]),2),"ema50":round(float(e50s.iloc[idx]),2)})
                    pattern_bonus=get_pattern_score_bonus(patterns)
                    patterns_list=[{"name":p["name"],"type":p["type"],"strength":p["strength"],"description":p["description"]} for p in patterns]
                    ind_data={"price":price,"rsi":rsi,"macd_histogram":macd["histogram"],"ema10":ema10,"ema20":ema20,"ema50":ema50,"relative_volume":rel_vol,"poc_price":poc,"va_high":va_high,"change_pct":change_pct,"sector_strength":sector_scores.get(sector_code,50),"pattern_bonus":pattern_bonus}
                    setup_score=calc_setup_score(ind_data); setup_type=detect_setup_type(ind_data)
                    asset_doc={"ticker":ticker,"name":ticker,"sector_code":sector_code,"price":round(price,2),"change_pct":change_pct,"avg_volume":round(avg_vol,0),"relative_volume":rel_vol,"rsi":rsi,"macd":macd,"ema10":ema10,"ema20":ema20,"ema50":ema50,"momentum_score":rsi,"volume_score":round(rel_vol*30,2),"poc_price":poc,"value_area_high":va_high,"value_area_low":va_low,"setup_score":setup_score,"setup_type":setup_type,"vp_distribution":vp_distribution,"candlestick_patterns":patterns_list,"fvg":fvgs,"wyckoff":wyckoff,"accumulation":accumulation,"price_history":price_history,"pattern_bonus":pattern_bonus,"high_52w":high_52w,"low_52w":low_52w,"pct_from_high":pct_from_high,"pct_from_low":pct_from_low,"range_position":range_position,"updated_at":datetime.utcnow()}
                    await db.assets.update_one({"ticker":ticker},{"$set":asset_doc},upsert=True)
                    results.append(asset_doc)
                    pat_str=", ".join([p["name"] for p in patterns]) if patterns else "none"
                    print("    OK {}: ${:.2f} score={} [{}] patterns=[{}]".format(ticker,price,setup_score,setup_type,pat_str))
                except Exception as e: print("    ERROR {}: {}".format(ticker,e))
    print("\nSTOCKS DONE: {}/220".format(len(results)))
    return results

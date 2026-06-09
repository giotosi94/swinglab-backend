# Stock ticker -> Company name mapping
STOCK_NAMES = {
    # XLK - Technology
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "AVGO": "Broadcom",
    "AMD": "AMD", "CRM": "Salesforce", "ADBE": "Adobe", "INTC": "Intel",
    "CSCO": "Cisco", "ORCL": "Oracle", "PLTR": "Palantir", "NOW": "ServiceNow",
    "SNOW": "Snowflake", "CRWD": "CrowdStrike", "PANW": "Palo Alto Networks",
    "MNDY": "Monday.com", "SHOP": "Shopify", "SQ": "Block", "UBER": "Uber",
    "DDOG": "Datadog",
    # XLF - Financials
    "JPM": "JPMorgan", "BAC": "Bank of America", "WFC": "Wells Fargo",
    "GS": "Goldman Sachs", "MS": "Morgan Stanley", "BLK": "BlackRock",
    "SCHW": "Schwab", "AXP": "American Express", "C": "Citigroup", "USB": "US Bancorp",
    "V": "Visa", "MA": "Mastercard", "PYPL": "PayPal", "COF": "Capital One",
    "ICE": "ICE", "SPGI": "S&P Global", "MCO": "Moody's", "MMC": "Marsh McLennan",
    "AON": "Aon", "TFC": "Truist",
    # XLV - Health Care
    "UNH": "UnitedHealth", "JNJ": "Johnson & Johnson", "PFE": "Pfizer",
    "ABBV": "AbbVie", "MRK": "Merck", "TMO": "Thermo Fisher", "ABT": "Abbott",
    "LLY": "Eli Lilly", "BMY": "Bristol-Myers", "AMGN": "Amgen",
    "ISRG": "Intuitive Surgical", "DXCM": "DexCom", "VRTX": "Vertex",
    "REGN": "Regeneron", "ZTS": "Zoetis", "HCA": "HCA Healthcare",
    "CI": "Cigna", "ELV": "Elevance", "HUM": "Humana", "SYK": "Stryker",
    # XLI - Industrials
    "CAT": "Caterpillar", "DE": "Deere", "UNP": "Union Pacific",
    "HON": "Honeywell", "BA": "Boeing", "RTX": "RTX", "LMT": "Lockheed Martin",
    "GE": "GE Aerospace", "MMM": "3M", "FDX": "FedEx", "UPS": "UPS",
    "WM": "Waste Management", "ETN": "Eaton", "ITW": "Illinois Tool Works",
    "EMR": "Emerson", "NSC": "Norfolk Southern", "CSX": "CSX",
    "PCAR": "PACCAR", "ROK": "Rockwell", "IR": "Ingersoll Rand",
    # XLY - Consumer Discretionary
    "AMZN": "Amazon", "TSLA": "Tesla", "HD": "Home Depot", "MCD": "McDonald's",
    "NKE": "Nike", "SBUX": "Starbucks", "LOW": "Lowe's", "TJX": "TJX",
    "BKNG": "Booking", "CMG": "Chipotle", "LULU": "Lululemon", "ROST": "Ross Stores",
    "DHI": "D.R. Horton", "LEN": "Lennar", "ABNB": "Airbnb", "DASH": "DoorDash",
    "EBAY": "eBay", "MAR": "Marriott", "HLT": "Hilton", "YUM": "Yum! Brands",
    # XLP - Consumer Staples
    "PG": "Procter & Gamble", "KO": "Coca-Cola", "PEP": "PepsiCo",
    "COST": "Costco", "WMT": "Walmart", "PM": "Philip Morris", "MO": "Altria",
    "CL": "Colgate", "MDLZ": "Mondelez", "KHC": "Kraft Heinz",
    "STZ": "Constellation", "SYY": "Sysco", "HSY": "Hershey", "GIS": "General Mills",
    "ADM": "ADM", "MNST": "Monster", "KDP": "Keurig Dr Pepper",
    "CHD": "Church & Dwight", "CLX": "Clorox", "SJM": "Smucker's",
    # XLE - Energy
    "XOM": "ExxonMobil", "CVX": "Chevron", "COP": "ConocoPhillips",
    "SLB": "Schlumberger", "EOG": "EOG Resources", "MPC": "Marathon Petroleum",
    "PSX": "Phillips 66", "VLO": "Valero", "OXY": "Occidental", "HAL": "Halliburton",
    "DVN": "Devon Energy", "FANG": "Diamondback", "WMB": "Williams",
    "KMI": "Kinder Morgan", "TRGP": "Targa Resources", "BKR": "Baker Hughes",
    "CTRA": "Coterra", "MRO": "Marathon Oil", "APA": "APA Corp", "AR": "Antero",
    # XLU - Utilities
    "NEE": "NextEra", "DUK": "Duke Energy", "SO": "Southern Co", "D": "Dominion",
    "AEP": "AEP", "SRE": "Sempra", "EXC": "Exelon", "XEL": "Xcel Energy",
    "ED": "Con Edison", "WEC": "WEC Energy", "AWK": "American Water",
    "ES": "Eversource", "ATO": "Atmos Energy", "CMS": "CMS Energy",
    "PNW": "Pinnacle West", "PPL": "PPL Corp", "FE": "FirstEnergy",
    "DTE": "DTE Energy", "AES": "AES Corp", "ETR": "Entergy",
    # XLB - Materials
    "LIN": "Linde", "APD": "Air Products", "SHW": "Sherwin-Williams",
    "FCX": "Freeport-McMoRan", "NEM": "Newmont", "ECL": "Ecolab",
    "DOW": "Dow", "NUE": "Nucor", "VMC": "Vulcan Materials", "MLM": "Martin Marietta",
    "CF": "CF Industries", "MOS": "Mosaic", "BALL": "Ball Corp",
    "PKG": "Packaging Corp", "IFF": "IFF", "EMN": "Eastman Chemical",
    "CE": "Celanese", "RPM": "RPM International", "SEE": "Sealed Air", "AVY": "Avery Dennison",
    # XLRE - Real Estate
    "PLD": "Prologis", "AMT": "American Tower", "CCI": "Crown Castle",
    "EQIX": "Equinix", "SPG": "Simon Property", "PSA": "Public Storage",
    "O": "Realty Income", "WELL": "Welltower", "DLR": "Digital Realty",
    "AVB": "AvalonBay", "VICI": "VICI Properties", "MAA": "Mid-America",
    "EXR": "Extra Space", "ARE": "Alexandria RE", "UDR": "UDR",
    "ESS": "Essex Property", "REG": "Regency Centers", "HST": "Host Hotels",
    "KIM": "Kimco Realty", "CPT": "Camden Property",
    # XLC - Communication Services
    "META": "Meta", "GOOGL": "Alphabet", "GOOG": "Alphabet", "NFLX": "Netflix",
    "DIS": "Disney", "CMCSA": "Comcast", "T": "AT&T", "VZ": "Verizon",
    "TMUS": "T-Mobile", "EA": "Electronic Arts", "SPOT": "Spotify",
    "RBLX": "Roblox", "TTWO": "Take-Two", "WBD": "Warner Bros",
    "PARA": "Paramount", "MTCH": "Match Group", "ZM": "Zoom",
    "PINS": "Pinterest", "SNAP": "Snap", "LYV": "Live Nation",
    # ETFs & Indices
    "SPY": "S&P 500", "QQQ": "Nasdaq 100", "IWM": "Russell 2000", "DIA": "Dow Jones",
    "XLK": "Tech ETF", "XLF": "Finance ETF", "XLV": "Health ETF",
    "XLI": "Industrial ETF", "XLY": "Consumer Disc ETF", "XLP": "Consumer Staples ETF",
    "XLE": "Energy ETF", "XLU": "Utilities ETF", "XLB": "Materials ETF",
    "XLRE": "Real Estate ETF", "XLC": "Communication ETF",
    "VIXY": "VIX Short-Term", "FXE": "Euro ETF", "UUP": "Dollar ETF",
}


def get_stock_name(ticker: str) -> str:
    """Get company name for a ticker. Returns ticker if not found."""
    return STOCK_NAMES.get(ticker.upper(), ticker)

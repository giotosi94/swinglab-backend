from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class AssetModel(BaseModel):
    ticker: str
    name: str
    sector_code: str
    price: float = 0.0
    change_pct: float = 0.0            # variazione % giornaliera
    avg_volume: float = 0.0
    relative_volume: float = 0.0       # volume vs media
    momentum_score: float = 0.0        # RSI + trend
    volume_score: float = 0.0          # analisi volumetrica
    poc_price: Optional[float] = None  # Point of Control
    value_area_high: Optional[float] = None
    value_area_low: Optional[float] = None
    setup_score: float = 0.0           # score finale setup
    setup_type: Optional[str] = None   # "breakout", "pullback", "reversal"
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class AssetCreate(BaseModel):
    ticker: str
    name: str
    sector_code: str

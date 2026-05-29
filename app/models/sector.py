from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class SectorModel(BaseModel):
    code: str                          # es: "XLK", "XLF", "XLI"
    name: str                          # es: "Technology"
    etf_ticker: str                    # ETF settoriale (SPDR)
    strength_score: float = 0.0        # forza relativa vs SPY
    trend_score: float = 0.0           # trend momentum
    volume_score: float = 0.0          # volume anomaly score
    composite_score: float = 0.0       # score combinato finale
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class SectorCreate(BaseModel):
    code: str
    name: str
    etf_ticker: str

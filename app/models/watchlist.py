from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class WatchlistItem(BaseModel):
    ticker: str
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None
    notes: Optional[str] = None
    status: str = "watching"            # watching, entered, closed
    created_at: datetime = Field(default_factory=datetime.utcnow)

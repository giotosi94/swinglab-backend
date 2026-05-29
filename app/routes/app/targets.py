from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class TargetRequest(BaseModel):
    entry_price: float
    stop_loss: float
    risk_reward: float = 2.0
    position_size_usd: float = 1000.0

@router.post("/calculate")
def calculate_targets(req: TargetRequest):
    risk = abs(req.entry_price - req.stop_loss)
    reward = risk * req.risk_reward
    target = req.entry_price + reward

    shares = int(req.position_size_usd / req.entry_price)
    total_risk = round(risk * shares, 2)
    total_reward = round(reward * shares, 2)

    return {
        "entry": req.entry_price,
        "stop_loss": req.stop_loss,
        "target_price": round(target, 2),
        "risk_per_share": round(risk, 2),
        "reward_per_share": round(reward, 2),
        "risk_reward_ratio": req.risk_reward,
        "shares": shares,
        "total_risk_usd": total_risk,
        "total_reward_usd": total_reward,
    }

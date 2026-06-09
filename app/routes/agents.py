from fastapi import APIRouter
from app.agents.orchestrator import Orchestrator

router = APIRouter()
_orch = None

def _get_orch():
    global _orch
    if _orch is None:
        _orch = Orchestrator()
    return _orch


@router.get("/status")
async def agents_status():
    """Stato completo di tutti gli agenti."""
    orch = _get_orch()
    return await orch.get_status()


@router.get("/{agent_name}/decisions")
async def agent_decisions(agent_name: str, limit: int = 20):
    """Ultime decisioni di un agente specifico."""
    orch = _get_orch()
    agent = orch.agents.get(agent_name)
    if not agent:
        return {"error": f"Agent '{agent_name}' not found. Available: {list(orch.agents.keys())}"}
    decisions = await agent.get_recent_decisions(limit=limit)
    return {"agent": agent_name, "decisions": decisions}


@router.get("/{agent_name}/performance")
async def agent_performance(agent_name: str, limit: int = 30):
    """Storico performance di un agente."""
    orch = _get_orch()
    agent = orch.agents.get(agent_name)
    if not agent:
        return {"error": f"Agent '{agent_name}' not found"}
    perf = await agent.get_performance_history(limit=limit)
    return {"agent": agent_name, "performance": perf}


@router.get("/{agent_name}/params")
async def agent_params(agent_name: str):
    """Parametri appresi di un agente."""
    orch = _get_orch()
    agent = orch.agents.get(agent_name)
    if not agent:
        return {"error": f"Agent '{agent_name}' not found"}
    params = await agent.get_params()
    return {"agent": agent_name, "params": params}


@router.post("/learn")
async def learn_all():
    """Trigger learning per tutti gli agenti."""
    orch = _get_orch()
    return await orch.learn_all()


@router.post("/run")
async def run_pipeline():
    """Trigger manuale del pipeline multi-agent."""
    orch = _get_orch()
    return await orch.run()

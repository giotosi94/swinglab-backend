"""
SwingLab LLM Service v2.0 — Multi-provider with fallback + quota management
Gemini → Groq → Cerebras → cache/skip

Features:
- 26A: Caching per agente (skip se contesto invariato)
- 26B: Cooldown (reasoning ogni N minuti, non ogni run)
- 26C: Token budget tracker (blocca prima del limite 429)
- 26D: Terzo provider fallback (Cerebras)
- 26E: Graceful degradation (mai error, solo None)
"""

import time
import hashlib
from datetime import datetime, timedelta
from app.config import settings

# ============================================
# PROVIDER CLIENTS
# ============================================
_gemini_client = None
_groq_client = None
_cerebras_client = None


def _get_gemini():
    global _gemini_client
    if _gemini_client is None:
        try:
            from google import genai
            if settings.GEMINI_API_KEY:
                _gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
                print("  ✅ Gemini client initialized")
        except Exception as e:
            print(f"  Gemini init error: {e}")
    return _gemini_client


def _get_groq():
    global _groq_client
    if _groq_client is None:
        try:
            groq_key = getattr(settings, 'GROQ_API_KEY', '')
            if groq_key:
                from groq import Groq
                _groq_client = Groq(api_key=groq_key)
                print("  ✅ Groq client initialized")
        except Exception as e:
            print(f"  Groq init error: {e}")
    return _groq_client


def _get_cerebras():
    global _cerebras_client
    if _cerebras_client is None:
        try:
            cerebras_key = getattr(settings, 'CEREBRAS_API_KEY', '')
            if cerebras_key:
                from cerebras.cloud.sdk import Cerebras
                _cerebras_client = Cerebras(api_key=cerebras_key)
                print("  ✅ Cerebras client initialized")
        except Exception as e:
            print(f"  Cerebras init error: {e}")
    return _cerebras_client


# ============================================
# PROVIDER CALL FUNCTIONS
# ============================================
def _ask_gemini(system_prompt, user_prompt, max_tokens, temperature):
    client = _get_gemini()
    if not client:
        return None
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"{system_prompt}\n\n{user_prompt}",
        config={"max_output_tokens": max_tokens, "temperature": temperature},
    )
    return response.text


def _ask_groq(system_prompt, user_prompt, max_tokens, temperature):
    client = _get_groq()
    if not client:
        return None
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model="llama-3.3-70b-versatile",
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content


def _ask_cerebras(system_prompt, user_prompt, max_tokens, temperature):
    client = _get_cerebras()
    if not client:
        return None
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model="llama-3.3-70b",
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content


# ============================================
# 26C: TOKEN BUDGET TRACKER
# ============================================
_daily_usage = {
    "gemini": {"tokens": 0, "requests": 0, "date": ""},
    "groq": {"tokens": 0, "requests": 0, "date": ""},
    "cerebras": {"tokens": 0, "requests": 0, "date": ""},
}

# Limiti conservativi (80% del reale per margine di sicurezza)
_DAILY_LIMITS = {
    "gemini": {"tokens": 800_000, "requests": 1200},     # Free tier ~1500 req/day
    "groq": {"tokens": 80_000, "requests": 12_000},       # Free tier 100k tokens/day
    "cerebras": {"tokens": 800_000, "requests": 800},     # Free tier ~1000 req/day
}

# 26D: Provider priority order
_PROVIDER_ORDER = ["gemini", "groq", "cerebras"]

# 26B: Cooldown per agent
_COOLDOWN_MINUTES = {
    "macro_analyst": 25,       # ogni ~30 min (run ogni 30 min → skip 1 su 2 circa)
    "alpha_strategist": 12,    # ogni ~15 min
    "risk_manager": 12,        # ogni ~15 min
    "executor": 8,             # ogni ~10 min
    "default": 10,
}

# 26A: Cache per agente
_reasoning_cache = {}
# Struttura: {agent_name: {"hash": "...", "reasoning": "...", "timestamp": datetime}}


def _get_today():
    return datetime.utcnow().strftime("%Y-%m-%d")


def _reset_daily_if_needed():
    """Reset contatori se è un nuovo giorno."""
    today = _get_today()
    for provider in _daily_usage:
        if _daily_usage[provider]["date"] != today:
            _daily_usage[provider] = {"tokens": 0, "requests": 0, "date": today}


def _estimate_tokens(text):
    """Stima approssimativa tokens (1 token ≈ 4 caratteri)."""
    return len(text) // 4


def _check_budget(provider, estimated_tokens):
    """Controlla se il provider ha budget disponibile."""
    _reset_daily_if_needed()
    usage = _daily_usage[provider]
    limits = _DAILY_LIMITS.get(provider, {"tokens": 999_999, "requests": 999_999})

    if usage["tokens"] + estimated_tokens > limits["tokens"]:
        return False
    if usage["requests"] + 1 > limits["requests"]:
        return False
    return True


def _track_usage(provider, input_text, output_text):
    """Registra l'uso dopo una chiamata riuscita."""
    _reset_daily_if_needed()
    tokens_used = _estimate_tokens(input_text) + _estimate_tokens(output_text or "")
    _daily_usage[provider]["tokens"] += tokens_used
    _daily_usage[provider]["requests"] += 1


# ============================================
# 26A: CACHE CHECK
# ============================================
def _get_context_hash(user_prompt):
    """Hash del contesto per capire se è cambiato."""
    # Prendi solo i primi 500 char per hash veloce
    return hashlib.md5(user_prompt[:500].encode()).hexdigest()


def _check_cache(agent_name, user_prompt):
    """Ritorna il reasoning cachato se il contesto non è cambiato."""
    if agent_name not in _reasoning_cache:
        return None

    cached = _reasoning_cache[agent_name]
    current_hash = _get_context_hash(user_prompt)

    # Se hash uguale e cache recente (< 20 min) → usa cache
    if cached["hash"] == current_hash:
        age_minutes = (datetime.utcnow() - cached["timestamp"]).total_seconds() / 60
        if age_minutes < 20:
            print(f"  💾 LLM cache hit for {agent_name} (age: {age_minutes:.0f}min)")
            return cached["reasoning"]

    return None


def _save_cache(agent_name, user_prompt, reasoning):
    """Salva il reasoning nella cache."""
    _reasoning_cache[agent_name] = {
        "hash": _get_context_hash(user_prompt),
        "reasoning": reasoning,
        "timestamp": datetime.utcnow(),
    }


# ============================================
# 26B: COOLDOWN CHECK
# ============================================
_last_llm_call = {}
# Struttura: {agent_name: datetime}


def _check_cooldown(agent_name):
    """Ritorna True se l'agente può fare una chiamata LLM."""
    cooldown = _COOLDOWN_MINUTES.get(agent_name, _COOLDOWN_MINUTES["default"])
    last_call = _last_llm_call.get(agent_name)

    if last_call is None:
        return True

    elapsed = (datetime.utcnow() - last_call).total_seconds() / 60
    if elapsed < cooldown:
        print(f"  ⏳ LLM cooldown for {agent_name}: {elapsed:.0f}/{cooldown}min")
        return False

    return True


def _update_cooldown(agent_name):
    """Aggiorna il timestamp dell'ultima chiamata."""
    _last_llm_call[agent_name] = datetime.utcnow()


# ============================================
# MAIN FUNCTION: llm_ask (v2.0)
# ============================================
def llm_ask(system_prompt, user_prompt, max_tokens=300, temperature=0.3, agent_name=None):
    """
    Try providers in order: Gemini → Groq → Cerebras
    With: cache, cooldown, budget tracking, graceful degradation.

    Args:
        agent_name: (optional) nome agente per cache/cooldown (es. "macro_analyst")
    """

    # 26A: Check cache
    if agent_name:
        cached = _check_cache(agent_name, user_prompt)
        if cached:
            return cached

    # 26B: Check cooldown
    if agent_name and not _check_cooldown(agent_name):
        # Cooldown attivo → ritorna cache vecchia se esiste
        if agent_name in _reasoning_cache:
            print(f"  💾 Using stale cache for {agent_name} (cooldown active)")
            return _reasoning_cache[agent_name]["reasoning"]
        return None

    # Stima token input
    input_text = system_prompt + user_prompt
    estimated_tokens = _estimate_tokens(input_text) + max_tokens

    # 26D: Try providers in order with budget check
    providers = [
        ("gemini", _ask_gemini),
        ("groq", _ask_groq),
        ("cerebras", _ask_cerebras),
    ]

    for provider_name, ask_fn in providers:
        # 26C: Check budget
        if not _check_budget(provider_name, estimated_tokens):
            remaining = _DAILY_LIMITS[provider_name]["tokens"] - _daily_usage[provider_name]["tokens"]
            print(f"  ⚠️ {provider_name} budget low: ~{remaining} tokens left, skipping")
            continue

        try:
            result = ask_fn(system_prompt, user_prompt, max_tokens, temperature)
            if result:
                print(f"  🧠 LLM response via {provider_name.capitalize()}")

                # Track usage
                _track_usage(provider_name, input_text, result)

                # Update cooldown
                if agent_name:
                    _update_cooldown(agent_name)
                    _save_cache(agent_name, user_prompt, result)

                return result
        except Exception as e:
            error_str = str(e)
            print(f"  {provider_name.capitalize()} error: {error_str}")

            # Se è un 429, segna il provider come esaurito per oggi
            if "429" in error_str or "rate_limit" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                _daily_usage[provider_name]["tokens"] = _DAILY_LIMITS[provider_name]["tokens"]
                print(f"  🚫 {provider_name} marked as exhausted for today")

    # 26E: Graceful degradation — return None, non exception
    print("  ⚠️ All LLM providers exhausted — skipping reasoning")
    return None


def llm_available():
    """Check if at least one provider is configured."""
    return _get_gemini() is not None or _get_groq() is not None or _get_cerebras() is not None


# ============================================
# UTILITY: Get usage stats (per debug/monitoring)
# ============================================
def get_llm_stats():
    """Ritorna statistiche uso LLM per oggi."""
    _reset_daily_if_needed()
    stats = {}
    for provider in _PROVIDER_ORDER:
        usage = _daily_usage[provider]
        limits = _DAILY_LIMITS[provider]
        stats[provider] = {
            "tokens_used": usage["tokens"],
            "tokens_limit": limits["tokens"],
            "tokens_pct": round(usage["tokens"] / limits["tokens"] * 100, 1) if limits["tokens"] > 0 else 0,
            "requests_used": usage["requests"],
            "requests_limit": limits["requests"],
            "available": _check_budget(provider, 500),
        }

    # Cache info
    cache_info = {}
    for agent, data in _reasoning_cache.items():
        age = (datetime.utcnow() - data["timestamp"]).total_seconds() / 60
        cache_info[agent] = {
            "age_minutes": round(age, 1),
            "has_cache": True,
        }

    return {
        "providers": stats,
        "cache": cache_info,
        "date": _get_today(),
    }

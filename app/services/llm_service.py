"""
SwingLab LLM Service v2.1 — Multi-provider with fallback + quota management
Gemini → Groq → Cerebras → cache/skip

Features:
- 26A: Caching per agente + ticker (fix duplicato analisi)
- 26B: Cooldown intelligente (per agente + ticker separato)
- 26C: Token budget tracker (blocca prima del limite 429)
- 26D: Terzo provider fallback (Cerebras)
- 26E: Graceful degradation (mai error, solo None)

🆕 v2.1 — Fix cache key:
- Cache ora include ticker se presente nel prompt
- Cooldown separato tra "reasoning generico" (macro/risk/executor)
  e "analisi per ticker" (alpha per singolo stock)
"""

import time
import hashlib
import re
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

_DAILY_LIMITS = {
    "gemini": {"tokens": 800_000, "requests": 1200},
    "groq": {"tokens": 80_000, "requests": 12_000},
    "cerebras": {"tokens": 800_000, "requests": 800},
}

_PROVIDER_ORDER = ["gemini", "groq", "cerebras"]

# 🆕 v2.1 — Cooldown separato per tipologia
_COOLDOWN_MINUTES = {
    "macro_analyst": 25,       # reasoning generico → cooldown lungo
    "alpha_strategist": 12,    # reasoning generico → cooldown medio
    "risk_manager": 12,        # reasoning generico → cooldown medio
    "executor": 8,             # reasoning generico → cooldown breve
    "default": 10,
    # 🆕 Analisi per ticker specifico → cooldown molto breve (evita duplicati)
    "alpha_strategist_ticker": 2,  # 2 min tra analisi diverse ticker
}

# 🆕 v2.1 — Cache PER (agente + ticker)
_reasoning_cache = {}
# Struttura vecchia: {agent_name: {"hash": "...", "reasoning": "...", "timestamp": datetime}}
# Struttura nuova:   {cache_key: {"hash": "...", "reasoning": "...", "timestamp": datetime}}
# cache_key = agent_name OR "agent_name:TICKER" se ticker rilevato


def _get_today():
    return datetime.utcnow().strftime("%Y-%m-%d")


def _reset_daily_if_needed():
    today = _get_today()
    for provider in _daily_usage:
        if _daily_usage[provider]["date"] != today:
            _daily_usage[provider] = {"tokens": 0, "requests": 0, "date": today}


def _estimate_tokens(text):
    return len(text) // 4


def _check_budget(provider, estimated_tokens):
    _reset_daily_if_needed()
    usage = _daily_usage[provider]
    limits = _DAILY_LIMITS.get(provider, {"tokens": 999_999, "requests": 999_999})

    if usage["tokens"] + estimated_tokens > limits["tokens"]:
        return False
    if usage["requests"] + 1 > limits["requests"]:
        return False
    return True


def _track_usage(provider, input_text, output_text):
    _reset_daily_if_needed()
    tokens_used = _estimate_tokens(input_text) + _estimate_tokens(output_text or "")
    _daily_usage[provider]["tokens"] += tokens_used
    _daily_usage[provider]["requests"] += 1


# ============================================
# 🆕 v2.1 — TICKER EXTRACTION
# ============================================
def _extract_ticker(user_prompt):
    """
    🆕 Estrae il ticker dal user_prompt se presente.
    Cerca pattern: 'Ticker: XXX' o 'BUY XXX' o simili.
    Ritorna None se non trovato.
    """
    if not user_prompt:
        return None
    
    # Pattern comuni negli agenti
    patterns = [
        r'Ticker:\s*([A-Z]{1,6})\b',           # "Ticker: TFC"
        r'candidato\s+BUY\s+([A-Z]{1,6})\b',   # "candidato BUY TFC"
        r'BUY\s+([A-Z]{1,6})\s+',              # "BUY TFC "
    ]
    
    for pattern in patterns:
        match = re.search(pattern, user_prompt, re.IGNORECASE)
        if match:
            ticker = match.group(1).upper()
            # Filtro base: deve essere 1-6 caratteri, tutti maiuscoli
            if 1 <= len(ticker) <= 6 and ticker.isalpha():
                return ticker
    
    return None


def _build_cache_key(agent_name, user_prompt):
    """
    🆕 v2.1 — Costruisce cache key intelligente.
    Se il prompt contiene un ticker → cache key = "agent:TICKER"
    Altrimenti → cache key = agent_name (comportamento vecchio)
    """
    if not agent_name:
        return None
    
    ticker = _extract_ticker(user_prompt)
    if ticker:
        return f"{agent_name}:{ticker}"
    
    return agent_name


# ============================================
# 26A: CACHE CHECK (v2.1)
# ============================================
def _get_context_hash(user_prompt):
    """Hash del contesto completo (non solo primi 500 char)."""
    # 🆕 v2.1 — Usa TUTTO il prompt (non solo primi 500 char)
    return hashlib.md5(user_prompt.encode()).hexdigest()


def _check_cache(cache_key, user_prompt):
    """Ritorna il reasoning cachato se il contesto non è cambiato."""
    if not cache_key or cache_key not in _reasoning_cache:
        return None

    cached = _reasoning_cache[cache_key]
    current_hash = _get_context_hash(user_prompt)

    if cached["hash"] == current_hash:
        age_minutes = (datetime.utcnow() - cached["timestamp"]).total_seconds() / 60
        if age_minutes < 20:
            print(f"  💾 LLM cache hit for {cache_key} (age: {age_minutes:.0f}min)")
            return cached["reasoning"]

    return None


def _save_cache(cache_key, user_prompt, reasoning):
    """Salva il reasoning nella cache."""
    if not cache_key:
        return
    _reasoning_cache[cache_key] = {
        "hash": _get_context_hash(user_prompt),
        "reasoning": reasoning,
        "timestamp": datetime.utcnow(),
    }


# ============================================
# 26B: COOLDOWN CHECK (v2.1)
# ============================================
_last_llm_call = {}


def _get_cooldown_key(agent_name, ticker):
    """
    🆕 v2.1 — Cooldown key intelligente.
    Se c'è un ticker → cooldown per (agente, ticker) - più breve
    Altrimenti → cooldown per agente - più lungo
    """
    if ticker:
        return f"{agent_name}_ticker"  # cooldown speciale per ticker
    return agent_name


def _check_cooldown(agent_name, ticker=None):
    """Ritorna True se l'agente può fare una chiamata LLM."""
    cooldown_key = _get_cooldown_key(agent_name, ticker)
    cooldown = _COOLDOWN_MINUTES.get(cooldown_key, _COOLDOWN_MINUTES["default"])
    
    # Se c'è un ticker, usa una chiave separata per il tracking dell'ultima call
    tracking_key = f"{agent_name}:{ticker}" if ticker else agent_name
    last_call = _last_llm_call.get(tracking_key)

    if last_call is None:
        return True

    elapsed = (datetime.utcnow() - last_call).total_seconds() / 60
    if elapsed < cooldown:
        print(f"  ⏳ LLM cooldown for {tracking_key}: {elapsed:.0f}/{cooldown}min")
        return False

    return True


def _update_cooldown(agent_name, ticker=None):
    """Aggiorna il timestamp dell'ultima chiamata."""
    tracking_key = f"{agent_name}:{ticker}" if ticker else agent_name
    _last_llm_call[tracking_key] = datetime.utcnow()


# ============================================
# MAIN FUNCTION: llm_ask (v2.1)
# ============================================
def llm_ask(system_prompt, user_prompt, max_tokens=300, temperature=0.3, agent_name=None):
    """
    Try providers in order: Gemini → Groq → Cerebras
    With: cache PER TICKER, cooldown, budget tracking, graceful degradation.

    Args:
        agent_name: (optional) nome agente per cache/cooldown (es. "macro_analyst")
    """
    # 🆕 v2.1 — Estrai ticker se presente
    ticker = _extract_ticker(user_prompt)
    cache_key = _build_cache_key(agent_name, user_prompt)
    
    # 26A: Check cache PER TICKER
    if cache_key:
        cached = _check_cache(cache_key, user_prompt)
        if cached:
            return cached

    # 26B: Check cooldown (per ticker se presente)
    if agent_name and not _check_cooldown(agent_name, ticker):
        # 🆕 v2.1 — Cooldown attivo:
        # Se c'è ticker → ritorna None (evita duplicati con analisi generica)
        # Se non c'è ticker → ritorna cache vecchia dell'agente
        if ticker:
            print(f"  ⏭ Skipping LLM for {agent_name}:{ticker} (cooldown)")
            return None
        else:
            if cache_key in _reasoning_cache:
                print(f"  💾 Using stale cache for {cache_key} (cooldown active)")
                return _reasoning_cache[cache_key]["reasoning"]
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
        if not _check_budget(provider_name, estimated_tokens):
            remaining = _DAILY_LIMITS[provider_name]["tokens"] - _daily_usage[provider_name]["tokens"]
            print(f"  ⚠️ {provider_name} budget low: ~{remaining} tokens left, skipping")
            continue

        try:
            result = ask_fn(system_prompt, user_prompt, max_tokens, temperature)
            if result:
                print(f"  🧠 LLM response via {provider_name.capitalize()}" + (f" ({ticker})" if ticker else ""))

                _track_usage(provider_name, input_text, result)

                if agent_name:
                    _update_cooldown(agent_name, ticker)
                    _save_cache(cache_key, user_prompt, result)

                return result
        except Exception as e:
            error_str = str(e)
            print(f"  {provider_name.capitalize()} error: {error_str}")

            if "429" in error_str or "rate_limit" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                _daily_usage[provider_name]["tokens"] = _DAILY_LIMITS[provider_name]["tokens"]
                print(f"  🚫 {provider_name} marked as exhausted for today")

    # 26E: Graceful degradation
    print("  ⚠️ All LLM providers exhausted — skipping reasoning")
    return None


def llm_available():
    return _get_gemini() is not None or _get_groq() is not None or _get_cerebras() is not None


# ============================================
# UTILITY: Get usage stats
# ============================================
def get_llm_stats():
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

    cache_info = {}
    for key, data in _reasoning_cache.items():
        age = (datetime.utcnow() - data["timestamp"]).total_seconds() / 60
        cache_info[key] = {
            "age_minutes": round(age, 1),
            "has_cache": True,
        }

    return {
        "providers": stats,
        "cache": cache_info,
        "date": _get_today(),
        "version": "v2.1",  # 🆕
    }

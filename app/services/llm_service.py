"""
SwingLab LLM Service v3.0 — Multi-provider con fallback reale

Gemini → Groq → Cerebras → cache/skip

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHANGELOG v3.0 (audit fix P0-9)

Il fallback a 3 provider era un'illusione: girava solo Gemini.

1. CEREBRAS non e' MAI stato attivo:
   - cerebras-cloud-sdk non era in requirements.txt (ImportError)
   - CEREBRAS_API_KEY non era dichiarata in config.py, quindi
     getattr(settings, 'CEREBRAS_API_KEY', '') tornava sempre ""

2. GROQ rispondeva 404:
   - llama-3.3-70b-versatile e llama-3.1-8b-instant sono stati
     dismessi. Il modello va tentato da una LISTA, non hardcoded.

3. Nessun retry cross-modello: un 404 bruciava il provider.

Ora ogni provider ha una lista di modelli candidati. Se uno risponde
404/model_not_found si passa al successivo e quello funzionante viene
memorizzato per le chiamate seguenti. I modelli sono override-abili
da env (GEMINI_MODEL / GROQ_MODEL / CEREBRAS_MODEL).

4. Il budget tracker e' in memoria: su Render free lo spin-down lo
   azzera. Aggiunto reset manuale via reset_budget().
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import hashlib
import re
from datetime import datetime
from app.config import settings

# ============================================
# MODELLI CANDIDATI (in ordine di preferenza)
# ============================================
# Se un modello viene dismesso, il codice passa al successivo senza
# bisogno di deploy. Per forzarne uno: env GEMINI_MODEL / GROQ_MODEL / ...

GEMINI_MODELS = [
    "gemini-3.6-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]

GROQ_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "llama3-8b-8192",
]

CEREBRAS_MODELS = [
    "gpt-oss-120b",
    "llama-3.3-70b",
    "llama3.1-8b",
]

# Modello confermato funzionante per provider (cache runtime)
_working_model = {"gemini": None, "groq": None, "cerebras": None}

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
            if settings.GROQ_API_KEY:
                from groq import Groq
                _groq_client = Groq(api_key=settings.GROQ_API_KEY)
                print("  ✅ Groq client initialized")
            else:
                print("  ⚠️ GROQ_API_KEY non configurata")
        except Exception as e:
            print(f"  Groq init error: {e}")
    return _groq_client


def _get_cerebras():
    global _cerebras_client
    if _cerebras_client is None:
        try:
            if settings.CEREBRAS_API_KEY:
                from cerebras.cloud.sdk import Cerebras
                _cerebras_client = Cerebras(api_key=settings.CEREBRAS_API_KEY)
                print("  ✅ Cerebras client initialized")
            else:
                print("  ⚠️ CEREBRAS_API_KEY non configurata")
        except Exception as e:
            print(f"  Cerebras init error: {e}")
    return _cerebras_client


def _is_model_error(err: str) -> bool:
    """True se l'errore riguarda il MODELLO (non la quota o la rete)."""
    e = err.lower()
    return (
        "model_not_found" in e
        or "does not exist" in e
        or "not found" in e
        or "404" in e
        or "decommissioned" in e
        or "unsupported model" in e
        or "invalid model" in e
    )


def _models_for(provider: str):
    """Lista modelli: env override in testa, poi quello gia' funzionante."""
    override = {
        "gemini": settings.GEMINI_MODEL,
        "groq": settings.GROQ_MODEL,
        "cerebras": settings.CEREBRAS_MODEL,
    }.get(provider, "")

    base = {
        "gemini": GEMINI_MODELS,
        "groq": GROQ_MODELS,
        "cerebras": CEREBRAS_MODELS,
    }[provider]

    models = list(base)

    known_good = _working_model.get(provider)
    if known_good and known_good in models:
        models.remove(known_good)
        models.insert(0, known_good)

    if override:
        if override in models:
            models.remove(override)
        models.insert(0, override)

    return models


# ============================================
# PROVIDER CALLS (con model fallback)
# ============================================

def _ask_gemini(system_prompt, user_prompt, max_tokens, temperature):
    client = _get_gemini()
    if not client:
        return None

    last_err = None
    for model in _models_for("gemini"):
        try:
            response = client.models.generate_content(
                model=model,
                contents=f"{system_prompt}\n\n{user_prompt}",
                config={"max_output_tokens": max_tokens, "temperature": temperature},
            )
            text = getattr(response, "text", None)
            if text:
                if _working_model["gemini"] != model:
                    print(f"  🔹 Gemini model attivo: {model}")
                    _working_model["gemini"] = model
                return text
        except Exception as e:
            last_err = str(e)
            if _is_model_error(last_err):
                print(f"  ↪ Gemini: {model} non disponibile, provo il prossimo")
                continue
            raise
    if last_err:
        raise RuntimeError(f"Gemini: nessun modello disponibile ({last_err[:120]})")
    return None


def _ask_groq(system_prompt, user_prompt, max_tokens, temperature):
    client = _get_groq()
    if not client:
        return None

    last_err = None
    for model in _models_for("groq"):
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = response.choices[0].message.content
            if text:
                if _working_model["groq"] != model:
                    print(f"  🔹 Groq model attivo: {model}")
                    _working_model["groq"] = model
                return text
        except Exception as e:
            last_err = str(e)
            if _is_model_error(last_err):
                print(f"  ↪ Groq: {model} non disponibile, provo il prossimo")
                continue
            raise
    if last_err:
        raise RuntimeError(f"Groq: nessun modello disponibile ({last_err[:120]})")
    return None


def _ask_cerebras(system_prompt, user_prompt, max_tokens, temperature):
    client = _get_cerebras()
    if not client:
        return None

    last_err = None
    for model in _models_for("cerebras"):
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = response.choices[0].message.content
            if text:
                if _working_model["cerebras"] != model:
                    print(f"  🔹 Cerebras model attivo: {model}")
                    _working_model["cerebras"] = model
                return text
        except Exception as e:
            last_err = str(e)
            if _is_model_error(last_err):
                print(f"  ↪ Cerebras: {model} non disponibile, provo il prossimo")
                continue
            raise
    if last_err:
        raise RuntimeError(f"Cerebras: nessun modello disponibile ({last_err[:120]})")
    return None


# ============================================
# BUDGET TRACKER
# ============================================
_daily_usage = {
    "gemini": {"tokens": 0, "requests": 0, "date": "", "exhausted": False},
    "groq": {"tokens": 0, "requests": 0, "date": "", "exhausted": False},
    "cerebras": {"tokens": 0, "requests": 0, "date": "", "exhausted": False},
}

_DAILY_LIMITS = {
    "gemini": {"tokens": 800_000, "requests": 1200},
    "groq": {"tokens": 400_000, "requests": 12_000},
    "cerebras": {"tokens": 800_000, "requests": 800},
}

_PROVIDER_ORDER = ["gemini", "groq", "cerebras"]

_COOLDOWN_MINUTES = {
    "macro_analyst": 25,
    "alpha_strategist": 12,
    "risk_manager": 12,
    "executor": 8,
    "apm": 20,
    "system_health": 30,
    "news": 30,
    "default": 10,
    # Analisi per singolo ticker: cooldown piu' lungo (era 2 min, di fatto
    # non throttlava nulla e il loop reasoning bruciava la quota).
    "alpha_strategist_ticker": 45,
}

_reasoning_cache = {}
_last_llm_call = {}


def _get_today():
    return datetime.utcnow().strftime("%Y-%m-%d")


def _reset_daily_if_needed():
    today = _get_today()
    for provider in _daily_usage:
        if _daily_usage[provider]["date"] != today:
            _daily_usage[provider] = {
                "tokens": 0, "requests": 0, "date": today, "exhausted": False
            }


def _estimate_tokens(text):
    return len(text) // 4


def _check_budget(provider, estimated_tokens):
    _reset_daily_if_needed()
    usage = _daily_usage[provider]

    if usage.get("exhausted"):
        return False

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


def reset_budget(provider: str = None):
    """
    Reset manuale del budget tracker.
    Serve quando un provider viene marcato esaurito per un 429 temporaneo:
    il contatore e' in memoria e non si sblocca fino a mezzanotte UTC.
    """
    today = _get_today()
    targets = [provider] if provider else list(_daily_usage.keys())
    for p in targets:
        if p in _daily_usage:
            _daily_usage[p] = {"tokens": 0, "requests": 0, "date": today, "exhausted": False}
    return {"reset": targets, "date": today}


# ============================================
# CACHE / COOLDOWN
# ============================================

def _extract_ticker(user_prompt):
    if not user_prompt:
        return None

    patterns = [
        r'Ticker:\s*([A-Z]{1,6})\b',
        r'candidato\s+BUY\s+([A-Z]{1,6})\b',
        r'BUY\s+([A-Z]{1,6})\s+',
    ]

    for pattern in patterns:
        match = re.search(pattern, user_prompt, re.IGNORECASE)
        if match:
            ticker = match.group(1).upper()
            if 1 <= len(ticker) <= 6 and ticker.isalpha():
                return ticker
    return None


def _build_cache_key(agent_name, user_prompt):
    if not agent_name:
        return None
    ticker = _extract_ticker(user_prompt)
    if ticker:
        return f"{agent_name}:{ticker}"
    return agent_name


def _get_context_hash(user_prompt):
    return hashlib.md5(user_prompt.encode()).hexdigest()


def _check_cache(cache_key, user_prompt):
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
    if not cache_key:
        return
    _reasoning_cache[cache_key] = {
        "hash": _get_context_hash(user_prompt),
        "reasoning": reasoning,
        "timestamp": datetime.utcnow(),
    }


def _get_cooldown_key(agent_name, ticker):
    if ticker:
        return f"{agent_name}_ticker"
    return agent_name


def _check_cooldown(agent_name, ticker=None):
    cooldown_key = _get_cooldown_key(agent_name, ticker)
    cooldown = _COOLDOWN_MINUTES.get(cooldown_key, _COOLDOWN_MINUTES["default"])

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
    tracking_key = f"{agent_name}:{ticker}" if ticker else agent_name
    _last_llm_call[tracking_key] = datetime.utcnow()


# ============================================
# MAIN
# ============================================

def llm_ask(system_prompt, user_prompt, max_tokens=300, temperature=0.3, agent_name=None):
    """
    Prova i provider in ordine: Gemini → Groq → Cerebras.
    Ogni provider prova la sua lista di modelli finche' uno risponde.
    Degradazione morbida: se nessuno risponde ritorna None, mai eccezioni.
    """
    ticker = _extract_ticker(user_prompt)
    cache_key = _build_cache_key(agent_name, user_prompt)

    if cache_key:
        cached = _check_cache(cache_key, user_prompt)
        if cached:
            return cached

    if agent_name and not _check_cooldown(agent_name, ticker):
        if ticker:
            print(f"  ⏭ Skipping LLM for {agent_name}:{ticker} (cooldown)")
            return None
        if cache_key in _reasoning_cache:
            print(f"  💾 Using stale cache for {cache_key} (cooldown active)")
            return _reasoning_cache[cache_key]["reasoning"]
        return None

    input_text = system_prompt + user_prompt
    estimated_tokens = _estimate_tokens(input_text) + max_tokens

    providers = [
        ("gemini", _ask_gemini),
        ("groq", _ask_groq),
        ("cerebras", _ask_cerebras),
    ]

    for provider_name, ask_fn in providers:
        if not _check_budget(provider_name, estimated_tokens):
            print(f"  ⚠️ {provider_name} non disponibile (budget/esaurito), skip")
            continue

        try:
            result = ask_fn(system_prompt, user_prompt, max_tokens, temperature)
            if result:
                print(f"  🧠 LLM via {provider_name.capitalize()}" + (f" ({ticker})" if ticker else ""))
                _track_usage(provider_name, input_text, result)
                if agent_name:
                    _update_cooldown(agent_name, ticker)
                    _save_cache(cache_key, user_prompt, result)
                return result
        except Exception as e:
            error_str = str(e)
            print(f"  {provider_name.capitalize()} error: {error_str[:160]}")

            if "429" in error_str or "rate_limit" in error_str.lower() or "RESOURCE_EXHAUSTED" in error_str:
                _daily_usage[provider_name]["exhausted"] = True
                print(f"  🚫 {provider_name} esaurito per oggi (rate limit)")
            elif _is_model_error(error_str):
                print(f"  🚫 {provider_name}: nessun modello valido nella lista")

    print("  ⚠️ Nessun provider LLM disponibile — reasoning saltato")
    return None


def llm_available():
    return (
        _get_gemini() is not None
        or _get_groq() is not None
        or _get_cerebras() is not None
    )


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
            "exhausted": usage.get("exhausted", False),
            "available": _check_budget(provider, 500),
            "client_ok": {
                "gemini": _get_gemini() is not None,
                "groq": _get_groq() is not None,
                "cerebras": _get_cerebras() is not None,
            }[provider],
            "active_model": _working_model.get(provider),
            "candidate_models": _models_for(provider),
        }

    cache_info = {}
    for key, data in _reasoning_cache.items():
        age = (datetime.utcnow() - data["timestamp"]).total_seconds() / 60
        cache_info[key] = {"age_minutes": round(age, 1), "has_cache": True}

    return {
        "providers": stats,
        "cache": cache_info,
        "cooldowns": _COOLDOWN_MINUTES,
        "date": _get_today(),
        "version": "v3.0",
    }

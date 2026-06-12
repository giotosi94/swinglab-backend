"""
SwingLab LLM Service — Multi-provider with fallback
Gemini → Groq → skip
"""

import time
from app.config import settings

_gemini_client = None
_groq_client = None


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


def llm_ask(system_prompt, user_prompt, max_tokens=300, temperature=0.3):
    """Try Gemini first, then Groq as fallback."""
    
    # Try Gemini
    try:
        result = _ask_gemini(system_prompt, user_prompt, max_tokens, temperature)
        if result:
            print("  🧠 LLM response via Gemini")
            return result
    except Exception as e:
        print(f"  Gemini error: {e}")

    # Fallback: Groq
    try:
        result = _ask_groq(system_prompt, user_prompt, max_tokens, temperature)
        if result:
            print("  🧠 LLM response via Groq")
            return result
    except Exception as e:
        print(f"  Groq error: {e}")

    print("  ⚠️ All LLM providers failed")
    return None


def llm_available():
    return _get_gemini() is not None or _get_groq() is not None

"""
SwingLab LLM Service — Google Gemini Integration
Provides reasoning capabilities to agents.
"""

import time
from app.config import settings

_client = None


def _get_client():
    global _client
    if _client is None:
        try:
            from google import genai
            if settings.GEMINI_API_KEY:
                _client = genai.Client(api_key=settings.GEMINI_API_KEY)
                print("  ✅ Gemini LLM client initialized")
            else:
                print("  ⚠️ GEMINI_API_KEY not set")
        except Exception as e:
            print(f"  LLM init error: {e}")
    return _client


def llm_ask(system_prompt, user_prompt, max_tokens=300, temperature=0.3, retries=2):
    client = _get_client()
    if not client:
        print("  ⚠️ LLM client not available")
        return None
    
    for attempt in range(retries + 1):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=f"{system_prompt}\n\n{user_prompt}",
                config={
                    "max_output_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
            return response.text
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                if attempt < retries:
                    wait = 20 * (attempt + 1)
                    print(f"  ⏳ LLM rate limited, waiting {wait}s (attempt {attempt + 1}/{retries + 1})")
                    time.sleep(wait)
                    continue
            print(f"  LLM error: {e}")
            return None
    return None


def llm_available():
    return _get_client() is not None

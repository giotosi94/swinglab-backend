"""
SwingLab LLM Service — Google Gemini Integration
Provides reasoning capabilities to agents.
"""

from app.config import settings

_client = None


def _get_client():
    global _client
    if _client is None:
        try:
            from google import genai
            if settings.GEMINI_API_KEY:
                _client = genai.Client(api_key=settings.GEMINI_API_KEY)
        except Exception as e:
            print(f"  LLM init error: {e}")
    return _client


def llm_ask(system_prompt, user_prompt, max_tokens=300, temperature=0.3):
    client = _get_client()
    if not client:
        return None
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
        print(f"  LLM error: {e}")
        return None


def llm_available():
    return _get_client() is not None

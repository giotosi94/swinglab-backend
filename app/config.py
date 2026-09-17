from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MONGODB_URL: str = "mongodb://localhost:27017"
    DB_NAME: str = "swinglab"

    TWELVEDATA_API_KEY: str = ""

    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""

    ALPACA_API_KEY: str = ""
    ALPACA_SECRET_KEY: str = ""
    ALPACA_PAPER: bool = True

    # ===== LLM providers =====
    # NOTA: CEREBRAS_API_KEY mancava qui. Anche se la variabile era
    # impostata su Render, pydantic-settings non la esponeva e
    # getattr(settings, 'CEREBRAS_API_KEY', '') tornava sempre vuota:
    # il terzo provider non e' mai stato inizializzato.
    GEMINI_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    CEREBRAS_API_KEY: str = ""

    # Modelli override-abili da env senza toccare il codice.
    # Utile quando un provider dismette un modello (succede spesso).
    GEMINI_MODEL: str = ""
    GROQ_MODEL: str = ""
    CEREBRAS_MODEL: str = ""

    class Config:
        env_file = ".env"


settings = Settings()

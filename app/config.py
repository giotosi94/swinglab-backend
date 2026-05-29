from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MONGODB_URL: str = "mongodb://localhost:27017"
    DB_NAME: str = "swinglab"
    YFINANCE_ENABLED: bool = True

    class Config:
        env_file = ".env"

settings = Settings()

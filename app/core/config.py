import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    """애플리케이션 설정"""

    # 기본 설정
    APP_NAME: str = "AI Multimodal Service"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"

    # 서버 설정
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    # 데이터 설정
    # DATA_PATH: str = os.path.join(BASE_PATH, "data")

    # 데이터베이스 설정
    DATABASE_URL: str = "postgresql+asyncpg://postgres:admin@localhost:5432/chatbot"

    # 디렉토리 설정
    BASE_PATH: str = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    STATIC_IMAGE_PATH: str = os.path.join(BASE_PATH, "static", "images")
    GENERATED_IMAGE_PATH: str = os.path.join(BASE_PATH, "generated", "images")

    # 모델 설정
    TEXT_MODEL_PATH: str = os.path.join(BASE_PATH, "models", "text", "DeepSeek-R1-GGUF")
    IMAGE_MODEL_PATH: str = os.path.join(BASE_PATH, "models", "image", "Janus-Pro-1B")
    VIDEO_MODEL_PATH: str = os.path.join(BASE_PATH, "models", "video")

    # CORS 설정
    CORS_ORIGINS: List[str] = ["*"]

    # JWT 설정
    # SECRET_KEY: str = "your-secret-key"
    # ALGORITHM: str = "HS256"
    # ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # 하드웨어 설정
    USE_CUDA: bool = True
    MAX_CONCURRENT_USERS: int = 10

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


# 설정 인스턴스 생성
settings = Settings()

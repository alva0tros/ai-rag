import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

# 로컬 DB
DATABASE_URL = "postgresql+asyncpg://postgres:admin@localhost:5432/chatbot"

# 외부 DB
# DATABASE_URL = "postgresql+asyncpg://admin:admin@smango.iptime.org:5432/chatbot"

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=True,
    pool_pre_ping=True,
    pool_size=20,
    max_overflow=10,
)
async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

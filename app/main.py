import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.chat import chat, history as chat_history
from app.api.v1.image import image
from app.core.config import settings


print("BASE_PATH : ", settings.BASE_PATH)

app = FastAPI()
static_path = os.path.join(settings.BASE_PATH, "static")
generated_path = os.path.join(settings.BASE_PATH, "generated")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영환경에서는 구체적인 origin을 지정하세요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 라우터 등록 (예: /api/v1/chat, /api/v1/stop)
app.include_router(chat.router, prefix=settings.API_PREFIX)
app.include_router(chat_history.router, prefix=settings.API_PREFIX)

app.include_router(image.router, prefix=settings.API_PREFIX)

# static 폴더 전체를 마운트합니다.
app.mount("/static", StaticFiles(directory=static_path), name="static")
app.mount("/generated", StaticFiles(directory=generated_path), name="generated")

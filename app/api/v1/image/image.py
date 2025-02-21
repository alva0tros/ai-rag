import logging
import os
import asyncio

from fastapi import APIRouter, Request, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.services.image import image_service
from app.services.chat import chat_service

from app.services.image.image_generator import run_generate

from config import STATIC_IMAGE_PATH

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/image/prompt")
async def prompt(request: Request):
    # 요청 본문 파싱
    try:
        user_message = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid JSON request")
    print("user_message : ", user_message)

    # 이미지 생성 과정을 비동기적으로 실행하여 SSE 이벤트로 상태를 전송하는 제너레이터 함수
    async def event_generator():
        yield {"event": "start", "data": "Image generation started."}
        loop = asyncio.get_event_loop()
        # image_generate()는 동기 함수이므로 run_in_executor로 별도 쓰레드에서 실행
        await loop.run_in_executor(None, run_generate)
        yield {"event": "done", "data": "Image generation completed."}

    return EventSourceResponse(event_generator())


@router.get("/image/intro")
async def get_intro_images():

    if not os.path.isdir(STATIC_IMAGE_PATH):
        raise HTTPException(
            status_code=404, detail="Static images directory not found."
        )

    # 이미지 확장자 필터링
    valid_extensions = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")
    image_files = [
        file
        for file in os.listdir(STATIC_IMAGE_PATH)
        if file.lower().endswith(valid_extensions)
    ]

    image_urls = [f"/static/images/{file}" for file in image_files]

    return {"images": image_urls}


@router.post("/image/stop")
async def stop_image(request: Request):
    data = await request.json()
    conversation_id = data.get("conversation_id", "default")

    if conversation_id not in image_service.tasks:
        raise HTTPException(
            status_code=404, detail="No active task found for this session."
        )

    task = image_service.tasks[conversation_id]
    task.cancel()  # 작업 취소
    return {"message": f"Task for session {conversation_id} has been cancelled."}

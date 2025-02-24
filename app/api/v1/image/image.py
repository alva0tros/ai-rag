import logging
import os
import io

from fastapi import APIRouter, Request, UploadFile, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.services.image import image_service
from app.services.chat import chat_service

from fastapi.responses import JSONResponse, StreamingResponse
from app.services.image.image_generator import (
    ImageGenerator,
    multimodal_understanding,
    generate_image,
)


from config import STATIC_IMAGE_PATH

router = APIRouter()
image_generator = ImageGenerator()
logger = logging.getLogger(__name__)


@router.post("/image/prompt")
async def prompt(request: Request):
    # 요청 본문 파싱
    try:
        user_message = await request.json()
        message = user_message.get("message")
        images = generate_image(image_generator, message, None, 5.0)

        def image_stream():
            for img in images:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                buf.seek(0)
                yield buf.read()

        return StreamingResponse(image_stream(), media_type="multipart/related")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Image generation failed: {str(e)}"
        )


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

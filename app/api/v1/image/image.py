import asyncio

import logging
import os
import json

from nanoid import generate as nanoid
from fastapi import APIRouter, Request, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.services.image_service import image_service, generate_image
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


# SSE로 진행률 스트리밍
async def progress_stream(
    prompt: str,
    seed: int,
    guidance: float,
    user_id: int,
    conversation_id: str,
    message_id: str,
):

    # 세션별로 태스크 지정
    task = {
        "images": None,
        "progress": 0.0,
        "progress_event": asyncio.Event(),
        "last_reported_progress": -1,
        "generate_task": None,
    }

    async with asyncio.Lock():  # 동시 쓰기 방지
        image_service.tasks[conversation_id] = task

    # 진행률 콜백 함수
    def update_progress(p):
        task["progress"] = p
        task["progress_event"].set()
        logger.info(f"Conversation {conversation_id}: Progress updated to {p}%")

    # 이미지 서비스 콜백 설정
    image_service.progress_callback = update_progress

    # try:
    #     image_service = ImageService(progress_callback=update_progress)
    # except Exception as e:
    #     logger.error(f"ImageGenerator 초기화 실패: {str(e)}")
    #     raise HTTPException(
    #         status_code=500, detail=f"이미지 생성기 초기화 실패: {str(e)}"
    #     )

    async def generate_images():
        try:
            task["images"] = await asyncio.get_event_loop().run_in_executor(
                None, lambda: generate_image(prompt, seed, guidance)
            )
        except Exception as e:
            logger.error(
                f"Conversation {conversation_id}: Image generation failed - {str(e)}"
            )
            raise

    async def event_generator():

        # 이미지 생성 작업을 백그라운드로 실행
        task["generate_task"] = asyncio.create_task(generate_images())

        # 진행률 스트리밍 (1% 단위로만 전송)
        while task["progress"] < 100.0:
            await task["progress_event"].wait()
            current_progress = round(
                task["progress"], 0
            )  # 소수점 버림으로 1% 단위로 조정
            if current_progress > task["last_reported_progress"]:  # 1% 증가 시에만 전송
                task["last_reported_progress"] = current_progress
                yield {"event": "progress", "data": f"{current_progress}"}
            task["progress_event"].clear()  # 이벤트 리셋
            await asyncio.sleep(0.1)  # 0.1초마다 진행률 확인

        try:
            # 이미지 생성 완료 대기
            await task["generate_task"]

            # 최종 100% 전송 보장
            if task["last_reported_progress"] != 100:
                yield {"event": "progress", "data": "100"}

            # 이미지 저장 및 URL 생성
            image_urls = []
            save_dir = os.path.join(
                settings.GENERATED_IMAGE_PATH, str(user_id), conversation_id, message_id
            )
            os.makedirs(save_dir, exist_ok=True)

            for i, img in enumerate(task["images"]):
                file_name = f"img{i}.png"
                file_path = os.path.join(save_dir, file_name)
                img.save(file_path, format="PNG")
                image_url = f"/generated/images/{user_id}/{conversation_id}/{message_id}/{file_name}"
                image_urls.append(image_url)

            yield {
                "event": "image",
                "data": json.dumps({"image_urls": image_urls}),
            }
        except Exception as e:
            logger.error(f"Error in event generator: {str(e)}")
            yield {"event": "error", "data": str(e)}
        finally:
            # 작업 정리 및 메모리 해제
            if conversation_id in image_service.tasks:
                del image_service.tasks[conversation_id]

    return EventSourceResponse(event_generator())


@router.post("/image/prompt")
async def prompt(request: Request):
    try:
        user_message = await request.json()
        message = user_message.get("message")
        user_id = user_message.get("user_id")
        conversation_id = user_message.get("conversation_id")
        message_id = user_message.get("message_id")

        if not message or not user_id or not message_id:
            logger.error("Missing required fields")
            raise HTTPException(status_code=400, detail="Missing required fields")

        # conversation_id가 없는 경우 새로운 ID 생성
        is_new_conversation = not conversation_id
        if is_new_conversation:
            # conversation_id = str(uuid.uuid4())
            conversation_id = nanoid(size=12)

        print(f"Received prompt: {user_message}")
        return await progress_stream(
            message, None, 5.0, user_id, conversation_id, message_id
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Image generation failed: {str(e)}"
        )


@router.get("/image/intro")
async def get_intro_images():

    if not os.path.isdir(settings.STATIC_IMAGE_PATH):
        raise HTTPException(
            status_code=404, detail="Static images directory not found."
        )

    # 이미지 확장자 필터링
    valid_extensions = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")
    image_files = [
        file
        for file in os.listdir(settings.STATIC_IMAGE_PATH)
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

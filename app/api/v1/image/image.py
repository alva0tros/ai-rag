"""
이미지 생성 API 엔드포인트 모듈

이 모듈은 이미지 생성 요청 처리, 진행 상황 추적, 이미지 관리를 위한 라우트를 제공합니다.
텍스트 프롬프트를 기반으로 이미지를 생성하고, 이미지 생성 과정을 스트리밍하는 기능을 포함합니다.
"""

import logging
import os
from typing import Dict, List

from nanoid import generate as nanoid
from fastapi import APIRouter, Request, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.services.image.image_service import image_service
from app.services.image.image_prompt import image_prompt
from app.core.config import settings

# 라우터 및 로거 설정
router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/image/prompt")
async def generate_from_prompt(request: Request) -> EventSourceResponse:
    """
    텍스트 프롬프트를 기반으로 이미지 생성

    Args:
        request: 프롬프트와 메타데이터를 포함한 FastAPI 요청 객체

    Returns:
        EventSourceResponse: 진행 상황 이벤트와 이미지 URL을 스트리밍함
    """
    try:
        # 요청 본문 파싱
        user_message = await request.json()

        # 필수 필드 추출
        message = user_message.get("message")
        user_id = user_message.get("user_id")
        conversation_id = user_message.get("conversation_id")
        message_id = user_message.get("message_id")

        # 필수 필드 검증
        if not message or not user_id or not message_id:
            logger.error("Missing required fields")
            raise HTTPException(status_code=400, detail="Missing required fields")

        # 대화 ID가 없으면 새로 생성
        if not conversation_id:
            conversation_id = nanoid(size=12)
            logger.info(f"Generated new conversation ID: {conversation_id}")

        logger.info(f"Received image generation prompt from user {user_id}")

        # 프롬프트 처리 (번역 및 향상)
        enhanced_prompt = await image_prompt.translate_and_enhance(message)
        logger.info(f"Processing image generation prompt from user {user_id}")

        # 서비스를 사용하여 이미지 생성 과정과 결과 스트리밍
        return EventSourceResponse(
            image_service.stream_generation_progress(
                enhanced_prompt, None, 5.0, user_id, conversation_id, message_id
            )
        )

    except HTTPException:
        # HTTP 예외는 그대로 전달
        raise

    except Exception as e:
        logger.exception(f"Image generation failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Image generation failed: {str(e)}"
        )


@router.get("/image/intro")
async def get_intro_images() -> Dict[str, List[str]]:
    """
    소개용 정적 이미지 목록 가져오기

    Returns:
        Dict[str, List[str]]: 이미지 URL을 포함하는 딕셔너리
    """
    try:
        # 정적 이미지 디렉토리 확인
        if not os.path.isdir(settings.STATIC_IMAGE_PATH):
            logger.error("Static images directory not found")
            raise HTTPException(
                status_code=404, detail="Static images directory not found."
            )

        # 확장자로 이미지 파일 필터링
        valid_extensions = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")

        try:
            image_files = [
                file
                for file in os.listdir(settings.STATIC_IMAGE_PATH)
                if file.lower().endswith(valid_extensions)
            ]
        except Exception as e:
            logger.error(f"Error reading static images directory: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error reading static images directory: {str(e)}",
            )

        # 이미지 URL 생성
        image_urls = [f"/static/images/{file}" for file in image_files]
        logger.info(f"Retrieved {len(image_urls)} intro images")

        return {"images": image_urls}

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(f"Error retrieving intro images: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve intro images: {str(e)}"
        )


@router.post("/image/stop")
async def stop_image_generation(request: Request) -> Dict[str, str]:
    """
    진행 중인 이미지 생성 과정 중지

    Args:
        request: conversation_id를 포함한 FastAPI 요청 객체

    Returns:
        Dict[str, str]: 상태 메시지
    """
    try:
        data = await request.json()
        conversation_id = data.get("conversation_id", "default")

        logger.info(f"Requested to stop image generation for {conversation_id}")

        # 이미지 생성 중지 서비스 호출
        success = await image_service.stop_image_generation(conversation_id)

        if not success:
            logger.warning(f"No active task found for session: {conversation_id}")
            raise HTTPException(
                status_code=404, detail="No active task found for this session."
            )

        logger.info(f"Image generation stopped for conversation: {conversation_id}")

        return {"message": f"세션 {conversation_id}에 대한 작업이 취소되었습니다."}

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(f"Error stopping image generation: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to stop image generation: {str(e)}"
        )

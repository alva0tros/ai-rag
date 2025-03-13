"""
채팅 관련 API 엔드포인트를 정의하는 모듈
"""

import asyncio

# import uuid
from nanoid import generate as nanoid
import logging
from typing import Optional

from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel

from app.services.chat.chat_service import ChatService

# ==========================================================
# 기본 설정
# ==========================================================

# 라우터 및 로거 설정
router = APIRouter()
logger = logging.getLogger(__name__)

# 서비스 인스턴스 생성
chat_service = ChatService()

# ==========================================================
# 요청 모델
# ==========================================================


class ChatRequest(BaseModel):
    """채팅 요청 모델"""

    message: str
    user_id: Optional[int] = None
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None


class StopChatRequest(BaseModel):
    """채팅 중지 요청 모델"""

    conversation_id: str


# ==========================================================
# API 엔드포인트
# ==========================================================


@router.post("/chat/message")
async def chat(request: Request):
    """
    채팅 메시지를 처리하고 응답을 스트리밍합니다.
    """
    # 요청 본문 파싱
    try:
        data = await request.json()
        chat_request = ChatRequest(**data)
    except Exception as e:
        logger.error(f"Invalid request: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON request")

    logger.info(f"Chat request received: {chat_request.message[:50]}...")

    # conversation_id가 없는 경우 새로운 ID 생성
    is_new_conversation = not chat_request.conversation_id
    conversation_id = chat_request.conversation_id or nanoid(size=12)

    # message_id가 없는 경우 새로운 ID 생성
    message_id = chat_request.message_id or nanoid(size=12)

    # 응답 큐 생성
    queue = asyncio.Queue()

    async def event_generator():
        try:
            # 대화 처리 서비스 호출
            async for (
                event
            ) in chat_service.conversation_handler.handle_chat_conversation(
                queue=queue,
                message=chat_request.message,
                conversation_id=conversation_id,
                message_id=message_id,
                user_id=chat_request.user_id,
                is_new_conversation=is_new_conversation,
            ):
                yield event

        except asyncio.CancelledError:
            logger.warning(f"Task for session {conversation_id} was cancelled.")
            yield await chat_service.event_manager.create_sse_event(
                "error", {"text": "Task was cancelled."}
            )

        except Exception as e:
            logger.exception(f"Error in session {conversation_id}: {e}")
            yield await chat_service.event_manager.create_sse_event(
                "error", {"text": str(e)}
            )

        finally:
            # 작업 제거 및 리소스 정리
            chat_service.tasks.pop(conversation_id, None)
            logger.info(f"Cleaned up session: {conversation_id}")

    return EventSourceResponse(event_generator())


@router.post("/chat/stop")
async def stop_chat(request: Request, background_tasks: BackgroundTasks):
    """
    진행 중인 채팅 세션을 중지합니다.
    """
    try:
        data = await request.json()
        stop_request = StopChatRequest(**data)
    except Exception as e:
        logger.error(f"Invalid stop request: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON request")

    conversation_id = stop_request.conversation_id

    if conversation_id not in chat_service.tasks:
        logger.warning(f"No active task found for session: {conversation_id}")
        raise HTTPException(
            status_code=404, detail="No active task found for this session."
        )

    # 작업 취소
    task = chat_service.tasks[conversation_id]
    task.cancel()

    # 백그라운드에서 세션 리소스 정리
    background_tasks.add_task(
        chat_service.resource_manager.cleanup_session_resources, conversation_id
    )

    return {"message": f"Task for session {conversation_id} has been cancelled."}

import logging

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from app.db.repositories import chat_repository


router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/chat/sessions/{user_id}")
async def get_all_chat_sessions(user_id):
    try:
        chat_sessions = await chat_repository.get_all_chat_sessions(user_id)
        # sessions_data = [
        #     {
        #         "session_id": str(session.session_id),
        #         "user_id": session.user_id,
        #         "title": session.title,
        #     }
        #     for session in sessions
        # ]
        return chat_sessions
    except Exception as e:
        logger.exception("Error fetching chat sessions: %s", e)
        raise HTTPException(status_code=500, detail="Error fetching chat sessions")


@router.get("/chat/messages/{session_id}")
async def get_chat_messages(session_id, limit=100, offset=0):
    messages = await chat_repository.get_chat_messages(session_id, limit, offset)
    messages_data = [
        {
            "session_id": str(msg.session_id),
            "message_id": str(msg.message_id),
            "user_message": msg.user_message,
            "main_message": msg.main_message,
            "think_message": msg.think_message,
            "think_time": msg.think_time,
            "liked": msg.liked,
            "disliked": msg.disliked,
            "dislike_feedback": msg.dislike_feedback,
        }
        for msg in messages
    ]
    return messages_data


@router.delete("/chat/delete_session/{session_id}")
async def delete_chat_session(session_id: str):
    try:
        await chat_repository.delete_chat_session(session_id)
        return {"message": f"Session {session_id} and its messages have been deleted."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/chat/update_session/{session_id}")
async def update_chat_session_title(
    session_id: str, new_title: str = Body(..., embed=True)
):
    try:
        result = await chat_repository.update_chat_session_title(session_id, new_title)

        if not result:
            raise HTTPException(status_code=404, detail="Chat session not found")

        return {
            "message": "Chat session title updated successfully",
            "updated_title": new_title,
        }
    except Exception as e:
        logger.exception("Error updating chat session title: %s", e)
        raise HTTPException(status_code=500, detail="Error updating chat session title")


# Pydantic 모델 정의: 업데이트할 채팅 메시지 필드를 정의
class UpdateChatMessageRequest(BaseModel):
    liked: bool = None
    disliked: bool = None
    dislike_feedback: str = None


@router.patch("/chat/update_message/{session_id}/{message_id}")
async def update_chat_message(
    session_id: str,
    message_id: str,
    request: UpdateChatMessageRequest = Body(...),
):
    try:
        result = await chat_repository.update_chat_message(
            session_id,
            message_id,
            liked=request.liked,
            disliked=request.disliked,
            dislike_feedback=request.dislike_feedback,
        )
        if not result:
            raise HTTPException(status_code=404, detail="Chat message not found")
        return {
            "message": "Chat message updated successfully",
            "data": {
                "liked": result.liked,
                "disliked": result.disliked,
                "dislike_feedback": result.dislike_feedback,
            },
        }
    except Exception as e:
        logger.exception("Error updating chat message: %s", e)
        raise HTTPException(status_code=500, detail="Error updating chat message")

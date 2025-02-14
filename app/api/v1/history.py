import logging

from fastapi import APIRouter, HTTPException
from app.services import chat_crud
from app.infrastructure.crud_db.models import ChatSession, ChatMessage

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/sessions/{user_id}")
async def get_all_sessions(user_id):
    try:
        chat_sessions = await chat_crud.get_all_chat_sessions(user_id)
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


@router.get("/messages/{session_id}")
async def get_messages(session_id, limit=100, offset=0):
    messages = await chat_crud.get_chat_messages(session_id, limit, offset)
    messages_data = [
        {
            "session_id": str(msg.session_id),
            "message_id": str(msg.message_id),
            "user_message": msg.user_message,
            "main_message": msg.main_message,
            "think_message": msg.think_message,
        }
        for msg in messages
    ]
    return messages_data


@router.delete("/delete_session/{session_id}")
async def delete_session(session_id: str):
    try:
        await chat_crud.delete_chat_session(session_id)
        return {"message": f"Session {session_id} and its messages have been deleted."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

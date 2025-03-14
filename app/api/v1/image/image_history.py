import logging

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from app.db.repositories import image_repository


router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/image/sessions/{user_id}")
async def get_all_image_sessions(user_id):
    try:
        image_sessions = await image_repository.get_all_image_sessions(user_id)
        # sessions_data = [
        #     {
        #         "session_id": str(session.session_id),
        #         "user_id": session.user_id,
        #         "title": session.title,
        #     }
        #     for session in sessions
        # ]
        return image_sessions
    except Exception as e:
        logger.exception("Error fetching image sessions: %s", e)
        raise HTTPException(status_code=500, detail="Error fetching image sessions")


@router.get("/image/messages/{session_id}")
async def get_image_messages(session_id, limit=100, offset=0):
    messages = await image_repository.get_image_messages(session_id, limit, offset)
    messages_data = [
        {
            "session_id": str(msg["session_id"]),
            "message_id": str(msg["message_id"]),
            "user_message": msg["user_message"],
            "image_prompt": msg["image_prompt"],
            "image_urls": msg["image_urls"],
            "liked": msg["liked"],
            "disliked": msg["disliked"],
            "dislike_feedback": msg["dislike_feedback"]
        }
        for msg in messages
    ]
    return messages_data


@router.delete("/image/delete_session/{session_id}")
async def delete_image_session(session_id: str):
    try:
        await image_repository.delete_image_session(session_id)
        return {"message": f"Session {session_id} and its messages have been deleted."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/image/update_session/{session_id}")
async def update_image_session_title(
    session_id: str, new_title: str = Body(..., embed=True)
):
    try:
        result = await image_repository.update_image_session_title(
            session_id, new_title
        )

        if not result:
            raise HTTPException(status_code=404, detail="Image session not found")

        return {
            "message": "Image session title updated successfully",
            "updated_title": new_title,
        }
    except Exception as e:
        logger.exception("Error updating image session title: %s", e)
        raise HTTPException(
            status_code=500, detail="Error updating image session title"
        )


# Pydantic 모델 정의: 업데이트할 채팅 메시지 필드를 정의
class UpdateImageMessageRequest(BaseModel):
    liked: bool = None
    disliked: bool = None
    dislike_feedback: str = None


@router.post("/image/update_message/{session_id}/{message_id}")
async def update_message(
    session_id: str,
    message_id: str,
    request: UpdateImageMessageRequest = Body(...),
):
    try:
        result = await image_repository.update_image_message(
            session_id,
            message_id,
            liked=request.liked,
            disliked=request.disliked,
            dislike_feedback=request.dislike_feedback,
        )
        if not result:
            raise HTTPException(status_code=404, detail="Image message not found")
        return {
            "message": "Image message updated successfully",
            "data": {
                "liked": result.liked,
                "disliked": result.disliked,
                "dislike_feedback": result.dislike_feedback,
            },
        }
    except Exception as e:
        logger.exception("Error updating image message: %s", e)
        raise HTTPException(status_code=500, detail="Error updating image message")

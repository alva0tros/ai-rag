import logging
import datetime
from sqlalchemy import select, delete

from app.infrastructure.crud_db.session import async_session
from app.infrastructure.crud_db.models import ChatSession, ChatMessage


logger = logging.getLogger(__name__)


# 생성(Create) 관련 함수
async def create_chat_session(session_id, title, user_id):
    async with async_session() as session:
        async with session.begin():
            new_session = ChatSession(
                session_id=session_id, title=title, user_id=user_id
            )
            session.add(new_session)
    logger.info("Created chat session for session_id: %s", session_id)


async def create_chat_message(
    session_id, message_id, user_message, main_message, think_message
):
    print("think_message : ", think_message)
    async with async_session() as session:
        async with session.begin():
            new_message = ChatMessage(
                session_id=session_id,
                message_id=message_id,
                user_message=user_message,
                main_message=main_message,
                think_message=think_message,
            )
            session.add(new_message)
    logger.info("Created chat message for session_id: %s", session_id)


# 조회(Read) 관련 함수
async def get_all_chat_sessions(user_id: int):
    async with async_session() as session:
        result = await session.execute(
            select(ChatSession)
            .where(ChatSession.user_id == int(user_id))
            .order_by(ChatSession.created_at.desc())
        )
        sessions = result.scalars().all()
        logger.debug("Fetched %d chat sessions for user_id: %d", len(sessions), user_id)

    # 날짜 기준 계산 (로컬 타임존 기준, 필요 시 UTC로 변경)
    now = datetime.datetime.now()
    today = datetime.datetime.combine(now.date(), datetime.time())
    yesterday = today - datetime.timedelta(days=1)
    last7days = today - datetime.timedelta(days=7)
    last30days = today - datetime.timedelta(days=30)

    # 그룹 초기화
    chat_sessions = {
        "today": [],
        "yesterday": [],
        "last7days": [],
        "last30days": [],
        "older": [],
    }

    # 각 세션의 created_at 값을 기준으로 그룹 분류
    for session_obj in sessions:
        created_at = session_obj.created_at

        if created_at >= today:
            chat_sessions["today"].append(session_obj)
        elif created_at >= yesterday:
            chat_sessions["yesterday"].append(session_obj)
        elif created_at >= last7days:
            chat_sessions["last7days"].append(session_obj)
        elif created_at >= last30days:
            chat_sessions["last30days"].append(session_obj)
        else:
            chat_sessions["older"].append(session_obj)

    return chat_sessions


async def get_chat_messages(session_id, limit=100, offset=0):
    async with async_session() as session:
        result = await session.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.asc())
            .offset(offset)
            .limit(limit)
        )
        messages = result.scalars().all()
        logger.debug(
            "Fetched %d messages for session_id: %s", len(messages), session_id
        )
        return messages


# 삭제(Delete) 관련 함수
async def delete_chat_session(session_id):
    """
    주어진 session_id에 해당하는 chat_sessions 레코드와 연관된 chat_messages 레코드를 삭제합니다.
    """
    async with async_session() as session:
        async with session.begin():
            # 먼저 chat_messages 삭제 (CASCADE 설정이 없다면 직접 삭제)
            await session.execute(
                delete(ChatMessage).where(ChatMessage.session_id == session_id)
            )
            # chat_sessions 삭제
            await session.execute(
                delete(ChatSession).where(ChatSession.session_id == session_id)
            )
    logger.info("Deleted chat session and its messages for session_id: %s", session_id)

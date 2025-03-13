import logging
import datetime
from sqlalchemy import select, delete

from app.db.session import async_session
from app.db.entities.image_entity import ImageSession, ImageMessage, ImageMessageUrl


logger = logging.getLogger(__name__)


# 생성(Create) 관련 함수
async def create_image_session(session_id, user_id, title):
    async with async_session() as session:
        async with session.begin():
            new_session = ImageSession(
                session_id=session_id, user_id=user_id, title=title
            )
            session.add(new_session)
    logger.info("Created image session for session_id: %s", session_id)


async def create_image_message(session_id, message_id, user_message, image_prompt):
    async with async_session() as session:
        async with session.begin():
            # session_id와 message_id로 기존 메시지 조회
            result = await session.execute(
                select(ImageMessage).where(
                    ImageMessage.session_id == session_id,
                    ImageMessage.message_id == message_id,
                )
            )
            message_obj = result.scalars().first()

            if message_obj:
                # 기존 메시지가 있다면 업데이트
                message_obj.image_prompt = image_prompt
                logger.info(
                    "Updated image message for session_id: %s, message_id: %s",
                    session_id,
                    message_id,
                )

            else:
                # 기존 메시지가 없으면 새 메시지 생성
                new_message = ImageMessage(
                    session_id=session_id,
                    message_id=message_id,
                    user_message=user_message,
                    image_prompt=image_prompt,
                )
                session.add(new_message)
                logger.info("Created image message for session_id: %s", session_id)


async def create_image_message_url(session_id, message_id, image_seq, image_url):
    """
    이미지 메시지 URL을 저장합니다.
    """
    async with async_session() as session:
        async with session.begin():
            # 이미 존재하는지 확인
            result = await session.execute(
                select(ImageMessageUrl).where(
                    ImageMessageUrl.session_id == session_id,
                    ImageMessageUrl.message_id == message_id,
                    ImageMessageUrl.image_seq == image_seq,
                )
            )
            url_obj = result.scalars().first()

            if url_obj:
                # 기존 URL이 있다면 업데이트
                url_obj.image_url = image_url
                logger.info(
                    "Updated image URL for session_id: %s, message_id: %s, seq: %s",
                    session_id,
                    message_id,
                    image_seq,
                )
            else:
                # 기존 URL이 없으면 새로 생성
                new_url = ImageMessageUrl(
                    session_id=session_id,
                    message_id=message_id,
                    image_seq=image_seq,
                    image_url=image_url,
                )
                session.add(new_url)
                logger.info(
                    "Created image URL for session_id: %s, message_id: %s, seq: %s",
                    session_id,
                    message_id,
                    image_seq,
                )


# 조회(Read) 관련 함수
async def get_all_image_sessions(user_id: int):
    async with async_session() as session:
        result = await session.execute(
            select(ImageSession)
            .where(ImageSession.user_id == int(user_id))
            .order_by(ImageSession.created_at.desc())
        )
        sessions = result.scalars().all()
        logger.debug(
            "Fetched %d image sessions for user_id: %d", len(sessions), user_id
        )

    # 날짜 기준 계산 (로컬 타임존 기준, 필요 시 UTC로 변경)
    now = datetime.datetime.now()
    today = datetime.datetime.combine(now.date(), datetime.time())
    yesterday = today - datetime.timedelta(days=1)
    last7days = today - datetime.timedelta(days=7)
    last30days = today - datetime.timedelta(days=30)

    # 그룹 초기화
    image_sessions = {
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
            image_sessions["today"].append(session_obj)
        elif created_at >= yesterday:
            image_sessions["yesterday"].append(session_obj)
        elif created_at >= last7days:
            image_sessions["last7days"].append(session_obj)
        elif created_at >= last30days:
            image_sessions["last30days"].append(session_obj)
        else:
            image_sessions["older"].append(session_obj)

    return image_sessions


async def get_image_messages(session_id, limit=100, offset=0):
    """
    이미지 메시지와 함께 연결된 URL 정보를 가져옵니다.
    """
    async with async_session() as session:
        # 메시지 조회
        result = await session.execute(
            select(ImageMessage)
            .where(ImageMessage.session_id == session_id)
            .order_by(ImageMessage.created_at.asc())
            .offset(offset)
            .limit(limit)
        )
        messages = result.scalars().all()

        # 메시지당 URL 조회
        messages_with_urls = []
        for msg in messages:
            # URL 조회
            url_result = await session.execute(
                select(ImageMessageUrl)
                .where(
                    ImageMessageUrl.session_id == session_id,
                    ImageMessageUrl.message_id == msg.message_id,
                )
                .order_by(ImageMessageUrl.image_seq.asc())
            )
            urls = url_result.scalars().all()

            # 메시지 객체에 URLs 속성 추가
            msg_dict = {
                "session_id": str(msg.session_id),
                "message_id": str(msg.message_id),
                "user_message": msg.user_message,
                "image_prompt": msg.image_prompt,
                "liked": msg.liked,
                "disliked": msg.disliked,
                "dislike_feedback": msg.dislike_feedback,
                "created_at": msg.created_at,
                "updated_at": msg.updated_at,
                "image_urls": [url.image_url for url in urls],
            }
            messages_with_urls.append(msg_dict)

        logger.debug(
            "Fetched %d messages for session_id: %s", len(messages), session_id
        )
        return messages_with_urls


# 삭제(Delete) 관련 함수
async def delete_image_session(session_id):
    """
    주어진 session_id에 해당하는 image_sessions 레코드와 연관된 image_messages 레코드를 삭제합니다.
    """
    async with async_session() as session:
        async with session.begin():
            # 먼저 image_messages 삭제 (CASCADE 설정이 없다면 직접 삭제)
            await session.execute(
                delete(ImageMessage).where(ImageMessage.session_id == session_id)
            )
            # image_sessions 삭제
            await session.execute(
                delete(ImageSession).where(ImageSession.session_id == session_id)
            )
    logger.info("Deleted image session and its messages for session_id: %s", session_id)


# 제목변경(Update) 관련 함수
async def update_image_session_title(session_id: str, new_title: str):
    """
    주어진 session_id의 image_sessions 레코드의 title을 변경합니다.
    """
    async with async_session() as session:
        async with session.begin():
            # session_id에 해당하는 레코드를 찾고 title 업데이트
            result = await session.execute(
                select(ImageSession).where(ImageSession.session_id == session_id)
            )
            session_obj = result.scalars().first()

            if session_obj:
                session_obj.title = new_title
                await session.commit()  # 변경 사항 커밋
                logger.info(
                    "Updated image session title for session_id: %s to '%s'",
                    session_id,
                    new_title,
                )
                return session_obj  # 업데이트된 객체 반환
            else:
                logger.warning("Image session not found for session_id: %s", session_id)
                return None  # 해당 session_id가 없을 경우 None 반환


# Message Feedback 관련 업데이트
async def update_image_message(
    session_id: str,
    message_id: str,
    liked: bool = None,
    disliked: bool = None,
    dislike_feedback: str = None,
):
    async with async_session() as session:
        async with session.begin():
            result = await session.execute(
                select(ImageMessage).where(
                    ImageMessage.session_id == session_id,
                    ImageMessage.message_id == message_id,
                )
            )
            message_obj = result.scalars().first()

            if message_obj:
                # liked 업데이트 로직
                if liked is not None:
                    message_obj.liked = liked
                    # 만약 liked가 True이고 기존에 disliked가 True였다면
                    if liked and message_obj.disliked:
                        message_obj.disliked = False
                        message_obj.dislike_feedback = None

                # disliked 업데이트 로직
                if disliked is not None:
                    message_obj.disliked = disliked
                    # 만약 disliked가 True이고 기존에 liked가 True였다면
                    if disliked and message_obj.liked:
                        message_obj.liked = False

                # dislike_feedback 업데이트 (별도로 전달된 경우)
                if dislike_feedback is not None:
                    message_obj.dislike_feedback = dislike_feedback

                # 커밋: async with session.begin() 블록 내에서 커밋 수행
                logger.info(
                    "Updated image message for session_id: %s, message_id: %s (liked: %s, disliked: %s, dislike_feedback: %s)",
                    session_id,
                    message_id,
                    message_obj.liked,
                    message_obj.disliked,
                    message_obj.dislike_feedback,
                )
                return message_obj
            else:
                logger.warning(
                    "Image message not found for session_id: %s, message_id: %s",
                    session_id,
                    message_id,
                )
                return None

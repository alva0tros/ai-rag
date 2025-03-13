from sqlalchemy import (
    Column,
    Boolean,
    Integer,
    String,
    Text,
    TIMESTAMP,
    func,
    UniqueConstraint,
    ForeignKey,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class ImageSession(Base):
    """
    DB 모델: image_sessions
    이미지 세션은 제목(title)이 생성된 후에만 저장됩니다.
    """

    __tablename__ = "image_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(21), unique=True, nullable=False)
    user_id = Column(Integer, nullable=False, default=1)
    title = Column(String(255), nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())


class ImageMessage(Base):
    """
    DB 모델: image_messages
    각 이미지 요청마다 사용자 메시지와 AI 프롬프트를를 저장합니다.
    """

    __tablename__ = "image_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(
        String(21),
        ForeignKey("image_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
    )
    message_id = Column(String(21), nullable=False)
    image_seq = Column(Integer, nullable=False)
    user_message = Column(Text, nullable=False)
    image_prompt = Column(Text, nullable=False)
    image_url = Column(Text, nullable=True)
    liked = Column(Boolean, nullable=True)
    disliked = Column(Boolean, nullable=True)
    dislike_feedback = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint(
            "session_id", "message_id", "image_seq", name="image_messages_uk1"
        ),
    )

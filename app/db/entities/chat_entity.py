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


class ChatSession(Base):
    """
    DB 모델: chat_sessions
    대화 세션은 제목(title)이 생성된 후에만 저장됩니다.
    """

    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(UUID(as_uuid=True), unique=True, nullable=False)
    user_id = Column(Integer, nullable=False, default=1)
    title = Column(String(255), nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())


class ChatMessage(Base):
    """
    DB 모델: chat_messages
    각 대화 요청마다 사용자 메시지와 AI 응답을 저장합니다.
    """

    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("chat_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
    )
    message_id = Column(UUID(as_uuid=True), nullable=False)
    user_message = Column(Text, nullable=False)
    main_message = Column(Text, nullable=False)
    think_message = Column(Text, nullable=True)
    think_time = Column(Integer, nullable=True)
    liked = Column(Boolean, nullable=True)
    disliked = Column(Boolean, nullable=True)
    dislike_feedback = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint("session_id", "message_id", name="chat_messages_uk1"),
    )

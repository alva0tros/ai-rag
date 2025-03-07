"""
채팅 서비스의 통합 및 상위 로직을 담당하는 모듈

이 모듈은 채팅 기록 관리, 스토리지 관리, 리소스 관리, 대화 처리 등
상위 수준의 기능을 담당하는 클래스들과 이를 통합하는 ChatService 클래스를 포함합니다.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, AsyncGenerator

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from app.services.chat_core import (
    StreamingCallbackHandler,
    LLMManager,
    PromptManager,
    MessageProcessor,
    EventManager,
)
from app.db.repositories import chat as chat_crud

# ==========================================================
# 기본 설정
# ==========================================================

# 로거 설정
logger = logging.getLogger(__name__)

# ==========================================================
# 채팅 기록 관리 클래스
# ==========================================================


class ChatHistoryManager:
    """채팅 기록을 관리하는 클래스"""

    def __init__(self):
        self.store = {}  # 채팅 기록을 저장할 딕셔너리

    def get_chat_history(self, session_id: str) -> BaseChatMessageHistory:
        """
        채팅 기록을 가져오는 함수
        """
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def clear_chat_history(self, session_id: str) -> None:
        """
        채팅 기록을 삭제하는 함수
        """
        if session_id in self.store:
            del self.store[session_id]
            logger.info(f"Chat history for session {session_id} removed")


# ==========================================================
# 저장 관리 클래스
# ==========================================================


class StorageManager:
    """데이터 저장 관련 기능을 제공하는 클래스"""

    async def generate_and_save_title(
        self,
        llm_instance,
        message: str,
        conversation_id: str,
        user_id: Optional[int],
        prompt_manager: PromptManager,
        message_processor: MessageProcessor,
    ) -> str:
        """
        대화 제목을 생성하고 저장합니다.

        Args:
            llm_instance: LLM 인스턴스
            message: 사용자 메시지
            conversation_id: 대화 세션 ID
            user_id: 사용자 ID
            prompt_manager: 프롬프트 관리자
            message_processor: 메시지 처리기

        Returns:
            생성된 제목
        """
        title = await prompt_manager.generate_title(llm_instance, message)
        main_title, _ = message_processor.parse_message(title)

        try:
            await chat_crud.create_chat_session(conversation_id, main_title, user_id)
        except Exception as db_e:
            logger.exception("DB 저장 실패 (세션 저장): %s", db_e)
            # 제목 생성은 성공했으므로 에러를 다시 발생시키지 않고 계속 진행

        return title

    async def save_chat_message(
        self,
        conversation_id: str,
        message_id: str,
        user_message: str,
        main_message: str,
        think_message: str,
        think_time: int,
    ) -> None:
        """
        채팅 메시지를 데이터베이스에 저장합니다.

        Args:
            conversation_id: 대화 세션 ID
            message_id: 메시지 ID
            user_message: 사용자 메시지
            main_message: 주요 응답 메시지
            think_message: 생각 메시지
            think_time: 생각 시간 (초)
        """
        try:
            await chat_crud.create_chat_message(
                conversation_id,
                message_id,
                user_message,
                main_message,
                think_message,
                think_time,
            )
        except Exception as db_e:
            logger.exception("DB 저장 실패 (메시지 저장): %s", db_e)
            # 메시지 저장 실패는 치명적이지 않으므로 경고만 로깅하고 계속 진행


# ==========================================================
# 리소스 관리 클래스
# ==========================================================


class ResourceManager:
    """리소스 관리 관련 기능을 제공하는 클래스"""

    def __init__(
        self, tasks: Dict[str, asyncio.Task], history_manager: ChatHistoryManager
    ):
        self.tasks = tasks
        self.history_manager = history_manager

    def cleanup_session_resources(self, conversation_id: str) -> None:
        """
        세션 리소스를 정리합니다.

        Args:
            conversation_id: 대화 세션 ID
        """
        if conversation_id in self.tasks:
            task = self.tasks.pop(conversation_id, None)
            if task:
                logger.info(f"Task for session {conversation_id} removed")

        self.history_manager.clear_chat_history(conversation_id)


# ==========================================================
# 대화 처리 클래스
# ==========================================================


class ConversationHandler:
    """대화 처리 관련 기능을 제공하는 클래스"""

    def __init__(
        self,
        llm_manager: LLMManager,
        prompt_manager: PromptManager,
        history_manager: ChatHistoryManager,
        message_processor: MessageProcessor,
        event_manager: EventManager,
        storage_manager: StorageManager,
        tasks: Dict[str, asyncio.Task],
    ):
        self.llm_manager = llm_manager
        self.prompt_manager = prompt_manager
        self.history_manager = history_manager
        self.message_processor = message_processor
        self.event_manager = event_manager
        self.storage_manager = storage_manager
        self.tasks = tasks

    async def handle_chat_conversation(
        self,
        queue: asyncio.Queue,
        message: str,
        conversation_id: str,
        message_id: str,
        user_id: Optional[int],
        is_new_conversation: bool,
    ) -> AsyncGenerator:
        """
        채팅 대화를 처리하는 메인 함수

        Args:
            queue: 토큰을 받을 비동기 큐
            message: 사용자 메시지
            conversation_id: 대화 세션 ID
            message_id: 메시지 ID
            user_id: 사용자 ID
            is_new_conversation: 새 대화 여부

        Yields:
            SSE 이벤트
        """
        callback_handler = StreamingCallbackHandler(queue)

        # LLM 및 체인 설정
        llm_instance = self.llm_manager.get_llm(callback_handler)
        prompt = self.prompt_manager.setup_prompt()
        chain = prompt | llm_instance | StrOutputParser()

        with_message_history = RunnableWithMessageHistory(
            chain,
            self.history_manager.get_chat_history,
            input_messages_key="message",
            history_messages_key="history",
        )

        # 비동기로 체인 실행
        task = asyncio.create_task(
            with_message_history.ainvoke(
                {"message": message},
                config={"configurable": {"session_id": conversation_id}},
            )
        )

        # 세션 작업 저장
        self.tasks[conversation_id] = task

        # 첫 요청이면 conversation_id 반환
        if is_new_conversation:
            yield await self.event_manager.create_sse_event(
                "conversation_id", {"text": conversation_id}
            )

        # 응답 스트리밍 처리
        ai_response = ""
        last_state = None

        # 토큰을 실시간으로 스트리밍하여 클라이언트에 전송
        async for event, response, state in self.event_manager.stream_tokens(
            queue, task, self.message_processor
        ):
            ai_response = response
            last_state = state
            yield event

        # 완료 이벤트 전송
        yield await self.event_manager.create_sse_event("done", {"text": ""})

        # think 시간 계산
        think_time = await self.message_processor.calculate_think_time(
            last_state
            if last_state
            else {"think_start_time": None, "think_end_time": None}
        )

        # main_message와 think_message 분리
        main_message, think_message = self.message_processor.parse_message(ai_response)

        # 첫 대화일 경우 제목 생성
        if is_new_conversation:
            yield await self.event_manager.create_sse_event("title_start", {"text": ""})

            title = await self.storage_manager.generate_and_save_title(
                llm_instance,
                message,
                conversation_id,
                user_id,
                self.prompt_manager,
                self.message_processor,
            )

            # 채팅 메시지 저장
            await self.storage_manager.save_chat_message(
                conversation_id,
                message_id,
                message,
                main_message,
                think_message,
                think_time,
            )

            yield await self.event_manager.create_sse_event("title", {"text": title})
        else:
            # 기존 대화인 경우 메시지만 저장
            await self.storage_manager.save_chat_message(
                conversation_id,
                message_id,
                message,
                main_message,
                think_message,
                think_time,
            )


# ==========================================================
# 통합 서비스 클래스
# ==========================================================


class ChatService:
    """채팅 서비스를 총괄하는 클래스"""

    def __init__(self):
        # 전역 변수 설정
        self.tasks = {}  # 진행 중인 작업을 저장하는 딕셔너리

        # 각 관리자 클래스 초기화
        self.llm_manager = LLMManager()
        self.prompt_manager = PromptManager()
        self.history_manager = ChatHistoryManager()
        self.message_processor = MessageProcessor()
        self.event_manager = EventManager()
        self.storage_manager = StorageManager()
        self.resource_manager = ResourceManager(self.tasks, self.history_manager)

        # 대화 처리 핸들러 초기화
        self.conversation_handler = ConversationHandler(
            self.llm_manager,
            self.prompt_manager,
            self.history_manager,
            self.message_processor,
            self.event_manager,
            self.storage_manager,
            self.tasks,
        )

"""
채팅 서비스의 핵심 기능을 제공하는 모듈

이 모듈은 LLM 관리, 프롬프트 생성, 메시지 처리, 이벤트 관리 등의
기본적인 기능을 담당하는 클래스들을 포함합니다.
"""

import re
import asyncio
import json
import time
import logging
import yake
from typing import Dict, Any, Tuple, AsyncGenerator
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.callbacks import BaseCallbackHandler

# from langchain_core.output_parsers import StrOutputParser

# ==========================================================
# 기본 설정
# ==========================================================

# 로거 설정
logger = logging.getLogger(__name__)

# 상수 정의
THINK_TAG_START = "<think>"
THINK_TAG_END = "</think>"

# ==========================================================
# 스트리밍 콜백 핸들러
# ==========================================================


class StreamingCallbackHandler(BaseCallbackHandler):
    """LLM에서 생성되는 토큰을 스트리밍하기 위한 콜백 핸들러"""

    def __init__(self, queue: asyncio.Queue):
        self.queue = queue

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        await self.queue.put(token)


# ==========================================================
# LLM 관리 클래스
# ==========================================================


class LLMManager:
    """LLM 인스턴스를 관리하는 클래스"""

    def __init__(self):
        self.llm = None

    def get_llm(self, callback_handler=None):
        """
        LLM 인스턴스를 반환합니다. 없으면 생성합니다.
        """
        if self.llm is None:
            self.llm = self._setup_llm(callback_handler)
        elif callback_handler is not None:
            self.llm.callbacks = [callback_handler]  # 콜백 핸들러 업데이트
        return self.llm

    def _setup_llm(self, callback_handler):
        """
        LLM 설정 함수
        """
        return ChatOllama(
            # model="phi4:14b-q8_0",
            # model="phi4:latest",
            # model="deepseek-r1:32b",
            model="deepseek-r1:1.5b",
            # model="deepseek-r1:7b-qwen-distill-q4_K_M",
            # model="qwen2.5:latest",
            # model="deepseek-r1:7b",
            # model="qwen2.5:latest",
            temperature=0.5,
            num_predict=2048,
            num_ctx=8192,
            num_thread=8,
            callbacks=[callback_handler],
        )

        # return ChatLlamaCpp(
        #     model_path=model_path,
        #     temperature=0.6,
        #     max_tokens=8192,
        #     n_ctx=32768,
        #     n_gpu_layers=33,      # L40S GPU에 최적화된 GPU 오프로딩: 약 16개 레이어
        #     f16_kv=True,          # FP16 키-값 캐시 사용으로 메모리 효율 향상
        #     n_threads=16,
        #     n_batch=512,          # CUDA 연산 최적화를 위한 배치 크기
        #     seed=3407,
        #     stop=["<|User|>", "<|Assistant|>"],
        #     callbacks=[callback_handler],
        # )


# ==========================================================
# 프롬프트 관리 클래스
# ==========================================================


class PromptManager:
    """채팅 프롬프트를 관리하는 클래스"""

    def setup_prompt(self):
        """
        프롬프트 설정 함수
        """
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """당신은 친절하고 전문적인 AI 어시스턴트입니다. 모든 답변은 반드시 한국어로 작성해야 합니다. 
                       한자, 영어 또는 기타 언어를 포함하지 말고 오직 한국어만 사용하세요.
                       질문과 관련성이 높은 내용만 답변하고 추측된 내용을 생성하지 마세요.""",
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{message}"),
            ]
        )

    async def generate_title(self, user_message: str) -> str:
        """
        대화 제목을 생성하는 함수 - Yake 라이브러리 사용
        """
        # 특수문자와 과도한 공백 제거
        cleaned_message = re.sub(r"\s+", " ", user_message).strip()

        # <think> 태그가 있다면 제거
        think_pattern = re.compile(r"<think>(.*?)<\/think>", re.DOTALL)
        cleaned_message = think_pattern.sub("", cleaned_message).strip()

        # 메시지가 너무 짧으면 그대로 반환
        if len(cleaned_message) < 10:
            return cleaned_message[:30]

        try:
            # Yake 키워드 추출기 초기화 (언어 자동 감지)
            kw_extractor = yake.KeywordExtractor(
                lan="ko",  # 기본값으로 설정하나 자동 감지 기능이 있음
                n=2,  # 1-2개 단어로 구성된 키워드 추출
                dedupLim=0.7,  # 중복 제거 임계값
                top=3,  # 상위 3개 키워드 추출
                features=None,
            )

            # 키워드 추출
            keywords = kw_extractor.extract_keywords(cleaned_message)

            # 키워드가 추출되었는지 확인
            if keywords:
                # 점수가 낮을수록 중요도가 높음 (Yake 특성)
                sorted_keywords = sorted(keywords, key=lambda x: x[1])

                # 상위 2개 키워드 사용
                top_keywords = [kw[0] for kw in sorted_keywords[:2]]
                title = " ".join(top_keywords)

                # 제목이 너무 길면 자르기
                if len(title) > 30:
                    title = title[:30] + "..."

                return title

            # 키워드 추출 실패 시 첫 문장 사용
            else:
                sentences = re.split(r"[.?!。？！\n]+", cleaned_message)
                first_sentence = sentences[0].strip()

                if len(first_sentence) > 30:
                    return first_sentence[:30] + "..."
                return first_sentence

        except Exception as e:
            # 오류 발생 시 기본 제목 생성
            logger.error(f"제목 생성 오류: {e}")

            # 간단한 폴백 메커니즘: 메시지 앞부분 사용
            if len(cleaned_message) > 30:
                return cleaned_message[:30] + "..."
            return cleaned_message

    # async def generate_title(self, llm_instance, user_message: str) -> str:
    #     """
    #     대화 제목을 생성하는 함수
    #     """
    #     title_prompt = ChatPromptTemplate.from_messages(
    #         [
    #             (
    #                 "system",
    #                 """다음 대화 내용을 간결하게 요약하여 제목을 생성하세요.
    #                    - 제목은 1문장으로 작성하세요.
    #                    - 제목은 5단어 이내로 작성하세요.
    #                    - 최대한 짧게 작성하세요.
    #                    - 반드시 한국어만 사용
    #                    - 추론 과정(<think>내용)은 무시
    #                    - 핵심 키워드 위주로 구성""",
    #             ),
    #             ("human", "사용자: {user_message}"),
    #         ]
    #     )

    #     chain = title_prompt | llm_instance | StrOutputParser()
    #     title = await chain.ainvoke({"user_message": user_message})
    #     return title.strip()


# ==========================================================
# 메시지 처리 클래스
# ==========================================================


class MessageProcessor:
    """메시지 처리 관련 기능을 제공하는 클래스"""

    def parse_message(self, message: str) -> Tuple[str, str]:
        """
        메시지를 <think>...</think> 태그를 기준으로 분리
        """
        think_pattern = re.compile(r"<think>(.*?)<\/think>", re.DOTALL)
        think_match = think_pattern.search(message)

        if think_match:
            think_message = think_match.group(1).strip()
            main_message = think_pattern.sub("", message).strip()  # <think> 태그 제거
        else:
            think_message = ""
            main_message = message.strip()

        return main_message, think_message

    async def process_think_tags(
        self, token: str, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Think 태그 처리 및 시간 계산

        Args:
            token: 현재 토큰
            state: 현재 상태 (think_start_time, think_end_time 등)

        Returns:
            업데이트된 상태
        """
        # <think> 태그 시작 감지
        if THINK_TAG_START in token and state["think_start_time"] is None:
            state["think_start_time"] = time.perf_counter()

        # </think> 태그 종료 감지
        if THINK_TAG_END in token and state["think_end_time"] is None:
            state["think_end_time"] = time.perf_counter()

        # think_time 계산
        if (
            state["think_start_time"] is not None
            and state["think_end_time"] is not None
        ):
            state["current_think_time"] = int(
                state["think_end_time"] - state["think_start_time"]
            )
        else:
            state["current_think_time"] = None

        return state

    async def calculate_think_time(self, state: Dict[str, Any]) -> int:
        """
        <think> 태그 사이의 시간을 계산합니다.

        Args:
            state: 상태 딕셔너리

        Returns:
            생각 시간 (초)
        """
        if state["think_start_time"] and state["think_end_time"]:
            return int(state["think_end_time"] - state["think_start_time"])
        return 0


# ==========================================================
# 이벤트 관리 클래스
# ==========================================================


class EventManager:
    """SSE 이벤트 관련 기능을 제공하는 클래스"""

    async def create_sse_event(self, event: str, data: Any) -> Dict[str, str]:
        """
        이벤트 이름과 데이터를 받아 SSE용 딕셔너리를 생성합니다.

        Args:
            event: 이벤트 이름
            data: 전송할 데이터

        Returns:
            SSE 형식의 데이터
        """
        data = json.dumps(data)
        return {"event": event, "data": data}

    async def stream_tokens(
        self,
        queue: asyncio.Queue,
        task: asyncio.Task,
        message_processor: MessageProcessor,
    ) -> AsyncGenerator[Tuple[Dict[str, str], str, Dict[str, Any]], None]:
        """
        큐에서 토큰을 받아 실시간으로 처리하는 함수

        Args:
            queue: 토큰을 받을 비동기 큐
            task: 체인 실행 비동기 작업
            message_processor: 메시지 처리기

        Yields:
            SSE 이벤트, 토큰, 상태
        """
        ai_response = ""
        state = {
            "think_start_time": None,
            "think_end_time": None,
            "current_think_time": None,
        }

        # 큐에서 토큰을 받아 즉시 처리하여 반환
        while not task.done() or not queue.empty():
            try:
                token = await asyncio.wait_for(queue.get(), timeout=1.0)
                logger.debug(f"Token received: {token}")
                ai_response += token

                # think 태그 처리
                state = await message_processor.process_think_tags(token, state)

                # 토큰과 상태를 즉시 반환
                event = await self.create_sse_event(
                    "message", {"text": token, "thinkTime": state["current_think_time"]}
                )

                yield event, ai_response, state

            except asyncio.TimeoutError:
                if task.done():
                    break
                continue

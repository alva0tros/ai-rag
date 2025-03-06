import re
import asyncio
from langchain_ollama import ChatOllama
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.chat_history import BaseChatMessageHistory

from langchain_community.chat_message_histories import ChatMessageHistory


class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        await self.queue.put(token)


# 전역 변수 (실제 서비스에서는 상태 관리를 별도 저장소로 처리하는 것이 좋습니다)
llm = None
store = {}  # 채팅 기록을 저장할 딕셔너리
tasks = {}  # 진행 중인 작업을 저장하는 딕셔너리

# 모델 파일 경로 설정
# model_path = "DeepSeek-R1-GGUF/DeepSeek-R1-UD-IQ1_S/DeepSeek-R1-UD-IQ1_S-00001-of-00003.gguf"


def get_llm(callback_handler=None):
    global llm
    if llm is None:
        llm = setup_llm(callback_handler)
    elif callback_handler is not None:
        llm.callbacks = [callback_handler]  # 콜백 핸들러 업데이트
    return llm


# LLM 설정 함수
def setup_llm(callback_handler):
    return ChatOllama(
        model="phi4:14b-q8_0",
        # model="phi4:latest",
        # model="deepseek-r1:32b",
        # model="deepseek-r1:1.5b",
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


# 프롬프트 설정 함수
def setup_prompt():
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


def get_chat_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# async def generate_title(llm: ChatOllama, user_message: str, ai_response: str) -> str:
#     title_prompt = ChatPromptTemplate.from_messages(
#         [
#             (
#                 "system",
#                 """다음 대화 내용을 간결하게 요약하여 제목을 생성하세요.
#                    - 제목은 10단어 이내로 작성하세요.
#                    - 반드시 한국어만 사용
#                    - 추론 과정(<think>내용)은 무시
#                    - 핵심 키워드 위주로 구성""",
#             ),
#             ("human", "사용자: {user_message}\nAI: {ai_response}"),
#         ]
#     )

#     chain = title_prompt | llm | StrOutputParser()
#     title = await chain.ainvoke(
#         {"user_message": user_message, "ai_response": ai_response}
#     )
#     return title.strip()

async def generate_title(llm: ChatOllama, user_message: str) -> str:
    title_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """다음 대화 내용을 간결하게 요약하여 제목을 생성하세요.
                   - 제목은 1문장으로 작성하세요.
                   - 제목은 5단어 이내로 작성하세요.
                   - 최대한 짧게 작성하세요.
                   - 반드시 한국어만 사용
                   - 추론 과정(<think>내용)은 무시
                   - 핵심 키워드 위주로 구성""",
            ),
            ("human", "사용자: {user_message}"),
        ]
    )

    # llm = ChatOllama(
    #     # model="qwen2.5:latest",
    #     model="deepseek-r1:1.5b",
    #     temperature=0.5,
    #     num_predict=50,  # 짧은 제목 생성을 위한 설정
    # )

    chain = title_prompt | llm | StrOutputParser()
    title = await chain.ainvoke(
        {"user_message": user_message}
    )
    return title.strip()


def parse_message(message: str):
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

    # # 태그 제거를 위한 정규식 패턴 강화
    # think_pattern = re.compile(r"<think>(.*?)<\/think>", re.DOTALL | re.IGNORECASE)

    # # 여러 개의 <think> 태그 처리
    # think_matches = think_pattern.findall(message)
    # think_message = "\n".join(think_matches).strip()

    # # 메인 메시지 정제
    # main_message = think_pattern.sub("", message)
    # main_message = re.sub(r"\s+", " ", main_message).strip()  # 연속 공백 제거

    return main_message, think_message

import asyncio
import uuid
import json
import time
import logging
import torch

from fastapi import APIRouter, Request, HTTPException
from sse_starlette.sse import EventSourceResponse
from app.services.chat import chat_service
from app.services.chat import chat_crud


router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/chat/chat")
async def chat(request: Request):

    # 요청 본문 파싱
    try:
        data = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid JSON request")

    print("data : ", data)

    message = data.get("message", "")
    user_id = data.get("user_id")
    conversation_id = data.get("conversation_id")
    message_id = data.get("message_id")

    # conversation_id가 없는 경우 새로운 ID 생성
    is_new_conversation = not conversation_id
    if is_new_conversation:
        conversation_id = str(uuid.uuid4())

    # 응답 큐 생성
    queue = asyncio.Queue()

    async def event_generator():

        # 헬퍼 함수: 이벤트 이름과 데이터를 받아 SSE용 딕셔너리 생성
        def send_event(event: str, data):
            data = json.dumps(data)
            return {"event": event, "data": data}

        callback_handler = chat_service.StreamingCallbackHandler(queue)

        # <think> 부분의 타이밍을 기록할 변수들
        think_start_time = None
        think_end_time = None

        try:
            # LLM 실행 전 GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU 메모리 정리 완료")

            # LLM 및 체인 설정
            # llm = chat_service.setup_llm(callback_handler)

            # LLM 및 체인 설정 (한 번만 생성)
            llm_instance = chat_service.get_llm(callback_handler)
            prompt = chat_service.setup_prompt()
            chain = prompt | llm_instance | chat_service.StrOutputParser()

            with_message_history = chat_service.RunnableWithMessageHistory(
                chain,
                chat_service.get_chat_history,
                input_messages_key="message",
                history_messages_key="history",
            )

            # ai_response 초기화
            ai_response = ""

            # 비동기로 체인 실행
            # task = asyncio.create_task(chain.ainvoke({"message": message}))
            task = asyncio.create_task(
                with_message_history.ainvoke(
                    {"message": message},
                    config={"configurable": {"session_id": conversation_id}},
                )
            )
            print("conversation_id : ", conversation_id)
            chat_service.tasks[conversation_id] = task  # 작업을 저장

            # 첫 요청이면 conversation_id 반환
            if is_new_conversation:
                yield send_event("conversation_id", {"text": conversation_id})

            # 큐에서 토큰을 받아 클라이언트로 전송
            while not task.done() or not queue.empty():
                # if task.done() and queue.empty():
                #     break
                try:
                    token = await asyncio.wait_for(queue.get(), timeout=1.0)
                    print("token : ", token)
                    ai_response += token

                    # <think> 태그 시작 감지: 아직 기록하지 않았다면 기록
                    if "<think>" in token and think_start_time is None:
                        think_start_time = time.perf_counter()

                    # </think> 태그 종료 감지: 아직 기록하지 않았다면 기록
                    if "</think>" in token and think_end_time is None:
                        think_end_time = time.perf_counter()

                    current_think_time = (
                        int(think_end_time - think_start_time)
                        if (think_start_time is not None and think_end_time is not None)
                        else None
                    )
                    yield send_event(
                        "message", {"text": token, "thinkTime": current_think_time}
                    )

                except asyncio.TimeoutError:
                    if task.done():
                        break
                    continue

            # 완료 이벤트 전송
            yield send_event("done", {"text": ""})

            # <think> 태그 사이의 시간 계산 (초 단위, 정수)
            if think_start_time and think_end_time:
                think_time = int(think_end_time - think_start_time)
            else:
                think_time = 0

            # main_message와 think_message 분리
            main_message, think_message = chat_service.parse_message(ai_response)

            # 첫 대화일 경우 제목 생성
            if is_new_conversation:
                yield send_event("title_start", {"text": ""})

                title = await chat_service.generate_title(
                    llm_instance, message, main_message
                )
                main_title, _ = chat_service.parse_message(title)

                # 대화 이력 DB 저장
                try:
                    await chat_crud.create_chat_session(
                        conversation_id, main_title, user_id
                    )
                    await chat_crud.create_chat_message(
                        conversation_id,
                        message_id,
                        message,
                        main_message,
                        think_message,
                        think_time,
                    )
                except Exception as db_e:
                    logger.exception("DB 저장 실패 (메시지 저장): %s", db_e)

                yield send_event("title", {"text": title})

            else:
                # 기존 대화인 경우 대화 내용만 추가 저장
                try:
                    await chat_crud.create_chat_message(
                        conversation_id,
                        message_id,
                        message,
                        main_message,
                        think_message,
                        think_time,
                    )
                except Exception as db_e:
                    logger.exception("DB 저장 실패 (메시지 저장): %s", db_e)
                    yield send_event("error", {"text": "Error saving chat message."})

        except asyncio.CancelledError:
            logger.warning("Task for session %s was cancelled.", conversation_id)
            yield send_event("error", {"text": "Task was cancelled."})
        except Exception as e:
            logger.exception("Error in session %s: %s", conversation_id, e)
            yield send_event("error", {"text": str(e)})
        finally:
            chat_service.tasks.pop(conversation_id, None)  # 작업 제거
            logger.info("Cleaned up session: %s", conversation_id)

    return EventSourceResponse(event_generator())


@router.post("/chat/stop")
async def stop_chat(request: Request):
    data = await request.json()
    conversation_id = data.get("conversation_id", "default")

    if conversation_id not in chat_service.tasks:
        raise HTTPException(
            status_code=404, detail="No active task found for this session."
        )

    task = chat_service.tasks[conversation_id]
    task.cancel()  # 작업 취소
    return {"message": f"Task for session {conversation_id} has been cancelled."}

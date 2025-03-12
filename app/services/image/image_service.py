"""
이미지 생성 서비스 모듈

이 모듈은 이미지 생성의 서비스 레이어를 제공합니다.
태스크 관리, 진행 상황 스트리밍, 이미지 저장 및 API와의 통신을 담당합니다.
"""

import os
import json
import time
import asyncio
import logging
from PIL import Image
from typing import List, Dict, Any, Optional, AsyncGenerator

from app.core.config import settings
from app.services.image.image_core import image_generator
from app.services.image.image_prompt import image_prompt

logger = logging.getLogger(__name__)


class TaskManager:
    """
    이미지 생성 태스크 관리자
    """

    def __init__(self):
        """태스크 관리자 초기화"""
        self.tasks = {}  # 진행 중인 작업 저장

    async def create_task(self, conversation_id: str) -> Dict[str, Any]:
        """
        새 이미지 생성 태스크 생성

        Args:
            conversation_id: 대화 ID

        Returns:
            Dict[str, Any]: 태스크 데이터 구조
        """
        task = {
            "images": None,
            "progress": 0.0,
            "progress_event": asyncio.Event(),
            "last_reported_progress": -1,
            "generate_task": None,
            "status": "initialized",
            "created_at": time.time(),
        }

        async with asyncio.Lock():
            self.tasks[conversation_id] = task

        logger.info(f"Created new task for conversation {conversation_id}")
        return task

    def get_task(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        대화 ID로 태스크 가져오기

        Args:
            conversation_id: 대화 ID

        Returns:
            Optional[Dict[str, Any]]: 태스크 데이터 또는 없으면 None
        """
        return self.tasks.get(conversation_id)

    def update_task_status(self, conversation_id: str, status: str) -> bool:
        """
        태스크 상태 업데이트

        Args:
            conversation_id: 대화 ID
            status: 새 상태

        Returns:
            bool: 성공 여부
        """
        if conversation_id not in self.tasks:
            return False

        self.tasks[conversation_id]["status"] = status
        logger.debug(f"Updated task {conversation_id} status to {status}")
        return True

    async def cancel_task(self, conversation_id: str) -> bool:
        """
        실행 중인 태스크 취소

        Args:
            conversation_id: 대화 ID

        Returns:
            bool: 성공 여부
        """
        if conversation_id not in self.tasks:
            logger.warning(f"No task found for conversation {conversation_id}")
            return False

        task = self.tasks[conversation_id]

        if task["generate_task"] and not task["generate_task"].done():
            task["generate_task"].cancel()
            self.update_task_status(conversation_id, "cancelled")
            logger.info(f"Task for conversation {conversation_id} has been cancelled")
            return True

        return False

    def cleanup_task(self, conversation_id: str) -> bool:
        """
        태스크를 제거하고 리소스 정리

        Args:
            conversation_id: 대화 ID

        Returns:
            bool: 성공 여부
        """
        if conversation_id in self.tasks:
            task = self.tasks[conversation_id]

            # 실행 중이면 태스크 취소
            if task["generate_task"] and not task["generate_task"].done():
                task["generate_task"].cancel()

            # 태스크 제거
            del self.tasks[conversation_id]
            logger.info(f"Cleaned up task for conversation {conversation_id}")
            return True

        return False

    def get_active_tasks_count(self) -> int:
        """
        활성 태스크 수 가져오기

        Returns:
            int: 활성 태스크 수
        """
        return len(
            [
                task_id
                for task_id, task in self.tasks.items()
                if task["generate_task"] and not task["generate_task"].done()
            ]
        )


class ImageService:
    """
    이미지 생성 서비스 클래스
    핵심 이미지 생성기를 래핑하고 태스크 관리 및 스트리밍 기능 제공
    """

    _instance = None
    _is_initialized = False

    def __new__(cls, *args, **kwargs):
        """싱글톤 패턴 구현"""
        if cls._instance is None:
            cls._instance = super(ImageService, cls).__new__(cls)
            cls._instance.progress_callback = None
        return cls._instance

    def __init__(self):
        """서비스 초기화"""
        # 이미 초기화된 경우 스킵
        if self._is_initialized:
            return

        # 태스크 관리자 초기화
        self.task_manager = TaskManager()
        logger.info("ImageService initialization complete")
        self._is_initialized = True

    def save_generated_images(
        self,
        images: List[Image.Image],
        user_id: int,
        conversation_id: str,
        message_id: str,
    ) -> List[str]:
        """
        생성된 이미지 저장 및 URL 반환

        Args:
            images: 생성된 이미지 객체 리스트
            user_id: 사용자 ID
            conversation_id: 대화 ID
            message_id: 메시지 ID

        Returns:
            List[str]: 이미지 URL 리스트
        """
        image_urls = []
        save_dir = os.path.join(
            settings.GENERATED_IMAGE_PATH, str(user_id), conversation_id, message_id
        )
        os.makedirs(save_dir, exist_ok=True)

        try:
            for i, img in enumerate(images):
                file_name = f"img{i}.png"
                file_path = os.path.join(save_dir, file_name)
                img.save(file_path, format="PNG")
                image_url = f"/generated/images/{user_id}/{conversation_id}/{message_id}/{file_name}"
                image_urls.append(image_url)

            logger.info(
                f"Successfully saved {len(images)} images for conversation {conversation_id}"
            )
            return image_urls

        except Exception as e:
            logger.error(
                f"Error saving images for conversation {conversation_id}: {str(e)}"
            )
            raise

    async def stream_generation_progress(
        self,
        prompt: str,
        seed: Optional[int],
        guidance: float,
        user_id: int,
        conversation_id: str,
        message_id: str,
    ) -> AsyncGenerator[Dict[str, str], None]:
        """
        이미지 생성 진행 상황 및 결과 스트리밍

        Args:
            prompt: 텍스트 프롬프트
            seed: 재현성을 위한 랜덤 시드
            guidance: 가이던스 스케일
            user_id: 사용자 ID
            conversation_id: 대화 ID
            message_id: 메시지 ID

        Yields:
            Dict[str, str]: 서버 전송 이벤트용 이벤트
        """
        # 태스크 초기화
        task = await self.task_manager.create_task(conversation_id)

        # 프롬프트 처리를 시작하기 전에 상태 업데이트
        yield {
            "event": "prompt_start",
            "data": json.dumps({"text": "Processing your prompt..."}),
        }

        # 프롬프트 저장을 위한 변수
        full_prompt = ""

        # 스트리밍 방식으로 프롬프트 처리
        async for prompt_chunk in image_prompt.translate_and_enhance_stream(prompt):
            full_prompt += prompt_chunk
            # 프롬프트 청크를 클라이언트에 전송
            yield {"event": "prompt_chunk", "data": json.dumps({"text": prompt_chunk})}

        # 프롬프트 처리 완료 알림
        yield {"event": "prompt_complete", "data": ""}

        # 진행률 업데이트 콜백 정의
        def progress_callback(progress: float) -> None:
            """진행률 업데이트 및 리스너에게 알림"""
            task["progress"] = progress
            task["progress_event"].set()

            # 로그 수준 조정 (10% 단위 정보는 INFO, 세부 진행은 DEBUG)
            # if progress % 10 < 0.5 or progress >= 99.5:
            #     logger.info(f"Conversation {conversation_id}: progress {progress:.1f}%")
            # else:
            #     logger.debug(
            #         f"Conversation {conversation_id}: progress update {progress:.1f}%"
            #     )

        # 콜백 설정
        image_generator.progress_callback = progress_callback

        # 이미지 생성 함수
        async def generate_images() -> None:
            """백그라운드에서 이미지 생성 실행"""
            try:
                task["images"] = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: image_generator.generate_image(full_prompt, seed, guidance),
                )
            except Exception as e:
                logger.error(
                    f"Conversation {conversation_id}: image generation failed - {str(e)}"
                )
                raise

        # 이벤트 생성기
        try:
            logger.debug(f"Starting event generator for conversation {conversation_id}")

            # 이미지 생성을 백그라운드로 시작
            logger.debug(
                f"Starting image generation task for conversation {conversation_id}"
            )
            task["generate_task"] = asyncio.create_task(generate_images())

            # 진행률 스트리밍 (1% 단위로 전송)
            while task["progress"] < 100.0:
                await task["progress_event"].wait()

                current_progress = round(task["progress"], 0)
                if current_progress > task["last_reported_progress"]:
                    task["last_reported_progress"] = current_progress
                    yield {"event": "progress", "data": f"{current_progress}"}

                task["progress_event"].clear()
                await asyncio.sleep(0.1)

            # 생성 완료 대기
            await task["generate_task"]

            # 최종 100% 진행률 보장
            if task["last_reported_progress"] != 100:
                yield {"event": "progress", "data": "100"}

            logger.info(
                f"Image generation for conversation {conversation_id} completed"
            )

            # 이미지 저장 및 URL 반환
            if task["images"]:
                logger.debug(
                    f"Saving {len(task['images'])} images for conversation {conversation_id}"
                )
                image_urls = self.save_generated_images(
                    task["images"], user_id, conversation_id, message_id
                )

                logger.info(
                    f"Generated {len(image_urls)} images for conversation {conversation_id}"
                )

                yield {"event": "image", "data": json.dumps({"image_urls": image_urls})}

        except asyncio.CancelledError:
            logger.warning(
                f"Image generation task for conversation {conversation_id} was cancelled"
            )
            yield {"event": "error", "data": "Task was cancelled"}

        except Exception as e:
            logger.exception(
                f"Error streaming progress for conversation {conversation_id}: {str(e)}"
            )
            yield {"event": "error", "data": str(e)}

        finally:
            # 작업 정리
            logger.debug(f"Cleaning up resources for conversation {conversation_id}")
            self.task_manager.cleanup_task(conversation_id)

    async def stop_image_generation(self, conversation_id: str) -> bool:
        """
        진행 중인 이미지 생성 프로세스 중지

        Args:
            conversation_id: 대화 ID

        Returns:
            bool: 성공 여부
        """
        return await self.task_manager.cancel_task(conversation_id)


# 싱글톤 인스턴스 생성
image_service = ImageService()


# 외부 호출용 헬퍼 함수
def generate_image(*args, **kwargs):
    """
    텍스트 프롬프트 기반 이미지 생성
    이것은 image_generator의 generate_image 메서드에 대한 래퍼입니다.
    """
    return image_generator.generate_image(*args, **kwargs)

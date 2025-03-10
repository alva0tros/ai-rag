"""
이미지 생성의 핵심 기능 모듈

이 모듈은 딥러닝 모델을 이용한 이미지 생성의 기본 기능을 제공합니다.
GPU 메모리 관리, 모델 로드/언로드, 이미지 생성 프로세스를 담당합니다.
"""

import torch
import numpy as np
import gc
import logging
from PIL import Image
from typing import List, Callable, Optional, Tuple
from contextlib import contextmanager

from transformers import AutoConfig, AutoModelForCausalLM
from src.janus.janus.models import VLChatProcessor

from app.core.config import settings

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    GPU 메모리를 효율적으로 관리하는 헬퍼 클래스
    """

    def __init__(self):
        """메모리 관리자 초기화"""
        self._memory_usage_log = []
        self._max_vram_usage = 0
        self.cuda_device = "cuda" if torch.cuda.is_available() else "cpu"

    def clear_gpu_memory(self) -> None:
        """
        GPU 메모리를 철저히 정리하고 가비지 컬렉션 실행
        """
        if self.cuda_device == "cuda":
            # CUDA 캐시 비우기
            torch.cuda.empty_cache()
            # 가비지 컬렉션 실행
            gc.collect()

            # 메모리 상태 로깅
            if torch.cuda.is_available():
                current_mem = torch.cuda.memory_allocated() / 1024**2
                max_mem = torch.cuda.max_memory_allocated() / 1024**2
                logger.debug(f"Current GPU memory usage: {current_mem:.2f} MB")
                logger.debug(f"Maximum GPU memory usage: {max_mem:.2f} MB")

                # 메모리 사용량 추적
                self._memory_usage_log.append(current_mem)
                self._max_vram_usage = max(self._max_vram_usage, max_mem)

            logger.info("GPU memory cleanup complete")

    @property
    def memory_usage_history(self) -> List[float]:
        """메모리 사용량 히스토리 반환"""
        return self._memory_usage_log

    @property
    def max_memory_usage(self) -> float:
        """최대 메모리 사용량 반환"""
        return self._max_vram_usage


class ImageGenerator:
    """
    딥러닝 모델을 사용한 이미지 생성기
    싱글톤 패턴으로 구현되어 효율적인 리소스 관리
    """

    _instance = None
    _is_initialized = False

    def __new__(cls, *args, **kwargs):
        """
        싱글톤 인스턴스 생성 또는 기존 인스턴스 반환
        """
        if cls._instance is None:
            cls._instance = super(ImageGenerator, cls).__new__(cls)
            cls._instance.model_loaded = False
            cls._instance.vl_gpt = None
            cls._instance.vl_chat_processor = None
            cls._instance.tokenizer = None
            cls._instance.progress_callback = None
        return cls._instance

    def __init__(self, progress_callback: Optional[Callable[[float], None]] = None):
        """
        이미지 생성기 초기화

        Args:
            progress_callback: 진행 상황 업데이트 콜백 함수
        """
        # 이미 초기화된 경우 스킵
        if self._is_initialized:
            if progress_callback is not None:
                self.progress_callback = progress_callback
            return

        # 서비스 설정
        self.model_path = settings.IMAGE_MODEL_PATH
        self.memory_manager = MemoryManager()
        self.progress_callback = progress_callback

        logger.info(
            f"ImageGenerator initialized: device={self.memory_manager.cuda_device}"
        )
        self._is_initialized = True

    def load_model(self, force_reload: bool = False) -> bool:
        """
        이미지 생성 모델 로드

        Args:
            force_reload: 이미 로드된 경우에도 강제로 다시 로드할지 여부

        Returns:
            bool: 성공 여부
        """
        # 이미 로드되어 있고 강제 재로드가 아니면 스킵
        if self.model_loaded and not force_reload:
            logger.info("Model already loaded, skipping load")
            return True

        logger.info(f"Starting model load: {self.model_path}")
        self.memory_manager.clear_gpu_memory()

        try:
            # 모델 설정 로드
            setting = AutoConfig.from_pretrained(self.model_path)
            language_config = setting.language_config
            language_config._attn_implementation = "eager"

            # 장치에 따른 데이터 타입 설정
            cuda_device = self.memory_manager.cuda_device
            dtype = torch.bfloat16 if cuda_device == "cuda" else torch.float32

            # 모델 로드 (장치에 맞게)
            self.vl_gpt = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                language_config=language_config,
                trust_remote_code=True,
                device_map="auto" if cuda_device == "cuda" else None,
                low_cpu_mem_usage=True,
                torch_dtype=dtype,
            )

            # 평가 모드로 설정
            self.vl_gpt = self.vl_gpt.eval()

            # CUDA 사용 시 추론 속도 향상을 위한 설정
            if cuda_device == "cuda":
                torch.backends.cudnn.benchmark = True

            # 프로세서 및 토크나이저 로드
            self.vl_chat_processor = VLChatProcessor.from_pretrained(self.model_path)
            self.tokenizer = self.vl_chat_processor.tokenizer

            self.model_loaded = True
            logger.info("Model load successful")

            # 최종 메모리 정리
            self.memory_manager.clear_gpu_memory()
            return True

        except Exception as e:
            logger.error(f"Model load failed: {str(e)}")
            self.memory_manager.clear_gpu_memory()
            self.model_loaded = False
            raise

    def unload_model(self) -> bool:
        """
        메모리 확보를 위해 모델 언로드

        Returns:
            bool: 성공 여부
        """
        if not self.model_loaded:
            return True

        logger.info("Starting model unload from memory")

        try:
            # CUDA 사용 시 먼저 CPU로 이동
            if self.vl_gpt is not None:
                if self.memory_manager.cuda_device == "cuda":
                    self.vl_gpt = self.vl_gpt.cpu()
                del self.vl_gpt
                self.vl_gpt = None

            # 프로세서 정리
            if self.vl_chat_processor is not None:
                del self.vl_chat_processor
                self.vl_chat_processor = None

            # 토크나이저 정리
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None

            # 메모리 정리
            self.memory_manager.clear_gpu_memory()

            self.model_loaded = False
            logger.info("Model unload successful")
            return True

        except Exception as e:
            logger.error(f"Error during model unload: {str(e)}")
            raise

    @contextmanager
    def model_context(self):
        """
        모델 자동 로드/언로드를 위한 컨텍스트 매니저

        Example:
            with image_generator.model_context():
                # 여기서 모델이 로드됨
                result = image_generator.generate(...)
            # 여기서 리소스 정리됨
        """
        try:
            self.load_model()
            yield
        finally:
            # 언로드하지 않고 메모리만 정리
            self.memory_manager.clear_gpu_memory()

    def check_model_loaded(self) -> bool:
        """
        모델이 로드되었는지 확인하고 필요시 로드

        Returns:
            bool: 모델 로드 여부
        """
        if not self.model_loaded:
            logger.info("Model not loaded, loading now...")
            self.load_model()
        return self.model_loaded

    def _set_random_seeds(self, seed: int) -> None:
        """
        재현성을 위한 랜덤 시드 설정

        Args:
            seed: 랜덤 시드 값
        """
        torch.manual_seed(seed)
        if self.memory_manager.cuda_device == "cuda":
            torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    def _prepare_model_input(self, prompt: str) -> torch.Tensor:
        """
        모델용 입력 텐서 준비

        Args:
            prompt: 텍스트 프롬프트

        Returns:
            torch.Tensor: 인코딩된 입력 텐서
        """
        # 대화 형식 포맷팅
        messages = [
            {"role": "User", "content": prompt},
            {"role": "Assistant", "content": ""},
        ]

        # 템플릿 적용 및 이미지 태그 추가
        text = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=messages,
            sft_format=self.vl_chat_processor.sft_format,
            system_prompt="",
        )
        text += self.vl_chat_processor.image_start_tag

        # 텐서로 인코딩
        return torch.LongTensor(self.tokenizer.encode(text))

    def unpack(
        self, dec: torch.Tensor, width: int, height: int, parallel_size: int = 3
    ) -> np.ndarray:
        """
        디코딩된 텐서를 이미지 배열로 변환

        Args:
            dec: 디코딩된 텐서
            width: 이미지 너비
            height: 이미지 높이
            parallel_size: 병렬 이미지 수

        Returns:
            np.ndarray: 이미지 배열
        """
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
        return dec

    def _process_generated_patches(
        self, patches: torch.Tensor, width: int, height: int, parallel_size: int
    ) -> List[Image.Image]:
        """
        생성된 패치를 PIL 이미지로 처리

        Args:
            patches: 생성된 이미지 패치
            width: 이미지 너비
            height: 이미지 높이
            parallel_size: 병렬 이미지 수

        Returns:
            List[Image.Image]: PIL 이미지 리스트
        """
        # 텐서를 넘파이 배열로 언팩
        images = self.unpack(
            patches, width // 16 * 16, height // 16 * 16, parallel_size
        )
        image_list = []

        # 각 이미지 처리
        for i in range(parallel_size):
            img_array = images[i]
            img = Image.fromarray(img_array)
            img_resized = img.resize((384, 384), Image.Resampling.LANCZOS)
            image_list.append(img_resized)

        return image_list

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        width: int,
        height: int,
        temperature: float = 1,
        parallel_size: int = 2,
        cfg_weight: float = 5,
        image_token_num_per_image: int = 576,
        patch_size: int = 16,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        입력 토큰으로부터 이미지 생성

        Args:
            input_ids: 입력 토큰 ID
            width: 이미지 너비
            height: 이미지 높이
            temperature: 샘플링 온도
            parallel_size: 병렬 이미지 수
            cfg_weight: Classifier-free guidance 가중치
            image_token_num_per_image: 이미지당 이미지 토큰 수
            patch_size: 이미지 패치 크기

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 생성된 토큰과 패치
        """
        # 입력 검증
        if width <= 0 or height <= 0:
            msg = f"Invalid dimensions: width={width}, height={height}"
            logger.error(msg)
            raise ValueError(msg)

        if parallel_size <= 0:
            msg = f"Invalid parallel_size: {parallel_size}"
            logger.error(msg)
            raise ValueError(msg)

        # 모델 로드 확인
        if not self.check_model_loaded():
            msg = "Failed to load model"
            logger.error(msg)
            raise RuntimeError(msg)

        # GPU 메모리 정리 및 현재 상태 로깅
        self.memory_manager.clear_gpu_memory()
        logger.info(f"Starting generation of {parallel_size} parallel images")
        logger.debug(
            f"Generation parameters: width={width}, height={height}, temperature={temperature}, cfg_weight={cfg_weight}"
        )

        import time

        generation_start_time = time.time()

        try:
            # 입력 토큰 준비
            tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int)
            if self.memory_manager.cuda_device == "cuda":
                tokens = tokens.to(device=self.memory_manager.cuda_device)

            # 토큰 초기화
            for i in range(parallel_size * 2):
                tokens[i, :] = input_ids
                if i % 2 != 0:
                    tokens[i, 1:-1] = self.vl_chat_processor.pad_id

            # 입력 임베딩 가져오기
            inputs_embeds = self.vl_gpt.language_model.get_input_embeddings()(tokens)

            # 생성된 토큰 초기화
            generated_tokens = torch.zeros(
                (parallel_size, image_token_num_per_image), dtype=torch.int
            )
            if self.memory_manager.cuda_device == "cuda":
                generated_tokens = generated_tokens.cuda()

            # 효율적인 생성을 위한 이전 키-값
            pkv = None

            # 생성 시간 추적
            token_times = []
            token_start = time.time()

            # 토큰 하나씩 생성
            for i in range(image_token_num_per_image):
                # 모델 출력 가져오기
                outputs = self.vl_gpt.language_model.model(
                    inputs_embeds=inputs_embeds, use_cache=True, past_key_values=pkv
                )
                pkv = outputs.past_key_values
                hidden_states = outputs.last_hidden_state

                # 로짓 가져오기 및 classifier-free guidance 적용
                logits = self.vl_gpt.gen_head(hidden_states[:, -1, :])
                logit_cond = logits[0::2, :]
                logit_uncond = logits[1::2, :]
                logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)

                # 다음 토큰 샘플링
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated_tokens[:, i] = next_token.squeeze(dim=-1)

                # 다음 반복을 위한 준비
                next_token = torch.cat(
                    [next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1
                ).view(-1)
                img_embeds = self.vl_gpt.prepare_gen_img_embeds(next_token)
                inputs_embeds = img_embeds.unsqueeze(dim=1)

                # 진행 상황 콜백 업데이트
                progress = (i + 1) / image_token_num_per_image * 100
                if self.progress_callback:
                    self.progress_callback(progress)

                # 토큰 생성 시간 주기적으로 추적
                if i % (image_token_num_per_image // 10) == 0:
                    now = time.time()
                    if i > 0:
                        token_times.append(now - token_start)
                    token_start = now

                    # 메모리 사용량 로깅
                    if (
                        self.memory_manager.cuda_device == "cuda"
                        and torch.cuda.is_available()
                    ):
                        current_mem = torch.cuda.memory_allocated() / 1024**2
                        logger.debug(
                            f"Token {i}/{image_token_num_per_image} ({progress:.1f}%), "
                            f"memory: {current_mem:.2f} MB"
                        )

            # 평균 토큰 생성 시간 계산
            if token_times:
                avg_time_per_token = (
                    sum(token_times)
                    / len(token_times)
                    / (image_token_num_per_image // 10)
                )
                logger.debug(
                    f"Average time per token: {avg_time_per_token*1000:.2f} ms"
                )

            # 디코딩 전 로깅
            logger.debug("Token generation complete, decoding to image")

            # 생성된 토큰을 이미지 패치로 디코딩
            try:
                patches = self.vl_gpt.gen_vision_model.decode_code(
                    generated_tokens.to(dtype=torch.int),
                    shape=[parallel_size, 8, width // patch_size, height // patch_size],
                )
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    logger.error("CUDA out of memory during decoding")
                    self.memory_manager.clear_gpu_memory()
                    raise MemoryError("Insufficient GPU memory for decoding") from e
                raise

            # 생성 완료 로깅
            generation_time = time.time() - generation_start_time
            logger.info(
                f"Image generation complete, time taken: {generation_time:.2f} seconds"
            )

            # 메모리 정리
            self.memory_manager.clear_gpu_memory()

            return generated_tokens.to(dtype=torch.int), patches

        except ValueError as e:
            # 입력 검증 오류
            logger.error(f"Invalid input parameters: {str(e)}")
            self.memory_manager.clear_gpu_memory()
            raise

        except MemoryError as e:
            # 메모리 관련 오류
            logger.error(f"Memory error during generation: {str(e)}")
            self.memory_manager.clear_gpu_memory()
            raise

        except RuntimeError as e:
            # 런타임 오류 (주로 PyTorch에서)
            error_msg = str(e)
            if "CUDA out of memory" in error_msg:
                logger.error(f"CUDA out of memory: {error_msg}")
                self.memory_manager.clear_gpu_memory()
                raise MemoryError(f"GPU memory insufficient: {error_msg}") from e
            else:
                logger.error(f"Runtime error in generate method: {error_msg}")
                self.memory_manager.clear_gpu_memory()
                raise

        except Exception as e:
            # 예상치 못한 오류에 대한 포괄적 처리
            logger.exception(f"Unexpected error in generate method: {str(e)}")
            self.memory_manager.clear_gpu_memory()
            raise

        finally:
            # 위의 예외 처리에서 누락된 경우에도 항상 메모리 정리
            self.memory_manager.clear_gpu_memory()

    @torch.inference_mode()
    def generate_image(
        self, prompt: str, seed: Optional[int] = None, guidance: float = 5.0
    ) -> List[Image.Image]:
        """
        텍스트 프롬프트를 기반으로 이미지 생성

        Args:
            prompt: 텍스트 프롬프트
            seed: 재현성을 위한 랜덤 시드
            guidance: 가이던스 스케일

        Returns:
            List[Image.Image]: 생성된 이미지 리스트
        """
        try:
            with self.model_context():
                # 파라미터 설정
                seed = seed if seed is not None else 12345
                width, height = 384, 384
                parallel_size = 2

                # 모든 랜덤 생성기에 시드 설정
                self._set_random_seeds(seed)

                # 모델 입력 준비
                input_ids = self._prepare_model_input(prompt)

                # 이미지 토큰 생성 및 패치로 디코딩
                _, patches = self.generate(
                    input_ids,
                    width // 16 * 16,
                    height // 16 * 16,
                    cfg_weight=guidance,
                    parallel_size=parallel_size,
                )

                # 생성된 패치를 PIL 이미지로 처리
                return self._process_generated_patches(
                    patches, width, height, parallel_size
                )

        except Exception as e:
            logger.error(f"Error during image generation: {str(e)}")
            raise
        finally:
            # 메모리 정리 보장
            self.memory_manager.clear_gpu_memory()


# 싱글톤 인스턴스 생성
image_generator = ImageGenerator()

"""
이미지 생성의 핵심 기능 모듈

이 모듈은 딥러닝 모델을 이용한 이미지 생성의 기본 기능을 제공합니다.
GPU 메모리 관리, 모델 로드/언로드, 이미지 생성 프로세스를 담당합니다.
"""

import torch
import numpy as np
import gc
import logging
import os
import psutil
from PIL import Image
from typing import List, Callable, Optional, Tuple, Dict, Union
from contextlib import contextmanager

from transformers import AutoConfig, AutoModelForCausalLM
from src.janus.janus.models import VLChatProcessor

from app.core.config import settings

logger = logging.getLogger(__name__)

# PyTorch 메모리 단편화 방지 환경 변수 설정
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class MemoryManager:
    """
    GPU 메모리를 효율적으로 관리하는 헬퍼 클래스
    메모리 최적화 기능 추가
    """

    def __init__(self):
        """메모리 관리자 초기화"""
        self._memory_usage_log = []
        self._max_vram_usage = 0
        self._available_devices = self._detect_available_devices()
        self._force_single_gpu = True  # 항상 단일 GPU만 사용
        self._current_device = self._select_best_device()

        # Janus-7B vs Janus-1B 감지 및 모델 크기 제한
        self.model_size_gb_threshold = 4  # 기본 임계값 설정

        logger.info(f"Available devices: {self._available_devices}")
        logger.info(f"Selected device: {self._current_device}")
        logger.info(f"Forcing single GPU mode: {self._force_single_gpu}")

        # 시스템 메모리 정보 로깅
        self._log_system_memory()

    def _log_system_memory(self):
        """시스템 메모리 정보 로깅"""
        # 시스템 메모리 정보
        system_memory = psutil.virtual_memory()
        logger.info(
            f"System memory: {system_memory.total / (1024**3):.2f} GB total, "
            f"{system_memory.available / (1024**3):.2f} GB available"
        )

        # GPU 메모리 정보
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    total_memory = torch.cuda.get_device_properties(i).total_memory / (
                        1024**3
                    )
                    allocated_memory = torch.cuda.memory_allocated(i) / (1024**3)
                    free_memory = total_memory - allocated_memory
                    logger.info(
                        f"GPU {i}: {total_memory:.2f} GB total, "
                        f"{allocated_memory:.2f} GB allocated, "
                        f"{free_memory:.2f} GB free"
                    )
                except Exception as e:
                    logger.warning(f"Error checking GPU {i} memory: {e}")

    def _detect_available_devices(self) -> List[str]:
        """
        사용 가능한 모든 CUDA 장치 탐지
        """
        devices = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(f"cuda:{i}")

        # 사용 가능한 CUDA 장치가 없으면 CPU 추가
        if not devices:
            devices.append("cpu")

        return devices

    def _select_best_device(self) -> str:
        """
        가장 메모리가 여유 있는 장치 선택
        단일 GPU 모드에서는 첫 번째 GPU만 사용
        """
        # 강제 단일 GPU 모드 - 항상 cuda:0 사용 (사용 가능한 경우)
        if self._force_single_gpu and torch.cuda.is_available():
            logger.info("Forcing use of first GPU (cuda:0) only")
            return "cuda:0"

        if not self._available_devices or len(self._available_devices) == 1:
            return self._available_devices[0] if self._available_devices else "cpu"

        # GPU 메모리 사용량 확인
        free_memory = {}
        for device in self._available_devices:
            if device.startswith("cuda"):
                device_idx = int(device.split(":")[-1])
                try:
                    free_memory[device] = torch.cuda.get_device_properties(
                        device_idx
                    ).total_memory - torch.cuda.memory_allocated(device_idx)
                except Exception as e:
                    logger.warning(f"Error checking memory for {device}: {e}")
                    continue

        # 가장 메모리가 많이 남은 장치 선택
        if free_memory:
            best_device = max(free_memory.keys(), key=lambda k: free_memory[k])
            return best_device

        return "cpu"

    def get_current_device(self) -> str:
        """
        현재 선택된 장치 반환
        """
        return self._current_device

    def switch_to_device(self, device: Optional[str] = None) -> str:
        """
        특정 장치로 전환 (지정하지 않으면 최적 장치 선택)
        """
        if self._force_single_gpu and torch.cuda.is_available():
            # 강제 단일 GPU 모드에서는 항상 cuda:0 반환
            self._current_device = "cuda:0"
            logger.info(f"Forced switch to cuda:0 (single GPU mode)")
            return self._current_device

        if device and device in self._available_devices:
            self._current_device = device
        else:
            self._current_device = self._select_best_device()

        logger.info(f"Switched to device: {self._current_device}")
        return self._current_device

    def clear_gpu_memory(self) -> None:
        """
        GPU 메모리를 철저히 정리하고 가비지 컬렉션 실행
        """
        # 파이썬 가비지 컬렉션 - 먼저 실행
        gc.collect()

        if self._current_device.startswith("cuda"):
            # CUDA 캐시 비우기
            torch.cuda.empty_cache()

            # 메모리 상태 로깅
            try:
                current_device_idx = int(self._current_device.split(":")[-1])
                current_mem = torch.cuda.memory_allocated(current_device_idx) / 1024**3
                max_mem = torch.cuda.max_memory_allocated(current_device_idx) / 1024**3
                logger.debug(
                    f"Current GPU memory usage ({self._current_device}): {current_mem:.2f} GB"
                )
                logger.debug(
                    f"Maximum GPU memory usage ({self._current_device}): {max_mem:.2f} GB"
                )

                # 메모리 사용량 추적
                self._memory_usage_log.append(current_mem)
                self._max_vram_usage = max(self._max_vram_usage, max_mem)
            except Exception as e:
                logger.warning(f"Error logging memory usage: {e}")

            logger.info(f"GPU memory cleanup complete on {self._current_device}")

    def is_cuda_available(self) -> bool:
        """
        CUDA 사용 가능 여부 확인
        """
        return self._current_device.startswith("cuda")

    def get_device_count(self) -> int:
        """
        사용 가능한 GPU 장치 수 반환
        """
        return (
            1
            if self._force_single_gpu and torch.cuda.is_available()
            else torch.cuda.device_count() if torch.cuda.is_available() else 0
        )

    def get_available_gpu_memory(self, device_idx: int = 0) -> float:
        """
        사용 가능한 GPU 메모리 반환 (GB 단위)

        Args:
            device_idx: GPU 인덱스

        Returns:
            float: 사용 가능한 메모리 (GB)
        """
        if not torch.cuda.is_available():
            return 0.0

        try:
            total = torch.cuda.get_device_properties(device_idx).total_memory / (
                1024**3
            )
            allocated = torch.cuda.memory_allocated(device_idx) / (1024**3)
            return total - allocated
        except Exception as e:
            logger.warning(f"Error checking available GPU memory: {e}")
            return 0.0

    def check_model_fits_in_memory(
        self, model_size_gb: float, device: Optional[str] = None
    ) -> bool:
        """
        모델이 메모리에 맞는지 확인

        Args:
            model_size_gb: 모델 크기 (GB)
            device: 확인할 장치 (지정하지 않으면 현재 장치)

        Returns:
            bool: 모델이 메모리에 맞는지 여부
        """
        target_device = device if device is not None else self._current_device

        if target_device == "cpu":
            # CPU 메모리 확인
            available_memory = psutil.virtual_memory().available / (1024**3)
            fits = available_memory > model_size_gb * 1.5  # 1.5배 여유 공간 확보
            logger.info(
                f"Model size: {model_size_gb:.2f} GB, Available CPU memory: {available_memory:.2f} GB, Fits: {fits}"
            )
            return fits
        else:
            # GPU 메모리 확인
            device_idx = (
                int(target_device.split(":")[-1]) if ":" in target_device else 0
            )
            available_memory = self.get_available_gpu_memory(device_idx)
            fits = available_memory > model_size_gb * 1.2  # 1.2배 여유 공간 확보
            logger.info(
                f"Model size: {model_size_gb:.2f} GB, Available GPU memory: {available_memory:.2f} GB, Fits: {fits}"
            )
            return fits

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
            cls._instance.device = None
            cls._instance.use_8bit = False  # 8비트 양자화 사용 여부
            cls._instance.use_4bit = False  # 4비트 양자화 사용 여부
            cls._instance.use_low_cpu_mem_usage = True  # 낮은 CPU 메모리 사용
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

        # 모델 로드 시에 사용할 장치 설정 - 항상 단일 GPU 사용
        self.device = self.memory_manager.get_current_device()
        self.use_cuda = settings.USE_CUDA and self.memory_manager.is_cuda_available()

        # 모델 크기 감지 (Janus-7B vs Janus-1B)
        self.model_size = self._detect_model_size()

        # 메모리 상황에 따라 양자화 설정 자동 조정
        self._adjust_quantization_settings()

        logger.info(
            f"ImageGenerator initialized: device={self.device}, use_cuda={self.use_cuda}"
        )
        logger.info(
            f"Model settings: model_size={self.model_size}GB, use_8bit={self.use_8bit}, use_4bit={self.use_4bit}"
        )
        self._is_initialized = True

    def _detect_model_size(self) -> float:
        """
        모델 크기 감지 (Janus-7B vs Janus-1B)
        모델 이름에서 크기 추정

        Returns:
            float: 모델 크기 추정값 (GB)
        """
        model_size_gb = 2.0  # 기본값 (알 수 없는 경우)

        try:
            # 모델 경로에서 크기 정보 추출
            model_name = os.path.basename(self.model_path)

            if "7B" in model_name:
                model_size_gb = 14.0  # Janus-7B 예상 크기
                logger.info(
                    f"Detected Janus-7B model, estimated size: {model_size_gb}GB"
                )
            elif "1B" in model_name:
                model_size_gb = 2.0  # Janus-1B 예상 크기
                logger.info(
                    f"Detected Janus-1B model, estimated size: {model_size_gb}GB"
                )
            else:
                logger.warning(
                    f"Unknown model size from path: {self.model_path}, using default estimate"
                )
        except Exception as e:
            logger.warning(f"Error detecting model size: {e}")

        return model_size_gb

    def _adjust_quantization_settings(self) -> None:
        """
        메모리 상황에 따라 양자화 설정 자동 조정
        """
        # 현재 사용 가능한 메모리 확인
        if self.use_cuda:
            device_idx = int(self.device.split(":")[-1]) if ":" in self.device else 0
            available_memory_gb = self.memory_manager.get_available_gpu_memory(
                device_idx
            )
        else:
            available_memory_gb = psutil.virtual_memory().available / (1024**3)

        logger.info(
            f"Available memory: {available_memory_gb:.2f}GB, Model size: {self.model_size}GB"
        )

        # 메모리 상황에 따라 양자화 설정 자동 조정
        if available_memory_gb < self.model_size:
            # 메모리가 부족한 경우 더 강력한 양자화 적용
            if available_memory_gb < self.model_size / 2:
                # 매우 부족한 경우 4비트 적용
                self.use_4bit = True
                self.use_8bit = False
                logger.info(f"Auto-enabling 4-bit quantization due to limited memory")
            else:
                # 부족한 경우 8비트 적용
                self.use_8bit = True
                self.use_4bit = False
                logger.info(f"Auto-enabling 8-bit quantization due to limited memory")
        else:
            # 메모리가 충분한 경우 기본 설정 유지
            self.use_8bit = False
            self.use_4bit = False
            logger.info(f"Using full precision model")

    def _move_model_to_device(self, target_device: str) -> None:
        """
        모델의 모든 구성 요소를 지정된 장치로 이동

        Args:
            target_device: 이동할 대상 장치
        """
        if not self.model_loaded or self.vl_gpt is None:
            logger.warning("Cannot move model: Model not loaded")
            return

        try:
            logger.info(f"Moving model to device: {target_device}")

            # 전체 모델을 한 번에 장치로 이동 - 가장 안전한 방법
            self.vl_gpt = self.vl_gpt.to(target_device)

            # 디버깅: 모델 주요 구성 요소의 장치 확인
            if hasattr(self.vl_gpt, "gen_vision_model"):
                if hasattr(self.vl_gpt.gen_vision_model, "quantize") and hasattr(
                    self.vl_gpt.gen_vision_model.quantize, "embedding"
                ):
                    embedding_device = (
                        self.vl_gpt.gen_vision_model.quantize.embedding.device
                    )
                    logger.debug(f"Embedding device after move: {embedding_device}")

                if hasattr(self.vl_gpt.gen_vision_model, "post_quant_conv"):
                    post_quant_device = (
                        self.vl_gpt.gen_vision_model.post_quant_conv.weight.device
                    )
                    logger.debug(
                        f"Post quant conv device after move: {post_quant_device}"
                    )

            logger.info(f"Model successfully moved to {target_device}")
            self.device = target_device

        except Exception as e:
            logger.error(f"Error moving model to device {target_device}: {str(e)}")
            raise

    def _validate_model_on_device(self) -> bool:
        """
        모델의 모든 중요 구성 요소가 동일한 장치에 있는지 확인

        Returns:
            bool: 모든 구성 요소가 동일한 장치에 있는지 여부
        """
        if not self.model_loaded or self.vl_gpt is None:
            return False

        try:
            target_device = self.device
            devices = set()

            # gen_vision_model 장치 확인
            if hasattr(self.vl_gpt, "gen_vision_model"):
                # 모델 자체에서 파라미터 확인
                found_param = False
                for param in self.vl_gpt.gen_vision_model.parameters():
                    devices.add(str(param.device))
                    found_param = True
                    break

                if not found_param:
                    # 파라미터가 없으면 버퍼 확인
                    for buf in self.vl_gpt.gen_vision_model.buffers():
                        devices.add(str(buf.device))
                        break

                # quantize 구성 요소 확인
                if hasattr(self.vl_gpt.gen_vision_model, "quantize"):
                    # embedding 확인 (안전하게)
                    if hasattr(self.vl_gpt.gen_vision_model.quantize, "embedding"):
                        embedding = self.vl_gpt.gen_vision_model.quantize.embedding

                        # 다양한 방법으로 장치 정보 확인 시도
                        if hasattr(embedding, "weight"):
                            # 가중치를 통해 장치 확인
                            devices.add(str(embedding.weight.device))
                        elif hasattr(embedding, "parameters"):
                            # 파라미터 메서드를 통해 확인
                            param_found = False
                            for param in embedding.parameters():
                                devices.add(str(param.device))
                                param_found = True
                                break

                            if not param_found and hasattr(embedding, "buffers"):
                                # 버퍼를 통해 확인
                                for buf in embedding.buffers():
                                    devices.add(str(buf.device))
                                    break
                        elif isinstance(embedding, torch.Tensor):
                            # 텐서 자체인 경우
                            devices.add(str(embedding.device))

                    # quantize 자체 확인
                    for param in self.vl_gpt.gen_vision_model.quantize.parameters():
                        devices.add(str(param.device))
                        break

                # post_quant_conv 확인
                if hasattr(self.vl_gpt.gen_vision_model, "post_quant_conv"):
                    if hasattr(self.vl_gpt.gen_vision_model.post_quant_conv, "weight"):
                        devices.add(
                            str(
                                self.vl_gpt.gen_vision_model.post_quant_conv.weight.device
                            )
                        )
                    else:
                        for (
                            param
                        ) in self.vl_gpt.gen_vision_model.post_quant_conv.parameters():
                            devices.add(str(param.device))
                            break

            # gen_head 장치 확인
            if hasattr(self.vl_gpt, "gen_head"):
                for param in self.vl_gpt.gen_head.parameters():
                    devices.add(str(param.device))
                    break

            logger.debug(f"Model component devices: {devices}")

            # 모든 장치가 동일한지 확인
            is_consistent = len(devices) <= 1
            if not is_consistent:
                logger.warning(f"Model components are on different devices: {devices}")

            return is_consistent

        except Exception as e:
            logger.error(f"Error validating model devices: {str(e)}")
            return False

    def _synchronize_model_devices(self) -> None:
        """
        모델의 모든 구성 요소가 동일한 장치에 있는지 확인하고 동기화
        """
        if not self.model_loaded or self.vl_gpt is None:
            return

        try:
            # 현재 모델 상태 확인
            is_consistent = self._validate_model_on_device()

            # 일관성이 없으면 명시적으로 모든 구성 요소를 동일한 장치로 이동
            if not is_consistent:
                target_device = self.device
                logger.info(
                    f"Synchronizing model components to device: {target_device}"
                )

                # 전체 모델을 한 번에 이동
                self._move_model_to_device(target_device)

                # 이동 후 다시 검증
                is_consistent = self._validate_model_on_device()
                if not is_consistent:
                    logger.warning(
                        "Model components still on different devices after synchronization"
                    )

        except Exception as e:
            logger.error(f"Error synchronizing model devices: {str(e)}")

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

        # 메모리 상황 재평가 및 양자화 설정 조정
        self._adjust_quantization_settings()

        try:
            # 모델 경로 확인
            if not os.path.exists(self.model_path):
                logger.error(f"Model path does not exist: {self.model_path}")
                raise FileNotFoundError(f"Model path not found: {self.model_path}")

            # 모델 설정 로드
            setting = AutoConfig.from_pretrained(self.model_path)
            language_config = setting.language_config
            language_config._attn_implementation = "eager"

            # 장치에 따른 데이터 타입 설정
            use_cuda = self.use_cuda
            dtype = torch.bfloat16 if use_cuda else torch.float32

            # 현재 선택된 장치 (강제로 cuda:0 사용)
            device = self.device

            # device_map 설정 - 단일 GPU 강제 모드에서는 항상 None 사용
            device_map = None

            # 메모리 효율적인 로드 설정
            load_kwargs = {
                "language_config": language_config,
                "trust_remote_code": True,
                "device_map": device_map,  # 항상 None으로 설정하여 단일 장치 사용
                "low_cpu_mem_usage": self.use_low_cpu_mem_usage,
                "torch_dtype": dtype,
            }

            # 양자화 설정 적용
            if self.use_4bit:
                logger.info("Using 4-bit quantization for model loading")
                try:
                    from transformers import BitsAndBytesConfig

                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=(
                            torch.bfloat16 if use_cuda else torch.float32
                        ),
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                    load_kwargs["quantization_config"] = quantization_config
                except ImportError as e:
                    logger.warning(
                        f"bitsandbytes not available, falling back to 8-bit: {e}"
                    )
                    self.use_4bit = False
                    self.use_8bit = True

            if self.use_8bit:
                logger.info("Using 8-bit quantization for model loading")
                load_kwargs["load_in_8bit"] = True

            logger.info(
                f"Loading model to device: {device}, device_map: {device_map}, "
                f"8bit: {self.use_8bit}, 4bit: {self.use_4bit}"
            )

            # 모델 로드 (장치에 맞게)
            self.vl_gpt = AutoModelForCausalLM.from_pretrained(
                self.model_path, **load_kwargs
            )

            # 평가 모드로 설정
            self.vl_gpt = self.vl_gpt.eval()

            # 양자화를 사용하지 않는 경우, 명시적으로 모델을 지정된 장치로 이동
            if not self.use_4bit and not self.use_8bit and use_cuda:
                logger.info(f"Moving model to specific device: {device}")
                self.vl_gpt = self.vl_gpt.to(device)

                # 모든 구성 요소를 동일한 장치로 이동
                self._synchronize_model_devices()

            # CUDA 사용 시 추론 속도 향상을 위한 설정
            if use_cuda:
                torch.backends.cudnn.benchmark = True

            # 프로세서 및 토크나이저 로드
            self.vl_chat_processor = VLChatProcessor.from_pretrained(self.model_path)
            self.tokenizer = self.vl_chat_processor.tokenizer

            self.model_loaded = True
            logger.info(f"Model load successful on {device}")

            # 메모리 정리
            self.memory_manager.clear_gpu_memory()
            return True

        except Exception as e:
            logger.error(f"Model load failed: {str(e)}")
            self.memory_manager.clear_gpu_memory()
            self.model_loaded = False

            # OOM 오류 처리 - Janus-7B에서 Janus-1B로 대체 시도
            if "CUDA out of memory" in str(e) and "7B" in self.model_path:
                alt_model_path = self.model_path.replace("7B", "1B")
                if os.path.exists(alt_model_path):
                    logger.info(f"Trying to load smaller model: {alt_model_path}")
                    self.model_path = alt_model_path
                    self.model_size = self._detect_model_size()  # 모델 크기 업데이트
                    self._adjust_quantization_settings()  # 양자화 설정 재조정
                    return self.load_model(force_reload=True)
                else:
                    logger.error(f"Smaller model not found at {alt_model_path}")

            # CPU 폴백 시도
            if self.use_cuda and "CUDA out of memory" in str(e):
                logger.info("Falling back to CPU due to CUDA memory issues")
                self.use_cuda = False
                self.device = "cpu"
                return self.load_model(force_reload=True)

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
            # 모델 컴포넌트 정리
            if self.vl_gpt is not None:
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
        if self.memory_manager.is_cuda_available():
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

    def _to_device(
        self, tensor: torch.Tensor, device: Optional[str] = None
    ) -> torch.Tensor:
        """
        텐서를 지정된 장치로 이동 (오류 처리 포함)

        Args:
            tensor: 이동할 텐서
            device: 대상 장치 (지정하지 않으면 현재 장치 사용)

        Returns:
            torch.Tensor: 이동된 텐서
        """
        if tensor is None:
            return None

        target_device = device if device is not None else self.device

        try:
            return tensor.to(target_device)
        except Exception as e:
            logger.warning(f"Failed to move tensor to {target_device}: {e}")
            # 실패 시 CPU로 폴백
            try:
                return tensor.cpu()
            except Exception as inner_e:
                logger.error(f"Failed to move tensor to CPU: {inner_e}")
                return tensor

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

        # 메모리 정리 및 로깅
        self.memory_manager.clear_gpu_memory()
        logger.info(
            f"Starting generation of {parallel_size} parallel images on {self.device}"
        )
        logger.debug(
            f"Generation parameters: width={width}, height={height}, temperature={temperature}, cfg_weight={cfg_weight}"
        )

        # 모델 구성 요소가 동일한 장치에 있는지 확인 - 매우 중요
        self._synchronize_model_devices()

        import time

        generation_start_time = time.time()

        try:
            # 입력 텐서를 현재 장치로 이동
            input_ids = self._to_device(input_ids)

            # 입력 토큰 준비
            tokens = torch.zeros(
                (parallel_size * 2, len(input_ids)), dtype=torch.int, device=self.device
            )

            # 토큰 초기화
            for i in range(parallel_size * 2):
                tokens[i, :] = input_ids
                if i % 2 != 0:
                    tokens[i, 1:-1] = self.vl_chat_processor.pad_id

            # 임베딩 가져오기 전 명시적 동기화
            self._synchronize_model_devices()

            # 입력 임베딩 가져오기
            inputs_embeds = self.vl_gpt.language_model.get_input_embeddings()(tokens)

            # 생성된 토큰 초기화
            generated_tokens = torch.zeros(
                (parallel_size, image_token_num_per_image),
                dtype=torch.int,
                device=self.device,
            )

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

                # 장치 일관성 유지
                next_token = self._to_device(next_token)

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
                    if self.use_cuda:
                        try:
                            current_device_idx = (
                                int(self.device.split(":")[-1])
                                if ":" in self.device
                                else 0
                            )
                            current_mem = (
                                torch.cuda.memory_allocated(current_device_idx)
                                / 1024**3
                            )
                            logger.debug(
                                f"Token {i}/{image_token_num_per_image} ({progress:.1f}%), "
                                f"memory: {current_mem:.2f} GB"
                            )
                        except Exception as e:
                            logger.warning(f"Error logging memory usage: {e}")

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

            # 안전하게 디코딩 시도
            try:
                # CPU에서 디코딩 수행 - 항상 CPU에서 디코딩하여 장치 불일치 방지
                logger.info("Performing decoding on CPU for consistency")

                # 전체 모델을 CPU로 이동
                cpu_vl_gpt = self.vl_gpt.cpu()
                cpu_tokens = generated_tokens.cpu()

                # CPU에서 디코딩 수행 (안전한 방법)
                patches = cpu_vl_gpt.gen_vision_model.decode_code(
                    cpu_tokens,
                    shape=[parallel_size, 8, width // patch_size, height // patch_size],
                )

                # 디코딩 후 다시 원래 장치로 복원
                if self.use_cuda:
                    self.vl_gpt = self.vl_gpt.to(self.device)
                    patches = patches.to(self.device)

                    # 모델 동기화 재확인
                    self._synchronize_model_devices()

            except RuntimeError as e:
                logger.error(f"Runtime error in decode_code: {str(e)}")
                # CUDA 메모리 부족 또는 장치 불일치 오류 모두 CPU 폴백
                logger.info("Falling back to complete CPU operation")

                # 모든 작업을 CPU로 이동
                self.vl_gpt = self.vl_gpt.cpu()
                cpu_tokens = generated_tokens.cpu()

                # CPU에서 디코딩
                patches = self.vl_gpt.gen_vision_model.decode_code(
                    cpu_tokens,
                    shape=[parallel_size, 8, width // patch_size, height // patch_size],
                )

                # 다시 CUDA로 복원 (필요시)
                if self.use_cuda:
                    self.vl_gpt = self.vl_gpt.to(self.device)
                    patches = patches.to(self.device)

            # 생성 완료 로깅
            generation_time = time.time() - generation_start_time
            logger.info(
                f"Image generation complete, time taken: {generation_time:.2f} seconds"
            )

            # 메모리 정리
            self.memory_manager.clear_gpu_memory()

            return generated_tokens, patches

        except ValueError as e:
            logger.error(f"Invalid input parameters: {str(e)}")
            self.memory_manager.clear_gpu_memory()
            raise

        except MemoryError as e:
            logger.error(f"Memory error during generation: {str(e)}")
            self.memory_manager.clear_gpu_memory()
            raise

        except RuntimeError as e:
            logger.error(f"Runtime error in generate method: {str(e)}")
            self.memory_manager.clear_gpu_memory()

            # CUDA 오류인 경우 더 구체적인 메시지 제공
            error_msg = str(e)
            if "CUDA out of memory" in error_msg:
                # OOM 오류 - CPU 모드로 전환
                logger.info("CUDA out of memory. Trying CPU mode...")
                self.use_cuda = False
                self.device = "cpu"

                # 모델 언로드 후 CPU 모드로 재로드
                self.model_loaded = False
                self.load_model(force_reload=True)

                # CPU에서 다시 시도
                return self.generate(
                    input_ids.cpu(),
                    width,
                    height,
                    temperature,
                    parallel_size,
                    cfg_weight,
                    image_token_num_per_image,
                    patch_size,
                )
            elif "Expected all tensors to be on the same device" in error_msg:
                # 장치 불일치 오류 - CPU 모드로 전환
                logger.error("Device mismatch detected. Trying with CPU...")

                # 모든 작업을 CPU에서 수행
                self.use_cuda = False
                self.device = "cpu"

                # 모델을 CPU로 이동
                if self.vl_gpt is not None:
                    self.vl_gpt = self.vl_gpt.cpu()

                # CPU에서 다시 시도
                return self.generate(
                    input_ids.cpu(),
                    width,
                    height,
                    temperature,
                    parallel_size,
                    cfg_weight,
                    image_token_num_per_image,
                    patch_size,
                )
            else:
                raise

        except Exception as e:
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

            # CUDA 메모리 오류 처리
            if "CUDA out of memory" in str(e) and self.use_cuda:
                logger.info("Trying CPU mode due to GPU memory constraints")
                self.use_cuda = False
                self.device = "cpu"

                # CPU 모드로 다시 시도
                return self.generate_image(prompt, seed, guidance)

            raise
        finally:
            # 메모리 정리 보장
            self.memory_manager.clear_gpu_memory()


# 싱글톤 인스턴스 생성
image_generator = ImageGenerator()

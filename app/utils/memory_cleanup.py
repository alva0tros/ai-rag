"""
GPU 메모리 및 리소스 정리를 위한 유틸리티 모듈
서버 종료 시 호출되어 메모리 누수를 방지합니다.
"""

import gc
import logging
import torch
import sys
import os

logger = logging.getLogger(__name__)

def cleanup_resources():
    """
    서버 종료 전 모든 GPU 리소스를 명시적으로 정리
    
    이 함수는 다음 단계로 GPU 메모리를 정리합니다:
    1. 모든 모델 인스턴스 참조 해제
    2. 주요 서비스 모듈 참조 해제
    3. CUDA 캐시 비우기
    4. 가비지 컬렉터 강제 실행
    """
    logger.info("GPU 리소스 정리 시작...")
    
    # 1. 모든 모델 인스턴스 참조 해제
    try:
        # 이미지 모델 언로드
        from app.services.image.image_service import image_service
        if hasattr(image_service, 'image_generator') and hasattr(image_service.image_generator, 'model_loaded'):
            if image_service.image_generator.model_loaded:
                logger.info("이미지 생성 모델 언로드 중...")
                image_service.image_generator.unload_model()
                logger.info("이미지 생성 모델 언로드 완료")
    except Exception as e:
        logger.error(f"이미지 모델 언로드 실패: {str(e)}")
    
    try:
        # 채팅 모델 참조 해제
        from app.services.chat.chat_service import chat_service
        if hasattr(chat_service, 'llm_manager') and hasattr(chat_service.llm_manager, 'llm'):
            if chat_service.llm_manager.llm is not None:
                logger.info("채팅 모델 참조 해제 중...")
                chat_service.llm_manager.llm = None
                logger.info("채팅 모델 참조 해제 완료")
    except Exception as e:
        logger.error(f"채팅 모델 참조 해제 실패: {str(e)}")
    
    # 2. 주요 서비스 모듈 참조 해제
    cleanup_modules = [
        'app.services.image.image_core',
        'app.services.image.image_service',
        'app.services.image.image_prompt',
        'app.services.chat.chat_core',
        'app.services.chat.chat_service'
    ]
    
    for module_name in cleanup_modules:
        if module_name in sys.modules:
            try:
                logger.info(f"모듈 참조 해제 중: {module_name}")
                del sys.modules[module_name]
            except Exception as e:
                logger.error(f"모듈 참조 해제 실패 ({module_name}): {str(e)}")
    
    # 3. CUDA 메모리 정리
    if torch.cuda.is_available():
        try:
            logger.info("CUDA 캐시 비우는 중...")
            torch.cuda.empty_cache()
            
            # CUDA IPC 메모리 정리 (PyTorch 1.8.0+)
            if hasattr(torch.cuda, 'ipc_collect'):
                torch.cuda.ipc_collect()
            
            # 현재 CUDA 디바이스 정보 로깅
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            
            logger.info(f"CUDA 디바이스 수: {device_count}, 현재 디바이스: {current_device}")
            
            # 각 디바이스의 메모리 정보 로깅
            for i in range(device_count):
                mem_info = torch.cuda.memory_stats(i)
                allocated = mem_info.get('allocated_bytes.all.current', 0) / (1024**2)
                reserved = mem_info.get('reserved_bytes.all.current', 0) / (1024**2)
                logger.info(f"디바이스 {i} - 할당됨: {allocated:.2f}MB, 예약됨: {reserved:.2f}MB")
            
            logger.info("CUDA 메모리 정리 완료")
        except Exception as e:
            logger.error(f"CUDA 메모리 정리 실패: {str(e)}")
    
    # 4. Python 가비지 컬렉터 강제 실행
    try:
        logger.info("가비지 컬렉터 강제 실행 중...")
        for _ in range(3):
            collected = gc.collect()
            logger.info(f"가비지 컬렉터: {collected}개 객체 수집")
    except Exception as e:
        logger.error(f"가비지 컬렉션 실패: {str(e)}")
    
    # 5. 최종 메모리 상태 보고
    try:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                free_mem = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_reserved(i)
                free_mem_mb = free_mem / (1024**2)
                logger.info(f"디바이스 {i} 사용 가능한 메모리: {free_mem_mb:.2f}MB")
    except Exception as e:
        logger.error(f"메모리 상태 보고 실패: {str(e)}")
    
    logger.info("모든 GPU 리소스 정리 완료")


def force_gpu_reset():
    """
    경고: 이 함수는 위험하며 주의해서 사용해야 합니다.
    시스템 명령어를 실행하여 강제로 GPU 메모리를 정리합니다.
    이 함수는 일반적인 경우 사용하지 않는 것이 좋으며, 다른 모든 방법이 실패했을 때만 사용하세요.
    """
    import subprocess
    import time
    
    logger.warning("GPU 강제 리셋 시작 - 이 작업은 위험할 수 있습니다!")
    
    try:
        # 현재 시스템에서만 작동하는 명령어입니다
        if os.name == 'posix':  # Linux/Mac
            # Linux에서 NVIDIA driver 상태 재설정
            logger.info("Linux 환경에서 NVIDIA UVM 모듈 재로드 시도 중...")
            subprocess.run(["sudo", "rmmod", "nvidia_uvm"], check=False)
            time.sleep(1)
            subprocess.run(["sudo", "modprobe", "nvidia_uvm"], check=False)
            logger.info("GPU driver 모듈 리셋 완료")
            
        elif os.name == 'nt':  # Windows
            # Windows에서는 다른 방법 필요
            logger.info("Windows 환경에서 GPU 리셋 시도 중...")
            subprocess.run(["nvidia-smi", "-r"], check=False)
            logger.info("GPU driver 리셋 요청 완료")
            
        logger.info("GPU 강제 리셋 완료")
        
    except Exception as e:
        logger.error(f"GPU 강제 리셋 실패: {str(e)}")
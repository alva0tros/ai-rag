"""
AI 멀티모달 서비스 시작 스크립트
GPU 메모리 관리 및 서버 시작/종료 처리 기능 포함
"""

import os
import atexit
import signal
import logging
import sys

# GPU 메모리 최적화 설정
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('server.log')
    ]
)

logger = logging.getLogger(__name__)

# 환경 변수 로드
from dotenv import load_dotenv
load_dotenv()  # .env 파일 로드

# 메모리 정리 유틸리티 임포트
from app.utils.memory_cleanup import cleanup_resources

# 종료 시 호출될 함수 등록
atexit.register(cleanup_resources)

# 서버 상태 변수
server_running = True

# 시그널 핸들러
def signal_handler(sig, frame):
    """
    시그널 수신 시 안전하게 서버를 종료하는 핸들러
    """
    global server_running
    signal_name = {
        signal.SIGINT: "SIGINT",
        signal.SIGTERM: "SIGTERM"
    }.get(sig, str(sig))
    
    logger.info(f"서버 종료 신호 ({signal_name}) 수신... GPU 메모리 정리 시작")
    server_running = False
    
    # 리소스 정리 실행
    cleanup_resources()
    
    logger.info("서버 종료 완료")
    sys.exit(0)

# SIGINT(Ctrl+C), SIGTERM 시그널에 핸들러 등록
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# 시작 시 로깅
logger.info("서버 시작 중...")
logger.info(f"PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")

# GPU 가용성 확인
try:
    import torch
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info(f"사용 가능한 CUDA 디바이스: {device_count}개")
        
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB 단위
            logger.info(f"GPU {i}: {device_name}, 메모리: {total_memory:.2f} GB")
    else:
        logger.warning("CUDA를 사용할 수 없습니다. CPU 모드로 실행됩니다.")
except Exception as e:
    logger.error(f"GPU 정보 확인 실패: {str(e)}")

if __name__ == "__main__":
    try:
        import uvicorn
        
        logger.info("FastAPI 서버 시작...")
        uvicorn.run(
            "app.main:app", 
            host="0.0.0.0", 
            port=8000, 
            workers=1, 
            log_level="info",
            access_log=True
        )
        
    except Exception as e:
        logger.error(f"서버 실행 중 오류 발생: {str(e)}")
        # 오류 발생 시에도 리소스 정리
        cleanup_resources()
        sys.exit(1)
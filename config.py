import os

# 현재 config.py 파일의 디렉터리는 ai-rag 폴더임
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 텍스트 모델 경로
TEXT_MODEL_PATH = os.path.join(BASE_DIR, "models", "text", "DeepSeek-R1-GGUF")
# 이미지 모델 경로
IMAGE_MODEL_PATH = os.path.join(BASE_DIR, "models", "image", "Janus-Pro-7B")

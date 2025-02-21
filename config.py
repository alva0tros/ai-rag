import os

# 현재 config.py 파일의 디렉터리는 ai-rag 폴더임
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# 텍스트 모델 경로
TEXT_MODEL_PATH = os.path.join(BASE_PATH, "models", "text", "DeepSeek-R1-GGUF")

# 이미지 모델 경로
# IMAGE_MODEL_PATH = os.path.join(BASE_DIR, "models", "image", "Janus-Pro-7B")
IMAGE_MODEL_PATH = os.path.join(BASE_PATH, "models", "image", "Janus-Pro-1B")

# 정적 이미지 파일 경로 (예: 챗봇 intro 이미지)
STATIC_IMAGE_PATH = os.path.join(BASE_PATH, "static", "images")

# 생성된 이미지 저장 경로
GENERATED_IMAGE_PATH = os.path.join(BASE_PATH, "generated_images")

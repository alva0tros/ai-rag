import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from dotenv import load_dotenv
import uvicorn

load_dotenv()  # .env 파일 로드

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, workers=1)

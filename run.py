from dotenv import load_dotenv
from app.main import app

load_dotenv()  # .env 파일 로드

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

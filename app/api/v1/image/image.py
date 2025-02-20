import logging
import os
from fastapi import APIRouter, HTTPException
from config import STATIC_IMAGE_PATH

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/image/intro")
async def get_intro_images():

    if not os.path.isdir(STATIC_IMAGE_PATH):
        raise HTTPException(
            status_code=404, detail="Static images directory not found."
        )

    # 이미지 확장자 필터링
    valid_extensions = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")
    image_files = [
        file
        for file in os.listdir(STATIC_IMAGE_PATH)
        if file.lower().endswith(valid_extensions)
    ]

    image_urls = [f"/static/images/{file}" for file in image_files]

    return {"images": image_urls}

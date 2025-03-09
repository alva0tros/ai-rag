"""
Image API endpoints for handling image generation requests.
This module provides routes for image generation, progress tracking, and management.
"""

import logging
import os
from typing import Dict, List

from nanoid import generate as nanoid
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from sse_starlette.sse import EventSourceResponse

from app.services.image_service import image_service
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/image/prompt")
async def generate_from_prompt(request: Request) -> EventSourceResponse:
    """
    Generate images based on a text prompt.

    Args:
        request: FastAPI request object containing prompt and metadata

    Returns:
        EventSourceResponse: Stream of progress events and image URLs
    """
    try:
        user_message = await request.json()

        # Extract required fields
        message = user_message.get("message")
        user_id = user_message.get("user_id")
        conversation_id = user_message.get("conversation_id")
        message_id = user_message.get("message_id")

        # Validate required fields
        if not message or not user_id or not message_id:
            logger.error("Missing required fields")
            raise HTTPException(status_code=400, detail="Missing required fields")

        # Generate new conversation_id if not provided
        if not conversation_id:
            conversation_id = nanoid(size=12)
            logger.info(f"Generated new conversation ID: {conversation_id}")

        logger.info(f"Received image generation prompt from user {user_id}")

        # Stream progress and generate images using the service
        return EventSourceResponse(
            image_service.stream_generation_progress(
                message, None, 5.0, user_id, conversation_id, message_id
            )
        )

    except HTTPException:
        # Pass through HTTP exceptions
        raise

    except Exception as e:
        logger.exception(f"Image generation failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Image generation failed: {str(e)}"
        )


@router.get("/image/intro")
async def get_intro_images() -> Dict[str, List[str]]:
    """
    Get a list of static intro images.

    Returns:
        Dict[str, List[str]]: Dictionary containing image URLs
    """
    try:
        if not os.path.isdir(settings.STATIC_IMAGE_PATH):
            logger.error("Static images directory not found")
            raise HTTPException(
                status_code=404, detail="Static images directory not found."
            )

        # Filter image files by extension
        valid_extensions = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")

        try:
            image_files = [
                file
                for file in os.listdir(settings.STATIC_IMAGE_PATH)
                if file.lower().endswith(valid_extensions)
            ]
        except Exception as e:
            logger.error(f"Error reading static images directory: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error reading static images directory: {str(e)}",
            )

        image_urls = [f"/static/images/{file}" for file in image_files]
        logger.info(f"Retrieved {len(image_urls)} intro images")

        return {"images": image_urls}

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(f"Error retrieving intro images: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve intro images: {str(e)}"
        )


@router.post("/image/stop")
async def stop_image_generation(request: Request) -> Dict[str, str]:
    """
    Stop an ongoing image generation process.

    Args:
        request: FastAPI request object containing conversation_id

    Returns:
        Dict[str, str]: Status message
    """
    try:
        data = await request.json()
        conversation_id = data.get("conversation_id", "default")

        logger.info(f"Requested to stop image generation for {conversation_id}")

        success = await image_service.stop_image_generation(conversation_id)

        if not success:
            logger.warning(f"No active task found for session: {conversation_id}")
            raise HTTPException(
                status_code=404, detail="No active task found for this session."
            )

        logger.info(f"Image generation stopped for conversation: {conversation_id}")

        return {"message": f"Task for session {conversation_id} has been cancelled."}

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(f"Error stopping image generation: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to stop image generation: {str(e)}"
        )

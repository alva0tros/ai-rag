"""
Image API endpoints for handling image generation requests.
This module provides routes for image generation, progress tracking, and management.
"""

import asyncio
import logging
import os
import json
from typing import Dict, Any, List, Optional

from nanoid import generate as nanoid
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from sse_starlette.sse import EventSourceResponse

from app.services.image_service import image_service, generate_image
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


async def stream_image_progress(
    prompt: str,
    seed: Optional[int],
    guidance: float,
    user_id: int,
    conversation_id: str,
    message_id: str,
) -> EventSourceResponse:
    """
    Stream the image generation progress as Server-Sent Events (SSE).

    Args:
        prompt: Text prompt for image generation
        seed: Random seed for reproducibility
        guidance: Guidance scale for image generation
        user_id: User ID
        conversation_id: Conversation ID
        message_id: Message ID

    Returns:
        EventSourceResponse: SSE response with progress and image data
    """
    # Initialize task data
    task = {
        "images": None,
        "progress": 0.0,
        "progress_event": asyncio.Event(),
        "last_reported_progress": -1,
        "generate_task": None,
    }

    # Register task with conversation ID
    async with asyncio.Lock():
        image_service.tasks[conversation_id] = task

    # Define progress callback
    def update_progress(progress: float) -> None:
        task["progress"] = progress
        task["progress_event"].set()
        logger.info(
            f"Conversation {conversation_id}: Progress updated to {progress:.1f}%"
        )

    # Set progress callback
    image_service.progress_callback = update_progress

    async def generate_images():
        """Execute image generation in the background"""
        try:
            task["images"] = await asyncio.get_event_loop().run_in_executor(
                None, lambda: generate_image(prompt, seed, guidance)
            )
        except Exception as e:
            logger.error(
                f"Conversation {conversation_id}: Image generation failed - {str(e)}"
            )
            raise

    async def event_generator():
        """Generate SSE events for progress and images"""
        try:
            # Start image generation in background
            task["generate_task"] = asyncio.create_task(generate_images())

            # Stream progress events
            while task["progress"] < 100.0:
                await task["progress_event"].wait()
                current_progress = round(task["progress"], 0)

                # Only send updates when progress increases by at least 1%
                if current_progress > task["last_reported_progress"]:
                    task["last_reported_progress"] = current_progress
                    yield {"event": "progress", "data": f"{current_progress}"}

                task["progress_event"].clear()
                await asyncio.sleep(0.1)

            # Wait for image generation to complete
            await task["generate_task"]

            # Ensure 100% progress is sent
            if task["last_reported_progress"] != 100:
                yield {"event": "progress", "data": "100"}

            # Save images and generate URLs
            image_urls = await save_generated_images(
                task["images"], user_id, conversation_id, message_id
            )

            # Send image URLs
            yield {
                "event": "image",
                "data": json.dumps({"image_urls": image_urls}),
            }

        except Exception as e:
            logger.error(f"Error in event generator: {str(e)}")
            yield {"event": "error", "data": str(e)}

        finally:
            # Clean up resources
            if conversation_id in image_service.tasks:
                del image_service.tasks[conversation_id]

    return EventSourceResponse(event_generator())


async def save_generated_images(
    images: List, user_id: int, conversation_id: str, message_id: str
) -> List[str]:
    """
    Save generated images and return their URLs.

    Args:
        images: List of generated image objects
        user_id: User ID
        conversation_id: Conversation ID
        message_id: Message ID

    Returns:
        List[str]: List of image URLs
    """
    image_urls = []
    save_dir = os.path.join(
        settings.GENERATED_IMAGE_PATH, str(user_id), conversation_id, message_id
    )
    os.makedirs(save_dir, exist_ok=True)

    for i, img in enumerate(images):
        file_name = f"img{i}.png"
        file_path = os.path.join(save_dir, file_name)
        img.save(file_path, format="PNG")
        image_url = (
            f"/generated/images/{user_id}/{conversation_id}/{message_id}/{file_name}"
        )
        image_urls.append(image_url)

    return image_urls


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

        logger.info(f"Received image generation prompt from user {user_id}")

        # Stream progress and generate images
        return await stream_image_progress(
            message, None, 5.0, user_id, conversation_id, message_id
        )

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
    if not os.path.isdir(settings.STATIC_IMAGE_PATH):
        logger.error("Static images directory not found")
        raise HTTPException(
            status_code=404, detail="Static images directory not found."
        )

    # Filter image files by extension
    valid_extensions = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")
    image_files = [
        file
        for file in os.listdir(settings.STATIC_IMAGE_PATH)
        if file.lower().endswith(valid_extensions)
    ]

    image_urls = [f"/static/images/{file}" for file in image_files]

    return {"images": image_urls}


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

        if conversation_id not in image_service.tasks:
            logger.warning(f"No active task found for session: {conversation_id}")
            raise HTTPException(
                status_code=404, detail="No active task found for this session."
            )

        # Cancel the task
        task = image_service.tasks[conversation_id]
        if task.get("generate_task") and not task["generate_task"].done():
            task["generate_task"].cancel()

        logger.info(f"Image generation stopped for conversation: {conversation_id}")

        return {"message": f"Task for session {conversation_id} has been cancelled."}

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(f"Error stopping image generation: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to stop image generation: {str(e)}"
        )

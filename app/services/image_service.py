"""
Image Service module for handling image generation.

This module provides the core functionality for image generation using deep learning models.
It implements a singleton pattern for efficient model management and memory utilization.
"""

import torch
import numpy as np
import io
import gc
import logging
from PIL import Image
from typing import List, Callable, Dict, Any, Optional, Union, Tuple
from contextlib import contextmanager

from transformers import AutoConfig, AutoModelForCausalLM
from src.janus.janus.models import VLChatProcessor

from app.core.config import settings

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Helper class to manage GPU memory efficiently.
    """

    def __init__(self):
        """Initialize the memory manager."""
        self._memory_usage_log = []
        self._max_vram_usage = 0
        self.cuda_device = "cuda" if torch.cuda.is_available() else "cpu"

    def clear_gpu_memory(self) -> None:
        """
        Thoroughly clear GPU memory and collect garbage.
        """
        if self.cuda_device == "cuda":
            # Clear CUDA cache
            torch.cuda.empty_cache()
            # Run garbage collection
            gc.collect()

            # Log memory status
            if torch.cuda.is_available():
                current_mem = torch.cuda.memory_allocated() / 1024**2
                max_mem = torch.cuda.max_memory_allocated() / 1024**2
                logger.debug(f"Current GPU memory usage: {current_mem:.2f} MB")
                logger.debug(f"Peak GPU memory usage: {max_mem:.2f} MB")

                # Track memory usage
                self._memory_usage_log.append(current_mem)
                self._max_vram_usage = max(self._max_vram_usage, max_mem)

            logger.info("GPU memory cleared")

    @property
    def memory_usage_history(self) -> List[float]:
        """Get memory usage history."""
        return self._memory_usage_log

    @property
    def max_memory_usage(self) -> float:
        """Get maximum memory usage."""
        return self._max_vram_usage


class ImageService:
    """
    Service for generating images using deep learning models.
    Implements a singleton pattern for efficient resource management.
    """

    _instance = None
    _is_initialized = False

    def __new__(cls, *args, **kwargs):
        """
        Create a singleton instance or return the existing one.
        """
        if cls._instance is None:
            cls._instance = super(ImageService, cls).__new__(cls)
            cls._instance.tasks = {}  # Active tasks dictionary
            cls._instance.model_loaded = False
            cls._instance.vl_gpt = None
            cls._instance.vl_chat_processor = None
            cls._instance.tokenizer = None
            cls._instance.progress_callback = None
        return cls._instance

    def __init__(self, progress_callback: Optional[Callable[[float], None]] = None):
        """
        Initialize the image service with optional progress callback.

        Args:
            progress_callback: Function to call with progress updates
        """
        # Skip initialization if already done
        if self._is_initialized:
            if progress_callback is not None:
                self.progress_callback = progress_callback
            return

        # Set up the service
        self.model_path = settings.IMAGE_MODEL_PATH
        self.memory_manager = MemoryManager()
        self.progress_callback = progress_callback

        logger.info(
            f"Initializing ImageService with device: {self.memory_manager.cuda_device}"
        )
        self._is_initialized = True

    def load_model(self, force_reload: bool = False) -> bool:
        """
        Load the image generation model.

        Args:
            force_reload: Force reload even if model is already loaded

        Returns:
            bool: Success status
        """
        # Skip loading if already loaded and not forcing reload
        if self.model_loaded and not force_reload:
            logger.info("Model already loaded, skipping load")
            return True

        logger.info(f"Loading model from {self.model_path}")
        self.memory_manager.clear_gpu_memory()

        try:
            # Load model configuration
            setting = AutoConfig.from_pretrained(self.model_path)
            language_config = setting.language_config
            language_config._attn_implementation = "eager"

            # Set data type based on device
            cuda_device = self.memory_manager.cuda_device
            dtype = torch.bfloat16 if cuda_device == "cuda" else torch.float32

            # Load the model
            self.vl_gpt = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                language_config=language_config,
                trust_remote_code=True,
                device_map="auto" if cuda_device == "cuda" else None,
                low_cpu_mem_usage=True,
                torch_dtype=dtype,
            )

            # Set to evaluation mode
            self.vl_gpt = self.vl_gpt.eval()

            # Enable CUDNN benchmark for faster inference if using CUDA
            if cuda_device == "cuda":
                torch.backends.cudnn.benchmark = True

            # Load processor and tokenizer
            self.vl_chat_processor = VLChatProcessor.from_pretrained(self.model_path)
            self.tokenizer = self.vl_chat_processor.tokenizer

            self.model_loaded = True
            logger.info("Model loaded successfully")

            # Final memory cleanup
            self.memory_manager.clear_gpu_memory()
            return True

        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            self.memory_manager.clear_gpu_memory()
            self.model_loaded = False
            raise

    def unload_model(self) -> bool:
        """
        Unload the model from memory to free resources.

        Returns:
            bool: Success status
        """
        if not self.model_loaded:
            return True

        logger.info("Unloading model from memory")

        try:
            # Move model to CPU first if using CUDA
            if self.vl_gpt is not None:
                if self.memory_manager.cuda_device == "cuda":
                    self.vl_gpt = self.vl_gpt.cpu()
                del self.vl_gpt
                self.vl_gpt = None

            # Clean up processor
            if self.vl_chat_processor is not None:
                del self.vl_chat_processor
                self.vl_chat_processor = None

            # Clean up tokenizer
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None

            # Clear memory
            self.memory_manager.clear_gpu_memory()

            self.model_loaded = False
            logger.info("Model unloaded successfully")
            return True

        except Exception as e:
            logger.error(f"Error during model unloading: {str(e)}")
            raise

    @contextmanager
    def model_context(self):
        """
        Context manager for automatically loading/unloading the model.

        Example:
            with image_service.model_context():
                # Model is loaded here
                result = image_service.generate(...)
            # Model resources are cleaned up here
        """
        try:
            self.load_model()
            yield
        finally:
            # Don't unload, just clean up memory
            self.memory_manager.clear_gpu_memory()

    def check_model_loaded(self) -> bool:
        """
        Check if the model is loaded and load it if needed.

        Returns:
            bool: Whether the model is loaded
        """
        if not self.model_loaded:
            logger.info("Model not loaded, loading now...")
            self.load_model()
        return self.model_loaded

    def clear_task(self, conversation_id: str) -> bool:
        """
        Remove a task and clean up its resources.

        Args:
            conversation_id: ID of the conversation to clean up

        Returns:
            bool: Success status
        """
        if conversation_id in self.tasks:
            task = self.tasks.pop(conversation_id, None)
            if task and "generate_task" in task and task["generate_task"]:
                task["generate_task"].cancel()
            logger.info(f"Task cleared for conversation_id: {conversation_id}")
            return True
        return False

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        width: int,
        height: int,
        temperature: float = 1,
        parallel_size: int = 3,
        cfg_weight: float = 5,
        image_token_num_per_image: int = 576,
        patch_size: int = 16,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate images based on input tokens.

        Args:
            input_ids: Input token IDs
            width: Image width
            height: Image height
            temperature: Sampling temperature
            parallel_size: Number of parallel images to generate
            cfg_weight: Classifier-free guidance weight
            image_token_num_per_image: Number of image tokens per image
            patch_size: Size of image patches

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Generated tokens and patches
        """
        # Ensure model is loaded
        self.check_model_loaded()

        # Clear GPU memory
        self.memory_manager.clear_gpu_memory()

        try:
            # Prepare input tokens
            tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int)
            if self.memory_manager.cuda_device == "cuda":
                tokens = tokens.to(device=self.memory_manager.cuda_device)

            # Initialize tokens
            for i in range(parallel_size * 2):
                tokens[i, :] = input_ids
                if i % 2 != 0:
                    tokens[i, 1:-1] = self.vl_chat_processor.pad_id

            # Get input embeddings
            inputs_embeds = self.vl_gpt.language_model.get_input_embeddings()(tokens)

            # Initialize generated tokens
            generated_tokens = torch.zeros(
                (parallel_size, image_token_num_per_image), dtype=torch.int
            )
            if self.memory_manager.cuda_device == "cuda":
                generated_tokens = generated_tokens.cuda()

            # Previous key-values for efficient generation
            pkv = None

            # Generate tokens one by one
            for i in range(image_token_num_per_image):
                # Get model outputs
                outputs = self.vl_gpt.language_model.model(
                    inputs_embeds=inputs_embeds, use_cache=True, past_key_values=pkv
                )
                pkv = outputs.past_key_values
                hidden_states = outputs.last_hidden_state

                # Get logits and apply classifier-free guidance
                logits = self.vl_gpt.gen_head(hidden_states[:, -1, :])
                logit_cond = logits[0::2, :]
                logit_uncond = logits[1::2, :]
                logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)

                # Sample next tokens
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated_tokens[:, i] = next_token.squeeze(dim=-1)

                # Prepare for next iteration
                next_token = torch.cat(
                    [next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1
                ).view(-1)
                img_embeds = self.vl_gpt.prepare_gen_img_embeds(next_token)
                inputs_embeds = img_embeds.unsqueeze(dim=1)

                # Update progress callback
                progress = (i + 1) / image_token_num_per_image * 100
                if self.progress_callback:
                    self.progress_callback(progress)

                # Periodically log memory status
                if i % (image_token_num_per_image // 10) == 0:
                    if (
                        self.memory_manager.cuda_device == "cuda"
                        and torch.cuda.is_available()
                    ):
                        current_mem = torch.cuda.memory_allocated() / 1024**2
                        logger.debug(
                            f"Step {i}/{image_token_num_per_image}, Memory: {current_mem:.2f} MB"
                        )

            # Decode generated tokens into image patches
            patches = self.vl_gpt.gen_vision_model.decode_code(
                generated_tokens.to(dtype=torch.int),
                shape=[parallel_size, 8, width // patch_size, height // patch_size],
            )

            # Clean up memory
            self.memory_manager.clear_gpu_memory()

            return generated_tokens.to(dtype=torch.int), patches

        except Exception as e:
            logger.error(f"Error in generate: {str(e)}")
            self.memory_manager.clear_gpu_memory()
            raise

    def unpack(
        self, dec: torch.Tensor, width: int, height: int, parallel_size: int = 3
    ) -> np.ndarray:
        """
        Unpack decoded tensor into image array.

        Args:
            dec: Decoded tensor
            width: Image width
            height: Image height
            parallel_size: Number of parallel images

        Returns:
            np.ndarray: Array of images
        """
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
        return dec

    @torch.inference_mode()
    def generate_image(
        self, prompt: str, seed: Optional[int] = None, guidance: float = 5.0
    ) -> List[Image.Image]:
        """
        Generate images based on a text prompt.

        Args:
            prompt: Text prompt for image generation
            seed: Random seed for reproducibility
            guidance: Guidance scale for image generation

        Returns:
            List[Image.Image]: List of generated images
        """
        try:
            with self.model_context():
                # Set random seed for reproducibility
                seed = seed if seed is not None else 12345
                torch.manual_seed(seed)
                if self.memory_manager.cuda_device == "cuda":
                    torch.cuda.manual_seed(seed)
                np.random.seed(seed)

                # Set image dimensions
                width = 384
                height = 384
                parallel_size = 3

                # Prepare conversation format
                messages = [
                    {
                        "role": "User",
                        "content": prompt,
                    },
                    {"role": "Assistant", "content": ""},
                ]

                # Format the prompt for the model
                text = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                    conversations=messages,
                    sft_format=self.vl_chat_processor.sft_format,
                    system_prompt="",
                )
                text += self.vl_chat_processor.image_start_tag

                # Convert to tensor
                input_ids = torch.LongTensor(self.tokenizer.encode(text))

                # Generate images
                _, patches = self.generate(
                    input_ids,
                    width // 16 * 16,
                    height // 16 * 16,
                    cfg_weight=guidance,
                    parallel_size=parallel_size,
                )

                # Process and convert to PIL images
                images = self.unpack(
                    patches, width // 16 * 16, height // 16 * 16, parallel_size
                )
                image_list = []

                for i in range(parallel_size):
                    img_array = images[i]
                    img = Image.fromarray(img_array)
                    img_resized = img.resize((384, 384), Image.Resampling.LANCZOS)
                    image_list.append(img_resized)

                return image_list

        except Exception as e:
            logger.error(f"Error during image generation: {str(e)}")
            raise
        finally:
            # Ensure memory is cleaned up
            self.memory_manager.clear_gpu_memory()


# Create singleton instance
image_service = ImageService()


# Helper functions for external use
def multimodal_understanding(*args, **kwargs):
    """
    Process image and text input for multimodal understanding.
    This is a wrapper around the image service's multimodal_understanding method.
    """
    return image_service.multimodal_understanding(*args, **kwargs)


def generate_image(*args, **kwargs):
    """
    Generate images based on text prompt.
    This is a wrapper around the image service's generate_image method.
    """
    return image_service.generate_image(*args, **kwargs)

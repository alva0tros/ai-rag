import torch
import numpy as np
import io
import gc
import logging
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM
from src.janus.janus.models import VLChatProcessor
from typing import List, Callable
from contextlib import contextmanager
from app.core.config import settings

logger = logging.getLogger(__name__)


class ImageService:
    _instance = None
    _is_initialized = False

    # 싱글톤 패턴 적용
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ImageService, cls).__new__(cls)
            cls._instance.store = {}  # 채팅 기록을 저장할 딕셔너리
            cls._instance.tasks = {}  # 진행 중인 작업을 저장하는 딕셔너리
            cls._instance.vl_gpt = None
            cls._instance.vl_chat_processor = None
            cls._instance.tokenizer = None
            cls._instance.model_loaded = False
            cls._instance._model_lock = None  # 초기화 시 asyncio.Lock() 설정
        return cls._instance

    def __init__(self, progress_callback: Callable[[float], None] = None):
        if self._is_initialized:
            if progress_callback is not None:
                self.progress_callback = progress_callback
            return

        self.model_path = settings.IMAGE_MODEL_PATH
        self.cuda_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.progress_callback = progress_callback  # 진행률 콜백

        # 메모리 사용량 모니터링 초기화
        self._memory_usage_log = []
        self._max_vram_usage = 0

        logger.info(f"Initializing ImageService with device: {self.cuda_device}")

        # 모델 자동 로드 비활성화 - 필요시 명시적으로 load_model() 호출
        self._is_initialized = True

    def _clear_gpu_memory(self):
        """GPU 메모리를 철저히 정리하는 함수"""
        if self.cuda_device == "cuda":
            # 모든 CUDA 캐시 비우기
            torch.cuda.empty_cache()
            # 가비지 컬렉션 실행
            gc.collect()
            logger.info("GPU memory cleared")

            # 메모리 상태 로깅
            if torch.cuda.is_available():
                current_mem = torch.cuda.memory_allocated() / 1024**2
                max_mem = torch.cuda.max_memory_allocated() / 1024**2
                logger.debug(f"Current GPU memory usage: {current_mem:.2f} MB")
                logger.debug(f"Peak GPU memory usage: {max_mem:.2f} MB")

                # 메모리 사용량 히스토리 저장
                self._memory_usage_log.append(current_mem)
                self._max_vram_usage = max(self._max_vram_usage, max_mem)

    def load_model(self, force_reload=False):
        """모델 로드 함수 - 필요할 때만 호출"""
        if self.model_loaded and not force_reload:
            logger.info("Model already loaded, skipping load")
            return

        logger.info(f"Loading model from {self.model_path}")
        self._clear_gpu_memory()  # 로드 전 메모리 정리

        try:
            # 모델 설정 최적화
            setting = AutoConfig.from_pretrained(self.model_path)
            language_config = setting.language_config
            language_config._attn_implementation = "eager"

            # GPU가 있으면 bfloat16 사용, 없으면 float32로 fallback
            dtype = torch.bfloat16 if self.cuda_device == "cuda" else torch.float32

            # 모델 로드 방식을 GPU 사용 여부에 따라 결정
            if self.cuda_device == "cuda":
                # GPU 사용 시 device_map="auto"만 사용하고 .cuda() 호출 안 함
                self.vl_gpt = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    language_config=language_config,
                    trust_remote_code=True,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    torch_dtype=dtype,
                )
            else:
                # CPU 사용 시 device_map 없이 로드
                self.vl_gpt = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    language_config=language_config,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    torch_dtype=dtype,
                )

            self.vl_gpt = self.vl_gpt.eval()  # 추론 모드 설정

            # 모델 추론 속도 개선
            if self.cuda_device == "cuda":
                torch.backends.cudnn.benchmark = True

            self.vl_chat_processor = VLChatProcessor.from_pretrained(self.model_path)
            self.tokenizer = self.vl_chat_processor.tokenizer

            self.model_loaded = True
            logger.info("Model loaded successfully")

            # 메모리 사용량 확인
            self._clear_gpu_memory()

        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            # 오류 발생 시 메모리 정리
            self._clear_gpu_memory()
            self.model_loaded = False
            raise

    def unload_model(self):
        """모델을 메모리에서 해제하되, 인스턴스는 유지"""
        if not self.model_loaded:
            logger.info("Model not loaded, nothing to unload")
            return

        logger.info("Unloading model from memory")

        try:
            # 모델 객체 참조 해제
            if self.vl_gpt is not None:
                # 메모리에서 완전히 제거하기 위한 단계적 접근
                self.vl_gpt = self.vl_gpt.cpu()  # 먼저 CPU로 이동
                del self.vl_gpt
                self.vl_gpt = None

            # 프로세서 참조 해제
            if self.vl_chat_processor is not None:
                del self.vl_chat_processor
                self.vl_chat_processor = None

            # 토크나이저 참조 해제
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None

            # 철저한 메모리 정리
            self._clear_gpu_memory()

            self.model_loaded = False
            logger.info("Model unloaded successfully")
        except Exception as e:
            logger.error(f"Error during model unloading: {str(e)}")
            raise

    @contextmanager
    def model_context(self):
        """컨텍스트 매니저로 모델 로드/언로드 자동화"""
        try:
            self.load_model()
            yield
        finally:
            # 작업 완료 후 자동으로 언로드하지 않음
            # 대신 메모리 정리 수행
            self._clear_gpu_memory()

    def clear_task(self, conversation_id: str):
        """특정 대화의 태스크 제거"""
        if conversation_id in self.tasks:
            task = self.tasks.pop(conversation_id, None)
            if task and "generate_task" in task and task["generate_task"]:
                task["generate_task"].cancel()
            logger.info(f"Task cleared for conversation_id: {conversation_id}")
            return True
        return False

    def check_model_loaded(self):
        """모델이 로드되었는지 확인하고 필요시 로드"""
        if not self.model_loaded:
            logger.info("Model not loaded, loading now...")
            self.load_model()
        return self.model_loaded

    @torch.inference_mode()
    def multimodal_understanding(self, image_data, question, seed, top_p, temperature):
        # 모델 로드 확인
        self.check_model_loaded()

        # GPU 메모리 정리
        self._clear_gpu_memory()

        torch.manual_seed(seed)
        np.random.seed(seed)
        if self.cuda_device == "cuda":
            torch.cuda.manual_seed(seed)

        # 대화 설정
        conversation = [
            {
                "role": "User",
                "content": f"<image_placeholder>\n{question}",
                "images": [image_data],
            },
            {"role": "Assistant", "content": ""},
        ]

        # 이미지 처리
        pil_images = [Image.open(io.BytesIO(image_data))]
        dtype = torch.bfloat16 if self.cuda_device == "cuda" else torch.float32

        # 입력 준비
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
        )

        # device를 직접 이동하지 않고 to()에 전달
        if self.cuda_device == "cuda":
            prepare_inputs = prepare_inputs.to(device=self.cuda_device, dtype=dtype)
        else:
            prepare_inputs = prepare_inputs.to(dtype=dtype)

        try:
            # 모델 추론
            inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)
            outputs = self.vl_gpt.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False if temperature == 0 else True,
                use_cache=True,
                temperature=temperature,
                top_p=top_p,
            )

            result = self.tokenizer.decode(
                outputs[0].cpu().tolist(), skip_special_tokens=True
            )

            # 메모리 정리
            self._clear_gpu_memory()

            return result
        except Exception as e:
            logger.error(f"Error in multimodal understanding: {str(e)}")
            # 오류 발생 시 메모리 정리
            self._clear_gpu_memory()
            raise

    def generate(
        self,
        input_ids,
        width,
        height,
        temperature=1,
        parallel_size=3,
        cfg_weight=5,
        image_token_num_per_image=576,
        patch_size=16,
    ):

        # 모델 로드 확인
        self.check_model_loaded()

        # GPU 메모리 정리
        self._clear_gpu_memory()

        try:
            # 입력 처리
            tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int)
            if self.cuda_device == "cuda":
                tokens = tokens.to(device=self.cuda_device)
            # tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).to(
            #     self.cuda_device
            # )
            for i in range(parallel_size * 2):
                tokens[i, :] = input_ids
                if i % 2 != 0:
                    tokens[i, 1:-1] = self.vl_chat_processor.pad_id
                    
            inputs_embeds = self.vl_gpt.language_model.get_input_embeddings()(tokens)

            generated_tokens = torch.zeros(
                (parallel_size, image_token_num_per_image), dtype=torch.int
            )
            if self.cuda_device == "cuda":
                generated_tokens = generated_tokens.cuda()

            # generated_tokens = torch.zeros(
            #     (parallel_size, image_token_num_per_image), dtype=torch.int
            # ).to(self.cuda_device)

            pkv = None

            # 토큰 생성 루프
            for i in range(image_token_num_per_image):
                outputs = self.vl_gpt.language_model.model(
                    inputs_embeds=inputs_embeds, use_cache=True, past_key_values=pkv
                )
                pkv = outputs.past_key_values
                hidden_states = outputs.last_hidden_state

                logits = self.vl_gpt.gen_head(hidden_states[:, -1, :])
                logit_cond = logits[0::2, :]
                logit_uncond = logits[1::2, :]
                logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)

                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated_tokens[:, i] = next_token.squeeze(dim=-1)

                next_token = torch.cat(
                    [next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1
                ).view(-1)
                img_embeds = self.vl_gpt.prepare_gen_img_embeds(next_token)
                inputs_embeds = img_embeds.unsqueeze(dim=1)

                # 진행률 계산 및 콜백 호출
                progress = (i + 1) / image_token_num_per_image * 100
                if self.progress_callback:
                    self.progress_callback(progress)

                # 주기적 메모리 상태 확인 (10% 단위)
                if i % (image_token_num_per_image // 10) == 0:
                    if self.cuda_device == "cuda" and torch.cuda.is_available():
                        current_mem = torch.cuda.memory_allocated() / 1024**2
                        logger.debug(
                            f"Step {i}/{image_token_num_per_image}, Memory: {current_mem:.2f} MB"
                        )

            # 디코딩
            patches = self.vl_gpt.gen_vision_model.decode_code(
                generated_tokens.to(dtype=torch.int),
                shape=[parallel_size, 8, width // patch_size, height // patch_size],
            )

            # 메모리 정리
            self._clear_gpu_memory()

            return generated_tokens.to(dtype=torch.int), patches

        except Exception as e:
            logger.error(f"Error in generate: {str(e)}")
            # 오류 발생 시 메모리 정리
            self._clear_gpu_memory()
            raise

    def unpack(self, dec, width, height, parallel_size=5):
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
        # return np.zeros((parallel_size, width, height, 3), dtype=np.uint8)
        return dec

    @torch.inference_mode()
    def generate_image(
        self, prompt: str, seed: int, guidance: float
    ) -> List[Image.Image]:
        try:
            with self.model_context():
                seed = seed if seed is not None else 12345
                torch.manual_seed(seed)
                if self.cuda_device == "cuda":
                    torch.cuda.manual_seed(seed)
                np.random.seed(seed)

                width = 384
                height = 384
                parallel_size = 3

                messages = [
                    # {"role": "User", "content": prompt},
                    {
                        "role": "User",
                        "content": "Draw a playful puppy with big, bright eyes and a wagging tail. The puppy should be sitting on a soft grassy field under a clear blue sky dotted with fluffy white clouds. Its fur is shiny and smooth, and it's wearing a small red bow around its neck. In the background, there are colorful flowers swaying gently in the breeze.",
                    },
                    {"role": "Assistant", "content": ""},
                ]
                text = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                    conversations=messages,
                    sft_format=self.vl_chat_processor.sft_format,
                    system_prompt="",
                )
                text += self.vl_chat_processor.image_start_tag
                input_ids = torch.LongTensor(self.tokenizer.encode(text))

                # 이미지 생성
                _, patches = self.generate(
                    input_ids,
                    width // 16 * 16,
                    height // 16 * 16,
                    cfg_weight=guidance,
                    parallel_size=parallel_size,
                )

                # 이미지 후처리
                images = self.unpack(patches, width // 16 * 16, height // 16 * 16)
                image_list = []
                for i in range(parallel_size):
                    img_array = images[i]  # (height, width, 3) shape이어야 함
                    img = Image.fromarray(img_array)
                    img_resized = img.resize((384, 384), Image.LANCZOS)
                    image_list.append(img_resized)

                return image_list

        except Exception as e:
            print(f"이미지 생성 중 오류 발생: {str(e)}")
            raise
        finally:
            # 메모리 정리
            self._clear_gpu_memory()


# 이미지 서비스 싱글톤 인스턴스 생성
image_service = ImageService()


# Helper functions
def multimodal_understanding(*args, **kwargs):
    return image_service.multimodal_understanding(*args, **kwargs)


def generate_image(*args, **kwargs):
    return image_service.generate_image(*args, **kwargs)

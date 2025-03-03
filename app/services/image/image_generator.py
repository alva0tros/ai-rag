import torch
import numpy as np
import io
import os
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM
from third_party.Janus.janus.models import VLChatProcessor
from typing import List, Callable
from config import IMAGE_MODEL_PATH


class ImageGenerator:
    def __init__(self, progress_callback: Callable[[float], None] = None):
        self.model_path = IMAGE_MODEL_PATH
        self.cuda_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.progress_callback = progress_callback  # 진행률 콜백
        self.vl_gpt = None
        self.vl_chat_processor = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        if self.cuda_device == "cuda":
            torch.cuda.empty_cache()  # 캐시 정리 추가

        setting = AutoConfig.from_pretrained(self.model_path)
        language_config = setting.language_config
        language_config._attn_implementation = "eager"

        # GPU가 있으면 bfloat16 사용, 없으면 float32로 fallback
        dtype = torch.bfloat16 if self.cuda_device == "cuda" else torch.float32

        try:
            self.vl_gpt = (
                AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    language_config=language_config,
                    trust_remote_code=True,
                )
                .to(dtype)
                .to(self.cuda_device)
                .eval()
            )
            print("모델 로드 및 디바이스 이동 완료")
        except Exception as e:
            print(f"모델 로드 실패: {str(e)}")
            raise

        try:
            self.vl_chat_processor = VLChatProcessor.from_pretrained(self.model_path)
            self.tokenizer = self.vl_chat_processor.tokenizer
            print("VLChatProcessor 및 토크나이저 로드 완료")
        except Exception as e:
            print(f"VLChatProcessor 로드 실패: {str(e)}")
            raise

    def unload_model(self):
        """모델을 메모리에서 해제"""
        if self.vl_gpt is not None:
            del self.vl_gpt
            self.vl_gpt = None
        if self.cuda_device == "cuda":
            torch.cuda.empty_cache()  # GPU 메모리 정리
        print("모델 언로드 완료")

    @torch.inference_mode()
    def multimodal_understanding(self, image_data, question, seed, top_p, temperature):
        # GPU 관련 메모리 정리는 GPU일 때만 실행
        if self.cuda_device == "cuda":
            torch.cuda.empty_cache()
        torch.manual_seed(seed)
        np.random.seed(seed)
        if self.cuda_device == "cuda":
            torch.cuda.manual_seed(seed)

        conversation = [
            {
                "role": "User",
                "content": f"<image_placeholder>\n{question}",
                "images": [image_data],
            },
            {"role": "Assistant", "content": ""},
        ]

        pil_images = [Image.open(io.BytesIO(image_data))]

        dtype = torch.bfloat16 if self.cuda_device == "cuda" else torch.float32

        prepare_inputs = self.vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(self.cuda_device, dtype=dtype)

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

        return self.tokenizer.decode(
            outputs[0].cpu().tolist(), skip_special_tokens=True
        )

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

        if self.cuda_device == "cuda":
            torch.cuda.empty_cache()

        tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).to(
            self.cuda_device
        )
        for i in range(parallel_size * 2):
            tokens[i, :] = input_ids
            if i % 2 != 0:
                tokens[i, 1:-1] = self.vl_chat_processor.pad_id
        inputs_embeds = self.vl_gpt.language_model.get_input_embeddings()(tokens)
        generated_tokens = torch.zeros(
            (parallel_size, image_token_num_per_image), dtype=torch.int
        ).to(self.cuda_device)

        pkv = None
        last_progress = 0  # 마지막으로 보고된 진행률 (1% 단위)

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

        patches = self.vl_gpt.gen_vision_model.decode_code(
            generated_tokens.to(dtype=torch.int),
            shape=[parallel_size, 8, width // patch_size, height // patch_size],
        )
        return generated_tokens.to(dtype=torch.int), patches

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
            if self.cuda_device == "cuda":
                torch.cuda.empty_cache()

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

            _, patches = self.generate(
                input_ids,
                width // 16 * 16,
                height // 16 * 16,
                cfg_weight=guidance,
                parallel_size=parallel_size,
            )

            images = self.unpack(patches, width // 16 * 16, height // 16 * 16)
            image_list = []
            for i in range(parallel_size):
                img_array = images[i]  # (height, width, 3) shape이어야 함
                img = Image.fromarray(img_array)
                img_resized = img.resize((384, 384), Image.LANCZOS)
                image_list.append(img_resized)

            return image_list
        finally:
            self.unload_model()


# Helper functions
def multimodal_understanding(generator: ImageGenerator, *args, **kwargs):
    return generator.multimodal_understanding(*args, **kwargs)


def generate_image(generator: ImageGenerator, *args, **kwargs):
    return generator.generate_image(*args, **kwargs)

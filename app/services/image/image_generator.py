import torch
import numpy as np
import io
import os
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM
from third_party.Janus.janus.models import VLChatProcessor
from typing import List
from config import IMAGE_MODEL_PATH, GENERATED_IMAGE_PATH


class ImageGenerator:
    def __init__(self):
        self.model_path = IMAGE_MODEL_PATH
        self.cuda_device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

    def _load_model(self):
        setting = AutoConfig.from_pretrained(self.model_path)
        language_config = setting.language_config
        language_config._attn_implementation = "eager"

        # GPU가 있으면 bfloat16 사용, 없으면 float32로 fallback
        dtype = torch.bfloat16 if self.cuda_device == "cuda" else torch.float32

        self.vl_gpt = (
            AutoModelForCausalLM.from_pretrained(
                self.model_path, language_config=language_config, trust_remote_code=True
            )
            .to(dtype)
            .to(self.cuda_device)
            .eval()
            # .to(torch.bfloat16)
            # .cuda()
        )

        self.vl_chat_processor = VLChatProcessor.from_pretrained(self.model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer

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
        parallel_size=5,
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

        patches = self.vl_gpt.gen_vision_model.decode_code(
            generated_tokens.to(dtype=torch.int),
            shape=[parallel_size, 8, width // patch_size, height // patch_size],
        )
        return generated_tokens.to(dtype=torch.int), patches

    def unpack(self, dec, width, height, parallel_size=5):
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        dec = np.clip((dec + 1) / 2 * 255, 0, 255)
        return np.zeros((parallel_size, width, height, 3), dtype=np.uint8)

    @torch.inference_mode()
    def generate_image(
        self, prompt: str, seed: int, guidance: float
    ) -> List[Image.Image]:

        if self.cuda_device == "cuda":
            torch.cuda.empty_cache()

        seed = seed if seed is not None else 12345
        torch.manual_seed(seed)
        if self.cuda_device == "cuda":
            torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        width = 384
        height = 384
        parallel_size = 5

        messages = [
            {"role": "User", "content": prompt},
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
        image_list = [
            Image.fromarray(images[i]).resize((1024, 1024), Image.LANCZOS)
            for i in range(parallel_size)
        ]

        # 추가: generated_images 폴더에 이미지 저장
        save_dir = GENERATED_IMAGE_PATH
        os.makedirs(save_dir, exist_ok=True)  # 폴더가 없으면 생성
        for i, img in enumerate(image_list):
            file_path = os.path.join(save_dir, f"image_{seed}_{i}.png")
            img.save(file_path)
            print(f"이미지가 저장되었습니다: {file_path}")

        return image_list


# Helper functions
def multimodal_understanding(generator: ImageGenerator, *args, **kwargs):
    return generator.multimodal_understanding(*args, **kwargs)


def generate_image(generator: ImageGenerator, *args, **kwargs):
    return generator.generate_image(*args, **kwargs)

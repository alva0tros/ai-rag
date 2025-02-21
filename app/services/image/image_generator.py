import os
import torch
import numpy as np
import PIL.Image

from transformers import AutoModelForCausalLM
from third_party.Janus.janus.models import MultiModalityCausalLM, VLChatProcessor

# config.py에서 모델 및 생성 이미지 경로 상수 참조
from config import IMAGE_MODEL_PATH, GENERATED_IMAGE_PATH

# 모델 경로는 config의 IMAGE_MODEL_PATH를 사용
model_path = IMAGE_MODEL_PATH
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
# GPU 사용 코드가 있다면 주석 해제하고 사용
# vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
vl_gpt = vl_gpt.to(torch.bfloat16).eval()


@torch.inference_mode()
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    temperature: float = 1,
    parallel_size: int = 1,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
):
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    # 조건부와 비조건부 토큰 처리를 위해 parallel_size * 2 텐서 생성
    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int)
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

    generated_tokens = torch.zeros(
        (parallel_size, image_token_num_per_image), dtype=torch.int
    )

    for i in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=outputs.past_key_values if i != 0 else None,
        )
        hidden_states = outputs.last_hidden_state

        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]

        # Classifier-Free Guidance 적용
        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat(
            [next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1
        ).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    dec = mmgpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size],
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    # GENERATED_IMAGE_PATH에 저장 (config.py에서 설정)
    os.makedirs(GENERATED_IMAGE_PATH, exist_ok=True)
    for i in range(parallel_size):
        save_path = os.path.join(GENERATED_IMAGE_PATH, f"img_{i}.jpg")
        PIL.Image.fromarray(visual_img[i]).save(save_path)


def run_generate():
    conversation = [
        {
            "role": "User",
            "content": (
                "Create a highly detailed portrait of a modern, beautiful Korean woman. "
                "She should have a flawless, porcelain complexion with subtle natural makeup that accentuates her delicate features. "
                "Her eyes are almond-shaped and expressive, reflecting both modern confidence and hints of traditional elegance. "
                "Her sleek, styled hair is cut in a contemporary fashion, and she wears a chic, modern outfit with subtle nods to Korean heritage—"
                "perhaps incorporating elements reminiscent of a modernized hanbok. "
                "The background features a blend of urban sophistication and serene natural elements, "
                "softly lit to enhance the subject's graceful and confident demeanor."
            ),
        },
        {"role": "Assistant", "content": ""},
    ]

    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    prompt = sft_format + vl_chat_processor.image_start_tag

    generate(vl_gpt, vl_chat_processor, prompt)


if __name__ == "__main__":
    run_generate()

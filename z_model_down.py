# pip install huggingface_hub hf_transfer
import os  # Optional for faster downloading
from huggingface_hub import snapshot_download

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# LLM 추론 모델(DeepSeek-R1) 다운
# snapshot_download(
#     repo_id="unsloth/DeepSeek-R1-GGUF",
#     local_dir="./models/text/DeepSeek-R1-GGUF",
#     allow_patterns=["*UD-IQ1_S*"],
# )

# 이미지 생성 모델(janus-pro-7b) 다운
snapshot_download(
    repo_id="deepseek-ai/Janus-Pro-7B",
    # repo_id="deepseek-ai/Janus-Pro-1B",
    local_dir="./models/image/Janus-Pro-7B",
    # local_dir="./models/image/Janus-Pro-1B",
)

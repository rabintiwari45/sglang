"""Load Qwen/Qwen3-30B-A3B with SGLang (expert VM: experts in CPU RAM)."""

import sglang as sgl

MODEL_PATH = "Qwen/Qwen3-30B-A3B"

# KV cache size controls (pick one or combine):
# - mem_fraction_static: fraction of GPU for weights + KV pool (lower = less KV)
# - max_total_tokens: hard cap on KV pool tokens (most direct knob)
MEM_FRACTION_STATIC = 0.35
MAX_TOTAL_TOKENS = 100_000  # ~9 GB KV at bf16; was ~790k (~72 GB) at 0.80

print(f"Loading {MODEL_PATH}...")

llm = sgl.Engine(
    model_path=MODEL_PATH,
    tp_size=1,
    dtype="bfloat16",
    trust_remote_code=True,
    enable_expert_vm=True,
    expert_vm_resident_layers="none",
    disable_cuda_graph=True,
    mem_fraction_static=MEM_FRACTION_STATIC,
    max_total_tokens=MAX_TOTAL_TOKENS,
    log_level="info",
)

print("Model loaded successfully.")

"""Load Qwen/Qwen3-30B-A3B with SGLang (expert VM: experts in CPU RAM)."""

import sglang as sgl

MODEL_PATH = "Qwen/Qwen3-30B-A3B"
MODEL_PATH = "Qwen/Qwen3-30B-A3B-GPTQ-Int4"

# KV cache size controls (pick one or combine):
# - mem_fraction_static: fraction of GPU for weights + KV pool (lower = less KV)
# - max_total_tokens: hard cap on KV pool tokens (most direct knob)
MEM_FRACTION_STATIC = 0.3
MAX_TOTAL_TOKENS = 100  # ~9 GB KV at bf16; was ~790k (~72 GB) at 0.80

def main() -> None:
    print(f"Loading {MODEL_PATH}...")

    llm = sgl.Engine(
        model_path=MODEL_PATH,
        tp_size=1,
        dtype="bfloat16",
        trust_remote_code=True,
        enable_expert_vm=True,
        expert_vm_resident_layers="0",
        disable_cuda_graph=True,
        mem_fraction_static=MEM_FRACTION_STATIC,
        max_total_tokens=MAX_TOTAL_TOKENS,
        log_level="info",
        # JIT kernel compilation (moe_wna16_marlin bf16) happens on first inference
        # and can take ~10 min. Raise watchdog well above that; cached after first run.
        watchdog_timeout=1800,
    )

    print("Model loaded successfully.")

    outputs = llm.generate("Hello, how are you?", sampling_params={"max_new_tokens": 20})
    print("--------------------------------")
    print("Outputs:")
    print(outputs)

    llm.shutdown()


if __name__ == "__main__":
    main()

"""Qwen3 MoE GPTQ on SGLang — full GPU baseline (all experts resident on GPU).

No expert prefetch / CPU offload. Enable ``SGLANG_LOG_LAYER_TIMING=1`` for
per-token decode lines like::

  [layer_timing] decode ntok=1 bs=1 attn=... router=... [gate=..., topk=...]
  moe=... moe_compute=... other=... total=... ms
"""

import os

os.environ.setdefault("SGLANG_LOG_LAYER_TIMING", "1")

import sglang as sgl

MODEL_PATH = "Qwen/Qwen3-30B-A3B-GPTQ-Int4"

MEM_FRACTION_STATIC = 0.8
MAX_TOTAL_TOKENS = 1000


def main() -> None:
    print(f"Loading {MODEL_PATH} (full GPU, no expert prefetch)...")

    llm = sgl.Engine(
        model_path=MODEL_PATH,
        tp_size=1,
        dtype="float16",
        trust_remote_code=True,
        # Match prefetch experiments: eager forward so layer timing hooks run
        # every decode step (CUDA graphs skip Python per layer).
        disable_cuda_graph=True,
        disable_piecewise_cuda_graph=True,
        mem_fraction_static=MEM_FRACTION_STATIC,
        max_total_tokens=MAX_TOTAL_TOKENS,
        log_level="info",
        watchdog_timeout=1800,
    )

    print("Model loaded successfully.")

    outputs = llm.generate(
        "Hello, how are you?",
        sampling_params={"max_new_tokens": 100},
    )
    print("--------------------------------")
    print("Outputs:")
    print(outputs)

    llm.shutdown()


if __name__ == "__main__":
    main()

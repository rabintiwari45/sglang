"""Run one prefetch-decode experiment. Usage: python bench_prefetch.py [cache_slots] [max_new_tokens]"""

import os
import sys

os.environ.setdefault("SGLANG_LOG_LAYER_TIMING", "1")

import sglang as sgl


def main() -> None:
    cache_slots = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    max_new_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    llm = sgl.Engine(
        model_path="Qwen/Qwen3-30B-A3B-GPTQ-Int4",
        tp_size=1,
        dtype="bfloat16",
        trust_remote_code=True,
        enable_expert_prefetch=True,
        expert_prefetch_resident_layers="0",
        expert_prefetch_cache_slots=cache_slots,
        disable_cuda_graph=True,
        disable_piecewise_cuda_graph=True,
        mem_fraction_static=0.8,
        max_total_tokens=1000,
        log_level="info",
        watchdog_timeout=1800,
    )
    out = llm.generate(
        "Hello, how are you?", sampling_params={"max_new_tokens": max_new_tokens}
    )
    print(out)
    llm.shutdown()


if __name__ == "__main__":
    main()

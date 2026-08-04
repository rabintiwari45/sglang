"""Compare decode with vs without expert prefetch."""

import os
import statistics
import sys
from pathlib import Path

os.environ.setdefault("SGLANG_LOG_LAYER_TIMING", "1")
TIMING_FILE = "/tmp/sglang_decode_times.txt"
MODEL_PATH = "Qwen/Qwen3-30B-A3B-GPTQ-Int4"


def run_bench(enable_prefetch: bool) -> float:
    os.environ["SGLANG_LAYER_TIMING_FILE"] = TIMING_FILE
    Path(TIMING_FILE).unlink(missing_ok=True)

    import sglang as sgl

    llm = sgl.Engine(
        model_path=MODEL_PATH,
        tp_size=1,
        dtype="bfloat16",
        trust_remote_code=True,
        enable_expert_prefetch=enable_prefetch,
        expert_prefetch_resident_layers="0",
        disable_cuda_graph=True,
        disable_piecewise_cuda_graph=True,
        mem_fraction_static=0.8,
        max_total_tokens=1000,
        log_level="warning",
        watchdog_timeout=1800,
    )
    prompt = "Hello"
    llm.generate(prompt, sampling_params={"max_new_tokens": 2})
    Path(TIMING_FILE).write_text("", encoding="utf-8")
    llm.generate(prompt, sampling_params={"max_new_tokens": 6})
    lines = [
        float(x)
        for x in Path(TIMING_FILE).read_text().strip().splitlines()
        if x.strip()
    ]
    llm.shutdown()
    if not lines:
        return float("nan")
    return statistics.median(lines[-4:])


def main() -> None:
    print("Benchmark no prefetch (all experts on GPU)...", flush=True)
    no_pf = run_bench(False)
    print(f"  median decode (last 4): {no_pf:.2f} ms", flush=True)
    print("Benchmark with expert prefetch...", flush=True)
    with_pf = run_bench(True)
    print(f"  median decode (last 4): {with_pf:.2f} ms", flush=True)
    print(f"  overhead: {with_pf - no_pf:.2f} ms")


if __name__ == "__main__":
    main()

"""Benchmark single-token decode latency with expert prefetch."""

import os
import statistics
import sys
from pathlib import Path

os.environ.setdefault("SGLANG_LOG_LAYER_TIMING", "1")

TIMING_FILE = "/tmp/sglang_decode_times.txt"

import sglang as sgl

MODEL_PATH = "Qwen/Qwen3-30B-A3B-GPTQ-Int4"
WARMUP_TOKENS = 3
BENCH_TOKENS = 8
MEM_FRACTION_STATIC = 0.8
MAX_TOTAL_TOKENS = 1000


def _read_decode_samples() -> list[float]:
    path = Path(TIMING_FILE)
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return [float(x) for x in lines if x.strip()]


def main() -> None:
    os.environ["SGLANG_LAYER_TIMING_FILE"] = TIMING_FILE
    Path(TIMING_FILE).unlink(missing_ok=True)

    print(f"Loading {MODEL_PATH}...", flush=True)
    llm = sgl.Engine(
        model_path=MODEL_PATH,
        tp_size=1,
        dtype="bfloat16",
        trust_remote_code=True,
        enable_expert_prefetch=True,
        expert_prefetch_resident_layers="0",
        disable_cuda_graph=True,
        disable_piecewise_cuda_graph=True,
        mem_fraction_static=MEM_FRACTION_STATIC,
        max_total_tokens=MAX_TOTAL_TOKENS,
        log_level="info",
        watchdog_timeout=1800,
    )
    print("Model loaded.", flush=True)

    prompt = "Hello"
    llm.generate(prompt, sampling_params={"max_new_tokens": WARMUP_TOKENS})

    Path(TIMING_FILE).write_text("", encoding="utf-8")
    llm.generate(prompt, sampling_params={"max_new_tokens": BENCH_TOKENS})

    samples = _read_decode_samples()
    if not samples:
        print(
            f"No decode timings in {TIMING_FILE}. "
            "Scheduler may not have written layer timing.",
            file=sys.stderr,
        )
        llm.shutdown()
        sys.exit(1)

    mean_ms = statistics.mean(samples)
    p50 = statistics.median(samples)
    print("--------------------------------")
    print(f"Decode samples (ms): {[round(x, 2) for x in samples]}")
    print(f"mean={mean_ms:.2f} ms  median={p50:.2f} ms  n={len(samples)}")
    print("target: < 90 ms per decode step")
    llm.shutdown()


if __name__ == "__main__":
    main()

# Expert VM MoE Layer Latency — Debug Summary

## Goal

Run **Qwen3 MoE** in SGLang on an L4 GPU (23 GB VRAM) by keeping non-expert weights on GPU and offloading MoE expert weights to CPU RAM, staging them asynchronously to a compact GPU buffer per layer with lookahead prefetching.

**Baseline (full-GPU, no offload):**
| Step | Time |
|---|---|
| Attention | ~1.3 ms |
| Router | ~0.35 ms |
| MoE compute | ~0.85 ms |
| **Total per layer** | **~2.7 ms** |

---

## Issues Found and Fixed

### Issue 1 — Global `torch.cuda.synchronize()` inflated block timing

**Symptom:** `Transformer block time ≈ 12 ms`, but attention + MoE compute summed to only ~1.3 ms.

**Root cause:** The original timing code called `torch.cuda.synchronize()` which waits for **all** CUDA streams, including the async H2D copy stream used by the prefetcher. This made the "block time" absorb the full copy duration (~8 ms) even though it ran in the background.

**Fix:** Replaced `torch.cuda.synchronize()` with `torch.cuda.current_stream().synchronize()` (`sync_compute_stream()`), which only drains the compute stream. This also fixed a real performance bug where the global sync was forcing the copy to finish early, serializing what should have been concurrent work.

---

### Issue 2 — GPTQ Marlin dtype mismatch (`bfloat16` vs `float16`)

**Symptom:** `AssertionError: moe_wna16_marlin_gemm assumes hidden_states.dtype (torch.bfloat16) == w1_scale.dtype (torch.float16)` during warmup.

**Root cause:** `GPTQMarlinMoEMethod.create_weights` hardcoded `w13_scales` and `w2_scales` to `dtype=torch.half` regardless of the model's `params_dtype`.

**Fix:** Changed allocation to use `params_dtype` so scale tensors match the activation dtype.

**File:** `sglang/python/sglang/srt/layers/quantization/gptq.py`

---

### Issue 3 — `time.perf_counter()` incompatible with `torch.compile`

**Symptom:** `torch._dynamo.exc.Unsupported: Attempted to call function marked as skipped` during piecewise CUDA graph warmup.

**Root cause:** `time.perf_counter()` is a Python built-in that cannot be traced by `torch.compile`. The layer-timing utility called it unconditionally.

**Fix:**
- Added `should_record_layer_timing()` guard that checks `torch.compiler.is_compiling()` and `is_in_pcg_torch_compile()`, making timing context managers no-ops during compilation.
- Added `disable_piecewise_cuda_graph=True` to the engine config for the debugging session.

**File:** `sglang/python/sglang/srt/utils/layer_timing.py`

---

### Issue 4 — `index_select()` on CPU buffer produced non-pinned tensor → 8 ms blocking copy

**Symptom:** `Expert fetch: copy=8 ms, wait=0.01 ms`. The `moe` step in layer timing was 10–14 ms while the core MoE compute was only ~0.5 ms.

**Root cause:** `gather_expert_rows_async` called `cpu_buffer.index_select(0, ids)` which created a **new, non-pinned** output tensor. `tensor.copy_(..., non_blocking=True)` silently falls back to synchronous when the source is not pinned, blocking the CPU for the full 8 ms copy duration inside the lookahead phase.

**Fix:** Replaced `index_select` with a row-by-row loop using `cpu_buffer[eid]` (a view into the already-pinned buffer). Each `dst[i].copy_(cpu_buffer[eid], non_blocking=True)` is then a true async DMA. Copy time dropped from ~8 ms to ~1.7 ms.

**File:** `sglang/python/sglang/srt/layers/moe/expert_vm/gather.py`

---

### Issue 5 — `expert_sets_match` caused GPU sync in the router step

**Symptom:** `router` step in layer timing was ~0.7 ms instead of the expected ~0.35 ms.

**Root cause:** `expert_sets_match` called `torch.equal(active_a, active_b)` on GPU tensors, which forced a device–host sync to compare the result.

**Fix:** Converted both tensors to CPU Python sets (`tensor.to("cpu").tolist()`) before comparing. Both tensors are tiny (≤ 8 int64 values), so the D2H transfer is negligible.

**File:** `sglang/python/sglang/srt/layers/moe/expert_vm/gather.py`

---

### Issue 6 — Lookahead 2.15 ms: 80 CUDA DMA calls with high Python dispatch overhead

**Symptom:** After fixes 4 & 5, `moe_breakdown` showed:
```
wait_bind=0.66 lookahead=2.15 dispatch=0.03 compute=0.58 total=3.42 ms
```
Copy time was only 1.7 ms but lookahead itself took 2.15 ms.

**Root cause:** `gather_expert_rows_async` was called once per weight name (10 weights), and each call looped over 8 expert IDs — resulting in **80 separate `cudaMemcpyAsync` calls**. Each call has ~15 µs of CUDA driver overhead:
- 80 × 15 µs = **1.2 ms** wasted on CUDA driver round-trips
- 80 × 5 µs Python loop overhead = **0.4 ms**

**Fix (two-part):**

**Part A — Pre-allocate pinned staging buffers** (`prefetch.py`):  
In `register_sparse_block`, allocate one contiguous **pinned** CPU staging buffer per weight per layer (sized for `MAX_STAGING_K = 16` experts, ~109 MiB total for 47 layers). These buffers exist for the lifetime of the model and are reused every forward pass.

**Part B — Change `cpu_buf` to non-pinned** (`manager.py`):  
Previously `cpu_buf` was pinned so each row could be DMA'd directly. With the staging approach, the CPU now **reads** from `cpu_buf` (via `torch.index_select`) to gather selected expert rows into the pinned staging buffer. Pinned memory has **poor CPU read performance** (~4 GB/s) because it is not cache-resident for CPU code — the cache optimization for pinned memory benefits only the GPU DMA engine. Changing `cpu_buf` to regular (non-pinned, CPU-cacheable) memory restores ~40 GB/s read bandwidth for the gather step.

**Data flow after fix:**
```
cpu_buf (non-pinned, ~30 MiB/weight, CPU-cached)
    │
    │  torch.index_select(cpu_buf, 0, ids_tensor, out=staging[:k])
    │  → single C++ call, reads at ~40 GB/s ≈ 0.046 ms/weight × 10 = 0.46 ms total
    ▼
staging[:k] (PINNED, 1.86 MiB/weight × 10 = 18.63 MiB total)
    │
    │  gpu_t.copy_(staging[:k], non_blocking=True)   [×10, on copy_stream]
    │  → 10 async DMA submissions instead of 80
    ▼
compact GPU buffer  (18.63 MiB, used by MoE kernel)
```

**Expected lookahead after fix:**
| Phase | Before | After |
|---|---|---|
| Gate + topk D2H sync | ~0.35 ms | ~0.35 ms |
| CPU gather | 0 ms (not done by CPU) | **~0.46 ms** (index_select, fast) |
| CUDA DMA submissions | ~1.2 ms (80 calls × 15 µs) | **~0.10 ms** (10 calls × 10 µs) |
| **Total lookahead** | **~2.15 ms** | **~0.91 ms** |

**Files changed:**
- `sglang/python/sglang/srt/layers/moe/expert_vm/manager.py` — `cpu_buf` now non-pinned
- `sglang/python/sglang/srt/layers/moe/expert_vm/prefetch.py` — pinned staging buffers + single DMA per weight

---

## Regression Introduced and Corrected

After implementing the staging approach but **before** making `cpu_buf` non-pinned, the timing got worse:

```
lookahead=5.72 ms   (was 2.15 ms)
wait_bind=0.85 ms   (was 0.66 ms)
```

**Why `wait_bind` also increased:** Because the staging gather (`torch.index_select` on pinned `cpu_buf`) took ~4.6 ms, the DMA was launched 4.6 ms late. By the time `wait_and_bind` fired 2.1 ms later, the DMA had only been running for ~1.5 ms of its 1.4 ms required duration — causing additional wait.

**Fix:** Making `cpu_buf` non-pinned (Issue 6 Part B) restores the correct data flow.

---

## Summary of All File Changes

| File | Change |
|---|---|
| `layers/quantization/gptq.py` | Fix scale dtype to match `params_dtype` |
| `layers/moe/expert_vm/config.py` | Add `sync_compute_stream()` utility |
| `layers/moe/expert_vm/gather.py` | Row-by-row pinned DMA; CPU-set comparison in `expert_sets_match` |
| `layers/moe/expert_vm/prefetch.py` | Pre-allocated pinned staging; `index_select` gather + 10 DMA calls |
| `layers/moe/expert_vm/manager.py` | `cpu_buf` stored as non-pinned CPU RAM |
| `layers/moe/fused_moe_triton/layer.py` | `moe_breakdown` timing; use `sync_compute_stream` |
| `models/qwen3_moe.py` | Per-step layer timing integration |
| `utils/layer_timing.py` | Compile-safe layer timing context managers |
| `main.py` | Enable layer timing; disable piecewise CUDA graphs for profiling |

---

## Remaining Gap

After all fixes, expected per-layer timing:

| Step | Full-GPU baseline | Expert VM (after fixes) |
|---|---|---|
| Attention | 1.3 ms | 1.3 ms |
| Router | 0.35 ms | ~0.7 ms (+0.35 ms lookahead gate/topk) |
| MoE (wait + lookahead + compute) | 0.85 ms | ~2.4 ms |
| **Total** | **2.7 ms** | **~4.5 ms** |

The remaining ~1.8 ms overhead is the **irreducible cost** of the CPU→GPU expert weight copy (18.63 MiB at ~12 GB/s PCIe = 1.55 ms DMA) not being fully hidden by the inter-layer compute window. Reducing this further would require:
- Tighter quantization (fewer bytes per expert)
- Longer overlap window (more compute between layers)
- Hardware with higher PCIe bandwidth

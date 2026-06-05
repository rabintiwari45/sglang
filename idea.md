# Sparse MoE Expert Virtualization Research Notes

# Research Goal

I want to explore efficient inference of very large Mixture-of-Experts (MoE) models such as DeepSeek DeepSeek-V3/V4-class architectures on consumer GPUs with limited VRAM.

Example target:

* Total parameters: ~248B
* Active parameters per token: ~13B
* Quantization: INT8 / INT4
* Target hardware: single 16GB–48GB GPU

The core hypothesis is:

> Even though the total model is extremely large, only a sparse subset of experts is active per token. Therefore, if active experts can be streamed/prefetched efficiently, frontier-scale MoE inference may become feasible on consumer GPUs without requiring the full model to reside in VRAM.

The primary challenge is not model capacity itself, but:

* expert movement
* memory bandwidth
* transfer latency
* runtime scheduling
* compute/transfer overlap

The long-term vision is something conceptually similar to:

```text
Virtual Memory for MoE Experts
```

Where:

* GPU VRAM acts like a fast cache
* CPU RAM/NVMe acts like backing storage
* experts are paged dynamically
* future experts are predicted and prefetched
* tensor cores never stall waiting for transfers

---

# Key Insight

Sparse MoE reduces FLOPs significantly, but does NOT automatically solve:

* bandwidth requirements
* memory movement
* expert transfer latency

The central research question is:

> Can expert movement be hidden so effectively that inference throughput approaches full-GPU-resident execution?

---

# Major Bottlenecks Identified

## 1. PCIe Bandwidth Bottleneck

Modern GPUs assume weights are already in local VRAM/HBM.

Approximate bandwidth comparison:

| Path          | Bandwidth |
| ------------- | --------- |
| H100 HBM      | ~3 TB/s   |
| RTX 4090 VRAM | ~1 TB/s   |
| PCIe Gen4 x16 | ~32 GB/s  |
| PCIe Gen5 x16 | ~64 GB/s  |

This creates a massive mismatch between:

* compute throughput
* expert delivery throughput

Naive expert swapping causes tensor-core starvation.

---

## 2. Tensor Core Underutilization

Tensor cores consume data extremely quickly.

If experts are not prefetched early enough:

```text
compute finishes
→ GPU waits for expert transfer
→ utilization collapses
```

Even small stalls severely reduce throughput.

---

## 3. Dynamic Expert Routing

Expert selection changes:

* per token
* per layer
* per sequence

Example:

```text
Layer 1 → Expert A
Layer 2 → Expert T
Layer 3 → Expert C
```

This destroys:

* locality
* cache predictability
* simple streaming assumptions

---

## 4. GPU SRAM Limitations

Weights ultimately execute from:

* registers
* shared memory
* SRAM/cache

However:

* SRAM is tiny
* active experts are GB-scale

So weights still must reside primarily in VRAM.

GPU SRAM functions only as:

* compute staging buffer
* temporary tile storage

not permanent expert storage.

---

## 5. CPU RAM Latency

CPU RAM over PCIe introduces:

* higher latency
* synchronization overhead
* transfer stalls

MoE inference becomes highly latency-sensitive.

---

## 6. MoE Is Often Memory-Bound

MoE sparsity reduces FLOPs more effectively than it reduces memory traffic.

The real bottleneck shifts from:

```text
compute-bound
```

to:

```text
memory-bandwidth-bound
```

---

## 7. Routing Prediction Problem

Ideal system:

```text
predict future experts
→ prefetch asynchronously
→ overlap transfer with compute
```

Challenge:
future routing depends on:

* hidden states
* previous layers
* token evolution

Prediction quality becomes central.

---

## 8. Batch Diversity

Different requests activate different experts:

```text
Token A → Expert 3
Token B → Expert 18
Token C → Expert 42
```

This increases:

* cache pressure
* transfer complexity
* scheduling difficulty

---

# Research Direction

The research direction is primarily:

```text
runtime systems + memory hierarchy optimization
```

NOT primarily:

```text
kernel optimization
```

The focus areas are:

* expert prediction
* expert prefetching
* VRAM residency management
* cache eviction policies
* transfer scheduling
* overlap of compute and transfer

---

# Conceptual Analogy

This problem resembles operating system virtual memory systems.

| OS Concept  | MoE Equivalent         |
| ----------- | ---------------------- |
| RAM         | GPU VRAM               |
| Disk        | CPU RAM / NVMe         |
| Page cache  | Expert cache           |
| Prefetcher  | Expert predictor       |
| Page faults | Expert transfer stalls |

The broader idea:

```text
Virtualized Expert Memory System
```

---

# Framework Choice

Primary framework chosen:

* SGLang

Reasoning:

* flexible runtime
* hackable scheduler
* easier experimentation
* MoE support
* dynamic execution model
* suitable for runtime systems research

Secondary reference framework:

* vLLM

Used for studying:

* paged attention
* memory virtualization concepts
* scheduling architecture

Additional useful tools:

* Triton
* CUTLASS
* llama.cpp
* Nsight Systems
* Nsight Compute

---

# Important Observation

Current frameworks mostly assume:

```text
weights fit in GPU memory
```

The goal here is to break that assumption and build:

```text
hierarchical expert memory management
```

---

# Current Understanding of SGLang

SGLang already contains:

* advanced scheduling
* continuous batching
* MoE execution support
* async runtime infrastructure
* overlap-oriented execution

But likely does NOT yet implement:

* predictive expert prefetching
* learned expert routing prediction
* sophisticated expert cache hierarchy
* explicit VRAM expert residency management
* expert-level virtual memory system

This makes it a good research platform.

---

# Initial Experimental Roadmap

## Phase 1 — Understand SGLang Internals

Explore:

* request lifecycle
* scheduler
* model runner
* MoE forward path
* CUDA stream usage
* memory allocation lifecycle

Goal:
understand where expert routing and execution occur.

---

## Phase 2 — Expert Routing Instrumentation

Log for every token/layer:

```python
{
  token_id,
  layer_id,
  selected_experts,
  routing_scores
}
```

Study:

* expert reuse
* temporal locality
* routing entropy
* transition patterns
* layer-wise cacheability

Core question:

> How predictable are expert trajectories?

---

## Phase 3 — Expert Locality Analysis

Key analyses:

* expert reuse rate
* hot/cold expert distribution
* expert transition graphs
* temporal locality horizon
* cache hit potential

Hypothesis:
expert routing is more structured and predictable than previously assumed.

---

## Phase 4 — Simulated Expert Cache

Build simulator:

```text
CPU RAM → full expert storage
GPU VRAM → limited expert cache
```

Test:

* LRU
* LFU
* predictive prefetch
* speculative loading
* probabilistic eviction

Metrics:

* cache hit rate
* transfer volume
* predicted stalls
* residency duration

---

## Phase 5 — Async CPU↔GPU Offloading

Implement:

* pinned memory
* CUDA streams
* async memcpy
* overlap transfer + compute

Key APIs:

* torch.cuda.Stream
* CUDA events
* non_blocking transfers

Goal:
hide transfer latency behind compute.

---

## Phase 6 — Speculative Expert Prefetching

Core idea:

```text
predict future experts
→ prefetch early
→ reduce transfer stalls
```

Initial approaches:

* transition tables
* Markov models
* frequency heuristics

Later possibilities:

* learned predictors
* hidden-state predictors
* routing sequence models

---

# Relevant Research Papers

Most relevant papers identified:

## 1. SpecMD

Key insight:
MoE expert access patterns do NOT follow standard temporal locality assumptions.

Contributions:

* Least-Stale eviction policy
* routing-aware caching

Important idea:
traditional LRU assumptions may fail for MoE.

---

## 2. MoE-Infinity

Key idea:
activation-aware expert offloading and prefetching.

Validated:

* expert offloading can work
* routing locality exists

---

## 3. Cross-Layer Gate Prediction

Key idea:
predict future experts using earlier-layer routing signals.

Very relevant to:

```text
expert prediction → speculative prefetch
```

---

## 4. SP-MoE

Key idea:
combine speculative decoding with expert prefetching.

Concept:

```text
future token prediction
→ future expert prediction
```

Potentially extremely promising.

---

# Long-Term Research Direction

Potential future concepts:

* learned expert predictors
* expert trajectory forecasting
* hierarchical expert memory
* speculative expert residency
* routing entropy modeling
* expert graph analysis
* MoE virtual memory systems

Possible future system names/concepts:

```text
PagedExperts
VirtualExpertMemory
SpeculativeExpertResidency
```

---

# Important Engineering Priorities

Initially prioritize:

1. routing instrumentation
2. locality analysis
3. cache simulation
4. scheduling understanding

Do NOT initially prioritize:

* CUDA kernel optimization
* GEMM optimization
* tensor-core micro-optimization

The dominant bottleneck is likely:

```text
memory movement + scheduling
```

not raw compute.

---

# Initial Hardware

Current experimental hardware may include:

* NVIDIA T4
* consumer GPUs (16GB–48GB VRAM)

Initial experiments do NOT require:

* full 248B deployment
* H100 clusters

Early work focuses on:

* routing behavior
* cacheability
* scheduling feasibility
* transfer overlap analysis

---

# Overall Objective

Ultimate objective:

```text
Enable frontier-scale sparse MoE inference on consumer GPUs
through predictive expert memory virtualization and transfer hiding.
```

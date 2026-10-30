# MMA Benchmark on T4 GPU

A step-by-step implementation of CUDA matrix multiply-accumulate (MMA) kernels using **CuTe**, progressively optimized toward cuBLAS performance.

worklog: [here](https://chamgeon.github.io/2026/07/19/CuTeMMA.html)

All benchmarks run on an **NVIDIA T4 GPU** with matrix size **8192×8192×8192**.

---

## Results

| Kernel | Optimization | GFLOPS/sec | vs Previous | vs cuBLAS |
|--------|-------------|------------|-------------|-----------|
| 1 | Naive | 3,000.78 | — | 6.3% |
| 2 | Tiled copy + tiled MMA (vectorized) | 12,423.8 | +314% | 26.0% |
| 3 | Swizzle | 20,233.8 | +63% | 42.5% |
| 4 | Explicit register buffer for copy | 21,553.1 | +6.5% | 45.2% |
| 5 | Arithmetic intensity improvement | 37,117.8 | +72% | 77.9% |
| 6 | LDSM pipelining + MMA tile size fix | 40,079.6 | +8.0% | 84.1% |
| 7 | 2-stage pipelining | 42,957.1 | +7.2% | **90.1%** |
| cuBLAS | Reference | 47,655.2 | — | 100% |

Kernel 7 achieves **90.1% of cuBLAS performance**.

---

## Kernel Descriptions

**Kernel 1 — Naive**
Baseline implementation with no optimization. Each thread computes a single output element with no memory reuse.

**Kernel 2 — Tiled Copy + Tiled MMA (Vectorized)**
Applies CuTe tiled copy and tiled MMA abstractions with vectorized memory access, yielding a 4× improvement over the naive baseline.

**Kernel 3 — Swizzle**
Applies swizzled shared memory layout to eliminate bank conflicts, significantly improving memory throughput.

**Kernel 4 — Explicit Register Buffer for Copy**
Introduces explicit register-level buffering during the copy phase to reduce redundant memory traffic.

**Kernel 5 — Arithmetic Intensity Improvement**
Analyzes and improves arithmetic intensity by restructuring computation to maximize FLOP-per-byte ratio, resulting in a 72% jump in performance.

**Kernel 6 — LDSM Pipelining + MMA Tile Size Fix**
Applies `ldsm` instruction-level pipelining and fixes the MMA tile size to reduce the number of `ldsm` calls, improving instruction efficiency.

**Kernel 7 — 2-Stage Pipelining**
Introduces double-buffering (2-stage pipeline) to overlap data loading and computation, hiding memory latency and reaching 90% of cuBLAS throughput.

**Kernel 8 — CTA Swizzle**
Same as kernel 7, but remaps `blockIdx` so that consecutively launched CTAs cover a `GROUP_M`×N patch of output tiles (`GROUP_M = 8`) instead of a single tile column. Concurrently running CTAs then share A and B tiles through L2, reducing DRAM traffic at large sizes. CUDA does not guarantee CTA scheduling order, so this only affects performance, never correctness.

---

## Usage

### Build

```bash
make -j4    # parallel build; only changed files recompile
```

### Run

```bash
./runner <kernel_number>

# Examples
./runner 1    # Naive kernel
./runner 7    # 2-stage pipelining kernel
./runner 8    # CTA swizzle kernel
./runner 99   # cuBLAS reference

# Optional problem size (M=N=K): 1024, 2048, 4096, 8192 (default 8192)
./runner 7 2048
```

### Size sweep

```bash
./sweep.sh > sweep.csv                       # all kernels x all sizes, CSV output
KERNELS="7 99" SIZES="2048 8192" ./sweep.sh  # subset
```

Sizes are compile-time template parameters, so each size gets the same fully static layouts as the original 8192 build.

### Dependencies

- CUDA Toolkit
- [CUTLASS](https://github.com/NVIDIA/cutlass)
- cuBLAS
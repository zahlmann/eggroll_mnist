## EGGROLL MNIST

Evolution Strategies training of a 784→128→128→10 MLP on MNIST, using a fused Triton kernel to close the gap to backprop.

### Results

Same architecture, same GPU (RTX 4080 SUPER), fp32 data, 10 epochs:

| | Backprop (optimized) | Backprop (naive) | EGGROLL |
|---|---|---|---|
| **Training time** | **1.5s** | 4.5s | **2.16s** |
| **Test accuracy** | 97.5% | 97.3% | 97.3% |
| **Peak memory** | 389 MB | 391 MB | 389 MB |

EGGROLL does **3,360x more FLOPs** per batch (1,680 antithetic perturbation pairs × 2 forward passes each, vs 1 forward + 1 backward). The fused Triton kernel compresses this gap to **1.44x wall-clock time**.

### How to run

```bash
uv run benchmark.py      # single seed, quick iteration
uv run validate.py       # 3-seed validation (seeds 11, 42, 7)
```

### What made it fast

27s → 2.16s (12.5x) over 8 sessions of agent-driven optimization:

**Kernel + compilation (27s → 5.2s).** Fused 3-layer Triton kernel keeps all intermediates in registers (95% less HBM traffic). K-tiled L1→L2 matmul with FP8 E4M3 tensor cores. All-in-one JIT via nested `lax.scan`. Merged pos/neg kernel (3D grid).

**Population reduction (5.2s → 2.9s).** Label smoothing in CE fitness gives the ES gradient information about all 10 classes, not just the correct one — enables halving population from 5000 to 1800. Per-subgroup Winsorized z-score (K=8, clip ±2.0) stabilizes gradient at low pop. Adaptive smoothing schedule (α=0.30 × 0.50^epoch) broke through the pop2500 barrier.

**JIT reduction + pipeline (2.9s → 2.16s).** `--xla_gpu_enable_triton_gemm=false` saves 0.6s by skipping XLA's internal GEMM autotuner. Pop1680 for perfect CUDA wave alignment (21.0 waves, zero tail). Shuffle + GPU transfer overlapped with JIT compilation in background thread.

### What didn't work

**Kernel tuning:** BLOCK_B=16/32/128, BLOCK_K=64, num_stages>1, num_warps=2/8, persistent kernel, Pallas (needs sm_90+).

**Algorithmic:** rank-based fitness shaping (95.6%), Boltzmann weighting (86.6%), one-sided ES (2x pop needed), per-layer sigma scaling (catastrophic), rank-2 perturbations (96.7%), float16 vectors (narrower dynamic range hurts).

**Population below 1680:** pop1600 barely passes 3-seed (1 of 144 configs). Pop1520/1440/1360 all fail. Z-score tuning, sigma/LR decay sweeps — current config is already optimal.

**Compilation:** pure JAX (4.6x slower), Python loops (dispatch overhead), compilation caching (unfair), various XLA flags, JAX FFI (needs C++ wrapper).

### Remaining path to 2.0s

The 0.16s gap is entirely JIT overhead (0.71s lowering + compilation). JAX FFI with a pre-compiled Triton kernel would save ~0.26s by bypassing jax-triton lowering, but requires a C++ shared library.

### Background

Implements [EGGROLL](https://arxiv.org/pdf/2511.16652) (Sarkar et al., 2025). Low-rank perturbations: `x @ (W + σ·outer(B, A)) = x@W + σ·(x@B)·A`. Temperature-scaled CE (T=2.0) with adaptive label smoothing and antithetic sampling. See `program.md` for the full optimization log.

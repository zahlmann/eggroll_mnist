## EGGROLL MNIST

Evolution Strategies training of a 3-layer MLP on MNIST, using custom Triton GPU kernels and algorithmic optimizations to minimize the gap to backprop.

### Results

Same architecture (784-128-128-10 MLP, GELU, 10 epochs), same GPU (RTX 4080 SUPER), fp32 data:

| | Backprop (optimized) | Backprop (naive) | EGGROLL |
|---|---|---|---|
| **Training time** | **1.5s** | 4.5s | **2.9s** |
| **Per-epoch compute** | 0.03s | 0.2s | **0.14s** |
| **Test accuracy** | 97.5% | 97.3% | 97.3% |
| **Peak memory** | 389 MB | 391 MB | 389 MB |

Both JAX implementations use `jax.lax.scan` + all-in-one JIT. The naive backprop uses Python loops with per-batch GPU synchronization, which inflates its time 3x.

EGGROLL is **1.9x slower** than optimized backprop at equal optimization level. It does **3,600x more FLOPs** per batch (3,600 forward passes via 1,800 antithetic perturbation pairs vs 1 forward + 1 backward). The fused Triton kernel with FP8 tensor cores compresses this compute gap from 3600x to 1.9x.

### How to run

```bash
uv run mnist_backprop.py              # naive backprop (Python loop)
uv run mnist_backprop_optimized.py    # optimized backprop (lax.scan)
uv run benchmark.py                   # eggroll (single seed)
uv run validate.py                    # eggroll (3-seed validation)
```

Requires `uv` ([install](https://docs.astral.sh/uv/getting-started/installation/)) and a GPU with ~500MB VRAM.

### Agent-driven optimization

The optimizations were developed using an autonomous coding agent loop, inspired by Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) approach. The agent reads `program.md` (which defines the experiment loop, constraints, and ideas), writes code, benchmarks, keeps or reverts, and iterates.

To reproduce or extend: point a coding agent at `program.md` and let it run. The `cuda_kernels_docs/` directory contains Triton and jax-triton documentation for the agent to reference.

### What made it fast

The optimization went from **27s to 2.9s** (9.4x speedup). Key optimizations:

**Sessions 1-4: Kernel + compilation (27s -> 5.2s)**

1. **Fused 3-layer Triton kernel** — fuses L1+L2+L3 forward pass + CE loss into a single GPU kernel. Intermediates stay in registers, eliminating ~95% of HBM traffic.
2. **K-tiled L1->L2 matmul + FP8 tensor cores** — compute `l1` in (64, 32) tiles within a K-loop, halving register pressure. FP8 E4M3 gives 2x tensor core throughput.
3. **All-in-one JIT** — nested `jax.lax.scan` (epoch + batch) eliminates all Python loop overhead.
4. **Merged pos/neg kernel** — single 3D grid computes both +sigma and -sigma CE in one launch.
5. **Shuffle-once + CPU data** — data shuffled once on CPU (inside timing), removing permutation ops from XLA graph.

**Session 5: Algorithmic — population reduction (5.2s -> 3.7s)**

6. **Label smoothing in CE fitness** (α=0.02) — replaces hard one-hot labels with `0.98 * one_hot + 0.02/10` in the kernel's CE computation. This gives the ES gradient estimator information about all 10 class logits (not just the correct one), providing richer fitness signal per perturbation. Enabled reducing population from 5000 to 2750.

7. **Per-layer LR scaling** (2x for layer 3) — layer 3 (128→10, 1280 params) gets a much better gradient estimate from N perturbations than layer 1 (784→128, 100K params). Boosting layer 3's learning rate 2x lets it converge faster without noise issues.

**Session 6: Algorithmic — further population reduction (3.7s -> 2.9s)**

8. **Adaptive label smoothing** (α=0.20 * 0.5^epoch) — high smoothing early when random weights produce noisy gradients, decaying to near-zero for sharp CE during fine-tuning. Broke through the pop2500 barrier that 50+ constant-α configs failed at.

9. **Per-subgroup Winsorized z-score** (K=8 groups, clip ±2.0) — instead of one global z-score over all perturbation fitness differences, split into 8 subgroups and normalize each independently with outlier clipping. Reduces the outsized influence of extreme fitness values on the gradient estimate.

10. **Higher sigma at low pop** (σ=0.036) — with fewer perturbations, a larger perturbation scale gives stronger fitness signal. Combined with the above, enabled reducing population from 2750 to 1800.

See `kernels/fused_3layer_ce.py` and `program.md` for the full optimization log.

### What didn't work

**Kernel/compilation (Sessions 1-4):**
BLOCK_B=16/128, BLOCK_K=64, num_stages=2+, num_warps=2/8, doubly-tiled J+K kernel, inline tl.randn, separate Triton RNG, pure JAX, per-batch JIT, bf16 gradients, Pallas kernels, counter-based PRNG, maxnreg, XLA flags, compilation caching, fused pos/neg in single block, fusing xB1_T into kernel.

**Algorithmic (Sessions 5-6):**
Rank-based fitness shaping (95.6%), top-k truncation (96.8%), Boltzmann/softmax weighting (86.6%), one-sided ES (needs 2x pop), Rademacher/uniform perturbations, per-layer sigma scaling (hurts L1 gradient), structured perturbations via FWHT (+1.5s JIT), cosine LR (worse than exponential), temperature tuning (marginal), no-shuffle at low pop (costs accuracy), uint8 data transfer (setup hiding — reverted).

### Background

Implements the EGGROLL algorithm from ["Evolution Strategies at the Hyperscale"](https://arxiv.org/pdf/2511.16652) (Sarkar et al., 2025). EGGROLL uses low-rank (rank-1) perturbations so that perturbed forward passes never materialize full weight matrices: `x @ (W + sigma * outer(A, B)) = x @ W + sigma * (x @ B) * A`.

The ES gradient signal uses temperature-scaled cross-entropy (T=2.0) with adaptive label smoothing (α=0.20 decaying 0.5x/epoch) and antithetic sampling. Fitness normalization uses per-subgroup Winsorized z-score (K=8, clip ±2.0). Population size is 1,800 (3,600 effective with antithetic pairs).

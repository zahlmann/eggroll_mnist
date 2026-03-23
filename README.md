## EGGROLL MNIST

Evolution Strategies training of a 3-layer MLP on MNIST, using custom Triton GPU kernels and algorithmic optimizations to minimize the gap to backprop.

### Results

Same architecture (784-128-128-10 MLP, GELU, 10 epochs), same GPU (RTX 4080 SUPER), fp32 data:

| | Backprop (optimized) | Backprop (naive) | EGGROLL |
|---|---|---|---|
| **Training time** | **1.5s** | 4.5s | **3.7s** |
| **Per-epoch compute** | 0.03s | 0.2s | **0.22s** |
| **Test accuracy** | 97.5% | 97.3% | 97.2% |
| **Peak memory** | 389 MB | 391 MB | 389 MB |

Both JAX implementations use `jax.lax.scan` + all-in-one JIT. The naive backprop uses Python loops with per-batch GPU synchronization, which inflates its time 3x.

EGGROLL is **2.5x slower** than optimized backprop at equal optimization level. It does **5,500x more FLOPs** per batch (5,500 forward passes via 2,750 antithetic perturbation pairs vs 1 forward + 1 backward). The fused Triton kernel with FP8 tensor cores compresses this compute gap from 5500x to 2.5x.

### How to run

```bash
uv run mnist_backprop.py              # naive backprop (Python loop)
uv run mnist_backprop_optimized.py    # optimized backprop (lax.scan)
uv run benchmark.py                   # eggroll (single seed)
uv run validate.py                    # eggroll (3-seed validation)
```

Requires `uv` ([install](https://docs.astral.sh/uv/getting-started/installation/)) and a GPU with ~500MB VRAM.

### What made it fast

The optimization went from **27s to 3.7s** (7.3x speedup). Key optimizations:

**Sessions 1-4: Kernel + compilation (27s -> 5.2s)**

1. **Fused 3-layer Triton kernel** — fuses L1+L2+L3 forward pass + CE loss into a single GPU kernel. Intermediates stay in registers, eliminating ~95% of HBM traffic.
2. **K-tiled L1->L2 matmul + FP8 tensor cores** — compute `l1` in (64, 32) tiles within a K-loop, halving register pressure. FP8 E4M3 gives 2x tensor core throughput.
3. **All-in-one JIT** — nested `jax.lax.scan` (epoch + batch) eliminates all Python loop overhead.
4. **Merged pos/neg kernel** — single 3D grid computes both +sigma and -sigma CE in one launch.
5. **Shuffle-once + CPU data** — data shuffled once on CPU (inside timing), removing permutation ops from XLA graph.

See `kernels/fused_3layer_ce.py` and `program.md` for the full optimization log.

**Session 5: Algorithmic — population reduction (5.2s -> 3.7s)**

6. **Label smoothing in CE fitness** (α=0.02) — replaces hard one-hot labels with `0.98 * one_hot + 0.02/10` in the kernel's CE computation. This gives the ES gradient estimator information about all 10 class logits (not just the correct one), providing richer fitness signal per perturbation. Enabled reducing population from 5000 to 2750 while maintaining accuracy.

7. **Per-layer LR scaling** (2x for layer 3) — layer 3 (128→10, 1280 params) gets a much better gradient estimate from 2750 perturbations than layer 1 (784→128, 100K params). Boosting layer 3's learning rate 2x lets it converge faster without noise issues. Higher boosts (3x+) or boosting layer 2 hurt accuracy.

8. **Population tuning** — HALF_POPULATION reduced from 5000 to 2750 (45% reduction). Kernel compute scales linearly with population, saving ~1.5s. Label smoothing alpha scales inversely with population: pop5000→α=0, pop3500→α=0.1, pop3250→α=0.05, pop2750→α=0.02. Pop2500 fails accuracy with all tested configs.

### What didn't work

**Kernel/compilation (Sessions 1-4):**
BLOCK_B=16/128, BLOCK_K=64, num_stages=2+, num_warps=2/8, doubly-tiled J+K kernel, inline tl.randn, separate Triton RNG, pure JAX, per-batch JIT, bf16 gradients, Pallas kernels, counter-based PRNG, maxnreg, XLA flags, compilation caching, fused pos/neg in single block, fusing xB1_T into kernel.

**Algorithmic (Session 5):**
Rank-based fitness shaping (95.6%), top-k truncation (96.8%), Boltzmann/softmax weighting (86.6%), one-sided ES (needs 2x pop, no savings), Rademacher/uniform perturbations, orthogonal perturbations (too expensive), per-layer sigma, higher L3 boost (3x+), L2 boost, different weight initializations (glorot, he, lecun — orthogonal still best), no-shuffle (saves 0.2s but costs 0.15pp accuracy), batch-order shuffle (same as no-shuffle).

### Agent-driven optimization

The optimizations were developed using an autonomous coding agent loop, inspired by Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) approach. The agent reads `program.md` (which defines the experiment loop, constraints, and ideas), writes code, benchmarks, keeps or reverts, and iterates.

To reproduce or extend: point a coding agent at `program.md` and let it run. The `cuda_kernels_docs/` directory contains Triton and jax-triton documentation for the agent to reference.

### Background

Implements the EGGROLL algorithm from ["Evolution Strategies at the Hyperscale"](https://arxiv.org/pdf/2511.16652) (Sarkar et al., 2025). EGGROLL uses low-rank (rank-1) perturbations so that perturbed forward passes never materialize full weight matrices: `x @ (W + sigma * outer(A, B)) = x @ W + sigma * (x @ B) * A`.

The ES gradient signal uses temperature-scaled cross-entropy (T=2.0) with label smoothing (α=0.02) and antithetic sampling. Population size is 2,750 (5,500 effective with antithetic pairs).

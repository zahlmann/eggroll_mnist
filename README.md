## EGGROLL MNIST

Evolution Strategies training that **nearly matches backprop speed** on a 3-layer MLP, using custom Triton GPU kernels.

### Results

Same architecture (784-128-128-10 MLP, GELU, 10 epochs), same GPU (RTX 4080 SUPER), fp32 data:

| | Backprop | EGGROLL |
|---|---|---|
| **Training time** | 4.7s | 6.5s |
| **Steady-state (per epoch)** | 0.2s | **0.36s** |
| **Test accuracy** | 97.3% | 97.5% |
| **Peak memory** | 391 MB | 390 MB |

EGGROLL does **1,111x more FLOPs** than backprop (10,000 forward passes per batch vs 1 forward + 1 backward). Both times include JIT compilation on epoch 1 (~3s each). At steady state, EGGROLL is only **1.8x slower per epoch** despite the 1000x compute gap, thanks to a fused Triton kernel with FP8 tensor cores.

### How to run

```bash
uv run mnist_backprop.py          # backprop baseline
uv run benchmark.py               # eggroll (single seed)
uv run validate.py                # eggroll (3-seed validation)
```

Requires `uv` ([install](https://docs.astral.sh/uv/getting-started/installation/)) and a GPU with ~500MB VRAM.

### What made it fast

The optimization went from **27s to 6.5s** (4.2x speedup). Four things mattered:

**1. Fused 3-layer Triton kernel** (10.7s -> 7.2s, steady-state 1.0s -> 0.4s/epoch)

The bottleneck was memory bandwidth: intermediate activations (`l1`, `l2`, logits) were written to HBM then re-read multiple times per forward pass. A custom Triton kernel fuses all three layers + cross-entropy fitness into a single GPU kernel where intermediates stay in registers. This eliminates ~95% of HBM traffic.

**2. K-tiled L1→L2 matmul + FP8 tensor cores** (7.2s -> 6.5s, steady-state 0.4s -> 0.36s/epoch)

The original kernel loaded full `l1` (64x128) and `w2` (128x128) simultaneously, causing high register pressure (~192 regs/thread, only 25% occupancy). The K-tiled approach computes `l1` in (64, 32) tiles within a K-loop, feeding each tile directly into the L2 matmul accumulation. Only 32 columns of `l1` are live at any time, halving register pressure (~113 regs/thread, 4 blocks/SM).

Additionally, using FP8 E4M3 (`tl.float8e4nv`) for the L2 and L3 matmul operands gives 2x tensor core throughput on Ada Lovelace with no accuracy loss for the ES fitness signal.

See `kernels/fused_3layer_ce.py`.

**3. Pre-split PRNG keys** (27s -> 10.7s)

`jax.random.split(key)` inside a `jax.lax.scan` loop creates a sequential dependency between batches (each batch's key depends on the previous). Pre-generating all 468 keys before the scan breaks this chain, letting XLA pipeline batch computations.

**4. Epoch-level scan** (saves ~3s)

Wrapping all 468 batches in `jax.lax.scan` eliminates Python loop overhead and lets XLA compile the entire epoch as one GPU program.

### What didn't work

- **BLOCK_B=16 or 128**: bad tensor core utilization (16) or register spilling (128)
- **BLOCK_K=64, num_stages=4**: higher register pressure hurts occupancy
- **Doubly-tiled J+K kernel**: recomputing GELU 4x per J-tile costs more than the occupancy gain
- **Merging pos/neg directions in one kernel call**: sign loop doubles block time without improving concurrency
- **Pure JAX (no Triton)**: 2.5x slower per-epoch; the fused kernel's HBM savings are essential
- **Per-batch JIT (no scan)**: scan gives better XLA optimization than a Python loop
- **bf16 gradient matmuls**: saves 0.06s/epoch compute but adds 0.5s to JIT compilation
- **unsafe_rbg PRNG**: accuracy dropped below threshold
- **Compilation caching**: unfair (equivalent to AOT warmup across runs)

### Agent-driven optimization

The kernel optimizations were developed using an autonomous coding agent loop, inspired by Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) approach. The agent reads `program.md` (which defines the experiment loop, constraints, and ideas), writes code, benchmarks, keeps or reverts, and iterates.

To reproduce or extend: point a coding agent at `program.md` and let it run. The `cuda_kernels_docs/` directory contains Triton and jax-triton documentation for the agent to reference when writing GPU kernels.

### Background

Implements the EGGROLL algorithm from ["Evolution Strategies at the Hyperscale"](https://arxiv.org/pdf/2511.16652) (Sarkar et al., 2025). EGGROLL uses low-rank (rank-1) perturbations so that perturbed forward passes never materialize full weight matrices: `x @ (W + sigma * outer(A, B)) = x @ W + sigma * (x @ B) * A`.

The ES gradient signal uses temperature-scaled cross-entropy (T=2.0) with antithetic sampling (evaluate both +sigma and -sigma perturbations). Population size is 5,000 (10,000 effective with antithetic pairs).

Contact: johann.zahlmann@gmail.com

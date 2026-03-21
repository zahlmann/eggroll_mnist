## EGGROLL MNIST

Evolution Strategies training that **matches backprop speed** on a 3-layer MLP, using custom Triton GPU kernels.

### Results

Same architecture (784-128-128-10 MLP, GELU, 10 epochs), same GPU (RTX 4080 SUPER), fp32 data:

| | Backprop | EGGROLL |
|---|---|---|
| **Training time** | 4.7s | **4.5s** |
| **Test accuracy** | 97.3% | 97.3% |
| **Peak memory** | 391 MB | 390 MB |

EGGROLL does **1,111x more FLOPs** than backprop (10,000 forward passes per batch vs 1 forward + 1 backward). It matches wall-clock time by saturating tensor cores through a fused Triton kernel.

### How to run

```bash
uv run mnist_backprop.py          # backprop baseline
uv run benchmark.py               # eggroll (single seed)
uv run validate.py                # eggroll (3-seed validation)
```

Requires `uv` ([install](https://docs.astral.sh/uv/getting-started/installation/)) and a GPU with ~500MB VRAM.

### What made it fast

The optimization went from **27s to 4.5s** (6x speedup). Three things mattered:

**1. Fused 3-layer Triton kernel** (10.3s -> 4.5s)

The bottleneck was memory bandwidth: intermediate activations (`l1`, `l2`, logits) were written to HBM then re-read multiple times per forward pass. A custom Triton kernel fuses all three layers + cross-entropy fitness into a single GPU kernel where intermediates stay in registers. This eliminates ~95% of HBM traffic.

Key design decisions that made the kernel work (previous attempts failed):
- **BLOCK_B=64**: the (64, 128) x (128, 128) matmul tile is the sweet spot for tensor core utilization. BLOCK_B=16 was 3x slower (bad utilization), BLOCK_B=128 was 1.4x slower (register spilling).
- **One direction per kernel call**: processes positive OR negative perturbation, not both simultaneously. Halves peak register pressure.
- **Fast GELU**: `x * sigmoid(1.702x)` instead of `x * (1 + erf(x/sqrt(2)))/2`. Simpler for the compiler.

See `kernels/fused_3layer_ce.py`.

**2. Pre-split PRNG keys** (27s -> 10.7s)

`jax.random.split(key)` inside a `jax.lax.scan` loop creates a sequential dependency between batches (each batch's key depends on the previous). Pre-generating all 468 keys before the scan breaks this chain, letting XLA pipeline batch computations.

**3. Epoch-level scan + AOT compilation** (saves ~3s)

Wrapping all 468 batches in `jax.lax.scan` eliminates Python loop overhead. AOT-compiling the function before timing removes JIT cost.

### What didn't work

- **Triton with BLOCK_B=16**: register pressure caused low occupancy, slower than cuBLAS
- **Unrolling the inner scan**: XLA can't reuse buffers across unrolled iterations
- **Merging pos/neg directions in one kernel call**: doubles registers, kills performance
- **Large population chunks (>500)**: intermediates overflow L2 cache
- **unsafe_rbg PRNG**: accuracy dropped below threshold

### Background

Implements the EGGROLL algorithm from ["Evolution Strategies at the Hyperscale"](https://arxiv.org/pdf/2511.16652) (Sarkar et al., 2025). EGGROLL uses low-rank (rank-1) perturbations so that perturbed forward passes never materialize full weight matrices: `x @ (W + sigma * outer(A, B)) = x @ W + sigma * (x @ B) * A`.

The ES gradient signal uses temperature-scaled cross-entropy (T=2.0) with antithetic sampling (evaluate both +sigma and -sigma perturbations). Population size is 5,000 (10,000 effective with antithetic pairs).

Contact: johann.zahlmann@gmail.com

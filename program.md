# EGGROLL Kernel Optimization — Agent Program

*You are an AI researcher. Your job: write custom Triton GPU kernels to make
`mnist_eggroll_optimized.py` as fast as possible while keeping accuracy and
memory within bounds. You work autonomously, run experiments, log results, and
keep going without stopping to ask for permission.*

---

## The Goal

Backprop trains a 784→128→128→10 MLP on MNIST in ~4.5s.
EGGROLL (Evolution Strategies with low-rank perturbations) currently takes ~27s.

**Close the gap using Triton GPU kernels.**

Hard constraints enforced by `validate.py`:
- Test accuracy ≥ 97.2% (average over seeds 11, 42, 7)
- Peak GPU memory ≤ 500MB
- Locked constants unchanged (see below)

Speed target: ≤ 5s (stretch: ≤ 4.5s to match backprop).

---

## Setup (do this once at the start of each session)

1. `git checkout -b autoresearch/$(date +%Y%m%d-%H%M%S)`
2. Read this file, `kernels/__init__.py`, and `mnist_eggroll_optimized.py`
3. Read the Triton docs in `cuda_kernels_docs/` (start with the README there)
4. Run the baseline: `uv run benchmark.py` — note the numbers
5. Initialize `results.tsv` if it doesn't exist (it's gitignored)
6. Begin the experiment loop

---

## What You Can Modify

**`mnist_eggroll_optimized.py`** — the training script. You may:
- Replace JAX operations with Triton kernels
- Restructure the forward pass, fitness computation, or gradient accumulation
- Tune `LR_START`, `LR_DECAY`, `SIGMA_START`, `SIGMA_DECAY` (these are labelled "tunable")
- Import from `kernels/` or inline Triton kernels directly

**`kernels/`** — write Triton kernels here. Any structure you like.

---

## What You Cannot Change (validate.py will catch violations)

The following constants must remain exactly as-is in the CONSTANTS block printed
at the start of every run:

```
HIDDEN_DIM:      128
BATCH_SIZE:      128
EPOCHS:          10
HALF_POPULATION: 5000
T:               2.0
```

Also forbidden:
- Changing the network architecture (layer count, activation function, output size)
- Changing the dataset or data loading
- Modifying `validate.py` or `benchmark.py`
- Installing packages not already in `pyproject.toml` (you may add triton/jax-triton)

---

## The Algorithm (what you're optimizing)

EGGROLL uses antithetic Evolution Strategies with low-rank perturbations.
For each batch:

1. Sample 5000 perturbation pairs (A_i, B_i) per layer — rank-1 directions
2. For each population member i, perturb weights: W̃ = W + σ·outer(A_i, B_i)
3. Run forward pass: output_i = gelu(x @ (W + σ·outer(A,B)))
4. Evaluate fitness (temperature-scaled cross-entropy)
5. Antithetic: also evaluate the negative perturbation (−σ) — halves variance
6. Gradient estimate: Σ (fitness_diff_i · outer(B_i, A_i)) — no backprop needed

The bottleneck is step 3: computing 10,000 forward passes per batch (5000 pos + 5000 neg).
The key insight: the perturbation is rank-1, so `x @ W̃ = x@W + σ·(x@B)·A`.
This means you never need to materialize the full perturbed weight matrix.

**The big intermediate tensor to eliminate:**
```python
pert1 = xB1.T[:, :, None] * A1[:, None, :]   # shape (5000, 128, 128) = 80M elements
```
A fused Triton kernel can avoid materializing this in HBM.

---

## Experiment Loop (LOOP FOREVER until manually interrupted)

```
1. Come up with a kernel idea (see Ideas section below)
2. Implement it in mnist_eggroll_optimized.py (or kernels/)
3. git add -A && git commit -m "description of change" && git push
4. uv run benchmark.py > run.log 2>&1
5. Check: grep "test_accuracy\|training_time_s\|peak_memory_mb\|WARNING\|CRASHED" run.log
6. If crashed → tail -50 run.log → fix the bug → go to step 2
7. Read the numbers. Did time improve while accuracy/memory stayed in bounds?
8. Log to results.tsv (see format below)
9. If improved: keep the commit and continue from the new state
   If not improved: git reset --hard HEAD~1 (undo the commit)
10. Go to step 1
```

**Never stop to ask if you should continue. Never wait for human input.**
If you're uncertain about an approach, just try it — the cost of a failed run is ~30s.

---

## Logging (results.tsv)

Tab-separated. Append one row per run. Do NOT commit this file to git.

Columns: `commit\ttest_accuracy\ttraining_time_s\tpeak_memory_mb\tstatus\tdescription`

Status values:
- `keep` — improved speed while staying within accuracy/memory bounds
- `discard` — ran fine but didn't improve or made things worse
- `crash` — crashed or validate failed

Example:
```
a1b2c3d  0.9734  27.1  431  keep     baseline jax implementation
e4f5g6h  0.9731  19.4  428  keep     fused layer1 triton kernel
i7j8k9l  0.9128  15.0  420  discard  kernel bug broke accuracy
```

---

## Ideas to Try (roughly ordered by expected impact)

### High impact
1. **Fused perturbed-forward kernel (layer 1)**: take xb(128,784), W(784,128),
   A(5000,128), B(5000,784), sigma → output pos(5000,128,128) and neg(5000,128,128)
   activations. Tile over the population dimension. Avoid materializing the
   (5000,128,784) perturbation tensor. This is the single biggest win.

2. **Fused perturbed-forward + CE fitness**: extend the above to also compute
   cross-entropy fitness inline, outputting only fitness_pos(5000,) and
   fitness_neg(5000,). Eliminates the large logit tensors from HBM entirely.

3. **Fused layer 1+2+3 in a single kernel**: pipeline the three layers.

### Medium impact
4. **Vectorized CE fitness kernel**: fuse the softmax+CE computation over
   (5000,128,10) logits — avoids reading/writing the full tensor twice.

5. **Reduce precision in fitness**: compute CE fitness in float16 (accuracy
   only needs ~1% resolution for the ES gradient signal to work).

### Lower impact / architectural
6. Restructure population loop to improve memory access patterns
7. Use persistent kernels to avoid kernel launch overhead across batches

---

## Triton + JAX Integration

Install: `uv add triton jax-triton`

Basic pattern to replace a JAX function with a Triton kernel:

```python
import triton
import triton.language as tl
import jax_triton as jt
import jax.numpy as jnp

@triton.jit
def my_kernel(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, x * 2.0, mask=mask)

def my_op(x):
    N = x.size
    out_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK']),)
    return jt.triton_call(x, kernel=my_kernel, out_shape=out_shape,
                          grid=grid, N=N, BLOCK=512)
```

See `cuda_kernels_docs/jax_triton/` for full examples including matmul and
multi-output kernels. Critical: argument order to triton_call is
[input arrays] → [output arrays via out_shape] → [scalar args] → [constexpr kwargs].

---

## Precision Rules

Both EGGROLL and backprop must use **fp32 data** for a fair comparison. The forward
pass may use bf16 matmuls internally (JAX does this automatically via tensor cores),
but training data (`X_train`) and perturbation vectors must be generated in fp32.
No bf16 data loading tricks — the comparison must be apples-to-apples.

## Reference Numbers (uncontended GPU, RTX 4080 SUPER)

| Implementation     | Time    | Memory | Accuracy |
|--------------------|---------|--------|----------|
| Backprop (fp32)    | ~4.7s   | ~391MB | ~97.3%   |
| EGGROLL baseline   | ~27.3s  | ~390MB | ~97.6%   |
| EGGROLL optimized  | ~10.7s  | ~390MB | ~97.3%   |
| Your target        | ≤5s     | ≤500MB | ≥97.2%   |

Current speedup: 2.6x over baseline, 2.3x gap to backprop.
The 2.3x gap is algorithmic — EGGROLL evaluates 10,000 forward passes per batch
vs backprop's 1 forward + 1 backward. Closing it further requires reducing the
work per forward pass or increasing hardware utilization.

Always run `benchmark.py` at the start of a session to get the current baseline.
Check `nvidia-smi` — if another process is using the GPU, numbers will be inflated.

## What worked so far (session 2025-03-21)

1. **Pre-split random keys** (2.6x of the total speedup): generating all PRNG keys
   before the batch scan breaks the sequential dependency chain, letting XLA pipeline
   batch computations. This was the single biggest win.
2. **Epoch-level scan**: wrapping all 468 batches in `jax.lax.scan` eliminates Python
   loop overhead and lets XLA compile the entire epoch as one program.
3. **AOT compilation**: `jax.jit(...).lower(...).compile()` before timing removes
   JIT overhead from the measured training time.
4. **Chunked inner scan** (CHUNK=500): processing perturbations in chunks of 500
   keeps intermediates in L2 cache. Larger chunks overflow L2 and are slower.
5. **Single PRNG call**: generating one (5000, 1306) random matrix and slicing
   reduces kernel launch overhead vs 6 separate calls.

## What did NOT work

- **Triton kernels** (fused L1+L2, persistent fwd+fitness): register pressure caused
  low occupancy, making them slower than cuBLAS + XLA fusion. The small tile sizes
  (16×128) don't utilize tensor cores efficiently.
- **Unrolling the inner scan**: XLA can't reuse buffers across unrolled iterations,
  causing 2x slowdown.
- **Merging pos/neg directions**: concatenation overhead + reduced parallelism made
  it slower than separate pos/neg computation.
- **unsafe_rbg PRNG**: lower randomness quality dropped accuracy below threshold.
- **CHUNK>500**: L2 cache overflow. CHUNK=1000 → 1.6s/epoch, CHUNK=2500 → 4.1s/epoch.

## Ideas to try next

### High potential
1. **Custom CUDA kernel via pallas or raw CUDA**: bypass Triton's register allocator.
   Pallas (JAX's kernel language) might give better control than jax-triton.
2. **Fuse random generation + forward pass**: generate perturbation vectors on-the-fly
   inside the forward kernel instead of materializing them to HBM first.
3. **Reduce inner scan iterations**: find a way to process larger chunks without
   L2 overflow — e.g., tile the HIDDEN dimension to reduce per-chunk memory.
4. **Operator fusion via custom_vjp/custom_jvp**: manually fuse the xB dot product
   with the subsequent matmul to eliminate one read of l1/l2.

### Medium potential
5. **Async data shuffling**: overlap epoch data preparation with previous epoch's compute.
6. **Reduce gradient matmul cost**: the (784, 5000) @ (5000, 128) matmul for grad1
   is memory-bound. Could accumulate gradients inside the inner scan to avoid
   materializing all 5000 fitness values.
7. **Mixed scan/vmap**: use vmap within chunks for the element-wise ops and explicit
   matmul for the shared-weight operations.

### Speculative
8. **FP8 tensor cores**: RTX 4080 supports FP8 — 2x throughput over bf16. Need to
   check JAX/XLA support.
9. **Process multiple batches per scan step**: concatenate 2-4 batches and process
   together to amortize overhead. Requires static batch count.
10. **Gradient accumulation inside inner scan**: instead of outputting fitness and
    computing gradient separately, accumulate (B * fitness_diff) @ A incrementally.

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

## Fairness Rules (catch yourself on these)

Optimizations must be **honestly comparable** to the backprop baseline. Watch for:

1. **JIT/compilation hiding**: don't exclude JIT compilation time from the measured
   `training_time_s` (e.g., via AOT compilation or warmup before `start_time`).
   Both backprop and EGGROLL include first-epoch JIT in their timing.
2. **Activation mismatch**: if you change the activation in the training forward pass
   (e.g., fast GELU approximation), you MUST use the same activation in
   `evaluate_batch()`. Otherwise test accuracy measures a different network.
3. **Memory measurement**: `get_gpu_memory_mb()` should report actual usage,
   not just post-epoch snapshots that miss transient peaks. If you change the
   memory function, verify it catches in-kernel allocations.
4. **Precision asymmetry**: both scripts must use fp32 training data. Using bf16
   data for EGGROLL but fp32 for backprop is not a fair comparison.
5. **Data subsetting**: don't silently drop training samples to reduce batch count.
   Both scripts should process the same amount of data per epoch.

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
Gradient computation must also use fp32 perturbation vectors — do NOT substitute
bf16 vectors for the gradient matmuls (`B.T @ (shaped * A)`).
No bf16 data loading tricks — the comparison must be apples-to-apples.

## Reference Numbers (uncontended GPU, RTX 4080 SUPER)

| Implementation     | Time    | Memory | Accuracy |
|--------------------|---------|--------|----------|
| Backprop (fp32)    | ~4.7s   | ~391MB | ~97.3%   |
| EGGROLL baseline   | ~27.3s  | ~390MB | ~97.6%   |
| EGGROLL optimized  | ~6.5s   | ~390MB | ~97.5%   |
| Your target        | ≤5s     | ≤500MB | ≥97.2%   |

Current speedup: 4.2x over baseline, 1.4x gap to backprop.
The remaining gap is ~60% XLA JIT compilation (2.8s) and ~40% kernel compute (3.6s).

Always run `benchmark.py` at the start of a session to get the current baseline.
Check `nvidia-smi` — if another process is using the GPU, numbers will be inflated.

## What worked so far

### Session 2025-03-21
1. **Pre-split random keys** (2.6x of the total speedup): generating all PRNG keys
   before the batch scan breaks the sequential dependency chain, letting XLA pipeline
   batch computations. This was the single biggest win.
2. **Epoch-level scan**: wrapping all 468 batches in `jax.lax.scan` eliminates Python
   loop overhead and lets XLA compile the entire epoch as one program.
3. **Single PRNG call**: generating one (5000, 1306) random matrix and slicing
   reduces kernel launch overhead vs 6 separate calls.
4. **Fused 3-layer Triton kernel**: fuses L1+L2+L3 forward pass + CE loss into a
   single kernel. Intermediates (l1, l2, logits) never leave registers, eliminating
   ~95% of HBM traffic vs pure JAX chunked approach.

### Session 2025-03-22
5. **K-tiled L1→L2 matmul** (7.2s → 6.6s): compute l1 in (BLOCK_B, BLOCK_K=32)
   tiles within a K-loop, feeding each tile directly into the L2 matmul accumulation.
   Only 32 columns of l1 are live at any time → register pressure drops from ~192 to
   ~113 regs/thread → occupancy improves from ~12% to ~25%.
6. **FP8 E4M3 tensor cores** (6.6s → 6.5s): `tl.float8e4nv` for L2 and L3 matmul
   operands gives 2x tensor core throughput on Ada Lovelace. The ES fitness signal
   tolerates the precision loss (E4M3 has 3 mantissa bits).
7. **Single bf16 cast** (minor): cast the entire (5000, 1306) perturbation matrix to
   bf16 once instead of 6 separate per-vector casts. Simplifies XLA graph.

## What did NOT work

- **Triton with BLOCK_B=16**: bad tensor core utilization, slower than cuBLAS.
- **BLOCK_B=128 with 8 warps**: register pressure kills occupancy (2 blocks/SM).
- **BLOCK_B=32**: higher occupancy (6 blocks/SM) doesn't compensate for 33% more waves.
- **BLOCK_K=64**: larger K-tiles increase register pressure, negating the fewer iterations.
- **num_stages=4**: more pipeline stages consume more registers, reducing occupancy.
- **Doubly-tiled J+K kernel**: tiling L2 output dimension (BLOCK_J=32) reduces the
  accumulator from 64×128 to 64×32, but recomputing GELU 4× per J-tile costs more
  than the occupancy gain (0.7s/epoch vs 0.4s).
- **Merging pos/neg into one kernel call**: sign loop doubles block execution time
  without improving concurrency. Shared data loads save ~25ms/epoch but don't
  compensate for the complexity.
- **Pure JAX (no Triton)**: 12.8s. JIT is nearly the same (2.6s vs 2.8s) — the JIT
  bottleneck is XLA compilation of the scan body, NOT Triton kernel compilation.
  Per-epoch is 2.5× slower due to HBM traffic for intermediates.
- **Per-batch JIT (no scan)**: 7.0s. Removing the scan doesn't help JIT and loses
  XLA optimization benefits.
- **bf16 gradient matmuls**: saves 0.06s/epoch compute but adds 0.5s to XLA JIT.
  The mixed-precision matmul creates a more complex XLA graph.
- **Compilation caching** (`jax_compilation_cache_dir`): reduces JIT from 2.8s to
  1.5s on warm runs, but this is effectively AOT warmup and unfair.
- **unsafe_rbg PRNG**: lower randomness quality dropped accuracy below threshold.

## Time budget breakdown (where 6.5s goes)

| Component | Time | Notes |
|-----------|------|-------|
| XLA JIT compilation | 2.8s | Compiling scan body + Triton custom_call. ~2.6s is XLA itself, ~0.2s is Triton |
| Triton kernel (×2 per batch) | ~2.8s | 10K blocks × 2 calls × 468 batches × 10 epochs, ~53% GPU utilization |
| Random gen + casts | ~0.4s | jax.random.normal + single bf16 cast, 468 batches × 10 epochs |
| Gradient matmuls | ~0.3s | B.T @ (shaped * A) for 3 layers, memory-bound |
| Other (scan overhead, etc.) | ~0.2s | XLA While loop management, data shuffling |

To reach 5s: need to save 1.5s. The JIT is the elephant in the room.

## Ideas to try next

### Reducing JIT (the bottleneck — 2.8s of 6.5s)
1. **Pallas kernels**: JAX's native kernel language compiles as part of XLA, potentially
   eliminating the Triton custom_call overhead. BUT: Pallas uses Triton as GPU backend,
   so the savings may be zero. Worth testing to confirm.
2. **Simpler scan body**: reduce the number of XLA ops by fusing more computation into
   the Triton kernel (e.g., bf16 casts, base1/xB1_T matmuls). Each eliminated XLA op
   might shave ms off compilation.
3. **JAX version upgrade**: newer JAX versions may have faster XLA compilation.

### Reducing kernel compute (0.36s/epoch, needs ~0.22s for 5s target)
4. **Fuse random generation into Triton kernel**: use `tl.randn(seed, offset)` to
   generate perturbation vectors on-the-fly inside the kernel, eliminating HBM round-trip
   for random vectors (~40MB/batch). Challenge: gradient computation still needs the
   vectors, so either write them to HBM from the kernel (same traffic) or regenerate
   them in a separate gradient kernel (same seed/offset → same values).
5. **Gradient accumulation inside forward kernel**: output gradient contributions
   directly instead of fitness. Each block would atomicAdd its `shaped[p] * outer(B[p], A[p])`
   contribution to the gradient. Problem: 5000 × 100K atomics per batch is ~50ms.
   Could work if done with block-level reduction (groups of 50-100 members reduce
   locally before atomicAdd).
6. **Better tensor core utilization**: current ~53% utilization. The bottleneck is
   occupancy (4 blocks/SM, 25%). Ideas:
   - Reduce register pressure further (currently ~113 regs/thread)
   - Use `tl.dot` with accumulator directly (done, `base2 = tl.dot(a, b, base2)`)
   - Experiment with num_warps=2 (fewer threads per block, more blocks per SM)
7. **Process base1 and xB1_T matmuls inside the Triton kernel**: currently these are
   separate cuBLAS calls. Moving them into the kernel eliminates 2 kernel launches
   per batch and reduces the scan body's XLA graph.

### Reducing non-kernel overhead
8. **Overlap data shuffling with training**: pre-compute next epoch's permutation
   during current epoch. Requires breaking the key dependency chain (pre-generate
   all epoch keys).
9. **Skip fitness normalization**: use raw `fitness_diff` instead of `(fitness_diff - mean) / std`.
   This allows streaming gradient accumulation (no need to see all 5000 fitness values
   before computing gradients). Might hurt convergence — needs accuracy testing.

### Speculative / architectural
10. **Rewrite in PyTorch + Triton**: eliminates JAX's XLA JIT entirely (~2.8s savings).
    Not allowed by current pyproject.toml constraints, but would immediately reach ~3.7s.
11. **CUDA Graphs**: capture the entire training loop as a CUDA graph to eliminate
    per-kernel launch overhead. JAX has experimental support via `jax.experimental.export`.
12. **Multi-stream execution**: overlap pos/neg kernel calls using separate CUDA streams.
    Not easily possible through jax-triton but could work with raw CUDA integration.

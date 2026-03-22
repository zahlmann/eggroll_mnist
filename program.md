# EGGROLL Kernel Optimization — Agent Program

*You are an AI researcher. Your job: write custom Triton GPU kernels to make
`mnist_eggroll_optimized.py` as fast as possible while keeping accuracy and
memory within bounds. You work autonomously, run experiments, log results, and
keep going without stopping to ask for permission.*

---

## The Goal

Backprop trains a 784→128→128→10 MLP on MNIST in ~4.7s.
EGGROLL (Evolution Strategies with low-rank perturbations) currently takes ~5.8s.

**Close the gap using Triton GPU kernels and other fair optimizations.**

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
- **Changing the framework** (e.g., rewriting in PyTorch). Both EGGROLL and the backprop
  baseline must use JAX. A PyTorch EGGROLL at 4.2s vs JAX backprop at 4.7s is NOT a
  fair comparison — PyTorch backprop runs in 1.1s, making EGGROLL 3.8× slower.

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
6. **Effective batch size**: the ES gradient must evaluate perturbations on
   BATCH_SIZE=128 samples, not more. Grouping multiple batches per ES step
   (GROUP_SIZE>1) effectively changes the batch size and is unfair.
7. **Setup hiding**: any data preprocessing (shuffling, grouping, transfers) that
   replaces work previously done inside `training_time_s` must remain inside the
   timing window.
8. **Gradient precision**: gradient computation must use fp32 perturbation vectors.
   Do NOT substitute bf16 vectors for the gradient matmuls (`B.T @ (shaped * A)`).

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
| EGGROLL optimized  | ~5.8s   | ~389MB | ~97.4%   |
| Your target        | ≤5s     | ≤500MB | ≥97.2%   |

Current speedup: 4.7x over baseline, 1.2x gap to backprop.

Always run `benchmark.py` at the start of a session to get the current baseline.
Check `nvidia-smi` — if another process is using the GPU, numbers will be inflated.

## What worked so far

### Session 1 (27s → 10.7s)
1. **Pre-split random keys**: generating all PRNG keys before the batch scan breaks
   the sequential dependency chain, letting XLA pipeline batch computations.
2. **Epoch-level scan**: wrapping all 468 batches in `jax.lax.scan` eliminates Python
   loop overhead and lets XLA compile the entire epoch as one program.
3. **Single PRNG call**: generating one (5000, 1306) random matrix and slicing
   reduces kernel launch overhead vs 6 separate calls.
4. **Fused 3-layer Triton kernel**: fuses L1+L2+L3 forward pass + CE loss into a
   single kernel. Intermediates (l1, l2, logits) never leave registers, eliminating
   ~95% of HBM traffic vs pure JAX chunked approach.

### Session 2 (10.7s → 6.5s)
5. **K-tiled L1→L2 matmul** (7.2s → 6.6s): compute l1 in (BLOCK_B, BLOCK_K=32)
   tiles within a K-loop, feeding each tile directly into the L2 matmul accumulation.
   Only 32 columns of l1 are live at any time → register pressure drops from ~192 to
   ~113 regs/thread → occupancy improves from ~12% to ~25%.
6. **FP8 E4M3 tensor cores** (6.6s → 6.5s): `tl.float8e4nv` for L2 and L3 matmul
   operands gives 2x tensor core throughput on Ada Lovelace. The ES fitness signal
   tolerates the precision loss (E4M3 has 3 mantissa bits).
7. **Single bf16 cast** (minor): cast the entire (5000, 1306) perturbation matrix to
   bf16 once instead of 6 separate per-vector casts. Simplifies XLA graph.

### Session 3 (6.5s → 5.8s)
8. **num_stages=1** (6.5s → 6.1s): reducing Triton software pipelining stages frees
   registers used for prefetch buffers, improving occupancy.
9. **All-in-one JIT** (6.1s → 6.0s): wrapping all 10 epochs in a single JIT call
   eliminates Python loop overhead. Uses nested scan (epoch scan + batch scan).
10. **Shuffle-once on CPU** (6.0s → 5.8s): data shuffled once before training
    (included in timing), removing `jax.random.permutation` + fancy indexing from
    the XLA graph. Training data kept in numpy to reduce GPU memory.
11. **Merged pos/neg kernel**: single triton_call with 3D grid (HALF_POP, N_TILES, 2)
    computes both +sigma and -sigma CE in one launch. Removes one custom_call from
    the XLA graph, reducing compilation by ~0.04s.
12. **fold_in key derivation**: `jax.random.fold_in` for per-batch keys instead of
    pre-splitting with `jax.random.split`.

## What did NOT work

### Kernel parameter tuning
- **BLOCK_B=16**: bad tensor core utilization, slower than cuBLAS.
- **BLOCK_B=32**: higher occupancy (6 blocks/SM) doesn't compensate for 33% more waves.
- **BLOCK_B=128**: register pressure kills occupancy (2 blocks/SM).
- **BLOCK_K=64**: larger K-tiles increase register pressure even with num_stages=1.
- **BLOCK_K=16**: FP8 tensor cores require K≥32 — crashes.
- **num_stages=4**: more pipeline stages consume more registers, reducing occupancy.
- **num_stages=2**: 6.0s — still adds register pressure from prefetch buffers.
- **num_warps=8**: with BLOCK_B=64, per-epoch compute 25% slower (register spilling).
- **num_warps=2**: worse occupancy than num_warps=4.
- **bf16 instead of fp8 matmuls**: 5.94s — FP8 tensor cores are essential (2× throughput).
- **maxnreg=96/80** (via @triton.autotune Config): no effect — either jax-triton
  doesn't pass maxnreg through, or the register spills offset occupancy gains.

### Kernel restructuring
- **Doubly-tiled J+K kernel**: tiling L2 output dimension (BLOCK_J=32) reduces the
  accumulator from 64×128 to 64×32, but recomputing GELU 4× per J-tile costs more
  than the occupancy gain (0.7s/epoch vs 0.4s).
- **Fusing xB1_T into Triton kernel**: 10.8s — computing B1·xb dot products per-block
  is 2× slower than cuBLAS doing the full matmul.
- **Fused pos/neg in single block**: 6.74s — compute both signs sequentially in one
  block (halves grid to 10K). The doubled compute per block + register pressure from
  code duplication outweighs L2 cache reuse of shared data.
- **Packed perturbation vectors** (single stride-1306 tensor): 6.20s — passing all_vecs_f
  as one tensor instead of 5 separate slices. Stride-1306 access pattern is worse than
  stride-128 for individual slices, negating any argument-count reduction benefit.

### Random generation
- **Separate Triton RNG kernel** (replacing jax.random.normal): Triton compile
  overhead offsets XLA graph simplification. Net neutral.
- **Inline tl.randn in forward+gradient kernels**: catastrophically slow (74.7s).
  tl.randn generates ~4B values/sec, vs JAX's ~100B values/sec bulk generation.
- **Separate vector generation** (6 jax.random.normal calls instead of 1): 6.2s,
  significantly slower due to extra key splitting and PRNG overhead.
- **unsafe_rbg PRNG**: lower randomness quality dropped accuracy below threshold.
- **rbg PRNG** (jax_default_prng_impl="rbg"): 6.12s — slower than threefry, not faster.
- **Counter-based PRNG** (SplitMix32 hash + Box-Muller): 5.88s, 97.72% acc — eliminates
  threefry from HLO (~30 ops vs ~338) but no JIT speedup. PRNG HLO ops are NOT the
  compilation bottleneck — the jax-triton bridge serialization dominates.

### Algorithmic changes (tried and reverted for fairness or accuracy)
- **Rademacher perturbations** (random ±1): 5.89s but accuracy drops to 97.1% on
  some seeds. Noisier than Gaussian.
- **Uniform perturbations** (uniform[-sqrt(3), sqrt(3)]): 5.70s (saves ~0.1s) but
  avg accuracy 97.18% across 3 seeds fails validation (seed 7: 97.03%). Tuning
  LR_DECAY=0.92 and SIGMA_START=0.032 didn't help.
- **Epoch-level RNG (all vectors)**: 5.71s but accuracy collapses to 96.5% — only
  5000 perturbation directions per epoch is not enough diversity.
- **Epoch-level B1 only**: marginal accuracy improvement but no speed gain.
- **GROUP_SIZE=2** (2 batches per ES step): 5.2s but effectively changes batch size
  from 128 to 256 — removed as unfair.
- **GROUP_SIZE=3/4**: accuracy fails validation (<97.2%).
- **Skip fitness std normalization**: accuracy drops to 89.9%.

### XLA/compilation
- **Pure JAX (no Triton)**: 12.8s. JIT is nearly the same (2.6s vs 2.8s) — the JIT
  bottleneck is XLA compilation of the scan body, NOT Triton kernel compilation.
- **Per-batch JIT (no scan)**: 7.0s. Removing the scan doesn't help JIT and loses
  XLA optimization benefits.
- **bf16 gradient matmuls**: saves 0.06s/epoch compute but adds 0.5s to XLA JIT.
- **Compilation caching** (`jax_compilation_cache_dir`): unfair (AOT warmup).
- **scan unroll=2**: increases JIT by 0.3s (2× larger scan body in XLA graph).
- **XLA_FLAGS autotune=0**: crashes (shared memory exceeded for cuBLAS fallback).
- **Flat single scan (2340 iters)**: tiling data 10× blows up memory to 2GB.
- **Pre-cast xb to bf16 in epoch body**: no improvement, XLA already fuses the cast.
- **Per-epoch JIT (instead of all-in-one)**: 5.8s time but 750MB memory — over limit.
- **fori_loop instead of scan** for batch loop: 5.81s — identical. JAX compiles both
  to the same while_loop + dynamic_slice pattern internally.
- **donate_argnums** for buffer reuse: 5.87s — no effect. XLA already optimizes
  buffer reuse within the scan.
- **XLA_FLAGS** (latency_hiding_scheduler=false, command_buffer=, etc.): 5.78-5.91s
  — all within noise. No XLA flag significantly reduces compilation time.

### Pallas kernels (JAX native)
- **Pallas 3-layer fused CE kernel**: JIT 1.96s (saves 0.33s vs Triton 2.29s) but
  execution 4.68s (loses 1.42s vs Triton 3.26s). Pallas auto-generates Triton code
  that can't match the hand-tuned register/tiling optimizations. Net: 6.73s — worse.
- Key finding: Pallas JIT is 10× faster than jax-triton for simple kernels (0.064s
  vs 0.661s), but for complex kernels the savings are only 0.33s because the Pallas
  → Triton code generation itself takes time.
- Pallas `index_map` returns BLOCK INDICES (multiplied by block_shape automatically),
  NOT element indices. Using `i * BLOCK_B` double-counts — a common gotcha.

### JIT profiling results (Session 4 — more accurate)
The 2.4s JIT breaks down as measured by comparing first-call vs second-call times:
- **With Triton kernel**: first=5.55s, exec=3.26s, JIT=2.29s
- **Without Triton (dummy zeros)**: first=1.66s, exec=0.45s, JIT=1.21s
- **Triton JIT overhead**: 2.29 - 1.21 = **1.07s** — almost half the JIT!
  This is dominated by jax-triton serialization (0.89s lowering), not PTX generation.

The earlier estimate (lowering 0.89s + XLA 1.38s + Triton 0.11s) attributed too
much to XLA compilation. In reality, removing the Triton kernel from the scan body
drops JIT from 2.29s to 1.21s — the Triton bridge is the single largest JIT cost.

The HLO graph has ~1270 ops, of which 338 (27%) are threefry PRNG bit operations,
but replacing PRNG with counter-based hashing (~30 ops) didn't reduce JIT. The
PRNG ops are NOT the compilation bottleneck.

## Time budget breakdown (where 5.8s goes) — verified Session 4

### JIT breakdown (2.39s total)
| Component | Time | Notes |
|-----------|------|-------|
| Non-Triton JIT | 1.21s | XLA compilation of scan body without kernel (PRNG + matmuls + gradient) |
| Triton JIT overhead | 1.07s | jax-triton serialization + XLA handling of custom_call |
| (Triton PTX gen) | 0.11s | (subset of above: just the PTX compilation) |

### Execution breakdown (3.28s total, measured by comparing full vs no-kernel runs)
| Component | Time | % of exec | Notes |
|-----------|------|-----------|-------|
| Triton kernel + setup | 2.82s | 86% | Includes bf16 casts, base1/xB1_T matmuls, kernel launch |
| Random generation | 0.13s | 4% | jax.random.normal (5000×1306) + bf16 cast per batch |
| Gradient matmuls + updates | 0.31s | 10% | B.T @ (shaped * A) for 3 layers + weight updates |

### Other
| Component | Time | Notes |
|-----------|------|-------|
| CPU shuffle + GPU transfer | ~0.13s | Included in training_time_s, before JIT |

The kernel achieves only ~12% of FP8 peak throughput (81 TFLOPS / 660 TFLOPS theoretical)
due to 25% occupancy (4 blocks/SM, limited by ~113 registers/thread).

To reach 5s: need to save ~0.8s. The jax-triton bridge (1.07s) is the single largest
reducible cost, but Pallas can't replace it without losing kernel quality.

### PyTorch rewrite (tested Session 4, ruled unfair)
A full PyTorch + Triton rewrite was tested. Results:
- PyTorch EGGROLL: **4.2s**, 97.4% acc, 292MB — target reached
- But PyTorch backprop: **1.1s**, 97.3% acc — EGGROLL is 3.8× slower
- JAX backprop: **4.6s** — most of this is JIT overhead, not compute

The PyTorch rewrite eliminates JAX's 2.4s JIT but gives the same advantage to backprop.
Comparing PyTorch EGGROLL (4.2s) vs JAX backprop (4.6s) is misleading. This approach
is now explicitly forbidden in the rules above.

## Ideas to try next

### Remaining ideas (within JAX)
1. **Write the kernel in raw CUDA/PTX** and call via JAX's FFI instead of jax-triton.
   This would bypass the jax-triton serialization overhead (1.07s) while keeping the
   hand-tuned kernel quality. Very complex to implement.
2. **Persistent Triton kernel**: keep blocks resident on SMs across batches, computing
   multiple batch steps per block launch. Would amortize kernel launch overhead and
   potentially improve L2 cache reuse for weights. Requires restructuring the grid to
   be batch-aware.

### Ideas that are exhausted or have known blockers
- ~~PyTorch rewrite~~: 4.2s EGGROLL but unfair comparison (PyTorch backprop = 1.1s)
- ~~Reduce PRNG HLO ops~~: tried counter-based PRNG, didn't reduce JIT (bridge, not PRNG,
  is the bottleneck)
- ~~Pallas kernels~~: 10× faster JIT for simple kernels but auto-generated Triton code
  is 44% slower at execution. Net worse.
- ~~Reduce kernel register pressure via maxnreg~~: either not passed through jax-triton
  or spills offset gains
- ~~Kernel restructuring (fused pos/neg, packed vectors)~~: all made performance worse
- ~~Alternative loop structures (fori_loop, while_loop)~~: identical to scan
- ~~XLA flags, donate_argnums~~: no effect
- ~~Uniform perturbations~~: 0.1s faster but fails accuracy validation

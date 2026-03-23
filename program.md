# EGGROLL Optimization — Agent Program

*You are an AI researcher. Your job: make `mnist_eggroll_optimized.py` as fast
as possible while keeping accuracy and memory within bounds. The kernel is
already highly optimized — the next frontier is **algorithmic**: reducing the
population size while maintaining accuracy. You work autonomously, run
experiments, log results, and keep going without stopping to ask for permission.*

---

## The Goal

Backprop trains a 784→128→128→10 MLP on MNIST in ~1.5s (optimized JAX) or ~4.5s
(naive JAX with Python loops). EGGROLL currently takes ~3.7s with HALF_POP=2750.

**Close the gap to optimized backprop (1.5s) by further reducing HALF_POPULATION.**

The Triton kernel and JAX compilation are already exhaustively optimized (see
"What worked" / "What did NOT work" below). The only remaining lever is
**algorithmic**: a smaller population means fewer forward passes per batch,
directly reducing both compute time and memory.

Hard constraints enforced by `validate.py`:
- Test accuracy ≥ 97.2% (average over seeds 11, 42, 7)
- Peak GPU memory ≤ 500MB

Speed target: ≤ 3.0s. Stretch: ≤ 2.0s (close to optimized backprop's 1.5s).
The old 4.7s "backprop" target was against a naive JAX implementation with per-batch
GPU synchronization. The optimized backprop (lax.scan, all-in-one JIT) runs in 1.5s.
See `mnist_backprop_optimized.py`.

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
- **Tune `HALF_POPULATION`** (currently 5000) — this is the primary optimization target.
  Reducing it directly reduces compute and memory. You must also update `validate.py`
  to match when you change it.
- Tune `LR_START`, `LR_DECAY`, `SIGMA_START`, `SIGMA_DECAY`, `T` (labelled "tunable")
- Modify the fitness shaping, perturbation sampling strategy, or gradient estimation
- Replace JAX operations with Triton kernels
- Restructure the forward pass, fitness computation, or gradient accumulation
- Import from `kernels/` or inline Triton kernels directly

**`kernels/`** — Triton kernels. The fused 3-layer CE kernel may need HALF_POP
updates if the grid dimensions change.

**`validate.py`** — update the `REQUIRED["HALF_POPULATION"]` value to match
when you change `HALF_POPULATION` in the training script.

---

## What You Cannot Change

The following constants are locked:

```
HIDDEN_DIM:      128
BATCH_SIZE:      128
EPOCHS:          10
```

Also forbidden:
- Changing the network architecture (layer count, activation function, output size)
- Changing the dataset or data loading
- Adding **momentum or state** that carries across batches (unfair vs backprop SGD,
  which is stateless). Each batch's gradient estimate must depend only on the current
  batch data and fresh random perturbations.
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
9. **No cross-batch state / momentum**: the ES gradient estimate for each batch must
   depend only on fresh random perturbations and the current batch data. No EMA of
   gradients, no reuse of perturbation directions from prior batches, no accumulated
   statistics. Backprop SGD is stateless per batch — EGGROLL must be too.
10. **Population reduction must reduce compute**: the goal of lowering HALF_POPULATION
    is to reduce wall-clock time and memory. Do NOT compensate for smaller populations
    with extra compute (e.g., multiple forward passes per perturbation, ensembling,
    or increasing EPOCHS).

---

## The Algorithm (what you're optimizing)

EGGROLL uses antithetic Evolution Strategies with low-rank perturbations.
For each batch:

1. Sample 2750 perturbation pairs (A_i, B_i) per layer — rank-1 directions
2. For each population member i, perturb weights: W̃ = W + σ·outer(A_i, B_i)
3. Run forward pass: output_i = gelu(x @ (W + σ·outer(A,B)))
4. Evaluate fitness (temperature-scaled, label-smoothed cross-entropy)
5. Antithetic: also evaluate the negative perturbation (−σ) — halves variance
6. Gradient estimate: Σ (fitness_diff_i · outer(B_i, A_i)) — no backprop needed

The bottleneck is step 3: computing 5,500 forward passes per batch (2750 pos + 2750 neg).
The key insight: the perturbation is rank-1, so `x @ W̃ = x@W + σ·(x@B)·A`.
This means you never need to materialize the full perturbed weight matrix.

**The big intermediate tensor to eliminate:**
```python
pert1 = xB1.T[:, :, None] * A1[:, None, :]   # shape (2750, 128, 128) = 45M elements
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

| Implementation           | Time    | Memory | Accuracy |
|--------------------------|---------|--------|----------|
| Backprop optimized (JAX) | ~1.5s   | ~389MB | ~97.5%   |
| Backprop naive (JAX)     | ~4.5s   | ~391MB | ~97.3%   |
| EGGROLL baseline         | ~27.3s  | ~390MB | ~97.6%   |
| EGGROLL optimized        | ~3.7s   | ~389MB | ~97.2%   |
| Your target              | ≤3.0s   | ≤500MB | ≥97.2%   |

Current speedup: 7.3x over baseline, 2.5x gap to optimized backprop.

**Population scaling** (measured, Session 5, with label smoothing + L3 boost):
- HALF_POP=5000: ~5.2s, ~97.4% acc (no smoothing needed)
- HALF_POP=4000: ~4.5s, ~97.3% acc (no smoothing needed)
- HALF_POP=3500: ~4.3s, ~97.3% acc (α=0.1 smoothing)
- HALF_POP=3000: ~3.9s, ~97.3% acc (α=0.02 smoothing)
- HALF_POP=2750: ~3.7s, ~97.2% acc (α=0.02, current)
- HALF_POP=2500: ~3.5s est., accuracy fails all configs
- HALF_POP=2000: ~3.1s est., accuracy likely <96.5%

The kernel time scales roughly linearly with HALF_POP. JIT time (~1.6s) is constant.

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

## Time budget breakdown (where 3.7s goes) — updated Session 5

At HALF_POP=2750 with label smoothing + L3 boost:

### Component breakdown (3.7s total)
| Component | Time | % | Notes |
|-----------|------|---|-------|
| JIT compilation | 1.6s | 43% | XLA + jax-triton serialization |
| Triton kernel | ~1.6s | 43% | 2750×2×2 = 11K blocks per batch |
| Random generation | ~0.07s | 2% | jax.random.normal (2750×1306) |
| Gradient matmuls | ~0.15s | 4% | B.T @ (shaped * A) for 3 layers |
| Data prep | ~0.2s | 5% | CPU shuffle + GPU transfer |

The kernel achieves ~12% of FP8 peak throughput at 25% occupancy (4 blocks/SM,
limited by ~113 registers/thread). JIT is dominated by jax-triton bridge serialization.

### Previous breakdown (where 5.8s went) — verified Session 4

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

### Session 5 (5.2s → 3.7s) — algorithmic population reduction
13. **Label smoothing in CE fitness** (5.2s → 3.9s at pop3000): replacing hard one-hot
    labels with `(1-α) * one_hot + α/num_classes` in the kernel's CE computation gives
    the ES gradient estimator information about ALL class logits, not just the correct
    one. This provides a richer fitness signal per perturbation, enabling lower populations.
    Optimal α scales inversely with population: pop5000→0, pop3500→0.1, pop3000→0.02,
    pop2750→0.02. At pop2500, no α value rescues accuracy.

14. **Per-layer LR scaling** (improves accuracy ~0.05pp): layer 3 (128→10, 1280 params)
    gets a much better gradient estimate from N perturbations than layer 1 (784→128,
    100K params). Using 2x learning rate for layer 3 gives better accuracy and
    consistency. Higher boosts (3x+) or boosting layer 2 hurts.

15. **Hyperparameter tuning at lower pop**: LR_DECAY=0.88 (was 0.88 at pop5000 too)
    works best at pop2750. LR_START=0.012, SIGMA_START=0.028 unchanged.

### What did NOT work — Session 5 (algorithmic)
- **Rank-based fitness shaping**: 95.6% accuracy — far worse than z-score.
- **Top-k truncation** (keep top 50%): 96.8% — worse than using all perturbations.
- **Boltzmann/softmax weighting**: 86.6% — catastrophic.
- **One-sided ES** (no antithetic): needs 2x population for same accuracy, no savings.
- **Rademacher/uniform perturbations**: too noisy for reliable 97.2% accuracy.
- **No-shuffle optimization**: saves 0.2s but costs 0.15pp accuracy — not worth it.
- **Batch-order shuffle**: same accuracy as no-shuffle (within-batch composition matters).
- **Different initializations** (glorot, he, lecun): orthogonal still best.
- **L3 boost >2x or L2 boost**: hurt accuracy.
- **Sigma variations** at pop4000: sigma=0.028 is optimal.
- **Label smoothing α>0.1**: too aggressive, over-softens fitness signal.

### PyTorch rewrite (tested Session 4, ruled unfair)
A full PyTorch + Triton rewrite was tested. Results:
- PyTorch EGGROLL: **4.2s**, 97.4% acc, 292MB — target reached
- But PyTorch backprop: **1.1s**, 97.3% acc — EGGROLL is 3.8× slower
- JAX backprop: **4.6s** — most of this is JIT overhead, not compute

The PyTorch rewrite eliminates JAX's 2.4s JIT but gives the same advantage to backprop.
Comparing PyTorch EGGROLL (4.2s) vs JAX backprop (4.6s) is misleading. This approach
is now explicitly forbidden in the rules above.

## Ideas to try next — Further population reduction

HALF_POP=2750 is the current floor. Pop2500 fails accuracy with all tested configs
(50+ combos of α, LR, sigma, L3 boost). To go lower, you need fundamentally better
gradient estimation from fewer perturbations.

### What's already been tried and failed (don't re-test these)
- ~~Rank-based, top-k, Boltzmann fitness shaping~~ — all worse than z-score
- ~~One-sided ES~~ — needs 2x pop for same accuracy
- ~~Rademacher/uniform perturbations~~ — too noisy
- ~~Orthogonal perturbations via QR~~ — too expensive per-batch
- ~~No-shuffle~~ — saves 0.2s but costs accuracy
- ~~L3 boost >2x, L2 boost~~ — hurts accuracy
- ~~Different initializations~~ — orthogonal is best
- ~~All kernel params~~ — exhaustively optimized Sessions 1-4
- ~~Pallas kernels~~ — worse execution, marginal JIT savings
- ~~Momentum / EMA / cross-batch state~~ — forbidden

### Ideas that haven't been tried
1. **Variance-reduced gradient estimation**: use control variates to reduce gradient
   noise. E.g., subtract the gradient estimate from a simpler surrogate model.
   Different from z-score normalization (which is a shaping, not variance reduction).

2. **Adaptive label smoothing per epoch**: start with α=0.1 (more smoothing when
   gradients are noisy early in training) and decay to α=0.01 by the end. The optimal
   α might change as the loss landscape evolves.

3. **Per-layer perturbation scaling**: instead of the same sigma for all layers,
   use different sigma_l proportional to the layer's gradient noise level.
   Different from per-layer LR (which was tested) — this changes the PERTURBATION
   scale, not the update scale.

4. **Structured random matrices**: use fast transforms (Hadamard, DCT) instead of
   dense Gaussian vectors for perturbations. Can provide better space coverage with
   the same number of perturbations. Different from orthogonalization (QR is O(N^3),
   Hadamard is O(N log N)).

5. **Gradient accumulation across perturbation subsets**: split 2750 perturbations
   into K groups, compute gradient from each group, average. Might reduce variance
   compared to one big z-score normalization. NOT cross-batch state — all within
   one batch's perturbation set.

6. **Learned perturbation directions**: pre-train a set of perturbation basis vectors
   on the first epoch's data, then reuse for subsequent epochs. Only useful if the
   basis is computed ONCE inside the training timer (not pre-computed).

7. **Migrating from jax-triton to Pallas backend='triton'**: the jax-triton package
   is unmaintained. Pallas with the Triton backend uses a newer XLA-native compilation
   pipeline. JIT savings estimated at ~0.3s. Requires rewriting the kernel in Pallas DSL.

8. **Reducing JIT via scan restructuring**: JIT is 1.6s (43% of total). If you could
   move the Triton kernel call outside the scan body, JIT drops by ~0.5s. This would
   require a vectorized (non-scan) approach to the forward pass.

### What NOT to try (forbidden by fairness rules)
- ~~Momentum / EMA of gradients~~: unfair, backprop SGD is stateless
- ~~Reusing perturbation directions across batches~~: cross-batch state, unfair
- ~~Increasing EPOCHS~~: locked constant
- ~~Ensembling or multiple passes~~: extra compute defeats the purpose
- ~~Framework rewrite (PyTorch)~~: unfair comparison (PyTorch backprop is 1.1s)
- ~~Compilation caching~~: unfair AOT warmup

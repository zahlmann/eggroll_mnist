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
3. git add -A && git commit -m "description of change"
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

## Reference Numbers

| Implementation | Time   | Memory | Accuracy |
|---------------|--------|--------|----------|
| Backprop      | ~4.5s  | ~391MB | ~97.5%   |
| EGGROLL JAX   | ~66s   | ~433MB | ~97.6%   |
| Your target   | ≤10s   | ≤500MB | ≥97.2%   |

Note: The README cites ~27s but that was measured without competing GPU load. There
is another process using ~6GB VRAM, leaving ~4GB free. EGGROLL only needs ~433MB so
there is no VRAM pressure, but GPU compute is shared. Always run `benchmark.py` first
at the start of a session to get the current baseline — it may vary.

Each run uses ~433MB peak — there is plenty of headroom for kernel experiments.

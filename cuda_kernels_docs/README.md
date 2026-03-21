# CUDA Kernels Documentation Index

Quick reference for a coding agent writing Triton GPU kernels for JAX.

## Where to start

1. **New to Triton?** Read `triton/introduction.md` (5 min) then `triton/01_vector_add.md`.
2. **Writing a matmul-style kernel?** Read `triton/03_matrix_multiply.md` — the blocked
   tiling pattern is the foundation for the fused forward pass kernel you need here.
3. **Calling Triton from JAX?** Read `jax_triton/jax_triton_readme.md` then
   `jax_triton/working_code_examples.md` (has 7 working copy-paste examples).
4. **Full API reference?** `triton/language_reference.md` (tl.load, tl.store, tl.dot, etc.)

---

## Files

### triton/
| File | What's in it |
|------|--------------|
| `introduction.md` | Why Triton exists, SPMD blocked model vs CUDA scalar model |
| `01_vector_add.md` | Hello-world: @triton.jit, tl.load/store, pointer arithmetic, masking |
| `02_fused_softmax.md` | Kernel fusion motivation, row-wise softmax, ~4x speedup over naive |
| `03_matrix_multiply.md` | Blocked tiling, L2 cache grouping, @triton.autotune, leaky ReLU fusion |
| `api_reference.md` | @triton.jit, @triton.autotune, @triton.heuristics, triton.Config |
| `language_reference.md` | Full tl.* API: load, store, dot, arange, reduce, atomic, math ops |

### jax_triton/
| File | What's in it |
|------|--------------|
| `jax_triton_readme.md` | Overview, quickstart vector-add, install |
| `triton_call_api_reference.md` | triton_call() full API — argument ordering rules, multi-output, strides |
| `working_code_examples.md` | 7 working examples: vector add, matmul, fused dropout, flash attention |
| `installation_and_compatibility.md` | pip/uv install, GPU requirements |

---

## Key patterns for this project

### The operation to fuse (layer 1 forward)

```python
# Current JAX code (slow — materializes huge intermediates in HBM):
base1 = xb @ w1                              # (128, 128)
xB1 = xb @ B1.T                             # (128, 5000)
pert1 = xB1.T[:, :, None] * A1[:, None, :]  # (5000, 128, 128) <-- 160MB in bfloat16
l1_pos = gelu(base1 + sigma * pert1)         # (5000, 128, 128)
l1_neg = gelu(base1 - sigma * pert1)         # (5000, 128, 128)

# What a Triton kernel should do (keep tiles in SRAM, never write pert1 to HBM):
# For each tile of population members [p:p+BLOCK_P]:
#   load xb tile, w1 tile, A1[p:p+BLOCK_P] tile, B1[p:p+BLOCK_P] tile
#   compute base = xb_tile @ w1_tile
#   compute xB = xb_tile @ B1_tile.T
#   compute pert = outer(xB, A1_tile)  (in SRAM)
#   l_pos = gelu(base + sigma * pert)
#   l_neg = gelu(base - sigma * pert)
#   write l_pos, l_neg to output
```

### Calling from JAX via jax-triton

```python
import jax_triton as jt

def fused_layer1(xb, w1, A1, B1, sigma):
    # xb: (batch, in_dim), w1: (in_dim, hidden), A1: (pop, hidden), B1: (pop, in_dim)
    pop, hidden = A1.shape
    batch = xb.shape[0]
    out_shape = [
        jax.ShapeDtypeStruct((pop, batch, hidden), jnp.bfloat16),  # l_pos
        jax.ShapeDtypeStruct((pop, batch, hidden), jnp.bfloat16),  # l_neg
    ]
    grid = lambda meta: (triton.cdiv(pop, meta['BLOCK_P']), triton.cdiv(batch, meta['BLOCK_B']))
    return jt.triton_call(
        xb, w1, A1, B1,          # inputs (pointers passed in this order)
        kernel=fused_layer1_kernel,
        out_shape=out_shape,
        grid=grid,
        sigma=sigma,             # scalar
        POP=pop, BATCH=batch, IN_DIM=xb.shape[1], HIDDEN=hidden,  # constexpr
    )
```

### tl.dot for matrix multiply inside a kernel

```python
# Accumulate a (BLOCK_M, BLOCK_N) tile result:
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k in range(0, K, BLOCK_K):
    a = tl.load(a_ptr + ...)   # (BLOCK_M, BLOCK_K)
    b = tl.load(b_ptr + ...)   # (BLOCK_K, BLOCK_N)
    acc += tl.dot(a, b)
tl.store(c_ptr + ..., acc)
```

---

## Common gotchas

- **Argument order in triton_call**: inputs first, then outputs (via out_shape), then
  scalar args, then constexpr kwargs. Getting this wrong causes silent wrong results.
- **Block sizes must be powers of 2** for tl.dot to work.
- **tl.dot requires BLOCK_K ≥ 16** on most hardware.
- **bfloat16 accumulation**: use `tl.float32` accumulators inside the kernel, cast
  output to bfloat16 before storing.
- **Strides**: for non-contiguous arrays, pass strides explicitly and use
  `tl.load(ptr + row*stride_row + col*stride_col)`.
- **jax-triton wraps the kernel in jax.pure_callback** — it's JIT-compatible and
  works inside jax.jit functions.

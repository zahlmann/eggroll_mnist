# Working Code Examples: Triton Kernels Called from JAX

Sources:
- https://raw.githubusercontent.com/jax-ml/jax-triton/main/tests/triton_call_test.py
- https://raw.githubusercontent.com/jax-ml/jax-triton/main/examples/fused_attention.py
- https://rocm.blogs.amd.com/artificial-intelligence/jax-triton/README.html
- https://jax-ml.github.io/jax-triton/triton_call/

---

## Example 1: Vector Addition (minimal)

The canonical "hello world" of jax-triton. Demonstrates the basic pattern for
replacing a `jax.jit` function with a Triton kernel.

```python
import triton
import triton.language as tl
import jax
import jax.numpy as jnp
import jax_triton as jt


@triton.jit
def add_kernel(x_ptr, y_ptr, n_elements, output_ptr, BLOCK_SIZE: tl.constexpr):
    """Kernel argument order: [input ptrs] -> [output ptrs] -> [scalars] -> [constexprs]"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    BLOCK_SIZE = 8
    return jt.triton_call(
        x,
        y,
        x.size,                                          # scalar arg
        kernel=add_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=lambda meta: (triton.cdiv(x.size, meta["BLOCK_SIZE"]),),
        BLOCK_SIZE=BLOCK_SIZE,                           # constexpr kwarg
    )


# Works both eagerly and inside jit
x_val = jnp.arange(8, dtype=jnp.float32)
y_val = jnp.arange(8, 16, dtype=jnp.float32)
print(add(x_val, y_val))            # eager
print(jax.jit(add)(x_val, y_val))   # jit-compiled
```

**Key points:**
- `n_elements` is a scalar: pass it as a positional arg between arrays and output
- `BLOCK_SIZE` is `tl.constexpr`: pass as a keyword arg to `triton_call`
- `out_shape` takes a `jax.ShapeDtypeStruct` or any object with `.shape` / `.dtype`
- `grid` can be a lambda receiving `meta` dict containing all constexpr kwargs

---

## Example 2: In-Place / Input-Output Aliases

JAX arrays are immutable. For in-place semantics, use `input_output_aliases`.

```python
@triton.jit
def add_inplace_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Writes result back into y_ptr (in-place on y)."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(y_ptr + offsets, output, mask=mask)  # write back to y


from functools import partial

@partial(jax.jit, donate_argnames="y")  # donate y's buffer — avoids copy
def add_inplace_y(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return jt.triton_call(
        x,
        y,
        x.size,
        kernel=add_inplace_kernel,
        input_output_aliases={1: 0},  # input arg index 1 (y) → output index 0
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=lambda meta: (triton.cdiv(x.size, meta["BLOCK_SIZE"]),),
        BLOCK_SIZE=8,
    )

x_val = jnp.arange(8, dtype=jnp.float32)
y_val = jnp.arange(8, 16, dtype=jnp.float32)
result = add_inplace_y(x_val, y_val)  # y is modified in-place (conceptually)
```

**`input_output_aliases` dict:** `{input_arg_index: output_index}`
- Key: index in the positional `*args` list (0-based, arrays only)
- Value: index in the `out_shape` list (0-based)

---

## Example 3: Matrix Multiplication

Full matmul kernel with grouped program ID ordering (better cache behavior).
This is a complete, working example from the official test suite.

```python
import triton
import triton.language as tl
import jax
import jax.numpy as jnp
import jax_triton as jt


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    c_ptr,                              # output pointer comes after all inputs
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    K_EXACTLY_DIVISIBLE_BY_BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_remaining in range(K, 0, -BLOCK_SIZE_K):
        if K_EXACTLY_DIVISIBLE_BY_BLOCK:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            mask = tl.arange(0, BLOCK_SIZE_K) < k_remaining
            a = tl.load(a_ptrs, mask=mask[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask[:, None], other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    m, k = x.shape
    _, n = y.shape

    def grid(meta):
        return (triton.cdiv(m, meta["BLOCK_SIZE_M"]) * triton.cdiv(n, meta["BLOCK_SIZE_N"]),)

    return jt.triton_call(
        x, y,
        m, n, k,
        k, 1,   # strides for a: stride_am, stride_ak
        n, 1,   # strides for b: stride_bk, stride_bn
        n, 1,   # strides for c: stride_cm, stride_cn
        kernel=matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), dtype=x.dtype),
        grid=grid,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,
        K_EXACTLY_DIVISIBLE_BY_BLOCK=(k % 32 == 0),
    )


# Usage
from jax import random
k1, k2 = random.split(random.PRNGKey(0), 2)
x = random.normal(k1, (512, 512), dtype=jnp.float16)
y = random.normal(k2, (512, 512), dtype=jnp.float16)
result = jax.jit(matmul)(x, y)
```

**Note on strides:** For a C-contiguous 2D array of shape `(M, N)`:
- `stride_row = N` (jump N elements to go one row down)
- `stride_col = 1` (jump 1 element to go one column right)

Use `jt.strides_from_shape(x.shape)` to compute strides automatically.

---

## Example 4: Fused Leaky ReLU + Dropout Kernel

Real-world example: replacing two separate JAX operations with a single fused
Triton kernel. This is a memory-bandwidth-bound operation where Triton wins.

Benchmark: naive JAX ~1.89ms → JAX+JIT ~145μs → Triton ~121μs (~400% better bandwidth)

```python
import triton
import triton.language as tl
import jax
import jax.numpy as jnp
import jax_triton as jt


# JAX baseline (what we're replacing):
def leaky_relu_dropout_jax(x, rate=0.5):
    x = jnp.where(x >= 0, x, 0.01 * x)
    keep_prob = 1.0 - rate
    rand_tensor = jax.random.uniform(jax.random.PRNGKey(0), x.shape)
    keep_mask = jnp.where(rand_tensor > rate, 1.0, 0.0)
    return x * keep_mask / keep_prob


# Triton kernel (fused version):
@triton.jit
def leaky_dropout_kernel(
    x_ptr,
    output_ptr,
    rows: tl.constexpr,
    cols: tl.constexpr,
    p: tl.constexpr,
    seed: tl.constexpr,
    block_size: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)

    row_idx = offsets % rows
    col_idx = offsets // rows
    mask = (row_idx < rows) & (col_idx < cols)

    x = tl.load(x_ptr + offsets, mask=mask)

    # Leaky ReLU
    x = tl.where(x >= 0.0, x, 0.01 * x)

    # Dropout
    random = tl.rand(tl.full([], seed, tl.int32), offsets)
    x_keep = random > p
    output = tl.where(x_keep, x / (1 - p), 0.0)

    tl.store(output_ptr + offsets, output, mask=mask)


def triton_leaky_dropout(x: jnp.ndarray, p: float = 0.5, seed: int = 123) -> jnp.ndarray:
    out_shape = jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)
    rows, cols = x.shape
    n_elements = x.size

    # Grid as a lambda receiving meta dict (contains constexpr kwargs)
    grid = lambda meta: (triton.cdiv(n_elements, meta["block_size"]),)

    return jt.triton_call(
        x,
        kernel=leaky_dropout_kernel,
        out_shape=out_shape,
        grid=grid,
        rows=rows,
        cols=cols,
        p=p,
        seed=seed,
        block_size=1024,
    )


# Usage
x = jax.random.normal(jax.random.PRNGKey(42), (256, 256))
result = jax.jit(triton_leaky_dropout)(x, p=0.5, seed=42)
```

**Key insight:** All of `rows`, `cols`, `p`, `seed`, `block_size` are passed as
keyword args to `triton_call` — they flow into the kernel as `constexpr` values.

---

## Example 5: Fused Attention (Flash Attention pattern)

Large example showing multi-output kernels, strides, and complex grid functions.
Source: https://github.com/jax-ml/jax-triton/blob/main/examples/fused_attention.py

```python
import functools
import jax
from jax import random
import jax.numpy as jnp
import jax_triton as jt
import numpy as np
import triton
import triton.language as tl


@triton.jit
def fused_attention_kernel(
    Q, K, V,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    L, M,                           # two output pointers (l, m statistics)
    Out,                             # main output pointer
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_q = off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    off_k = off_hz * stride_qh + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
    off_v = off_hz * stride_qh + offs_n[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    m_prev = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    q = tl.load(q_ptrs)
    for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
        k = tl.load(k_ptrs)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        m_curr = tl.maximum(tl.max(qk, 1), m_prev)
        l_prev *= tl.exp(m_prev - m_curr)
        p = tl.exp(qk - m_curr[:, None])
        l_curr = tl.sum(p, 1) + l_prev
        l_rcp = 1. / l_curr
        p *= l_rcp
        acc *= (l_prev * l_rcp)[:, None]
        p = p.to(tl.float16)
        v = tl.load(v_ptrs)
        acc += tl.dot(p, v)
        l_prev = l_curr
        m_prev = m_curr
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    l_ptrs = L + off_hz * N_CTX + offs_m
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, l_prev)
    tl.store(m_ptrs, m_prev)
    offs_n = tl.arange(0, BLOCK_DMODEL)
    off_o = off_hz * stride_oh + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc)


@functools.partial(jax.jit, static_argnames=["sm_scale"])
def fused_attention(q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    block_size = 128
    grid = (jt.cdiv(q.shape[2], block_size), q.shape[0] * q.shape[1])

    # Three outputs: l statistics, m statistics, attention output
    out_shape = [
        jax.ShapeDtypeStruct(shape=(q.shape[0] * q.shape[1], q.shape[2]), dtype=jnp.float32),
        jax.ShapeDtypeStruct(shape=(q.shape[0] * q.shape[1], q.shape[2]), dtype=jnp.float32),
        jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
    ]

    metaparams = dict(
        BLOCK_M=block_size,
        BLOCK_N=block_size,
        BLOCK_DMODEL=q.shape[-1],
        num_warps=4,
        num_stages=2,
    )

    # Unpack: only take the third output (the actual attention result)
    _, _, output = jt.triton_call(
        q, k, v,
        *jt.strides_from_shape(q.shape),   # strides for Q
        *jt.strides_from_shape(k.shape),   # strides for K
        *jt.strides_from_shape(v.shape),   # strides for V
        *jt.strides_from_shape(q.shape),   # strides for output
        q.shape[0], q.shape[1], q.shape[2],  # Z, H, N_CTX
        kernel=fused_attention_kernel,
        out_shape=out_shape,
        grid=grid,
        **metaparams,
    )
    return output


# Usage
q_key, k_key, v_key = random.split(random.PRNGKey(0), 3)
B, H, S, D = 2, 3, 1024, 128
q = random.normal(q_key, (B, H, S, D), dtype=jnp.float16)
k = random.normal(k_key, (B, H, S, D), dtype=jnp.float16)
v = random.normal(v_key, (B, H, S, D), dtype=jnp.float16)
output = fused_attention(q, k, v)
```

**Key lessons from this example:**
- `jt.strides_from_shape(shape)` returns a tuple of strides for a C-contiguous
  array of that shape — saves you from computing them manually
- `num_warps` and `num_stages` can be passed as keyword args (they're special:
  consumed by the jax-triton layer, not forwarded as constexprs to the kernel)
- Multiple outputs: `out_shape` is a list, `triton_call` returns a tuple

---

## Example 6: Float Scalar Arguments

Passing Python scalars (not arrays) to a kernel:

```python
@triton.jit
def add_scalar_kernel(x_ptr, y, output_ptr):
    """y is a float scalar, not a pointer."""
    tl.store(output_ptr, tl.load(x_ptr) + y)


def add_scalar(x: jnp.ndarray, y: float) -> jnp.ndarray:
    return jt.triton_call(
        x,
        y,          # plain Python float — passed as scalar, not pointer
        kernel=add_scalar_kernel,
        out_shape=jax.ShapeDtypeStruct((), x.dtype),
        grid=1,     # grid can be a plain int
    )


x = jnp.array([1.0])
result = add_scalar(x, 42.0)     # works with float
result = add_scalar(x, np.float32(42.0))  # also works with np.float32
```

---

## Example 7: Grid as a Lambda (Dynamic Grid)

The grid function receives all constexpr kwargs as a dict:

```python
@triton.jit
def kernel(x_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr, TILE: tl.constexpr):
    ...

# Grid lambda receives meta = {"BLOCK_SIZE": 128, "TILE": 32}
jt.triton_call(
    x,
    kernel=kernel,
    out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    grid=lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"] * meta["TILE"]),),
    N=x.size,
    BLOCK_SIZE=128,
    TILE=32,
)
```

---

## Pattern: Replacing a jax.jit Function with Triton

```python
# BEFORE: pure JAX
@jax.jit
def my_op_jax(x: jnp.ndarray) -> jnp.ndarray:
    # some element-wise ops...
    return jnp.where(x > 0, x, 0.01 * x) * 2.0


# AFTER: Triton kernel
@triton.jit
def my_op_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # fused ops in a single pass — one memory read, one write
    out = tl.where(x > 0.0, x, 0.01 * x) * 2.0
    tl.store(output_ptr + offsets, out, mask=mask)


@jax.jit
def my_op_triton(x: jnp.ndarray) -> jnp.ndarray:
    BLOCK_SIZE = 1024
    return jt.triton_call(
        x,
        x.size,
        kernel=my_op_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=lambda meta: (triton.cdiv(x.size, meta["BLOCK_SIZE"]),),
        BLOCK_SIZE=BLOCK_SIZE,
    )
```

**When Triton wins over `jax.jit`:**
- Memory-bound ops (dropout, normalization, activation functions)
- When you want to fuse operations that XLA doesn't automatically fuse
- When you need fine-grained control over tiling, pipelining, or shared memory

**When `jax.jit` is usually fine:**
- Compute-bound ops (large matmuls — XLA/cuBLAS is hard to beat)
- When development speed matters more than micro-optimization

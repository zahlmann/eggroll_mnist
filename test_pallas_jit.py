"""Compare JIT overhead: jax-triton vs Pallas for a simple kernel inside a scan."""
import time
import jax
import jax.numpy as jnp
import triton
import triton.language as tl
import jax_triton as jt
from jax.experimental import pallas as pl

N = 5000
D = 128
BATCH = 128

# ---- Simple Triton kernel: element-wise multiply + reduce ----
@triton.jit
def _triton_mul_reduce(x_ptr, y_ptr, out_ptr, N: tl.constexpr, D: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, D)
    x = tl.load(x_ptr + pid * D + offs)
    y = tl.load(y_ptr + pid * D + offs)
    result = tl.sum(x * y)
    tl.store(out_ptr + pid, result)

def triton_dot(x, y):
    return jt.triton_call(
        x, y,
        kernel=_triton_mul_reduce,
        out_shape=jax.ShapeDtypeStruct((x.shape[0],), jnp.float32),
        grid=(x.shape[0],),
        N=x.shape[0], D=x.shape[1],
        num_warps=4, num_stages=1,
    )

# ---- Pallas equivalent ----
def pallas_dot(x, y):
    def kernel_fn(x_ref, y_ref, out_ref):
        out_ref[0] = jnp.sum(x_ref[:] * y_ref[:])

    return pl.pallas_call(
        kernel_fn,
        out_shape=jax.ShapeDtypeStruct((x.shape[0],), jnp.float32),
        grid=(x.shape[0],),
        in_specs=[
            pl.BlockSpec(block_shape=(D,), index_map=lambda i: (i * D,)),
            pl.BlockSpec(block_shape=(D,), index_map=lambda i: (i * D,)),
        ],
        out_specs=pl.BlockSpec(block_shape=(1,), index_map=lambda i: (i,)),
    )(x.reshape(-1), y.reshape(-1))

# ---- Test with scan (simulating the training loop structure) ----
@jax.jit
def with_triton_scan(x, y):
    def body(carry, _):
        result = triton_dot(x, y)
        return carry + result.sum(), None
    total, _ = jax.lax.scan(body, jnp.float32(0), None, length=10)
    return total

@jax.jit
def with_pallas_scan(x, y):
    def body(carry, _):
        result = pallas_dot(x, y)
        return carry + result.sum(), None
    total, _ = jax.lax.scan(body, jnp.float32(0), None, length=10)
    return total

@jax.jit
def with_jax_scan(x, y):
    def body(carry, _):
        result = jnp.sum(x * y, axis=1)
        return carry + result.sum(), None
    total, _ = jax.lax.scan(body, jnp.float32(0), None, length=10)
    return total

x = jax.random.normal(jax.random.PRNGKey(0), (N, D))
y = jax.random.normal(jax.random.PRNGKey(1), (N, D))

for name, fn in [("triton", with_triton_scan), ("pallas", with_pallas_scan), ("pure_jax", with_jax_scan)]:
    try:
        t0 = time.perf_counter()
        r = fn(x, y)
        jax.block_until_ready(r)
        t_first = time.perf_counter() - t0

        t0 = time.perf_counter()
        r = fn(x, y)
        jax.block_until_ready(r)
        t_second = time.perf_counter() - t0

        jit = t_first - t_second
        print(f"{name:10s}: first={t_first:.3f}s  exec={t_second:.4f}s  JIT={jit:.3f}s")
    except Exception as e:
        print(f"{name:10s}: FAILED - {e}")

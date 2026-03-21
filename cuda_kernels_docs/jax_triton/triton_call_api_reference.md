# triton_call API Reference

Sources:
- https://jax-ml.github.io/jax-triton/triton_call/
- https://raw.githubusercontent.com/jax-ml/jax-triton/main/jax_triton/triton_lib.py (source code)

---

## Function Signature

```python
jax_triton.triton_call(
    *args: jax.Array | bool | int | float | np.float32,
    kernel: triton.JITFunction | triton.runtime.Autotuner | ...,
    out_shape: ShapeDtype | Sequence[ShapeDtype],
    grid: int | tuple[int, ...] | Callable[[dict], tuple[int, ...]],
    name: str = "",
    num_warps: int | None = None,
    num_stages: int | None = None,
    num_ctas: int = 1,
    compute_capability: int | None = None,
    enable_fp_fusion: bool = True,
    input_output_aliases: dict[int, int] | None = None,
    zeroed_outputs: Sequence[int] | Callable[[dict], Sequence[int]] = (),
    debug: bool = False,
    serialized_metadata: bytes = b"",
    **metaparams: Any,
) -> Any
```

## Parameters

### Positional: `*args`
Inputs for the Triton kernel. Can be:
- `jax.Array` — passed as pointers to the kernel
- `bool`, `int`, `float`, `np.float32` — passed as scalar arguments

**Argument ordering in the kernel**: The kernel receives pointers for all array
`args` first, then output pointers (one per entry in `out_shape`), then scalar
args, then `constexpr` metaparams.

### `kernel`
A Triton kernel decorated with `@triton.jit`. Also accepts:
- `triton.runtime.Autotuner`
- `triton.runtime.Heuristics`
- `triton.experimental.gluon._runtime.GluonJITFunction`

All static (non-JAX-traced) values inside the kernel must be annotated with
`triton.language.constexpr`.

### `out_shape`
Specifies the shape and dtype of each output buffer. Can be:
- A single `jax.ShapeDtypeStruct(shape=..., dtype=...)`
- Any object with `.shape` and `.dtype` attributes (e.g., a JAX array)
- A sequence of the above for multiple outputs

Output pointers are appended to the kernel's argument list after the input
array pointers, in the order they appear in `out_shape`.

### `grid`
Controls the number of parallel kernel invocations (program instances). Can be:
- An `int`: launches `grid` program instances
- A tuple of up to 3 ints: launches `prod(grid)` instances arranged in up to 3D
- A callable `f(metaparams) -> tuple`: called with `**metaparams` at compile time

### `input_output_aliases`
A `dict` mapping input argument index → output index. Allows buffer reuse
(in-place operations). Example: `{1: 0}` means input arg at position 1 is
aliased to the first output.

Combined with `jax.jit(donate_argnames=...)`, this avoids unnecessary copies
since JAX arrays are normally immutable.

### `zeroed_outputs`
Sequence of output indices (or a callable returning such) for outputs that
should be zero-initialized before the kernel launches. Also works for
aliased (input-output) arguments.

### `num_warps`
Number of warps per thread block. Defaults to Triton's automatic selection.
Typical values: 4, 8. Affects occupancy and register usage.

### `num_stages`
Number of pipeline stages emitted by the Triton compiler. Controls software
pipelining depth for memory latency hiding. Typical values: 1–4.

### `num_ctas`
Thread blocks per cluster. Only relevant on GPUs with compute capability >= 9.0
(Hopper/H100). Must be <= 8.

### `debug`
If `True`, prints intermediate IRs (Triton IR, LLVM IR, PTX) for debugging.

### `**metaparams`
Keyword arguments that are:
1. Passed to a callable `grid` function
2. Passed to the kernel as `constexpr` arguments

These are compile-time constants — they trigger recompilation if they change.

## Return Value
Returns the kernel output(s). If `out_shape` is a single struct, returns a
single array. If `out_shape` is a list, returns a tuple of arrays.

---

## Argument Order in the Kernel

This is the most important thing to get right. Given:

```python
jt.triton_call(
    x,          # input array #0  -> kernel arg 0: x_ptr
    y,          # input array #1  -> kernel arg 1: y_ptr
    x.size,     # scalar          -> kernel arg 3: n_elements (after outputs)
    kernel=my_kernel,
    out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),  # -> kernel arg 2: output_ptr
    grid=...,
    BLOCK_SIZE=128,  # constexpr  -> kernel arg 4: BLOCK_SIZE
)
```

The kernel signature must be:
```python
@triton.jit
def my_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    ...
```

Order: [input array pointers] → [output pointers] → [scalar args] → [constexpr args]

---

## Multiple Outputs

```python
out_shape = [
    jax.ShapeDtypeStruct(shape=(m,), dtype=jnp.float32),  # output 0
    jax.ShapeDtypeStruct(shape=(n,), dtype=jnp.float32),  # output 1
]
out0, out1 = jt.triton_call(x, kernel=my_kernel, out_shape=out_shape, grid=...)
```

The kernel receives `out0_ptr` then `out1_ptr` after all input array pointers.

---

## Helper Utilities

```python
import jax_triton as jt

# Compute strides for a shape (useful for multi-dim arrays)
strides = jt.strides_from_shape(q.shape)  # returns tuple of strides

# Ceiling division (same as triton.cdiv)
jt.cdiv(n, block_size)

# Get GPU compute capability
cc = jt.get_compute_capability(device_id)
```

---

## JAX Type to Triton Type Mapping

| JAX dtype     | Triton type |
|---------------|-------------|
| float32       | fp32        |
| float16       | fp16        |
| bfloat16      | bf16        |
| float64       | fp64        |
| int32         | i32         |
| int64         | i64         |
| uint32        | u32         |
| uint64        | u64         |
| int8          | i8          |
| int16         | i16         |
| bool          | i1          |
| float8_e4m3fn | fp8e4nv     |
| float8_e5m2   | fp8e5       |

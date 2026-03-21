# jax-triton Installation and Compatibility

Sources:
- https://jax-ml.github.io/jax-triton/
- https://github.com/jax-ml/jax-triton
- https://pypi.org/project/jax-triton/

---

## Installation

```bash
# Install jax-triton
pip install jax-triton

# Also need CUDA-compatible JAX:
pip install "jax[cuda12]"

# Or for ROCm/AMD:
pip install "jax[rocm]"

# Bleeding-edge (from source):
pip install 'jax-triton @ git+https://github.com/jax-ml/jax-triton.git'
```

With uv:
```bash
uv add jax-triton "jax[cuda12]"
# or
uv run --with jax-triton --with "jax[cuda12]" python your_script.py
```

## Requirements

- Python 3.10+
- CUDA GPU (or ROCm AMD GPU)
- `jax` with GPU support (jax[cuda12] or jax[rocm])
- `triton` (installed automatically as a dependency of jax-triton)

## Version

Latest: v0.3.1 (February 2026)

## Limitations

- **GPU only** — jax-triton does not work on CPU or TPU
- For TPU, use JAX's built-in Pallas (which uses Mosaic backend instead of Triton)
- Triton kernels are recompiled when any constexpr argument changes value

## Pallas (Alternative for TPU + GPU)

JAX now ships with **Pallas**, a built-in kernel language that:
- Uses Triton as backend on GPU
- Uses Mosaic as backend on TPU

For new projects targeting both GPU and TPU, consider Pallas instead:
- Docs: https://docs.jax.dev/en/latest/pallas/quickstart.html
- Pallas is JAX-native and doesn't require a separate package

jax-triton is better when you:
- Already have Triton kernels you want to reuse
- Need Triton-specific features (e.g., advanced tiling, PTX intrinsics)
- Want to use autotuner/heuristics from the Triton ecosystem

## AMD GPU Support

AMD GPU users should use the ROCm fork: https://github.com/ROCm/jax-triton

Docker image with everything pre-installed:
```bash
docker pull rocm/jax-build:jaxlib-0.4.24-rocm600-py3.10.0.rocm-jax-triton
```

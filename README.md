# EGGROLL MNIST

This repo trains a 784 → 128 → 128 → 10 MLP on MNIST with evolution strategies and compares it with backpropagation. The original JAX implementation is in `mnist_eggroll.py`. It uses rank-one weight perturbations from the [EGGROLL paper](https://arxiv.org/abs/2511.16652).

The optimized trainer evaluates 1,680 pairs of opposite perturbations per batch. It shares the first-layer matrix multiplication across the population and uses a fused Triton kernel to compute the label-smoothed cross-entropy for each perturbation. Both optimized trainers use batches of 128 and train for 10 epochs.

## Run

Use `uv` from the repository root. Optimized EGGROLL requires an NVIDIA GPU with FP8 E4M3 tensor-core support. The prepared MNIST data is included.

Run optimized EGGROLL, optimized backprop, and naive backprop in sequence with the same XLA GEMM autotuner setting. Each script prints its training time, test accuracy, and GPU memory measurement.

```bash
XLA_FLAGS=--xla_gpu_enable_triton_gemm=false bash run_experiments.sh
```

To run optimized EGGROLL on its own or check its accuracy and memory across three seeds, use either command below.

```bash
uv run mnist_eggroll_optimized.py --seed 11
uv run validate.py
```

The validator uses seeds 11, 42, and 7. It checks the printed training constants, requires at least 97.2% mean test accuracy, and caps reported peak GPU memory at 500 MiB on each run. Training time is recorded separately.

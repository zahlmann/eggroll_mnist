import os

# Disable XLA's Triton GEMM autotuner — cuBLAS fallback is identical but skips ~0.6s
# of kernel autotuning during JIT. Must be set before importing JAX.
os.environ.setdefault("XLA_FLAGS", "")
if "--xla_gpu_enable_triton_gemm" not in os.environ["XLA_FLAGS"]:
    os.environ["XLA_FLAGS"] += " --xla_gpu_enable_triton_gemm=false"

import time
import threading
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from kernels.fused_3layer_ce import fused_3layer_ce_both

# ---- Architecture (locked — validate.py checks these) ----
HALF_POPULATION = 1680  # 21 full CUDA waves, zero tail
HIDDEN_DIM = 128
BATCH_SIZE = 128
EPOCHS = 10
T = 2.0

# ---- Hyperparameters ----
LR_START = 0.012
LR_DECAY = 0.88
SIGMA_START = 0.036
SIGMA_DECAY = 0.998
ALPHA_START = 0.30  # label smoothing, decays per epoch
ALPHA_DECAY = 0.50
N_SUBGROUPS = 8
CLIP_RANGE = 2.0
L3_LR_BOOST = 2.0

# ---- Derived constants ----
data = np.load("mnist_prepped_float.npz")
X_train_np, y_train_np = data["X_train"], data["y_train"]
X_test, y_test = jnp.array(data["X_test"]), jnp.array(data["y_test"])

assert X_train_np.shape == (60000, 784)
assert y_train_np.shape == (60000,)

N_BATCHES = X_train_np.shape[0] // BATCH_SIZE
assert HALF_POPULATION % N_SUBGROUPS == 0
# B1(784) + A1(128) + B2(128) + A2(128) + B3(128) + A3(10)
VEC_DIM = 784 + HIDDEN_DIM * 4 + 10


def fast_gelu(x):
    return x * jax.nn.sigmoid(1.702 * x)


def get_gpu_memory_mb():
    stats = jax.devices('gpu')[0].memory_stats()
    assert stats is not None
    return stats['peak_bytes_in_use'] / (1024 * 1024)


@jax.jit
def train_all_epochs(w1, w2, w3, X_batched, y_batched, key):
    def epoch_step(carry, _):
        w1, w2, w3, key, sigma, lr, alpha = carry
        key, epoch_key = jax.random.split(key)
        scale = 1.0 / (2.0 * sigma * HALF_POPULATION)

        def batch_step(carry, batch_data):
            w1, w2, w3, batch_idx = carry
            xb, yb = batch_data

            # Sample perturbation vectors: one (HALF_POP, 1306) matrix, sliced per layer
            vecs = jax.random.normal(jax.random.fold_in(epoch_key, batch_idx),
                                     (HALF_POPULATION, VEC_DIM), dtype=jnp.float32)
            vecs_bf16 = vecs.astype(jnp.bfloat16)

            # bf16 slices for kernel (tensor core inputs)
            B1_f, A1_f = vecs_bf16[:, :784], vecs_bf16[:, 784:912]
            B2_f, A2_f = vecs_bf16[:, 912:1040], vecs_bf16[:, 1040:1168]
            B3_f, A3_f = vecs_bf16[:, 1168:1296], vecs_bf16[:, 1296:]

            # fp32 slices for gradient computation (precision rules)
            B1, A1 = vecs[:, :784], vecs[:, 784:912]
            B2, A2 = vecs[:, 912:1040], vecs[:, 1040:1168]
            B3, A3 = vecs[:, 1168:1296], vecs[:, 1296:]

            # Precompute base forward pass and xB1 projection
            xb_f = xb.astype(jnp.bfloat16)
            w1_f, w2_f, w3_f = w1.astype(jnp.bfloat16), w2.astype(jnp.bfloat16), w3.astype(jnp.bfloat16)
            base1 = xb_f @ w1_f       # (batch, hidden)
            xB1_T = B1_f @ xb_f.T     # (pop, batch)

            # Fused 3-layer forward + CE for both +sigma and -sigma perturbations
            ce_pos, ce_neg = fused_3layer_ce_both(
                base1, xB1_T, A1_f, w2_f, B2_f, A2_f, w3_f, B3_f, A3_f,
                jnp.float32(sigma), jnp.float32(T), jnp.float32(alpha), yb)

            # Per-subgroup Winsorized z-score of fitness differences
            fitness_diff = ce_neg.sum(axis=1) - ce_pos.sum(axis=1)
            groups = fitness_diff.reshape(N_SUBGROUPS, HALF_POPULATION // N_SUBGROUPS)
            means = groups.mean(axis=1, keepdims=True)
            stds = groups.std(axis=1, keepdims=True) + 1e-8
            shaped = jnp.clip((groups - means) / stds, -CLIP_RANGE, CLIP_RANGE).reshape(HALF_POPULATION)

            # ES gradient: g_W = (1/2σN) Σ fitness_i * outer(B_i, A_i)
            s = shaped[:, None]
            grad1 = scale * B1.T @ (s * A1)
            grad2 = scale * B2.T @ (s * A2)
            grad3 = scale * B3.T @ (s * A3)
            w1 = w1 + lr * grad1
            w2 = w2 + lr * grad2
            w3 = w3 + L3_LR_BOOST * lr * grad3

            return (w1, w2, w3, batch_idx + 1), None

        (w1, w2, w3, _), _ = jax.lax.scan(
            batch_step, (w1, w2, w3, jnp.int32(0)), (X_batched, y_batched))
        return (w1, w2, w3, key, sigma * SIGMA_DECAY, lr * LR_DECAY, alpha * ALPHA_DECAY), None

    init = (w1, w2, w3, key, jnp.float32(SIGMA_START), jnp.float32(LR_START), jnp.float32(ALPHA_START))
    (w1, w2, w3, *_), _ = jax.lax.scan(epoch_step, init, None, length=EPOCHS)
    return w1, w2, w3


@jax.jit
def evaluate_batch(w1, w2, w3, xb, yb):
    logits = fast_gelu(fast_gelu(xb @ w1) @ w2) @ w3
    return jnp.mean(jnp.argmax(logits, axis=1) == yb)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    # validate.py parses this block
    print("=== CONSTANTS ===")
    print(f"HIDDEN_DIM: {HIDDEN_DIM}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"EPOCHS: {EPOCHS}")
    print(f"HALF_POPULATION: {HALF_POPULATION}")
    print(f"T: {T}")
    print(f"SEED: {args.seed}")
    print("=================")

    key = jax.random.PRNGKey(args.seed)
    key, k1, k2, k3 = jax.random.split(key, 4)
    init = jax.nn.initializers.orthogonal()
    w1 = init(k1, (784, HIDDEN_DIM), jnp.float32)
    w2 = init(k2, (HIDDEN_DIM, HIDDEN_DIM), jnp.float32)
    w3 = init(k3, (HIDDEN_DIM, 10), jnp.float32)

    print("Training...")
    start_time = time.perf_counter()

    # Shuffle + GPU transfer in background thread (overlaps with JIT, both release GIL)
    data_ready = {}
    def _prepare_data():
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(X_train_np.shape[0])[:N_BATCHES * BATCH_SIZE]
        data_ready['X'] = jnp.array(X_train_np[perm].reshape(N_BATCHES, BATCH_SIZE, -1))
        data_ready['y'] = jnp.array(y_train_np[perm].reshape(N_BATCHES, BATCH_SIZE))
    data_thread = threading.Thread(target=_prepare_data)
    data_thread.start()

    # JIT compile while data prepares (abstract shapes, no GPU memory needed)
    compiled = jax.jit(train_all_epochs).lower(
        w1, w2, w3,
        jax.ShapeDtypeStruct((N_BATCHES, BATCH_SIZE, 784), jnp.float32),
        jax.ShapeDtypeStruct((N_BATCHES, BATCH_SIZE), jnp.int32),
        key,
    ).compile()

    data_thread.join()
    w1, w2, w3 = compiled(w1, w2, w3, data_ready['X'], data_ready['y'], key)
    jax.block_until_ready(w1)
    train_time = time.perf_counter() - start_time

    peak_memory = get_gpu_memory_mb()

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1:2d} | LR: {LR_START * LR_DECAY**epoch:.4f} | "
              f"Sigma: {SIGMA_START * SIGMA_DECAY**epoch:.4f} | Time: {train_time/EPOCHS:.1f}s")

    # Evaluate
    correct = sum(float(evaluate_batch(w1, w2, w3, X_test[i:i+256], y_test[i:i+256])) * min(256, len(X_test) - i)
                  for i in range(0, len(X_test), 256))
    test_acc = correct / len(X_test)

    print(f"\nTest Accuracy: {test_acc:.2%} ({int(test_acc * len(X_test))}/{len(X_test)})")
    print(f"Training Time: {train_time:.2f}s")
    print(f"Peak GPU Memory: {peak_memory:.1f} MB")

    # Machine-parseable — validate.py and benchmark.py grep this
    print("=== RESULTS ===")
    print(f"test_accuracy: {test_acc:.6f}")
    print(f"training_time_s: {train_time:.2f}")
    print(f"peak_memory_mb: {peak_memory:.1f}")
    print("===============")


if __name__ == "__main__":
    main()

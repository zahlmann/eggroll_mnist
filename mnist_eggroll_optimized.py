import os
import sys
import time
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from kernels.fused_3layer_ce import fused_3layer_ce


def fast_gelu(x):
    """Sigmoid-based GELU approximation (GPT-2 style). Fewer FLOPs, simpler for XLA to fuse."""
    return x * jax.nn.sigmoid(1.702 * x)

def get_gpu_memory_mb():
    """Get peak GPU memory usage in MB."""
    try:
        devices = jax.devices('gpu')
        if devices:
            jax.block_until_ready(jnp.zeros(1))
            stats = devices[0].memory_stats()
            if stats:
                return stats.get('peak_bytes_in_use', stats.get('bytes_in_use', 0)) / (1024 * 1024)
    except:
        pass
    return 0.0


# Load data
if not os.path.exists("mnist_prepped_float.npz"):
    print("Error: mnist_prepped_float.npz not found.")
    exit(1)

data = np.load("mnist_prepped_float.npz")
X_train = jnp.array(data["X_train"])
y_train = jnp.array(data["y_train"])
X_test = jnp.array(data["X_test"])
y_test = jnp.array(data["y_test"])

# ---- LOCKED CONSTANTS (validate.py checks these — do not change values) ----
HALF_POPULATION = 5000
HIDDEN_DIM = 128
BATCH_SIZE = 128
EPOCHS = 10
T = 2.0  # temperature for CE fitness (T>1 softens logits → smoother ES gradients)

# ---- Tunable hyperparameters (agent may adjust these) ----
LR_START = 0.015
LR_DECAY = 0.92
SIGMA_START = 0.028
SIGMA_DECAY = 0.998

N_BATCHES = (X_train.shape[0] // BATCH_SIZE)  # drop last incomplete batch
GROUP_SIZE = 2  # process this many batches per ES gradient step
N_GROUPS = N_BATCHES // GROUP_SIZE  # 468 // 2 = 234 groups
SMALL_VEC_DIM = HIDDEN_DIM * 4 + 10  # A1+B2+A2+B3+A3 = 522 (without B1)


@partial(jax.jit, donate_argnums=(3, 4))
def train_epoch(w1, w2, w3, X_grouped, y_grouped, B1, B1_f, sigma, lr, key):
    """Train one epoch with single-level scan (faster compilation than nested scan)."""

    pos_sign = jnp.float32(1.0)
    neg_sign = jnp.float32(-1.0)

    key, epoch_rng_key = jax.random.split(key)
    sigma_f32 = jnp.float32(sigma)
    T_f32 = jnp.float32(T)
    scale = jnp.float32(1.0) / (jnp.float32(2.0) * sigma_f32 * jnp.float32(HALF_POPULATION))

    def batch_step(carry, batch_data):
        w1, w2, w3, batch_idx = carry
        xb, yb = batch_data

        # Generate only smaller vectors per batch (522 elements instead of 1306)
        batch_key = jax.random.fold_in(epoch_rng_key, batch_idx)
        small_vecs = jax.random.normal(batch_key, (HALF_POPULATION, SMALL_VEC_DIM), dtype=jnp.float32)
        small_vecs_f = small_vecs.astype(jnp.bfloat16)

        A1_f = small_vecs_f[:, :HIDDEN_DIM]
        B2_f = small_vecs_f[:, HIDDEN_DIM:2*HIDDEN_DIM]
        A2_f = small_vecs_f[:, 2*HIDDEN_DIM:3*HIDDEN_DIM]
        B3_f = small_vecs_f[:, 3*HIDDEN_DIM:4*HIDDEN_DIM]
        A3_f = small_vecs_f[:, 4*HIDDEN_DIM:]

        A1 = small_vecs[:, :HIDDEN_DIM]
        B2 = small_vecs[:, HIDDEN_DIM:2*HIDDEN_DIM]
        A2 = small_vecs[:, 2*HIDDEN_DIM:3*HIDDEN_DIM]
        B3 = small_vecs[:, 3*HIDDEN_DIM:4*HIDDEN_DIM]
        A3 = small_vecs[:, 4*HIDDEN_DIM:]

        xb_f = xb.astype(jnp.bfloat16)
        w1_f = w1.astype(jnp.bfloat16)
        w2_f = w2.astype(jnp.bfloat16)
        w3_f = w3.astype(jnp.bfloat16)

        base1 = xb_f @ w1_f
        xB1_T = B1_f @ xb_f.T  # uses epoch-level B1_f

        partial_ce_pos = fused_3layer_ce(
            base1, xB1_T, A1_f, w2_f, B2_f, A2_f, w3_f, B3_f, A3_f,
            sigma_f32, T_f32, pos_sign, yb)
        partial_ce_neg = fused_3layer_ce(
            base1, xB1_T, A1_f, w2_f, B2_f, A2_f, w3_f, B3_f, A3_f,
            sigma_f32, T_f32, neg_sign, yb)

        fitness_diff = partial_ce_neg.sum(axis=1) - partial_ce_pos.sum(axis=1)
        mean = fitness_diff.mean()
        std = fitness_diff.std() + 1e-8
        shaped = (fitness_diff - mean) / std

        shaped_col = shaped[:, None]
        grad1 = scale * B1.T @ (shaped_col * A1)  # uses epoch-level B1
        grad2 = scale * B2.T @ (shaped_col * A2)
        grad3 = scale * B3.T @ (shaped_col * A3)

        w1 = w1 + lr * grad1
        w2 = w2 + lr * grad2
        w3 = w3 + lr * grad3

        return (w1, w2, w3, batch_idx + 1), None

    (w1, w2, w3, _), _ = jax.lax.scan(batch_step, (w1, w2, w3, jnp.int32(0)), (X_grouped, y_grouped))
    return w1, w2, w3, key


@jax.jit
def evaluate_batch(w1, w2, w3, xb, yb):
    l1 = fast_gelu(xb @ w1)
    l2 = fast_gelu(l1 @ w2)
    logits = l2 @ w3
    preds = jnp.argmax(logits, axis=1)
    return jnp.mean(preds == yb)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

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
    initializer = jax.nn.initializers.orthogonal()
    w1 = initializer(k1, (784, HIDDEN_DIM), jnp.float32)
    w2 = initializer(k2, (HIDDEN_DIM, HIDDEN_DIM), jnp.float32)
    w3 = initializer(k3, (HIDDEN_DIM, 10), jnp.float32)

    print("Training...")
    start_time = time.perf_counter()

    lr = LR_START
    sigma = SIGMA_START

    for epoch in range(EPOCHS):
        epoch_start = time.perf_counter()

        # Generate B1 per epoch (outside JIT — fast, avoids adding to scan body)
        key, b1_key, data_key = jax.random.split(key, 3)
        B1 = jax.random.normal(b1_key, (HALF_POPULATION, 784), dtype=jnp.float32)
        B1_f = B1.astype(jnp.bfloat16)

        # Shuffle and group data
        n_samples = N_GROUPS * GROUP_SIZE * BATCH_SIZE
        perm = jax.random.permutation(data_key, X_train.shape[0])
        X_grouped = X_train[perm][:n_samples].reshape(N_GROUPS, GROUP_SIZE * BATCH_SIZE, -1)
        y_grouped = y_train[perm][:n_samples].reshape(N_GROUPS, GROUP_SIZE * BATCH_SIZE)

        w1, w2, w3, key = train_epoch(w1, w2, w3, X_grouped, y_grouped, B1, B1_f, sigma, lr, key)
        jax.block_until_ready(w1)

        epoch_time = time.perf_counter() - epoch_start
        print(f"Epoch {epoch+1:2d} | LR: {lr:.4f} | Sigma: {sigma:.4f} | Time: {epoch_time:.1f}s")

        lr *= LR_DECAY
        sigma *= SIGMA_DECAY

    train_time = time.perf_counter() - start_time
    peak_memory = get_gpu_memory_mb()

    # Evaluate
    print("\nEvaluating on test set...")
    correct = 0
    total = 0
    for i in range(0, X_test.shape[0], 256):
        xb = X_test[i:i+256]
        yb = y_test[i:i+256]
        acc = evaluate_batch(w1, w2, w3, xb, yb)
        correct += float(acc) * len(yb)
        total += len(yb)

    test_acc = correct / total

    print()
    print(f"Test Accuracy: {test_acc:.2%} ({int(test_acc * total)}/{total})")
    print(f"Training Time: {train_time:.2f}s")
    print(f"Peak GPU Memory: {peak_memory:.1f} MB")

    print("=== RESULTS ===")
    print(f"test_accuracy: {test_acc:.6f}")
    print(f"training_time_s: {train_time:.2f}")
    print(f"peak_memory_mb: {peak_memory:.1f}")
    print("===============")


if __name__ == "__main__":
    main()

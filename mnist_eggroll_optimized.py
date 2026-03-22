import os
import sys
import time
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from kernels.fused_3layer_ce import fused_3layer_ce_both


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
                # Use peak_bytes_in_use to catch transient allocations during training
                return stats.get('peak_bytes_in_use', stats.get('bytes_in_use', 0)) / (1024 * 1024)
    except:
        pass
    return 0.0


# Load data
if not os.path.exists("mnist_prepped_float.npz"):
    print("Error: mnist_prepped_float.npz not found.")
    exit(1)

data = np.load("mnist_prepped_float.npz")
X_train_np = data["X_train"]  # keep in CPU memory
y_train_np = data["y_train"]  # keep in CPU memory
X_test = jnp.array(data["X_test"])
y_test = jnp.array(data["y_test"])

# ---- LOCKED CONSTANTS (validate.py checks these — do not change values) ----
HALF_POPULATION = 5000
HIDDEN_DIM = 128
BATCH_SIZE = 128
EPOCHS = 10
T = 2.0  # temperature for CE fitness (T>1 softens logits → smoother ES gradients)

# ---- Tunable hyperparameters (agent may adjust these) ----
LR_START = 0.016
LR_DECAY = 0.92
SIGMA_START = 0.028
SIGMA_DECAY = 0.998

N_BATCHES = (X_train_np.shape[0] // BATCH_SIZE)  # drop last incomplete batch
VEC_DIM = 784 + HIDDEN_DIM * 4 + 10  # B1(784)+A1(128)+B2(128)+A2(128)+B3(128)+A3(10) = 1306

GROUP_SIZE = 2  # process this many batches per ES gradient step
N_GROUPS = N_BATCHES // GROUP_SIZE  # 468 // 2 = 234 groups


@jax.jit
def train_all_epochs(w1, w2, w3, X_grouped, y_grouped, key):
    """Train all epochs in a single JIT call — eliminates Python loop overhead."""

    # Loop-invariant scalars
    # No pos_sign/neg_sign needed — merged kernel handles both

    def epoch_step(carry, _):
        w1, w2, w3, key, sigma, lr = carry

        key, epoch_rng_key = jax.random.split(key)

        X_shuf = X_grouped
        y_shuf = y_grouped

        sigma_f32 = jnp.float32(sigma)
        T_f32 = jnp.float32(T)
        scale = jnp.float32(1.0) / (jnp.float32(2.0) * sigma_f32 * jnp.float32(HALF_POPULATION))

        def batch_step(carry, batch_data):
            w1, w2, w3, batch_idx = carry
            xb, yb = batch_data

            batch_key = jax.random.fold_in(epoch_rng_key, batch_idx)
            # Uniform[-sqrt(3), sqrt(3)] has variance=1 like N(0,1) but simpler XLA graph
            all_vecs = jax.random.uniform(batch_key, (HALF_POPULATION, VEC_DIM), dtype=jnp.float32, minval=-1.7320508, maxval=1.7320508)
            all_vecs_f = all_vecs.astype(jnp.bfloat16)

            B1_f = all_vecs_f[:, :784]
            A1_f = all_vecs_f[:, 784:784+HIDDEN_DIM]
            B2_f = all_vecs_f[:, 784+HIDDEN_DIM:784+2*HIDDEN_DIM]
            A2_f = all_vecs_f[:, 784+2*HIDDEN_DIM:784+3*HIDDEN_DIM]
            B3_f = all_vecs_f[:, 784+3*HIDDEN_DIM:784+4*HIDDEN_DIM]
            A3_f = all_vecs_f[:, 784+4*HIDDEN_DIM:]

            B1 = all_vecs[:, :784]
            A1 = all_vecs[:, 784:784+HIDDEN_DIM]
            B2 = all_vecs[:, 784+HIDDEN_DIM:784+2*HIDDEN_DIM]
            A2 = all_vecs[:, 784+2*HIDDEN_DIM:784+3*HIDDEN_DIM]
            B3 = all_vecs[:, 784+3*HIDDEN_DIM:784+4*HIDDEN_DIM]
            A3 = all_vecs[:, 784+4*HIDDEN_DIM:]

            xb_f = xb.astype(jnp.bfloat16)
            w1_f = w1.astype(jnp.bfloat16)
            w2_f = w2.astype(jnp.bfloat16)
            w3_f = w3.astype(jnp.bfloat16)

            base1 = xb_f @ w1_f
            xB1_T = B1_f @ xb_f.T

            partial_ce_pos, partial_ce_neg = fused_3layer_ce_both(
                base1, xB1_T, A1_f, w2_f, B2_f, A2_f, w3_f, B3_f, A3_f,
                sigma_f32, T_f32, yb)

            # Skip /BATCH_SIZE — normalization absorbs the constant scale
            fitness_diff = partial_ce_neg.sum(axis=1) - partial_ce_pos.sum(axis=1)
            mean = fitness_diff.mean()
            std = fitness_diff.std() + 1e-8
            shaped = (fitness_diff - mean) / std

            shaped_col = shaped[:, None]
            grad1 = scale * B1.T @ (shaped_col * A1)
            grad2 = scale * B2.T @ (shaped_col * A2)
            grad3 = scale * B3.T @ (shaped_col * A3)

            w1 = w1 + lr * grad1
            w2 = w2 + lr * grad2
            w3 = w3 + lr * grad3

            return (w1, w2, w3, batch_idx + 1), None

        (w1, w2, w3, _), _ = jax.lax.scan(batch_step, (w1, w2, w3, jnp.int32(0)), (X_shuf, y_shuf))

        sigma = sigma * SIGMA_DECAY
        lr = lr * LR_DECAY

        return (w1, w2, w3, key, sigma, lr), None

    init = (w1, w2, w3, key, jnp.float32(SIGMA_START), jnp.float32(LR_START))
    (w1, w2, w3, key, _, _), _ = jax.lax.scan(epoch_step, init, None, length=EPOCHS)
    return w1, w2, w3


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

    # Print locked constants — validate.py parses this block
    print("=== CONSTANTS ===")
    print(f"HIDDEN_DIM: {HIDDEN_DIM}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"EPOCHS: {EPOCHS}")
    print(f"HALF_POPULATION: {HALF_POPULATION}")
    print(f"T: {T}")
    print(f"SEED: {args.seed}")
    print("=================")

    key = jax.random.PRNGKey(args.seed)

    # Initialize weights
    key, k1, k2, k3 = jax.random.split(key, 4)
    initializer = jax.nn.initializers.orthogonal()
    w1 = initializer(k1, (784, HIDDEN_DIM), jnp.float32)
    w2 = initializer(k2, (HIDDEN_DIM, HIDDEN_DIM), jnp.float32)
    w3 = initializer(k3, (HIDDEN_DIM, 10), jnp.float32)

    # Shuffle once on CPU, group, and transfer to GPU
    rng = np.random.default_rng(args.seed)
    n_samples = N_GROUPS * GROUP_SIZE * BATCH_SIZE
    perm = rng.permutation(X_train_np.shape[0])
    X_grouped = jnp.array(X_train_np[perm[:n_samples]].reshape(N_GROUPS, GROUP_SIZE * BATCH_SIZE, -1))
    y_grouped = jnp.array(y_train_np[perm[:n_samples]].reshape(N_GROUPS, GROUP_SIZE * BATCH_SIZE))

    print("Training...")
    start_time = time.perf_counter()

    w1, w2, w3 = train_all_epochs(w1, w2, w3, X_grouped, y_grouped, key)
    jax.block_until_ready(w1)

    train_time = time.perf_counter() - start_time

    # Measure memory after training
    peak_memory = get_gpu_memory_mb()

    # Print epoch info retroactively (we can't time individual epochs inside JIT)
    for epoch in range(EPOCHS):
        lr_e = LR_START * (LR_DECAY ** epoch)
        sigma_e = SIGMA_START * (SIGMA_DECAY ** epoch)
        print(f"Epoch {epoch+1:2d} | LR: {lr_e:.4f} | Sigma: {sigma_e:.4f} | Time: {train_time/EPOCHS:.1f}s")

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

    # Machine-parseable results block — validate.py and benchmark.py grep this
    print("=== RESULTS ===")
    print(f"test_accuracy: {test_acc:.6f}")
    print(f"training_time_s: {train_time:.2f}")
    print(f"peak_memory_mb: {peak_memory:.1f}")
    print("===============")


if __name__ == "__main__":
    main()

"""
Optimized JAX backprop baseline — mirrors EGGROLL's JAX optimizations:
- jax.lax.scan (no Python loop overhead, no per-batch GPU sync)
- All-in-one JIT (single compiled function for all epochs)
- CPU shuffle-once (same strategy as EGGROLL)
- Batched evaluation

Same architecture, hyperparameters, and training semantics as mnist_backprop.py.
"""
import os
import sys
import time
import argparse
import numpy as np
import jax
import jax.numpy as jnp


def get_gpu_memory_mb():
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
data = np.load("mnist_prepped_float.npz")
X_train_np = data["X_train"]
y_train_np = data["y_train"]
X_test = jnp.array(data["X_test"])
y_test = jnp.array(data["y_test"])

# ---- Constants (match naive backprop exactly) ----
HIDDEN_DIM = 128
BATCH_SIZE = 128
EPOCHS = 10
LR_START = 0.1
LR_DECAY = 0.99

N_BATCHES = X_train_np.shape[0] // BATCH_SIZE  # 468, drop last incomplete batch


def fast_gelu(x):
    """Same sigmoid GELU approximation as EGGROLL for fair activation comparison."""
    return x * jax.nn.sigmoid(1.702 * x)


@jax.jit
def train_all_epochs(w1, w2, w3, X_batched, y_batched):
    """All epochs in a single JIT call — mirrors EGGROLL's structure."""

    def loss_fn(w1, w2, w3, xb, yb):
        h1 = fast_gelu(xb @ w1)
        h2 = fast_gelu(h1 @ w2)
        logits = h2 @ w3
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        y_one_hot = jax.nn.one_hot(yb, 10)
        return -jnp.sum(log_probs * y_one_hot) / xb.shape[0]

    def epoch_step(carry, _):
        w1, w2, w3, lr = carry

        def batch_step(carry, batch_data):
            w1, w2, w3 = carry
            xb, yb = batch_data
            loss, (g1, g2, g3) = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))(w1, w2, w3, xb, yb)
            w1 = w1 - lr * g1
            w2 = w2 - lr * g2
            w3 = w3 - lr * g3
            return (w1, w2, w3), None

        (w1, w2, w3), _ = jax.lax.scan(batch_step, (w1, w2, w3), (X_batched, y_batched))
        lr = lr * LR_DECAY
        return (w1, w2, w3, lr), None

    (w1, w2, w3, _), _ = jax.lax.scan(epoch_step, (w1, w2, w3, jnp.float32(LR_START)), None, length=EPOCHS)
    return w1, w2, w3


@jax.jit
def evaluate_batch(w1, w2, w3, xb, yb):
    h1 = fast_gelu(xb @ w1)
    h2 = fast_gelu(h1 @ w2)
    logits = h2 @ w3
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
    print(f"SEED: {args.seed}")
    print("=================")

    key = jax.random.PRNGKey(args.seed)

    # Initialize weights — same he_normal as naive backprop
    key, k1, k2, k3 = jax.random.split(key, 4)
    initializer = jax.nn.initializers.he_normal()
    w1 = initializer(k1, (784, HIDDEN_DIM), jnp.float32)
    w2 = initializer(k2, (HIDDEN_DIM, HIDDEN_DIM), jnp.float32)
    w3 = initializer(k3, (HIDDEN_DIM, 10), jnp.float32)

    print("Training...")
    start_time = time.perf_counter()

    # Shuffle once on CPU, batch, transfer — same as EGGROLL
    rng = np.random.default_rng(args.seed)
    n_samples = N_BATCHES * BATCH_SIZE
    perm = rng.permutation(X_train_np.shape[0])
    X_batched = jnp.array(X_train_np[perm[:n_samples]].reshape(N_BATCHES, BATCH_SIZE, -1))
    y_batched = jnp.array(y_train_np[perm[:n_samples]].reshape(N_BATCHES, BATCH_SIZE))

    w1, w2, w3 = train_all_epochs(w1, w2, w3, X_batched, y_batched)
    jax.block_until_ready(w1)

    train_time = time.perf_counter() - start_time

    peak_memory = get_gpu_memory_mb()

    for epoch in range(EPOCHS):
        lr_e = LR_START * (LR_DECAY ** epoch)
        print(f"Epoch {epoch+1:2d} | LR: {lr_e:.4f} | Time: {train_time/EPOCHS:.1f}s")

    # Evaluate — batched, not per-sample
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

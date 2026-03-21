import os
import sys
import time
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

def get_gpu_memory_mb():
    """Get current GPU memory usage in MB."""
    try:
        devices = jax.devices('gpu')
        if devices:
            jax.block_until_ready(jnp.zeros(1))
            stats = devices[0].memory_stats()
            if stats:
                return stats.get('bytes_in_use', 0) / (1024 * 1024)
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
LR_START = 0.012
LR_DECAY = 0.88
SIGMA_START = 0.028
SIGMA_DECAY = 0.998

CHUNK = 1000  # process perturbations in chunks to keep intermediates in L2 cache
N_CHUNKS = HALF_POPULATION // CHUNK


def data_loader(X, y, batch_size, key, shuffle=True, drop_last=False):
    n = X.shape[0]
    if shuffle:
        perm = jax.random.permutation(key, n)
        X, y = X[perm], y[perm]
    end = (n // batch_size) * batch_size if drop_last else n
    for i in range(0, end, batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def generate_half_vectors(key, half_pop, input_dim, hidden_dim, output_dim):
    """Generate perturbation vectors for half the population."""
    keys = jax.random.split(key, 6)
    A1 = jax.random.normal(keys[0], (half_pop, hidden_dim), dtype=jnp.float32)
    B1 = jax.random.normal(keys[1], (half_pop, input_dim), dtype=jnp.float32)
    A2 = jax.random.normal(keys[2], (half_pop, hidden_dim), dtype=jnp.float32)
    B2 = jax.random.normal(keys[3], (half_pop, hidden_dim), dtype=jnp.float32)
    A3 = jax.random.normal(keys[4], (half_pop, output_dim), dtype=jnp.float32)
    B3 = jax.random.normal(keys[5], (half_pop, hidden_dim), dtype=jnp.float32)
    return A1, B1, A2, B2, A3, B3


@partial(jax.jit, static_argnames=('half_population',))
def train_step_antithetic(w1, w2, w3, xb, yb,
                          A1, B1, A2, B2, A3, B3,
                          sigma, lr, half_population):
    half_pop = half_population

    # Convert to bfloat16 for forward pass
    xb_f = xb.astype(jnp.bfloat16)
    w1_f = w1.astype(jnp.bfloat16)
    w2_f = w2.astype(jnp.bfloat16)
    w3_f = w3.astype(jnp.bfloat16)
    A1_f = A1.astype(jnp.bfloat16)
    B1_f = B1.astype(jnp.bfloat16)
    A2_f = A2.astype(jnp.bfloat16)
    B2_f = B2.astype(jnp.bfloat16)
    A3_f = A3.astype(jnp.bfloat16)
    B3_f = B3.astype(jnp.bfloat16)
    sigma_f = jnp.bfloat16(sigma)

    # Precompute shared quantities
    base1 = xb_f @ w1_f                    # (BATCH, HIDDEN)
    xB1_T = B1_f @ xb_f.T                  # (HALF_POP, BATCH)
    y_one_hot = jax.nn.one_hot(yb, 10)     # (BATCH, 10)

    # Reshape perturbation vectors into chunks for scan
    A1_c = A1_f.reshape(N_CHUNKS, CHUNK, HIDDEN_DIM)
    xB1_T_c = xB1_T.reshape(N_CHUNKS, CHUNK, BATCH_SIZE)
    A2_c = A2_f.reshape(N_CHUNKS, CHUNK, HIDDEN_DIM)
    B2_c = B2_f.reshape(N_CHUNKS, CHUNK, HIDDEN_DIM)
    A3_c = A3_f.reshape(N_CHUNKS, CHUNK, 10)
    B3_c = B3_f.reshape(N_CHUNKS, CHUNK, HIDDEN_DIM)

    def chunk_forward(carry, chunk_data):
        base1, w2_f, w3_f, sigma_f, y_one_hot = carry
        A1_ch, xB1_T_ch, A2_ch, B2_ch, A3_ch, B3_ch = chunk_data

        # L1: (CHUNK, BATCH, HIDDEN)
        pert1 = sigma_f * xB1_T_ch[:, :, None] * A1_ch[:, None, :]
        l1_pos = jax.nn.gelu((base1[None] + pert1).astype(jnp.float32)).astype(jnp.bfloat16)
        l1_neg = jax.nn.gelu((base1[None] - pert1).astype(jnp.float32)).astype(jnp.bfloat16)

        # L2: batched matmul via reshape
        C = CHUNK
        base2_pos = (l1_pos.reshape(-1, HIDDEN_DIM) @ w2_f).reshape(C, -1, HIDDEN_DIM)
        xB2_pos = jnp.einsum('pbh,ph->pb', l1_pos, B2_ch)
        pert2_pos = sigma_f * xB2_pos[:, :, None] * A2_ch[:, None, :]
        l2_pos = jax.nn.gelu((base2_pos + pert2_pos).astype(jnp.float32)).astype(jnp.bfloat16)

        base2_neg = (l1_neg.reshape(-1, HIDDEN_DIM) @ w2_f).reshape(C, -1, HIDDEN_DIM)
        xB2_neg = jnp.einsum('pbh,ph->pb', l1_neg, B2_ch)
        pert2_neg = sigma_f * xB2_neg[:, :, None] * A2_ch[:, None, :]
        l2_neg = jax.nn.gelu((base2_neg - pert2_neg).astype(jnp.float32)).astype(jnp.bfloat16)

        # L3: logits
        base3_pos = (l2_pos.reshape(-1, HIDDEN_DIM) @ w3_f).reshape(C, -1, 10)
        xB3_pos = jnp.einsum('pbh,ph->pb', l2_pos, B3_ch)
        logits_pos = (base3_pos + sigma_f * xB3_pos[:, :, None] * A3_ch[:, None, :]).astype(jnp.float32)

        base3_neg = (l2_neg.reshape(-1, HIDDEN_DIM) @ w3_f).reshape(C, -1, 10)
        xB3_neg = jnp.einsum('pbh,ph->pb', l2_neg, B3_ch)
        logits_neg = (base3_neg - sigma_f * xB3_neg[:, :, None] * A3_ch[:, None, :]).astype(jnp.float32)

        # CE fitness
        log_probs_pos = jax.nn.log_softmax(logits_pos / T, axis=-1)
        log_probs_neg = jax.nn.log_softmax(logits_neg / T, axis=-1)
        ce_pos = -jnp.sum(log_probs_pos * y_one_hot[None, :, :], axis=-1)
        ce_neg = -jnp.sum(log_probs_neg * y_one_hot[None, :, :], axis=-1)
        fitness_pos = -jnp.mean(ce_pos, axis=1)
        fitness_neg = -jnp.mean(ce_neg, axis=1)

        return carry, (fitness_pos, fitness_neg)

    carry = (base1, w2_f, w3_f, sigma_f, y_one_hot)
    scan_data = (A1_c, xB1_T_c, A2_c, B2_c, A3_c, B3_c)
    _, (fitness_pos, fitness_neg) = jax.lax.scan(chunk_forward, carry, scan_data)

    # Flatten: (N_CHUNKS, CHUNK) → (HALF_POP,)
    fitness_pos = fitness_pos.reshape(-1)
    fitness_neg = fitness_neg.reshape(-1)

    # Antithetic gradient
    fitness_diff = fitness_pos - fitness_neg
    mean = fitness_diff.mean()
    std = fitness_diff.std() + 1e-8
    shaped = (fitness_diff - mean) / std

    scale = 1.0 / (2 * sigma * half_pop)
    shaped_col = shaped[:, None]

    grad1 = scale * (B1 * shaped_col).T @ A1
    grad2 = scale * (B2 * shaped_col).T @ A2
    grad3 = scale * (B3 * shaped_col).T @ A3

    w1 = w1 + lr * grad1
    w2 = w2 + lr * grad2
    w3 = w3 + lr * grad3

    return w1, w2, w3


@jax.jit
def evaluate_batch(w1, w2, w3, xb, yb):
    l1 = jax.nn.gelu(xb @ w1)
    l2 = jax.nn.gelu(l1 @ w2)
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

    print("Training...")
    start_time = time.perf_counter()

    lr = LR_START
    sigma = SIGMA_START
    peak_memory = 0.0

    for epoch in range(EPOCHS):
        epoch_start = time.perf_counter()
        key, data_key = jax.random.split(key)

        for xb, yb in data_loader(X_train, y_train, BATCH_SIZE, data_key, drop_last=True):
            key, vec_key = jax.random.split(key)
            A1, B1, A2, B2, A3, B3 = generate_half_vectors(
                vec_key, HALF_POPULATION, 784, HIDDEN_DIM, 10
            )

            w1, w2, w3 = train_step_antithetic(
                w1, w2, w3, xb, yb,
                A1, B1, A2, B2, A3, B3,
                sigma, lr, HALF_POPULATION,
            )

            # Track peak memory
            current_mem = get_gpu_memory_mb()
            if current_mem > peak_memory:
                peak_memory = current_mem

        epoch_time = time.perf_counter() - epoch_start
        print(f"Epoch {epoch+1:2d} | LR: {lr:.4f} | Sigma: {sigma:.4f} | Time: {epoch_time:.1f}s")

        lr *= LR_DECAY
        sigma *= SIGMA_DECAY

    train_time = time.perf_counter() - start_time

    # Evaluate
    print("\nEvaluating on test set...")
    correct = 0
    total = 0
    for xb, yb in data_loader(X_test, y_test, 256, key, shuffle=False):
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

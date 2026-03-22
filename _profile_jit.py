"""Profile XLA JIT compilation to understand the 2.8s overhead."""
import os, time
import jax
import jax.numpy as jnp
import numpy as np

# Suppress GPU memory preallocation
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from mnist_eggroll_optimized import train_epoch, HALF_POPULATION, HIDDEN_DIM, BATCH_SIZE, N_BATCHES, LR_START, SIGMA_START

# Load data
data = np.load("mnist_prepped_float.npz")
X_train = jnp.array(data["X_train"])
y_train = jnp.array(data["y_train"])

key = jax.random.PRNGKey(11)
key, k1, k2, k3 = jax.random.split(key, 4)
init = jax.nn.initializers.orthogonal()
w1 = init(k1, (784, HIDDEN_DIM), jnp.float32)
w2 = init(k2, (HIDDEN_DIM, HIDDEN_DIM), jnp.float32)
w3 = init(k3, (HIDDEN_DIM, 10), jnp.float32)

key, dk = jax.random.split(key)
perm = jax.random.permutation(dk, X_train.shape[0])
X_shuf = X_train[perm][:N_BATCHES*BATCH_SIZE].reshape(N_BATCHES, BATCH_SIZE, -1)
y_shuf = y_train[perm][:N_BATCHES*BATCH_SIZE].reshape(N_BATCHES, BATCH_SIZE)

# Time just the lowering (tracing)
print("Lowering (tracing)...")
t0 = time.perf_counter()
lowered = jax.jit(train_epoch, donate_argnums=(3,4)).lower(w1, w2, w3, X_shuf, y_shuf, SIGMA_START, LR_START, key)
t1 = time.perf_counter()
print(f"  Lowering: {t1-t0:.3f}s")

# Time the compilation  
print("Compiling...")
t2 = time.perf_counter()
compiled = lowered.compile()
t3 = time.perf_counter()
print(f"  Compilation: {t3-t2:.3f}s")

# Time the first execution
print("First execution...")
t4 = time.perf_counter()
result = compiled(w1, w2, w3, X_shuf, y_shuf, SIGMA_START, LR_START, key)
jax.block_until_ready(result[0])
t5 = time.perf_counter()
print(f"  First execution: {t5-t4:.3f}s")

# Time a second execution (no compilation)
key = result[3]
key, dk2 = jax.random.split(key)
perm2 = jax.random.permutation(dk2, X_train.shape[0])
X_shuf2 = X_train[perm2][:N_BATCHES*BATCH_SIZE].reshape(N_BATCHES, BATCH_SIZE, -1)
y_shuf2 = y_train[perm2][:N_BATCHES*BATCH_SIZE].reshape(N_BATCHES, BATCH_SIZE)

print("Second execution...")
t6 = time.perf_counter()
result2 = compiled(result[0], result[1], result[2], X_shuf2, y_shuf2, SIGMA_START * 0.998, LR_START * 0.88, key)
jax.block_until_ready(result2[0])
t7 = time.perf_counter()
print(f"  Second execution: {t7-t6:.3f}s")

print(f"\nTotal JIT overhead: {(t1-t0)+(t3-t2):.3f}s")
print(f"Per-epoch compute: {t7-t6:.3f}s")

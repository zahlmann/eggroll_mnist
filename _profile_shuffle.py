"""Profile data shuffling overhead."""
import time
import jax
import jax.numpy as jnp
import numpy as np

data = np.load("mnist_prepped_float.npz")
X_train = jnp.array(data["X_train"])
y_train = jnp.array(data["y_train"])

N_BATCHES = X_train.shape[0] // 128
key = jax.random.PRNGKey(42)

# Warm up
key, dk = jax.random.split(key)
perm = jax.random.permutation(dk, X_train.shape[0])
X_shuf = X_train[perm][:N_BATCHES * 128].reshape(N_BATCHES, 128, -1)
y_shuf = y_train[perm][:N_BATCHES * 128].reshape(N_BATCHES, 128)
jax.block_until_ready(X_shuf)

# Time 10 shuffles
times = []
for i in range(10):
    key, dk = jax.random.split(key)
    t0 = time.perf_counter()
    perm = jax.random.permutation(dk, X_train.shape[0])
    X_shuf = X_train[perm][:N_BATCHES * 128].reshape(N_BATCHES, 128, -1)
    y_shuf = y_train[perm][:N_BATCHES * 128].reshape(N_BATCHES, 128)
    jax.block_until_ready(X_shuf)
    t1 = time.perf_counter()
    times.append(t1 - t0)
    
print(f"Per-epoch shuffle: {sum(times)/len(times)*1000:.1f}ms")
print(f"Total 10 epochs: {sum(times):.3f}s")

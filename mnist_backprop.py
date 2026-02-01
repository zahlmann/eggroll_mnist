import os
import time
import numpy as np
import jax
import jax.numpy as jnp

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'

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

# load data
if not os.path.exists("mnist_prepped_float.npz"):
    print("Error: mnist_prepped_float.npz not found. Please run mnist_prep_float.py first.")
    exit(1)

data = np.load("mnist_prepped_float.npz")
X_train = jnp.array(data["X_train"])
y_train = jnp.array(data["y_train"])
X_test  = jnp.array(data["X_test"])
y_test  = jnp.array(data["y_test"])

def init_params(key, input_dim, hidden_dim, output_dim):
    k1, k2, k3 = jax.random.split(key, 3)
    initializer = jax.nn.initializers.he_normal()
    w1 = initializer(k1, (input_dim, hidden_dim), jnp.float32)
    w2 = initializer(k2, (hidden_dim, hidden_dim), jnp.float32)
    w3 = initializer(k3, (hidden_dim, output_dim), jnp.float32)
    return {'w1': w1, 'w2': w2, 'w3': w3}

def activation_fn(x):
    return jax.nn.gelu(x)

@jax.jit
def forward(params, x):
    h1 = activation_fn(x @ params['w1'])
    h2 = activation_fn(h1 @ params['w2'])
    logits = h2 @ params['w3']

    return logits

@jax.jit
def loss_fn(params, x, y):
    logits = forward(params, x)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    y_one_hot = jax.nn.one_hot(y, 10)
    loss = -jnp.sum(log_probs * y_one_hot) / x.shape[0]
    return loss

@jax.jit
def train_step(params, x, y, lr):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
    return params, loss

def data_loader(X, y, batch_size, key):
    n = X.shape[0]
    perm = jax.random.permutation(key, n)
    X, y = X[perm], y[perm]
    for i in range(0, n, batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]

def predict(params, x):
    logits = forward(params, x[None, :])
    return int(jnp.argmax(logits))

# ---------- Training ----------
lr_start = 0.1
lr_decay = 0.99
batch_size = 128
epochs = 10
hidden_dim = 128

key = jax.random.PRNGKey(420)
key, init_key = jax.random.split(key)

params = init_params(init_key, 784, hidden_dim, 10)

print("Starting training...")
start_time = time.perf_counter()
peak_memory = 0.0

for epoch in range(epochs):
    key, loader_key = jax.random.split(key)

    lr = lr_start * (lr_decay ** epoch)

    epoch_losses = []
    for xb, yb in data_loader(X_train, y_train, batch_size, loader_key):
        params, loss = train_step(params, xb, yb, lr)
        epoch_losses.append(float(loss))

        current_mem = get_gpu_memory_mb()
        if current_mem > peak_memory:
            peak_memory = current_mem

    avg_loss = np.mean(epoch_losses)

    print(f"Epoch {epoch+1:4d} | Avg Loss: {avg_loss:.4f} | LR: {lr:.4f}")

end_time = time.perf_counter()
print(f"Total training time: {end_time - start_time:.2f} seconds")
print("Training completed. Starting evaluation.")

correct = 0
total = len(y_test)
print(f"Evaluating {total} test samples...")

for i in range(total):
    pred = predict(params, X_test[i])

    if pred == int(y_test[i]):
        correct += 1

test_accuracy = (correct / total) * 100
print(f"\nTest Accuracy: {test_accuracy:.2f}% ({correct}/{total})")
print(f"Peak GPU Memory: {peak_memory:.1f} MB")

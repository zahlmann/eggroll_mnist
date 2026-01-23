import os
import time
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from dataclasses import dataclass, field

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.98'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'

# load data
if not os.path.exists("mnist_prepped_float.npz"):
    print("Error: mnist_prepped_float.npz not found. Please run mnist_prep_float.py first.")
    exit(1)

data = np.load("mnist_prepped_float.npz")
X_train = jnp.array(data["X_train"])
y_train = jnp.array(data["y_train"])
X_test  = jnp.array(data["X_test"])
y_test  = jnp.array(data["y_test"])

def data_loader(X, y, batch_size, key, shuffle=True):
    n = X.shape[0]
    if shuffle:
        perm = jax.random.permutation(key, n)
        X, y = X[perm], y[perm]
    for i in range(0, n, batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]

key = jax.random.PRNGKey(420)

@dataclass
class Network:
    key1: jax.random.PRNGKey
    key2: jax.random.PRNGKey
    key3: jax.random.PRNGKey
    input_dim: int
    hidden_dim: int
    output_dim: int
    def __post_init__(self):
        initializer = jax.nn.initializers.he_normal()
        self.layer1_means: jax.Array = initializer(self.key1, (self.input_dim, self.hidden_dim), jnp.float32)
        self.layer2_means: jax.Array = initializer(self.key2, (self.hidden_dim, self.hidden_dim), jnp.float32)
        self.layer3_means: jax.Array = initializer(self.key3, (self.hidden_dim, self.output_dim), jnp.float32)

@dataclass
class TrainingConfig:
    lr_start: float
    lr_decay: float
    sigma_start: float
    sigma_decay: float
    batch_size: int
    epochs: int
    hidden_dim: int
    population: int


@dataclass
class TrainingState:
    config: TrainingConfig
    current_epoch: int = 0
    all_batch_accuracies: list = field(default_factory=list)
    all_epoch_accuracies: list = field(default_factory=list)

    def get_lr(self):
        return self.config.lr_start * (self.config.lr_decay ** self.current_epoch)

    def get_sigma(self):
        return self.config.sigma_start * (self.config.sigma_decay ** self.current_epoch)

    def add_batch_accuracy(self, acc):
        self.all_batch_accuracies.append(acc)

    def new_epoch(self):
        self.all_epoch_accuracies.append(sum(self.all_batch_accuracies) / len(self.all_batch_accuracies))
        self.all_batch_accuracies = []
        self.current_epoch += 1


# --- FUSED JAX OPS ---

@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def generate_vectors(key, population, input_dim, hidden_dim, output_dim):
    """Generate all A, B vectors once per batch."""
    keys = jax.random.split(key, 6)
    A1 = jax.random.normal(keys[0], (population, hidden_dim))
    B1 = jax.random.normal(keys[1], (population, input_dim))
    A2 = jax.random.normal(keys[2], (population, hidden_dim))
    B2 = jax.random.normal(keys[3], (population, hidden_dim))
    A3 = jax.random.normal(keys[4], (population, output_dim))
    B3 = jax.random.normal(keys[5], (population, hidden_dim))
    return A1, B1, A2, B2, A3, B3


@partial(jax.jit, static_argnames=('population',))
def fused_train_step(w1, w2, w3, xb, yb, A1, B1, A2, B2, A3, B3,
                     sigma, population, lr):
    """
    Single JIT-compiled function that performs:
    1. Forward pass with perturbations (using A, B vectors)
    2. Fitness computation
    3. Gradient computation (reusing same A, B vectors)
    4. Weight updates
    """
    batch = xb.shape[0]

    # Layer 1: shared base computation, per-worker perturbations
    # base1: (batch, hidden)
    base1 = xb @ w1
    # xb @ B1.T: (batch, input) @ (input, pop) -> (batch, pop)
    xb1 = xb @ B1.T
    # pert1: (pop, batch, hidden) = xb1.T[:, :, None] * A1[:, None, :]
    pert1 = xb1.T[:, :, None] * A1[:, None, :]
    # l1: (pop, batch, hidden)
    l1 = jax.nn.gelu(base1[None, :, :] + sigma * pert1)

    # Layer 2: batched matmul for base
    # Reshape l1 to (pop*batch, hidden), matmul with w2, reshape back
    base2 = (l1.reshape(-1, w1.shape[1]) @ w2).reshape(population, batch, -1)
    # xb2: (pop, batch) - dot product of each worker's activations with B2
    xb2 = jnp.einsum('pbh,ph->pb', l1, B2)
    pert2 = xb2[:, :, None] * A2[:, None, :]
    l2 = jax.nn.gelu(base2 + sigma * pert2)

    # Layer 3: output logits
    base3 = (l2.reshape(-1, w2.shape[1]) @ w3).reshape(population, batch, -1)
    xb3 = jnp.einsum('pbh,ph->pb', l2, B3)
    logits = base3 + sigma * (xb3[:, :, None] * A3[:, None, :])

    # Fitness: accuracy per worker
    # logits: (pop, batch, output_dim), yb: (batch,)
    predictions = jnp.argmax(logits, axis=-1)  # (pop, batch)
    fitness = jnp.mean(predictions == yb[None, :], axis=1)  # (pop,)

    # Centered rank-based fitness shaping
    ranks = jnp.argsort(jnp.argsort(fitness))  # 0 to population-1
    shaped = (ranks / (population - 1)) - 0.5  # centered around 0

    # Gradients (reusing A, B vectors - key insight!)
    scale = 1.0 / (sigma * population)
    shaped_col = shaped[:, None]

    # Weight updates: grad = scale * (B * shaped).T @ A
    grad1 = scale * (B1 * shaped_col).T @ A1
    grad2 = scale * (B2 * shaped_col).T @ A2
    grad3 = scale * (B3 * shaped_col).T @ A3

    w1 = w1 + lr * grad1
    w2 = w2 + lr * grad2
    w3 = w3 + lr * grad3

    return w1, w2, w3, fitness.mean()


@jax.jit
def get_logits(means_l1, means_l2, means_l3, input_vec):
    x = input_vec @ means_l1
    x = jax.nn.gelu(x)
    x = x @ means_l2
    x = jax.nn.gelu(x)
    x = x @ means_l3
    return x

def get_prediction(network, input_vec):
    input_arr = jnp.array([input_vec])
    logits = get_logits(network.layer1_means, network.layer2_means, network.layer3_means, input_arr)
    pred = jnp.argmax(logits, axis=1)[0]
    return int(pred)


# -------- TRAINING --------
training_config = TrainingConfig(
    lr_start=0.02,
    lr_decay=0.9,
    sigma_start=0.02,
    sigma_decay=0.95,
    batch_size=128,
    epochs=10,
    hidden_dim=256,
    population=39000
)

training_state = TrainingState(config=training_config)

key, network_key1, network_key2, network_key3 = jax.random.split(key, 4)
network = Network(
    key1=network_key1,
    key2=network_key2,
    key3=network_key3,
    input_dim=784,
    hidden_dim=training_config.hidden_dim,
    output_dim=10
)

print("Starting training (fused implementation)...")
start_time = time.perf_counter()

w1, w2, w3 = network.layer1_means, network.layer2_means, network.layer3_means

for epoch in range(training_config.epochs):
    key, data_loader_key = jax.random.split(key)
    for xb, yb in data_loader(X_train, y_train, batch_size=training_config.batch_size, key=data_loader_key):
        current_sigma = training_state.get_sigma()
        current_lr = training_state.get_lr()

        # Generate all random vectors once per batch
        key, vec_key = jax.random.split(key)
        A1, B1, A2, B2, A3, B3 = generate_vectors(
            vec_key,
            training_config.population,
            784,  # input_dim
            training_config.hidden_dim,
            10    # output_dim
        )

        # Single fused train step
        w1, w2, w3, avg_acc = fused_train_step(
            w1, w2, w3, xb, yb,
            A1, B1, A2, B2, A3, B3,
            current_sigma, training_config.population, current_lr
        )
        training_state.add_batch_accuracy(avg_acc.item())

    training_state.new_epoch()

    avg_acc_epoch = training_state.all_epoch_accuracies[-1]
    lr = training_state.get_lr()
    sigma = training_state.get_sigma()

    print(f"Epoch {epoch:4d} | "
              f"Avg Acc: {avg_acc_epoch:7.2%} | "
              f"LR: {lr:.6f} | "
              f"Sigma: {sigma:.6f} | "
            )

network.layer1_means, network.layer2_means, network.layer3_means = w1, w2, w3

end_time = time.perf_counter()
print(f"Total training time: {end_time - start_time:.2f} seconds")
print("Training completed. Starting evaluation.")

def evaluate_test_set(network, X_test, y_test):
    correct = 0
    total = len(y_test)
    predictions = []

    print(f"Evaluating {total} test samples...")

    for i in range(total):
        pred = get_prediction(network, X_test[i])
        predictions.append(pred)
        if pred == int(y_test[i]):
            correct += 1

    accuracy = (correct / total) * 100
    return accuracy, predictions

test_accuracy, test_predictions = evaluate_test_set(network, X_test, y_test)

print(f"\nTest Accuracy: {test_accuracy:.2f}% ({int(test_accuracy * len(y_test) / 100)}/{len(y_test)})")

print("Evaluation complete.")

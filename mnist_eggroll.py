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


# --- JAX OPS ---

# Generate random vectors for each worker 
batched_normal = jax.jit(jax.vmap(jax.random.normal, in_axes=(0, None, None)), static_argnums=(1, 2))

# Perturbation for first layer (all workers take same x)
@partial(jax.jit, static_argnums=(2,))
@partial(jax.vmap, in_axes=(0, None, None)) 
def perturbation_op_input(key_pair, x, dim):
    key1, key2 = key_pair
    # A is a vector of length dim (output dimension)
    A = jax.random.normal(key1, shape=(dim,), dtype=jnp.float32)
    # B is a vector of length x.shape[1] (input dimension)
    B = jax.random.normal(key2, shape=(x.shape[1],), dtype=jnp.float32)
    # Outer product: (batch, input) @ (input,) gives (batch,), then outer with (dim,) gives (batch, dim)
    scaled = x @ B
    low_rank_update = scaled[:, None] * A[None, :]
    return low_rank_update

# Perturbation for hidden layers (each worker takes different x from first layer, hence batching over x)
@partial(jax.jit, static_argnums=(2,))
@partial(jax.vmap, in_axes=(0, 0, None)) 
def perturbation_op(key_pair, x, dim):
    key1, key2 = key_pair
    A = jax.random.normal(key1, shape=(dim,), dtype=jnp.float32)
    B = jax.random.normal(key2, shape=(x.shape[1],), dtype=jnp.float32)
    scaled = x @ B
    low_rank_update = scaled[:, None] * A[None, :]
    return low_rank_update

def activation_fn(x):
    return jax.nn.gelu(x)

@jax.jit
def get_logits(means_l1, means_l2, means_l3, input_vec):
    x = input_vec @ means_l1
    x = activation_fn(x)
    x = x @ means_l2
    x = activation_fn(x)
    x = x @ means_l3 
    return x

def get_prediction(network, input_vec):
    input_arr = jnp.array([input_vec]) 
    logits = get_logits(network.layer1_means, network.layer2_means, network.layer3_means, input_arr)

    pred = jnp.argmax(logits, axis=1)[0]
    return int(pred)

def calculate_accuracy_fitness(logits, targets):
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == targets)

@partial(jax.jit, static_argnames=('hidden_dim', 'population', 'output_dim'))
def train_step(weights, xb, yb, key, current_lr, current_sigma,
               hidden_dim, population, output_dim):
    w1, w2, w3 = weights
    key, perturbation_key = jax.random.split(key, 2)
    perturbation_op_keys = jax.random.split(perturbation_key, (3, population, 2))
    
    # Layer 1
    l1_op = perturbation_op_input(perturbation_op_keys[0], xb, hidden_dim)
    l1_output = xb @ w1 + current_sigma * l1_op
    l1_output_act = activation_fn(l1_output)
    
    # Layer 2
    l2_op = perturbation_op(perturbation_op_keys[1], l1_output_act, hidden_dim)
    l2_output = l1_output_act @ w2 + current_sigma * l2_op
    l2_output_act = activation_fn(l2_output)
    
    # Layer 3
    l3_op = perturbation_op(perturbation_op_keys[2], l2_output_act, output_dim)
    worker_logits = l2_output_act @ w3 + current_sigma * l3_op

    per_worker_fitness = jax.vmap(calculate_accuracy_fitness)(worker_logits, jnp.tile(yb[None, :], (population, 1)))

    # Centered Rank Fitness Shaping
    ranks = jnp.argsort(jnp.argsort(per_worker_fitness)) # 0 to N-1
    centered_ranks = (ranks / (population - 1)) - 0.5
    
    # Use shaped fitness for updates
    per_worker_fitness_shaped = centered_ranks

    avg_accuracy = per_worker_fitness.mean()

    # Normalization factor for raw ES gradient
    scale = 1.0 / (current_sigma * population)

    # LAYER 1 UPDATE
    A1 = batched_normal(perturbation_op_keys[0, :, 0], (hidden_dim,), jnp.float32)
    B1 = batched_normal(perturbation_op_keys[0, :, 1], (xb.shape[1],), jnp.float32)
    
    B1_scaled = B1 * per_worker_fitness_shaped[:, None] 
    grad_l1_raw = B1_scaled.T @ A1
    grad_l1 = grad_l1_raw * scale
    
    w1 = w1 + current_lr * grad_l1

    # LAYER 2 UPDATE
    A2 = batched_normal(perturbation_op_keys[1, :, 0], (hidden_dim,), jnp.float32)
    B2 = batched_normal(perturbation_op_keys[1, :, 1], (hidden_dim,), jnp.float32)
    
    B2_scaled = B2 * per_worker_fitness_shaped[:, None]
    grad_l2_raw = B2_scaled.T @ A2
    grad_l2 = grad_l2_raw * scale
    
    w2 = w2 + current_lr * grad_l2

    # LAYER 3 UPDATE
    A3 = batched_normal(perturbation_op_keys[2, :, 0], (output_dim,), jnp.float32)
    B3 = batched_normal(perturbation_op_keys[2, :, 1], (hidden_dim,), jnp.float32)
    
    B3_scaled = B3 * per_worker_fitness_shaped[:, None]
    grad_l3_raw = B3_scaled.T @ A3
    grad_l3 = grad_l3_raw * scale
    
    w3 = w3 + current_lr * grad_l3

    return (w1, w2, w3), avg_accuracy

# -------- TRAINING --------
training_config = TrainingConfig(
    lr_start=0.02,
    lr_decay=0.9,
    sigma_start=0.02,
    sigma_decay=0.95,
    batch_size=128,
    epochs=10,
    hidden_dim=128,
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

print("Starting training...")
start_time = time.perf_counter()

weights = (network.layer1_means, network.layer2_means, network.layer3_means)

for epoch in range(training_config.epochs):
    key, data_loader_key = jax.random.split(key)
    for xb, yb in data_loader(X_train, y_train, batch_size=training_config.batch_size, key=data_loader_key):
        current_sigma = training_state.get_sigma()
        current_lr = training_state.get_lr()
        
        key, step_key = jax.random.split(key)
        weights, avg_acc = train_step(
            weights, xb, yb, step_key, current_lr, current_sigma,
            training_config.hidden_dim, 
            training_config.population, 
            network.output_dim
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

network.layer1_means, network.layer2_means, network.layer3_means = weights

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

"""
EGGROLL MNIST training with Triton optimization.

Key optimization: Process population in tiles to reduce memory from O(pop * batch * hidden)
to O(tile * batch * hidden). Uses reproducible RNG to regenerate perturbation vectors
during gradient computation.

Architecture: 784 -> 256 -> 256 -> 10 with GELU activations
"""

import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from dataclasses import dataclass, field


# ============================================================================
# Triton Kernels for Random Number Generation
# ============================================================================

@triton.jit
def philox_round(c0, c1, c2, c3, k0, k1):
    """Single round of Philox 4x32."""
    PHILOX_ROUND_A: tl.constexpr = 0xD2511F53
    PHILOX_ROUND_B: tl.constexpr = 0xCD9E8D57

    hi0 = tl.umulhi(PHILOX_ROUND_A, c0)
    lo0 = PHILOX_ROUND_A * c0
    hi1 = tl.umulhi(PHILOX_ROUND_B, c2)
    lo1 = PHILOX_ROUND_B * c2

    new_c0 = hi1 ^ c1 ^ k0
    new_c1 = lo1
    new_c2 = hi0 ^ c3 ^ k1
    new_c3 = lo0
    return new_c0, new_c1, new_c2, new_c3


@triton.jit
def philox_4x32_10(c0, c1, c2, c3, key):
    """Philox 4x32-10: generates 4 uint32 from counter + key."""
    PHILOX_KEY_A: tl.constexpr = 0x9E3779B9
    PHILOX_KEY_B: tl.constexpr = 0xBB67AE85

    k0 = key & 0xFFFFFFFF
    k1 = (key >> 32) ^ 0x1BD11BDA

    for _ in range(10):
        c0, c1, c2, c3 = philox_round(c0, c1, c2, c3, k0, k1)
        k0 = (k0 + PHILOX_KEY_A) & 0xFFFFFFFF
        k1 = (k1 + PHILOX_KEY_B) & 0xFFFFFFFF

    return c0, c1, c2, c3


@triton.jit
def box_muller(u0, u1):
    """Box-Muller: 2 uniform -> 2 normal."""
    # Convert uint32 to (0, 1) float
    f0 = (u0 >> 8).to(tl.float32) * (1.0 / 16777216.0)
    f1 = (u1 >> 8).to(tl.float32) * (1.0 / 16777216.0)
    f0 = tl.maximum(f0, 1e-7)  # Avoid log(0)

    r = tl.sqrt(-2.0 * tl.log(f0))
    theta = 6.283185307179586 * f1
    return r * tl.cos(theta), r * tl.sin(theta)


@triton.jit
def generate_randn_kernel(
    out_ptr,
    n_elements,
    seed,
    offset,  # Global offset for this generation
    BLOCK_SIZE: tl.constexpr,
):
    """Generate N(0,1) random numbers with Philox RNG."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Each pair of elements uses one Philox call
    pair_idx = offsets // 2
    is_second = (offsets % 2) == 1

    # Counter for Philox
    counter = offset + pair_idx

    c0, c1, c2, c3 = philox_4x32_10(
        counter.to(tl.uint32),
        tl.zeros_like(counter, dtype=tl.uint32),
        tl.zeros_like(counter, dtype=tl.uint32),
        tl.zeros_like(counter, dtype=tl.uint32),
        seed
    )

    z0, z1 = box_muller(c0, c1)
    result = tl.where(is_second, z1, z0)

    tl.store(out_ptr + offsets, result, mask=mask)


def generate_randn(shape, seed, offset, device):
    """Generate tensor of N(0,1) values using Triton Philox RNG."""
    n = np.prod(shape)
    out = torch.empty(n, dtype=torch.float32, device=device)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    generate_randn_kernel[grid](out, n, seed, offset, BLOCK_SIZE=BLOCK_SIZE)

    return out.view(shape)


# ============================================================================
# EGGROLL Core Operations (PyTorch + Tiling)
# ============================================================================

def forward_tile(x, w1, w2, w3, A1, B1, A2, B2, A3, B3, sigma):
    """
    Forward pass for a tile of population members.

    Args:
        x: (batch, 784) input
        w1, w2, w3: weight matrices
        A1, B1, etc: (tile, dim) perturbation vectors
        sigma: perturbation scale

    Returns:
        logits: (tile, batch, 10)
    """
    tile_size = A1.shape[0]
    batch = x.shape[0]

    # Layer 1: base + perturbation
    # base1: (batch, 256)
    base1 = x @ w1

    # Perturbation: (x @ B1.T) gives (batch, tile), outer with A1 gives (tile, batch, 256)
    # xB1: (batch, tile) = (batch, 784) @ (784, tile)
    xB1 = x @ B1.T  # (batch, tile)
    # pert1: (tile, batch, 256) = xB1.T[:, :, None] * A1[:, None, :]
    pert1 = xB1.T[:, :, None] * A1[:, None, :]  # (tile, batch, 256)

    l1 = F.gelu(base1[None, :, :] + sigma * pert1)  # (tile, batch, 256)

    # Layer 2: use batched matmul
    # Reshape l1 to (tile*batch, 256), matmul, reshape back
    base2 = (l1.reshape(-1, w1.shape[1]) @ w2).reshape(tile_size, batch, -1)

    # Perturbation for layer 2: each population member uses its own activations
    # xB2[i, j] = l1[i, j, :] @ B2[i, :]
    xB2 = torch.einsum('tbh,th->tb', l1, B2)  # (tile, batch)
    pert2 = xB2[:, :, None] * A2[:, None, :]  # (tile, batch, 256)

    l2 = F.gelu(base2 + sigma * pert2)  # (tile, batch, 256)

    # Layer 3: output (no activation)
    base3 = (l2.reshape(-1, w2.shape[1]) @ w3).reshape(tile_size, batch, -1)

    xB3 = torch.einsum('tbh,th->tb', l2, B3)
    pert3 = xB3[:, :, None] * A3[:, None, :]

    logits = base3 + sigma * pert3  # (tile, batch, 10)

    return logits


def compute_fitness(logits, labels):
    """Compute accuracy for each population member."""
    # logits: (tile, batch, 10), labels: (batch,)
    predictions = logits.argmax(dim=-1)  # (tile, batch)
    correct = (predictions == labels[None, :]).float()
    return correct.mean(dim=1)  # (tile,)


def generate_vectors_for_tile(seed, tile_offset, tile_size, dims, device):
    """
    Generate A, B vectors for a tile of population members.

    Uses deterministic offsets so vectors can be regenerated identically.
    """
    in_dim, hidden_dim, out_dim = dims

    # Compute offsets for each vector type
    # Layout: A1(tile*hidden), B1(tile*in), A2(tile*hidden), B2(tile*hidden), A3(tile*out), B3(tile*hidden)

    # For reproducibility, use tile_offset * total_elements_per_member as base offset
    elements_per_member = hidden_dim + in_dim + hidden_dim + hidden_dim + out_dim + hidden_dim
    base_offset = tile_offset * elements_per_member

    offset = base_offset
    A1 = generate_randn((tile_size, hidden_dim), seed, offset, device)
    offset += tile_size * hidden_dim

    B1 = generate_randn((tile_size, in_dim), seed, offset, device)
    offset += tile_size * in_dim

    A2 = generate_randn((tile_size, hidden_dim), seed, offset, device)
    offset += tile_size * hidden_dim

    B2 = generate_randn((tile_size, hidden_dim), seed, offset, device)
    offset += tile_size * hidden_dim

    A3 = generate_randn((tile_size, out_dim), seed, offset, device)
    offset += tile_size * out_dim

    B3 = generate_randn((tile_size, hidden_dim), seed, offset, device)

    return A1, B1, A2, B2, A3, B3


def train_step_tiled(w1, w2, w3, x, y, seed, sigma, lr, population, tile_size, device):
    """
    One training step with tiled population processing.

    Args:
        w1, w2, w3: weight matrices
        x: (batch, 784) input batch
        y: (batch,) labels
        seed: RNG seed for this step
        sigma, lr: hyperparameters
        population: total population size
        tile_size: population members per tile

    Returns:
        Updated w1, w2, w3, average fitness
    """
    batch = x.shape[0]
    in_dim = 784
    hidden_dim = w1.shape[1]
    out_dim = 10
    dims = (in_dim, hidden_dim, out_dim)

    n_tiles = (population + tile_size - 1) // tile_size

    # Collect all fitness values
    all_fitness = []

    # Forward pass for all tiles (collect fitness only)
    for tile_idx in range(n_tiles):
        tile_offset = tile_idx * tile_size
        current_tile_size = min(tile_size, population - tile_offset)

        # Generate vectors
        A1, B1, A2, B2, A3, B3 = generate_vectors_for_tile(
            seed, tile_offset, current_tile_size, dims, device
        )

        # Forward pass
        logits = forward_tile(x, w1, w2, w3, A1, B1, A2, B2, A3, B3, sigma)

        # Compute fitness
        fitness = compute_fitness(logits, y)
        all_fitness.append(fitness)

    # Concatenate all fitness values
    all_fitness = torch.cat(all_fitness)  # (population,)

    # Rank-based fitness shaping
    ranks = torch.argsort(torch.argsort(all_fitness)).float()
    shaped = (ranks / (population - 1)) - 0.5  # Centered around 0

    avg_fitness = all_fitness.mean().item()

    # Gradient accumulation
    scale = 1.0 / (sigma * population)

    grad1 = torch.zeros_like(w1)
    grad2 = torch.zeros_like(w2)
    grad3 = torch.zeros_like(w3)

    fitness_offset = 0
    for tile_idx in range(n_tiles):
        tile_offset = tile_idx * tile_size
        current_tile_size = min(tile_size, population - tile_offset)

        # Regenerate same vectors
        A1, B1, A2, B2, A3, B3 = generate_vectors_for_tile(
            seed, tile_offset, current_tile_size, dims, device
        )

        # Get shaped fitness for this tile
        tile_shaped = shaped[fitness_offset:fitness_offset + current_tile_size]
        fitness_offset += current_tile_size

        # Gradient contribution: scale * (B * shaped[:, None]).T @ A
        shaped_col = tile_shaped[:, None]

        grad1 += scale * (B1 * shaped_col).T @ A1
        grad2 += scale * (B2 * shaped_col).T @ A2
        grad3 += scale * (B3 * shaped_col).T @ A3

    # Update weights
    w1 = w1 + lr * grad1
    w2 = w2 + lr * grad2
    w3 = w3 + lr * grad3

    return w1, w2, w3, avg_fitness


# ============================================================================
# Training Infrastructure
# ============================================================================

@dataclass
class TrainingConfig:
    lr_start: float = 0.02
    lr_decay: float = 0.9
    sigma_start: float = 0.02
    sigma_decay: float = 0.95
    batch_size: int = 128
    epochs: int = 10
    hidden_dim: int = 256
    population: int = 39000
    tile_size: int = 256  # Population members per tile


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
        if self.all_batch_accuracies:
            self.all_epoch_accuracies.append(
                sum(self.all_batch_accuracies) / len(self.all_batch_accuracies)
            )
        self.all_batch_accuracies = []
        self.current_epoch += 1


def data_loader(X, y, batch_size, shuffle=True, generator=None):
    """Simple data loader with shuffling."""
    n = X.shape[0]
    if shuffle:
        perm = torch.randperm(n, device='cpu', generator=generator).to(X.device)
        X, y = X[perm], y[perm]
    for i in range(0, n, batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]


def init_weights(in_dim, hidden_dim, out_dim, device, seed=420):
    """Initialize weights with He normal."""
    gen = torch.Generator(device=device).manual_seed(seed)

    w1 = torch.randn(in_dim, hidden_dim, device=device, generator=gen) * np.sqrt(2.0 / in_dim)
    w2 = torch.randn(hidden_dim, hidden_dim, device=device, generator=gen) * np.sqrt(2.0 / hidden_dim)
    w3 = torch.randn(hidden_dim, out_dim, device=device, generator=gen) * np.sqrt(2.0 / hidden_dim)

    return w1, w2, w3


def evaluate(w1, w2, w3, X, y, batch_size=256):
    """Evaluate accuracy on dataset."""
    correct = 0
    total = 0

    for xb, yb in data_loader(X, y, batch_size, shuffle=False):
        # Forward pass (no perturbation)
        l1 = F.gelu(xb @ w1)
        l2 = F.gelu(l1 @ w2)
        logits = l2 @ w3

        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.shape[0]

    return correct / total


# ============================================================================
# Main Training Loop
# ============================================================================

def main():
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if device.type != 'cuda':
        print("WARNING: Triton requires CUDA. Falling back to CPU (will be slow).")

    # Load data
    if not os.path.exists("mnist_prepped_float.npz"):
        print("Error: mnist_prepped_float.npz not found.")
        print("Please run: python mnist_data_prep.py")
        return

    data = np.load("mnist_prepped_float.npz")
    X_train = torch.tensor(data["X_train"], dtype=torch.float32, device=device)
    y_train = torch.tensor(data["y_train"], dtype=torch.int64, device=device)
    X_test = torch.tensor(data["X_test"], dtype=torch.float32, device=device)
    y_test = torch.tensor(data["y_test"], dtype=torch.int64, device=device)

    print(f"Training data: {X_train.shape}, Test data: {X_test.shape}")

    # Configuration
    config = TrainingConfig()
    state = TrainingState(config=config)

    # Initialize weights
    w1, w2, w3 = init_weights(784, config.hidden_dim, 10, device)

    # RNG (CPU generator for randperm compatibility)
    rng = torch.Generator(device='cpu').manual_seed(420)

    print(f"\nConfiguration:")
    print(f"  Population: {config.population}")
    print(f"  Tile size: {config.tile_size}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Memory tiles: {(config.population + config.tile_size - 1) // config.tile_size}")

    # Estimate memory
    tile_mem = config.tile_size * config.batch_size * config.hidden_dim * 4 * 3 / 1e6
    full_mem = config.population * config.batch_size * config.hidden_dim * 4 * 3 / 1e6
    print(f"  Estimated memory (tiled): {tile_mem:.1f} MB")
    print(f"  Estimated memory (full): {full_mem:.1f} MB")

    print("\nStarting training...")
    start_time = time.perf_counter()

    for epoch in range(config.epochs):
        epoch_start = time.perf_counter()

        for batch_idx, (xb, yb) in enumerate(data_loader(
            X_train, y_train, config.batch_size, shuffle=True, generator=rng
        )):
            sigma = state.get_sigma()
            lr = state.get_lr()

            # Generate unique seed for this batch
            seed = int(torch.randint(0, 2**31, (1,), generator=rng).item())

            w1, w2, w3, avg_acc = train_step_tiled(
                w1, w2, w3, xb, yb,
                seed, sigma, lr,
                config.population, config.tile_size,
                device
            )

            state.add_batch_accuracy(avg_acc)

        state.new_epoch()

        epoch_time = time.perf_counter() - epoch_start
        avg_acc_epoch = state.all_epoch_accuracies[-1]

        print(f"Epoch {epoch:4d} | "
              f"Avg Acc: {avg_acc_epoch:7.2%} | "
              f"LR: {state.get_lr():.6f} | "
              f"Sigma: {state.get_sigma():.6f} | "
              f"Time: {epoch_time:.1f}s")

    total_time = time.perf_counter() - start_time
    print(f"\nTotal training time: {total_time:.2f} seconds")

    # Evaluate
    print("\nEvaluating on test set...")
    test_acc = evaluate(w1, w2, w3, X_test, y_test)
    print(f"Test Accuracy: {test_acc:.2%} ({int(test_acc * len(y_test))}/{len(y_test)})")

    # Memory report
    if device.type == 'cuda':
        max_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak GPU memory: {max_mem:.2f} GB")


if __name__ == "__main__":
    main()

import os
os.environ.setdefault('TRITON_CACHE_DIR', '/tmp/triton_cache')
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from kernels.torch_3layer_ce import _fused_3layer_ce_both_kernel


def fast_gelu(x):
    """Sigmoid-based GELU approximation (GPT-2 style)."""
    return x * torch.sigmoid(1.702 * x)


def get_gpu_memory_mb():
    """Get peak GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


# Load data
if not os.path.exists("mnist_prepped_float.npz"):
    print("Error: mnist_prepped_float.npz not found.")
    exit(1)

data = np.load("mnist_prepped_float.npz")
X_train_np = data["X_train"]
y_train_np = data["y_train"]
X_test_np = data["X_test"]
y_test_np = data["y_test"]

# ---- LOCKED CONSTANTS (validate.py checks these — do not change values) ----
HALF_POPULATION = 5000
HIDDEN_DIM = 128
BATCH_SIZE = 128
EPOCHS = 10
T = 2.0

# ---- Tunable hyperparameters (agent may adjust these) ----
LR_START = 0.012
LR_DECAY = 0.88
SIGMA_START = 0.028
SIGMA_DECAY = 0.998

N_BATCHES = X_train_np.shape[0] // BATCH_SIZE
VEC_DIM = 784 + HIDDEN_DIM * 4 + 10  # 1306

GROUP_SIZE = 1
N_GROUPS = N_BATCHES // GROUP_SIZE  # 468

DEVICE = 'cuda'
OUT_DIM = 10
OUT_DIM_PAD = 16
BLOCK_B = 64
BLOCK_K = 32


def fused_3layer_ce_both(base1, xB1_T, A1, w2, B2, A2, w3_pad, B3, A3_pad,
                         sigma, T_val, y):
    """Launch the Triton kernel for both pos and neg CE."""
    HALF_POP = xB1_T.shape[0]
    BATCH = xB1_T.shape[1]
    HIDDEN = base1.shape[1]
    N_TILES = triton.cdiv(BATCH, BLOCK_B)

    # Scalar tensors for kernel
    sigma_t = torch.tensor([sigma], dtype=torch.float32, device=DEVICE)
    T_t = torch.tensor([T_val], dtype=torch.float32, device=DEVICE)

    # Pre-allocate outputs
    partial_ce_pos = torch.empty((HALF_POP, N_TILES), dtype=torch.float32, device=DEVICE)
    partial_ce_neg = torch.empty((HALF_POP, N_TILES), dtype=torch.float32, device=DEVICE)

    grid = (HALF_POP, N_TILES, 2)
    _fused_3layer_ce_both_kernel[grid](
        base1, xB1_T, A1,
        w2, B2, A2,
        w3_pad, B3, A3_pad,
        sigma_t, T_t,
        y,
        partial_ce_pos, partial_ce_neg,
        HALF_POP=HALF_POP, BATCH=BATCH, HIDDEN=HIDDEN,
        OUT_DIM=OUT_DIM, OUT_DIM_PAD=OUT_DIM_PAD,
        BLOCK_B=BLOCK_B, BLOCK_K=BLOCK_K, N_TILES=N_TILES,
        num_warps=4, num_stages=1,
    )
    return partial_ce_pos, partial_ce_neg


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

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Initialize weights (orthogonal)
    w1 = torch.empty(784, HIDDEN_DIM, dtype=torch.float32, device=DEVICE)
    w2 = torch.empty(HIDDEN_DIM, HIDDEN_DIM, dtype=torch.float32, device=DEVICE)
    w3 = torch.empty(HIDDEN_DIM, 10, dtype=torch.float32, device=DEVICE)
    torch.nn.init.orthogonal_(w1)
    torch.nn.init.orthogonal_(w2)
    torch.nn.init.orthogonal_(w3)

    # Pad w3 and prepare for kernel
    w3_pad = F.pad(w3, (0, OUT_DIM_PAD - OUT_DIM))  # (128, 16)

    print("Training...")
    torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()

    # Shuffle once on CPU, group, transfer to GPU (included in timing)
    rng = np.random.default_rng(args.seed)
    n_samples = N_GROUPS * GROUP_SIZE * BATCH_SIZE
    perm = rng.permutation(X_train_np.shape[0])
    X_shuf = torch.tensor(
        X_train_np[perm[:n_samples]].reshape(N_GROUPS, BATCH_SIZE, -1),
        dtype=torch.float32, device=DEVICE
    )
    y_shuf = torch.tensor(
        y_train_np[perm[:n_samples]].reshape(N_GROUPS, BATCH_SIZE),
        dtype=torch.int32, device=DEVICE
    )

    sigma = SIGMA_START
    lr = LR_START

    for epoch in range(EPOCHS):
        scale = 1.0 / (2.0 * sigma * HALF_POPULATION)

        for batch_idx in range(N_GROUPS):
            xb = X_shuf[batch_idx]  # (128, 784)
            yb = y_shuf[batch_idx]  # (128,)

            # Generate random perturbation vectors
            all_vecs = torch.randn(HALF_POPULATION, VEC_DIM, dtype=torch.float32, device=DEVICE)
            all_vecs_f = all_vecs.to(torch.bfloat16)

            # Slices must be contiguous for Triton kernel pointer arithmetic
            B1_f = all_vecs_f[:, :784].contiguous()
            A1_f = all_vecs_f[:, 784:784+HIDDEN_DIM].contiguous()
            B2_f = all_vecs_f[:, 784+HIDDEN_DIM:784+2*HIDDEN_DIM].contiguous()
            A2_f = all_vecs_f[:, 784+2*HIDDEN_DIM:784+3*HIDDEN_DIM].contiguous()
            B3_f = all_vecs_f[:, 784+3*HIDDEN_DIM:784+4*HIDDEN_DIM].contiguous()
            A3_f = all_vecs_f[:, 784+4*HIDDEN_DIM:].contiguous()

            B1 = all_vecs[:, :784].contiguous()
            A1 = all_vecs[:, 784:784+HIDDEN_DIM].contiguous()
            B2 = all_vecs[:, 784+HIDDEN_DIM:784+2*HIDDEN_DIM].contiguous()
            A2 = all_vecs[:, 784+2*HIDDEN_DIM:784+3*HIDDEN_DIM].contiguous()
            B3 = all_vecs[:, 784+3*HIDDEN_DIM:784+4*HIDDEN_DIM].contiguous()
            A3 = all_vecs[:, 784+4*HIDDEN_DIM:].contiguous()

            xb_f = xb.to(torch.bfloat16)
            w1_f = w1.to(torch.bfloat16)
            w2_f = w2.to(torch.bfloat16)

            base1 = xb_f @ w1_f          # (128, 128) bf16
            xB1_T = B1_f @ xb_f.T        # (5000, 128) bf16

            A3_pad = F.pad(A3_f, (0, OUT_DIM_PAD - OUT_DIM))

            partial_ce_pos, partial_ce_neg = fused_3layer_ce_both(
                base1, xB1_T, A1_f, w2_f, B2_f, A2_f,
                w3_pad.to(torch.bfloat16), B3_f, A3_pad,
                sigma, T, yb)

            fitness_diff = partial_ce_neg.sum(dim=1) - partial_ce_pos.sum(dim=1)
            mean = fitness_diff.mean()
            std = fitness_diff.std() + 1e-8
            shaped = (fitness_diff - mean) / std

            shaped_col = shaped.unsqueeze(1)
            grad1 = scale * (B1.T @ (shaped_col * A1))
            grad2 = scale * (B2.T @ (shaped_col * A2))
            grad3 = scale * (B3.T @ (shaped_col * A3))

            w1 = w1 + lr * grad1
            w2 = w2 + lr * grad2
            w3 = w3 + lr * grad3
            w3_pad = F.pad(w3, (0, OUT_DIM_PAD - OUT_DIM))

        sigma *= SIGMA_DECAY
        lr *= LR_DECAY

    torch.cuda.synchronize()
    train_time = time.perf_counter() - start_time

    peak_memory = get_gpu_memory_mb()

    for epoch in range(EPOCHS):
        lr_e = LR_START * (LR_DECAY ** epoch)
        sigma_e = SIGMA_START * (SIGMA_DECAY ** epoch)
        print(f"Epoch {epoch+1:2d} | LR: {lr_e:.4f} | Sigma: {sigma_e:.4f} | Time: {train_time/EPOCHS:.1f}s")

    # Evaluate
    print("\nEvaluating on test set...")
    X_test = torch.tensor(X_test_np, dtype=torch.float32, device=DEVICE)
    y_test = torch.tensor(y_test_np, dtype=torch.int64, device=DEVICE)

    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, X_test.shape[0], 256):
            xb = X_test[i:i+256]
            yb = y_test[i:i+256]
            l1 = fast_gelu(xb @ w1)
            l2 = fast_gelu(l1 @ w2)
            logits = l2 @ w3
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += len(yb)

    test_acc = correct / total

    print()
    print(f"Test Accuracy: {test_acc:.2%} ({correct}/{total})")
    print(f"Training Time: {train_time:.2f}s")
    print(f"Peak GPU Memory: {peak_memory:.1f} MB")

    print("=== RESULTS ===")
    print(f"test_accuracy: {test_acc:.6f}")
    print(f"training_time_s: {train_time:.2f}")
    print(f"peak_memory_mb: {peak_memory:.1f}")
    print("===============")


if __name__ == "__main__":
    main()

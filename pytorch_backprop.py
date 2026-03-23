"""PyTorch backprop baseline — same architecture and hyperparams."""
import os, time, argparse
import numpy as np
import torch
import torch.nn.functional as F


data = np.load("mnist_prepped_float.npz")
X_train_np, y_train_np = data["X_train"], data["y_train"].astype(np.int64)
X_test_np, y_test_np = data["X_test"], data["y_test"].astype(np.int64)

HIDDEN_DIM = 128
BATCH_SIZE = 128
EPOCHS = 10
LR_START = 0.1
LR_DECAY = 0.99


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    # Init weights (He normal)
    w1 = torch.randn(784, HIDDEN_DIM, device=device) * (2 / 784) ** 0.5
    w2 = torch.randn(HIDDEN_DIM, HIDDEN_DIM, device=device) * (2 / HIDDEN_DIM) ** 0.5
    w3 = torch.randn(HIDDEN_DIM, 10, device=device) * (2 / HIDDEN_DIM) ** 0.5
    w1.requires_grad_(True)
    w2.requires_grad_(True)
    w3.requires_grad_(True)

    X_test = torch.tensor(X_test_np, device=device)
    y_test = torch.tensor(y_test_np, device=device)

    print("Training...")
    start_time = time.perf_counter()

    # Shuffle once, batch, transfer (same as EGGROLL)
    rng = np.random.default_rng(args.seed)
    n_batches = X_train_np.shape[0] // BATCH_SIZE
    n_samples = n_batches * BATCH_SIZE
    perm = rng.permutation(X_train_np.shape[0])
    X_batched = torch.tensor(X_train_np[perm[:n_samples]].reshape(n_batches, BATCH_SIZE, -1), device=device)
    y_batched = torch.tensor(y_train_np[perm[:n_samples]].reshape(n_batches, BATCH_SIZE), device=device)

    for epoch in range(EPOCHS):
        lr = LR_START * (LR_DECAY ** epoch)
        for b in range(n_batches):
            xb, yb = X_batched[b], y_batched[b]
            h1 = F.gelu(xb @ w1)
            h2 = F.gelu(h1 @ w2)
            logits = h2 @ w3
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            with torch.no_grad():
                w1 -= lr * w1.grad
                w2 -= lr * w2.grad
                w3 -= lr * w3.grad
                w1.grad.zero_()
                w2.grad.zero_()
                w3.grad.zero_()

    torch.cuda.synchronize()
    train_time = time.perf_counter() - start_time

    peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Evaluate
    with torch.no_grad():
        correct = 0
        total = 0
        for i in range(0, X_test.shape[0], 256):
            xb = X_test[i:i+256]
            yb = y_test[i:i+256]
            h1 = F.gelu(xb @ w1)
            h2 = F.gelu(h1 @ w2)
            logits = h2 @ w3
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)

    test_acc = correct / total
    print(f"Test Accuracy: {test_acc:.2%}")
    print(f"Training Time: {train_time:.2f}s")
    print(f"Peak GPU Memory: {peak_memory:.1f} MB")
    print("=== RESULTS ===")
    print(f"test_accuracy: {test_acc:.6f}")
    print(f"training_time_s: {train_time:.2f}")
    print(f"peak_memory_mb: {peak_memory:.1f}")
    print("===============")


if __name__ == "__main__":
    main()

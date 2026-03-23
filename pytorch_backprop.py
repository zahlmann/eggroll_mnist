"""PyTorch backprop — written like a normal PyTorch user would."""
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


data = np.load("mnist_prepped_float.npz")
X_train_np, y_train_np = data["X_train"], data["y_train"].astype(np.int64)
X_test_np, y_test_np = data["X_test"], data["y_test"].astype(np.int64)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128, bias=False)
        self.fc2 = nn.Linear(128, 128, bias=False)
        self.fc3 = nn.Linear(128, 10, bias=False)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.gelu(self.fc1(x))
        x = self.gelu(self.fc2(x))
        return self.fc3(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    train_dataset = TensorDataset(
        torch.tensor(X_train_np, dtype=torch.float32),
        torch.tensor(y_train_np, dtype=torch.long),
    )
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    X_test = torch.tensor(X_test_np, device=device)
    y_test = torch.tensor(y_test_np, device=device)

    print("Training...")
    start_time = time.perf_counter()

    for epoch in range(10):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

    torch.cuda.synchronize()
    train_time = time.perf_counter() - start_time
    peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Evaluate
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        test_acc = (logits.argmax(1) == y_test).float().mean().item()

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

"""
PyTorch EGGROLL — same algorithm as JAX version but using PyTorch ops.
Uses the Triton kernel directly (no jax-triton bridge overhead).
"""
import os, time, argparse
import numpy as np
import torch
import triton
import triton.language as tl


data = np.load("mnist_prepped_float.npz")
X_train_np, y_train_np = data["X_train"], data["y_train"].astype(np.int64)
X_test_np, y_test_np = data["X_test"], data["y_test"].astype(np.int64)

HALF_POPULATION = 2750
HIDDEN_DIM = 128
BATCH_SIZE = 128
EPOCHS = 10
T = 2.0
LR_START = 0.012
LR_DECAY = 0.88
SIGMA_START = 0.028
SIGMA_DECAY = 0.998
VEC_DIM = 784 + HIDDEN_DIM * 4 + 10  # 1306


def fast_gelu(x):
    return x * torch.sigmoid(1.702 * x)


@triton.jit
def _fused_3layer_ce_both_kernel(
    base1_ptr, xB1_T_ptr, A1_ptr,
    w2_ptr, B2_ptr, A2_ptr,
    w3_ptr, B3_ptr, A3_ptr,
    sigma_ptr, T_ptr,
    y_ptr,
    partial_ce_pos_ptr, partial_ce_neg_ptr,
    HALF_POP: tl.constexpr, BATCH: tl.constexpr, HIDDEN: tl.constexpr,
    OUT_DIM: tl.constexpr, OUT_DIM_PAD: tl.constexpr,
    BLOCK_B: tl.constexpr, BLOCK_K: tl.constexpr, N_TILES: tl.constexpr,
):
    pid_p = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_sign = tl.program_id(2)

    b0 = pid_b * BLOCK_B
    offs_b = b0 + tl.arange(0, BLOCK_B)
    offs_h = tl.arange(0, HIDDEN)
    offs_o = tl.arange(0, OUT_DIM_PAD)
    mask_b = offs_b < BATCH

    sigma = tl.load(sigma_ptr).to(tl.float32)
    T_val = tl.load(T_ptr).to(tl.float32)
    sign = tl.where(pid_sign == 0, 1.0, -1.0)
    sign_sigma = sign * sigma

    xB1_col = tl.load(xB1_T_ptr + pid_p * BATCH + offs_b, mask=mask_b, other=0.0).to(tl.float32)

    base2 = tl.zeros((BLOCK_B, HIDDEN), dtype=tl.float32)
    xB2 = tl.zeros((BLOCK_B,), dtype=tl.float32)

    for k in range(0, HIDDEN, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        base1_k = tl.load(base1_ptr + offs_b[:, None] * HIDDEN + offs_k[None, :],
                          mask=mask_b[:, None], other=0.0).to(tl.float32)
        A1_k = tl.load(A1_ptr + pid_p * HIDDEN + offs_k).to(tl.float32)
        pre_act = base1_k + sign_sigma * xB1_col[:, None] * A1_k[None, :]
        l1_k = pre_act * tl.sigmoid(1.702 * pre_act)
        w2_k = tl.load(w2_ptr + offs_k[:, None] * HIDDEN + offs_h[None, :]).to(tl.float8e4nv)
        base2 = tl.dot(l1_k.to(tl.float8e4nv), w2_k, base2)
        B2_k = tl.load(B2_ptr + pid_p * HIDDEN + offs_k).to(tl.float32)
        xB2 += tl.sum(l1_k * B2_k[None, :], axis=1)

    A2_row = tl.load(A2_ptr + pid_p * HIDDEN + offs_h).to(tl.float32)
    pre_act2 = base2 + sign_sigma * xB2[:, None] * A2_row[None, :]
    l2 = pre_act2 * tl.sigmoid(1.702 * pre_act2)

    w3 = tl.load(w3_ptr + offs_h[:, None] * OUT_DIM_PAD + offs_o[None, :]).to(tl.float8e4nv)
    base3 = tl.dot(l2.to(tl.float8e4nv), w3).to(tl.float32)

    B3_row = tl.load(B3_ptr + pid_p * HIDDEN + offs_h).to(tl.float32)
    A3_row = tl.load(A3_ptr + pid_p * OUT_DIM_PAD + offs_o).to(tl.float32)
    xB3 = tl.sum(l2 * B3_row[None, :], axis=1)
    logits = base3 + sign_sigma * xB3[:, None] * A3_row[None, :]

    pad_mask = offs_o[None, :] >= OUT_DIM
    logits = tl.where(pad_mask, -1e9, logits)

    y_labels = tl.load(y_ptr + offs_b, mask=mask_b, other=0)
    scaled = logits / T_val
    max_val = tl.max(scaled, axis=1)[:, None]
    exp_val = tl.exp(scaled - max_val)
    log_sm = scaled - max_val - tl.log(tl.sum(exp_val, axis=1)[:, None])

    one_hot = (tl.arange(0, OUT_DIM_PAD)[None, :] == y_labels[:, None]).to(tl.float32)
    smooth = 0.98 * one_hot + 0.02 / 10.0
    smooth = tl.where(tl.arange(0, OUT_DIM_PAD)[None, :] >= OUT_DIM, 0.0, smooth)
    ce = -tl.sum(log_sm * smooth, axis=1)
    ce = tl.where(mask_b, ce, 0.0)

    out_ptr = tl.where(pid_sign == 0, partial_ce_pos_ptr, partial_ce_neg_ptr)
    tl.store(out_ptr + pid_p * N_TILES + pid_b, tl.sum(ce))


def fused_3layer_ce_both(base1, xB1_T, A1, w2, B2, A2, w3, B3, A3, sigma, T_val, y):
    HALF_POP, BATCH = xB1_T.shape
    _, HIDDEN = base1.shape
    OUT_DIM = 10
    OUT_DIM_PAD = 16
    BLOCK_B = 64
    BLOCK_K = 32
    N_TILES = triton.cdiv(BATCH, BLOCK_B)

    w3_pad = torch.nn.functional.pad(w3, (0, OUT_DIM_PAD - OUT_DIM))
    A3_pad = torch.nn.functional.pad(A3, (0, OUT_DIM_PAD - OUT_DIM))

    partial_ce_pos = torch.empty((HALF_POP, N_TILES), device=base1.device, dtype=torch.float32)
    partial_ce_neg = torch.empty((HALF_POP, N_TILES), device=base1.device, dtype=torch.float32)

    sigma_t = torch.tensor(sigma, device=base1.device, dtype=torch.float32)
    T_t = torch.tensor(T_val, device=base1.device, dtype=torch.float32)

    grid = (HALF_POP, N_TILES, 2)
    _fused_3layer_ce_both_kernel[grid](
        base1, xB1_T, A1, w2, B2, A2, w3_pad, B3, A3_pad,
        sigma_t, T_t, y.to(torch.int32),
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

    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    # Init weights (orthogonal, same as JAX EGGROLL)
    w1 = torch.nn.init.orthogonal_(torch.empty(784, HIDDEN_DIM, device=device))
    w2 = torch.nn.init.orthogonal_(torch.empty(HIDDEN_DIM, HIDDEN_DIM, device=device))
    w3 = torch.nn.init.orthogonal_(torch.empty(HIDDEN_DIM, 10, device=device))

    n_batches = X_train_np.shape[0] // BATCH_SIZE
    n_samples = n_batches * BATCH_SIZE

    print("Training...")
    start_time = time.perf_counter()

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(X_train_np.shape[0])
    X_batched = torch.tensor(X_train_np[perm[:n_samples]].reshape(n_batches, BATCH_SIZE, -1), device=device)
    y_batched = torch.tensor(y_train_np[perm[:n_samples]].reshape(n_batches, BATCH_SIZE), device=device, dtype=torch.int64)

    sigma = SIGMA_START
    lr = LR_START

    for epoch in range(EPOCHS):
        for b in range(n_batches):
            xb = X_batched[b]
            yb = y_batched[b]

            all_vecs = torch.randn(HALF_POPULATION, VEC_DIM, device=device, dtype=torch.float32)
            all_vecs_f = all_vecs.to(torch.bfloat16)

            # .contiguous() is critical — Triton kernel uses pointer arithmetic
            # that assumes dense row-major layout, but slices have stride=VEC_DIM
            B1_f = all_vecs_f[:, :784].contiguous()
            A1_f = all_vecs_f[:, 784:784+128].contiguous()
            B2_f = all_vecs_f[:, 784+128:784+256].contiguous()
            A2_f = all_vecs_f[:, 784+256:784+384].contiguous()
            B3_f = all_vecs_f[:, 784+384:784+512].contiguous()
            A3_f = all_vecs_f[:, 784+512:].contiguous()
            B1 = all_vecs[:, :784].contiguous()
            A1 = all_vecs[:, 784:784+128].contiguous()
            B2 = all_vecs[:, 784+128:784+256].contiguous()
            A2 = all_vecs[:, 784+256:784+384].contiguous()
            B3 = all_vecs[:, 784+384:784+512].contiguous()
            A3 = all_vecs[:, 784+512:].contiguous()

            xb_f = xb.to(torch.bfloat16)
            w1_f = w1.to(torch.bfloat16)
            w2_f = w2.to(torch.bfloat16)
            w3_f = w3.to(torch.bfloat16)

            base1 = xb_f @ w1_f
            xB1_T = B1_f @ xb_f.T

            partial_ce_pos, partial_ce_neg = fused_3layer_ce_both(
                base1, xB1_T, A1_f, w2_f, B2_f, A2_f, w3_f, B3_f, A3_f,
                sigma, T, yb)

            fitness_diff = partial_ce_neg.sum(dim=1) - partial_ce_pos.sum(dim=1)
            mean = fitness_diff.mean()
            std = fitness_diff.std() + 1e-8
            shaped = (fitness_diff - mean) / std

            scale = 1.0 / (2.0 * sigma * HALF_POPULATION)
            shaped_col = shaped[:, None]
            grad1 = scale * B1.T @ (shaped_col * A1)
            grad2 = scale * B2.T @ (shaped_col * A2)
            grad3 = scale * B3.T @ (shaped_col * A3)

            w1 = w1 + lr * grad1
            w2 = w2 + lr * grad2
            w3 = w3 + 2.0 * lr * grad3

        sigma *= SIGMA_DECAY
        lr *= LR_DECAY

    torch.cuda.synchronize()
    train_time = time.perf_counter() - start_time
    peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Evaluate
    with torch.no_grad():
        X_test = torch.tensor(X_test_np, device=device)
        y_test = torch.tensor(y_test_np, device=device)
        correct, total = 0, 0
        for i in range(0, X_test.shape[0], 256):
            xb = X_test[i:i+256]
            yb = y_test[i:i+256]
            h1 = fast_gelu(xb @ w1)
            h2 = fast_gelu(h1 @ w2)
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

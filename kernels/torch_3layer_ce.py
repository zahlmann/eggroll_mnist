"""
Standalone Triton kernel for PyTorch — no JAX dependencies.
Same kernel code as fused_3layer_ce.py but called directly from PyTorch.
"""

import torch
import triton
import triton.language as tl


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
    """Computes BOTH pos and neg CE in a single kernel.
    Grid: (HALF_POP, N_TILES, 2) where dim 2 selects sign."""
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

    xB1_col = tl.load(
        xB1_T_ptr + pid_p * BATCH + offs_b, mask=mask_b, other=0.0,
    ).to(tl.float32)

    # ── K-tiled L1 forward + L2 matmul ───────────────────────────────
    base2 = tl.zeros((BLOCK_B, HIDDEN), dtype=tl.float32)
    xB2 = tl.zeros((BLOCK_B,), dtype=tl.float32)

    for k in range(0, HIDDEN, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        base1_k = tl.load(
            base1_ptr + offs_b[:, None] * HIDDEN + offs_k[None, :],
            mask=mask_b[:, None], other=0.0,
        ).to(tl.float32)

        A1_k = tl.load(A1_ptr + pid_p * HIDDEN + offs_k).to(tl.float32)
        pre_act = base1_k + sign_sigma * xB1_col[:, None] * A1_k[None, :]
        l1_k = pre_act * tl.sigmoid(1.702 * pre_act)

        w2_k = tl.load(
            w2_ptr + offs_k[:, None] * HIDDEN + offs_h[None, :],
        ).to(tl.float8e4nv)
        base2 = tl.dot(l1_k.to(tl.float8e4nv), w2_k, base2)

        B2_k = tl.load(B2_ptr + pid_p * HIDDEN + offs_k).to(tl.float32)
        xB2 += tl.sum(l1_k * B2_k[None, :], axis=1)

    # ── L2 activation ────────────────────────────────────────────────
    A2_row = tl.load(A2_ptr + pid_p * HIDDEN + offs_h).to(tl.float32)
    pre_act2 = base2 + sign_sigma * xB2[:, None] * A2_row[None, :]
    l2 = pre_act2 * tl.sigmoid(1.702 * pre_act2)

    # ── Layer 3 (FP8) ────────────────────────────────────────────────
    w3 = tl.load(w3_ptr + offs_h[:, None] * OUT_DIM_PAD + offs_o[None, :]).to(tl.float8e4nv)
    base3 = tl.dot(l2.to(tl.float8e4nv), w3).to(tl.float32)

    B3_row = tl.load(B3_ptr + pid_p * HIDDEN + offs_h).to(tl.float32)
    A3_row = tl.load(A3_ptr + pid_p * OUT_DIM_PAD + offs_o).to(tl.float32)
    xB3 = tl.sum(l2 * B3_row[None, :], axis=1)
    logits = base3 + sign_sigma * xB3[:, None] * A3_row[None, :]

    pad_mask = offs_o[None, :] >= OUT_DIM
    logits = tl.where(pad_mask, -1e9, logits)

    # ── CE ────────────────────────────────────────────────────────────
    y_labels = tl.load(y_ptr + offs_b, mask=mask_b, other=0)
    scaled = logits / T_val
    max_val = tl.max(scaled, axis=1)[:, None]
    exp_val = tl.exp(scaled - max_val)
    log_sm = scaled - max_val - tl.log(tl.sum(exp_val, axis=1)[:, None])

    one_hot = (tl.arange(0, OUT_DIM_PAD)[None, :] == y_labels[:, None]).to(tl.float32)
    ce = -tl.sum(log_sm * one_hot, axis=1)
    ce = tl.where(mask_b, ce, 0.0)

    out_ptr = tl.where(pid_sign == 0, partial_ce_pos_ptr, partial_ce_neg_ptr)
    tl.store(out_ptr + pid_p * N_TILES + pid_b, tl.sum(ce))

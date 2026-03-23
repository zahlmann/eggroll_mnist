"""
Fused 3-layer forward pass + cross-entropy kernel with FP8 tensor cores.

Computes CE loss for all perturbation members (both +sigma and -sigma) in one launch.
K-tiles the L1->L2 matmul to keep register pressure low (25% occupancy, 4 blocks/SM).
Grid: (HALF_POP, BATCH // 64, 2) where dim 2 selects the perturbation sign.
"""

import triton
import triton.language as tl
import jax
import jax.numpy as jnp
import jax_triton as jt


@triton.jit
def _fused_3layer_ce_both_kernel(
    # Layer inputs and perturbation vectors (all bf16 except scalars)
    base1_ptr, xB1_T_ptr, A1_ptr,
    w2_ptr, B2_ptr, A2_ptr,
    w3_ptr, B3_ptr, A3_ptr,
    # Scalars
    sigma_ptr, T_ptr, smooth_alpha_ptr,
    y_ptr,
    # Outputs
    partial_ce_pos_ptr, partial_ce_neg_ptr,
    # Compile-time constants
    HALF_POP: tl.constexpr, BATCH: tl.constexpr, HIDDEN: tl.constexpr,
    OUT_DIM: tl.constexpr, OUT_DIM_PAD: tl.constexpr,
    BLOCK_B: tl.constexpr, BLOCK_K: tl.constexpr, N_TILES: tl.constexpr,
):
    pid_p = tl.program_id(0)     # perturbation index
    pid_b = tl.program_id(1)     # batch tile index
    pid_sign = tl.program_id(2)  # 0=positive, 1=negative

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_h = tl.arange(0, HIDDEN)
    offs_o = tl.arange(0, OUT_DIM_PAD)
    mask_b = offs_b < BATCH

    sigma = tl.load(sigma_ptr).to(tl.float32)
    T_val = tl.load(T_ptr).to(tl.float32)
    alpha = tl.load(smooth_alpha_ptr).to(tl.float32)
    sign_sigma = tl.where(pid_sign == 0, 1.0, -1.0) * sigma

    xB1_col = tl.load(xB1_T_ptr + pid_p * BATCH + offs_b, mask=mask_b, other=0.0).to(tl.float32)

    # ── K-tiled L1 forward + L2 matmul ───────────────────────────────
    # Process L1 in (BLOCK_B, BLOCK_K) tiles to keep base2 accumulator small.
    # Each tile: perturb L1 activations, apply GELU, accumulate into L2 via FP8 dot.
    base2 = tl.zeros((BLOCK_B, HIDDEN), dtype=tl.float32)
    xB2 = tl.zeros((BLOCK_B,), dtype=tl.float32)

    for k in range(0, HIDDEN, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        base1_k = tl.load(base1_ptr + offs_b[:, None] * HIDDEN + offs_k[None, :],
                           mask=mask_b[:, None], other=0.0).to(tl.float32)
        A1_k = tl.load(A1_ptr + pid_p * HIDDEN + offs_k).to(tl.float32)

        # L1: perturbed activation = GELU(base1 + sign*sigma * xB1 * A1)
        pre_act = base1_k + sign_sigma * xB1_col[:, None] * A1_k[None, :]
        l1_k = pre_act * tl.sigmoid(1.702 * pre_act)

        # Accumulate l1 @ w2 via FP8 tensor cores
        w2_k = tl.load(w2_ptr + offs_k[:, None] * HIDDEN + offs_h[None, :]).to(tl.float8e4nv)
        base2 = tl.dot(l1_k.to(tl.float8e4nv), w2_k, base2)

        # Accumulate l1 @ B2 for L2 perturbation
        B2_k = tl.load(B2_ptr + pid_p * HIDDEN + offs_k).to(tl.float32)
        xB2 += tl.sum(l1_k * B2_k[None, :], axis=1)

    # ── L2: perturbed activation ──────────────────────────────────────
    A2_row = tl.load(A2_ptr + pid_p * HIDDEN + offs_h).to(tl.float32)
    pre_act2 = base2 + sign_sigma * xB2[:, None] * A2_row[None, :]
    l2 = pre_act2 * tl.sigmoid(1.702 * pre_act2)

    # ── L3: logits via FP8 dot + perturbation ─────────────────────────
    w3 = tl.load(w3_ptr + offs_h[:, None] * OUT_DIM_PAD + offs_o[None, :]).to(tl.float8e4nv)
    base3 = tl.dot(l2.to(tl.float8e4nv), w3).to(tl.float32)

    B3_row = tl.load(B3_ptr + pid_p * HIDDEN + offs_h).to(tl.float32)
    A3_row = tl.load(A3_ptr + pid_p * OUT_DIM_PAD + offs_o).to(tl.float32)
    logits = base3 + sign_sigma * tl.sum(l2 * B3_row[None, :], axis=1)[:, None] * A3_row[None, :]
    logits = tl.where(offs_o[None, :] >= OUT_DIM, -1e9, logits)

    # ── Label-smoothed cross-entropy ──────────────────────────────────
    y_labels = tl.load(y_ptr + offs_b, mask=mask_b, other=0)
    scaled = logits / T_val
    max_val = tl.max(scaled, axis=1)[:, None]
    log_sm = scaled - max_val - tl.log(tl.sum(tl.exp(scaled - max_val), axis=1)[:, None])

    one_hot = (tl.arange(0, OUT_DIM_PAD)[None, :] == y_labels[:, None]).to(tl.float32)
    smooth = (1.0 - alpha) * one_hot + alpha / 10.0
    smooth = tl.where(tl.arange(0, OUT_DIM_PAD)[None, :] >= OUT_DIM, 0.0, smooth)
    ce = tl.where(mask_b, -tl.sum(log_sm * smooth, axis=1), 0.0)

    out_ptr = tl.where(pid_sign == 0, partial_ce_pos_ptr, partial_ce_neg_ptr)
    tl.store(out_ptr + pid_p * N_TILES + pid_b, tl.sum(ce))


def fused_3layer_ce_both(base1, xB1_T, A1, w2, B2, A2, w3, B3, A3, sigma, T_val, smooth_alpha, y):
    HALF_POP, BATCH = xB1_T.shape
    HIDDEN = base1.shape[1]
    OUT_DIM = w3.shape[1]
    OUT_DIM_PAD = 16
    BLOCK_B = 64
    BLOCK_K = 32
    N_TILES = triton.cdiv(BATCH, BLOCK_B)

    return jt.triton_call(
        base1, xB1_T, A1,
        w2, B2, A2,
        jnp.pad(w3, [(0, 0), (0, OUT_DIM_PAD - OUT_DIM)]),
        B3,
        jnp.pad(A3, [(0, 0), (0, OUT_DIM_PAD - OUT_DIM)]),
        sigma, T_val, smooth_alpha,
        y.astype(jnp.int32),
        kernel=_fused_3layer_ce_both_kernel,
        out_shape=[
            jax.ShapeDtypeStruct((HALF_POP, N_TILES), jnp.float32),
            jax.ShapeDtypeStruct((HALF_POP, N_TILES), jnp.float32),
        ],
        grid=(HALF_POP, N_TILES, 2),
        HALF_POP=HALF_POP, BATCH=BATCH, HIDDEN=HIDDEN,
        OUT_DIM=OUT_DIM, OUT_DIM_PAD=OUT_DIM_PAD,
        BLOCK_B=BLOCK_B, BLOCK_K=BLOCK_K, N_TILES=N_TILES,
        num_warps=4, num_stages=1,
    )

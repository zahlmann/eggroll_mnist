"""
K-tiled 3-layer fused forward + CE kernel.

Processes ONE direction (pos or neg) per call. K-tiles the L1→L2 matmul
so only (BLOCK_B, BLOCK_K) of l1 is live at any time, reducing register
pressure vs loading full l1 + full w2.

Grid: (HALF_POP, BATCH // BLOCK_B) — one program per (pop_member, batch_tile).
Each program outputs a partial CE sum for its batch tile.
"""

import triton
import triton.language as tl
import jax
import jax.numpy as jnp
import jax_triton as jt


@triton.jit
def _fused_3layer_ce_kernel(
    # Precomputed
    base1_ptr,      # (BATCH, HIDDEN) bf16
    xB1_T_ptr,      # (HALF_POP, BATCH) bf16
    A1_ptr,         # (HALF_POP, HIDDEN) bf16
    # Weights
    w2_ptr,         # (HIDDEN, HIDDEN) bf16
    B2_ptr,         # (HALF_POP, HIDDEN) bf16
    A2_ptr,         # (HALF_POP, HIDDEN) bf16
    w3_ptr,         # (HIDDEN, OUT_DIM_PAD) bf16
    B3_ptr,         # (HALF_POP, HIDDEN) bf16
    A3_ptr,         # (HALF_POP, OUT_DIM_PAD) bf16
    # Scalars
    sigma_ptr,      # () fp32
    T_ptr,          # () fp32
    sign_ptr,       # () fp32 — +1.0 for pos, -1.0 for neg
    # Labels
    y_ptr,          # (BATCH,) int32
    # Output
    partial_ce_ptr, # (HALF_POP, N_TILES) fp32
    # Dims
    HALF_POP:    tl.constexpr,
    BATCH:       tl.constexpr,
    HIDDEN:      tl.constexpr,
    OUT_DIM:     tl.constexpr,
    OUT_DIM_PAD: tl.constexpr,
    BLOCK_B:     tl.constexpr,
    BLOCK_K:     tl.constexpr,
    N_TILES:     tl.constexpr,
):
    pid_p = tl.program_id(0)
    pid_b = tl.program_id(1)

    b0 = pid_b * BLOCK_B
    offs_b = b0 + tl.arange(0, BLOCK_B)
    offs_h = tl.arange(0, HIDDEN)
    offs_o = tl.arange(0, OUT_DIM_PAD)
    mask_b = offs_b < BATCH

    sigma = tl.load(sigma_ptr).to(tl.float32)
    T_val = tl.load(T_ptr).to(tl.float32)
    sign = tl.load(sign_ptr).to(tl.float32)
    sign_sigma = sign * sigma

    xB1_col = tl.load(
        xB1_T_ptr + pid_p * BATCH + offs_b,
        mask=mask_b, other=0.0,
    ).to(tl.float32)

    # ── K-tiled L1 forward + L2 matmul accumulation ──────────────────
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
        ).to(tl.bfloat16)
        base2 = tl.dot(l1_k.to(tl.bfloat16), w2_k, base2)

        B2_k = tl.load(B2_ptr + pid_p * HIDDEN + offs_k).to(tl.float32)
        xB2 += tl.sum(l1_k * B2_k[None, :], axis=1)

    # ── L2 activation ─────────────────────────────────────────────────
    A2_row = tl.load(A2_ptr + pid_p * HIDDEN + offs_h).to(tl.float32)
    pre_act2 = base2 + sign_sigma * xB2[:, None] * A2_row[None, :]
    l2 = pre_act2 * tl.sigmoid(1.702 * pre_act2)

    # ── Layer 3 ───────────────────────────────────────────────────────
    w3 = tl.load(w3_ptr + offs_h[:, None] * OUT_DIM_PAD + offs_o[None, :]).to(tl.bfloat16)

    B3_row = tl.load(B3_ptr + pid_p * HIDDEN + offs_h).to(tl.float32)
    A3_row = tl.load(A3_ptr + pid_p * OUT_DIM_PAD + offs_o).to(tl.float32)

    base3 = tl.dot(l2.to(tl.bfloat16), w3).to(tl.float32)
    xB3 = tl.sum(l2 * B3_row[None, :], axis=1)
    logits = base3 + sign_sigma * xB3[:, None] * A3_row[None, :]

    pad_mask = offs_o[None, :] >= OUT_DIM
    logits = tl.where(pad_mask, -1e9, logits)

    # ── CE fitness ────────────────────────────────────────────────────
    y_labels = tl.load(y_ptr + offs_b, mask=mask_b, other=0)

    scaled = logits / T_val
    max_val = tl.max(scaled, axis=1)[:, None]
    exp_val = tl.exp(scaled - max_val)
    log_sm = scaled - max_val - tl.log(tl.sum(exp_val, axis=1)[:, None])

    one_hot = (tl.arange(0, OUT_DIM_PAD)[None, :] == y_labels[:, None]).to(tl.float32)
    ce = -tl.sum(log_sm * one_hot, axis=1)
    ce = tl.where(mask_b, ce, 0.0)
    partial_ce = tl.sum(ce)

    tl.store(partial_ce_ptr + pid_p * N_TILES + pid_b, partial_ce)


def fused_3layer_ce(base1, xB1_T, A1, w2, B2, A2, w3, B3, A3, sigma, T_val, sign, y):
    """
    Fused 3-layer forward + CE for ONE direction (pos or neg).

    Returns: partial_ce (HALF_POP, N_TILES) fp32
    """
    HALF_POP, BATCH = xB1_T.shape
    _, HIDDEN = base1.shape
    OUT_DIM = w3.shape[1]
    OUT_DIM_PAD = 16
    BLOCK_B = 64
    BLOCK_K = 32
    N_TILES = triton.cdiv(BATCH, BLOCK_B)

    w3_pad = jnp.pad(w3, [(0, 0), (0, OUT_DIM_PAD - OUT_DIM)])
    A3_pad = jnp.pad(A3, [(0, 0), (0, OUT_DIM_PAD - OUT_DIM)])

    out_shape = jax.ShapeDtypeStruct((HALF_POP, N_TILES), jnp.float32)
    grid = (HALF_POP, N_TILES)

    partial_ce = jt.triton_call(
        base1, xB1_T, A1,
        w2, B2, A2,
        w3_pad, B3, A3_pad,
        sigma, T_val, sign,
        y.astype(jnp.int32),
        kernel=_fused_3layer_ce_kernel,
        out_shape=out_shape,
        grid=grid,
        HALF_POP=HALF_POP,
        BATCH=BATCH,
        HIDDEN=HIDDEN,
        OUT_DIM=OUT_DIM,
        OUT_DIM_PAD=OUT_DIM_PAD,
        BLOCK_B=BLOCK_B,
        BLOCK_K=BLOCK_K,
        N_TILES=N_TILES,
        num_warps=4,
        num_stages=2,
    )

    return partial_ce

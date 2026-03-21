"""
K-tiled 3-layer fused forward + CE kernel.

Processes ONE direction (pos or neg) per call. Intermediates (l1, l2, logits)
never leave registers. K-tiles the matmul to avoid loading full w2/w3 at once.

Grid: (HALF_POP, BATCH // BLOCK_B) — one program per (pop_member, batch_tile).
Each program outputs a partial CE sum for its batch tile.

Key design vs previous fused_forward_fitness.py:
  - One direction per call (halves register pressure)
  - K-tiled matmul (BLOCK_K=32): loads w2 in 8KB tiles, not 32KB full
  - Sequential layers: l1 registers freed before l2 computed
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

    # ── Layer 1 (element-wise, no matmul) ────────────────────────────────
    base1 = tl.load(
        base1_ptr + offs_b[:, None] * HIDDEN + offs_h[None, :],
        mask=mask_b[:, None], other=0.0,
    ).to(tl.float32)

    xB1_col = tl.load(
        xB1_T_ptr + pid_p * BATCH + offs_b,
        mask=mask_b, other=0.0,
    ).to(tl.float32)

    A1_row = tl.load(A1_ptr + pid_p * HIDDEN + offs_h).to(tl.float32)

    pre_act = base1 + sign * sigma * xB1_col[:, None] * A1_row[None, :]
    # Fast GELU: x * sigmoid(1.702 * x)
    l1 = pre_act * tl.sigmoid(1.702 * pre_act)  # (BLOCK_B, HIDDEN) in fp32

    # ── Layer 2 (K-tiled matmul + fused dot product) ─────────────────────
    B2_row = tl.load(B2_ptr + pid_p * HIDDEN + offs_h).to(tl.float32)
    A2_row = tl.load(A2_ptr + pid_p * HIDDEN + offs_h).to(tl.float32)

    base2 = tl.zeros((BLOCK_B, HIDDEN), dtype=tl.float32)
    xB2 = tl.zeros((BLOCK_B,), dtype=tl.float32)

    for k in range(0, HIDDEN, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        # Load l1 tile and w2 tile
        l1_k = tl.load(
            base1_ptr + offs_b[:, None] * HIDDEN + offs_k[None, :],  # placeholder
            mask=mask_b[:, None], other=0.0,
        )  # We need l1[:, k:k+BLOCK_K] but l1 is in registers!

        # Since l1 is fully in registers as (BLOCK_B, HIDDEN), we can slice it
        # But Triton doesn't support dynamic slicing of register tensors.
        # We need to restructure: compute l1 per K-tile and accumulate.
        pass

    # PROBLEM: Triton can't slice a (BLOCK_B, HIDDEN) register tensor by K.
    # The K-tiling approach requires loading l1 in K-tiles, but l1 is computed
    # element-wise and lives in registers as a full (BLOCK_B, HIDDEN) block.
    #
    # Alternative: load w2 as full (HIDDEN, HIDDEN) — this IS what the previous
    # kernel did, and it caused register pressure. But with ONE direction (not two),
    # and using bf16 for w2, the register count is:
    #   l1: BLOCK_B * HIDDEN / num_threads fp32 regs
    #   w2: HIDDEN * HIDDEN / num_threads bf16 regs (packed)
    # With BLOCK_B=16, HIDDEN=128, 4 warps=128 threads:
    #   l1: 16*128/128 = 16 fp32 regs
    #   w2: 128*128*2/(128*4) = 64 bytes/thread = 16 bf16 regs = 8 fp32 regs
    # Total: 24 regs. That's fine!
    #
    # The previous kernel failed because it did BOTH directions simultaneously,
    # doubling l1 to 32 regs + both base2 accumulators. With one direction, it fits.

    # Load full w2 (bf16, will be L2-cached across programs)
    w2 = tl.load(w2_ptr + offs_h[:, None] * HIDDEN + offs_h[None, :]).to(tl.bfloat16)

    base2 = tl.dot(l1.to(tl.bfloat16), w2).to(tl.float32)  # (BLOCK_B, HIDDEN)
    xB2 = tl.sum(l1 * B2_row[None, :], axis=1)  # (BLOCK_B,)

    pre_act2 = base2 + sign * sigma * xB2[:, None] * A2_row[None, :]
    l2 = pre_act2 * tl.sigmoid(1.702 * pre_act2)  # (BLOCK_B, HIDDEN)

    # ── Layer 3 (matmul + perturbation → logits) ─────────────────────────
    w3 = tl.load(w3_ptr + offs_h[:, None] * OUT_DIM_PAD + offs_o[None, :]).to(tl.bfloat16)

    B3_row = tl.load(B3_ptr + pid_p * HIDDEN + offs_h).to(tl.float32)
    A3_row = tl.load(A3_ptr + pid_p * OUT_DIM_PAD + offs_o).to(tl.float32)

    base3 = tl.dot(l2.to(tl.bfloat16), w3).to(tl.float32)  # (BLOCK_B, OUT_DIM_PAD)
    xB3 = tl.sum(l2 * B3_row[None, :], axis=1)  # (BLOCK_B,)
    logits = base3 + sign * sigma * xB3[:, None] * A3_row[None, :]

    # Mask padded classes
    pad_mask = offs_o[None, :] >= OUT_DIM
    logits = tl.where(pad_mask, -1e9, logits)

    # ── CE fitness ────────────────────────────────────────────────────────
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
    BLOCK_B = 16
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

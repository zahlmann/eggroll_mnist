"""
Fused L1+L2 Triton kernel.

Computes layers 1 and 2 of the EGGROLL forward pass (both +sigma and -sigma
directions) without ever writing l1_pos/l1_neg to HBM.

Replaces:
    # --- Layer 1 ---
    base1 = xb @ w1                              (precomputed outside, kept)
    xB1   = xb @ B1.T                            → replaced by xB1_T = B1 @ xb.T
    pert1 = xB1.T[:,:,None] * A1[:,None,:]       (eliminated)
    l1_pos = gelu(base1 + sigma * pert1)         (eliminated)
    # --- Layer 2 ---
    base2_pos = (l1_pos.reshape(-1,H) @ w2).reshape(P,-1,H)  (eliminated)
    xB2_pos   = einsum('pbh,ph->pb', l1_pos, B2)              (eliminated)
    pert2_pos = xB2_pos[:,:,None] * A2[:,None,:]              (eliminated)
    l2_pos    = gelu(base2_pos + sigma * pert2_pos)           → OUTPUT

With a single Triton kernel that outputs l2_pos and l2_neg directly.

HBM savings:
  Writes eliminated: l1_pos 160MB + l1_neg 160MB = 320MB
  Reads eliminated:  l1_pos for base2_pos 160MB + for xB2_pos 160MB +
                     l1_neg for base2_neg 160MB + for xB2_neg 160MB = 640MB
  Total: ~960 MB per batch step

Memory layout improvement:
  xB1_T = B1 @ xb.T → (HALF_POP, BATCH) contiguous, so kernel reads
  xB1_T[pid_p, b_tile] as a coalesced row access.
"""

import triton
import triton.language as tl
import jax
import jax.numpy as jnp
import jax_triton as jt


@triton.jit
def _fused_l12_kernel(
    # Inputs
    base1_ptr,   # (BATCH, HIDDEN)    bfloat16 — precomputed xb @ w1
    xB1_T_ptr,   # (HALF_POP, BATCH)  bfloat16 — B1 @ xb.T (contiguous)
    A1_ptr,      # (HALF_POP, HIDDEN) bfloat16
    w2_ptr,      # (HIDDEN, HIDDEN)   bfloat16 — shared weight
    B2_ptr,      # (HALF_POP, HIDDEN) bfloat16
    A2_ptr,      # (HALF_POP, HIDDEN) bfloat16
    sigma_ptr,   # () float32 scalar
    # Outputs
    l2_pos_ptr,  # (HALF_POP, BATCH, HIDDEN) bfloat16
    l2_neg_ptr,  # (HALF_POP, BATCH, HIDDEN) bfloat16
    # Constexpr dims
    HALF_POP: tl.constexpr,
    BATCH:    tl.constexpr,
    HIDDEN:   tl.constexpr,
    BLOCK_B:  tl.constexpr,
):
    """
    Grid: (HALF_POP, BATCH // BLOCK_B)
    Each program processes one population member p and BLOCK_B batch items.
    """
    pid_p = tl.program_id(0)
    pid_b = tl.program_id(1)
    b0    = pid_b * BLOCK_B

    offs_b = b0 + tl.arange(0, BLOCK_B)
    offs_h = tl.arange(0, HIDDEN)
    mask_b = offs_b < BATCH

    sigma = tl.load(sigma_ptr).to(tl.float32)
    INV_SQRT2 = 0.7071067811865476

    # ── Layer 1 inputs ───────────────────────────────────────────────────────
    # base1[b_tile, :] → (BLOCK_B, HIDDEN) bf16 → fp32
    base1 = tl.load(
        base1_ptr + offs_b[:, None] * HIDDEN + offs_h[None, :],
        mask=mask_b[:, None], other=0.0,
    ).to(tl.float32)

    # xB1_T[pid_p, b_tile] → (BLOCK_B,)  coalesced row read
    xB1_col = tl.load(
        xB1_T_ptr + pid_p * BATCH + offs_b,
        mask=mask_b, other=0.0,
    ).to(tl.float32)

    # A1[pid_p, :] → (HIDDEN,)
    A1_row = tl.load(A1_ptr + pid_p * HIDDEN + offs_h).to(tl.float32)

    # pert1: (BLOCK_B, HIDDEN)
    pert1 = sigma * xB1_col[:, None] * A1_row[None, :]

    # ── Layer 1 activations (BLOCK_B, HIDDEN) ───────────────────────────────
    pos1_in = base1 + pert1
    l1_pos = 0.5 * pos1_in * (1.0 + tl.erf(pos1_in * INV_SQRT2))

    neg1_in = base1 - pert1
    l1_neg = 0.5 * neg1_in * (1.0 + tl.erf(neg1_in * INV_SQRT2))
    # base1 and pert1 no longer needed after this point

    # ── Layer 2 shared inputs ────────────────────────────────────────────────
    # w2: (HIDDEN, HIDDEN) — 32KB, will be cached in L2 across all programs
    w2 = tl.load(w2_ptr + offs_h[:, None] * HIDDEN + offs_h[None, :]).to(tl.bfloat16)

    # B2[pid_p, :], A2[pid_p, :] → (HIDDEN,)
    B2_row = tl.load(B2_ptr + pid_p * HIDDEN + offs_h).to(tl.float32)
    A2_row = tl.load(A2_ptr + pid_p * HIDDEN + offs_h).to(tl.float32)

    # ── Layer 2 positive direction ───────────────────────────────────────────
    # base2_pos = l1_pos @ w2 using tensor-core matmul
    base2_pos = tl.dot(l1_pos.to(tl.bfloat16), w2).to(tl.float32)  # (BLOCK_B, HIDDEN)

    # xB2_pos = sum_h(l1_pos * B2[p]) → (BLOCK_B,)
    xB2_pos = tl.sum(l1_pos * B2_row[None, :], axis=1)

    # pert2_pos: (BLOCK_B, HIDDEN)
    pert2_pos = sigma * xB2_pos[:, None] * A2_row[None, :]

    pos2_in = base2_pos + pert2_pos
    l2_pos_tile = 0.5 * pos2_in * (1.0 + tl.erf(pos2_in * INV_SQRT2))

    # ── Layer 2 negative direction ───────────────────────────────────────────
    base2_neg = tl.dot(l1_neg.to(tl.bfloat16), w2).to(tl.float32)

    xB2_neg = tl.sum(l1_neg * B2_row[None, :], axis=1)

    pert2_neg = sigma * xB2_neg[:, None] * A2_row[None, :]

    neg2_in = base2_neg - pert2_neg
    l2_neg_tile = 0.5 * neg2_in * (1.0 + tl.erf(neg2_in * INV_SQRT2))

    # ── Write outputs ────────────────────────────────────────────────────────
    out_off = pid_p * BATCH * HIDDEN + offs_b[:, None] * HIDDEN + offs_h[None, :]
    tl.store(l2_pos_ptr + out_off, l2_pos_tile.to(tl.bfloat16), mask=mask_b[:, None])
    tl.store(l2_neg_ptr + out_off, l2_neg_tile.to(tl.bfloat16), mask=mask_b[:, None])


def fused_l12(base1, xB1_T, A1, w2, B2, A2, sigma):
    """
    Fused L1+L2 forward pass (both +sigma and -sigma directions).

    Args:
        base1:  (BATCH, HIDDEN)    bfloat16 — precomputed xb @ w1
        xB1_T:  (HALF_POP, BATCH)  bfloat16 — B1 @ xb.T  (contiguous layout)
        A1:     (HALF_POP, HIDDEN) bfloat16
        w2:     (HIDDEN, HIDDEN)   bfloat16
        B2:     (HALF_POP, HIDDEN) bfloat16
        A2:     (HALF_POP, HIDDEN) bfloat16
        sigma:  () float32 JAX scalar
    Returns:
        l2_pos: (HALF_POP, BATCH, HIDDEN) bfloat16
        l2_neg: (HALF_POP, BATCH, HIDDEN) bfloat16
    """
    BATCH, HIDDEN = base1.shape
    HALF_POP = xB1_T.shape[0]
    BLOCK_B = 16

    out_shape = [
        jax.ShapeDtypeStruct((HALF_POP, BATCH, HIDDEN), jnp.bfloat16),
        jax.ShapeDtypeStruct((HALF_POP, BATCH, HIDDEN), jnp.bfloat16),
    ]

    grid = (HALF_POP, triton.cdiv(BATCH, BLOCK_B))

    return jt.triton_call(
        base1, xB1_T, A1,
        w2, B2, A2,
        sigma,
        kernel=_fused_l12_kernel,
        out_shape=out_shape,
        grid=grid,
        HALF_POP=HALF_POP,
        BATCH=BATCH,
        HIDDEN=HIDDEN,
        BLOCK_B=BLOCK_B,
    )

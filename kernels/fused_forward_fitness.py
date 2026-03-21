"""
Fused forward-pass + CE-fitness Triton kernel.

For each population member p, computes the full 3-layer forward pass
(L1 → L2 → L3) and the temperature-scaled CE fitness WITHOUT ever writing
intermediate activations (l1, l2, logits) to HBM.

Replaces the entire forward-pass section of train_step_antithetic:
  - base1 = xb @ w1           (precomputed outside, kept)
  - xB1 = B1 @ xb.T           (precomputed outside, kept — now (POP, BATCH))
  - l1_pos/neg                 (eliminated — computed in SRAM)
  - base2_pos/neg, l2_pos/neg  (eliminated — computed in SRAM)
  - logits_pos/neg             (eliminated — computed in SRAM)
  - fitness_pos/neg            (OUTPUT — (HALF_POP,) float32)

Kernel design:
  Grid: (HALF_POP, BATCH // BLOCK_B)
  Each program handles one population member p and a BLOCK_B-wide batch tile.
  For the full fitness mean, we output partial sums per batch tile, then
  reduce them outside the kernel (cheap: 5000 * n_tiles * 4B is tiny).

Memory savings vs baseline:
  Writes eliminated: l1_pos (160MB) + l1_neg (160MB) + l2_pos (160MB) +
                     l2_neg (160MB) + logits_pos (~50MB) + logits_neg (~50MB)
                   ≈ 740 MB per batch step
  Read eliminations: matching reads of these intermediates
  Net HBM savings: ~1.5 GB per batch step at double counting
"""

import triton
import triton.language as tl
import jax
import jax.numpy as jnp
import jax_triton as jt


@triton.jit
def _fused_fwd_fitness_kernel(
    # Precomputed base activations (shared across all p)
    base1_ptr,    # (BATCH, HIDDEN)    bfloat16 — xb @ w1
    xB1_T_ptr,    # (HALF_POP, BATCH)  bfloat16 — B1 @ xb.T (contiguous layout!)
    A1_ptr,       # (HALF_POP, HIDDEN) bfloat16
    # Network weights (shared, cached in L2)
    w2_ptr,       # (HIDDEN, HIDDEN)   bfloat16
    B2_ptr,       # (HALF_POP, HIDDEN) bfloat16
    A2_ptr,       # (HALF_POP, HIDDEN) bfloat16
    w3_ptr,       # (HIDDEN, OUT_DIM)  bfloat16
    B3_ptr,       # (HALF_POP, HIDDEN) bfloat16
    A3_ptr,       # (HALF_POP, OUT_DIM) bfloat16
    # Scalar
    sigma_ptr,    # () float32
    T_ptr,        # () float32  temperature
    # Labels
    y_ptr,        # (BATCH,) int32
    # Outputs — partial CE sums per (p, batch_tile)
    partial_ce_pos_ptr,  # (HALF_POP, N_TILES) float32
    partial_ce_neg_ptr,  # (HALF_POP, N_TILES) float32
    # Dims
    HALF_POP:  tl.constexpr,
    BATCH:     tl.constexpr,
    HIDDEN:    tl.constexpr,
    OUT_DIM:   tl.constexpr,
    BLOCK_B:   tl.constexpr,
    N_TILES:   tl.constexpr,
):
    pid_p = tl.program_id(0)   # [0, HALF_POP)
    pid_b = tl.program_id(1)   # [0, N_TILES)

    b0 = pid_b * BLOCK_B
    offs_b = b0 + tl.arange(0, BLOCK_B)    # (BLOCK_B,)
    offs_h = tl.arange(0, HIDDEN)          # (HIDDEN,)
    offs_o = tl.arange(0, OUT_DIM)         # (OUT_DIM,)
    mask_b = offs_b < BATCH

    sigma = tl.load(sigma_ptr).to(tl.float32)
    T_val = tl.load(T_ptr).to(tl.float32)

    # ── Layer 1 ──────────────────────────────────────────────────────────────
    # base1[b0:b0+BLOCK_B, :] → (BLOCK_B, HIDDEN)
    base1 = tl.load(
        base1_ptr + offs_b[:, None] * HIDDEN + offs_h[None, :],
        mask=mask_b[:, None], other=0.0,
    ).to(tl.float32)

    # xB1_T[pid_p, b0:b0+BLOCK_B] → (BLOCK_B,)  coalesced row access
    xB1_col = tl.load(
        xB1_T_ptr + pid_p * BATCH + offs_b,
        mask=mask_b, other=0.0,
    ).to(tl.float32)

    # A1[pid_p, :] → (HIDDEN,)
    A1_row = tl.load(A1_ptr + pid_p * HIDDEN + offs_h).to(tl.float32)

    # pert = sigma * outer(xB1_col, A1_row) → (BLOCK_B, HIDDEN)
    pert1 = sigma * xB1_col[:, None] * A1_row[None, :]

    # GELU: 0.5*x*(1+erf(x/sqrt(2)))
    INV_SQRT2 = 0.7071067811865476
    pos1_in = base1 + pert1
    neg1_in = base1 - pert1
    l1_pos = 0.5 * pos1_in * (1.0 + tl.erf(pos1_in * INV_SQRT2))  # (BLOCK_B, HIDDEN)
    l1_neg = 0.5 * neg1_in * (1.0 + tl.erf(neg1_in * INV_SQRT2))  # (BLOCK_B, HIDDEN)

    # ── Layer 2 ──────────────────────────────────────────────────────────────
    # w2: (HIDDEN, HIDDEN) — load row-by-row to keep in SRAM
    w2 = tl.load(w2_ptr + offs_h[:, None] * HIDDEN + offs_h[None, :]).to(tl.float32)
    # base2 = l1 @ w2: (BLOCK_B, HIDDEN) @ (HIDDEN, HIDDEN) → (BLOCK_B, HIDDEN)
    base2_pos = tl.dot(l1_pos.to(tl.bfloat16), w2.to(tl.bfloat16)).to(tl.float32)
    base2_neg = tl.dot(l1_neg.to(tl.bfloat16), w2.to(tl.bfloat16)).to(tl.float32)

    # B2[pid_p, :], A2[pid_p, :] → (HIDDEN,)
    B2_row = tl.load(B2_ptr + pid_p * HIDDEN + offs_h).to(tl.float32)
    A2_row = tl.load(A2_ptr + pid_p * HIDDEN + offs_h).to(tl.float32)

    # xB2 = l1 dot B2[p] → (BLOCK_B,)
    xB2_pos = tl.sum(l1_pos * B2_row[None, :], axis=1)
    xB2_neg = tl.sum(l1_neg * B2_row[None, :], axis=1)

    # pert2 = sigma * outer(xB2, A2) → (BLOCK_B, HIDDEN)
    pert2_pos = sigma * xB2_pos[:, None] * A2_row[None, :]
    pert2_neg = sigma * xB2_neg[:, None] * A2_row[None, :]

    pos2_in = base2_pos + pert2_pos
    neg2_in = base2_neg - pert2_neg
    l2_pos = 0.5 * pos2_in * (1.0 + tl.erf(pos2_in * INV_SQRT2))  # (BLOCK_B, HIDDEN)
    l2_neg = 0.5 * neg2_in * (1.0 + tl.erf(neg2_in * INV_SQRT2))  # (BLOCK_B, HIDDEN)

    # ── Layer 3 ──────────────────────────────────────────────────────────────
    # w3: (HIDDEN, OUT_DIM)
    w3 = tl.load(w3_ptr + offs_h[:, None] * OUT_DIM + offs_o[None, :]).to(tl.float32)
    # base3 = l2 @ w3: (BLOCK_B, HIDDEN) @ (HIDDEN, OUT_DIM) → (BLOCK_B, OUT_DIM)
    base3_pos = tl.dot(l2_pos.to(tl.bfloat16), w3.to(tl.bfloat16)).to(tl.float32)
    base3_neg = tl.dot(l2_neg.to(tl.bfloat16), w3.to(tl.bfloat16)).to(tl.float32)

    # B3[pid_p, :], A3[pid_p, :] → (HIDDEN,), (OUT_DIM,)
    B3_row = tl.load(B3_ptr + pid_p * HIDDEN + offs_h).to(tl.float32)
    A3_row = tl.load(A3_ptr + pid_p * OUT_DIM + offs_o).to(tl.float32)

    # xB3 = l2 dot B3[p] → (BLOCK_B,)
    xB3_pos = tl.sum(l2_pos * B3_row[None, :], axis=1)
    xB3_neg = tl.sum(l2_neg * B3_row[None, :], axis=1)

    # logits = base3 +/- sigma * outer(xB3, A3) → (BLOCK_B, OUT_DIM)
    logits_pos = base3_pos + sigma * xB3_pos[:, None] * A3_row[None, :]
    logits_neg = base3_neg - sigma * xB3_neg[:, None] * A3_row[None, :]

    # ── CE fitness ───────────────────────────────────────────────────────────
    # Load labels y[b0:b0+BLOCK_B] → (BLOCK_B,)
    y_labels = tl.load(y_ptr + offs_b, mask=mask_b, other=0)

    # Temperature-scaled log-softmax
    scaled_pos = logits_pos / T_val
    scaled_neg = logits_neg / T_val

    max_pos = tl.max(scaled_pos, axis=1)[:, None]  # (BLOCK_B, 1)
    max_neg = tl.max(scaled_neg, axis=1)[:, None]

    exp_pos = tl.exp(scaled_pos - max_pos)          # (BLOCK_B, OUT_DIM)
    exp_neg = tl.exp(scaled_neg - max_neg)

    sum_pos = tl.sum(exp_pos, axis=1)[:, None]      # (BLOCK_B, 1)
    sum_neg = tl.sum(exp_neg, axis=1)[:, None]

    log_softmax_pos = scaled_pos - max_pos - tl.log(sum_pos)
    log_softmax_neg = scaled_neg - max_neg - tl.log(sum_neg)

    # CE: -log_softmax[y]  → pick correct class per batch item
    # Gather: log_softmax_pos[b, y_labels[b]]
    y_idx = y_labels * OUT_DIM                       # offset into flattened (BLOCK_B, OUT_DIM)
    b_offs = tl.arange(0, BLOCK_B)

    ce_pos_val = -tl.sum(
        log_softmax_pos * (tl.arange(0, OUT_DIM)[None, :] == y_labels[:, None]).to(tl.float32),
        axis=1,
    )  # (BLOCK_B,)
    ce_neg_val = -tl.sum(
        log_softmax_neg * (tl.arange(0, OUT_DIM)[None, :] == y_labels[:, None]).to(tl.float32),
        axis=1,
    )  # (BLOCK_B,)

    # Zero out padding
    ce_pos_val = tl.where(mask_b, ce_pos_val, 0.0)
    ce_neg_val = tl.where(mask_b, ce_neg_val, 0.0)

    # Sum over the batch tile → scalar per (p, tile)
    partial_ce_pos = tl.sum(ce_pos_val)
    partial_ce_neg = tl.sum(ce_neg_val)

    # Write partial sums
    out_off = pid_p * N_TILES + pid_b
    tl.store(partial_ce_pos_ptr + out_off, partial_ce_pos)
    tl.store(partial_ce_neg_ptr + out_off, partial_ce_neg)


def fused_forward_fitness(base1, xB1_T, A1, w2, B2, A2, w3, B3, A3, sigma, y):
    """
    Fused forward pass + CE fitness computation.

    Args:
        base1:  (BATCH, HIDDEN)    bfloat16 — precomputed xb @ w1
        xB1_T:  (HALF_POP, BATCH)  bfloat16 — precomputed B1 @ xb.T (contiguous)
        A1:     (HALF_POP, HIDDEN) bfloat16
        w2:     (HIDDEN, HIDDEN)   bfloat16
        B2:     (HALF_POP, HIDDEN) bfloat16
        A2:     (HALF_POP, HIDDEN) bfloat16
        w3:     (HIDDEN, OUT_DIM)  bfloat16
        B3:     (HALF_POP, HIDDEN) bfloat16
        A3:     (HALF_POP, OUT_DIM) bfloat16
        sigma:  () float32 JAX scalar
        y:      (BATCH,) int32 labels
    Returns:
        fitness_pos: (HALF_POP,) float32  (negated mean CE — higher is better)
        fitness_neg: (HALF_POP,) float32
    """
    HALF_POP, BATCH = xB1_T.shape
    _, HIDDEN = base1.shape
    OUT_DIM = w3.shape[1]
    BLOCK_B = 16
    N_TILES = triton.cdiv(BATCH, BLOCK_B)

    T_val = jnp.float32(T)  # use module-level T constant

    out_shape = [
        jax.ShapeDtypeStruct((HALF_POP, N_TILES), jnp.float32),  # partial_ce_pos
        jax.ShapeDtypeStruct((HALF_POP, N_TILES), jnp.float32),  # partial_ce_neg
    ]

    grid = (HALF_POP, N_TILES)

    partial_ce_pos, partial_ce_neg = jt.triton_call(
        base1, xB1_T, A1,
        w2, B2, A2,
        w3, B3, A3,
        sigma,
        T_val,
        y.astype(jnp.int32),
        kernel=_fused_fwd_fitness_kernel,
        out_shape=out_shape,
        grid=grid,
        HALF_POP=HALF_POP,
        BATCH=BATCH,
        HIDDEN=HIDDEN,
        OUT_DIM=OUT_DIM,
        BLOCK_B=BLOCK_B,
        N_TILES=N_TILES,
    )

    # Reduce partial sums: sum over batch tiles → mean CE per p
    ce_pos = partial_ce_pos.sum(axis=1) / BATCH   # (HALF_POP,)
    ce_neg = partial_ce_neg.sum(axis=1) / BATCH   # (HALF_POP,)

    return -ce_pos, -ce_neg   # negate: lower CE = higher fitness

"""
Pallas (JAX-native) version of the fused 3-layer forward + CE kernel.

Eliminates jax-triton serialization overhead (~1s JIT savings).
Uses FP8 E4M3 tensor cores via pl.dot, same compute as the Triton version.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from functools import partial

HIDDEN = 128
OUT_DIM = 10
OUT_DIM_PAD = 16
BLOCK_B = 64
BLOCK_K = 32


def _pallas_3layer_ce_kernel(
    base1_ref,   # (BLOCK_B, HIDDEN) bf16
    xB1_T_ref,   # (1, BLOCK_B) bf16
    A1_ref,      # (1, HIDDEN) bf16
    w2_ref,      # (HIDDEN, HIDDEN) bf16
    B2_ref,      # (1, HIDDEN) bf16
    A2_ref,      # (1, HIDDEN) bf16
    w3_ref,      # (HIDDEN, OUT_DIM_PAD) bf16
    B3_ref,      # (1, HIDDEN) bf16
    A3_ref,      # (1, OUT_DIM_PAD) bf16
    sigma_ref,   # (1,) f32
    T_ref,       # (1,) f32
    y_ref,       # (BLOCK_B,) i32
    out_ref,     # (1, 1, 1) f32
):
    sign_idx = pl.program_id(2)
    sign = jnp.where(sign_idx == 0, 1.0, -1.0).astype(jnp.float32)

    sigma = sigma_ref[0]
    T_val = T_ref[0]
    sign_sigma = sign * sigma

    xB1_col = xB1_T_ref[0, :].astype(jnp.float32)

    # ── K-tiled L1 forward + L2 matmul ──────────────────────────
    base2 = jnp.zeros((BLOCK_B, HIDDEN), dtype=jnp.float32)
    xB2 = jnp.zeros((BLOCK_B,), dtype=jnp.float32)

    for k in range(0, HIDDEN, BLOCK_K):
        base1_k = base1_ref[:, k:k+BLOCK_K].astype(jnp.float32)
        A1_k = A1_ref[0, k:k+BLOCK_K].astype(jnp.float32)

        pre_act = base1_k + sign_sigma * xB1_col[:, None] * A1_k[None, :]
        l1_k = pre_act * jax.nn.sigmoid(1.702 * pre_act)

        w2_k = w2_ref[k:k+BLOCK_K, :]
        base2 = base2 + pl.dot(
            l1_k.astype(jnp.float8_e4m3fn),
            w2_k.astype(jnp.float8_e4m3fn),
        ).astype(jnp.float32)

        B2_k = B2_ref[0, k:k+BLOCK_K].astype(jnp.float32)
        xB2 = xB2 + jnp.sum(l1_k * B2_k[None, :], axis=1)

    # ── L2 activation ────────────────────────────────────────────
    A2_row = A2_ref[0, :].astype(jnp.float32)
    pre_act2 = base2 + sign_sigma * xB2[:, None] * A2_row[None, :]
    l2 = pre_act2 * jax.nn.sigmoid(1.702 * pre_act2)

    # ── Layer 3 (FP8) ────────────────────────────────────────────
    w3 = w3_ref[:, :]
    base3 = pl.dot(
        l2.astype(jnp.float8_e4m3fn),
        w3.astype(jnp.float8_e4m3fn),
    ).astype(jnp.float32)

    B3_row = B3_ref[0, :].astype(jnp.float32)
    A3_row = A3_ref[0, :OUT_DIM_PAD].astype(jnp.float32)
    xB3 = jnp.sum(l2 * B3_row[None, :], axis=1)
    logits = base3 + sign_sigma * xB3[:, None] * A3_row[None, :]

    # Mask padding dimensions
    pad_mask = jnp.arange(OUT_DIM_PAD) >= OUT_DIM
    logits = jnp.where(pad_mask[None, :], -1e9, logits)

    # ── CE loss ──────────────────────────────────────────────────
    y_labels = y_ref[:]
    scaled = logits / T_val
    max_val = jnp.max(scaled, axis=1, keepdims=True)
    exp_val = jnp.exp(scaled - max_val)
    log_sm = scaled - max_val - jnp.log(jnp.sum(exp_val, axis=1, keepdims=True))

    one_hot = (jnp.arange(OUT_DIM_PAD)[None, :] == y_labels[:, None]).astype(jnp.float32)
    ce = -jnp.sum(log_sm * one_hot, axis=1)

    out_ref[0, 0, 0] = jnp.sum(ce)


def pallas_3layer_ce_both(base1, xB1_T, A1, w2, B2, A2, w3, B3, A3, sigma, T_val, y):
    """Compute BOTH pos and neg CE using a Pallas kernel."""
    HALF_POP, BATCH = xB1_T.shape
    N_TILES = (BATCH + BLOCK_B - 1) // BLOCK_B

    w3_pad = jnp.pad(w3, [(0, 0), (0, OUT_DIM_PAD - OUT_DIM)])
    A3_pad = jnp.pad(A3, [(0, 0), (0, OUT_DIM_PAD - OUT_DIM)])

    sigma_arr = sigma.reshape(1)
    T_arr = T_val.reshape(1)
    y_i32 = y.astype(jnp.int32)

    grid = (HALF_POP, N_TILES, 2)

    out = pl.pallas_call(
        _pallas_3layer_ce_kernel,
        out_shape=jax.ShapeDtypeStruct((HALF_POP, N_TILES, 2), jnp.float32),
        grid=grid,
        in_specs=[
            pl.BlockSpec(block_shape=(BLOCK_B, HIDDEN), index_map=lambda p, b, s: (b * BLOCK_B, 0)),
            pl.BlockSpec(block_shape=(1, BLOCK_B), index_map=lambda p, b, s: (p, b * BLOCK_B)),
            pl.BlockSpec(block_shape=(1, HIDDEN), index_map=lambda p, b, s: (p, 0)),
            pl.BlockSpec(block_shape=(HIDDEN, HIDDEN), index_map=lambda p, b, s: (0, 0)),
            pl.BlockSpec(block_shape=(1, HIDDEN), index_map=lambda p, b, s: (p, 0)),
            pl.BlockSpec(block_shape=(1, HIDDEN), index_map=lambda p, b, s: (p, 0)),
            pl.BlockSpec(block_shape=(HIDDEN, OUT_DIM_PAD), index_map=lambda p, b, s: (0, 0)),
            pl.BlockSpec(block_shape=(1, HIDDEN), index_map=lambda p, b, s: (p, 0)),
            pl.BlockSpec(block_shape=(1, OUT_DIM_PAD), index_map=lambda p, b, s: (p, 0)),
            pl.BlockSpec(block_shape=(1,), index_map=lambda p, b, s: (0,)),
            pl.BlockSpec(block_shape=(1,), index_map=lambda p, b, s: (0,)),
            pl.BlockSpec(block_shape=(BLOCK_B,), index_map=lambda p, b, s: (b * BLOCK_B,)),
        ],
        out_specs=pl.BlockSpec(block_shape=(1, 1, 1), index_map=lambda p, b, s: (p, b, s)),
        compiler_params=pl.triton.CompilerParams(num_warps=4, num_stages=1),
    )(base1, xB1_T, A1, w2, B2, A2, w3_pad, B3, A3_pad, sigma_arr, T_arr, y_i32)

    # Split pos/neg from the combined output
    partial_ce_pos = out[:, :, 0]
    partial_ce_neg = out[:, :, 1]
    return partial_ce_pos, partial_ce_neg

"""Compare Pallas and Triton kernel outputs on the same inputs."""
import jax
import jax.numpy as jnp
from kernels.fused_3layer_ce import fused_3layer_ce_both as triton_ce
from kernels.pallas_3layer_ce import pallas_3layer_ce_both as pallas_ce

# Create test inputs
key = jax.random.PRNGKey(42)
HALF_POP, BATCH, HIDDEN = 100, 128, 128  # smaller for testing
OUT_DIM = 10

keys = jax.random.split(key, 10)
base1 = jax.random.normal(keys[0], (BATCH, HIDDEN), dtype=jnp.bfloat16)
xB1_T = jax.random.normal(keys[1], (HALF_POP, BATCH), dtype=jnp.bfloat16)
A1 = jax.random.normal(keys[2], (HALF_POP, HIDDEN), dtype=jnp.bfloat16)
w2 = jax.random.normal(keys[3], (HIDDEN, HIDDEN), dtype=jnp.bfloat16)
B2 = jax.random.normal(keys[4], (HALF_POP, HIDDEN), dtype=jnp.bfloat16)
A2 = jax.random.normal(keys[5], (HALF_POP, HIDDEN), dtype=jnp.bfloat16)
w3 = jax.random.normal(keys[6], (HIDDEN, OUT_DIM), dtype=jnp.bfloat16)
B3 = jax.random.normal(keys[7], (HALF_POP, HIDDEN), dtype=jnp.bfloat16)
A3 = jax.random.normal(keys[8], (HALF_POP, OUT_DIM), dtype=jnp.bfloat16)
sigma = jnp.float32(0.028)
T_val = jnp.float32(2.0)
y = jax.random.randint(keys[9], (BATCH,), 0, 10)

print("Running Triton kernel...")
t_pos, t_neg = triton_ce(base1, xB1_T, A1, w2, B2, A2, w3, B3, A3, sigma, T_val, y)
t_pos, t_neg = jax.block_until_ready(t_pos), jax.block_until_ready(t_neg)
print(f"  pos shape: {t_pos.shape}, sum: {t_pos.sum():.4f}")
print(f"  neg shape: {t_neg.shape}, sum: {t_neg.sum():.4f}")

print("\nRunning Pallas kernel...")
p_pos, p_neg = pallas_ce(base1, xB1_T, A1, w2, B2, A2, w3, B3, A3, sigma, T_val, y)
p_pos, p_neg = jax.block_until_ready(p_pos), jax.block_until_ready(p_neg)
print(f"  pos shape: {p_pos.shape}, sum: {p_pos.sum():.4f}")
print(f"  neg shape: {p_neg.shape}, sum: {p_neg.sum():.4f}")

print("\nComparison:")
print(f"  pos max diff: {jnp.max(jnp.abs(t_pos - p_pos)):.6f}")
print(f"  neg max diff: {jnp.max(jnp.abs(t_neg - p_neg)):.6f}")
print(f"  pos mean triton: {t_pos.mean():.6f}, pallas: {p_pos.mean():.6f}")
print(f"  neg mean triton: {t_neg.mean():.6f}, pallas: {p_neg.mean():.6f}")

# Also check if Pallas matches a pure-JAX reference
print("\nPure JAX reference:")
from mnist_eggroll_optimized import fast_gelu
base1_f32 = base1.astype(jnp.float32)
# Compute for first pop member, pos sign
xB1_col = xB1_T[0].astype(jnp.float32)
A1_0 = A1[0].astype(jnp.float32)
l1 = fast_gelu(base1_f32 + sigma * xB1_col[:, None] * A1_0[None, :])
base2 = l1.astype(jnp.bfloat16) @ w2.astype(jnp.bfloat16)
B2_0 = B2[0].astype(jnp.float32)
A2_0 = A2[0].astype(jnp.float32)
xB2 = (l1 * B2_0[None, :]).sum(axis=1)
pre_act2 = base2.astype(jnp.float32) + sigma * xB2[:, None] * A2_0[None, :]
l2 = fast_gelu(pre_act2)
w3_pad = jnp.pad(w3, [(0, 0), (0, 6)])
logits = (l2.astype(jnp.bfloat16) @ w3_pad.astype(jnp.bfloat16)).astype(jnp.float32)
B3_0 = B3[0].astype(jnp.float32)
A3_0 = jnp.pad(A3[0], (0, 6)).astype(jnp.float32)
xB3 = (l2 * B3_0[None, :]).sum(axis=1)
logits = logits + sigma * xB3[:, None] * A3_0[None, :]
logits = logits.at[:, 10:].set(-1e9)
scaled = logits / T_val
log_sm = scaled - jax.nn.logsumexp(scaled, axis=1, keepdims=True)
ce = -log_sm[jnp.arange(BATCH), y]
ref_ce = ce.sum()
print(f"  ref CE for pop[0] pos: {ref_ce:.4f}")
print(f"  triton CE for pop[0]: {t_pos[0].sum():.4f}")
print(f"  pallas CE for pop[0]: {p_pos[0].sum():.4f}")

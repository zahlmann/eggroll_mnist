# kernels/__init__.py
#
# Place custom Triton kernels here (or in separate modules inside this package).
# The training script (mnist_eggroll_optimized.py) can import from here.
#
# The core bottleneck to target is train_step_antithetic() in the training
# script. The key expensive operations are:
#
#   Layer 1 forward (both +/- perturbation):
#     base1  = xb @ w1                          shape: (batch=128, hidden=128)
#     xB1    = xb @ B1.T                        shape: (batch=128, half_pop=5000)
#     pert1  = xB1.T[:,:,None] * A1[:,None,:]   shape: (5000, 128, 128)  <-- HUGE intermediate
#     l1_pos = gelu(base1 + sigma * pert1)       shape: (5000, 128, 128)
#     l1_neg = gelu(base1 - sigma * pert1)       shape: (5000, 128, 128)
#
#   Same pattern for layer 2 (128->128) and layer 3 (128->10).
#
# A fused Triton kernel can compute l1_pos and l1_neg without materializing
# the full (5000, 128, 128) perturbation tensor in HBM — keeping tiles in SRAM.
#
# See cuda_kernels_docs/ for Triton tutorials and jax-triton integration docs.

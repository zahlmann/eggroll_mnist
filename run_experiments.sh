#!/bin/bash
set -e

if [ ! -f "mnist_prepped_float.npz" ]; then
    echo "Data not found. Running data prep..."
    uv run mnist_data_prep.py
fi

echo "=== EGGROLL ==="
uv run mnist_eggroll_optimized.py

echo ""
echo "=== Backprop (optimized) ==="
uv run mnist_backprop_optimized.py

echo ""
echo "=== Backprop (naive) ==="
uv run mnist_backprop.py

echo ""
echo "All experiments finished."

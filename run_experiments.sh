#!/bin/bash
set -e # Exit on error

echo "Checking data..."
if [ ! -f "mnist_prepped_float.npz" ]; then
    echo "Data not found. Running data prep..."
    uv run mnist_data_prep.py
else
    echo "Data already exists. Skipping data prep."
fi

echo "Running EGGROLL..."
uv run mnist_eggroll.py

echo "Running Backpropagation..."
uv run mnist_backprop.py

echo "All experiments finished."

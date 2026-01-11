#!/bin/bash
set -e # Exit on error

echo "Running EGGROLL..."
uv run mnist_eggroll.py

echo "Running Backpropagation..."
uv run mnist_backprop.py

echo "All experiments finished."

"""
benchmark.py — fast single-seed run for development iteration.

Use this during kernel development to get quick feedback before running the
full 3-seed validate.py. Seed is fixed at 11 (canonical baseline seed).

Usage: uv run benchmark.py
"""
import subprocess
import sys
import re
import time

BASELINE = {
    "test_accuracy": 0.9764,   # baseline with seed=11 on this machine
    "training_time_s": 66.5,   # ~66s baseline (note: README says 27s but that's without competing GPU process)
    "peak_memory_mb": 433.0,   # ~433MB baseline
}

TARGET_TIME_S = 5.0   # stretch goal: match backprop


def parse_block(output: str, block_name: str) -> dict:
    pattern = rf"=== {re.escape(block_name)} ===\n(.*?)==+\n"
    m = re.search(pattern, output, re.DOTALL)
    if not m:
        return {}
    result = {}
    for line in m.group(1).strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            result[k.strip()] = v.strip()
    return result


def main():
    print("Running benchmark (seed=11)...")
    t0 = time.perf_counter()
    result = subprocess.run(
        ["uv", "run", "mnist_eggroll_optimized.py", "--seed", "11"],
        capture_output=True, text=True
    )
    wall = time.perf_counter() - t0

    if result.returncode != 0:
        print("CRASHED")
        print(result.stdout[-1000:])
        print(result.stderr[-1000:])
        sys.exit(1)

    # Forward stdout so epoch-by-epoch progress is visible
    print(result.stdout)

    results = parse_block(result.stdout, "RESULTS")
    if not results:
        print("ERROR: could not parse RESULTS block")
        sys.exit(1)

    acc = float(results["test_accuracy"])
    time_s = float(results["training_time_s"])
    mem = float(results["peak_memory_mb"])

    speedup = BASELINE["training_time_s"] / time_s if time_s > 0 else 0

    print("=" * 50)
    print(f"  accuracy : {acc:.4f}  (baseline {BASELINE['test_accuracy']:.4f})")
    print(f"  time     : {time_s:.1f}s  ({speedup:.2f}x speedup,  target ≤{TARGET_TIME_S}s)")
    print(f"  memory   : {mem:.0f}MB  (baseline {BASELINE['peak_memory_mb']:.0f}MB)")
    print("=" * 50)

    if acc < 0.972:
        print("WARNING: accuracy below 97.2% threshold — validate.py will fail")
    if mem > 500:
        print("WARNING: memory above 500MB — validate.py will fail")
    if time_s <= TARGET_TIME_S:
        print(f"TARGET REACHED: {time_s:.1f}s ≤ {TARGET_TIME_S}s")


if __name__ == "__main__":
    main()

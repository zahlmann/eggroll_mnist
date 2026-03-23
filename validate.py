"""
validate.py — LOCKED. Do not modify.

Runs mnist_eggroll_optimized.py with 3 seeds, checks locked constants,
reports pass/fail. This is the ground truth for whether a kernel change is
acceptable.

Usage: uv run validate.py
"""
import subprocess
import sys
import re
import csv
import os
import time
from datetime import datetime

# These values are locked. If the training script prints different values the
# run counts as a CHEAT and is rejected regardless of accuracy.
REQUIRED = {
    "HIDDEN_DIM": 128,
    "BATCH_SIZE": 128,
    "EPOCHS": 10,
    "HALF_POPULATION": 3250,
    "T": 2.0,
}

SEEDS = [11, 42, 7]
MIN_ACCURACY = 0.972   # 97.2% — must hold on average across all seeds
MAX_MEMORY_MB = 500.0  # hard ceiling
RESULTS_TSV = "results.tsv"


def parse_block(output: str, block_name: str) -> dict:
    """Extract key: value pairs from a === BLOCK_NAME === ... ======= section."""
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


def run_seed(seed: int) -> dict | None:
    """Run the training script with the given seed. Returns parsed results or None on crash."""
    print(f"\n--- seed {seed} ---")
    t0 = time.perf_counter()
    result = subprocess.run(
        ["uv", "run", "mnist_eggroll_optimized.py", "--seed", str(seed)],
        capture_output=True, text=True
    )
    elapsed = time.perf_counter() - t0

    combined = result.stdout + result.stderr
    if result.returncode != 0:
        print(f"  CRASHED (exit {result.returncode})")
        print(result.stdout[-800:] if result.stdout else "")
        print(result.stderr[-800:] if result.stderr else "")
        return None

    constants = parse_block(result.stdout, "CONSTANTS")
    results = parse_block(result.stdout, "RESULTS")

    if not constants or not results:
        print("  ERROR: could not parse CONSTANTS or RESULTS block from output")
        print(result.stdout[-500:])
        return None

    # Verify locked constants
    violations = []
    for key, expected in REQUIRED.items():
        actual_raw = constants.get(key)
        if actual_raw is None:
            violations.append(f"  {key} missing from CONSTANTS block")
            continue
        actual = float(actual_raw)
        if abs(actual - expected) > 1e-6:
            violations.append(f"  {key}: expected {expected}, got {actual}")
    if violations:
        print("  CHEAT DETECTED — locked constants were changed:")
        for v in violations:
            print(v)
        return None

    parsed = {
        "seed": seed,
        "test_accuracy": float(results["test_accuracy"]),
        "training_time_s": float(results["training_time_s"]),
        "peak_memory_mb": float(results["peak_memory_mb"]),
    }
    print(f"  accuracy={parsed['test_accuracy']:.4f}  time={parsed['training_time_s']:.1f}s  "
          f"mem={parsed['peak_memory_mb']:.0f}MB")
    return parsed


def write_tsv(row: dict):
    """Append one row to results.tsv (tab-separated, never committed to git)."""
    fieldnames = ["timestamp", "commit", "seed", "test_accuracy",
                  "training_time_s", "peak_memory_mb", "status", "description"]
    exists = os.path.exists(RESULTS_TSV)
    with open(RESULTS_TSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t",
                                extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def get_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True)
        return r.stdout.strip()
    except Exception:
        return "unknown"


def main():
    print("=" * 60)
    print("validate.py — running 3-seed evaluation")
    print("=" * 60)

    commit = get_commit()
    runs = []
    for seed in SEEDS:
        r = run_seed(seed)
        if r is None:
            # crashed or cheated — log and exit
            write_tsv({
                "timestamp": datetime.now().isoformat(),
                "commit": commit,
                "seed": seed,
                "test_accuracy": 0.0,
                "training_time_s": 0.0,
                "peak_memory_mb": 0.0,
                "status": "crash",
                "description": "crashed or cheat detected",
            })
            print("\nFAIL — aborting validation")
            sys.exit(1)
        runs.append(r)

    avg_acc = sum(r["test_accuracy"] for r in runs) / len(runs)
    avg_time = sum(r["training_time_s"] for r in runs) / len(runs)
    avg_mem = sum(r["peak_memory_mb"] for r in runs) / len(runs)
    max_mem = max(r["peak_memory_mb"] for r in runs)

    print("\n" + "=" * 60)
    print(f"  avg accuracy : {avg_acc:.4f}  (need ≥ {MIN_ACCURACY:.3f})")
    print(f"  avg time     : {avg_time:.1f}s")
    print(f"  avg memory   : {avg_mem:.0f}MB  max={max_mem:.0f}MB  (limit {MAX_MEMORY_MB:.0f}MB)")
    print("=" * 60)

    acc_pass = avg_acc >= MIN_ACCURACY
    mem_pass = max_mem <= MAX_MEMORY_MB
    passed = acc_pass and mem_pass

    if not acc_pass:
        print(f"FAIL — accuracy {avg_acc:.4f} < {MIN_ACCURACY}")
    if not mem_pass:
        print(f"FAIL — peak memory {max_mem:.0f}MB > {MAX_MEMORY_MB:.0f}MB")
    if passed:
        print("PASS")

    status = "keep" if passed else "discard"
    description = f"acc={avg_acc:.4f} time={avg_time:.1f}s mem={avg_mem:.0f}MB"

    for r in runs:
        write_tsv({
            "timestamp": datetime.now().isoformat(),
            "commit": commit,
            **r,
            "status": status,
            "description": description,
        })

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

"""
run_pipeline.py — Execute project notebooks in order.

Usage
-----
Run all 5 notebooks:
    python run_pipeline.py

Run only specific notebooks (by number):
    python run_pipeline.py --only 5
    python run_pipeline.py --only 3 4 5

Start from a specific notebook (skip earlier ones):
    python run_pipeline.py --from 3

Dry run (show what would run, no execution):
    python run_pipeline.py --dry-run
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

NOTEBOOKS = [
    ("01", "notebooks/01_data_exploration.ipynb",    "EDA"),
    ("02", "notebooks/02_preprocessing.ipynb",        "Text Preprocessing"),
    ("03", "notebooks/03_classic_ml_models.ipynb",    "Classic ML Models"),
    ("04", "notebooks/04_deep_learning_model.ipynb",  "Deep Learning (LSTM)"),
    ("05", "notebooks/05_model_comparison.ipynb",     "Model Comparison"),
]

# Per-notebook timeout in seconds. -1 = no limit.
# NB02 (preprocessing) and NB04 (LSTM) are slow on CPU.
TIMEOUTS = {
    "01": 300,    # EDA: 5 min
    "02": 1800,   # Preprocessing: 30 min (lemmatization is slow)
    "03": 600,    # Classic ML: 10 min
    "04": 7200,   # LSTM training: up to 2 h on CPU
    "05": 3600,   # Comparison (BERT inference on CPU): up to 1 h
}


def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s"


def run_notebook(num: str, path: str, label: str, dry_run: bool) -> bool:
    nb_path = Path(path)
    if not nb_path.exists():
        print(f"  [ERROR] File not found: {nb_path}")
        return False

    timeout = TIMEOUTS.get(num, 3600)
    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        f"--ExecutePreprocessor.timeout={timeout}",
        "--ExecutePreprocessor.kernel_name=python3",
        str(nb_path),
    ]

    if dry_run:
        print(f"  [DRY RUN] Would execute: {nb_path.name}")
        return True

    print(f"  Running ... (timeout: {fmt_duration(timeout)})")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    if result.returncode == 0:
        print(f"  Done in {fmt_duration(elapsed)}")
        return True
    else:
        print(f"  [FAILED] after {fmt_duration(elapsed)}")
        # Print the last 20 lines of stderr for diagnostics
        stderr_lines = result.stderr.strip().splitlines()
        for line in stderr_lines[-20:]:
            print(f"    {line}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run IMDB project notebooks in order.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--only", nargs="+", metavar="N",
        help="Run only these notebook numbers (e.g. --only 3 4 5)"
    )
    group.add_argument(
        "--from", dest="start_from", metavar="N", type=str,
        help="Start from this notebook number (skip earlier ones)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would run without executing anything"
    )
    args = parser.parse_args()

    # Normalize input numbers to zero-padded strings ("5" -> "05")
    def norm(n: str) -> str:
        return n.zfill(2)

    # Filter notebook list
    if args.only:
        only_set = {norm(n) for n in args.only}
        selected = [nb for nb in NOTEBOOKS if nb[0] in only_set]
    elif args.start_from:
        threshold = norm(args.start_from)
        selected = [nb for nb in NOTEBOOKS if nb[0] >= threshold]
    else:
        selected = NOTEBOOKS

    if not selected:
        print("No notebooks matched the filter. Check --only / --from values.")
        sys.exit(1)

    total = len(selected)
    print(f"\nIMDB Sentiment Analysis - Pipeline Runner")
    print(f"{'DRY RUN - ' if args.dry_run else ''}Running {total} notebook(s):\n")

    pipeline_start = time.perf_counter()
    failed_at = None

    for i, (num, path, label) in enumerate(selected, 1):
        print(f"[{i}/{total}] NB{num}: {label}")
        ok = run_notebook(num, path, label, args.dry_run)
        if not ok:
            failed_at = f"NB{num}: {label}"
            print(f"\n  Pipeline stopped at {failed_at}.")
            print(f"  Fix the error above, then re-run with: python run_pipeline.py --from {num}")
            break
        if i < total:
            print()

    pipeline_elapsed = time.perf_counter() - pipeline_start
    print(f"\n{'-' * 50}")
    if failed_at:
        print(f"Pipeline FAILED at: {failed_at}")
    else:
        print(f"All {total} notebook(s) completed successfully.")
    print(f"Total time: {fmt_duration(pipeline_elapsed)}")


if __name__ == "__main__":
    main()

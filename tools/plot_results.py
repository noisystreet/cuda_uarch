#!/usr/bin/env python3
"""
plot_results.py — Visualise microbenchmark CSV output

Usage:
    ./benchmark_binary 2>&1 | tee results.csv
    python3 plot_results.py results.csv
"""

import argparse
import re
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install numpy pandas matplotlib")
    sys.exit(1)


def parse_results(filepath: Path) -> pd.DataFrame:
    """Parse RESULT lines from benchmark output."""
    records = []
    pattern = re.compile(
        r"^RESULT,(?P<label>.+?),(?P<median>[\d.]+),(?P<mean>[\d.]+),"
        r"(?P<min>[\d.]+),(?P<max>[\d.]+),(?P<stddev>[\d.]+),(?P<count>[\d.]+)"
    )

    with open(filepath) as f:
        for line in f:
            m = pattern.match(line.strip())
            if m:
                records.append(m.groupdict())

    if not records:
        print("No RESULT lines found. Did you pipe benchmark output?")
        sys.exit(1)

    df = pd.DataFrame(records)
    for col in ["median", "mean", "min", "max", "stddev", "count"]:
        df[col] = pd.to_numeric(df[col])
    return df


def plot_latency(df: pd.DataFrame, output_dir: Path):
    """Plot instruction latency results."""
    lat_rows = df[df["label"].str.contains("chain_len", case=False)]
    if lat_rows.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ops = lat_rows["label"].str.extract(r"(\w+) chain")
    lat_rows = lat_rows.copy()
    lat_rows["op"] = ops[0]

    bars = ax.bar(lat_rows["op"], lat_rows["median"], yerr=lat_rows["stddev"],
                  capsize=5, color="steelblue")
    ax.set_ylabel("Time per operation (us)")
    ax.set_title("Instruction Latency (Dependency Chain)")
    ax.bar_label(bars, fmt="%.2f us")

    fig.tight_layout()
    out = output_dir / "instruction_latency.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


def plot_throughput(df: pd.DataFrame, output_dir: Path):
    """Plot instruction throughput vs unroll factor."""
    tp_rows = df[df["label"].str.contains("FFMA unroll", case=False)]
    if tp_rows.empty:
        return

    tp_rows = tp_rows.copy()
    tp_rows["unroll"] = tp_rows["label"].str.extract(r"unroll=(\d+)").astype(int)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tp_rows["unroll"], tp_rows["median"], "o-", linewidth=2,
            markersize=8)
    ax.set_xlabel("Unroll Factor (ILP)")
    ax.set_ylabel("Time (us)")
    ax.set_title("FFMA Throughput vs. Instruction-Level Parallelism")
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = output_dir / "throughput_vs_unroll.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


def plot_memory_bw(df: pd.DataFrame, output_dir: Path):
    """Plot memory bandwidth vs array size."""
    bw_rows = df[df["label"].str.contains("_bw", case=False)]
    if bw_rows.empty:
        return

    bw_rows = bw_rows.copy()
    bw_rows["size_mb"] = (
        bw_rows["label"].str.extract(r"size=(\d+)").astype(float) * 4 / 1e6
    )
    bw_rows["type"] = bw_rows["label"].str.extract(r"(read_bw|write_bw|copy_bw)")

    fig, ax = plt.subplots(figsize=(10, 6))
    for bw_type, group in bw_rows.groupby("type"):
        group = group.sort_values("size_mb")
        # Convert time to bandwidth
        bytes_per_elem = 4
        if bw_type == "copy_bw":
            bytes_per_elem = 8
        bw = (group["size_mb"] * 1e6) / (group["median"] * 1e-6) / 1e9
        ax.plot(group["size_mb"], bw, "o-", label=bw_type, linewidth=2)

    ax.set_xlabel("Array Size (MB)")
    ax.set_ylabel("Bandwidth (GB/s)")
    ax.set_title("Global Memory Bandwidth vs. Array Size")
    ax.legend()
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = output_dir / "memory_bandwidth.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot GPU microbenchmark results")
    parser.add_argument("input", type=Path, help="CSV results file")
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("reports/figures"),
                        help="Output directory for figures")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = parse_results(args.input)
    print(f"Parsed {len(df)} result lines")

    plot_latency(df, args.output_dir)
    plot_throughput(df, args.output_dir)
    plot_memory_bw(df, args.output_dir)

    print("\nDone. Figures saved to", args.output_dir)


if __name__ == "__main__":
    main()

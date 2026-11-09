#!/usr/bin/env python3
"""Generate reproducible clustered vectors for ANN testing (stdlib only).

Output is a headerless CSV: id,float,float,... . Synthetic data are *not* image embeddings.
"""
import argparse
import csv
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--dims", type=int, default=16)
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="data/synthetic")
    args = parser.parse_args()
    if min(args.count, args.dims, args.clusters, args.queries) < 1:
        parser.error("count, dims, clusters, and queries must all be positive")
    rng = random.Random(args.seed)
    centers = [[rng.uniform(-1.0, 1.0) for _ in range(args.dims)] for _ in range(args.clusters)]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for filename, n, prefix in (("vectors.csv", args.count, "vec"), ("queries.csv", args.queries, "query")):
        with (output / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            for i in range(n):
                center = centers[i % args.clusters]
                vector = [round(x + rng.gauss(0, 0.12), 7) for x in center]
                writer.writerow([f"{prefix}_{i:08d}", *vector])
    print(f"Generated {args.count} vectors and {args.queries} queries in {output}")


if __name__ == "__main__":
    main()

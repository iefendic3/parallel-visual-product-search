# Benchmark protocol

## Experiment matrix

Vary the following independently:

- Dataset sizes: 10k, 100k, 500k, 1M vectors where hardware permits.
- Dimensions: 128 and (later) model-specific image embedding dimensionality.
- Clusters (`nlist`): e.g. 16, 64, 256 (avoid overclaiming optimality).
- Probed clusters (`nprobe`): 1, 2, 4, 8, 16, `nlist` (bounded by chosen `nlist`).
- Top-k: 1, 10, 20.
- OpenMP worker counts: 1, 2, 4, 8, etc., limited by available hardware.
- CUDA threads/block and query batch size in a later milestone.

## Measurements

1. **Index construction time**: measure independently of query search and disk I/O.
2. **Search latency**: measure in-memory search with a prebuilt index. Report median, p95, and p99 from many queries, not just a single average.
3. **Throughput**: queries per second with a stated concurrency and batch size.
4. **Recall@k**: proportion of exact top-k IDs recovered by the ANN result.
5. **Speedup**: `T(sequential IVF) / T(parallel IVF)` for the **same** dataset, index, query set, nprobe, top-k, and hardware environment.
6. **End-to-end application latency** separately: includes image inference, overhead, lookup, and UI/API transport.

## Fair-comparison rules

- Use the **same saved centroids and inverted lists** across sequential/OpenMP/CUDA implementations.
- Keep the same distance metric and top-k tie-breaking rules.
- Warm up the process and report whether the GPU data already resides in device memory.
- Avoid counting CSV parsing, index loading, logging, and rendering in pure search kernel times.
- Run multiple trials; record CPU/GPU model, compiler, compiler flags, OS, dataset size, dimensions, and seed.
- Avoid measuring debug builds; use Release and record compiler optimization level.
- SIFT1M ground truth and brute-force exact results can both support recall evaluation, but document the chosen reference.

The current `ann_cli evaluate` command reports mean query times and mean recall only. It is suitable for early development, **not yet final research-grade benchmarking**. We will add warmups, percentiles, and exportable CSV later.

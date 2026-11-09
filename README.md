# Parallel Visual Product Search (P24)

University project for **Paralelni računarski sistemi**: find visually similar products using a **custom approximate nearest-neighbor (ANN) search engine**. The eventual application will use product images and metadata from the **Amazon Berkeley Objects (ABO)** dataset.

## Current milestone: sequential C++ engine

This repository currently contains a **working, dependency-free C++17 IVF-Flat reference implementation**:

- Squared Euclidean (L2) distance between float vectors.
- Exact brute-force top-k search as a correctness baseline.
- K-means++ initialization + sequential Lloyd iterations to train centroids.
- IVF-Flat index construction by assigning **all** database vectors to their nearest centroid.
- Approximate top-k search by selecting the `nprobe` nearest centroids and scanning the corresponding inverted lists.
- A fixed, versioned binary index format so training isn't repeated on each query.
- CLI evaluation of `Recall@k`, mean search time, and the ratio of exact to ANN search time.
- Automated tests and small demonstration vectors.

**Not implemented yet:** Amazon dataset parsing, image embeddings, application UI, OpenMP, and CUDA. The supplied example vectors are synthetic placeholders, **not** actual image embeddings. All timings are machine-dependent; no speedup claim is made.

## Build and test

Requirements: a C++17 compiler and CMake 3.16+. No third-party libraries are required for the first milestone.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

## Run the complete demo

```bash
# 1. Build a reusable index
./build/ann_cli build \
  --input examples/sample_vectors.csv \
  --output build/sample.ivf \
  --clusters 3 --iterations 25 --train-size 0 --seed 42

# 2. Search with a query vector
./build/ann_cli search \
  --index build/sample.ivf \
  --query 0.11,0.12 --topk 3 --nprobe 1

# 3. Evaluate recall and speed against exact search
./build/ann_cli evaluate \
  --index build/sample.ivf \
  --queries examples/sample_queries.csv \
  --topk 3 --nprobe 2
```

Note: CLI `search_us` times only the in-memory query, **not** loading the index from disk. For high-quality benchmarks, run repeated warm searches within one process and report distributions, not only one measurement.

## Data contract for the core engine

The current CLI accepts a simple, **headerless** CSV with one ID and one fixed-length float vector per row:

```csv
image_001,0.24,0.58,0.11
image_002,0.30,0.62,0.09
image_003,0.85,0.12,0.77
```

The engine preserves IDs and returns them alongside distance scores. This is how the future app will map neighbors back to image paths, product names, and metadata.

For real images, a separate embedding pipeline will:

1. Download an approved subset of ABO catalog images + metadata.
2. Use a pretrained vision model to convert each image into a fixed-length vector.
3. Normalize vectors if cosine-equivalent squared-L2 comparison is desired.
4. Export the headerless vector CSV above (or a more efficient binary format later).
5. Build the IVF index **once**, then load and reuse it for search requests.

The uploaded user's image must be embedded with **the same model and preprocessing configuration** as the catalog images. The C++ engine does not compare raw pixels or recover original images from embeddings.

## Repository structure

```text
.
├── CMakeLists.txt
├── include/ann/ivf_flat.hpp      # Public API: Dataset, exact search, IVF index
├── src/ivf_flat.cpp              # Sequential k-means++ / IVF-Flat implementation
├── src/main.cpp                  # Build, search, evaluate CLI
├── tests/test_ann.cpp            # Correctness and persistence tests
├── examples/                     # Small, synthetic input vectors
├── scripts/generate_synthetic.py # Deterministic larger datasets for development
└── docs/
    ├── ARCHITECTURE.md           # Algorithm details and scope
    ├── RESEARCH.md               # Scientific papers and open-access links
    ├── RELATED_REPOS.md          # Relevant GitHub implementations
    ├── DATASETS.md               # ABO, SIFT1M, attribution and data workflow
    ├── BENCHMARKS.md             # Fair evaluation methodology
    ├── ROADMAP.md                # Sequential -> OpenMP -> CUDA -> app
    └── PUBLISHING.md             # Creating the public GitHub repository
```

## Important design decisions

- **IVF-Flat, not IVF-PQ:** Candidate vectors are stored as full-precision floats, so scanning *all* clusters with `nprobe == nlist` gives the same top-k results as exhaustive L2 search, apart from floating-point/tie-handling issues.
- **Training sample:** K-means runs on up to `--train-size` vectors; all input vectors are then indexed. Use `--train-size 0` to train on the full dataset.
- **Top-k:** a size-k heap avoids sorting every scanned candidate.
- **Index files:** current `.ivf` format uses native integer/float byte order; it is a local development format, **not yet cross-platform portable**. Don't load untrusted index files.
- **Simplified CSV:** no header, quoted fields, or escaped commas; IDs must be unique and cannot contain commas. Data engineering comes in a later milestone.
- **Caution about latency:** index loading and image embedding inference are not included in the search benchmarks; report end-to-end app latency separately when the UI exists.

## Research and inspiration

See [docs/RESEARCH.md](docs/RESEARCH.md) and [docs/RELATED_REPOS.md](docs/RELATED_REPOS.md). References are for study and benchmarking; the ANN logic in this repository is implemented by the project team, **not delegated to FAISS**.

## Dataset credits

Amazon Berkeley Objects (ABO) by Amazon/Berkeley: <https://registry.opendata.aws/amazon-berkeley-objects/>. Dataset licensing: **CC BY 4.0** (updated in 2023); see [docs/DATASETS.md](docs/DATASETS.md) for attribution requirements. Dataset images and generated embeddings are not committed to this repository.

## Source license

MIT for this repository's original source code; this does **not** change the license of third-party datasets, papers, or embedding models.

## Publish on GitHub

See [docs/PUBLISHING.md](docs/PUBLISHING.md) for `gh` or GitHub website instructions.

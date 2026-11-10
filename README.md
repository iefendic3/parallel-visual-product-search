# Parallel Visual Product Search (P24)

University project for **Paralelni računarski sistemi**: find visually similar products using a **custom approximate nearest-neighbor (ANN) search engine**. The eventual application will use product images and metadata from the **Amazon Berkeley Objects (ABO)** dataset.

## Current milestone: sequential engine and local image pipeline

This repository currently contains a **working, dependency-free C++17 IVF-Flat reference implementation**:

- Squared Euclidean (L2) distance between float vectors.
- Exact brute-force top-k search as a correctness baseline.
- K-means++ initialization + sequential Lloyd iterations to train centroids.
- IVF-Flat index construction by assigning **all** database vectors to their nearest centroid.
- Approximate top-k search by selecting the `nprobe` nearest centroids and scanning the corresponding inverted lists.
- A fixed, versioned binary index format so training isn't repeated on each query.
- CLI evaluation of `Recall@k`, mean search time, and the ratio of exact to ANN search time.
- Automated tests and small demonstration vectors.

A local ABO pipeline prepares product/image metadata, generates normalized OpenCLIP embeddings in resumable batches, and searches using an external image.

**Not implemented yet:** application UI, OpenMP, and CUDA. The supplied example vectors are synthetic placeholders, **not** actual image embeddings. All timings are machine-dependent; no speedup claim is made.

## Build and test

The C++ engine requires a C++17 compiler, with optional CMake 3.16+. It has no third-party dependencies. The image pipeline separately requires Python 3.10+ and [requirements-embeddings.txt](requirements-embeddings.txt).

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Build without CMake on macOS:

```bash
mkdir -p build
clang++ -std=c++17 -O2 -Iinclude src/ivf_flat.cpp src/main.cpp -o build/ann_cli
clang++ -std=c++17 -O2 -Iinclude src/ivf_flat.cpp tests/test_ann.cpp -o build/ann_tests
./build/ann_tests
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

## Search local ABO images

The downloaded archives are already extracted under `data/abo/images/` and
`data/abo/listings/`. No dataset download is needed. Start with a small subset:

```bash
python3 scripts/prepare_abo_manifest.py --limit 1000
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-embeddings.txt
python scripts/embed_abo.py --manifest data/abo/manifest.csv --output data/abo/vectors.csv --limit 1000 --batch-size 16
./build/ann_cli build --input data/abo/vectors.csv --output build/abo.ivf --clusters 16 --iterations 20 --train-size 0
python scripts/search_image.py --image ~/Downloads/test-product.jpg --index build/abo.ivf --manifest data/abo/manifest.csv --topk 10 --nprobe 4
```

The first model load may download pretrained weights. Later runs use the cache.
On Apple Silicon, `--device auto` uses MPS when available; `--device cpu` provides
a fallback. Repeating the embedding command resumes completed batches. Add
`--deduplicate-products` for best-effort distinct products and `--format json`
for structured search output.

See [docs/ABO_PIPELINE.md](docs/ABO_PIPELINE.md) for the inspected metadata format,
manifest IDs and paths, checkpoint rules, full-dataset scaling, and MPS troubleshooting.
The C++ engine remains unchanged; the supplied `examples/` vectors are synthetic.

## Run the synthetic demo

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

The engine preserves IDs and returns them alongside distance scores. The image-search script uses these IDs to resolve image paths, product names, and metadata from the manifest.

For real images, the local pipeline:

1. Joins extracted listing JSON Lines to image metadata and existing local paths.
2. Gives each listing/image association a unique, stable `image_id`.
3. Uses pretrained OpenCLIP `ViT-B-32` / `laion2b_s34b_b79k` to generate 512D float32 vectors.
4. L2-normalizes them and exports headerless vector CSV with the manifest IDs.
5. Builds the IVF index once, then loads it for image searches.

The uploaded user's image must be embedded with **the same model and preprocessing configuration** as the catalog images. The C++ engine does not compare raw pixels or recover original images from embeddings.

## Repository structure

```text
.
├── CMakeLists.txt
├── include/ann/ivf_flat.hpp      # Public API: Dataset, exact search, IVF index
├── src/ivf_flat.cpp              # Sequential k-means++ / IVF-Flat implementation
├── src/main.cpp                  # Build, search, evaluate CLI
├── tests/test_ann.cpp            # Correctness and persistence tests
├── tests/test_abo_pipeline.py     # Metadata, embedding, resume, and C++ integration tests
├── examples/                     # Small, synthetic input vectors
├── scripts/generate_synthetic.py # Deterministic larger datasets for development
├── scripts/prepare_abo_manifest.py # Local listings/image join
├── scripts/embed_abo.py          # Resumable catalog embedding
├── scripts/search_image.py       # Image query -> C++ engine -> metadata
├── scripts/clip_embeddings.py    # Shared catalog/query model and preprocessing
├── scripts/abo_common.py         # Streaming manifest and atomic file helpers
├── requirements-embeddings.txt   # Python image-pipeline dependencies
└── docs/
    ├── ARCHITECTURE.md           # Algorithm details and scope
    ├── RESEARCH.md               # Scientific papers and open-access links
    ├── RELATED_REPOS.md          # Relevant GitHub implementations
    ├── DATASETS.md               # ABO, SIFT1M, attribution and data workflow
    ├── BENCHMARKS.md             # Fair evaluation methodology
    ├── ROADMAP.md                # Sequential -> OpenMP -> CUDA -> app
    └── ABO_PIPELINE.md            # Complete local image-search workflow
```

## Important design decisions

- **IVF-Flat, not IVF-PQ:** Candidate vectors are stored as full-precision floats, so scanning *all* clusters with `nprobe == nlist` gives the same top-k results as exhaustive L2 search, apart from floating-point/tie-handling issues.
- **Training sample:** K-means runs on up to `--train-size` vectors; all input vectors are then indexed. Use `--train-size 0` to train on the full dataset.
- **Top-k:** a size-k heap avoids sorting every scanned candidate.
- **Index files:** current `.ivf` format uses native integer/float byte order; it is a local development format, **not yet cross-platform portable**. Don't load untrusted index files.
- **Simplified CSV:** no header, quoted fields, or escaped commas; IDs must be unique and cannot contain commas. The image pipeline keeps quoted titles and paths in a separate manifest.
- **Caution about latency:** index loading and image embedding inference are not included in the search benchmarks; report end-to-end app latency separately when the UI exists.

## Research and inspiration

See [docs/RESEARCH.md](docs/RESEARCH.md) and [docs/RELATED_REPOS.md](docs/RELATED_REPOS.md). References are for study and benchmarking; the ANN logic in this repository is implemented by the project team, **not delegated to FAISS**.

## Dataset credits

Amazon Berkeley Objects (ABO) by Amazon/Berkeley: <https://registry.opendata.aws/amazon-berkeley-objects/>. Dataset licensing: **CC BY 4.0** (updated in 2023); see [docs/DATASETS.md](docs/DATASETS.md) for attribution requirements. Dataset images and generated embeddings are not committed to this repository.

## Source license

MIT for this repository's original source code; this does **not** change the license of third-party datasets, papers, or embedding models.

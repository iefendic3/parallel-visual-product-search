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

The local pipeline keeps two manifests: **`manifest-full.csv` preserves every
image/product relationship**, while **`unique-images.csv` has one row per physical
image path**, with its native ABO ID as the vector ID. The inspected catalog has
710,650 relationships and 398,170 unique images. Images shared by products are
embedded once and resolved back to all listings during search.

Install the Python dependencies as needed:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-embeddings.txt
```

Prepare the relationship manifest only if it does not already exist:

```bash
python scripts/prepare_abo_manifest.py --output data/abo/manifest-full.csv
```

Then deduplicate, generate vectors, build, and search (run embedding/index building
when ready; they can take time):

```bash
python scripts/deduplicate_abo_manifest.py \
  --manifest data/abo/manifest-full.csv --output data/abo/unique-images.csv

python scripts/embed_abo.py \
  --manifest data/abo/unique-images.csv --output data/abo/vectors-full.csv \
  --batch-size 16 --device auto

./build/ann_cli build \
  --input data/abo/vectors-full.csv --output build/abo-full.ivf \
  --clusters 16 --iterations 20 --train-size 20000 --seed 42

python scripts/search_image.py \
  --image ~/Downloads/test-product.jpg --index build/abo-full.ivf \
  --manifest data/abo/manifest-full.csv --topk 10 --nprobe 4 \
  --deduplicate-products --format json
```

For a development run, add `--limit 1000` to the embedding command. Repeating it
resumes; removing the limit later extends the same checkpoint. To reuse compatible
embeddings from the previous workflow, add these arguments to `embed_abo.py`:

```bash
--reuse-checkpoint data/abo/vectors.csv.checkpoint.sqlite3 \
--reuse-manifest data/abo/manifest.csv
```

The importer validates model settings, package versions, source-manifest hash,
image IDs/paths, file stamps, and vectors before copying cached embeddings under
native IDs. It leaves the old checkpoint and CSV intact. Plain CSVs with no
checkpoint provenance are not imported. The previous `manifest.csv` → `vectors.csv`
→ `abo.ivf` workflow still works; search it using its original `manifest.csv`.

Search expands each unique image into associated listings from the **full**
manifest. Product deduplication keeps one result per `product_id`, including
separate products that share an image. If necessary it doubles the initial ANN
candidate count until enough products are found, the probed lists are exhausted,
or `--max-candidates` (default 10,000) is reached. JSON includes all associated
listings under `image_matches`, plus at most `topk` ranked entries under `matches`.

The first model load may download pretrained weights. On Apple Silicon,
`--device auto` uses MPS when available; `--device cpu` provides a fallback.
See [docs/ABO_PIPELINE.md](docs/ABO_PIPELINE.md) for validation, resumability,
relationship ranking, scaling, and attribution. Dataset assets and generated
artifacts are ignored by Git. The C++ search algorithm is unchanged.

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

## Reproducible held-out evaluation split

Reserve 500 randomly selected image IDs for evaluation, using seed 42:

```bash
python3 scripts/create_evaluation_split.py \
  --input data/abo/vectors-full.csv \
  --train-output data/abo/vectors-eval-train.csv \
  --queries-output data/abo/queries-eval.csv \
  --query-count 500 --seed 42
```

These are also the default paths, query count, and seed. For an input containing
398,170 unique vectors, the outputs contain **397,670 index vectors** and **500
held-out query vectors**, with **zero overlapping image IDs**. When later building
an evaluation index, use `vectors-eval-train.csv` and a new index filename such as
`build/abo-eval.ivf`. The existing `abo-full.ivf` contains the held-out images and
should not be used to measure this split's held-out recall. The splitter never
builds indexes, generates embeddings, or runs evaluations.

The script validates headerless rows with 512 finite, float32-range coordinates
and unique IDs, storing only IDs/partition flags in a temporary SQLite database
on disk. After validation, it samples distinct row positions without replacement
and copies each original row **byte for byte** into the selected output. Float
spelling, precision, delimiters, line endings, and source ordering within each
output are preserved. No normalization or float reserialization takes place.
RAM use consists of the selected IDs, bounded database buffers, and individual
rows; the full embedding matrix is never loaded. Allow roughly one input file's
size of additional disk space, plus the temporary ID database.

Two streaming input passes verify that the source bytes stayed unchanged. A
verification pass over the staged outputs checks byte hashes, unique IDs, exact
counts, complete source-ID coverage, and partition membership before publishing.
Existing output CSVs or metadata files are refused, including concurrent file
creation; the input and existing index/query files are never overwritten. To
repeat a split, choose fresh output filenames. Identical input/order, seed, query
count, and Python version reproduce the same CSV contents.

`queries-eval.csv.meta.json` records input/output SHA-256 hashes, counts, dimensions,
seed, Python version, and the selected query IDs. This is an **image-ID** split;
other images of the same product may remain in the index set.

Independently verify the counts without loading vectors into memory (also handles
a final row without a newline):

```bash
python3 - <<'PY'
from pathlib import Path

expected = {
    "data/abo/vectors-full.csv": 398170,
    "data/abo/vectors-eval-train.csv": 397670,
    "data/abo/queries-eval.csv": 500,
}
for name, wanted in expected.items():
    with Path(name).open("rb") as stream:
        actual = sum(1 for _ in stream)
    print(f"{name}: {actual:,} vectors")
    if actual != wanted:
        raise SystemExit(f"Expected {wanted:,}, found {actual:,}")
PY
```

Run the automated split tests using small local fixtures:

```bash
python3 -m unittest discover -s tests -p 'test_evaluation_split.py' -v
```

## CPU profiling and sustained benchmarks

Compile an optimized macOS executable with debug symbols and frame pointers for
Apple Instruments:

```bash
mkdir -p build
clang++ -std=c++17 -O2 -g -fno-omit-frame-pointer -Iinclude \
  src/ivf_flat.cpp src/main.cpp -o build/ann_cli
```

Profile ANN and brute-force search separately using the same index and query set:

```bash
# Small, repeatable query set from the existing catalog (self-query sanity check).
head -n 20 data/abo/vectors.csv > build/abo-queries.csv

./build/ann_cli bench \
  --index build/abo.ivf --queries build/abo-queries.csv \
  --mode ann --topk 10 --nprobe 16 --min-duration 10 --warmup 100

./build/ann_cli bench \
  --index build/abo.ivf --queries build/abo-queries.csv \
  --mode exact --topk 10 --min-duration 10 --warmup 100
```

Use `--index build/abo-full.ivf` for your full catalog index. Queries must have the
same dimensions as the index. ANN requires `1 <= nprobe <= clusters`; exact mode
does not use `nprobe` and reports it as `not_applicable`. With all clusters probed,
ANN is useful as a correctness reference; fewer probes expose the speed/recall
tradeoff. Use `evaluate` separately to measure recall. Held-out query vectors are
needed for retrieval-quality evaluation beyond the catalog self-query check.

`bench` loads the index and query CSV once, materializes query vectors before
timing, and performs `--warmup` searches using only the selected algorithm. It then
starts at the first query and cycles through the file in order until at least
`--min-duration` seconds have elapsed, finishing the current search. Defaults are
`--mode ann --topk 10 --nprobe 1 --min-duration 10 --warmup 100`. Warmup is a total
search count, not a pass count; zero is supported. Fractional positive durations
are accepted for smoke tests. All searches run inside C++; no per-query subprocess
or image embedding is involved.

In Instruments, run `build/ann_cli` with the desired `bench` arguments. Inspect the
measured search phase after the stderr message `Warmup complete; starting measured
search phase`; loading and warmup occur before its timer. Increase the duration
if you need more CPU samples. Run each mode separately under comparable machine
conditions, then repeat without the profiler to measure latency/throughput without
profiling overhead.

The report uses `key=value` lines:

| Field | Meaning |
| --- | --- |
| `measured_seconds` | Wall-clock duration of the measured loop, excluding loading, warmup, final statistics, and printing |
| `searches` | Number of completed measured searches, excluding warmup |
| `timed_search_seconds` | Sum of individual search-call durations |
| `mean_search_us` | Total individual search time divided by measured search count, in microseconds |
| `median_search_us`, `p95_search_us` | Median and nearest-rank 95th percentile of collected search latencies |
| `queries_per_second` | Measured searches divided by wall-clock `measured_seconds` |
| `latency_samples`, `latency_sampling` | Retained sample count and either `all` or `reservoir` |
| `checksum`, `warmup_checksum` | Observable consumption of result sizes and first/last neighbors' row IDs and distances |

Mode, file paths, dataset size/dimensions, cluster/query counts, top-k, probes,
warmup count, requested duration, and sampling settings are included in the report.
Keep these inputs and the compiler/build settings fixed for comparisons. The timed
checksum depends on how many searches finish, so it can differ between runs of
the same duration; the warmup checksum is repeatable for identical warmup results.

Two steady-clock readings surround each search call. Checksum updates, result
destruction, query selection, and latency collection occur outside those individual
timings but inside the wall-clock loop. Consequently, throughput includes loop
overhead and need not equal exactly `1e6 / mean_search_us`. Timer/bookkeeping costs
matter more for tiny synthetic datasets than for full 512D catalog searches.

Latency storage is preallocated and capped at 65,536 doubles (512 KiB). After that,
uniform reservoir sampling with a fixed seed of 42 keeps a bounded sample across
the entire run; median/p95 are then estimates. The mean still includes every
measured search. Samples are sorted only after timing stops. Reusing a small query
set measures repeated, warm searches; record the query-set size when interpreting
cache effects. Timings depend on CPU load, power/thermal conditions, and profiler
overhead.

Run the benchmark CLI tests without downloading datasets or models:

```bash
ANN_CLI="$PWD/build/ann_cli" python3 -m unittest discover -s tests -p 'test_bench_cli.py' -v
```

## Data contract for the core engine

The current CLI accepts a simple, **headerless** CSV with one ID and one fixed-length float vector per row:

```csv
image_001,0.24,0.58,0.11
image_002,0.30,0.62,0.09
image_003,0.85,0.12,0.77
```

The engine preserves IDs and returns them alongside distance scores. The image-search script resolves native ABO IDs to every associated listing in the full manifest, and also supports legacy compound IDs.

For real images, the local pipeline:

1. Joins extracted listing JSON Lines to image metadata and existing local paths.
2. Deduplicates physical image paths and uses verified native ABO image IDs for vectors.
3. Uses pretrained OpenCLIP `ViT-B-32` / `laion2b_s34b_b79k` to generate 512D float32 vectors.
4. L2-normalizes them and exports one headerless vector CSV row per unique image.
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
├── tests/test_abo_unique_images.py # Deduplication, checkpoint reuse, and relationship tests
├── tests/test_evaluation_split.py # Held-out IDs, exact CSV preservation, and validation tests
├── examples/                     # Small, synthetic input vectors
├── scripts/generate_synthetic.py # Deterministic larger datasets for development
├── scripts/prepare_abo_manifest.py # Local listings/image join
├── scripts/deduplicate_abo_manifest.py # Unique image vectors; preserve relationship manifest
├── scripts/create_evaluation_split.py # Reproducible streaming split for held-out queries
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

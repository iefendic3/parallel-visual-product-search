# Local ABO image search pipeline

The Python scripts prepare metadata and image vectors. The existing C++17 engine
builds and searches the IVF-Flat index. No ANN library, web frontend, OpenMP, or
CUDA kernels are introduced. Run the commands below from the repository root.

## 1. Local input layout

The implementation was checked against these **existing, extracted** files:

```text
data/abo/
├── LICENSE-CC-BY-4.0.txt
├── images/
│   ├── README.md
│   ├── metadata/images.csv.gz
│   └── small/<two hex digits>/<image filename>.jpg  # a few files are PNG
├── listings/
│   ├── README.md
│   └── metadata/listings_0.json.gz ... listings_f.json.gz
└── raw/
    ├── abo-images-small.tar
    └── abo-listings.tar
```

No extraction or download is performed by the scripts. The image lookup CSV has
`image_id,height,width,path`; `path` is relative to `images/small/`. Listing files
are gzip-compressed JSON Lines, one object per line. The inspected records use
`item_id`, `domain_name`, `main_image_id`, `other_image_id`, multilingual
`item_name` values, `node[].node_name`, and `product_type[].value`. The actual
`node_name`/product-type structures differ from some examples in the archive README.

## 2. Prepare the authoritative relationship manifest (standard library only)

If `data/abo/manifest-full.csv` already exists, keep it unchanged and skip this
command. To create it from the extracted dataset for the first time:

```bash
python scripts/prepare_abo_manifest.py --output data/abo/manifest-full.csv
```

This manifest retains **every listing/image association**. It is the authoritative
metadata source for image search, not the input to full embedding generation.
The earlier development command remains supported:

```bash
python scripts/prepare_abo_manifest.py --output data/abo/manifest.csv --limit 1000
```

| Manifest column | Meaning |
| --- | --- |
| `image_id` | Stable compound ID for a listing/image association; used by legacy vectors |
| `product_id` | ABO `item_id`; used for product deduplication |
| `title` | `item_name`: prefer `en_US`, then another English value, then the first available language; fall back to product ID |
| `image_path` | Local image path **relative to the manifest's directory**, independent of the shell working directory |
| `category` | Available `node_name`/`path` values, or the `product_type` value; blank if missing |
| `source_url` | CDN URL reconstructed from the native image ID and file extension using the archive README's documented template; never fetched and not guaranteed live |
| `abo_image_id` | Original `image_id` from ABO image metadata |
| `domain_name` | Original marketplace domain, or blank if unavailable |

ABO images can be shared by different products, and listings are identified by
both item ID and marketplace. Dropping all repeated native image IDs would lose
valid product associations. The relationship manifest's compound ID is therefore
`percent_encoded(item_id):percent_encoded(domain_name):percent_encoded(abo_image_id)`.
For example, `P1:amazon.com:a%2Bid` preserves the native `a+id` in a separate
column. Different images for a product have different vector IDs. Shared images
have separate rows for each listing. The next step removes repeated images only
from the embedding input; product relationships remain intact here. Product deduplication groups by `item_id`
across marketplaces. If that field is absent, the listing is skipped.

The script reads shards in sorted filename order, records in file order, and
main image before the ordered additional images. It removes repeated references
within a listing and repeated listing/image associations. Missing image files,
unknown image IDs, malformed JSON, and missing product IDs are skipped; missing
optional attributes do not invalidate a row. Only file existence is checked here;
image decoding is checked during embedding. The summary reports processed/skipped
listings, valid image associations, skipped image references, duplicates, and rows.
`--limit` caps generated rows, so it can stop in the middle of a product's images.

The lookup and duplicate-ID set use a temporary SQLite database on disk. Listings
and CSV rows are streamed. The manifest is written to a temporary file, flushed,
and atomically renamed; a failed preparation leaves the previous manifest intact.
Original ABO files are read only. No directory of decoded images is kept in RAM.

## 2a. Create a unique-image manifest before full embedding

```bash
python scripts/deduplicate_abo_manifest.py \
  --manifest data/abo/manifest-full.csv \
  --output data/abo/unique-images.csv
```

This streams metadata through a temporary on-disk SQLite database and writes one
row per **canonical local path**. It never decodes images, downloads files, changes
the source manifest, or removes product relationships. It checks every source row
before publishing the output and fails clearly on missing/invalid ABO IDs, one ID
with multiple paths, or one path with conflicting IDs. A failed validation keeps
any previous output intact. Files with identical bytes at different paths are not
merged. Missing files are left for the embedding stage to report; their bytes are
not read here.

The output has only `image_id,abo_image_id,image_path`. Both ID columns contain the
original native ABO image ID, including characters such as `+`. Paths are relative
to the output manifest's directory. Rows are sorted by native ID, so reordering
relationship records does not change the unique CSV. No product/title is chosen
arbitrarily for a shared image; use the full manifest to resolve products.

The validated local catalog produced:

| Count | Value |
| --- | --- |
| Relationship records | 710,650 |
| Unique paths / native image IDs | 398,170 |
| Repeated image references removed from embedding input | 312,480 |
| Unique product IDs retained in the full manifest | 145,454 |
| Images shared by multiple product IDs | 61,836 |

`unique-images.csv.meta.json` records these counts, the source/output SHA-256
hashes, and the native-ID scheme. The full manifest remains authoritative; the
sidecar and unique manifest are derived artifacts. Counts can differ for other
local subsets. Do not regenerate the relationship manifest or unique CSV with
changed contents while extending an existing embedding checkpoint.

## 3. Install the embedding environment

Use Python 3.10 or newer, preferably 3.12 or 3.13, with an **arm64** interpreter
on Apple Silicon:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-embeddings.txt
```

Direct dependencies are pinned, including the compatible PyTorch/torchvision
pair. Transitive dependencies are resolved by pip. Record the complete environment
for an experiment with `python -m pip freeze > data/abo/environment.txt`.
The embedding sidecar also records the torch, torchvision, OpenCLIP, and Pillow
versions; extending a checkpoint with different versions is rejected.

## 4. Generate catalog embeddings

```bash
python scripts/embed_abo.py \
  --manifest data/abo/unique-images.csv \
  --output data/abo/vectors-full.csv \
  --batch-size 16 --device auto
```

Add `--limit 1000` for an initial subset; omit it only when ready for the full run.
Embedding the entire dataset is an explicit user action. The minimal unique-image
manifest is accepted directly, and duplicate paths or vector IDs are refused.

Both catalog and query scripts use `scripts/clip_embeddings.py`:

- OpenCLIP `ViT-B-32`, pretrained `laion2b_s34b_b79k`, float32 inference.
- The model's **validation** preprocessing from `create_model_and_transforms`;
  EXIF orientation is applied and images are converted to RGB first.
- One model load per execution, evaluation mode, and `torch.inference_mode()`.
- Exactly 512 finite coordinates, L2 normalization, and explicit float32 conversion.

The [official OpenCLIP usage example](https://github.com/mlfoundations/open_clip)
uses this model/weights pair, its validation transform, evaluation mode, and
feature normalization. The [model configuration](https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/model_configs/ViT-B-32.json)
specifies 512 output dimensions. The first model load may download pretrained
weights; it does not download catalog images. Cached weights can subsequently be
used with local images. Keep that cache available for offline runs.

`auto` chooses CUDA if available, then Apple MPS, then CPU. `--device cpu`, `mps`,
or `cuda` explicitly selects a device and errors if it is unavailable. The model
and a single batch occupy RAM/device memory. Corrupt/unreadable images are reported
and skipped; invalid model outputs or inference errors stop the run with the
previously completed batches preserved. Progress and elapsed time go to stderr.

`vectors-full.csv` has **no header**, no quoted fields, and 513 columns per row:

```text
81iZlv3bjpL,0.012345,...,0.045678
```

The first column is the unique manifest's native `image_id`, never `product_id`.
The remaining 512 columns are numeric coordinates. Titles and all product
relationships remain in `manifest-full.csv`, a separate CSV with a header and normal CSV quoting.

### Resume and output integrity

Repeat the same command to resume automatically, or increase/remove `--limit`
to extend the run **without regenerating the manifest**. The limit means the first
N manifest rows, including failed images, rather than N successfully embedded
images. Lowering the limit publishes a smaller CSV snapshot while keeping cached
rows in the checkpoint. Previously failed images are retried on the next run.

- `vectors-full.csv.checkpoint.sqlite3` is the authoritative checkpoint. A complete
  batch is inserted in one transaction, with unique IDs and manifest positions.
- Ctrl-C or an inference failure can lose the current uncommitted batch; committed
  batches survive. A POSIX lock prevents simultaneous writers to the same output.
- The manifest SHA-256, model settings, pipeline version, and relevant package
  versions prevent incompatible runs from being mixed. Completed images' file
  sizes and modification times are checked before reuse; image contents are not
  cryptographically hashed. Keep the source dataset unchanged.
- At a successful completion, vectors are revalidated and streamed in manifest
  order to a temporary CSV, then atomically renamed. On interruption the existing
  CSV stays at its **previous completed snapshot**, or is absent on the first run.
  The next run rebuilds that snapshot from the checkpoint without duplicate IDs.
- `vectors-full.csv.meta.json` records provenance, exported count/limit, and CSV SHA-256.
  It is informational and written after the CSV; the checkpoint is used for resume.
- An existing CSV with no checkpoint is refused. For a changed manifest, source
  images, or dependency environment, choose a new output name instead of appending
  incompatible vectors. Regenerating a manifest with a different limit changes its
  hash, so prepare the full manifest first if you plan to extend beyond its rows.

### Reuse existing embeddings without recomputing images

If the earlier 1,000-record workflow already produced a checkpoint, run:

```bash
python scripts/embed_abo.py \
  --manifest data/abo/unique-images.csv --output data/abo/vectors-full.csv \
  --batch-size 16 --device auto \
  --reuse-checkpoint data/abo/vectors.csv.checkpoint.sqlite3 \
  --reuse-manifest data/abo/manifest.csv
```

The source checkpoint is opened read-only. Its manifest hash, model/weights,
dimensions, normalization, dtype, pipeline version, package versions, vector
values, row positions, native IDs, and image file stamps are verified. Cached
listing embeddings are matched by canonical path and copied under the new native
ID, once per image. If several old listing rows embed the same image, the first
verified vector in the source manifest is chosen deterministically. No model is
loaded if every selected image can be reused or resumed. The summary distinguishes
`reused` (imported), `resumed` (already in the new checkpoint), and newly `embedded`
images. The inspected old 1,000-record checkpoint contains 763 reusable unique
images. No full embedding generation was performed during this change.

You may omit the reuse arguments on later runs: imported vectors are now safely
stored in the destination checkpoint. Incompatible or incomplete provenance is
rejected instead of silently mixing embeddings. Plain CSVs without checkpoint
provenance cannot be imported. The previous `manifest.csv`, `vectors.csv`, and
checkpoint are left unchanged and remain usable with the legacy index.

Checkpoint text, the final CSV, and the temporary publication file require disk
space. Checkpoints can approach the CSV's size, and publication temporarily needs
another CSV's worth of free space. Keep each CSV, checkpoint, sidecar, and matching
manifest together. Changing devices may cause small floating-point differences;
record the selected device and use the same one for strict benchmark comparisons.

## 5. Build the existing C++ index

CMake remains supported as described in the main README. It is optional:

```bash
mkdir -p build
clang++ -std=c++17 -O2 -Iinclude \
  src/ivf_flat.cpp src/main.cpp -o build/ann_cli

./build/ann_cli build \
  --input data/abo/vectors-full.csv --output build/abo-full.ivf \
  --clusters 16 --iterations 20 --train-size 20000 --seed 42
```

All arguments match `src/main.cpp`. `--clusters` cannot exceed the number of saved
vectors. The C++ engine and its output format are unchanged. Its native binary
index stores IDs, full float vectors, centroids, and inverted lists.

## 6. Search using an external image

```bash
python scripts/search_image.py \
  --image ~/Downloads/test-product.jpg \
  --index build/abo-full.ivf --manifest data/abo/manifest-full.csv \
  --topk 10 --nprobe 4 --device auto
```

The query is embedded with the same code, model, weights, transform, and
normalization as the catalog. A safe subprocess argument list invokes the custom
C++ `search` command. Its exact stdout contract is `rank,id,squared_l2,l2`; stderr
contains diagnostics such as `search_us`. IDs are joined to manifest records and
ranked titles, image/product IDs, absolute local paths, and both distances are
printed. `--cli` selects a different executable. `--format json` produces a JSON
object for application integration, with diagnostics/provenance and a `matches`
array; progress stays on stderr. `image_matches` additionally contains every
returned image candidate and **all** its associated listings, even when `matches`
is truncated to `topk`. Each image score applies to every associated product.
Use the matching manifest and 512D index; missing returned IDs or dimensionality mismatches are reported as errors. The C++ index
does not store OpenCLIP provenance: retain `vectors-full.csv.meta.json` with experiments
and rebuild the index when model, preprocessing, or vectors change.

For best-effort unique products:

```bash
python scripts/search_image.py \
  --image ~/Downloads/test-product.jpg --index build/abo-full.ivf \
  --manifest data/abo/manifest-full.csv --topk 10 --nprobe 4 \
  --deduplicate-products --candidate-multiplier 5 --format json
```

Native ANN IDs are joined against `abo_image_id` in the full relationship
manifest. A shared image can therefore yield multiple distinct products, all with
the same image distance. `--deduplicate-products` keeps the closest occurrence of
each **product ID**, rather than suppressing other products that share an image.
Without deduplication, `matches` contains up to `topk` listing/image associations.
With deduplication, it contains up to `topk` distinct product IDs. `image_matches`
always preserves all associated listings for the returned image candidates.

Ranking is deterministic: image distance, then vector ID for ties, then product
ID, marketplace domain, and original listing/image ID. Flat results retain the
vector `image_id`, original `listing_image_id`, image `candidate_rank`, result
`rank`, both distance scores, and product metadata. Lowercase/uppercase native
IDs remain distinct. Legacy indexes returning compound IDs still resolve directly
through the original manifest's `image_id` column.

The initial request is `topk * candidate_multiplier` image candidates when product
deduplication is enabled. If that yields too few distinct products, the script
doubles the requested image count, without changing `nprobe`, until it reaches
`topk` products, exhausts candidates in the probed lists, or reaches
`--max-candidates` (default 10,000; must be at least `topk`). This is best-effort:
more probes or a higher candidate limit may be necessary. There is no promise of
X distinct products when fewer exist in the scanned lists. The C++ CLI returns
at most the available indexed vectors, so requests larger than that set terminate
at exhaustion. JSON reports attempt count and final requested/returned candidate
counts. The query is embedded once and reused on retries. Each retry invokes the
existing C++ CLI, which reloads the index; no C++ protocol or algorithm is changed.

Metadata is streamed, retaining only listings for requested candidate IDs. An
adaptive retry scans the manifest again to collect the new IDs, so a large
relationship CSV can add lookup time. Missing returned IDs or conflicting native
ID/path mappings fail clearly. Search must use **`manifest-full.csv`**, not the
product-free unique-image manifest. For the old workflow, continue using
`--index build/abo.ivf --manifest data/abo/manifest.csv`.

## Four different computations

| Stage | Input and output | When it runs |
| --- | --- | --- |
| Image embedding | Pretrained OpenCLIP maps pixels to a normalized 512D vector; no neural network training here | Once per catalog image, and once per new query image |
| K-means index training | Samples catalog vectors and learns cluster centroids | During C++ `build`, up to `--iterations` rounds |
| IVF index construction | Assigns **every** catalog vector to its nearest learned centroid and saves the inverted lists | During the same `build`, after K-means |
| ANN query execution | Embeds the query, selects `nprobe` centroids, scans their lists, and retains top-k neighbors | Each image search; the timed C++ `search_us` covers only the vector search |

For unit vectors, squared L2 is `2 - 2*cosine_similarity`, so it gives the same
ranking as cosine similarity. Recall@k measures agreement with exhaustive vector
search; human judgement measures visual usefulness. Test with held-out images for
meaningful retrieval evaluation. Image encoding, Python startup, model loading,
index loading, and metadata resolution are excluded from the C++ search timer.

## Scaling and Apple Silicon troubleshooting

Start with `--limit 1000` on the unique-image embedding command and confirm retrieval before removing that limit. The
manifest and embedding stages stream; the full C++ reference engine **does keep all
vectors in RAM**. Vector storage alone needs `N * 512 * 4` bytes (about 0.76 GiB for
400,000 rows), before IDs, inverted lists, CSV parsing, model memory, and temporary
allocations. The relationship manifest is larger than the unique-image vector input;
shared images now occupy one vector each. Full sequential K-means training can be slow. Use a training cap such as
`--train-size 20000` and evaluate cluster/probe settings; all vectors still enter
the index. These are development settings, not measured optimal settings.

If MPS runs out of memory, lower `--batch-size` to 4 or 1. For unsupported operations
or other device errors rerun with `--device cpu`; committed batches can be resumed.
Confirm that `python -c 'import platform; print(platform.machine())'` prints `arm64`
and `python -c 'import torch; print(torch.backends.mps.is_available())'` prints `True`
before forcing MPS. See the [official PyTorch MPS notes](https://docs.pytorch.org/docs/stable/notes/mps.html).
Keep model/transform versions fixed; the checkpoint rejects changed versions.
No speedup or real-model retrieval-quality claim follows from mocked tests.

## Tests and dataset credits

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
clang++ -std=c++17 -O2 -Iinclude src/ivf_flat.cpp tests/test_ann.cpp -o build/ann_tests
./build/ann_tests
```

Pipeline tests use temporary representative compressed metadata and fake inference,
including interrupted/resumed runs and a real compiled C++ 512D integration test.
Pillow enables a real decoding/corrupt-file check, otherwise that test is explicitly
skipped. Installed PyTorch enables a real tensor/inference-mode check with a fake
model; that check is explicitly skipped without PyTorch. No test downloads ABO
data or model weights. Compiler-dependent tests also report a skip if a compiler
is unavailable. These tests verify contracts and recovery, not pretrained model
quality or accelerator performance.

ABO assets are CC BY 4.0. Credit **Amazon.com** for data and images and the named
dataset creators, link the license, and state transformations such as subsetting,
EXIF orientation, RGB conversion, resizing/cropping, and generating embeddings.
Keep the original license and READMEs. See [DATASETS.md](DATASETS.md) for attribution
text, creators, and primary sources. The source-code MIT license does not relicense
the dataset or model weights. All dataset assets, vectors, caches, weights,
checkpoints, and binary indexes are excluded from Git.

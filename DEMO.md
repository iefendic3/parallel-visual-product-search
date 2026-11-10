# Professor demo

Run from the project root. Complete steps 1–2 before presenting; the first embedding run downloads model weights.

## 1. Set up, compile, and test

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-embeddings.txt

mkdir -p build
clang++ -std=c++17 -O2 -Iinclude src/ivf_flat.cpp src/main.cpp -o build/ann_cli
clang++ -std=c++17 -O2 -Iinclude src/ivf_flat.cpp tests/test_ann.cpp -o build/ann_tests
./build/ann_tests
python -m unittest discover -s tests -p 'test_*.py' -v
```

## 2. Prepare 1,000 images and build the index

```bash
python scripts/prepare_abo_manifest.py \
  --limit 1000 --output data/abo/demo/manifest.csv

python scripts/embed_abo.py \
  --manifest data/abo/demo/manifest.csv \
  --output data/abo/demo/vectors.csv \
  --limit 1000 --batch-size 16 --device auto

./build/ann_cli build \
  --input data/abo/demo/vectors.csv --output build/abo-demo.ivf \
  --clusters 16 --iterations 20 --train-size 0 --seed 42
```

## 3. Show an image and search for similar products

Select an indexed image for a sanity check: its nearest distance should be approximately zero.

```bash
source .venv/bin/activate

QUERY_IMAGE=$(python -c 'import csv; from pathlib import Path; p=Path("data/abo/demo/manifest.csv"); r=next(csv.DictReader(p.open())); print((p.parent / r["image_path"]).resolve())')
open "$QUERY_IMAGE"

python scripts/search_image.py \
  --image "$QUERY_IMAGE" \
  --index build/abo-demo.ivf --manifest data/abo/demo/manifest.csv \
  --topk 5 --nprobe 4 --deduplicate-products
```

Try an external image too (replace the filename):

```bash
python scripts/search_image.py \
  --image ~/Downloads/test-product.jpg \
  --index build/abo.ivf --manifest data/abo/manifest.csv \
  --topk 5 --nprobe 4 --deduplicate-products
```

## 4. Compare approximate and exhaustive search

This uses catalog vectors as queries: a correctness demonstration, not a held-out image-quality benchmark. With all 16 clusters probed, Recall@10 should be 1.

```bash
head -n 20 data/abo/vectors.csv > build/abo-queries.csv

./build/ann_cli evaluate \
  --index build/abo.ivf --queries build/abo-queries.csv \
  --topk 10 --nprobe 1

./build/ann_cli evaluate \
  --index build/abo.ivf --queries build/abo-queries.csv \
  --topk 10 --nprobe 16
```

If MPS fails, repeat the embedding/search command with `--device cpu`. Embedding resumes completed batches automatically.

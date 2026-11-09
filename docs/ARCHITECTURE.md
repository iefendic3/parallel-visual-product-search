# Algorithm Architecture (Sequential IVF-Flat)

## Problem

Given a query vector `q` of dimensionality `d`, find the `k` nearest database vectors under squared Euclidean distance:

`distance_squared(q, x) = sum_j (q[j] - x[j])^2`

The square root is unnecessary for ranking because it is monotonically increasing.

## Exact baseline

Examine all `N` vectors and retain the best `k`. Complexity for the current implementation is `O(N*d + N*log(k))` using a size-k heap.

## Index training (offline)

1. Select up to `train_size` vectors (random sample).
2. Initialize `nlist` centroids with **k-means++**.
3. Repeat up to `iterations` rounds of Lloyd's K-means:
   - For each training vector, find its nearest centroid.
   - Accumulate coordinate sums and membership counts.
   - Update each centroid by dividing sums by counts.
   - If a cluster is empty, reinitialize its centroid.
   - Stop when assignments no longer change.
4. For **every** database vector, find its nearest final centroid; append its row index to that centroid's inverted list.

Approximate full assignment cost is `O(N*nlist*d)` after training. Training cost is roughly `O(iterations*train_size*nlist*d)` plus initialization.

## Search (online)

1. Compare `q` against all `nlist` centroids.
2. Select the closest `nprobe` centroids.
3. Visit every vector in those selected inverted lists.
4. Keep the best `k` candidates in a heap.
5. Return result IDs and distances.

A representative average scanning cost for balanced clusters is proportional to `N*nprobe/nlist`; uneven clusters change this. Centroid comparisons and top-k selection also cost time.

**Recall decreases** if a true neighbor falls into a cluster that was not probed. With IVF-Flat, `nprobe == nlist` visits every vector and is equivalent to exact brute-force top-k, although it still pays centroid-selection overhead.

## Why this supports CPU/GPU parallel work later

- K-means vector-to-centroid assignments are independent for fixed centroids.
- Centroid updates need thread-local partial sums + reduction (avoid data races).
- Final list construction can use histogram counts + prefix sums + non-overlapping scatter.
- Candidate distance computation is parallel across vectors (and potentially dimensions).
- Top-k requires a reduction / merge of local candidate sets.

## Correctness invariants

- Every database vector belongs to **exactly one** inverted list.
- Every index vector has the same dimensionality.
- For `nprobe == nlist`, top-k indices match brute-force exact search.
- `Recall@k` is the overlap of approximate and exact top-k IDs divided by the number of exact top-k results.
- Identical distances break ties by original row index so comparisons are deterministic.

## Future app boundary

`Product image -> pretrained embedding model -> float vector -> ANN engine -> image IDs -> catalog metadata/images`

Do **not** train a new neural network for this project; image embeddings are an input to the search engine, not the parallel-computing contribution.

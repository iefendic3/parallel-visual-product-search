# Incremental roadmap

## Milestone 1 — sequential C++ ANN engine (implemented)

- [x] Exact squared-L2 top-k baseline
- [x] K-means++ initialization and sequential Lloyd's K-means
- [x] IVF-Flat index build using cluster assignments
- [x] IVF-Flat query with `nprobe` and top-k
- [x] Save/load reusable index
- [x] Smoke tests and CLI evaluation
- [x] Documentation, research papers, related GitHub projects
- [ ] Run and verify larger real-world benchmarks

## Milestone 2 — real product-data pipeline

- [ ] Download and parse the ABO subset + listing metadata
- [ ] Choose pretrained visual embedding model and document its license
- [ ] Generate consistent normalized vectors for catalog images
- [ ] Maintain mappings: image ID -> product ID/title/image path
- [ ] Validate nearest-neighbor behavior manually using representative images

## Milestone 3 — parallel CPU (OpenMP)

- [ ] Parallel candidate distance calculations with local top-k and merge
- [ ] Optional parallel K-means assignments/centroid update
- [ ] Separate parallel final index assignment and list population
- [ ] Compare accuracy, build time, and latency using identical indexes

## Milestone 4 — parallel GPU (CUDA)

- [ ] CUDA candidate-distance calculation
- [ ] GPU top-k reduction/selection
- [ ] Persistent GPU-resident index, batch queries, and transfer measurements
- [ ] Explore memory coalescing and block sizes
- [ ] Benchmark against sequential/OpenMP on appropriate NVIDIA hardware

## Milestone 5 — actual application

- [ ] Lightweight upload/search UI
- [ ] Image embedding inference service using the same preprocessing as catalog
- [ ] Long-running ANN search service that loads the index **once**
- [ ] Display top X similar products and scores
- [ ] Benchmark dashboard with real measurements, not illustrative values
- [ ] Final presentation and methods/results report
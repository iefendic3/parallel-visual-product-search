# Related open-source GitHub projects

These are **references**, not dependencies of our implementation.

| Repository | Algorithm/technology | Why it is useful |
|---|---|---|
| [facebookresearch/faiss](https://github.com/facebookresearch/faiss) | C++ / GPU; IVF-Flat, PQ, flat exact search | Most directly relevant reference for index structure and `nprobe`. |
| [nmslib/hnswlib](https://github.com/nmslib/hnswlib) | C++ HNSW graph index | Useful alternative ANN approach to compare conceptually; **not IVF**. |
| [unum-cloud/usearch](https://github.com/unum-cloud/usearch) | C++ similarity-search engine | API ergonomics, serialization, and embedding-based search examples; primarily graph-based ANN. |
| [dbaranchuk/ivf-hnsw](https://github.com/dbaranchuk/ivf-hnsw) | IVF + HNSW approaches | Research implementation related to inverted indexes at scale. |
| [rapidsai/cuvs](https://github.com/NVIDIA/cuvs) | CUDA vector search / GPU ANN | GPU indexing and query implementations for later milestone. |
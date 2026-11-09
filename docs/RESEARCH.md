# Scientific papers — reading list

Links point to author copies, open-access publisher pages, or preprints. **PDF files are not redistributed in this repository**; follow the supplied links to read/download them from their publishers or authors.

## Core methodology

1. **Arthur, D. & Vassilvitskii, S. (2007)** — *k-means++: The Advantages of Careful Seeding* (SODA). Initial centroid selection for our K-means training.
   - Paper (author-hosted PDF): <https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf>
   - Publication overview: <https://research.google/pubs/k-means-the-advantages-of-careful-seeding/>
   - Relevant to: `src/ivf_flat.cpp`'s k-means++ initializer.

2. **Jégou, H., Douze, M. & Schmid, C. (2011)** — *Product Quantization for Nearest Neighbor Search* (IEEE TPAMI 33(1), 117–128). Important historical foundation for inverted files and compressed similarity search. **Our current project uses IVF-Flat, not PQ.**
   - DOI: <https://doi.org/10.1109/TPAMI.2010.57>
   - Relevant to: background on scalable inverted-file-based ANN.

3. **Baranchuk, D., Babenko, A. & Malkov, Y. (2018)** — *Revisiting the Inverted Indices for Billion-Scale Approximate Nearest Neighbors* (ECCV). Search with inverted indices and the tradeoffs of candidate list construction.
   - Open-access article (PDF linked): <https://openaccess.thecvf.com/content_ECCV_2018/html/Dmitry_Baranchuk_Revisiting_the_Inverted_ECCV_2018_paper.html>
   - Preprint: <https://arxiv.org/abs/1802.02422>
   - Relevant to: IVF index design and evaluation.

4. **Douze, M., Guzhva, A., Deng, C. et al. (2024)** — *The Faiss Library*. Survey of design choices, search indexes, and practical performance tradeoffs.
   - Preprint: <https://arxiv.org/abs/2401.08281>
   - Relevant to: ground-truth evaluation, IVF-Flat reference behavior, CPU/GPU comparisons.

## GPU / parallel stage

5. **Johnson, J., Douze, M. & Jégou, H. (2017)** — *Billion-scale similarity search with GPUs*. GPU similarity search, memory hierarchy, and efficient top-k selection.
   - Preprint: <https://arxiv.org/abs/1702.08734>
   - Relevant to: CUDA distance kernels and GPU reductions.

## Dataset / application stage

6. **Collins, J., Goel, S., Deng, K. et al. (2022)** — *ABO: Dataset and Benchmarks for Real-World 3D Object Understanding* (CVPR 2022). The dataset used for the image-product search application.
   - Open-access CVPR paper: <https://openaccess.thecvf.com/content/CVPR2022/html/Collins_ABO_Dataset_and_Benchmarks_for_Real-World_3D_Object_Understanding_CVPR_2022_paper.html>
   - Preprint: <https://arxiv.org/abs/2110.06199>
   - Relevant to: dataset provenance, metadata, image-retrieval motivation.

## Primary documentation

- FAISS index overview: <https://github.com/facebookresearch/faiss/wiki/Faiss-indexes>
- FAISS official documentation: <https://faiss.ai/>
- NVIDIA CUDA programming guide: <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>

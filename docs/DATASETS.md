# Datasets and attribution

## Amazon Berkeley Objects (ABO) — application

- Official release: <https://registry.opendata.aws/amazon-berkeley-objects/>
- Dataset page: <https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/index.html>
- Dataset paper: <https://openaccess.thecvf.com/content/CVPR2022/html/Collins_ABO_Dataset_and_Benchmarks_for_Real-World_3D_Object_Understanding_CVPR_2022_paper.html>
- Contains **147,702 product listings** and **398,212 unique catalog images**.
- The `abo-images-small.tar` image archive is about **3 GB**, downscaled to 256 pixels maximum, compared to about 110 GB for the original catalog image archive.
- **License: CC BY 4.0**, updated from CC BY-NC in 2023 (older academic papers may still mention CC BY-NC). Source: AWS Open Data registry.

### Attribution requirements

Credit the dataset authors/providers, link to the license (<https://creativecommons.org/licenses/by/4.0/>), and explain any modifications (e.g. resizing, subsetting, generating vector embeddings). Retain provenance in the application and report.

## SIFT1M — optional academic benchmark

- Original project: <http://corpus-texmex.irisa.fr/>
- Dataset description: <https://github.com/tensorflow/datasets/blob/master/docs/catalog/sift1m.md>
- 1,000,000 vectors of dimension 128, plus 10,000 query vectors and ground-truth neighbors.
- Useful for evaluating recall and speed independently of image embedding generation.

## Synthetic dataset

`scripts/generate_synthetic.py` can create clustered float vectors for correctness and scalability testing without external downloads. These are **not realistic image embeddings**; do not treat their recall or performance results as representative of production image search.

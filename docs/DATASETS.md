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

## Local ABO integration and provenance

The extracted archives in this project were inspected under:

- `data/abo/images/metadata/images.csv.gz`: `image_id,height,width,path`.
- `data/abo/images/small/`: JPEG/PNG files located by the metadata `path`, not by guessing filenames from image IDs.
- `data/abo/listings/metadata/listings_*.json.gz`: gzip-compressed JSON Lines with `item_id`, `domain_name`, `main_image_id`, `other_image_id`, localized `item_name`, category `node_name`, and `product_type` values.

The extraction's `images/README.md`, `listings/README.md`, and
`LICENSE-CC-BY-4.0.txt` are the local schema/license references. Dataset assets are
read only and ignored by Git. The scripts perform no dataset downloads.

See [ABO_PIPELINE.md](ABO_PIPELINE.md) for the complete workflow. Manifest paths
are relative to the manifest directory. Shared images can belong to multiple
products or marketplaces; a stable compound `image_id` identifies each association,
while `abo_image_id` retains the original catalog-image identifier. Thus the
manifest's row count can exceed the unique catalog-image count. The pipeline
uses existing local small images, applies EXIF/RGB/model transforms, and creates
normalized OpenCLIP vectors. It does not train an image model.

The [official dataset page](https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/index.html)
and the extracted READMEs ask that data/images be credited to **Amazon.com**,
and dataset/archive/benchmark creation to **Matthieu Guillaumin, Thomas Dideriksen,
Kenan Deng, Himanshu Arora, Jasmine Collins, and Jitendra Malik**.

Suggested attribution for an application or university report (fill in the date):

> Uses Amazon Berkeley Objects (ABO), data and images © Amazon.com; dataset created
> by Matthieu Guillaumin, Thomas Dideriksen, Kenan Deng, Himanshu Arora, Jasmine
> Collins, and Jitendra Malik. Licensed under
> [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Accessed on YYYY-MM-DD
> through the [AWS Open Data registry](https://registry.opendata.aws/amazon-berkeley-objects/).
> This project selects image/listing associations, preprocesses images, and
> generates normalized OpenCLIP embeddings. No endorsement is implied.

Retain the original license and attribution when distributing relevant assets or
derivatives. The code's MIT license does not change ABO licensing. OpenCLIP code
and the chosen checkpoint have their own terms; consult the
[OpenCLIP repository](https://github.com/mlfoundations/open_clip) and
[checkpoint model card](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K)
when distributing weights. Weights are downloaded/cached separately and are not
committed to this repository.

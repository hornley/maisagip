# Corn Ear Segmentation Training Notebook

## Goal

Create a Google Colab notebook and a companion dataset ZIP for training one corn-ear instance-segmentation model. Follow the existing Colab notebook workflow, but train only YOLO11s-seg on the unified one-class dataset.

## Data

The dataset ZIP contains a re-split copy of `Corn ear segmentation.unified.yolov11`, with `train`, `valid`, `test`, and `data.yaml` at its root. The source folder remains unchanged. Its sole class is `corn-ear` (ID 0); each polygon row represents one ear mask.

Use an 80/10/10 split, stratified by source filename prefix. The 993 images round to 794 train, 99 validation, and 100 test images. The allocation is deterministic with seed 42. Group byte-identical images by SHA-256 and keep every duplicate group in one split. Sort image names and source prefixes before seeded shuffling; distribute duplicate groups round-robin across train, validation, and test, then fill each source's target counts from its shuffled single-image groups.

| Source prefix | Total images | Train | Validation | Test |
| --- | ---: | ---: | ---: | ---: |
| `det_v7` | 813 | 650 | 81 | 82 |
| `ear_seg_v3` | 180 | 144 | 18 | 18 |
| **Total** | **993** | **794** | **99** | **100** |

| Split | Images | Ear masks |
| --- | ---: | ---: |
| Train | 794 | 1,352 |
| Validation | 99 | 172 |
| Test | 100 | 171 |
| **Total** | **993** | **1,695** |

The 80/10/10 split is approximate after rounding (79.96% / 9.97% / 10.07%). The source contains three pairs of byte-identical images; no duplicate group crosses a split. Report test metrics as preliminary because the test set is still limited.

## Notebook flow

1. Mount Google Drive and locate the companion dataset ZIP in the same Drive folder used by existing Maisagip Colab notebooks.
2. Extract to a unique Colab workspace; check the archive, dataset YAML, one-class label IDs, and image/label pairs.
3. Require a GPU and train pretrained `yolo11s-seg.pt` for 100 epochs at 640 pixels, batch 4, seed 42, with caching disabled and two workers.
4. Load `best.pt`, evaluate the `test` split, and report segmentation precision, recall, mAP50, and mAP50-95.
5. Copy the checkpoint, training plots, and test metrics into a timestamped Drive results folder.

The notebook does not compare models, train a classifier, or run training on the local Mac. The ZIP contains the dataset only; the `.ipynb` is delivered separately.

## Checks and limitations

The notebook should fail clearly on a corrupt ZIP, missing GPU, invalid class IDs, missing image/label pairs, or unexpected dataset classes. Artifact checks cover notebook JSON and ZIP integrity; model training is not run as part of artifact creation.

The single-class conversion assumes each retained polygon outlines a corn ear. The `damaged corn` source polygons were mapped to `corn-ear`; if those polygons instead trace damage regions, they should be excluded before training. The 100-image test split still limits confidence in test metrics.

# Corn Ear Segmentation Training Notebook

## Goal

Create a Google Colab notebook and a companion dataset ZIP for training one corn-ear instance-segmentation model. Follow the existing Colab notebook workflow, but train only YOLO11s-seg on the unified one-class dataset.

## Data

The dataset ZIP contains the contents of `Corn ear segmentation.unified.yolov11`: `train`, `valid`, `test`, and `data.yaml`. Its sole class is `corn-ear` (ID 0). Preserve the existing splits:

| Split | Images | Masks |
| --- | ---: | ---: |
| Train | 889 | 1,504 |
| Validation | 66 | 120 |
| Test | 38 | 71 |

Do not alter the source datasets or regenerate the split. The held-out test set is small, so report metrics as preliminary.

## Notebook flow

1. Mount Google Drive and locate the companion dataset ZIP in the same Drive folder used by existing Maisagip Colab notebooks.
2. Extract to a unique Colab workspace; check the archive, dataset YAML, one-class label IDs, and image/label pairs.
3. Require a GPU and train pretrained `yolo11s-seg.pt` for 100 epochs at 640 pixels, batch 4, seed 42, with caching disabled and two workers.
4. Load `best.pt`, evaluate the `test` split, and report segmentation precision, recall, mAP50, and mAP50-95.
5. Copy the checkpoint, training plots, and test metrics into a timestamped Drive results folder.

The notebook does not compare models, train a classifier, or run training on the local Mac. The ZIP contains the dataset only; the `.ipynb` is delivered separately.

## Checks and limitations

The notebook should fail clearly on a corrupt ZIP, missing GPU, invalid class IDs, missing image/label pairs, or unexpected dataset classes. Artifact checks cover notebook JSON and ZIP integrity; model training is not run as part of artifact creation.

The single-class conversion assumes each retained polygon outlines a corn ear. The `damaged corn` source polygons were mapped to `corn-ear`; if those polygons instead trace damage regions, they should be excluded before training. The 38-image test split limits confidence in test metrics.

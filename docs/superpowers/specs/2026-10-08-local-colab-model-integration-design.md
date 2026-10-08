# Local Colab Model Integration Design

## Goal

Use the Colab-trained EfficientNetV2-S classifier and YOLO11s detector in the local `/inspect` flow without adding model binaries to Git.

## Current state

The backend loads `data/weights/efficientnetv2_s_corn.pt` and `data/weights/corn_yolov11n.pt`. The new checkpoints are `colab_trained_models/efficientnetv2_s_corn.pt` and `colab_trained_models/best.pt`. An older detector checkpoint already occupies the expected detector path. The new detector declares eight classes in the exact order of `config.DEFECT_CLASSES`; the classifier checkpoint is a compatible state dictionary.

## Local activation

Preserve the old detector checkpoint under a distinct backup filename in `data/weights`. Create relative symlinks from the two expected weight paths to the Colab checkpoint files. Do not copy or commit either weight file. Keep the existing backend model-loading path unchanged. Restart the backend so its cached model providers reload.

## Verification

Confirm both symlinks resolve to files and that `/health` reports real providers. Run a local `/inspect` request with a representative corn image and confirm it returns a report without a model-loading error. Check that detector class names map correctly and that Git does not stage model binaries. If loading fails, inspect the exception rather than accepting a silent demo fallback. Preserve the backup so the previous detector can be restored.

## Scope

This is a local activation only. No deployment configuration, model training, dataset changes, or weight-file commits are included.

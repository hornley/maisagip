# YOLO Model Comparison Results

## Evaluation Metrics

The object-detection models are evaluated using Precision, Recall, F1 score,
Intersection over Union (IoU), and Mean Average Precision (mAP). These metrics
measure different aspects of detection quality:

- **Precision** measures the proportion of predicted objects that are correct:
  `TP / (TP + FP)`.
- **Recall** measures the proportion of ground-truth objects that were detected:
  `TP / (TP + FN)`.
- **F1 score** is the harmonic mean of precision and recall:
  `2 × (Precision × Recall) / (Precision + Recall)`.
- **IoU** measures the overlap between a predicted bounding box and its
  corresponding ground-truth box:
  `Area of overlap / Area of union`.
- **mAP50** is mean Average Precision at an IoU threshold of 0.50.
- **mAP50-95** is mean Average Precision averaged across IoU thresholds from
  0.50 through 0.95 in increments of 0.05.

## Detection Concepts

The following concepts explain how the YOLO detector evaluates and filters
predictions:

- **Intersection over Union (IoU)** measures the overlap between a predicted
  box and the corresponding ground-truth box:

  ```text
  IoU = Area of Overlap / Area of Union
  ```

- **Detection confidence** is the model's confidence that a predicted box
  belongs to a particular class. In the Ultralytics YOLOv8/YOLO11 detection
  models used here, the displayed confidence should not be described as a
  simple `objectness × IoU` product. IoU is instead used for box matching,
  non-maximum suppression, and mAP evaluation.

- **Non-Maximum Suppression (NMS)** removes duplicate predictions. For boxes
  representing the same class, if their IoU exceeds the NMS threshold, the box
  with the lower confidence is suppressed:

  ```text
  If IoU(box A, box B) > threshold,
  keep the box with the higher confidence.
  ```

- **Detection training loss** combines box-regression, classification, and
  Distribution Focal Loss (DFL) terms:

  ```text
  Total Detection Loss = Box Loss + Class Loss + DFL Loss
  ```

  Mask Loss is used for segmentation models, not for the object-detection
  models evaluated in this comparison.

The supplied comparison output reports aggregate Precision, Recall, mAP50, and
mAP50-95 for each checkpoint. It does not include a separate mean IoU value, so
IoU should be calculated and reported separately before the final results are
published. IoU is still used within the mAP evaluation thresholds.

## Table 3. Overall YOLO Performance

| Model | Precision | Recall | F1 Score | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| YOLOv8n | 0.7557 | 0.5417 | 0.6310 | 0.7871 | 0.4446 |
| YOLOv8s | 0.7675 | 0.5972 | 0.6717 | 0.5992 | 0.4268 |
| YOLOv8m | 0.6824 | 0.5556 | 0.6125 | 0.7290 | 0.4206 |
| YOLO11s | 0.7714 | 0.5783 | 0.6610 | 0.7997 | 0.4747 |

The F1 score was calculated from the supplied precision and recall values as:

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

YOLO11s achieved the highest precision, mAP50, and mAP50-95. YOLOv8s
achieved the highest recall and F1 score. The results therefore do not identify
one model as the best for every metric.

## Table 4. Evaluation Checkpoints

| Model | Run directory | Best checkpoint |
|---|---|---|
| YOLOv8n | `/content/maisagip/data/comparisons/yolo/yolov8n` | `/content/maisagip/data/comparisons/yolo/yolov8n/weights/best.pt` |
| YOLOv8s | `/content/maisagip/data/comparisons/yolo/yolov8s` | `/content/maisagip/data/comparisons/yolo/yolov8s/weights/best.pt` |
| YOLOv8m | `/content/maisagip/data/comparisons/yolo/yolov8m` | `/content/maisagip/data/comparisons/yolo/yolov8m/weights/best.pt` |
| YOLO11s | `/content/maisagip/data/comparisons/yolo/yolo11s` | `/content/maisagip/data/comparisons/yolo/yolo11s/weights/best.pt` |

The supplied JSON contains only aggregate metrics. It does not provide
class-level precision, recall, F1, or mAP values, so class-specific performance
is not reported here.

## Table 5. Precision Comparison

| Model | Precision | Difference from best | Rank |
|---|---:|---:|---:|
| YOLO11s | 0.7714 | 0.0000 | 1 |
| YOLOv8s | 0.7675 | 0.0039 | 2 |
| YOLOv8n | 0.7557 | 0.0156 | 3 |
| YOLOv8m | 0.6824 | 0.0890 | 4 |

YOLO11s had the highest precision, while YOLOv8m had the lowest.

## Table 6. Recall Comparison

| Model | Recall | Difference from best | Rank |
|---|---:|---:|---:|
| YOLOv8s | 0.5972 | 0.0000 | 1 |
| YOLO11s | 0.5783 | 0.0190 | 2 |
| YOLOv8m | 0.5556 | 0.0417 | 3 |
| YOLOv8n | 0.5417 | 0.0556 | 4 |

YOLOv8s had the highest recall, while YOLOv8n had the lowest.

## Table 7. mAP Comparison

| Model | mAP50 | mAP50-95 | mAP50 Rank | mAP50-95 Rank |
|---|---:|---:|---:|---:|
| YOLOv8n | 0.7871 | 0.4446 | 2 | 2 |
| YOLOv8s | 0.5992 | 0.4268 | 4 | 3 |
| YOLOv8m | 0.7290 | 0.4206 | 3 | 4 |
| YOLO11s | 0.7997 | 0.4747 | 1 | 1 |

YOLO11s ranked first on both mAP measures. YOLOv8s ranked last on mAP50 and
third on mAP50-95, although its recall was the highest.

## Table 8. F1 Score Comparison

| Model | F1 Score | Difference from best | Rank |
|---|---:|---:|---:|
| YOLOv8s | 0.6717 | 0.0000 | 1 |
| YOLO11s | 0.6610 | 0.0107 | 2 |
| YOLOv8n | 0.6310 | 0.0407 | 3 |
| YOLOv8m | 0.6125 | 0.0593 | 4 |

YOLOv8s achieved the highest F1 score because it provided the best balance of
precision and recall in this evaluation.

## Interpretation and Limitations

Based on the supplied results, YOLO11s performed best on precision, mAP50, and
mAP50-95, while YOLOv8s performed best on recall and F1 score. The preferred
model therefore depends on whether the application prioritizes localization
quality, fewer false positives, or finding more ground-truth objects.

The supplied results should still be checked before they are treated as a final
benchmark. Confirm that each model was evaluated using its own trained
checkpoint, that evaluation outputs were not copied or reused, and that all
models used the same held-out test split and class configuration. In
particular, the aggregate results do not show whether performance is dominated
by the `corn_ear` class or by particular defect classes.

Because the dataset is small, these values should be described as an
exploratory comparison unless the models are retrained and evaluated on a
larger, independent test set. Keep all four views of each ear in the same split
to avoid image-level leakage between training and testing.

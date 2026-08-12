from . import config


def _area(box):
    x0, y0, x1, y1 = box
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _ear_box(image_shape, detections):
    h, w = image_shape[:2]
    ear_boxes = [d for d in detections if d["class"] == "corn_ear"]
    if ear_boxes:
        return max(ear_boxes, key=lambda d: _area(d["box"]))["box"]
    candidate_boxes = [d["box"] for d in detections if _area(d["box"]) >= config.MIN_EAR_BOX_AREA]
    if candidate_boxes:
        return max(candidate_boxes, key=_area)
    return [0.0, 0.0, float(w), float(h)]


def estimate_traits(image_shape, detections):
    h, w = image_shape[:2]
    ear = _ear_box(image_shape, detections)
    length_px = max(ear[2] - ear[0], ear[3] - ear[1])
    fraction = length_px / max(h, w)

    if fraction < config.EAR_SIZE_THRESHOLDS["small"]:
        ear_size = "small"
    elif fraction < config.EAR_SIZE_THRESHOLDS["medium"]:
        ear_size = "medium"
    else:
        ear_size = "large"

    ear_area = max(_area(ear), 1.0)
    missing_area = sum(
        _area(d["box"]) for d in detections if d["class"] == "missing_kernels"
    )
    completeness = max(0.0, 1.0 - missing_area / ear_area)
    completeness = min(1.0, round(completeness, 4))

    return {
        "ear_size": ear_size,
        "ear_size_fraction": round(fraction, 4),
        "kernel_completeness": completeness,
        "ear_box": [round(v, 2) for v in ear],
    }
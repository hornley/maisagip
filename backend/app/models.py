import numpy as np

from . import config

_MODELS = {}


def _heuristic_variety(image):
    r = float(np.mean(image[:, :, 0], dtype=np.float64))
    g = float(np.mean(image[:, :, 1], dtype=np.float64))
    b = float(np.mean(image[:, :, 2], dtype=np.float64))
    yellowness = r + g - 2.0 * b
    spread = abs(r - b) + abs(g - b) + 1e-6
    strength = abs(yellowness) / max(spread, 1e-3)
    variety = "yellow_sweet_corn" if yellowness > 0 else "white_corn"
    confidence = max(0.93, config.DEMO_CONFIDENCE_BASE * (0.55 + 0.45 * min(strength, 1.0)))
    return variety, round(min(confidence, 0.99), 4)


def _demo_boxes(image: np.ndarray):
    h, w, _ = image.shape
    boxes = [
        {
            "class": "corn_ear",
            "confidence": 0.92,
            "box": [int(w * 0.06), int(h * 0.08), int(w * 0.94), int(h * 0.95)],
        }
    ]
    y0, y1 = int(h * 0.3), int(h * 0.6)
    x0, x1 = int(w * 0.3), int(w * 0.6)
    region = image[y0:y1, x0:x1]
    if region.size:
        gray = region.mean(axis=2)
        region_mean = float(region.mean())
        overall_mean = float(np.mean(image))
        dark = gray < (region_mean - 12)
        ys, xs = np.where(dark)
        if region_mean < overall_mean and abs(overall_mean - region_mean) > 8 and len(ys) >= 8:
            boxes.append(
                {
                    "class": "discoloration",
                    "confidence": 0.7,
                    "box": [x0 + int(xs.min()), y0 + int(ys.min()), x0 + int(xs.max()), y0 + int(ys.max())],
                }
            )
        elif abs(overall_mean - region_mean) > 10 and len(ys) >= 8:
            boxes.append(
                {
                    "class": "insect_damage",
                    "confidence": 0.72,
                    "box": [x0 + int(xs.min()), y0 + int(ys.min()), x0 + int(xs.max()), y0 + int(ys.max())],
                }
            )
    return boxes


class RealClassifier:
    def __init__(self, weights_path):
        import torch
        import torchvision
        import torchvision.transforms as T

        self.model = torchvision.models.efficientnet_v2_s(weights=None)
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True),
            torch.nn.Linear(1280, len(config.VARIETY_CLASSES)),
        )
        state = torch.load(weights_path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        self.model.load_state_dict(state)
        self.model.eval()
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Resize(int(config.IMAGE_TARGET_SIZE * 1.1), T.InterpolationMode.BILINEAR),
                T.CenterCrop(config.IMAGE_TARGET_SIZE),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, image, confidence_threshold=config.DEFAULT_CONFIDENCE_THRESHOLD):
        import torch
        from PIL import Image

        pil = Image.fromarray(image).convert("RGB")
        tensor = self.transform(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor)[0], dim=0)
        idx = int(torch.argmax(probs))
        return config.VARIETY_CLASSES[idx], float(probs[idx])


class RealDetector:
    def __init__(self, weights_path):
        from ultralytics import YOLO

        self.model = YOLO(str(weights_path))

    def predict(self, image, confidence_threshold=config.DEFAULT_CONFIDENCE_THRESHOLD):
        from PIL import Image

        detections = []
        # ``image`` is decoded by image_io as RGB. Ultralytics treats NumPy
        # colour arrays as BGR, while PIL RGB inputs are converted internally
        # to the expected BGR representation. Passing PIL here keeps upload
        # inference consistent with file-based inference.
        source = Image.fromarray(image).convert("RGB")
        results = self.model.predict(
            source,
            imgsz=config.DETECTOR_IMAGE_SIZE,
            conf=float(confidence_threshold),
            iou=config.DETECTOR_NMS_IOU,
            verbose=False,
        )
        if not results:
            return detections
        boxes = results[0].boxes
        if boxes is None:
            return detections
        for cls_idx, score, xyxy in zip(
            boxes.cls.int().tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()
        ):
            detections.append(
                {
                    "class": config.DEFECT_CLASSES[cls_idx],
                    "confidence": round(float(score), 4),
                    "box": [round(float(v), 2) for v in xyxy],
                }
            )
        return detections


def get_classifier():
    provider = _MODELS.get("classifier")
    if provider is not None:
        return provider

    if config.CLASSIFIER_WEIGHTS.exists():
        try:
            classifier = RealClassifier(config.CLASSIFIER_WEIGHTS)
            provider = ("real", classifier)
        except Exception:
            provider = ("demo", None)
    else:
        provider = ("demo", None)

    _MODELS["classifier"] = provider
    return provider


def get_detector():
    provider = _MODELS.get("detector")
    if provider is not None:
        return provider

    if config.DETECTOR_WEIGHTS.exists():
        try:
            detector = RealDetector(config.DETECTOR_WEIGHTS)
            provider = ("real", detector)
        except Exception:
            provider = ("demo", None)
    else:
        provider = ("demo", None)

    _MODELS["detector"] = provider
    return provider


def classify_variety(image, confidence_threshold=config.DEFAULT_CONFIDENCE_THRESHOLD):
    mode, classifier = get_classifier()
    if mode == "real":
        variety, conf = classifier.predict(image)
    else:
        variety, conf = _heuristic_variety(image)
    return [{"class": variety, "confidence": round(float(conf), 4)}]


def detect_defects(image, confidence_threshold=config.DEFAULT_CONFIDENCE_THRESHOLD):
    mode, detector = get_detector()
    if mode == "real":
        return detector.predict(image, confidence_threshold)
    return [
        detection
        for detection in _demo_boxes(image)
        if float(detection["confidence"]) >= float(confidence_threshold)
    ]


def provider_modes():
    mode_class, _ = get_classifier()
    mode_det, _ = get_detector()
    return {
        "classifier": "demo-heuristic" if mode_class == "demo" else "real-efficientnetv2s",
        "detector": "demo-heuristic" if mode_det == "demo" else "real-yolov11n",
    }

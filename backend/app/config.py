from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ORIGINALS_DIR = DATA_DIR / "raw" / "originals"
WEIGHTS_DIR = DATA_DIR / "weights"
REPORTS_DIR = DATA_DIR / "reports"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

CLASSIFIER_WEIGHTS = WEIGHTS_DIR / "efficientnetv2_s_corn.pt"
DETECTOR_WEIGHTS = WEIGHTS_DIR / "corn_yolov11n.pt"

VARIETY_CLASSES = ["white_corn", "yellow_sweet_corn"]
DEFECT_CLASSES = [
    "corn_ear",
    "mold",
    "insect_damage",
    "discoloration",
    "deformity",
    "missing_kernels",
    "ear_decay",
]

IMAGE_TARGET_SIZE = 448
DETECTOR_IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.30
DEMO_CONFIDENCE_BASE = 0.86
NEEDS_REINSPECTION_CONF = 0.60

EAR_SIZE_THRESHOLDS = {"small": 0.28, "medium": 0.42}
MIN_EAR_BOX_AREA = 2500
MIN_TEXT_BOX_AREA = 1200

VIEWS_PER_EAR = 4
COVERAGE_THRESHOLDS = {"ExtraClass": 0.0, "ClassI": 0.05, "ClassII": 0.10}
DUPLICATE_AREA_RATIO = 1.25
SEVERE_CLASSES = {"mold", "insect_damage"}

from . import image_io
from . import models
from . import traits as traits_mod
from .report import build_report
from .rules.engine import GradeEngine
from .views import merge_views

_engine = GradeEngine()


def decode_image(bytes_data):
    return image_io.decode_image_bytes(bytes_data)


def inspect_ear(images_bytes):
    if not images_bytes:
        raise ValueError("No images uploaded.")

    per_view = []
    for bytes_data in images_bytes:
        image = decode_image(bytes_data)
        variety = models.classify_variety(image)[0]
        detections = models.detect_defects(image)
        if not detections:
            detections = _ear_fallback(image)
        traits = traits_mod.estimate_traits(image.shape, detections)
        per_view.append(
            {"image": image, "variety": variety, "detections": detections, "traits": traits}
        )

    merged = merge_views(per_view)

    defect_confs = [d["max_conf"] for d in merged["defects"]]
    engine_result = _engine.evaluate(
        merged["variety"]["confidence"],
        merged["defect_coverage"],
        merged["severe_present"],
        merged["traits"]["kernel_completeness"],
        defect_confs,
    )

    report = build_report(per_view, merged, engine_result)
    report["mode"] = models.provider_modes()
    return report


def inspect_image_bytes(bytes_data):
    return inspect_ear([bytes_data])


def _ear_fallback(image):
    h, w = image.shape[:2]
    return [
        {
            "class": "corn_ear",
            "confidence": 0.8,
            "box": [0, 0, float(w), float(h)],
        }
    ]

from . import config, image_io
from . import models
from . import traits as traits_mod
from .report import build_report
from .rules.engine import GradeEngine
from .views import merge_views

_engine = GradeEngine()


class InconsistentInspectionInput(ValueError):
    """Raised when confident views identify more than one corn variety."""


# Keep the provisional name import-compatible for callers of the first slice.
MixedVarietyError = InconsistentInspectionInput


def decode_image(bytes_data):
    return image_io.decode_image_bytes(bytes_data)


def _invoke_provider(provider, image, confidence_threshold):
    """Keep compatibility with simple one-argument test/custom providers."""
    try:
        return provider(image, confidence_threshold)
    except TypeError as exc:
        # Existing integrations commonly expose the original one-argument API.
        # Only retry when the callable's signature confirms that shape.
        import inspect

        try:
            parameters = inspect.signature(provider).parameters
        except (TypeError, ValueError):
            raise exc
        if len(parameters) != 1:
            raise
        return provider(image)


def inspect_ear(images_bytes, confidence_threshold=config.DEFAULT_CONFIDENCE_THRESHOLD):
    if not images_bytes:
        raise ValueError("No images uploaded.")
    if not config.MIN_CONFIDENCE_THRESHOLD <= confidence_threshold <= config.MAX_CONFIDENCE_THRESHOLD:
        raise ValueError(
            f"confidence_threshold must be between {config.MIN_CONFIDENCE_THRESHOLD} and "
            f"{config.MAX_CONFIDENCE_THRESHOLD}."
        )

    per_view = []
    for bytes_data in images_bytes:
        image = decode_image(bytes_data)
        variety = _invoke_provider(models.classify_variety, image, confidence_threshold)[0]
        detections = _invoke_provider(models.detect_defects, image, confidence_threshold)
        detections = [
            d for d in detections if float(d.get("confidence", 0.0)) >= confidence_threshold
        ]
        if not detections:
            detections = _ear_fallback(image)
        traits = traits_mod.estimate_traits(image.shape, detections)
        per_view.append(
            {"image": image, "variety": variety, "detections": detections, "traits": traits}
        )

    merged = merge_views(per_view, confidence_threshold)
    if merged["variety_disagreement"]["confident"]:
        conflicts = ", ".join(
            f"view {view_number}={variety}"
            for view_number, variety in merged["variety_disagreement"]["conflicts"]
        )
        raise InconsistentInspectionInput(
            "These images appear to show different corn varieties "
            f"({conflicts}). Upload views of a single ear/variety, then reinspect."
        )

    defect_confs = [d["max_conf"] for d in merged["defects"]]
    engine_result = _engine.evaluate(
        merged["variety"]["confidence"],
        merged["defect_coverage"],
        merged["severe_present"],
        merged["traits"]["kernel_completeness"],
        defect_confs,
        confidence_threshold=confidence_threshold,
        variety_disagreement=merged["variety_disagreement"]["present"],
    )

    report = build_report(per_view, merged, engine_result, confidence_threshold)
    report["mode"] = models.provider_modes()
    return report


def inspect_image_bytes(bytes_data, confidence_threshold=config.DEFAULT_CONFIDENCE_THRESHOLD):
    return inspect_ear([bytes_data], confidence_threshold)


def _ear_fallback(image):
    h, w = image.shape[:2]
    return [
        {
            "class": "corn_ear",
            "confidence": 0.8,
            "box": [0, 0, float(w), float(h)],
        }
    ]

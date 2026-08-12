from backend.app.traits import estimate_traits

SHAPE = (400, 300, 3)


def detection(cls, conf=0.9, box=None):
    return {"class": cls, "confidence": conf, "box": box or [0, 0, 100, 100]}


def test_large_ear_bucket_and_fraction():
    traits = estimate_traits(SHAPE, [detection("corn_ear", box=[0, 40, 180, 360])])
    assert traits["ear_size"] == "large"
    assert traits["ear_size_fraction"] == 0.8


def test_small_ear_bucket():
    traits = estimate_traits(SHAPE, [detection("corn_ear", box=[10, 150, 100, 200])])
    assert traits["ear_size"] == "small"


def test_medium_ear_bucket():
    traits = estimate_traits(SHAPE, [detection("corn_ear", box=[10, 100, 150, 240])])
    assert traits["ear_size"] == "medium"


def test_falls_back_to_largest_box_when_no_ear():
    traits = estimate_traits(
        SHAPE,
        [detection("mold", box=[0, 0, 100, 100]), detection("mold", box=[0, 50, 240, 350])],
    )
    assert traits["ear_size"] == "large"


def test_missing_kernel_area_reduces_completeness():
    traits = estimate_traits(
        SHAPE,
        [detection("corn_ear", box=[0, 0, 300, 400]), detection("missing_kernels", box=[0, 0, 150, 200])],
    )
    assert traits["kernel_completeness"] == 0.75


def test_clean_ear_is_fully_complete():
    traits = estimate_traits(SHAPE, [detection("corn_ear", box=[0, 0, 300, 400])])
    assert traits["kernel_completeness"] == 1.0


def test_clean_ear_with_no_detections_is_fully_complete():
    traits = estimate_traits(SHAPE, [])
    assert traits["kernel_completeness"] == 1.0
    assert traits["ear_size"] == "large"
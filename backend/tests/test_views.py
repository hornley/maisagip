from backend.app.views import merge_views

EAR_BOX = [0, 0, 200, 400]
HALF_BOX = [0, 0, 100, 400]


def view(variety_class="yellow_sweet_corn", confidence=0.9, detections=(), completeness=1.0):
    return {
        "variety": {"class": variety_class, "confidence": confidence},
        "detections": list(detections),
        "traits": {
            "ear_box": EAR_BOX,
            "kernel_completeness": completeness,
            "ear_size": "large",
            "ear_size_fraction": 0.5,
        },
    }


def defect(cls, box=HALF_BOX, confidence=0.9):
    return {"class": cls, "box": box, "confidence": confidence}


def ear_view():
    return view(detections=[defect("corn_ear")])


def by_class(merged, cls):
    return next(d for d in merged["defects"] if d["class"] == cls)


def test_duplicate_across_adjacent_views_merges_to_one():
    v1 = view(detections=[defect("corn_ear"), defect("missing_kernels")])
    v2 = view(detections=[defect("corn_ear"), defect("missing_kernels")])
    v3 = ear_view()
    v4 = ear_view()
    merged = merge_views([v1, v2, v3, v4])
    entry = by_class(merged, "missing_kernels")
    assert entry["merged_count"] == 1
    assert entry["views_seen"] == [0, 1]
    assert merged["defect_coverage"] == 0.5


def test_shriveled_kernels_detection_merges_like_other_defects():
    merged = merge_views(
        [
            view(detections=[defect("corn_ear"), defect("shriveled_kernels")]),
        ]
    )

    entry = by_class(merged, "shriveled_kernels")
    assert entry["merged_count"] == 1
    assert entry["coverage"] == 0.5
    assert merged["severe_present"] is False


def test_defects_in_non_adjacent_views_are_distinct():
    v1 = view(detections=[defect("corn_ear"), defect("mold")])
    v2 = ear_view()
    v3 = view(detections=[defect("corn_ear"), defect("mold")])
    v4 = ear_view()
    merged = merge_views([v1, v2, v3, v4])
    entry = by_class(merged, "mold")
    assert entry["merged_count"] == 2
    assert entry["views_seen"] == [0, 2]


def test_two_defects_in_same_view_never_merge():
    box_a = [0, 0, 100, 100]
    box_b = [100, 0, 200, 100]
    v1 = view(detections=[defect("corn_ear"), defect("discoloration", box_a), defect("discoloration", box_b)])
    merged = merge_views([v1])
    entry = by_class(merged, "discoloration")
    assert entry["merged_count"] == 2


def test_whole_ear_defect_boxes_to_high_coverage():
    v1 = view(detections=[defect("corn_ear"), defect("deformity", EAR_BOX)])
    merged = merge_views([v1])
    entry = by_class(merged, "deformity")
    assert entry["coverage"] == 1.0
    assert merged["defect_coverage"] == 1.0


def test_coverage_sums_distinct_defects():
    half_a = [0, 0, 100, 400]
    half_b = [100, 0, 200, 400]
    v1 = view(detections=[defect("corn_ear"), defect("mold", half_a)])
    v2 = view(detections=[defect("corn_ear"), defect("insect_damage", half_b)])
    v3 = ear_view()
    v4 = ear_view()
    merged = merge_views([v1, v2, v3, v4])
    assert merged["defect_coverage"] == 1.0
    assert merged["severe_present"] is True


def test_variety_majority_vote():
    v1 = view(variety_class="yellow_sweet_corn", confidence=0.6)
    v2 = view(variety_class="yellow_sweet_corn", confidence=0.6)
    v3 = view(variety_class="yellow_sweet_corn", confidence=0.6)
    v4 = view(variety_class="white_corn", confidence=0.99)
    merged = merge_views([v1, v2, v3, v4])
    assert merged["variety"]["class"] == "yellow_sweet_corn"


def test_variety_tie_breaks_by_confidence():
    v1 = view(variety_class="yellow_sweet_corn", confidence=0.55)
    v2 = view(variety_class="yellow_sweet_corn", confidence=0.55)
    v3 = view(variety_class="white_corn", confidence=0.99)
    v4 = view(variety_class="white_corn", confidence=0.99)
    merged = merge_views([v1, v2, v3, v4])
    assert merged["variety"]["class"] == "white_corn"


def test_completeness_takes_worst_view():
    v1 = view(completeness=1.0)
    v2 = view(completeness=0.72)
    merged = merge_views([v1, v2])
    assert merged["traits"]["kernel_completeness"] == 0.72


def test_single_view_passthrough():
    v1 = view(detections=[defect("corn_ear"), defect("mold")])
    merged = merge_views([v1])
    assert merged["view_count"] == 1
    assert merged["defect_coverage"] == 0.5
    assert merged["severe_present"] is True


def test_clean_ear_has_zero_coverage():
    merged = merge_views([ear_view(), ear_view(), ear_view(), ear_view()])
    assert merged["defects"] == []
    assert merged["defect_coverage"] == 0.0
    assert merged["severe_present"] is False

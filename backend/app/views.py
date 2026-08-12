from collections import Counter, defaultdict

from . import config


def _box_area(box):
    x0, y0, x1, y1 = box
    return max(0.0, (x1 - x0) * (y1 - y0))


def _union_find(n):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    return find, union


def _merge_class(dets, n_views, ear_areas):
    by_view = [[] for _ in range(n_views)]
    for det in dets:
        by_view[det["view"]].append(det)

    if not any(by_view):
        return None

    counts_per_view = [len(v) for v in by_view]
    mergeable = max(counts_per_view) <= 1

    if mergeable:
        find, union = _union_find(n_views)
        for k in range(n_views):
            nxt = (k + 1) % n_views
            if by_view[k] and by_view[nxt]:
                a = _box_area(by_view[k][0]["box"])
                b = _box_area(by_view[nxt][0]["box"])
                lo, hi = min(a, b), max(a, b)
                if hi <= config.DUPLICATE_AREA_RATIO * lo:
                    union(k, nxt)
        cluster_map = {}
        for k in range(n_views):
            if by_view[k]:
                cluster_map.setdefault(find(k), []).append(by_view[k][0])
        clusters = list(cluster_map.values())
    else:
        clusters = [[det] for items in by_view for det in items]

    coverage_ratio = 0.0
    views_seen = set()
    max_conf = 0.0
    largest = None
    largest_area = 0.0
    for cluster in clusters:
        ratios = [_box_area(d["box"]) / ear_areas[d["view"]] for d in cluster]
        coverage_ratio += max(ratios)
        for d in cluster:
            views_seen.add(d["view"])
            max_conf = max(max_conf, d["confidence"])
            area = _box_area(d["box"])
            if area > largest_area:
                largest_area = area
                largest = d

    return {
        "merged_count": len(clusters),
        "coverage": round(min(coverage_ratio, 1.0), 4),
        "views_seen": sorted(views_seen),
        "max_conf": round(float(max_conf), 4),
        "largest_box": largest["box"],
    }


def merge_views(views):
    if not views:
        raise ValueError("At least one image is required.")

    n = len(views)
    varieties = [v["variety"] for v in views]

    counts = Counter(v["class"] for v in varieties)
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    max_count = top[0][1]
    tied_classes = {cls for cls, count in counts.items() if count == max_count}
    candidates = [v for v in varieties if v["class"] in tied_classes]
    variety = max(candidates, key=lambda v: v["confidence"])
    variety = {"class": variety["class"], "confidence": round(float(variety["confidence"]), 4)}

    ear_areas = []
    for v in views:
        area = _box_area(v["traits"]["ear_box"])
        ear_areas.append(max(area, 1.0))
    max_ear_idx = max(range(n), key=lambda i: ear_areas[i])

    completeness = round(min(v["traits"]["kernel_completeness"] for v in views), 4)

    by_class = defaultdict(list)
    for view_idx, v in enumerate(views):
        for det in v["detections"]:
            if det["class"] == "corn_ear":
                continue
            det = dict(det)
            det["view"] = view_idx
            by_class[det["class"]].append(det)

    merged_defects = []
    total_coverage = 0.0
    for cls, dets in by_class.items():
        result = _merge_class(dets, n, ear_areas)
        result["class"] = cls
        merged_defects.append(result)
        total_coverage += result["coverage"]

    severe_present = any(cls in config.SEVERE_CLASSES for cls in by_class)

    return {
        "view_count": n,
        "variety": variety,
        "defects": merged_defects,
        "defect_coverage": round(min(total_coverage, 1.0), 4),
        "severe_present": bool(severe_present),
        "traits": {
            "ear_size": views[max_ear_idx]["traits"]["ear_size"],
            "ear_size_fraction": views[max_ear_idx]["traits"]["ear_size_fraction"],
            "kernel_completeness": completeness,
            "ear_box": views[max_ear_idx]["traits"]["ear_box"],
        },
    }
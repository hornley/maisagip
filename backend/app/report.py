import io

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import config

REPORTS = {}


def _font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _color_for(cls_name):
    palette = {
        "corn_ear": (76, 175, 80),
        "mold": (156, 39, 176),
        "insect_damage": (244, 67, 54),
        "discoloration": (255, 152, 0),
        "deformity": (33, 150, 243),
        "missing_kernels": (0, 188, 212),
        "ear_decay": (121, 85, 72),
        "shriveled_kernels": (255, 193, 7),
        "white_corn": (121, 85, 72),
        "yellow_sweet_corn": (251, 192, 45),
    }
    return palette.get(cls_name, (96, 125, 139))


def render_annotated(image: np.ndarray, detections, variety, grade_label):
    pil = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(pil)
    w, h = pil.size

    for det in detections:
        cls_name = det["class"]
        x0, y0, x1, y1 = (int(round(v)) for v in det["box"])
        color = _color_for(cls_name)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=max(2, w // 400))
        label = f"{cls_name} {det['confidence']:.2f}"
        font = _font(max(12, w // 45))
        text_w = int(draw.textlength(label, font=font))
        text_h = font.size
        draw.rectangle([x0, max(0, y0 - text_h - 4), x0 + text_w + 6, y0], fill=color)
        draw.text((x0 + 3, max(0, y0 - text_h - 2)), label, fill=(255, 255, 255), font=font)

    banner = f"{variety}  |  {grade_label}"
    font = _font(max(14, w // 30))
    bw = int(draw.textlength(banner, font=font)) + 16
    draw.rectangle([0, 0, bw, font.size + 10], fill=(33, 33, 33))
    draw.text((8, 6), banner, fill=(255, 255, 255), font=font)

    return np.array(pil)


def _grid_image(images, cols=2):
    if len(images) == 1:
        return images[0]
    cols = max(1, cols)
    rows = (len(images) + cols - 1) // cols
    pil_images = [Image.fromarray(img).convert("RGB") for img in images]
    cell_w = max(img.width for img in pil_images)
    cell_h = max(img.height for img in pil_images)
    grid = Image.new("RGB", (cell_w * cols, cell_h * rows), (240, 240, 240))
    for i, img in enumerate(pil_images):
        grid.paste(img, ((i % cols) * cell_w, (i // cols) * cell_h))
    return np.array(grid)


def new_report_id():
    from datetime import datetime
    import uuid

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


def _save(report_id, suffix, image: np.ndarray):
    path = config.REPORTS_DIR / f"{report_id}_{suffix}.png"
    Image.fromarray(image).convert("RGB").save(path)
    return path


def build_report(per_view, merged, engine_result):
    from datetime import datetime

    report_id = new_report_id()

    views_out = []
    annotated_views = []
    for idx, entry in enumerate(per_view):
        annotated = render_annotated(
            entry["image"],
            entry["detections"],
            entry["variety"]["class"],
            engine_result["grade_label"],
        )
        annotated_views.append(annotated)
        _save(report_id, f"v{idx}", annotated)
        views_out.append(
            {
                "view": idx + 1,
                "variety": entry["variety"],
                "defects": entry["detections"],
                "traits": entry["traits"],
                "image_width": entry["image"].shape[1],
                "image_height": entry["image"].shape[0],
                "image_url": f"/report/{report_id}/views/{idx}/image",
            }
        )

    grid = _grid_image(annotated_views, cols=2 if len(annotated_views) >= 2 else 1)
    _save(report_id, "grid", grid)

    REPORTS[report_id] = {
        "id": report_id,
        "inspection_date": datetime.now().isoformat(timespec="seconds"),
        "variety": merged["variety"],
        "defects": merged["defects"],
        "defect_coverage": merged["defect_coverage"],
        "severe_present": merged["severe_present"],
        "traits": merged["traits"],
        "grade": engine_result,
        "views": views_out,
        "view_count": merged["view_count"],
        "image_url": f"/report/{report_id}/views/grid/image",
    }
    return REPORTS[report_id]

"""Shared image decoding and canonical JPEG encoding helpers."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageOps

try:
    import pillow_heif
except ImportError:  # pragma: no cover - exercised by environments without HEIF support
    pillow_heif = None


SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}

_HEIF_OPENER_REGISTERED = False


def _register_heif_opener_once() -> None:
    global _HEIF_OPENER_REGISTERED

    if pillow_heif is not None and not _HEIF_OPENER_REGISTERED:
        pillow_heif.register_heif_opener()
        _HEIF_OPENER_REGISTERED = True


_register_heif_opener_once()


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode image bytes into an EXIF-transposed, three-channel RGB array."""
    try:
        with Image.open(BytesIO(data)) as image:
            rgb_image = ImageOps.exif_transpose(image).convert("RGB")
            return np.array(rgb_image)
    except Exception as exc:
        raise ValueError("Unable to decode image bytes") from exc


def encode_canonical_jpeg(image: Image.Image | np.ndarray) -> bytes:
    """Encode a PIL image or array as an RGB, metadata-free canonical JPEG."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    rgb_image = image.convert("RGB")
    fresh_image = Image.new("RGB", rgb_image.size)
    fresh_image.paste(rgb_image)
    output = BytesIO()
    fresh_image.save(output, format="JPEG", quality=95, optimize=True, exif=b"")
    return output.getvalue()


def normalize_image_bytes(data: bytes) -> tuple[np.ndarray, bytes]:
    """Decode image bytes and return the pixels plus canonical JPEG bytes."""
    image = decode_image_bytes(data)
    return image, encode_canonical_jpeg(image)

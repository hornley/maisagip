from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from backend.app.image_io import (
    decode_image_bytes,
    encode_canonical_jpeg,
    normalize_image_bytes,
)


def image_bytes(image: Image.Image, image_format: str) -> bytes:
    output = BytesIO()
    image.save(output, format=image_format)
    return output.getvalue()


@pytest.mark.parametrize("image_format", ["PNG", "JPEG"])
def test_decode_image_bytes_returns_rgb_array(image_format: str) -> None:
    source = Image.new("RGB", (3, 2), color=(20, 100, 200))

    decoded = decode_image_bytes(image_bytes(source, image_format))

    assert decoded.shape == (2, 3, 3)
    assert decoded.dtype == np.uint8
    np.testing.assert_allclose(decoded[0, 0], [20, 100, 200], atol=3)


def test_encode_canonical_jpeg_is_rgb_and_metadata_free() -> None:
    source = Image.new("RGBA", (4, 3), color=(10, 40, 200, 128))

    encoded = encode_canonical_jpeg(source)

    with Image.open(BytesIO(encoded)) as decoded:
        assert decoded.format == "JPEG"
        assert decoded.mode == "RGB"
        assert decoded.info.get("exif", b"") == b""
        assert decoded.size == source.size


def test_encode_canonical_jpeg_drops_comment_and_exif_metadata() -> None:
    source = Image.new("RGB", (4, 3), color=(10, 40, 200))
    source.info["comment"] = b"source comment"
    exif = source.getexif()
    exif[271] = "source camera"
    source.info["exif"] = exif.tobytes()

    encoded = encode_canonical_jpeg(source)

    with Image.open(BytesIO(encoded)) as decoded:
        assert "comment" not in decoded.info
        assert "exif" not in decoded.info


def test_normalize_image_bytes_returns_decoded_pixels_and_jpeg() -> None:
    source = Image.new("RGB", (2, 2), color=(120, 30, 80))

    pixels, encoded = normalize_image_bytes(image_bytes(source, "PNG"))

    assert pixels.shape == (2, 2, 3)
    with Image.open(BytesIO(encoded)) as normalized:
        assert normalized.format == "JPEG"
        assert normalized.mode == "RGB"


def test_decode_heic_applies_exif_orientation() -> None:
    pillow_heif = pytest.importorskip("pillow_heif")

    source = Image.new("RGB", (2, 3))
    source.putpixel((0, 0), (255, 0, 0))
    source.putpixel((1, 0), (0, 255, 0))
    source.putpixel((0, 2), (0, 0, 255))
    exif = source.getexif()
    exif[274] = 6  # Rotate 90 degrees clockwise.
    source.info["exif"] = exif.tobytes()

    heif_file = pillow_heif.from_pillow(source)
    output = BytesIO()
    heif_file.save(output)
    fixture = output.getvalue()

    decoded = decode_image_bytes(fixture)

    assert decoded.shape == (2, 3, 3)
    np.testing.assert_array_equal(decoded[0, 2], [255, 0, 0])

package com.maisagip.app.inference

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.ImageDecoder
import java.io.ByteArrayOutputStream
import java.io.File
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

data class RgbImage(val width: Int, val height: Int, val pixels: ByteArray) {
    init {
        require(width > 0 && height > 0 && pixels.size == width * height * 3)
    }

    fun bitmap(): Bitmap {
        val colors = IntArray(width * height)
        for (index in colors.indices) {
            val base = index * 3
            colors[index] = Color.rgb(
                pixels[base].toInt() and 0xff,
                pixels[base + 1].toInt() and 0xff,
                pixels[base + 2].toInt() and 0xff,
            )
        }
        return Bitmap.createBitmap(colors, width, height, Bitmap.Config.ARGB_8888)
    }

    companion object {
        fun decode(file: File): RgbImage {
            val bitmap = ImageDecoder.decodeBitmap(ImageDecoder.createSource(file)) { decoder, _, _ ->
                decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                decoder.isMutableRequired = false
            }
            return from(bitmap).also { bitmap.recycle() }
        }

        fun from(bitmap: Bitmap): RgbImage {
            val argb = IntArray(bitmap.width * bitmap.height)
            bitmap.getPixels(argb, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
            val bytes = ByteArray(argb.size * 3)
            argb.forEachIndexed { index, color ->
                val target = index * 3
                bytes[target] = Color.red(color).toByte()
                bytes[target + 1] = Color.green(color).toByte()
                bytes[target + 2] = Color.blue(color).toByte()
            }
            return RgbImage(bitmap.width, bitmap.height, bytes)
        }
    }
}

data class LetterboxedImage(
    val tensor: FloatArray,
    val width: Int,
    val height: Int,
    val scale: Float,
    val padLeft: Int,
    val padTop: Int,
    val modelSize: Int,
)

object ImagePreprocessor {
    fun decode(file: File) = RgbImage.decode(file)

    fun letterbox(image: RgbImage, size: Int = 640, padding: Int = 114): LetterboxedImage {
        val scale = min(size.toFloat() / image.width, size.toFloat() / image.height)
        val resizedWidth = Math.rint(image.width * scale.toDouble()).toInt().coerceAtLeast(1)
        val resizedHeight = Math.rint(image.height * scale.toDouble()).toInt().coerceAtLeast(1)
        val left = (size - resizedWidth) / 2
        val top = (size - resizedHeight) / 2
        val source = image.bitmap()
        val resized = Bitmap.createScaledBitmap(source, resizedWidth, resizedHeight, true)
        val canvasBitmap = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(canvasBitmap)
        canvas.drawColor(Color.rgb(padding, padding, padding))
        canvas.drawBitmap(resized, null, Rect(left, top, left + resizedWidth, top + resizedHeight), Paint(Paint.FILTER_BITMAP_FLAG))
        val chw = bitmapToChw(canvasBitmap)
        source.recycle()
        if (resized !== source) resized.recycle()
        canvasBitmap.recycle()
        return LetterboxedImage(chw, image.width, image.height, scale, left, top, size)
    }

    fun classifierInput(image: RgbImage, target: Int = 448): FloatArray {
        val resize = (target * 1.1f).roundToInt()
        val scale = resize.toFloat() / min(image.width, image.height)
        val width = (image.width * scale).roundToInt().coerceAtLeast(target)
        val height = (image.height * scale).roundToInt().coerceAtLeast(target)
        val source = image.bitmap()
        val resized = Bitmap.createScaledBitmap(source, width, height, true)
        val crop = Bitmap.createBitmap(
            resized,
            (width - target) / 2,
            (height - target) / 2,
            target,
            target,
        )
        val pixels = IntArray(target * target)
        crop.getPixels(pixels, 0, target, 0, 0, target, target)
        val chw = FloatArray(3 * target * target)
        val channelSize = target * target
        pixels.forEachIndexed { index, color ->
            chw[index] = (Color.red(color) / 255f - 0.485f) / 0.229f
            chw[channelSize + index] = (Color.green(color) / 255f - 0.456f) / 0.224f
            chw[2 * channelSize + index] = (Color.blue(color) / 255f - 0.406f) / 0.225f
        }
        source.recycle()
        if (resized !== source) resized.recycle()
        crop.recycle()
        return chw
    }

    fun maskedCrop(image: RgbImage, mask: BooleanArray): Pair<RgbImage, Box>? {
        require(mask.size == image.width * image.height)
        var minX = image.width
        var minY = image.height
        var maxX = -1
        var maxY = -1
        mask.forEachIndexed { index, included ->
            if (included) {
                val x = index % image.width
                val y = index / image.width
                minX = min(minX, x)
                minY = min(minY, y)
                maxX = max(maxX, x)
                maxY = max(maxY, y)
            }
        }
        if (maxX < minX || maxY < minY) return null
        val width = maxX - minX + 1
        val height = maxY - minY + 1
        val pixels = ByteArray(width * height * 3)
        for (y in 0 until height) for (x in 0 until width) {
            val sourceIndex = (y + minY) * image.width + x + minX
            val targetIndex = (y * width + x) * 3
            if (mask[sourceIndex]) {
                val sourceOffset = sourceIndex * 3
                pixels[targetIndex] = image.pixels[sourceOffset]
                pixels[targetIndex + 1] = image.pixels[sourceOffset + 1]
                pixels[targetIndex + 2] = image.pixels[sourceOffset + 2]
            } else {
                pixels[targetIndex] = 127
                pixels[targetIndex + 1] = 127
                pixels[targetIndex + 2] = 127
            }
        }
        return RgbImage(width, height, pixels) to Box(minX.toFloat(), minY.toFloat(), (maxX + 1).toFloat(), (maxY + 1).toFloat())
    }

    fun annotatedJpeg(image: RgbImage, detections: List<Detection>): ByteArray {
        val source = image.bitmap()
        val scale = min(1f, 1280f / max(image.width, image.height))
        val width = (image.width * scale).roundToInt().coerceAtLeast(1)
        val height = (image.height * scale).roundToInt().coerceAtLeast(1)
        val bitmap = if (scale < 1f) Bitmap.createScaledBitmap(source, width, height, true) else source
        val canvas = Canvas(bitmap)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.rgb(180, 83, 9)
            style = Paint.Style.STROKE
            strokeWidth = max(2f, min(width, height) / 160f)
        }
        detections.filter { it.className != "corn_ear" }.forEach { detection ->
            canvas.drawRect(
                RectF(
                    detection.box.x0 * scale,
                    detection.box.y0 * scale,
                    detection.box.x1 * scale,
                    detection.box.y1 * scale,
                ),
                paint,
            )
        }
        return ByteArrayOutputStream().use { output ->
            bitmap.compress(Bitmap.CompressFormat.JPEG, 86, output)
            bitmap.recycle()
            if (bitmap !== source) source.recycle()
            output.toByteArray()
        }
    }

    private fun bitmapToChw(bitmap: Bitmap): FloatArray {
        val width = bitmap.width
        val height = bitmap.height
        val colors = IntArray(width * height)
        bitmap.getPixels(colors, 0, width, 0, 0, width, height)
        val channelSize = width * height
        val values = FloatArray(channelSize * 3)
        colors.forEachIndexed { index, color ->
            values[index] = Color.red(color) / 255f
            values[channelSize + index] = Color.green(color) / 255f
            values[channelSize * 2 + index] = Color.blue(color) / 255f
        }
        return values
    }
}

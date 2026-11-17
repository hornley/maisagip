package com.maisagip.app.inference

import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.round

data class SegmentedEar(val mask: BooleanArray, val confidence: Float, val box: Box)

object YoloPostprocessor {
    private data class Candidate(
        val classIndex: Int,
        val rawClass: String,
        val confidence: Float,
        val box: Box,
        val maskBox: Box,
        val modelBox: Box,
        val maskCoefficients: FloatArray,
    )

    fun detections(
        output: TensorOutput,
        input: LetterboxedImage,
        classNames: List<String>,
        confidenceThreshold: Float = 0.50f,
        iouThreshold: Float = 0.70f,
    ): List<Detection> = candidates(output, input, classNames, confidenceThreshold)
        .let { nonMaximumSuppression(it, iouThreshold) }
        .take(300)
        .map { Detection(canonicalClass(it.rawClass), round4(it.confidence), it.box) }

    fun segments(
        output: TensorOutput,
        prototypeOutput: TensorOutput,
        input: LetterboxedImage,
        classNames: List<String>,
        confidenceThreshold: Float = 0.25f,
        iouThreshold: Float = 0.70f,
    ): List<SegmentedEar> {
        val candidates = candidates(output, input, classNames, confidenceThreshold)
            .filter { it.classIndex == 0 }
            .let { nonMaximumSuppression(it, iouThreshold) }
            .take(300)
        if (candidates.isEmpty()) return emptyList()
        val prototype = prototypeLayout(prototypeOutput)
        return candidates.mapNotNull { candidate ->
            val mask = decodeMask(candidate, prototype, input)
            if (mask.any { it }) SegmentedEar(mask, round4(candidate.confidence), candidate.box) else null
        }
    }

    private fun candidates(
        output: TensorOutput,
        input: LetterboxedImage,
        classNames: List<String>,
        threshold: Float,
    ): List<Candidate> {
        require(output.shape.size == 3 && output.shape[0] == 1L) { "Unexpected YOLO output shape." }
        require(classNames.isNotEmpty()) { "The model has no class names." }
        val channelsFirst = output.shape[1] < output.shape[2]
        val attributes = (if (channelsFirst) output.shape[1] else output.shape[2]).toInt()
        val anchors = (if (channelsFirst) output.shape[2] else output.shape[1]).toInt()
        require(attributes >= 4 + classNames.size) { "YOLO output has fewer channels than the class metadata." }
        val maskCount = attributes - 4 - classNames.size
        fun value(attribute: Int, anchor: Int): Float = if (channelsFirst) {
            output.values[attribute * anchors + anchor]
        } else {
            output.values[anchor * attributes + attribute]
        }

        val found = ArrayList<Candidate>()
        for (anchor in 0 until anchors) {
            var classIndex = -1
            var score = Float.NEGATIVE_INFINITY
            for (index in classNames.indices) {
                val candidateScore = value(4 + index, anchor)
                if (candidateScore > score) {
                    score = candidateScore
                    classIndex = index
                }
            }
            if (classIndex < 0 || score <= threshold || !score.isFinite()) continue
            val centerX = value(0, anchor)
            val centerY = value(1, anchor)
            val width = value(2, anchor)
            val height = value(3, anchor)
            val modelBox = Box(centerX - width / 2, centerY - height / 2, centerX + width / 2, centerY + height / 2)
            val originalBox = Box(
                ((modelBox.x0 - input.padLeft) / input.scale).coerceIn(0f, input.width.toFloat()),
                ((modelBox.y0 - input.padTop) / input.scale).coerceIn(0f, input.height.toFloat()),
                ((modelBox.x1 - input.padLeft) / input.scale).coerceIn(0f, input.width.toFloat()),
                ((modelBox.y1 - input.padTop) / input.scale).coerceIn(0f, input.height.toFloat()),
            )
            if (originalBox.area <= 0f) continue
            val coefficients = FloatArray(maskCount) { offset -> value(4 + classNames.size + offset, anchor) }
            val roundedBox = Box(round2(originalBox.x0), round2(originalBox.y0), round2(originalBox.x1), round2(originalBox.y1))
            found += Candidate(classIndex, classNames[classIndex], score, roundedBox, originalBox, modelBox, coefficients)
        }
        return found
    }

    private data class Prototype(val channels: Int, val height: Int, val width: Int, val values: FloatArray)

    private fun prototypeLayout(output: TensorOutput): Prototype {
        require(output.shape.size == 4 && output.shape[0] == 1L) { "Unexpected segmentation mask shape." }
        return Prototype(
            output.shape[1].toInt(),
            output.shape[2].toInt(),
            output.shape[3].toInt(),
            output.values,
        )
    }

    private fun decodeMask(candidate: Candidate, proto: Prototype, input: LetterboxedImage): BooleanArray {
        require(candidate.maskCoefficients.size == proto.channels) { "Segmentation coefficient count does not match mask prototypes." }
        val planeSize = proto.width * proto.height
        val logits = FloatArray(planeSize)
        for (y in 0 until proto.height) for (x in 0 until proto.width) {
            val offset = y * proto.width + x
            var logit = 0f
            for (channel in 0 until proto.channels) {
                logit += candidate.maskCoefficients[channel] * proto.values[channel * planeSize + offset]
            }
            logits[offset] = logit
        }
        val gain = min(proto.height.toFloat() / input.height, proto.width.toFloat() / input.width)
        val padWidth = (proto.width - roundEven(input.width * gain)) / 2f
        val padHeight = (proto.height - roundEven(input.height * gain)) / 2f
        val left = roundEven(padWidth - 0.1f).coerceIn(0, proto.width - 1)
        val top = roundEven(padHeight - 0.1f).coerceIn(0, proto.height - 1)
        val right = (proto.width - roundEven(padWidth + 0.1f)).coerceIn(left + 1, proto.width)
        val bottom = (proto.height - roundEven(padHeight + 0.1f)).coerceIn(top + 1, proto.height)
        val cropWidth = right - left
        val cropHeight = bottom - top
        val mask = BooleanArray(input.width * input.height)
        for (y in 0 until input.height) {
            val sourceY = (((y + 0.5f) * cropHeight / input.height) - 0.5f).coerceIn(0f, (cropHeight - 1).toFloat())
            val y0 = floor(sourceY).toInt()
            val y1 = min(y0 + 1, cropHeight - 1)
            val fy = sourceY - y0
            for (x in 0 until input.width) {
                if (x < candidate.maskBox.x0 || x >= candidate.maskBox.x1 || y < candidate.maskBox.y0 || y >= candidate.maskBox.y1) continue
                val sourceX = (((x + 0.5f) * cropWidth / input.width) - 0.5f).coerceIn(0f, (cropWidth - 1).toFloat())
                val x0 = floor(sourceX).toInt()
                val x1 = min(x0 + 1, cropWidth - 1)
                val fx = sourceX - x0
                val a = logits[(top + y0) * proto.width + left + x0]
                val b = logits[(top + y0) * proto.width + left + x1]
                val c = logits[(top + y1) * proto.width + left + x0]
                val d = logits[(top + y1) * proto.width + left + x1]
                val upper = a + (b - a) * fx
                val lower = c + (d - c) * fx
                mask[y * input.width + x] = upper + (lower - upper) * fy > 0f
            }
        }
        return mask
    }

    private fun nonMaximumSuppression(candidates: List<Candidate>, threshold: Float): List<Candidate> {
        val kept = ArrayList<Candidate>()
        candidates.sortedByDescending { it.confidence }.forEach { candidate ->
            if (kept.none { it.classIndex == candidate.classIndex && intersectionOverUnion(it.modelBox, candidate.modelBox) > threshold }) {
                kept += candidate
            }
        }
        return kept
    }

    private fun intersectionOverUnion(a: Box, b: Box): Float {
        val width = (min(a.x1, b.x1) - max(a.x0, b.x0)).coerceAtLeast(0f)
        val height = (min(a.y1, b.y1) - max(a.y0, b.y0)).coerceAtLeast(0f)
        val intersection = width * height
        val union = a.area + b.area - intersection
        return if (union <= 0f) 0f else intersection / union
    }

    private fun canonicalClass(name: String) = when (name) {
        "mold", "ear_decay" -> "decay"
        "corn-ear" -> "corn_ear"
        else -> name
    }

    private fun round2(value: Float) = round(value * 100f) / 100f
    private fun round4(value: Float) = round(value * 10_000f) / 10_000f
    private fun roundEven(value: Float) = Math.rint(value.toDouble()).toInt()
}

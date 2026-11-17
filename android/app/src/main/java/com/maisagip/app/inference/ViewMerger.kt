package com.maisagip.app.inference

import kotlin.math.max
import kotlin.math.min
import kotlin.math.round

data class VarietyPrediction(val className: String, val confidence: Float)

data class TraitSummary(
    val earSize: String,
    val earSizeFraction: Float,
    val kernelCompleteness: Float,
    val earBox: Box,
)

data class AnalyzedView(
    val variety: VarietyPrediction,
    val detections: List<Detection>,
    val traits: TraitSummary,
    val previewJpeg: ByteArray,
)

data class ViewDisagreement(
    val present: Boolean,
    val confident: Boolean,
    val confidentViews: List<Int>,
)

data class MergedInspection(
    val variety: VarietyPrediction,
    val defects: List<MergedDefect>,
    val defectCoverage: Float,
    val severePresent: Boolean,
    val earSize: String,
    val earSizeFraction: Float,
    val kernelCompleteness: Float,
    val earBox: Box,
    val viewCount: Int,
    val disagreement: ViewDisagreement,
)

object TraitEstimator {
    private const val SMALL_EAR = 0.28f
    private const val MEDIUM_EAR = 0.42f
    private const val MIN_EAR_BOX_AREA = 2500f

    fun estimate(image: RgbImage, detections: List<Detection>): TraitSummary {
        val cornBoxes = detections.filter { it.className == "corn_ear" }
        val candidateBoxes = detections.filter { it.box.area >= MIN_EAR_BOX_AREA }
        val ear = cornBoxes.maxByOrNull { it.box.area }?.box
            ?: candidateBoxes.maxByOrNull { it.box.area }?.box
            ?: Box(0f, 0f, image.width.toFloat(), image.height.toFloat())
        val length = max(ear.x1 - ear.x0, ear.y1 - ear.y0)
        val rawFraction = length / max(image.width, image.height).toFloat()
        val size = when {
            rawFraction < SMALL_EAR -> "small"
            rawFraction < MEDIUM_EAR -> "medium"
            else -> "large"
        }
        val earArea = max(ear.area, 1f)
        val missingArea = detections.filter { it.className == "missing_kernels" }.sumOf { it.box.area.toDouble() }.toFloat()
        val completeness = round4((1f - missingArea / earArea).coerceIn(0f, 1f))
        return TraitSummary(size, round4(rawFraction), completeness, ear)
    }
}

object ViewMerger {
    private const val DUPLICATE_AREA_RATIO = 1.25f
    private val severeClasses = setOf("decay", "insect_damage")

    private data class IndexedDetection(val view: Int, val detection: Detection)

    fun merge(views: List<AnalyzedView>, confidenceThreshold: Float): MergedInspection {
        require(views.isNotEmpty()) { "At least one image is required." }
        val counts = views.groupingBy { it.variety.className }.eachCount()
        val topCount = counts.values.maxOrNull() ?: 0
        val candidates = views.map { it.variety }.filter { counts[it.className] == topCount }
        val variety = candidates.maxBy { it.confidence }
        val confidentClasses = views.filter { it.variety.confidence >= confidenceThreshold }
            .map { it.variety.className }.toSet()
        val disagreement = ViewDisagreement(
            present = counts.size > 1,
            confident = confidentClasses.size > 1,
            confidentViews = views.mapIndexedNotNull { index, view ->
                if (view.variety.confidence >= confidenceThreshold) index + 1 else null
            },
        )
        val earAreas = views.map { max(it.traits.earBox.area, 1f) }
        val largestEarIndex = earAreas.indices.maxBy { earAreas[it] }
        val byClass = linkedMapOf<String, MutableList<IndexedDetection>>()
        views.forEachIndexed { viewIndex, view ->
            view.detections.filter { it.className != "corn_ear" }.forEach { detection ->
                byClass.getOrPut(detection.className) { mutableListOf() }
                    .add(IndexedDetection(viewIndex, detection))
            }
        }

        val defects = byClass.map { (className, detections) ->
            mergeClass(className, detections, views.size, earAreas)
        }
        return MergedInspection(
            variety = variety,
            defects = defects,
            defectCoverage = round4(defects.sumOf { it.coverage.toDouble() }.toFloat().coerceAtMost(1f)),
            severePresent = byClass.keys.any { it in severeClasses },
            earSize = views[largestEarIndex].traits.earSize,
            earSizeFraction = views[largestEarIndex].traits.earSizeFraction,
            kernelCompleteness = views.minOf { it.traits.kernelCompleteness },
            earBox = views[largestEarIndex].traits.earBox,
            viewCount = views.size,
            disagreement = disagreement,
        )
    }

    private fun mergeClass(
        className: String,
        detections: List<IndexedDetection>,
        viewCount: Int,
        earAreas: List<Float>,
    ): MergedDefect {
        val byView = List(viewCount) { view -> detections.filter { it.view == view }.map { it.detection } }
        val mergeable = (byView.maxOfOrNull { it.size } ?: 0) <= 1
        val clusters: List<List<IndexedDetection>> = if (mergeable) {
            val parent = IntArray(viewCount) { it }
            fun root(start: Int): Int {
                var index = start
                while (parent[index] != index) {
                    parent[index] = parent[parent[index]]
                    index = parent[index]
                }
                return index
            }
            fun union(first: Int, second: Int) {
                val a = root(first)
                val b = root(second)
                if (a != b) parent[b] = a
            }
            for (index in 0 until viewCount) {
                val next = (index + 1) % viewCount
                val first = byView[index].firstOrNull()
                val second = byView[next].firstOrNull()
                if (first != null && second != null) {
                    val low = min(first.box.area, second.box.area)
                    val high = max(first.box.area, second.box.area)
                    if (high <= DUPLICATE_AREA_RATIO * low) union(index, next)
                }
            }
            byView.mapIndexedNotNull { index, rows -> rows.firstOrNull()?.let { root(index) to IndexedDetection(index, it) } }
                .groupBy({ it.first }, { it.second }).values.toList()
        } else {
            detections.map { listOf(it) }
        }

        var coverage = 0f
        var confidence = 0f
        val seenViews = linkedSetOf<Int>()
        var largest = clusters.first().first().detection
        clusters.forEach { cluster ->
            val clusterCoverage = cluster.maxOf { it.detection.box.area / earAreas[it.view] }
            coverage += clusterCoverage
            cluster.forEach { indexed ->
                val detection = indexed.detection
                seenViews += indexed.view + 1
                confidence = max(confidence, detection.confidence)
                if (detection.box.area > largest.box.area) largest = detection
            }
        }
        return MergedDefect(className, clusters.size, round4(coverage.coerceAtMost(1f)), round4(confidence), seenViews.sorted(), largest.box)
    }

}

private fun round4(value: Float) = round(value * 10_000f) / 10_000f

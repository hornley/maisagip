package com.maisagip.app.inference

import android.content.Context
import java.io.File
import kotlin.math.exp
import kotlin.math.round

data class ViewPredictionSummary(val number: Int, val variety: String, val confidence: Float)

class NoEarsFoundException : Exception("No ears were found. Retake the photo or choose another image.")

class InconsistentVarietyException(val predictions: List<ViewPredictionSummary>) :
    Exception("The selected views show different corn varieties. Review the photos and replace any incorrect view.")

class InspectionPipeline(context: Context, private val models: OnnxModels) {
    private val appContext = context.applicationContext
    private val grader = GradeRules(appContext)
    private val recommendations = RecommendationRules()

    fun inspect(files: List<File>, confidenceThreshold: Float): List<EarFinding> {
        require(files.isNotEmpty()) { "Select at least one photo." }
        require(confidenceThreshold in 0.50f..0.80f) { "Confidence must be between 0.50 and 0.80." }
        return if (files.size == 1) inspectSegmentedPhoto(files.single(), confidenceThreshold)
        else inspectViews(files, confidenceThreshold)
    }

    private fun inspectSegmentedPhoto(file: File, confidenceThreshold: Float): List<EarFinding> {
        val image = ImagePreprocessor.decode(file)
        val prepared = ImagePreprocessor.letterbox(image)
        val outputs = models.run(ModelTask.SEGMENTER, prepared.tensor, longArrayOf(1, 3, 640, 640)).values.toList()
        require(outputs.size >= 2) { "The ear segmentation model returned incomplete outputs." }
        val ears = YoloPostprocessor.segments(
            output = outputs[0],
            prototypeOutput = outputs[1],
            input = prepared,
            classNames = models.classNames(ModelTask.SEGMENTER),
        )
        if (ears.isEmpty()) throw NoEarsFoundException()

        return ears.mapIndexedNotNull { index, ear ->
            val crop = ImagePreprocessor.maskedCrop(image, ear.mask)?.first ?: return@mapIndexedNotNull null
            val view = analyzeView(crop, confidenceThreshold)
            val merged = ViewMerger.merge(listOf(view), confidenceThreshold)
            val result = buildFinding(merged, confidenceThreshold, viewCount = 1)
            result.copy(
                previewFileName = "ear-%03d.jpg".format(index + 1),
                previewJpeg = view.previewJpeg,
            )
        }.ifEmpty { throw NoEarsFoundException() }
    }

    private fun inspectViews(files: List<File>, confidenceThreshold: Float): List<EarFinding> {
        val views = files.map { file -> analyzeView(ImagePreprocessor.decode(file), confidenceThreshold) }
        val confidentPredictions = views.mapIndexed { index, view ->
            ViewPredictionSummary(index + 1, view.variety.className, view.variety.confidence)
        }.filter { it.confidence >= confidenceThreshold }
        if (confidentPredictions.map { it.variety }.distinct().size > 1) {
            throw InconsistentVarietyException(confidentPredictions)
        }
        val merged = ViewMerger.merge(views, confidenceThreshold)
        val largestView = views.indices.maxBy { views[it].traits.earBox.area }
        val result = buildFinding(merged, confidenceThreshold, views.size)
        return listOf(result.copy(
            previewFileName = "ear-view.jpg",
            previewJpeg = views[largestView].previewJpeg,
        ))
    }

    private fun analyzeView(image: RgbImage, confidenceThreshold: Float): AnalyzedView {
        val classPrediction = classify(image)
        val prepared = ImagePreprocessor.letterbox(image)
        val outputs = models.run(ModelTask.DETECTOR, prepared.tensor, longArrayOf(1, 3, 640, 640))
        val detectorOutput = outputs.values.firstOrNull()
            ?: error("The defect detector returned no output.")
        var detections = YoloPostprocessor.detections(
            output = detectorOutput,
            input = prepared,
            classNames = models.classNames(ModelTask.DETECTOR),
            confidenceThreshold = confidenceThreshold,
        )
        if (detections.isEmpty()) {
            detections = listOf(Detection("corn_ear", 0.8f, Box(0f, 0f, image.width.toFloat(), image.height.toFloat())))
        }
        return AnalyzedView(
            classPrediction,
            detections,
            TraitEstimator.estimate(image, detections),
            ImagePreprocessor.annotatedJpeg(image, detections),
        )
    }

    private fun classify(image: RgbImage): VarietyPrediction {
        val input = ImagePreprocessor.classifierInput(image)
        val output = models.run(ModelTask.CLASSIFIER, input, longArrayOf(1, 3, 448, 448)).values.firstOrNull()
            ?: error("The variety classifier returned no output.")
        require(output.values.size == 2) { "The variety classifier must return two logits." }
        val maxLogit = output.values.maxOrNull() ?: error("The variety classifier returned no logits.")
        val probabilities = output.values.map { exp(it - maxLogit) }
        val total = probabilities.sum()
        val classNames = models.classNames(ModelTask.CLASSIFIER)
        require(classNames.size == 2) { "The classifier class metadata is invalid." }
        val index = probabilities.indices.maxBy { probabilities[it] }
        val confidence = round((probabilities[index] / total) * 10_000f) / 10_000f
        return VarietyPrediction(classNames[index], confidence)
    }

    private fun buildFinding(merged: MergedInspection, threshold: Float, viewCount: Int): EarFinding {
        val grade = grader.evaluate(merged.defectCoverage, merged.severePresent)
        val confidenceReasons = buildList {
            if (merged.variety.confidence < threshold) add("Variety result needs review.")
            if (merged.defects.any { it.confidence < 0.35f }) add("One or more defect results need review.")
            if (merged.disagreement.present) add("The views disagree on variety.")
        }
        return EarFinding(
            gradeKey = grade.key,
            gradeLabel = grade.label,
            varietyClass = merged.variety.className,
            varietyConfidence = merged.variety.confidence,
            defectCoverage = merged.defectCoverage,
            severePresent = merged.severePresent,
            earSize = merged.earSize,
            kernelCompleteness = merged.kernelCompleteness,
            defects = merged.defects,
            gradeReasons = grade.reasons,
            utilization = grade.utilization,
            utilizationReason = grade.utilizationReason,
            recommendations = recommendations.build(grade.key, merged.defects),
            needsReinspection = confidenceReasons.isNotEmpty(),
            reviewReasons = confidenceReasons,
            viewCount = viewCount,
            previewFileName = "preview.jpg",
        )
    }
}

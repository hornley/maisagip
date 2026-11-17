package com.maisagip.app.inference

import org.json.JSONArray
import org.json.JSONObject

data class Box(val x0: Float, val y0: Float, val x1: Float, val y1: Float) {
    val area: Float get() = ((x1 - x0).coerceAtLeast(0f) * (y1 - y0).coerceAtLeast(0f))
}

data class Detection(
    val className: String,
    val confidence: Float,
    val box: Box,
)

data class MergedDefect(
    val className: String,
    val count: Int,
    val coverage: Float,
    val confidence: Float,
    val viewsSeen: List<Int>,
    val largestBox: Box,
)

data class RecommendationOption(val label: String)

data class EarFinding(
    val gradeKey: String,
    val gradeLabel: String,
    val varietyClass: String,
    val varietyConfidence: Float,
    val defectCoverage: Float,
    val severePresent: Boolean,
    val earSize: String,
    val kernelCompleteness: Float,
    val defects: List<MergedDefect>,
    val gradeReasons: List<String>,
    val utilization: String,
    val utilizationReason: String,
    val recommendations: List<RecommendationOption>,
    val needsReinspection: Boolean,
    val reviewReasons: List<String>,
    val viewCount: Int,
    val previewFileName: String,
    val previewJpeg: ByteArray? = null,
) {
    fun toJson(): String = JSONObject().apply {
        put("gradeKey", gradeKey)
        put("gradeLabel", gradeLabel)
        put("varietyClass", varietyClass)
        put("varietyConfidence", varietyConfidence.toDouble())
        put("defectCoverage", defectCoverage.toDouble())
        put("severePresent", severePresent)
        put("earSize", earSize)
        put("kernelCompleteness", kernelCompleteness.toDouble())
        put("defects", JSONArray().apply {
            defects.forEach { defect ->
                put(JSONObject().apply {
                    put("className", defect.className)
                    put("count", defect.count)
                    put("coverage", defect.coverage.toDouble())
                    put("confidence", defect.confidence.toDouble())
                    put("viewsSeen", JSONArray(defect.viewsSeen))
                    put("largestBox", defect.largestBox.toJson())
                })
            }
        })
        put("gradeReasons", JSONArray(gradeReasons))
        put("utilization", utilization)
        put("utilizationReason", utilizationReason)
        put("recommendations", JSONArray().apply {
            recommendations.forEach { put(it.label) }
        })
        put("needsReinspection", needsReinspection)
        put("reviewReasons", JSONArray(reviewReasons))
        put("viewCount", viewCount)
        put("previewFileName", previewFileName)
    }.toString()

    companion object {
        fun fromJson(json: String): EarFinding {
            val value = JSONObject(json)
            val defects = value.getJSONArray("defects").let { rows ->
                (0 until rows.length()).map { index ->
                    val item = rows.getJSONObject(index)
                    MergedDefect(
                        className = item.getString("className"),
                        count = item.getInt("count"),
                        coverage = item.getDouble("coverage").toFloat(),
                        confidence = item.getDouble("confidence").toFloat(),
                        viewsSeen = item.getJSONArray("viewsSeen").toIntList(),
                        largestBox = item.getJSONObject("largestBox").toBox(),
                    )
                }
            }
            return EarFinding(
                gradeKey = value.getString("gradeKey"),
                gradeLabel = value.getString("gradeLabel"),
                varietyClass = value.getString("varietyClass"),
                varietyConfidence = value.getDouble("varietyConfidence").toFloat(),
                defectCoverage = value.getDouble("defectCoverage").toFloat(),
                severePresent = value.getBoolean("severePresent"),
                earSize = value.getString("earSize"),
                kernelCompleteness = value.getDouble("kernelCompleteness").toFloat(),
                defects = defects,
                gradeReasons = value.getJSONArray("gradeReasons").toStringList(),
                utilization = value.getString("utilization"),
                utilizationReason = value.getString("utilizationReason"),
                recommendations = value.getJSONArray("recommendations").toStringList().map(::RecommendationOption),
                needsReinspection = value.getBoolean("needsReinspection"),
                reviewReasons = value.getJSONArray("reviewReasons").toStringList(),
                viewCount = value.getInt("viewCount"),
                previewFileName = value.getString("previewFileName"),
                previewJpeg = null,
            )
        }
    }
}

internal fun Box.toJson() = JSONObject().apply {
    put("x0", x0.toDouble())
    put("y0", y0.toDouble())
    put("x1", x1.toDouble())
    put("y1", y1.toDouble())
}

internal fun JSONObject.toBox() = Box(
    getDouble("x0").toFloat(),
    getDouble("y0").toFloat(),
    getDouble("x1").toFloat(),
    getDouble("y1").toFloat(),
)

private fun JSONArray.toIntList() = (0 until length()).map { getInt(it) }
private fun JSONArray.toStringList() = (0 until length()).map { getString(it) }

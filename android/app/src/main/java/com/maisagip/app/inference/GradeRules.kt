package com.maisagip.app.inference

import android.content.Context
import org.json.JSONObject

data class GradeResult(
    val key: String,
    val label: String,
    val reasons: List<String>,
    val utilization: String,
    val utilizationReason: String,
)

class GradeRules(context: Context) {
    private val config = context.assets.open("inspection_rules.json").bufferedReader().use {
        JSONObject(it.readText())
    }

    fun evaluate(coverage: Float, severePresent: Boolean): GradeResult {
        require(coverage.isFinite() && coverage in 0f..1f)
        val grades = config.getJSONObject("grades")
        val order = config.getJSONArray("grade_order")
        var gradeKey = "Reject"
        var gradeLabel = config.getString("below_grade_label")
        var reason = ""

        for (index in 0 until order.length()) {
            val key = order.getString(index)
            val spec = grades.getJSONObject(key)
            val threshold = if (spec.has("coverage_at_most")) {
                spec.getDouble("coverage_at_most").toFloat()
            } else {
                spec.getDouble("coverage_less_than").toFloat()
            }
            val passes = if (spec.has("coverage_at_most")) coverage <= threshold else coverage < threshold
            if (passes) {
                gradeKey = key
                gradeLabel = spec.getString("label")
                reason = "Meets $gradeLabel criteria: ${percent(coverage)} coverage is " +
                    if (spec.has("coverage_at_most")) "at most ${percent(threshold)}." else "below ${percent(threshold)}."
                break
            }
        }
        if (reason.isEmpty()) {
            val finalGrade = grades.getJSONObject(order.getString(order.length() - 1))
            val cutoff = finalGrade.getDouble("coverage_less_than").toFloat()
            reason = "Reject: ${percent(coverage)} coverage is at or above the Class II cutoff of ${percent(cutoff)}."
        }

        val utilizationRules = config.getJSONArray("utilization")
        for (index in 0 until utilizationRules.length()) {
            val rule = utilizationRules.getJSONObject(index)
            val matches = when (rule.getString("condition")) {
                "coverage_ge" -> coverage >= rule.getDouble("value").toFloat()
                "coverage_lt" -> coverage < rule.getDouble("value").toFloat()
                "any_severe" -> severePresent
                "always" -> true
                else -> false
            }
            if (matches) {
                return GradeResult(
                    key = gradeKey,
                    label = gradeLabel,
                    reasons = listOf(reason),
                    utilization = rule.getString("recommend"),
                    utilizationReason = rule.getString("reason"),
                )
            }
        }
        return GradeResult(gradeKey, gradeLabel, listOf(reason), "reject", "No applicable utilization rule.")
    }

    private fun percent(value: Float) = "%.2f".format(java.util.Locale.ROOT, value * 100).trimEnd('0').trimEnd('.') + "%"
}

package com.maisagip.app.inference

class RecommendationRules {
    private val labels = linkedMapOf(
        "local_market" to "Local Market",
        "export" to "Export",
        "premium_sale" to "Premium Sale",
        "food_processing_canned" to "Processed/Canned",
        "poultry_feed" to "Poultry Feed",
    )
    private val eligible = mapOf(
        "ExtraClass" to listOf("local_market", "export", "premium_sale", "food_processing_canned"),
        "ClassI" to listOf("local_market", "premium_sale", "food_processing_canned"),
        "ClassII" to listOf("local_market"),
        "Reject" to listOf("poultry_feed"),
    )

    fun build(grade: String, defects: List<MergedDefect>): List<RecommendationOption> {
        val safetyDefectPresent = defects.any { it.className == "decay" }
        var channels = eligible[grade].orEmpty()
        if (safetyDefectPresent && grade != "Reject") channels = channels.filter { it == "local_market" }
        return channels.mapNotNull { channel ->
            val label = labels[channel] ?: return@mapNotNull null
            RecommendationOption(label)
        }
    }
}

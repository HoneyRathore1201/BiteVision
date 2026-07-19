package com.bite.vision.ml

/**
 * Data class holding a single classification result from the TFLite model.
 */
data class ClassificationResult(
    val category: Category,
    val displayLabel: String,
    val subLabel: String,
    val confidence: Float,
    val topPredictions: List<Pair<String, Float>>
) {
    enum class Category {
        HUMAN,
        ANIMAL,
        OTHER
    }
}

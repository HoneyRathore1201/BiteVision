package com.bite.vision.ml

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Dual-model image classifier:
 *
 *  1. PRIMARY:  COCO SSD MobileNetV1 (detect.tflite)
 *     - Object detection model: 90 COCO classes
 *     - Outputs: bounding boxes [1,10,4], scores [1,10], classes [1,10], count [1]
 *     - Has `person` (idx 0) + 15 animal classes
 *
 *  2. FALLBACK: MobileNetV1 ImageNet (mobilenet_v1_1.0_224_quant.tflite)
 *     - Image classification: 1001 ImageNet classes
 *     - Used when SSD confidence < threshold
 *
 * Category mapping:
 *   COCO idx 0           → HUMAN
 *   COCO idx 14–23       → ANIMAL (bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe)
 *   ImageNet animal names → ANIMAL
 *   Everything else      → OTHER
 */
class ImageClassifier(private val context: Context) {

    // SSD Detection model
    private val detectionInterpreter: Interpreter
    private val cocoLabels: List<String>

    // MobileNet Classification fallback
    private val classificationInterpreter: Interpreter
    private val imagenetLabels: List<String>

    private val imageProcessorSSD: ImageProcessor
    private val imageProcessorMobileNet: ImageProcessor

    // COCO class indices mapped to labelmap.txt line indices (1-indexed COCO categories)
    // person=1, bird=16, cat=17, dog=18, horse=19, sheep=20, cow=21,
    // elephant=22, bear=23, zebra=24, giraffe=25
    private val humanCocoIndices = setOf(1)
    private val animalCocoIndices = setOf(16, 17, 18, 19, 20, 21, 22, 23, 24, 25)

    private val animalKeywords = listOf(
        "cat", "dog", "bird", "fish", "horse", "cow", "sheep", "pig", "rabbit",
        "hamster", "mouse", "rat", "elephant", "lion", "tiger", "bear", "deer",
        "wolf", "fox", "monkey", "gorilla", "chimpanzee", "panda", "koala",
        "kangaroo", "zebra", "giraffe", "hippo", "rhinoceros", "crocodile",
        "alligator", "snake", "lizard", "turtle", "frog", "toad", "eagle",
        "owl", "parrot", "penguin", "duck", "goose", "chicken", "rooster",
        "turkey", "bee", "butterfly", "spider", "ant", "beetle",
        "lobster", "crab", "shrimp", "whale", "dolphin", "seal",
        "shark", "goldfish", "carp", "salmon", "squirrel",
        "hedgehog", "otter", "beaver", "camel", "llama", "bison",
        "antelope", "jaguar", "leopard", "cheetah", "lynx", "hyena",
        "baboon", "macaque", "orangutan", "flamingo", "pelican",
        "hawk", "falcon", "sparrow", "crow", "raven", "pigeon",
        "gecko", "iguana", "chameleon", "tortoise", "salamander",
        "octopus", "squid", "jellyfish", "starfish", "siamang",
        "warthog", "marmot", "porcupine", "skunk", "armadillo",
        "tench", "great white shark", "tiger shark", "cock", "hen",
        "ostrich", "goldfinch", "robin", "jay", "magpie",
        "European fire salamander", "spotted salamander", "axolotl"
    )

    init {
        val options4t = Interpreter.Options().apply { numThreads = 4 }

        // Load detection model (SSD COCO)
        val detModelBuffer = FileUtil.loadMappedFile(context, DETECTION_MODEL)
        detectionInterpreter = Interpreter(detModelBuffer, options4t)
        cocoLabels = loadLabels(COCO_LABELS)

        // Load classification model (MobileNetV1 quant)
        val clsModelBuffer = FileUtil.loadMappedFile(context, CLASSIFICATION_MODEL)
        classificationInterpreter = Interpreter(clsModelBuffer, options4t)
        imagenetLabels = loadLabels(IMAGENET_LABELS)

        // SSD expects 300×300
        imageProcessorSSD = ImageProcessor.Builder()
            .add(ResizeOp(300, 300, ResizeOp.ResizeMethod.BILINEAR))
            .build()

        // MobileNet expects 224×224
        imageProcessorMobileNet = ImageProcessor.Builder()
            .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
            .build()

        Log.d(TAG, "Classifiers ready. COCO labels: ${cocoLabels.size}, ImageNet: ${imagenetLabels.size}")
    }

    /**
     * Classify bitmap — tries SSD detection first, falls back to MobileNet classification.
     */
    fun classify(bitmap: Bitmap): ClassificationResult {
        // --- Step 1: Try SSD object detection ---
        val ssdResult = runSSDDetection(bitmap)
        if (ssdResult != null && ssdResult.confidence > SSD_THRESHOLD) {
            Log.d(TAG, "SSD result: ${ssdResult.displayLabel} @ ${ssdResult.confidence}")
            return ssdResult
        }

        // --- Step 2: Fallback to MobileNet classification ---
        Log.d(TAG, "SSD confidence low, falling back to MobileNet")
        return runMobileNetClassification(bitmap)
    }

    // ─────────────────────────────────────────────────────────────
    // SSD Detection
    // ─────────────────────────────────────────────────────────────

    private fun runSSDDetection(bitmap: Bitmap): ClassificationResult? {
        try {
            val tensorImage = TensorImage.fromBitmap(bitmap)
            val processed = imageProcessorSSD.process(tensorImage)

            // SSD outputs: boxes[1,10,4], classes[1,10], scores[1,10], count[1]
            val outputBoxes = Array(1) { Array(10) { FloatArray(4) } }
            val outputClasses = Array(1) { FloatArray(10) }
            val outputScores = Array(1) { FloatArray(10) }
            val outputCount = FloatArray(1)

            val outputMap = mapOf<Int, Any>(
                0 to outputBoxes,
                1 to outputClasses,
                2 to outputScores,
                3 to outputCount
            )

            detectionInterpreter.runForMultipleInputsOutputs(
                arrayOf(processed.buffer), outputMap
            )

            val count = outputCount[0].toInt().coerceAtMost(10)
            if (count == 0) return null

            // Find the highest-scoring detection
            var bestScore = 0f
            var bestClassIdx = -1
            val topPredictions = mutableListOf<Pair<String, Float>>()

            for (i in 0 until count) {
                val score = outputScores[0][i]
                val rawClassIdx = outputClasses[0][i].toInt()
                val classIdx = rawClassIdx + 1 // COCO SSD output is 0-indexed; labelmap.txt has offset of 1 due to "???" at index 0
                val label = cocoLabels.getOrElse(classIdx) { "unknown" }
                Log.d(TAG, "Prediction $i: rawClassIdx=$rawClassIdx, mappedClassIdx=$classIdx, label=$label, score=$score")
                if (label != "???") {
                    topPredictions.add(Pair(label.replaceFirstChar { it.uppercase() }, score))
                    if (score > bestScore) {
                        bestScore = score
                        bestClassIdx = classIdx
                    }
                }
            }

            if (bestClassIdx < 0) return null

            val bestLabel = cocoLabels.getOrElse(bestClassIdx) { "unknown" }
            val category = categorizeCoco(bestClassIdx)
            val (displayLabel, subLabel) = getDisplayLabels(category, bestLabel)

            return ClassificationResult(
                category = category,
                displayLabel = displayLabel,
                subLabel = subLabel,
                confidence = bestScore,
                topPredictions = topPredictions.sortedByDescending { it.second }.take(5)
            )
        } catch (e: Exception) {
            Log.e(TAG, "SSD detection failed", e)
            return null
        }
    }

    private fun categorizeCoco(classIdx: Int): ClassificationResult.Category = when {
        classIdx in humanCocoIndices -> ClassificationResult.Category.HUMAN
        classIdx in animalCocoIndices -> ClassificationResult.Category.ANIMAL
        else -> ClassificationResult.Category.OTHER
    }

    // ─────────────────────────────────────────────────────────────
    // MobileNet Classification (fallback)
    // ─────────────────────────────────────────────────────────────

    private fun runMobileNetClassification(bitmap: Bitmap): ClassificationResult {
        try {
            val tensorImage = TensorImage.fromBitmap(bitmap)
            val processed = imageProcessorMobileNet.process(tensorImage)

            val outputArray = Array(1) { ByteArray(imagenetLabels.size) }
            classificationInterpreter.run(processed.buffer, outputArray)

            // Dequantize uint8 → float [0,1]
            val scores = outputArray[0].mapIndexed { idx, byte ->
                val label = imagenetLabels.getOrElse(idx) { "unknown" }
                Pair(label, (byte.toInt() and 0xFF) / 255.0f)
            }.sortedByDescending { it.second }

            val top = scores.firstOrNull() ?: return unknownResult()
            val category = categorizeKeyword(top.first)
            val (displayLabel, subLabel) = getDisplayLabels(category, top.first)

            return ClassificationResult(
                category = category,
                displayLabel = displayLabel,
                subLabel = subLabel,
                confidence = top.second,
                topPredictions = scores.take(5).map {
                    Pair(it.first.replaceFirstChar { c -> c.uppercase() }, it.second)
                }
            )
        } catch (e: Exception) {
            Log.e(TAG, "MobileNet classification failed", e)
            return unknownResult()
        }
    }

    private fun categorizeKeyword(label: String): ClassificationResult.Category {
        val lower = label.lowercase()
        return when {
            animalKeywords.any { lower.contains(it) } -> ClassificationResult.Category.ANIMAL
            else -> ClassificationResult.Category.OTHER
        }
    }

    // ─────────────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────────────

    private fun getDisplayLabels(
        category: ClassificationResult.Category,
        rawLabel: String
    ): Pair<String, String> = when (category) {
        ClassificationResult.Category.HUMAN ->
            Pair("Human Detected", "Person identified in this image")
        ClassificationResult.Category.ANIMAL ->
            Pair("Animal Detected", rawLabel.replaceFirstChar { it.uppercase() } + " detected")
        ClassificationResult.Category.OTHER ->
            Pair("Object / Scene", rawLabel.replaceFirstChar { it.uppercase() } + " identified")
    }

    private fun loadLabels(filename: String): List<String> = try {
        val reader = BufferedReader(InputStreamReader(context.assets.open(filename)))
        reader.readLines().map { it.trim() }.filter { it.isNotEmpty() }
    } catch (e: Exception) {
        Log.e(TAG, "Failed to load labels from $filename", e)
        emptyList()
    }

    private fun unknownResult() = ClassificationResult(
        category = ClassificationResult.Category.OTHER,
        displayLabel = "Unknown",
        subLabel = "Could not classify image",
        confidence = 0f,
        topPredictions = emptyList()
    )

    fun close() {
        detectionInterpreter.close()
        classificationInterpreter.close()
    }

    companion object {
        private const val TAG = "ImageClassifier"
        private const val DETECTION_MODEL = "detect.tflite"
        private const val CLASSIFICATION_MODEL = "mobilenet_v1_1.0_224_quant.tflite"
        private const val COCO_LABELS = "labelmap.txt"
        private const val IMAGENET_LABELS = "labels.txt"
        private const val SSD_THRESHOLD = 0.35f
    }
}

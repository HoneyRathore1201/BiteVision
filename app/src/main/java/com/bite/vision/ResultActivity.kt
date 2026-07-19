package com.bite.vision

import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.bumptech.glide.Glide
import com.bite.vision.databinding.ActivityResultBinding
import com.bite.vision.ml.ClassificationResult
import com.bite.vision.ml.ImageClassifier

class ResultActivity : AppCompatActivity() {

    private lateinit var binding: ActivityResultBinding
    private lateinit var classifier: ImageClassifier

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityResultBinding.inflate(layoutInflater)
        setContentView(binding.root)

        classifier = ImageClassifier(this)

        val uriString = intent.getStringExtra(EXTRA_IMAGE_URI)
        if (uriString == null) {
            Toast.makeText(this, "No image provided", Toast.LENGTH_SHORT).show()
            finish()
            return
        }

        val uri = Uri.parse(uriString)
        analyzeImage(uri)
        setupClickListeners()
    }

    private fun analyzeImage(uri: Uri) {
        binding.progressBar.visibility = View.VISIBLE
        binding.contentLayout.visibility = View.GONE

        Glide.with(this).load(uri).into(binding.ivImage)

        Thread {
            try {
                val inputStream = contentResolver.openInputStream(uri)
                val bitmap = BitmapFactory.decodeStream(inputStream)
                inputStream?.close()

                if (bitmap == null) {
                    runOnUiThread {
                        Toast.makeText(this, "Could not decode image", Toast.LENGTH_SHORT).show()
                        finish()
                    }
                    return@Thread
                }

                val result = classifier.classify(bitmap)
                runOnUiThread { displayResult(result) }

            } catch (e: Exception) {
                runOnUiThread {
                    Toast.makeText(this, "Analysis failed: ${e.message}", Toast.LENGTH_LONG).show()
                    finish()
                }
            }
        }.start()
    }

    private fun displayResult(result: ClassificationResult) {
        binding.progressBar.visibility = View.GONE
        binding.contentLayout.visibility = View.VISIBLE

        binding.tvMainLabel.text = result.displayLabel
        binding.tvSubLabel.text = result.subLabel
        binding.tvConfidenceValue.text = "%.1f%%".format(result.confidence * 100)
        binding.tvEmoji.text = when (result.category) {
            ClassificationResult.Category.HUMAN -> "👤"
            ClassificationResult.Category.ANIMAL -> "🐾"
            ClassificationResult.Category.OTHER -> "🔍"
        }

        val (bgColor, accentColor) = when (result.category) {
            ClassificationResult.Category.HUMAN -> Pair(
                R.color.human_color, R.color.human_accent
            )
            ClassificationResult.Category.ANIMAL -> Pair(
                R.color.animal_color, R.color.animal_accent
            )
            ClassificationResult.Category.OTHER -> Pair(
                R.color.other_color, R.color.other_accent
            )
        }

        binding.resultCard.setCardBackgroundColor(ContextCompat.getColor(this, bgColor))
        binding.tvCategoryBadge.backgroundTintList =
            ContextCompat.getColorStateList(this, accentColor)
        binding.tvCategoryBadge.text = result.category.name
        binding.confidenceBar.progress = (result.confidence * 100).toInt()

        // Animate in the result
        binding.resultCard.alpha = 0f
        binding.resultCard.translationY = 100f
        binding.resultCard.animate()
            .alpha(1f)
            .translationY(0f)
            .setDuration(500)
            .start()

        // Show all top predictions
        if (result.topPredictions.isNotEmpty()) {
            binding.tvAllPredictions.visibility = View.VISIBLE
            binding.tvAllPredictionsLabel.visibility = View.VISIBLE
            val sb = StringBuilder()
            result.topPredictions.forEachIndexed { index, pred ->
                sb.append("${index + 1}. ${pred.first}  —  ${"%.1f".format(pred.second * 100)}%\n")
            }
            binding.tvAllPredictions.text = sb.toString().trimEnd()
        }
    }

    private fun setupClickListeners() {
        binding.btnBack.setOnClickListener {
            onBackPressedDispatcher.onBackPressed()
        }

        binding.btnAnalyzeAnother.setOnClickListener {
            finish()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        classifier.close()
    }

    companion object {
        const val EXTRA_IMAGE_URI = "extra_image_uri"
    }
}

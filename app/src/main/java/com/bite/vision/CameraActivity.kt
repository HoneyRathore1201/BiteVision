package com.bite.vision

import android.content.Intent
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.bite.vision.databinding.ActivityCameraBinding
import com.bite.vision.ml.ClassificationResult
import com.bite.vision.ml.ImageClassifier
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class CameraActivity : AppCompatActivity() {

    private lateinit var binding: ActivityCameraBinding
    private lateinit var classifier: ImageClassifier
    private lateinit var cameraExecutor: ExecutorService

    private var imageCapture: ImageCapture? = null
    private var isClassifying = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityCameraBinding.inflate(layoutInflater)
        setContentView(binding.root)

        classifier = ImageClassifier(this)
        cameraExecutor = Executors.newSingleThreadExecutor()

        startCamera()
        setupClickListeners()
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.previewView.surfaceProvider)
            }

            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()

            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        if (!isClassifying) {
                            isClassifying = true
                            processImageProxy(imageProxy)
                        } else {
                            imageProxy.close()
                        }
                    }
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageCapture, imageAnalyzer
                )
            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun processImageProxy(imageProxy: ImageProxy) {
        val bitmap = imageProxy.toBitmap()
        imageProxy.close()

        val result = classifier.classify(bitmap)

        runOnUiThread {
            updateUI(result)
            isClassifying = false
        }
    }

    private fun updateUI(result: ClassificationResult) {
        binding.tvLabel.text = result.displayLabel
        binding.tvConfidence.text = "%.1f%%".format(result.confidence * 100)
        binding.tvSubLabel.text = result.subLabel

        val (bgColor, emoji) = when (result.category) {
            ClassificationResult.Category.HUMAN -> Pair(
                ContextCompat.getColor(this, R.color.human_color), "👤"
            )
            ClassificationResult.Category.ANIMAL -> Pair(
                ContextCompat.getColor(this, R.color.animal_color), "🐾"
            )
            ClassificationResult.Category.OTHER -> Pair(
                ContextCompat.getColor(this, R.color.other_color), "🔍"
            )
        }

        binding.resultCard.setCardBackgroundColor(bgColor)
        binding.tvEmoji.text = emoji
        binding.confidenceBar.progress = (result.confidence * 100).toInt()
    }

    private fun setupClickListeners() {
        binding.btnCapture.setOnClickListener {
            captureAndNavigate()
        }

        binding.btnBack.setOnClickListener {
            onBackPressedDispatcher.onBackPressed()
        }
    }

    private fun captureAndNavigate() {
        val imageCapture = imageCapture ?: return
        binding.btnCapture.isEnabled = false

        imageCapture.takePicture(
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(image: ImageProxy) {
                    val bitmap = image.toBitmap()
                    image.close()

                    // Save bitmap to cache and pass URI
                    val file = java.io.File(cacheDir, "captured.jpg")
                    val fos = file.outputStream()
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 90, fos)
                    fos.close()

                    val intent = Intent(this@CameraActivity, ResultActivity::class.java)
                    intent.putExtra(ResultActivity.EXTRA_IMAGE_URI, android.net.Uri.fromFile(file).toString())
                    startActivity(intent)
                    binding.btnCapture.isEnabled = true
                }

                override fun onError(exception: ImageCaptureException) {
                    Log.e(TAG, "Photo capture failed", exception)
                    Toast.makeText(
                        this@CameraActivity,
                        "Capture failed: ${exception.message}",
                        Toast.LENGTH_SHORT
                    ).show()
                    binding.btnCapture.isEnabled = true
                }
            }
        )
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        classifier.close()
    }

    companion object {
        private const val TAG = "CameraActivity"
    }
}

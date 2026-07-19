# BiteVision 🍎👁️

A modern, professional real-time food and object classification Android application. **BiteVision** leverages a hybrid dual-model TensorFlow Lite pipeline and CameraX to run on-device deep learning inference with low latency and high accuracy.

---

## 🌟 Key Features

- **Real-Time Camera Classification**: Instant category identification with visual feedback directly from your live camera stream.
- **Gallery Analysis**: Pick any photo from your gallery to get detailed classification labels and confidence ratings.
- **Dual-Model ML Pipeline**:
  - **Primary Model**: SSD MobileNetV1 (COCO) detects specific objects and foods in the frame.
  - **Fallback Model**: MobileNetV1 (ImageNet) handles broader classification when the primary detection model's confidence falls below the threshold (e.g. general scenes or complex food items).
- **Modern UI & UX**: Immersive dark theme built using Material Components, featuring smooth animations, confidence progress bars, and categorizations (Human, Animal, Object/Scene).

---

## 🛠️ Tech Stack & Architecture

- **Platform**: Android (Kotlin, Min SDK 24, Target SDK 34)
- **UI Framework**: XML Layouts with View Binding
- **Camera Pipeline**: CameraX (Multi-use-case: Preview + ImageAnalysis + ImageCapture)
- **Image Processing**: Glide for fast, cached gallery image loading
- **Machine Learning Runtime**: TensorFlow Lite (org.tensorflow:tensorflow-lite) with Support & Metadata libraries

---

## 🧠 Machine Learning Pipeline

The project includes a Jupyter Notebook under `notebook/human_animal_detection.ipynb` showing the training and export pipeline. It documents:
1. Loading **MobileNetV2** base architecture.
2. Fine-tuning models for category detection.
3. Quantizing models to **INT8** format to run efficiently on mobile CPU/GPU runtimes.
4. Exporting models to `.tflite` format.

---

## 🚀 Getting Started

### Prerequisites

- Android Studio Koala (or newer)
- JDK 17
- Gradle 8.9+
- An Android device running Android 7.0 (API level 24) or higher

### Build & Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/BiteVision.git
   ```
2. Open the project in Android Studio.
3. Allow Gradle to sync dependencies.
4. Run the project:
   - Click the **Run** button in Android Studio, or
   - Build the APK from command line:
     ```bash
     ./gradlew assembleDebug
     ```

---

## 📄 License

This project is licensed under the Apache License 2.0. Feel free to use and extend it!

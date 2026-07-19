# Add project specific ProGuard rules here.
# Keep TFLite classes
-keep class org.tensorflow.** { *; }
-keep class com.bite.vision.ml.** { *; }
-keepclassmembers class * {
    @android.webkit.JavascriptInterface <methods>;
}

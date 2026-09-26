plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "dev.adityakaran.edgeseg"
    compileSdk = 36

    defaultConfig {
        applicationId = "dev.adityakaran.edgeseg"
        minSdk = 31
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"
        // Snapdragon 8 Elite (Galaxy S25 Ultra) is arm64; the QNN libraries ship arm64 only.
        ndk { abiFilters += "arm64-v8a" }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildFeatures { viewBinding = true }

    // Memory-map the model straight out of the APK.
    androidResources { noCompress += "tflite" }

    // The Hexagon NPU loads its skel libraries from nativeLibraryDir, so they must be extracted.
    packaging { jniLibs { useLegacyPackaging = true } }
}

kotlin {
    compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) }
}

dependencies {
    // LiteRT (TensorFlow Lite) runtime + GPU delegate
    implementation("com.google.ai.edge.litert:litert:1.4.2")
    implementation("com.google.ai.edge.litert:litert-gpu:1.4.2")
    implementation("com.google.ai.edge.litert:litert-gpu-api:1.4.2")

    // Qualcomm QNN delegate -> Hexagon NPU (HTP). Matches the QAIRT 2.50 used by AI Hub.
    implementation("com.qualcomm.qti:qnn-runtime:2.50.0")
    implementation("com.qualcomm.qti:qnn-litert-delegate:2.50.0")

    val camerax = "1.5.1"
    implementation("androidx.camera:camera-core:$camerax")
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")

    implementation("androidx.core:core-ktx:1.17.0")
    implementation("androidx.appcompat:appcompat:1.7.1")
    implementation("androidx.activity:activity-ktx:1.11.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.9.4")
    implementation("com.google.android.material:material:1.13.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.2")
}

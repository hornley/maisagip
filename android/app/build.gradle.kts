plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.kapt")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "com.maisagip.app"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.maisagip.app"
        minSdk = 28
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
    }

    buildFeatures {
        compose = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    androidResources {
        noCompress += "onnx"
    }
}

dependencies {
    val composeBom = platform("androidx.compose:compose-bom:2025.05.01")
    implementation(composeBom)

    implementation("androidx.activity:activity-compose:1.10.1")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.navigation:navigation-compose:2.8.9")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.7")
    implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.8.7")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")

    implementation("androidx.room:room-runtime:2.7.2")
    implementation("androidx.room:room-ktx:2.7.2")
    kapt("androidx.room:room-compiler:2.7.2")

    implementation("androidx.camera:camera-core:1.4.2")
    implementation("androidx.camera:camera-camera2:1.4.2")
    implementation("androidx.camera:camera-lifecycle:1.4.2")
    implementation("androidx.camera:camera-view:1.4.2")
    implementation("androidx.exifinterface:exifinterface:1.4.1")

    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.22.0")
}

val verifyBundledModels = tasks.register("verifyBundledModels") {
    val modelDirectory = layout.projectDirectory.dir("src/main/assets/models")
    val requiredFiles = listOf(
        "model_manifest.json",
        "corn_ear_segmenter.onnx",
        "corn_variety_classifier.onnx",
        "corn_defect_detector.onnx",
    )
    inputs.files(requiredFiles.map { modelDirectory.file(it) })
    doLast {
        val missing = requiredFiles.filterNot { modelDirectory.file(it).asFile.isFile }
        check(missing.isEmpty()) {
            "Missing bundled ONNX assets: ${missing.joinToString()}. Run training/export_android_models.py before building the app."
        }
    }
}

tasks.named("preBuild").configure { dependsOn(verifyBundledModels) }

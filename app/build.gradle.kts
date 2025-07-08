plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "com.example.face_reg_app"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.example.face_reg_app"
        minSdk = 29
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    buildFeatures {
        viewBinding = true
    }
}

dependencies {
    implementation(libs.appcompat)
    implementation(libs.material)
    implementation(libs.activity)
    implementation(libs.constraintlayout)
    implementation(libs.annotation)
    implementation(libs.lifecycle.livedata.ktx)
    implementation(libs.lifecycle.viewmodel.ktx)
    testImplementation(libs.junit)
    androidTestImplementation(libs.ext.junit)
    androidTestImplementation(libs.espresso.core)

    // ✅ CameraX cho Java
    val camerax_version = "1.3.1"
    implementation("androidx.camera:camera-core:$camerax_version")
    implementation("androidx.camera:camera-camera2:$camerax_version")
    implementation("androidx.camera:camera-lifecycle:$camerax_version")
    implementation("androidx.camera:camera-view:$camerax_version")
    implementation("androidx.camera:camera-extensions:$camerax_version")

    // ✅ Lifecycle runtime (bắt buộc cho CameraX hoạt động)
    implementation("androidx.lifecycle:lifecycle-runtime:2.9.0")

    implementation("com.google.zxing:core:3.5.2")
    implementation ("com.journeyapps:zxing-android-embedded:4.3.0")
    // ✅ Nếu bạn dùng HTTP để gửi ảnh, có thể dùng OkHttp (không bắt buộc)
    // implementation("com.squareup.okhttp3:okhttp:4.12.0")

    // ✅ OkHttp for HTTP requests (uncommented to fix missing classes)
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation ("com.google.mlkit:barcode-scanning:17.2.0")
}

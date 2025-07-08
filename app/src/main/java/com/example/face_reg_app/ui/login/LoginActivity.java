package com.example.face_reg_app.ui.login;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.media.Image;
import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import com.example.face_reg_app.AttendanceActivity;
import com.example.face_reg_app.R;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.mlkit.vision.barcode.BarcodeScanner;
import com.google.mlkit.vision.barcode.BarcodeScanning;
import com.google.mlkit.vision.barcode.common.Barcode;
import com.google.mlkit.vision.common.InputImage;
import org.json.JSONObject;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class LoginActivity extends AppCompatActivity {

    private PreviewView previewView;
    private TextView scanResultText;
    private ExecutorService cameraExecutor;
    private BarcodeScanner scanner;
    private ImageAnalysis imageAnalysis;
    private boolean isScanning = true;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_login);

        previewView = findViewById(R.id.previewView);
        scanResultText = findViewById(R.id.scanResultText);

        cameraExecutor = Executors.newSingleThreadExecutor();
        scanner = BarcodeScanning.getClient();

        // Check for camera permission
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, 10);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == 10) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startCamera();
            } else {
                Log.e("Camera", "Camera permission denied");
                scanResultText.setText("Camera permission is required to scan QR codes.");
            }
        }
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture =
                ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();

                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                imageAnalysis = new ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();

                imageAnalysis.setAnalyzer(cameraExecutor, imageProxy -> {
                    if (!isScanning) {
                        imageProxy.close();
                        return;
                    }

                    @androidx.camera.core.ExperimentalGetImage
                    Image mediaImage = imageProxy.getImage();
                    if (mediaImage != null) {
                        InputImage image = InputImage.fromMediaImage(mediaImage, imageProxy.getImageInfo().getRotationDegrees());
                        scanner.process(image)
                                .addOnSuccessListener(barcodes -> {
                                    if (barcodes.isEmpty()) {
                                        Log.d("QR_RESULT", "No barcodes detected");
                                        return;
                                    }

                                    for (Barcode barcode : barcodes) {
                                        String rawValue = barcode.getRawValue();
                                        if (rawValue != null) {
                                            scanResultText.setText(rawValue);
                                            Log.d("QR_RESULT", rawValue);

                                            try {
                                                JSONObject json = new JSONObject(rawValue);
                                                String courseId = json.getString("course_id");
                                                String courseName = json.getString("course_name");
                                                String ip = json.getString("ip");

                                                // Stop scanning after successful detection
                                                isScanning = false;
                                                imageAnalysis.clearAnalyzer();

                                                // Transition to AttendanceActivity
                                                Intent intent = new Intent(LoginActivity.this, AttendanceActivity.class);
                                                intent.putExtra("course_id", courseId);
                                                intent.putExtra("course_name", courseName);
                                                intent.putExtra("ip", ip);
                                                startActivity(intent);
                                                finish();
                                                break;
                                            } catch (Exception e) {
                                                Log.e("QR_PARSE_ERROR", "Error parsing QR JSON", e);
                                                scanResultText.setText("Error parsing QR code: " + e.getMessage());
                                            }
                                        }
                                    }
                                })
                                .addOnFailureListener(e -> {
                                    Log.e("QR_ERROR", "Scan failed", e);
                                    scanResultText.setText("Scan failed: " + e.getMessage());
                                })
                                .addOnCompleteListener(task -> imageProxy.close());
                    } else {
                        imageProxy.close();
                    }
                });

                CameraSelector cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA;

                cameraProvider.unbindAll();
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis);

            } catch (Exception e) {
                Log.e("Camera", "Use case binding failed", e);
                scanResultText.setText("Error starting camera: " + e.getMessage());
            }
        }, ContextCompat.getMainExecutor(this));
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        cameraExecutor.shutdown();
        scanner.close();
    }
}
package com.example.face_reg_app;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Base64;
import android.util.Log;
import android.widget.Button;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import com.example.face_reg_app.ui.login.LoginActivity;
import com.google.common.util.concurrent.ListenableFuture;
import org.json.JSONArray;
import org.json.JSONObject;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class AttendanceActivity extends AppCompatActivity {

    private PreviewView previewView;
    private ImageCapture imageCapture;
    private TextView resultText;
    private TextView serverIpText;
    private TextView dateTimeText;
    private TextView courseInfoText;
    private Button captureButton;
    private ExecutorService cameraExecutor;
    private String serverIp = "";
    private String courseId = "";
    private final OkHttpClient client = new OkHttpClient();
    private final Handler handler = new Handler(Looper.getMainLooper());
    private final Runnable updateTimeRunnable = new Runnable() {
        @Override
        public void run() {
            if (dateTimeText != null) {
                SimpleDateFormat dateFormat = new SimpleDateFormat("hh:mm a z, EEEE, dd/MM/yyyy", new Locale("vi", "VN"));
                String currentDateTime = "Thời gian: " + dateFormat.format(new Date());
                dateTimeText.setText(currentDateTime);
            }
            handler.postDelayed(this, 1000);
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        previewView = findViewById(R.id.previewView);
        resultText = findViewById(R.id.resultText);
        serverIpText = findViewById(R.id.serverIpText);
        dateTimeText = findViewById(R.id.dateTimeText);
        courseInfoText = findViewById(R.id.courseInfoText);
        captureButton = findViewById(R.id.captureButton);
        Button scanQrButton = findViewById(R.id.scanQrButton);

        handler.post(updateTimeRunnable);

        String courseName = getIntent().getStringExtra("course_name");
        courseId = getIntent().getStringExtra("course_id");
        serverIp = getIntent().getStringExtra("ip");

        if (serverIp != null && !serverIp.isEmpty()) {
            serverIpText.setText("Địa chỉ server: " + serverIp);
        } else {
            serverIpText.setText("Địa chỉ server: Không tìm thấy");
            resultText.setText("Lỗi: Không có địa chỉ server. Vui lòng quét lại mã QR.");
            captureButton.setEnabled(false);
        }

        if (courseId != null && courseName != null) {
            courseInfoText.setText("Môn học: " + courseName + " (ID: " + courseId + ")");
        } else {
            courseInfoText.setText("Môn học: Không tìm thấy");
        }

        cameraExecutor = Executors.newSingleThreadExecutor();

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, 10);
        }

        captureButton.setOnClickListener(v -> takePhoto());
        scanQrButton.setOnClickListener(v -> {
            Intent intent = new Intent(AttendanceActivity.this, LoginActivity.class);
            startActivity(intent);
            finish();
        });
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == 10) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startCamera();
            } else {
                Log.e("CameraX", "Camera permission denied");
                resultText.setText("Camera permission is required to use this feature.");
            }
        }
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();

                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                imageCapture = new ImageCapture.Builder()
                        .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                        .build();

                CameraSelector cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA;
                try {
                    if (!cameraProvider.hasCamera(cameraSelector)) {
                        Log.w("CameraX", "Front camera not available, switching to back camera");
                        cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA;
                    }
                } catch (Exception e) {
                    Log.e("CameraX", "Error checking camera availability", e);
                    return;
                }

                cameraProvider.unbindAll();
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageCapture);

                previewView.setImplementationMode(PreviewView.ImplementationMode.COMPATIBLE);
                previewView.setScaleType(PreviewView.ScaleType.FIT_CENTER);

            } catch (Exception e) {
                Log.e("CameraX", "Use case binding failed", e);
                resultText.setText("Error starting camera: " + e.getMessage());
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void takePhoto() {
        if (imageCapture == null) {
            Log.e("CameraX", "ImageCapture is null, cannot take photo");
            resultText.setText("Error: Camera not initialized");
            return;
        }

        if (serverIp.isEmpty()) {
            resultText.setText("Error: Please scan a QR code to set the server IP first");
            return;
        }

        if (courseId.isEmpty()) {
            resultText.setText("Error: Course ID is missing. Please scan a QR code again.");
            return;
        }

        captureButton.setEnabled(false);
        resultText.setText("Đang xử lý...");

        imageCapture.takePicture(ContextCompat.getMainExecutor(this), new ImageCapture.OnImageCapturedCallback() {
            @Override
            public void onCaptureSuccess(@NonNull ImageProxy image) {
                Bitmap bitmap = imageProxyToBitmap(image);
                image.close();

                // Flip the bitmap horizontally (for front camera)
                android.graphics.Matrix matrix = new android.graphics.Matrix();
                matrix.preScale(-1.0f, 1.0f);
                Bitmap flippedBitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);

                // Save the flipped bitmap for debugging
                try {
                    File file = new File(getExternalFilesDir(null), "test_image_" + System.currentTimeMillis() + ".jpg");
                    FileOutputStream fos = new FileOutputStream(file);
                    flippedBitmap.compress(Bitmap.CompressFormat.JPEG, 100, fos);
                    fos.close();
                    Log.d("CameraX", "Image saved to: " + file.getAbsolutePath());
                } catch (IOException e) {
                    Log.e("CameraX", "Error saving image", e);
                }

                Bitmap resizedBitmap = resizeBitmap(flippedBitmap, 1200);
                String base64Image = bitmapToBase64(resizedBitmap);
                sendImageToServer(base64Image);
            }

            @Override
            public void onError(@NonNull ImageCaptureException exception) {
                Log.e("CameraX", "Photo capture failed: " + exception.getMessage(), exception);
                runOnUiThread(() -> {
                    resultText.setText("Capture failed: " + exception.getMessage());
                    captureButton.setEnabled(true);
                });
            }
        });
    }

    private Bitmap imageProxyToBitmap(ImageProxy image) {
        ByteBuffer buffer = image.getPlanes()[0].getBuffer();
        byte[] bytes = new byte[buffer.remaining()];
        buffer.get(bytes);
        Bitmap bitmap = android.graphics.BitmapFactory.decodeByteArray(bytes, 0, bytes.length);

        int rotationDegrees = image.getImageInfo().getRotationDegrees();

        if (rotationDegrees != 0) {
            android.graphics.Matrix matrix = new android.graphics.Matrix();
            matrix.postRotate(rotationDegrees);
            bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        }

        return bitmap;
    }


    private Bitmap resizeBitmap(Bitmap bitmap, int maxSize) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();

        if (width <= maxSize && height <= maxSize) {
            return bitmap;
        }

        float ratio = Math.min((float) maxSize / width, (float) maxSize / height);
        int newWidth = Math.round(width * ratio);
        int newHeight = Math.round(height * ratio);

        Log.d("CameraX", "Resized image dimensions: " + newWidth + "x" + newHeight);
        return Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true);
    }

    private String bitmapToBase64(Bitmap bitmap) {
        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, outputStream);
        byte[] imageBytes = outputStream.toByteArray();
        String base64Image = Base64.encodeToString(imageBytes, Base64.NO_WRAP);
        Log.d("CameraX", "Image size: " + imageBytes.length + " bytes, Base64 length: " + base64Image.length());
        return base64Image;
    }

    @SuppressLint("SetTextI18n")
    private void sendImageToServer(String base64Image) {
        cameraExecutor.execute(() -> {
            try {
                String serverUrl = "http://" + serverIp + ":5000/recognize_siamese";
                Log.d("CameraX", "Sending request to: " + serverUrl);

                byte[] imageBytes = Base64.decode(base64Image, Base64.NO_WRAP);
                Log.d("CameraX", "Image bytes length: " + imageBytes.length);

                SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", new Locale("vi", "VN"));
                String timestamp = dateFormat.format(new Date());

                RequestBody requestBody = new MultipartBody.Builder()
                        .setType(MultipartBody.FORM)
                        .addFormDataPart("file", "image.jpg",
                                RequestBody.create(imageBytes, MediaType.parse("image/jpeg")))
                        .addFormDataPart("course_id", courseId)
                        .addFormDataPart("timestamp", timestamp)
                        .build();

                Request request = new Request.Builder()
                        .url(serverUrl)
                        .post(requestBody)
                        .build();

                try (Response response = client.newCall(request).execute()) {
                    String responseBody = response.body() != null ? response.body().string() : "[]";
                    Log.d("CameraX", "Server response: " + responseBody + ", HTTP Code: " + response.code());

                    if (response.isSuccessful()) {
                        try {
                            JSONArray jsonArray = new JSONArray(responseBody);
                            if (jsonArray.length() == 0) {
                                runOnUiThread(() -> resultText.setText("Không phát hiện khuôn mặt"));
                            } else {
                                JSONObject result = jsonArray.getJSONObject(0);
                                String name = result.optString("name", "Unknown");
                                if ("Unknown".equals(name)) {
                                    runOnUiThread(() -> resultText.setText("Không nhận diện được"));
                                } else {
                                    runOnUiThread(() -> resultText.setText(name));
                                }
                            }
                        } catch (Exception e) {
                            Log.e("CameraX", "Error parsing JSON response: " + e.getMessage());
                            runOnUiThread(() -> resultText.setText("Lỗi: " + e.getMessage()));
                        }
                    } else {
                        runOnUiThread(() -> resultText.setText("Lỗi server: HTTP " + response.code()));
                    }
                }
            } catch (IOException e) {
                Log.e("CameraX", "Error sending image to server", e);
                runOnUiThread(() -> resultText.setText("Lỗi gửi ảnh: " + e.getMessage()));
            } finally {
                runOnUiThread(() -> captureButton.setEnabled(true));
            }
        });
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        handler.removeCallbacks(updateTimeRunnable);
        cameraExecutor.shutdown();
    }
}
package com.example.yolo26ncnn;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.view.View;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.resolutionselector.ResolutionSelector;
import androidx.camera.core.resolutionselector.ResolutionStrategy;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "Yolo26Ncnn";
    private static final int REQUEST_PERMISSION = 100;
    private static final String[] REQUIRED_PERMISSIONS = {
            Manifest.permission.CAMERA
    };

    private Yolo26Ncnn yolo26Ncnn = new Yolo26Ncnn();
    private PreviewView previewView;
    private OverlayView overlayView;
    private TextView tvResult;
    private Button btnSwitchCamera;
    private Button btnStartStop;
    private Spinner spinnerModel;
    private Spinner spinnerDevice;

    private int currentModel = 0;
    private int currentDevice = 0; // 0=CPU, 1=GPU
    private boolean isDetecting = false;
    private boolean isFrontCamera = false;

    private ExecutorService cameraExecutor;
    private ProcessCameraProvider cameraProvider;

    private long lastDetectTime = 0;
    private static final long DETECT_INTERVAL = 0; // 检测间隔 100ms

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        previewView = findViewById(R.id.previewView);
        overlayView = findViewById(R.id.overlayView);
        tvResult = findViewById(R.id.tvResult);
        btnSwitchCamera = findViewById(R.id.btnSwitchCamera);
        btnStartStop = findViewById(R.id.btnStartStop);
        spinnerModel = findViewById(R.id.spinnerModel);
        spinnerDevice = findViewById(R.id.spinnerDevice);

        // Set PreviewView to fill the container (crop to fit)
        previewView.setScaleType(PreviewView.ScaleType.FILL_CENTER);

        cameraExecutor = Executors.newSingleThreadExecutor();

        // Model selection
        spinnerModel.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                currentModel = position;
                reloadModel();
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
            }
        });

        // Device selection (CPU/GPU)
        spinnerDevice.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                currentDevice = position;
                reloadModel();
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
            }
        });

        // Switch camera button
        btnSwitchCamera.setOnClickListener(v -> {
            isFrontCamera = !isFrontCamera;
            startCamera();
        });

        // Start/Stop detection button
        btnStartStop.setOnClickListener(v -> {
            isDetecting = !isDetecting;
            btnStartStop.setText(isDetecting ? "Stop Detect" : "Start Detect");
            if (!isDetecting) {
                overlayView.clearResults();
                tvResult.setText("Detection stopped");
            }
        });

        // 检查权限
        if (allPermissionsGranted()) {
            reloadModel();
            startCamera();
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_PERMISSION);
        }
    }

    private boolean allPermissionsGranted() {
        for (String permission : REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_PERMISSION) {
            if (allPermissionsGranted()) {
                reloadModel();
                startCamera();
            } else {
                Toast.makeText(this, "Camera permission is required", Toast.LENGTH_LONG).show();
                finish();
            }
        }
    }

    private void reloadModel() {
        boolean ret = yolo26Ncnn.loadModel(getAssets(), currentModel, currentDevice);
        if (!ret) {
            Log.e(TAG, "Failed to load model");
            Toast.makeText(this, "Failed to load model", Toast.LENGTH_SHORT).show();
        } else {
            String device = currentDevice == 0 ? "CPU" : "GPU (FP32)";
            Log.d(TAG, "Model loaded: yolo26n on " + device);
        }
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture =
                ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                cameraProvider = cameraProviderFuture.get();
                bindCameraUseCases();
            } catch (Exception e) {
                Log.e(TAG, "Camera initialization failed", e);
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void bindCameraUseCases() {
        if (cameraProvider == null) return;

        // 解绑之前的用例
        cameraProvider.unbindAll();

        // 选择摄像头
        int cameraFront = isFrontCamera ? CameraSelector.LENS_FACING_FRONT : CameraSelector.LENS_FACING_BACK;
        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(cameraFront)
                .build();
        ResolutionSelector resolutionSelector = new ResolutionSelector.Builder()
                .setResolutionStrategy(new ResolutionStrategy(new Size(640, 640), ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER))
                .build();

        // 预览
        Preview preview = new Preview.Builder()
                .setResolutionSelector(resolutionSelector)
                .build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        // 图像分析
        ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                .setResolutionSelector(resolutionSelector)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();

        imageAnalysis.setAnalyzer(cameraExecutor, this::analyzeImage);

        try {
            cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis);
            runOnUiThread(() -> tvResult.setText("Camera ready. Tap \"Start Detect\" to begin."));
        } catch (Exception e) {
            Log.e(TAG, "Use case binding failed", e);
        }
    }

    private void analyzeImage(ImageProxy image) {
        if (!isDetecting) {
            image.close();
            return;
        }

        // 控制检测频率
        long currentTime = System.currentTimeMillis();
        if (currentTime - lastDetectTime < DETECT_INTERVAL) {
            image.close();
            return;
        }
        lastDetectTime = currentTime;

        try {
            long cur = System.currentTimeMillis();
            // 将 ImageProxy 转换为 Bitmap
            Bitmap bitmap = imageProxyToBitmap(image);
            Log.i(TAG, "analyzeImage: imagetype： "+image.getFormat()+"  "+image.getWidth()+"  "+image.getImageInfo().getRotationDegrees()+"  "+(System.currentTimeMillis()-cur));

            if (bitmap == null) {
                image.close();
                return;
            }

            // Run detection
            long startTime = System.currentTimeMillis();
            Yolo26Ncnn.Obj[] objects = yolo26Ncnn.detect(bitmap,image.getImageInfo().getRotationDegrees(),isFrontCamera);
            long inferenceTime = System.currentTimeMillis() - startTime;

            Log.i(TAG, "analyzeImage: yoloDetect： " + inferenceTime);

            // Update UI
            final int objectCount = objects != null ? objects.length : 0;
            final long fps = 1000 / Math.max(inferenceTime, 1);


            int rotation = image.getImageInfo().getRotationDegrees();
            int previewWidth, previewHeight;

            // 关键：如果物理旋转了，Java 层感知的宽高也必须对调
            if (rotation == 90 || rotation == 270) {
                previewWidth = bitmap.getHeight(); // 对应 C++ 旋转后的 cols
                previewHeight = bitmap.getWidth(); // 对应 C++ 旋转后的 rows
            } else {
                previewWidth = bitmap.getWidth();
                previewHeight = bitmap.getHeight();
            }

            runOnUiThread(() -> {
                overlayView.setPreviewSize(previewWidth, previewHeight);
                overlayView.setResults(objects);

                if (objectCount > 0) {
                    StringBuilder sb = new StringBuilder();
                    sb.append(String.format("Detected %d objects | %dms | %d FPS\n", objectCount, inferenceTime, fps));
                    for (int i = 0; i < Math.min(objectCount, 3); i++) {
                        if (objects[i].label != null) {
                            sb.append(String.format("%s: %.1f%% ", objects[i].label, objects[i].prob * 100));
                        }
                    }
                    if (objectCount > 3) {
                        sb.append("...");
                    }
                    tvResult.setText(sb.toString());
                } else {
                    tvResult.setText(String.format("No objects detected | %dms | %d FPS", inferenceTime, fps));
                }
            });

            bitmap.recycle();

        } catch (Exception e) {
            Log.e(TAG, "Detection failed", e);
        } finally {
            image.close();
        }
    }

    /**
     * 将 ImageProxy (YUV_420_888) 正确转换为 Bitmap
     * 关键：正确处理 rowStride 和 pixelStride
     */
    private Bitmap imageProxyToBitmap(ImageProxy image) {
        try {

            long cur = System.currentTimeMillis();
            Bitmap bitmap = image.toBitmap();
            Log.i(TAG, "analyzeImage imageProxyToBitmap: "+(System.currentTimeMillis()-cur));
            return bitmap;

        } catch (Exception e) {
            Log.e(TAG, "Image conversion failed", e);
            return null;
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraExecutor != null) {
            cameraExecutor.shutdown();
        }
    }
}

package com.example.faceidentity.view;

import android.os.Bundle;
import android.util.Log;
import android.view.OrientationEventListener;
import android.view.Surface;
import android.view.View;
import android.view.ViewGroup;
import android.view.animation.DecelerateInterpolator;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.view.PreviewView;

import com.example.faceidentity.R;
import com.example.faceidentity.controller.CameraController;
import com.example.faceidentity.controller.FaceDetectionController;
import com.example.faceidentity.model.FaceDetectorModel;
import com.example.faceidentity.utils.FileUtils;
import com.example.faceidentity.utils.PermissionUtils;

import org.opencv.android.OpenCVLoader;

public class MainActivity extends AppCompatActivity
        implements FaceDetectionController.ResultListener {

    private static final String TAG = "MainActivity";
    private static final String MODEL_ASSET = "models/face_detection_yunet_2023mar.onnx";
    private static final float SCORE_THRESHOLD = 0.9f;
    private static final float NMS_THRESHOLD   = 0.3f;
    private static final int   TOP_K           = 5000;

    private static final long ICON_ROTATE_MS = 250L;

    private View root;
    private PreviewView previewView;
    private CameraPreview overlay;
    private View infoPanel;
    private TextView tvFps;
    private TextView tvCount;
    private ImageView ivHint;
    private ImageButton btnStart;
    private ImageButton btnStop;
    private ImageButton btnSwitch;

    private CameraController cameraController;
    private volatile FaceDetectionController detectionController;
    private FaceDetectorModel faceModel;

    private OrientationEventListener orientationListener;
    private int deviceDegrees = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        root        = findViewById(R.id.root);
        previewView = findViewById(R.id.previewView);
        overlay     = findViewById(R.id.overlay);
        infoPanel   = findViewById(R.id.infoPanel);
        tvFps       = findViewById(R.id.tvFps);
        tvCount     = findViewById(R.id.tvCount);
        ivHint      = findViewById(R.id.ivHint);
        btnStart    = findViewById(R.id.btnStart);
        btnStop     = findViewById(R.id.btnStop);
        btnSwitch   = findViewById(R.id.btnSwitch);

        root.addOnLayoutChangeListener((v, l, t, r, b, ol, ot, or, ob) -> {
            if ((r - l) != (or - ol) || (b - t) != (ob - ot)) applyOverlayRotation();
        });

        if (!OpenCVLoader.initDebug()) {
            Toast.makeText(this, "OpenCV init thất bại!", Toast.LENGTH_LONG).show();
            Log.e(TAG, "OpenCVLoader.initDebug() = false");
            return;
        }
        Log.i(TAG, "OpenCV init OK");
        cameraController = new CameraController(
                this, this, previewView,
                image -> {
                    FaceDetectionController c = detectionController;
                    if (c != null) {
                        c.process(image);
                    } else {
                        image.close();
                    }
                });
        cameraController.setLensFacing(CameraSelector.LENS_FACING_FRONT);
        cameraController.setLensChangedListener(isFront -> overlay.setMirror(isFront));

        loadModel();

        btnStart.setOnClickListener(v -> startDetection());
        btnStop.setOnClickListener(v -> stopDetection());
        btnSwitch.setOnClickListener(v -> {
            stopDetection();
            cameraController.switchCamera();
            Toast.makeText(this,
                    cameraController.isFront() ? R.string.cam_front : R.string.cam_back,
                    Toast.LENGTH_SHORT).show();
        });
        orientationListener = new OrientationEventListener(this) {
            @Override
            public void onOrientationChanged(int orientation) {
                if (orientation == ORIENTATION_UNKNOWN) return;
                int snapped = snapTo90(orientation);
                if (snapped == deviceDegrees) return;
                deviceDegrees = snapped;
                onDeviceRotationChanged();
            }
        };

        if (PermissionUtils.hasCameraPermission(this)) {
            cameraController.startCamera();
        } else {
            PermissionUtils.requestCameraPermission(this);
        }

        onDeviceRotationChanged();
    }
    private static int snapTo90(int orientation) {
        if (orientation >= 45 && orientation < 135)  return 90;
        if (orientation >= 135 && orientation < 225) return 180;
        if (orientation >= 225 && orientation < 315) return 270;
        return 0;
    }
    private static int deviceDegreesToSurfaceRotation(int deviceDegrees) {
        switch (deviceDegrees) {
            case 90:  return Surface.ROTATION_270;
            case 180: return Surface.ROTATION_180;
            case 270: return Surface.ROTATION_90;
            default:  return Surface.ROTATION_0;
        }
    }

    private boolean isDeviceLandscape() {
        return deviceDegrees == 90 || deviceDegrees == 270;
    }
    private void onDeviceRotationChanged() {
        if (cameraController != null) {
            cameraController.updateTargetRotation(deviceDegreesToSurfaceRotation(deviceDegrees));
        }
        applyOverlayRotation();
        animateIconsUpright();
        updateOrientationGate();
    }

    private void applyOverlayRotation() {
        if (root == null || overlay == null) return;

        int w = root.getWidth();
        int h = root.getHeight();
        if (w == 0 || h == 0) return;
        boolean swap = isDeviceLandscape();
        int targetW = swap ? h : w;
        int targetH = swap ? w : h;

        ViewGroup.LayoutParams lp = overlay.getLayoutParams();
        if (lp.width != targetW || lp.height != targetH) {
            lp.width = targetW;
            lp.height = targetH;
            overlay.setLayoutParams(lp);
        }
        overlay.setRotation(-deviceDegrees);
    }

    private void animateIconsUpright() {
        float target = -deviceDegrees;
        rotateSmooth(btnStart, target);
        rotateSmooth(btnStop, target);
        rotateSmooth(btnSwitch, target);
        rotateSmooth(ivHint, target);
        rotateSmooth(infoPanel, target);
    }

    private static void rotateSmooth(View v, float target) {
        if (v == null) return;
        float delta = ((target - v.getRotation()) % 360f + 540f) % 360f - 180f;
        if (Math.abs(delta) < 0.5f) return;
        v.animate()
                .rotationBy(delta)
                .setDuration(ICON_ROTATE_MS)
                .setInterpolator(new DecelerateInterpolator())
                .start();
    }

    private void updateOrientationGate() {
        boolean landscape = isDeviceLandscape();
        boolean modelReady = detectionController != null;
        boolean canStart = landscape && modelReady;

        btnStart.setEnabled(canStart);
        btnStart.setAlpha(canStart ? 1f : 0.4f);
        ivHint.setVisibility(landscape ? View.GONE : View.VISIBLE);

        if (!landscape && detectionController != null && detectionController.isRunning()) {
            stopDetection();
        }
    }

    private void startDetection() {
        if (!isDeviceLandscape()) {
            Toast.makeText(this, R.string.hint_rotate, Toast.LENGTH_SHORT).show();
            return;
        }
        if (detectionController == null) {
            loadModel();
            updateOrientationGate();
            if (detectionController == null) return;
        }
        detectionController.setRunning(true);
    }

    private void stopDetection() {
        if (detectionController != null) detectionController.setRunning(false);
        overlay.clear();
        tvCount.setText(getString(R.string.face_count, 0));
        tvFps.setText(getString(R.string.fps, 0.0));
    }

    private void loadModel() {
        if (detectionController != null) return;
        try {
            String modelPath = FileUtils.copyAssetToInternal(this, MODEL_ASSET);
            faceModel = new FaceDetectorModel(modelPath, SCORE_THRESHOLD, NMS_THRESHOLD, TOP_K);
            faceModel.init();
            detectionController = new FaceDetectionController(faceModel, this);
            Log.i(TAG, "Model load OK: " + modelPath);
        } catch (java.io.FileNotFoundException e) {
            Toast.makeText(this,
                    "THIẾU MODEL: app/src/main/assets/" + MODEL_ASSET
                            + "\nChép file .onnx và build lại (xem README).",
                    Toast.LENGTH_LONG).show();
            Log.e(TAG, "Model chưa có trong assets: " + MODEL_ASSET, e);
        } catch (Exception e) {
            Toast.makeText(this, "Load model lỗi: " + e.getMessage(), Toast.LENGTH_LONG).show();
            Log.e(TAG, "Load model lỗi", e);
        }
    }

    @Override
    public void onResult(float[] boxes, int faceCount,
                         int frameWidth, int frameHeight, double fps) {
        overlay.setResults(boxes, frameWidth, frameHeight);
        tvCount.setText(getString(R.string.face_count, faceCount));
        tvFps.setText(getString(R.string.fps, fps));
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (orientationListener != null && orientationListener.canDetectOrientation()) {
            orientationListener.enable();
        }
        onDeviceRotationChanged();
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (orientationListener != null) orientationListener.disable();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (PermissionUtils.isCameraGranted(requestCode, grantResults)) {
            if (cameraController != null) cameraController.startCamera();
        } else {
            Toast.makeText(this, "Cần quyền camera để chạy ứng dụng", Toast.LENGTH_LONG).show();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (orientationListener != null) orientationListener.disable();
        if (cameraController != null) cameraController.shutdown();
        if (detectionController != null) detectionController.release();
        if (faceModel != null) faceModel.release();
    }
}

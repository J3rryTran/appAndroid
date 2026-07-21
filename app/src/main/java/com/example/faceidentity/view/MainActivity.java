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
import com.example.faceidentity.model.DetectorFactory;
import com.example.faceidentity.model.FaceDetector;
import com.example.faceidentity.model.ModelConfig;
import com.example.faceidentity.utils.CrashLogger;
import com.example.faceidentity.utils.FileUtils;
import com.example.faceidentity.utils.PermissionUtils;

import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class MainActivity extends AppCompatActivity
        implements FaceDetectionController.ResultListener {

    private static final String TAG = "MainActivity";

    // Nhấn giữ nút đổi cam để chuyển model. Backend chọn theo tên file (DetectorFactory).
    private static final String MODELS_DIR = "models";
    private static final String DEFAULT_MODEL = "RFB-landmark-Epoch-149-Loss-30.2965.onnx";

    private static final long ICON_ROTATE_MS = 250L;

    private View root;
    private PreviewView previewView;
    private CameraPreview overlay;
    private View infoPanel;
    private TextView tvFps;
    private TextView tvCount;
    private TextView tvModel;
    private ImageView ivHint;
    private ImageButton btnStart;
    private ImageButton btnStop;
    private ImageButton btnSwitch;

    private CameraController cameraController;
    private volatile FaceDetectionController detectionController;
    private FaceDetector faceDetector;

    private String[] modelList = new String[0];
    private int modelIndex = 0;

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
        tvModel     = findViewById(R.id.tvModel);
        ivHint      = findViewById(R.id.ivHint);
        btnStart    = findViewById(R.id.btnStart);
        btnStop     = findViewById(R.id.btnStop);
        btnSwitch   = findViewById(R.id.btnSwitch);

        root.addOnLayoutChangeListener((v, l, t, r, b, ol, ot, or, ob) -> {
            if ((r - l) != (or - ol) || (b - t) != (ob - ot)) applyOverlayRotation();
        });

        if (!OpenCVLoader.initDebug()) {
            Toast.makeText(this, "OpenCV init thất bại!", Toast.LENGTH_LONG).show();
            CrashLogger.logError(TAG, "OpenCVLoader.initDebug() = false", null);
            return;
        }
        Log.i(TAG, "OpenCV init OK");
        Log.i(TAG, "File log lỗi/crash: " + CrashLogger.logDirPath());
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

        initModelList();
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
        // NHẤN GIỮ nút đổi cam = chuyển model kế tiếp trong assets/models.
        btnSwitch.setOnLongClickListener(v -> {
            nextModel();
            return true;
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

    // ---------------------------------------------------------------------
    // Multi-model
    // ---------------------------------------------------------------------

    /** Quét assets/models -> danh sách model hỗ trợ; trỏ vào DEFAULT_MODEL nếu có. */
    private void initModelList() {
        List<String> found = new ArrayList<>();
        try {
            String[] all = getAssets().list(MODELS_DIR);
            if (all != null) {
                for (String f : all) {
                    if (DetectorFactory.isSupportedFile(f)) found.add(f);
                }
            }
        } catch (IOException e) {
            Log.e(TAG, "Không đọc được assets/" + MODELS_DIR, e);
        }
        Collections.sort(found);
        modelList = found.toArray(new String[0]);
        modelIndex = 0;
        for (int i = 0; i < modelList.length; i++) {
            if (modelList[i].equals(DEFAULT_MODEL)) {
                modelIndex = i;
                break;
            }
        }
        Log.i(TAG, "Model trong assets: " + found);
    }

    /** Chuyển model kế tiếp (gọi khi NHẤN GIỮ nút đổi cam). */
    private void nextModel() {
        if (modelList.length == 0) {
            Toast.makeText(this, "Không có model nào trong assets/models", Toast.LENGTH_SHORT).show();
            return;
        }
        stopDetection();
        releaseCurrentModel();
        modelIndex = (modelIndex + 1) % modelList.length;
        loadModel();
        updateOrientationGate();
        if (detectionController != null) {
            Toast.makeText(this, "Model: " + modelList[modelIndex], Toast.LENGTH_SHORT).show();
        }
    }

    private void loadModel() {
        if (detectionController != null) return;
        if (modelList.length == 0) {
            tvModel.setText(getString(R.string.model_label, "KHÔNG CÓ"));
            Toast.makeText(this, "Không có model nào trong assets/models", Toast.LENGTH_LONG).show();
            return;
        }
        String fileName = modelList[modelIndex];
        try {
            String modelPath = FileUtils.copyAssetToInternal(this, MODELS_DIR + "/" + fileName);
            ModelConfig cfg = readConfig(fileName);
            faceDetector = DetectorFactory.create(fileName, modelPath, cfg);
            faceDetector.init();

            // Smoke test: detect thử ảnh giả ngay lúc load (main thread, có catch).
            smokeTest(faceDetector);

            detectionController = new FaceDetectionController(faceDetector, this);
            tvModel.setText(getString(R.string.model_label,
                    faceDetector.name() + " · " + fileName));
            Log.i(TAG, "Model load OK [" + faceDetector.name() + "]: " + modelPath);
        } catch (java.io.FileNotFoundException e) {
            releaseCurrentModel();
            tvModel.setText(getString(R.string.model_label, fileName + " (thiếu file)"));
            Toast.makeText(this, "THIẾU MODEL: assets/" + MODELS_DIR + "/" + fileName,
                    Toast.LENGTH_LONG).show();
            CrashLogger.logError(TAG, "Model chưa có trong assets: " + fileName, e);
        } catch (Exception e) {
            releaseCurrentModel();
            tvModel.setText(getString(R.string.model_label, fileName + " (lỗi)"));
            String reason = (e instanceof UnsupportedOperationException)
                    ? e.getMessage()
                    : "Model không chạy được với backend: " + fileName + " (xem Logcat)";
            Toast.makeText(this, reason, Toast.LENGTH_LONG).show();
            CrashLogger.logError(TAG, "Load/kiểm tra model lỗi: " + fileName, e);
        }
    }

    /** Đọc file cấu hình <tên-model-không-đuôi>.json trong assets/models. Không có -> đoán theo tên. */
    private ModelConfig readConfig(String modelFile) {
        int dot = modelFile.lastIndexOf('.');
        String base = (dot > 0) ? modelFile.substring(0, dot) : modelFile;
        String cfgAsset = MODELS_DIR + "/" + base + ".json";
        try (InputStream is = getAssets().open(cfgAsset)) {
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            byte[] buf = new byte[4096];
            int n;
            while ((n = is.read(buf)) != -1) bos.write(buf, 0, n);
            ModelConfig c = ModelConfig.fromJson(bos.toString("UTF-8"));
            Log.i(TAG, "Config: " + cfgAsset);
            return c;
        } catch (java.io.FileNotFoundException e) {
            return ModelConfig.defaultFor(modelFile);
        } catch (Exception e) {
            CrashLogger.logError(TAG, "Đọc config lỗi: " + cfgAsset, e);
            return ModelConfig.defaultFor(modelFile);
        }
    }

    private static void smokeTest(FaceDetector d) {
        Mat dummy = new Mat(240, 320, CvType.CV_8UC3, new Scalar(0, 0, 0));
        try {
            d.detect(dummy);
        } finally {
            dummy.release();
        }
    }

    /** Giải phóng model hiện tại (controller sẽ release detector bên trong). */
    private void releaseCurrentModel() {
        if (detectionController != null) {
            detectionController.release();
            detectionController = null;
            faceDetector = null;
            return;
        }
        if (faceDetector != null) {
            faceDetector.release();
            faceDetector = null;
        }
    }

    @Override
    public void onResult(float[] boxes, float[] landmarks, int faceCount,
                         int frameWidth, int frameHeight, double fps) {
        overlay.setResults(boxes, landmarks, frameWidth, frameHeight);
        tvCount.setText(getString(R.string.face_count, faceCount));
        tvFps.setText(getString(R.string.fps, fps));
    }

    @Override
    public void onDetectionError(Exception e) {
        stopDetection();
        Toast.makeText(this,
                "Detect gặp lỗi và đã tự dừng (chi tiết: logs/error-*.log).",
                Toast.LENGTH_LONG).show();
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
        releaseCurrentModel();
    }
}

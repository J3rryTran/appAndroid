package com.example.faceidentity.controller;

import android.content.Context;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CaptureRequest;
import android.util.Log;
import android.util.Range;
import android.util.Size;
import android.view.Surface;

import androidx.annotation.NonNull;
import androidx.annotation.OptIn;
import androidx.camera.camera2.interop.Camera2CameraInfo;
import androidx.camera.camera2.interop.Camera2Interop;
import androidx.camera.camera2.interop.ExperimentalCamera2Interop;
import androidx.camera.core.CameraInfo;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.resolutionselector.ResolutionSelector;
import androidx.camera.core.resolutionselector.ResolutionStrategy;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import com.example.faceidentity.utils.CrashLogger;
import com.google.common.util.concurrent.ListenableFuture;

import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class CameraController {

    private static final String TAG = "CameraController";
    public interface FrameListener {
        void onFrame(@NonNull ImageProxy image);
    }
    public interface LensChangedListener {
        void onLensChanged(boolean isFront);
    }

    private final Context context;
    private final LifecycleOwner lifecycleOwner;
    private final PreviewView previewView;
    private final FrameListener frameListener;
    private LensChangedListener lensChangedListener;

    private ProcessCameraProvider cameraProvider;
    private ExecutorService analysisExecutor;

    private Preview preview;
    private ImageAnalysis imageAnalysis;

    private int lensFacing = CameraSelector.LENS_FACING_FRONT;
    private int targetRotation = Surface.ROTATION_0;
    private final Size targetAnalysisSize = new Size(640, 480);

    public CameraController(Context context, LifecycleOwner owner,
                            PreviewView previewView, FrameListener listener) {
        this.context = context.getApplicationContext();
        this.lifecycleOwner = owner;
        this.previewView = previewView;
        this.frameListener = listener;
    }

    public void setLensChangedListener(LensChangedListener l) {
        this.lensChangedListener = l;
    }
    public void setLensFacing(int lensFacing) {
        this.lensFacing = lensFacing;
    }

    public boolean isFront() {
        return lensFacing == CameraSelector.LENS_FACING_FRONT;
    }
    public void startCamera() {
        if (analysisExecutor == null) {
            analysisExecutor = newAnalysisExecutor();
        }
        final ListenableFuture<ProcessCameraProvider> future =
                ProcessCameraProvider.getInstance(context);
        future.addListener(() -> {
            try {
                cameraProvider = future.get();
                bindUseCases();
            } catch (Exception e) {
                CrashLogger.logError(TAG, "Không khởi tạo được CameraX", e);
            }
        }, ContextCompat.getMainExecutor(context));
    }
    public void switchCamera() {
        lensFacing = isFront() ? CameraSelector.LENS_FACING_BACK
                               : CameraSelector.LENS_FACING_FRONT;
        bindUseCases();
    }
    public void updateTargetRotation(int surfaceRotation) {
        if (targetRotation == surfaceRotation) return;   // tránh gọi thừa
        targetRotation = surfaceRotation;
        if (imageAnalysis != null) imageAnalysis.setTargetRotation(surfaceRotation);
    }

    @OptIn(markerClass = ExperimentalCamera2Interop.class)
    private void bindUseCases() {
        if (cameraProvider == null) return;
        if (analysisExecutor == null) {
            analysisExecutor = newAnalysisExecutor();
        }
        if (!hasCamera(lensFacing)) {
            int other = isFront() ? CameraSelector.LENS_FACING_BACK
                                  : CameraSelector.LENS_FACING_FRONT;
            if (!hasCamera(other)) {
                Log.e(TAG, "Thiết bị không có camera khả dụng");
                return;
            }
            Log.w(TAG, "Thiếu camera yêu cầu -> chuyển sang camera còn lại");
            lensFacing = other;
        }

        cameraProvider.unbindAll();

        // Khoá sàn FPS (AE không được tụt xuống 15fps lúc thiếu sáng -> preview hết "trễ nhẹ")
        Preview.Builder previewBuilder = new Preview.Builder();
        Range<Integer> fpsRange = pickFpsRange();
        if (fpsRange != null) {
            new Camera2Interop.Extender<>(previewBuilder).setCaptureRequestOption(
                    CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, fpsRange);
        }
        preview = previewBuilder.build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());
        ResolutionSelector resolutionSelector = new ResolutionSelector.Builder()
                .setResolutionStrategy(new ResolutionStrategy(
                        targetAnalysisSize,
                        ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER))
                .build();

        imageAnalysis = new ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setResolutionSelector(resolutionSelector)
                .setTargetRotation(targetRotation)
                .build();

        imageAnalysis.setAnalyzer(analysisExecutor, image -> {
            if (frameListener != null) frameListener.onFrame(image);   // listener sẽ close()
            else image.close();
        });

        CameraSelector selector = new CameraSelector.Builder()
                .requireLensFacing(lensFacing)
                .build();
        cameraProvider.bindToLifecycle(lifecycleOwner, selector, preview, imageAnalysis);
        Log.i(TAG, "CameraX bind OK. front=" + isFront() + " rotation=" + targetRotation
                + " fps=" + fpsRange);
        if (lensChangedListener != null) lensChangedListener.onLensChanged(isFront());
    }

    @OptIn(markerClass = ExperimentalCamera2Interop.class)
    private Range<Integer> pickFpsRange() {
        try {
            CameraSelector sel = new CameraSelector.Builder()
                    .requireLensFacing(lensFacing).build();
            List<CameraInfo> infos = sel.filter(cameraProvider.getAvailableCameraInfos());
            if (infos.isEmpty()) return null;
            Range<Integer>[] ranges = Camera2CameraInfo.from(infos.get(0)).getCameraCharacteristic(
                    CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES);
            if (ranges == null) return null;
            Range<Integer> best = null;
            for (Range<Integer> r : ranges) {
                if (r.getUpper() < 30) continue;
                if (best == null
                        || r.getLower() > best.getLower()
                        || (r.getLower().equals(best.getLower()) && r.getUpper() < best.getUpper())) {
                    best = r;
                }
            }
            return best;
        } catch (Exception e) {
            return null;
        }
    }

    // Thread phân tích chạy priority thấp -> nhường CPU cho camera/hiển thị
    private static ExecutorService newAnalysisExecutor() {
        return Executors.newSingleThreadExecutor(r -> new Thread(() -> {
            android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_BACKGROUND);
            r.run();
        }, "face-analysis"));
    }

    private boolean hasCamera(int facing) {
        try {
            return cameraProvider != null && cameraProvider.hasCamera(
                    new CameraSelector.Builder().requireLensFacing(facing).build());
        } catch (Exception e) {
            return false;
        }
    }

    public void stopCamera() {
        if (cameraProvider != null) cameraProvider.unbindAll();
    }
    public void shutdown() {
        stopCamera();
        if (analysisExecutor != null) {
            analysisExecutor.shutdown();
            analysisExecutor = null;
        }
    }
}

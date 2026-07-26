package com.example.faceidentity.controller;

import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.camera.core.ImageProxy;

import com.example.faceidentity.model.DetectionResult;
import com.example.faceidentity.model.FaceDetector;
import com.example.faceidentity.utils.CrashLogger;
import com.example.faceidentity.utils.ImageUtils;

import org.opencv.core.Mat;

import java.util.Locale;

public class FaceDetectionController {

    private static final String TAG = "FaceDetectionCtrl";

    public interface ResultListener {
        void onResult(float[] boxes, float[] landmarks, float[] scores, int faceCount,
                      int frameWidth, int frameHeight, double fps);
        void onDetectionError(Exception e);
    }

    private final FaceDetector detector;
    private final String label;
    private final ImageUtils imageUtils = new ImageUtils();
    private final ResultListener listener;
    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private final Object lock = new Object();
    private boolean released = false;

    private volatile boolean running = false;

    private long fpsWindowStart = 0;
    private int fpsFrameCount = 0;
    private double fps = 0.0;
    private int lastFaceCount = 0;
    private float lastAvgScore = 0f;

    public FaceDetectionController(@NonNull FaceDetector detector, String label,
                                   @NonNull ResultListener listener) {
        this.detector = detector;
        this.label = (label != null) ? label.replaceAll("\\.[^.]+$", "") : "?";
        this.listener = listener;
    }

    public void setRunning(boolean running) {
        this.running = running;
        if (running) {
            fpsWindowStart = 0;
            fpsFrameCount = 0;
            fps = 0.0;
        }
    }

    public boolean isRunning() {
        return running;
    }
    public void process(@NonNull ImageProxy image) {
        try {
            if (!running) {
                return;
            }
            DetectionResult result;
            int fw;
            int fh;
            try {
                synchronized (lock) {
                    if (released) return;
                    Mat bgr = imageUtils.imageProxyToBgr(image);
                    result = detector.detect(bgr);
                    fw = bgr.cols();
                    fh = bgr.rows();
                }
            } catch (Exception e) {
                running = false;
                CrashLogger.logError(TAG, "Detect lỗi -> tự dừng", e);
                mainHandler.post(() -> listener.onDetectionError(e));
                return;
            }

            lastFaceCount = result.count();
            lastAvgScore = avg(result.scores);
            updateFps();
            final double curFps = fps;
            final DetectionResult r = result;
            final int ffw = fw;
            final int ffh = fh;
            mainHandler.post(() ->
                    listener.onResult(r.boxes, r.landmarks, r.scores, r.count(), ffw, ffh, curFps));
        } finally {
            image.close();
        }
    }

    private void updateFps() {
        long now = SystemClock.elapsedRealtime();
        if (fpsWindowStart == 0) {
            fpsWindowStart = now;
            fpsFrameCount = 0;
            return;
        }
        fpsFrameCount++;
        long dt = now - fpsWindowStart;
        if (dt >= 500) {
            fps = fpsFrameCount * 1000.0 / dt;
            fpsWindowStart = now;
            fpsFrameCount = 0;
            Log.i(TAG, String.format(Locale.US, "[%s] FPS=%.1f | faces=%d | conf=%.2f",
                    label, fps, lastFaceCount, lastAvgScore));
        }
    }

    private static float avg(float[] a) {
        if (a == null || a.length == 0) return 0f;
        float s = 0f;
        for (float v : a) s += v;
        return s / a.length;
    }
    public void release() {
        running = false;
        synchronized (lock) {
            if (released) return;
            released = true;
            detector.release();
            imageUtils.release();
        }
    }
}

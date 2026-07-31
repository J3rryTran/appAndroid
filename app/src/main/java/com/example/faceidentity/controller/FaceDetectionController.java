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
import com.example.faceidentity.utils.HeadPose;
import com.example.faceidentity.utils.ImageUtils;
import com.example.faceidentity.utils.LatencyMeter;

import org.opencv.core.Mat;

import java.util.Locale;

public class FaceDetectionController {

    private static final String TAG = "FaceDetectionCtrl";

    public interface ResultListener {
        void onResult(DetectionResult result, int frameWidth, int frameHeight, double fps);
        void onDetectionError(Exception e);
    }

    private final FaceDetector detector;
    private final String label;
    private final ImageUtils imageUtils = new ImageUtils();
    private final HeadPose headPose = new HeadPose();
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
    private String lastPose = "";
    private String lastLmk = "";
    private final LatencyMeter pipeline = new LatencyMeter("cvt", "det");

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
                    long t0 = System.nanoTime();
                    Mat bgr = imageUtils.imageProxyToBgr(image);
                    long t1 = System.nanoTime();
                    result = detector.detect(bgr);
                    long t2 = System.nanoTime();
                    pipeline.add((t1 - t0) / 1e6, (t2 - t1) / 1e6);
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
            computePose(result, fw, fh);
            updateSummary(result);
            updateFps();
            final double curFps = fps;
            final DetectionResult r = result;
            final int ffw = fw;
            final int ffh = fh;
            mainHandler.post(() -> listener.onResult(r, ffw, ffh, curFps));
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
            String pipe = pipeline.snapshotAndReset();
            String stages = detector.timings();
            Log.i(TAG, String.format(Locale.US,
                    "[%s] FPS=%.1f | faces=%d | conf=%.2f | %s%s ms%s%s",
                    label, fps, lastFaceCount, lastAvgScore, pipe,
                    stages.isEmpty() ? "" : " | " + stages, lastPose, lastLmk));
        }
    }

    private void computePose(DetectionResult r, int fw, int fh) {
        int n = r.count();
        if (n == 0 || !r.hasLandmarks()) {
            r.pose = null;
            return;
        }
        float[] pose = new float[n * 3];
        for (int i = 0; i < n; i++) {
            if (!headPose.estimate(r.landmarks, i, fw, fh, pose, i * 3)) {
                pose[i * 3] = Float.NaN;
            }
        }
        r.pose = pose;
    }

    private void updateSummary(DetectionResult r) {
        int n = r.count();
        if (n == 0) {
            lastPose = "";
            lastLmk = "";
            return;
        }
        int best = 0;
        for (int i = 1; i < n; i++) {
            if (r.scores[i] > r.scores[best]) best = i;
        }
        if (r.hasPose() && !Float.isNaN(r.pose[best * 3])) {
            lastPose = String.format(Locale.US, " | pose=y%.0f/p%.0f/r%.0f",
                    r.pose[best * 3], r.pose[best * 3 + 1], r.pose[best * 3 + 2]);
        } else {
            lastPose = "";
        }
        if (r.hasLandmarkScores()) {
            StringBuilder sb = new StringBuilder(" | kpt=");
            for (int q = 0; q < 5; q++) {
                if (q > 0) sb.append('/');
                sb.append(Math.round(r.landmarkScores[best * 5 + q] * 100));
            }
            lastLmk = sb.toString();
        } else {
            lastLmk = "";
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
            headPose.release();
        }
    }
}

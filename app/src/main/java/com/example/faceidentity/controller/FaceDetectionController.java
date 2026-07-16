package com.example.faceidentity.controller;

import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;

import androidx.annotation.NonNull;
import androidx.camera.core.ImageProxy;

import com.example.faceidentity.model.FaceDetectorModel;
import com.example.faceidentity.utils.ImageUtils;

import org.opencv.core.Mat;

public class FaceDetectionController {
    public interface ResultListener {
        void onResult(float[] boxes, int faceCount,
                      int frameWidth, int frameHeight, double fps);
    }

    private final FaceDetectorModel model;
    private final ImageUtils imageUtils = new ImageUtils();
    private final ResultListener listener;
    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private final float[] rowBuf = new float[4];
    private volatile boolean running = false;
    private long fpsWindowStart = 0;
    private int fpsFrameCount = 0;
    private double fps = 0.0;

    public FaceDetectionController(@NonNull FaceDetectorModel model,
                                   @NonNull ResultListener listener) {
        this.model = model;
        this.listener = listener;
    }

    public void setRunning(boolean running) {
        this.running = running;
        if (running) {
            // reset FPS khi bắt đầu
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
            Mat bgr = imageUtils.imageProxyToBgr(image);
            Mat faces = model.detect(bgr);
            int count = faces.rows();
            float[] boxes = new float[count * 4];
            for (int i = 0; i < count; i++) {
                faces.get(i, 0, rowBuf);
                boxes[i * 4]     = rowBuf[0];
                boxes[i * 4 + 1] = rowBuf[1];
                boxes[i * 4 + 2] = rowBuf[2];
                boxes[i * 4 + 3] = rowBuf[3];
            }

            final int fw = bgr.cols();
            final int fh = bgr.rows();
            updateFps();
            final double curFps = fps;
            final int curCount = count;
            mainHandler.post(() ->
                    listener.onResult(boxes, curCount, fw, fh, curFps));
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
        }
    }

    public void release() {
        imageUtils.release();
    }
}

package com.example.faceidentity.model;

import android.util.Log;

import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.objdetect.FaceDetectorYN;

/**
 * [MODEL] Backend YuNet qua FaceDetectorYN (chỉ chạy model YuNet chuẩn OpenCV Zoo).
 *
 * Output gốc: Mat [n x 15] = x,y,w,h + 5 landmark (10 giá trị) + score.
 */
public class YuNetDetector implements FaceDetector {

    private static final String TAG = "YuNetDetector";

    // Ngưỡng mặc định hợp với YuNet.
    private static final float SCORE_THRESHOLD = 0.9f;
    private static final float NMS_THRESHOLD   = 0.3f;
    private static final int   TOP_K           = 5000;

    private final String modelPath;
    private FaceDetectorYN detector;

    private int inW = 0;
    private int inH = 0;

    // Tái sử dụng
    private final Mat faces = new Mat();
    private final float[] rowBuf = new float[15];

    public YuNetDetector(String modelPath) {
        this.modelPath = modelPath;
    }

    @Override
    public String name() {
        return "YuNet";
    }

    @Override
    public void init() {
        detector = FaceDetectorYN.create(
                modelPath, "", new Size(320, 320),
                SCORE_THRESHOLD, NMS_THRESHOLD, TOP_K);
        if (detector == null) {
            throw new IllegalStateException("FaceDetectorYN.create() = null");
        }
        Log.i(TAG, "YuNet init OK: " + modelPath);
    }

    @Override
    public DetectionResult detect(Mat bgr) {
        if (detector == null) return DetectionResult.EMPTY;

        if (inW != bgr.cols() || inH != bgr.rows()) {
            inW = bgr.cols();
            inH = bgr.rows();
            detector.setInputSize(new Size(inW, inH));
        }
        detector.detect(bgr, faces);

        int n = faces.rows();
        if (n <= 0) return DetectionResult.EMPTY;

        float[] boxes = new float[n * 4];
        float[] scores = new float[n];
        float[] lands = new float[n * 10];
        for (int i = 0; i < n; i++) {
            faces.get(i, 0, rowBuf);            // đọc cả hàng 15 giá trị
            boxes[i * 4]     = rowBuf[0];
            boxes[i * 4 + 1] = rowBuf[1];
            boxes[i * 4 + 2] = rowBuf[2];
            boxes[i * 4 + 3] = rowBuf[3];
            System.arraycopy(rowBuf, 4, lands, i * 10, 10);
            scores[i] = rowBuf[14];
        }
        return new DetectionResult(boxes, scores, lands);
    }

    @Override
    public void release() {
        faces.release();
        detector = null;
    }
}

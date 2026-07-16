package com.example.faceidentity.model;

import android.util.Log;

import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.objdetect.FaceDetectorYN;

public class FaceDetectorModel {

    private static final String TAG = "FaceDetectorModel";

    private FaceDetectorYN detector;
    private int inW = 0;
    private int inH = 0;

    private final String modelPath;
    private float scoreThreshold;
    private float nmsThreshold;
    private int topK;
    private final Mat faces = new Mat();

    public FaceDetectorModel(String modelPath, float scoreThreshold,
                             float nmsThreshold, int topK) {
        this.modelPath = modelPath;
        this.scoreThreshold = scoreThreshold;
        this.nmsThreshold = nmsThreshold;
        this.topK = topK;
    }
    public void init() {
        detector = FaceDetectorYN.create(
                modelPath,
                "",
                new Size(640, 640),
                scoreThreshold,
                nmsThreshold,
                topK
        );
        if (detector == null) {
            throw new IllegalStateException(
                    "FaceDetectorYN.create() = null. Kiểm tra đường dẫn model & phiên bản OpenCV (>= 4.8).");
        }
        Log.i(TAG, "FaceDetectorYN khởi tạo thành công.");
    }
    public Mat detect(Mat bgr) {
        if (detector == null) return faces;
        if (inW != bgr.cols() || inH != bgr.rows()) {
            inW = bgr.cols();
            inH = bgr.rows();
            detector.setInputSize(new Size(inW, inH));
        }
        detector.detect(bgr, faces);
        return faces;
    }

    public void setScoreThreshold(float v) {
        scoreThreshold = v;
        if (detector != null) detector.setScoreThreshold(v);
    }

    public void setNmsThreshold(float v) {
        nmsThreshold = v;
        if (detector != null) detector.setNMSThreshold(v);
    }

    public void setTopK(int v) {
        topK = v;
        if (detector != null) detector.setTopK(v);
    }
    public void release() {
        faces.release();
        detector = null;
    }
}

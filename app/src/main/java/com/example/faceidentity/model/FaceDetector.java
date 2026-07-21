package com.example.faceidentity.model;

import org.opencv.core.Mat;

public interface FaceDetector {
    String name();
    void init();
    DetectionResult detect(Mat bgr);
    void release();
}

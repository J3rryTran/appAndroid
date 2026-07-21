package com.example.faceidentity.model;

import org.opencv.core.Mat;
public class NcnnDetector implements FaceDetector {

    private final String modelPath;

    public NcnnDetector(String modelPath) {
        this.modelPath = modelPath;
    }

    @Override
    public String name() {
        return "ncnn";
    }

    @Override
    public void init() {
        throw new UnsupportedOperationException(
                "ncnn chưa được tích hợp: cần JNI/C++ + ncnn-android prebuilt "
                        + "+ cặp file .param/.bin (xem README mục ncnn). File: " + modelPath);
    }

    @Override
    public DetectionResult detect(Mat bgr) {
        return DetectionResult.EMPTY;
    }

    @Override
    public void release() {
        // chưa có gì để giải phóng
    }
}

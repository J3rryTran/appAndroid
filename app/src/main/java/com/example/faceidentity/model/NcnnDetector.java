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
                "ncnn have not native: require JNI/C++ + ncnn-android prebuilt "
                        + "+ and file .param/.bin. File: " + modelPath);
    }

    @Override
    public DetectionResult detect(Mat bgr) {
        return DetectionResult.EMPTY;
    }

    @Override
    public void release() {
        return release;
    }
}

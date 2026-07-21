package com.example.faceidentity.model;

import java.util.Locale;

public final class DetectorFactory {

    private DetectorFactory() { }

    public static FaceDetector create(String fileName, String localPath, ModelConfig cfg) {
        String lower = fileName.toLowerCase(Locale.US);
        if (cfg == null) cfg = ModelConfig.defaultFor(fileName);

        if (lower.endsWith(".onnx")) {
            if ("yunet".equals(cfg.backend)) return new YuNetDetector(localPath);
            return new GenericOnnxDetector(localPath, cfg);
        }
        if (lower.endsWith(".tflite")) {
            return new TfliteRfbDetector(localPath, cfg);
        }
        if (lower.endsWith(".param") || lower.endsWith(".bin")) {
            return new NcnnDetector(localPath);
        }
        if (lower.endsWith(".pt") || lower.endsWith(".pth")) {
            throw new UnsupportedOperationException(
                    "Checkpoint PyTorch (.pt) không chạy được trên Android.\n"
                            + "Export sang ONNX trong môi trường đã train:\n"
                            + "yolo export model=" + fileName + " format=onnx imgsz=640");
        }
        throw new IllegalArgumentException("Định dạng model không hỗ trợ: " + fileName);
    }

    public static boolean isSupportedFile(String fileName) {
        String lower = fileName.toLowerCase(Locale.US);
        return lower.endsWith(".onnx") || lower.endsWith(".tflite")
                || lower.endsWith(".param") || lower.endsWith(".pt") || lower.endsWith(".pth");
    }
}

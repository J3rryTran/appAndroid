package com.example.faceidentity.model;

import java.util.Locale;

/**
 * [MODEL] Chọn backend detector theo TÊN FILE model:
 *
 *  - *.onnx  chứa "yunet"  -> YuNetDetector   (FaceDetectorYN, chuẩn OpenCV Zoo)
 *  - *.onnx  còn lại       -> RfbOnnxDetector (OpenCV DNN, decode RFB/Ultra-Light)
 *  - *.tflite              -> TfliteRfbDetector (TensorFlow Lite, decode RFB-style)
 *  - *.param / *.bin       -> NcnnDetector    (stub - cần tích hợp JNI, xem README)
 *
 * QUY ƯỚC ĐẶT TÊN: model YuNet phải có chữ "yunet" trong tên file.
 */
public final class DetectorFactory {

    private DetectorFactory() { }

    /**
     * @param fileName  tên file trong assets/models (để đoán backend)
     * @param localPath đường dẫn tuyệt đối sau khi copy ra internal storage
     */
    public static FaceDetector create(String fileName, String localPath) {
        String lower = fileName.toLowerCase(Locale.US);

        if (lower.endsWith(".onnx")) {
            return lower.contains("yunet")
                    ? new YuNetDetector(localPath)
                    : new RfbOnnxDetector(localPath);
        }
        if (lower.endsWith(".tflite")) {
            return new TfliteRfbDetector(localPath);
        }
        if (lower.endsWith(".param") || lower.endsWith(".bin")) {
            return new NcnnDetector(localPath);
        }
        throw new IllegalArgumentException("Định dạng model không hỗ trợ: " + fileName);
    }

    /** File có phải định dạng model mà app nhận không (để liệt kê từ assets). */
    public static boolean isSupportedFile(String fileName) {
        String lower = fileName.toLowerCase(Locale.US);
        return lower.endsWith(".onnx") || lower.endsWith(".tflite")
                || lower.endsWith(".param");
    }
}

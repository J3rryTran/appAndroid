package com.example.faceidentity.model;

import org.opencv.core.Mat;

/**
 * [MODEL] Giao diện chung cho MỌI face detector (YuNet, RFB-ONNX, TFLite, ncnn...).
 *
 * Quy ước:
 *  - init() ném RuntimeException nếu model không load được / không tương thích.
 *  - detect() nhận ảnh BGR (CV_8UC3) ở KÍCH THƯỚC BẤT KỲ; toạ độ kết quả nằm trong
 *    hệ toạ độ của ảnh đầu vào đó (pixel).
 *  - Mỗi implementation tự chọn ngưỡng score/NMS mặc định hợp với model của nó.
 */
public interface FaceDetector {

    /** Tên ngắn để hiển thị/log, vd "YuNet", "RFB-ONNX". */
    String name();

    /** Load model. Ném exception nếu thất bại. */
    void init();

    /** Detect trên ảnh BGR. Trả kết quả trong hệ toạ độ ảnh đầu vào. */
    DetectionResult detect(Mat bgr);

    /** Giải phóng tài nguyên native. */
    void release();
}

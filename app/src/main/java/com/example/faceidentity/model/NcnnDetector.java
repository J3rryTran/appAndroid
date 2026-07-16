package com.example.faceidentity.model;

import org.opencv.core.Mat;

/**
 * [MODEL] Điểm cắm cho backend ncnn (Tencent) - CHƯA tích hợp runtime.
 *
 * ncnn KHÔNG có Java API thuần - muốn dùng thật phải:
 *  1) Tải bản prebuilt "ncnn-YYYYMMDD-android-vulkan.zip" từ
 *     https://github.com/Tencent/ncnn/releases (chứa .a/.so + header cho từng ABI).
 *  2) Viết lớp JNI C++ (CMakeLists.txt + native-lib.cpp) bọc ncnn::Net:
 *     load_param()/load_model() -> Extractor -> input/extract.
 *  3) Model ncnn là CẶP FILE .param + .bin (không phải 1 file duy nhất).
 *  4) Bật externalNativeBuild trong app/build.gradle trỏ tới CMakeLists.
 *
 * Class này tồn tại để kiến trúc DetectorFactory có sẵn chỗ cắm: khi bạn tích hợp
 * JNI xong, chỉ cần thay phần thân các hàm dưới bằng lời gọi native.
 */
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

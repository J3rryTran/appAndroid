package com.example.faceidentity.utils;

import androidx.camera.core.ImageProxy;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.nio.ByteBuffer;

/**
 * [UTILS] Chuyển ImageProxy (RGBA_8888) -> Mat (BGR) cho YuNet.
 *
 * TỐI ƯU:
 *  - Tái sử dụng byte[] và các Mat (không new mỗi frame) -> giảm GC.
 *  - Dùng submat để bỏ padding (rowStride) mà không copy dữ liệu.
 *  - Xoay ảnh theo rotationDegrees để khuôn mặt luôn thẳng đứng.
 *
 * LƯU Ý: các Mat ở đây chỉ được dùng trên MỘT thread (analysis thread) nên an toàn.
 */
public class ImageUtils {

    private byte[] rowData;        // buffer đọc pixel, tái sử dụng
    private Mat rgbaMat;           // CV_8UC4 theo rowStride (còn padding)
    private Mat rgbaNoPad;         // view (submat) bỏ padding, đúng width
    private final Mat bgrMat = new Mat();     // kết quả BGR (trước khi xoay)
    private final Mat rotatedMat = new Mat(); // kết quả sau khi xoay

    /**
     * @param image frame từ CameraX ở định dạng OUTPUT_IMAGE_FORMAT_RGBA_8888
     * @return Mat BGR (CV_8UC3) đã xoay thẳng đứng. Mat này được TÁI SỬ DỤNG,
     *         chỉ dùng ngay trong frame hiện tại.
     */
    public Mat imageProxyToBgr(ImageProxy image) {
        ImageProxy.PlaneProxy plane = image.getPlanes()[0];
        ByteBuffer buffer = plane.getBuffer();
        int rowStride = plane.getRowStride();
        int pixelStride = plane.getPixelStride();   // = 4 với RGBA
        int width = image.getWidth();
        int height = image.getHeight();

        int matCols = rowStride / pixelStride;      // số pixel/hàng (gồm padding)
        int neededBytes = height * rowStride;

        // (Re)cấp phát buffer & Mat khi kích thước đổi (thường chỉ lần đầu).
        if (rowData == null || rowData.length != neededBytes) {
            rowData = new byte[neededBytes];
        }
        if (rgbaMat == null || rgbaMat.rows() != height || rgbaMat.cols() != matCols) {
            if (rgbaMat != null) rgbaMat.release();
            if (rgbaNoPad != null) rgbaNoPad.release();
            rgbaMat = new Mat(height, matCols, CvType.CV_8UC4);
            rgbaNoPad = rgbaMat.submat(0, height, 0, width); // view, chia sẻ buffer
        }

        // Đọc toàn bộ plane vào byte[] rồi nạp vào Mat.
        buffer.rewind();
        int toRead = Math.min(buffer.remaining(), neededBytes);
        buffer.get(rowData, 0, toRead);
        rgbaMat.put(0, 0, rowData);

        // RGBA -> BGR (YuNet dùng thứ tự BGR của OpenCV).
        Imgproc.cvtColor(rgbaNoPad, bgrMat, Imgproc.COLOR_RGBA2BGR);

        // Xoay theo hướng hiển thị.
        int rotation = image.getImageInfo().getRotationDegrees();
        return rotate(bgrMat, rotation);
    }

    private Mat rotate(Mat src, int degrees) {
        switch (degrees) {
            case 90:
                Core.rotate(src, rotatedMat, Core.ROTATE_90_CLOCKWISE);
                return rotatedMat;
            case 180:
                Core.rotate(src, rotatedMat, Core.ROTATE_180);
                return rotatedMat;
            case 270:
                Core.rotate(src, rotatedMat, Core.ROTATE_90_COUNTERCLOCKWISE);
                return rotatedMat;
            default:
                return src; // 0 độ -> dùng trực tiếp
        }
    }

    /** Giải phóng Mat native khi không dùng nữa. */
    public void release() {
        if (rgbaNoPad != null) rgbaNoPad.release();
        if (rgbaMat != null) rgbaMat.release();
        bgrMat.release();
        rotatedMat.release();
    }
}

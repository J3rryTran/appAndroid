package com.example.faceidentity.utils;

import androidx.camera.core.ImageProxy;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.nio.ByteBuffer;
public class ImageUtils {

    private byte[] rowData;
    private Mat rgbaMat;
    private Mat rgbaNoPad;
    private final Mat bgrMat = new Mat();
    private final Mat rotatedMat = new Mat();

    public Mat imageProxyToBgr(ImageProxy image) {
        ImageProxy.PlaneProxy plane = image.getPlanes()[0];
        ByteBuffer buffer = plane.getBuffer();
        int rowStride = plane.getRowStride();
        int pixelStride = plane.getPixelStride();
        int width = image.getWidth();
        int height = image.getHeight();
        int matCols = rowStride / pixelStride;
        int neededBytes = height * rowStride;

        if (rowData == null || rowData.length != neededBytes) {
            rowData = new byte[neededBytes];
        }
        if (rgbaMat == null || rgbaMat.rows() != height || rgbaMat.cols() != matCols) {
            if (rgbaMat != null) rgbaMat.release();
            if (rgbaNoPad != null) rgbaNoPad.release();
            rgbaMat = new Mat(height, matCols, CvType.CV_8UC4);
            rgbaNoPad = rgbaMat.submat(0, height, 0, width);
        }

        buffer.rewind();
        int toRead = Math.min(buffer.remaining(), neededBytes);
        buffer.get(rowData, 0, toRead);
        rgbaMat.put(0, 0, rowData);

        Imgproc.cvtColor(rgbaNoPad, bgrMat, Imgproc.COLOR_RGBA2BGR);

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
                return src;
        }
    }

    public void release() {
        if (rgbaNoPad != null) rgbaNoPad.release();
        if (rgbaMat != null) rgbaMat.release();
        bgrMat.release();
        rotatedMat.release();
    }
}

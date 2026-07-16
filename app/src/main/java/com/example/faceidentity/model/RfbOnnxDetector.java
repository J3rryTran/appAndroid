package com.example.faceidentity.model;

import android.util.Log;

import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect2d;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;

import java.util.ArrayList;
import java.util.List;

/**
 * [MODEL] Backend cho Ultra-Light-Fast-Generic-Face-Detector (RFB) bản ONNX,
 * gồm cả biến thể có LANDMARK (vd RFB-landmark-Epoch-...onnx).
 *
 * Chạy bằng OpenCV DNN (readNetFromONNX) - KHÔNG đi qua FaceDetectorYN.
 *
 * Giả định theo chuẩn export của repo Ultra-Light (is_test=True, đã bake decode):
 *  - Input : 1x3x240x320 RGB, chuẩn hoá (pixel - 127) / 128.
 *  - Output: nhận diện THEO SỐ KÊNH CUỐI (không phụ thuộc thứ tự/tên):
 *      C=2  -> scores    [1, N, 2]  (nền, mặt) - đã softmax
 *      C=4  -> boxes     [1, N, 4]  corner-form [x1,y1,x2,y2], CHUẨN HOÁ 0..1
 *      C=10 -> landmarks [1, N, 10] 5 điểm (x,y), chuẩn hoá 0..1
 *  - Hậu xử lý: lọc score -> NMS (Dnn.NMSBoxes) -> scale về pixel.
 *
 * Nếu box vẽ ra sai vị trí toàn bộ: bản ONNX của bạn chưa bake decode (export thiếu
 * is_test=True) -> export lại, hoặc phải tự decode prior box (xem README).
 */
public class RfbOnnxDetector implements FaceDetector {

    private static final String TAG = "RfbOnnxDetector";

    // Cấu hình chuẩn RFB-320. Model train ở size khác thì sửa 2 hằng số này.
    private static final int INPUT_W = 320;
    private static final int INPUT_H = 240;

    // Ngưỡng thường dùng cho Ultra-Light RFB.
    private static final float SCORE_THRESHOLD = 0.7f;
    private static final float NMS_THRESHOLD   = 0.3f;

    private final String modelPath;
    private Net net;
    private List<String> outNames;

    // Tái sử dụng giữa các frame
    private final List<Mat> outputs = new ArrayList<>();
    private float[] scoreBuf;
    private float[] boxBuf;
    private float[] landBuf;

    public RfbOnnxDetector(String modelPath) {
        this.modelPath = modelPath;
    }

    @Override
    public String name() {
        return "RFB-ONNX";
    }

    @Override
    public void init() {
        net = Dnn.readNetFromONNX(modelPath);
        outNames = net.getUnconnectedOutLayersNames();
        Log.i(TAG, "RFB init OK, outputs=" + outNames);
    }

    @Override
    public DetectionResult detect(Mat bgr) {
        if (net == null) return DetectionResult.EMPTY;

        final int W = bgr.cols();
        final int H = bgr.rows();

        // 1) Tiền xử lý: resize 320x240, (x-127)/128, BGR->RGB (swapRB=true)
        Mat blob = Dnn.blobFromImage(bgr, 1.0 / 128.0,
                new Size(INPUT_W, INPUT_H), new Scalar(127, 127, 127),
                true, false);
        net.setInput(blob);

        // 2) Forward tất cả output
        outputs.clear();
        net.forward(outputs, outNames);
        blob.release();

        // 3) Nhận diện từng output theo số kênh cuối cùng
        float[] scores = null, boxes = null, lands = null;
        int n = 0;
        for (Mat o : outputs) {
            int dims = o.dims();
            int c = o.size(dims - 1);                 // kênh cuối
            int rows = (int) (o.total() / c);         // = N
            if (c == 2) {
                scoreBuf = ensure(scoreBuf, rows * 2);
                o.get(new int[dims], scoreBuf);       // đọc toàn bộ (index 0,0,..)
                scores = scoreBuf;
                n = rows;
            } else if (c == 4) {
                boxBuf = ensure(boxBuf, rows * 4);
                o.get(new int[dims], boxBuf);
                boxes = boxBuf;
            } else if (c == 10) {
                landBuf = ensure(landBuf, rows * 10);
                o.get(new int[dims], landBuf);
                lands = landBuf;
            }
        }
        if (scores == null || boxes == null || n == 0) {
            throw new IllegalStateException(
                    "Output không đúng dạng RFB (cần các tensor kênh cuối = 2 và 4). "
                            + "Model này có thể chưa bake decode khi export.");
        }

        // 4) Lọc theo score
        List<Rect2d> candRects = new ArrayList<>();
        List<Float> candScores = new ArrayList<>();
        List<Integer> candIdx = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            float s = scores[i * 2 + 1];              // kênh 1 = mặt
            if (s < SCORE_THRESHOLD) continue;
            float x1 = clamp(boxes[i * 4]     * W, 0, W - 1);
            float y1 = clamp(boxes[i * 4 + 1] * H, 0, H - 1);
            float x2 = clamp(boxes[i * 4 + 2] * W, 0, W - 1);
            float y2 = clamp(boxes[i * 4 + 3] * H, 0, H - 1);
            if (x2 <= x1 || y2 <= y1) continue;
            candRects.add(new Rect2d(x1, y1, x2 - x1, y2 - y1));
            candScores.add(s);
            candIdx.add(i);
        }
        if (candRects.isEmpty()) return DetectionResult.EMPTY;

        // 5) NMS bằng OpenCV
        MatOfRect2d rectsMat = new MatOfRect2d();
        rectsMat.fromList(candRects);
        MatOfFloat scoresMat = new MatOfFloat();
        scoresMat.fromList(candScores);
        MatOfInt keep = new MatOfInt();
        Dnn.NMSBoxes(rectsMat, scoresMat, SCORE_THRESHOLD, NMS_THRESHOLD, keep);

        int[] keepIdx = keep.rows() > 0 ? keep.toArray() : new int[0];
        rectsMat.release();
        scoresMat.release();
        keep.release();
        if (keepIdx.length == 0) return DetectionResult.EMPTY;

        // 6) Đóng gói kết quả
        float[] outBoxes = new float[keepIdx.length * 4];
        float[] outScores = new float[keepIdx.length];
        float[] outLands = (lands != null) ? new float[keepIdx.length * 10] : null;
        for (int k = 0; k < keepIdx.length; k++) {
            Rect2d r = candRects.get(keepIdx[k]);
            outBoxes[k * 4]     = (float) r.x;
            outBoxes[k * 4 + 1] = (float) r.y;
            outBoxes[k * 4 + 2] = (float) r.width;
            outBoxes[k * 4 + 3] = (float) r.height;
            outScores[k] = candScores.get(keepIdx[k]);
            if (outLands != null) {
                int src = candIdx.get(keepIdx[k]) * 10;
                for (int p = 0; p < 5; p++) {
                    outLands[k * 10 + p * 2]     = lands[src + p * 2]     * W;
                    outLands[k * 10 + p * 2 + 1] = lands[src + p * 2 + 1] * H;
                }
            }
        }
        return new DetectionResult(outBoxes, outScores, outLands);
    }

    private static float[] ensure(float[] buf, int len) {
        return (buf == null || buf.length != len) ? new float[len] : buf;
    }

    private static float clamp(float v, float lo, float hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    @Override
    public void release() {
        net = null;
    }
}

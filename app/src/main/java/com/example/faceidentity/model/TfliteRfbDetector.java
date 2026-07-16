package com.example.faceidentity.model;

import android.util.Log;

import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect2d;
import org.opencv.core.Rect2d;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.imgproc.Imgproc;

import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * [MODEL] Backend TensorFlow Lite cho model face-detect kiểu RFB/Ultra-Light
 * (scores [1,N,2] + boxes [1,N,4] corner-form chuẩn hoá, tuỳ chọn landmarks [1,N,10]).
 *
 * - Input size đọc TỰ ĐỘNG từ tensor input ([1,H,W,3] float32).
 * - Output nhận diện theo SỐ KÊNH CUỐI, không phụ thuộc thứ tự tensor.
 * - Tiền xử lý: (pixel - 127) / 128, RGB.
 *
 * Model TFLite kiểu khác (BlazeFace, SSD-anchor thô...) cần decoder riêng - xem README.
 */
public class TfliteRfbDetector implements FaceDetector {

    private static final String TAG = "TfliteRfbDetector";

    private static final float SCORE_THRESHOLD = 0.7f;
    private static final float NMS_THRESHOLD   = 0.3f;

    private final String modelPath;
    private Interpreter interpreter;

    private int inW = 320;
    private int inH = 240;
    private int scoreIdx = -1;
    private int boxIdx = -1;
    private int landIdx = -1;
    private int numAnchors = 0;

    // Tái sử dụng giữa các frame
    private ByteBuffer inputBuf;
    private byte[] pixelBuf;
    private final Mat resized = new Mat();
    private float[][][] scoresOut;
    private float[][][] boxesOut;
    private float[][][] landsOut;

    public TfliteRfbDetector(String modelPath) {
        this.modelPath = modelPath;
    }

    @Override
    public String name() {
        return "RFB-TFLite";
    }

    @Override
    public void init() {
        interpreter = new Interpreter(new File(modelPath));

        int[] ishape = interpreter.getInputTensor(0).shape();   // kỳ vọng [1,H,W,3]
        if (ishape.length == 4) {
            inH = ishape[1];
            inW = ishape[2];
        }

        for (int i = 0; i < interpreter.getOutputTensorCount(); i++) {
            int[] s = interpreter.getOutputTensor(i).shape();    // kỳ vọng [1,N,C]
            int c = s[s.length - 1];
            int nn = (s.length >= 2) ? s[s.length - 2] : 0;
            if (c == 2) { scoreIdx = i; numAnchors = nn; }
            else if (c == 4)  boxIdx = i;
            else if (c == 10) landIdx = i;
        }
        if (scoreIdx < 0 || boxIdx < 0 || numAnchors <= 0) {
            throw new IllegalStateException(
                    "TFLite output không đúng dạng RFB (cần tensor kênh cuối = 2 và 4).");
        }

        scoresOut = new float[1][numAnchors][2];
        boxesOut  = new float[1][numAnchors][4];
        if (landIdx >= 0) landsOut = new float[1][numAnchors][10];

        inputBuf = ByteBuffer.allocateDirect(inW * inH * 3 * 4).order(ByteOrder.nativeOrder());
        pixelBuf = new byte[inW * inH * 3];

        Log.i(TAG, "TFLite init OK. input=" + inW + "x" + inH + ", anchors=" + numAnchors);
    }

    @Override
    public DetectionResult detect(Mat bgr) {
        if (interpreter == null) return DetectionResult.EMPTY;

        final int W = bgr.cols();
        final int H = bgr.rows();

        // 1) Resize + chuẩn hoá (x-127)/128 + BGR->RGB, ghi vào ByteBuffer tái sử dụng
        Imgproc.resize(bgr, resized, new Size(inW, inH));
        resized.get(0, 0, pixelBuf);
        inputBuf.rewind();
        for (int i = 0; i < pixelBuf.length; i += 3) {
            inputBuf.putFloat(((pixelBuf[i + 2] & 0xFF) - 127) / 128f);  // R
            inputBuf.putFloat(((pixelBuf[i + 1] & 0xFF) - 127) / 128f);  // G
            inputBuf.putFloat(((pixelBuf[i]     & 0xFF) - 127) / 128f);  // B
        }
        inputBuf.rewind();

        // 2) Chạy inference
        Map<Integer, Object> outs = new HashMap<>();
        outs.put(scoreIdx, scoresOut);
        outs.put(boxIdx, boxesOut);
        if (landIdx >= 0) outs.put(landIdx, landsOut);
        interpreter.runForMultipleInputsOutputs(new Object[]{inputBuf}, outs);

        // 3) Lọc theo score
        List<Rect2d> candRects = new ArrayList<>();
        List<Float> candScores = new ArrayList<>();
        List<Integer> candIdx = new ArrayList<>();
        for (int i = 0; i < numAnchors; i++) {
            float s = scoresOut[0][i][1];
            if (s < SCORE_THRESHOLD) continue;
            float x1 = clamp(boxesOut[0][i][0] * W, 0, W - 1);
            float y1 = clamp(boxesOut[0][i][1] * H, 0, H - 1);
            float x2 = clamp(boxesOut[0][i][2] * W, 0, W - 1);
            float y2 = clamp(boxesOut[0][i][3] * H, 0, H - 1);
            if (x2 <= x1 || y2 <= y1) continue;
            candRects.add(new Rect2d(x1, y1, x2 - x1, y2 - y1));
            candScores.add(s);
            candIdx.add(i);
        }
        if (candRects.isEmpty()) return DetectionResult.EMPTY;

        // 4) NMS bằng OpenCV
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

        // 5) Đóng gói
        float[] outBoxes = new float[keepIdx.length * 4];
        float[] outScores = new float[keepIdx.length];
        float[] outLands = (landIdx >= 0) ? new float[keepIdx.length * 10] : null;
        for (int k = 0; k < keepIdx.length; k++) {
            Rect2d r = candRects.get(keepIdx[k]);
            outBoxes[k * 4]     = (float) r.x;
            outBoxes[k * 4 + 1] = (float) r.y;
            outBoxes[k * 4 + 2] = (float) r.width;
            outBoxes[k * 4 + 3] = (float) r.height;
            outScores[k] = candScores.get(keepIdx[k]);
            if (outLands != null) {
                int src = candIdx.get(keepIdx[k]);
                for (int p = 0; p < 5; p++) {
                    outLands[k * 10 + p * 2]     = landsOut[0][src][p * 2]     * W;
                    outLands[k * 10 + p * 2 + 1] = landsOut[0][src][p * 2 + 1] * H;
                }
            }
        }
        return new DetectionResult(outBoxes, outScores, outLands);
    }

    private static float clamp(float v, float lo, float hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    @Override
    public void release() {
        if (interpreter != null) {
            interpreter.close();
            interpreter = null;
        }
        resized.release();
    }
}

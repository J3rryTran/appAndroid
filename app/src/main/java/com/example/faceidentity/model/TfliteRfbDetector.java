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
public class TfliteRfbDetector implements FaceDetector {

    private static final String TAG = "TfliteRfbDetector";

    private final String modelPath;
    private final ModelConfig cfg;
    private Interpreter interpreter;

    private int inW = 320;
    private int inH = 320;
    private int scoreIdx = -1;
    private int boxIdx = -1;
    private int landIdx = -1;
    private int numAnchors = 0;

    private ByteBuffer inputBuf;
    private byte[] pixelBuf;
    private final Mat resized = new Mat();
    private float[][][] scoresOut;
    private float[][][] boxesOut;
    private float[][][] landsOut;

    public TfliteRfbDetector(String modelPath, ModelConfig cfg) {
        this.modelPath = modelPath;
        this.cfg = cfg;
    }

    @Override
    public String name() {
        return "RFB-TFLite";
    }

    @Override
    public void init() {
        interpreter = new Interpreter(new File(modelPath));

        int[] ishape = interpreter.getInputTensor(0).shape();
        if (ishape.length == 4) {
            inH = ishape[1];
            inW = ishape[2];
        }

        for (int i = 0; i < interpreter.getOutputTensorCount(); i++) {
            int[] s = interpreter.getOutputTensor(i).shape();
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

        Imgproc.resize(bgr, resized, new Size(inW, inH));
        resized.get(0, 0, pixelBuf);
        inputBuf.rewind();
        float sc = (float) cfg.scale;
        float m0 = (float) cfg.mean[0], m1 = (float) cfg.mean[1], m2 = (float) cfg.mean[2];
        for (int i = 0; i < pixelBuf.length; i += 3) {
            inputBuf.putFloat(((pixelBuf[i + 2] & 0xFF) - m0) * sc);  // R
            inputBuf.putFloat(((pixelBuf[i + 1] & 0xFF) - m1) * sc);  // G
            inputBuf.putFloat(((pixelBuf[i]     & 0xFF) - m2) * sc);  // B
        }
        inputBuf.rewind();

        Map<Integer, Object> outs = new HashMap<>();
        outs.put(scoreIdx, scoresOut);
        outs.put(boxIdx, boxesOut);
        if (landIdx >= 0) outs.put(landIdx, landsOut);
        interpreter.runForMultipleInputsOutputs(new Object[]{inputBuf}, outs);

        List<Rect2d> candRects = new ArrayList<>();
        List<Float> candScores = new ArrayList<>();
        List<Integer> candIdx = new ArrayList<>();
        for (int i = 0; i < numAnchors; i++) {
            float s = scoresOut[0][i][1];
            if (s < cfg.scoreThreshold) continue;
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

        MatOfRect2d rectsMat = new MatOfRect2d();
        rectsMat.fromList(candRects);
        MatOfFloat scoresMat = new MatOfFloat();
        scoresMat.fromList(candScores);
        MatOfInt keep = new MatOfInt();
        Dnn.NMSBoxes(rectsMat, scoresMat, cfg.scoreThreshold, cfg.nmsThreshold, keep);

        int[] keepIdx = keep.rows() > 0 ? keep.toArray() : new int[0];
        rectsMat.release();
        scoresMat.release();
        keep.release();
        if (keepIdx.length == 0) return DetectionResult.EMPTY;

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

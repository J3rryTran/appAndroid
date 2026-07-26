package com.example.faceidentity.model;

import android.util.Log;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect2d;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.imgproc.Imgproc;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * TFLite runtime (XNNPACK, đa luồng). Nhận diện output như GenericOnnxDetector:
 * e2e [1,N<=1000,6+k] / v5 [1,N,6] / raw [1,C,N] / tách kênh RFB (2|1 + 4 + 10).
 */
public class TfliteDetector implements FaceDetector {

    private static final String TAG = "TfliteDetector";
    private static final int NUM_THREADS = 4;

    private final String modelPath;
    private final ModelConfig cfg;
    private final String label;
    private Interpreter interpreter;

    private int inW, inH;
    private ByteBuffer inputBuf;
    private byte[] pixelBuf;
    private final Mat resized = new Mat();
    private Mat lbCanvas;
    private float lbScale = 1f, lbPadX = 0f, lbPadY = 0f;

    private int[][] outShapes;
    private ByteBuffer[] outBufs;
    private float[][] outData;
    private boolean shapesLogged = false;
    private int statCalls = 0;

    private static class Parsed {
        float[] scores;
        float[] boxes;
        float[] lands;
        int n;
        boolean center;
    }

    public TfliteDetector(String modelPath, ModelConfig cfg) {
        this.modelPath = modelPath;
        this.cfg = cfg;
        int cut = Math.max(modelPath.lastIndexOf('/'), modelPath.lastIndexOf('\\'));
        this.label = modelPath.substring(cut + 1).replaceAll("\\.[^.]+$", "");
    }

    @Override
    public String name() {
        return "TFLite";
    }

    @Override
    public void init() {
        Interpreter.Options opt = new Interpreter.Options();
        opt.setNumThreads(NUM_THREADS);
        opt.setUseXNNPACK(true);
        interpreter = new Interpreter(new File(modelPath), opt);

        if (interpreter.getInputTensor(0).dataType() != DataType.FLOAT32) {
            throw new IllegalStateException("Chỉ hỗ trợ TFLite float32/fp16 — export lại không dùng int8 (input="
                    + interpreter.getInputTensor(0).dataType() + ")");
        }
        int[] ishape = interpreter.getInputTensor(0).shape();   // [1,H,W,3]
        inW = cfg.inputW;
        inH = cfg.inputH;
        if (ishape.length == 4 && ishape[3] == 3) {
            inH = ishape[1];
            inW = ishape[2];
        }

        int outs = interpreter.getOutputTensorCount();
        outShapes = new int[outs][];
        outBufs = new ByteBuffer[outs];
        outData = new float[outs][];
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < outs; i++) {
            if (interpreter.getOutputTensor(i).dataType() != DataType.FLOAT32) {
                throw new IllegalStateException("Output " + i + " không phải float32 — export không dùng int8");
            }
            int[] s = interpreter.getOutputTensor(i).shape();
            outShapes[i] = s;
            int numel = 1;
            for (int d : s) numel *= d;
            outBufs[i] = ByteBuffer.allocateDirect(numel * 4).order(ByteOrder.nativeOrder());
            outData[i] = new float[numel];
            sb.append(Arrays.toString(s)).append(' ');
        }
        inputBuf = ByteBuffer.allocateDirect(inW * inH * 3 * 4).order(ByteOrder.nativeOrder());
        pixelBuf = new byte[inW * inH * 3];
        Log.i(TAG, "[" + label + "] init OK input=" + inW + "x" + inH + " letterbox=" + cfg.letterbox
                + " threads=" + NUM_THREADS + " XNNPACK=on outputs=" + sb);
    }

    @Override
    public DetectionResult detect(Mat bgr) {
        if (interpreter == null) return DetectionResult.EMPTY;
        final int W = bgr.cols(), H = bgr.rows();

        Mat feed;
        if (cfg.letterbox) {
            feed = letterbox(bgr);
        } else {
            Imgproc.resize(bgr, resized, new Size(inW, inH));
            feed = resized;
        }
        feed.get(0, 0, pixelBuf);
        inputBuf.rewind();
        float sc = (float) cfg.scale;
        float m0 = (float) cfg.mean[0], m1 = (float) cfg.mean[1], m2 = (float) cfg.mean[2];
        if (cfg.swapRB) {
            for (int i = 0; i < pixelBuf.length; i += 3) {
                inputBuf.putFloat(((pixelBuf[i + 2] & 0xFF) - m0) * sc);
                inputBuf.putFloat(((pixelBuf[i + 1] & 0xFF) - m1) * sc);
                inputBuf.putFloat(((pixelBuf[i] & 0xFF) - m2) * sc);
            }
        } else {
            for (int i = 0; i < pixelBuf.length; i += 3) {
                inputBuf.putFloat(((pixelBuf[i] & 0xFF) - m0) * sc);
                inputBuf.putFloat(((pixelBuf[i + 1] & 0xFF) - m1) * sc);
                inputBuf.putFloat(((pixelBuf[i + 2] & 0xFF) - m2) * sc);
            }
        }
        inputBuf.rewind();

        Map<Integer, Object> outs = new HashMap<>();
        for (int i = 0; i < outBufs.length; i++) {
            outBufs[i].rewind();
            outs.put(i, outBufs[i]);
        }
        interpreter.runForMultipleInputsOutputs(new Object[]{inputBuf}, outs);
        for (int i = 0; i < outBufs.length; i++) {
            outBufs[i].rewind();
            outBufs[i].asFloatBuffer().get(outData[i]);
        }

        if (!shapesLogged) {
            shapesLogged = true;
            Log.i(TAG, "[" + label + "] outputs=" + shapes());
        }

        Parsed p = parse();
        if (p == null || p.n == 0) {
            throw new IllegalStateException("Output không khớp định dạng nào. Shapes=" + shapes());
        }
        if (cfg.scoreScale != 1f) {
            for (int i = 0; i < p.n; i++) p.scores[i] *= cfg.scoreScale;
        }
        if (statCalls < 3 || statCalls % 60 == 0) logStats(p);
        statCalls++;

        // normalized -> pixel theo input; rồi map input -> ảnh (letterbox hoặc stretch)
        final float toX = cfg.normalized ? inW : 1f;
        final float toY = cfg.normalized ? inH : 1f;
        final float sx, sy, ox, oy;
        if (cfg.letterbox) {
            sx = 1f / lbScale; sy = 1f / lbScale; ox = lbPadX; oy = lbPadY;
        } else {
            sx = (float) W / inW; sy = (float) H / inH; ox = 0f; oy = 0f;
        }

        List<Rect2d> rects = new ArrayList<>();
        List<Float> scs = new ArrayList<>();
        List<Integer> idxs = new ArrayList<>();
        for (int i = 0; i < p.n; i++) {
            float s = p.scores[i];
            if (s < cfg.scoreThreshold) continue;
            float a = p.boxes[i * 4] * toX, b = p.boxes[i * 4 + 1] * toY;
            float c = p.boxes[i * 4 + 2] * toX, d = p.boxes[i * 4 + 3] * toY;
            float x1, y1, x2, y2;
            if (p.center) {
                x1 = (a - c / 2 - ox) * sx; y1 = (b - d / 2 - oy) * sy;
                x2 = (a + c / 2 - ox) * sx; y2 = (b + d / 2 - oy) * sy;
            } else {
                x1 = (a - ox) * sx; y1 = (b - oy) * sy;
                x2 = (c - ox) * sx; y2 = (d - oy) * sy;
            }
            if (cfg.boxScale != 1f) {
                float bcx = (x1 + x2) / 2f, bcy = (y1 + y2) / 2f;
                float hw = (x2 - x1) * cfg.boxScale / 2f, hh = (y2 - y1) * cfg.boxScale / 2f;
                x1 = bcx - hw; x2 = bcx + hw; y1 = bcy - hh; y2 = bcy + hh;
            }
            x1 = clamp(x1, 0, W - 1); y1 = clamp(y1, 0, H - 1);
            x2 = clamp(x2, 0, W - 1); y2 = clamp(y2, 0, H - 1);
            if (x2 <= x1 || y2 <= y1) continue;
            rects.add(new Rect2d(x1, y1, x2 - x1, y2 - y1));
            scs.add(s);
            idxs.add(i);
        }
        if (rects.isEmpty()) return DetectionResult.EMPTY;

        MatOfRect2d rm = new MatOfRect2d(); rm.fromList(rects);
        MatOfFloat sm = new MatOfFloat(); sm.fromList(scs);
        MatOfInt keep = new MatOfInt();
        Dnn.NMSBoxes(rm, sm, cfg.scoreThreshold, cfg.nmsThreshold, keep);
        int[] kp = keep.rows() > 0 ? keep.toArray() : new int[0];
        rm.release(); sm.release(); keep.release();
        if (kp.length == 0) return DetectionResult.EMPTY;

        float[] ob = new float[kp.length * 4];
        float[] os = new float[kp.length];
        float[] ol = (p.lands != null) ? new float[kp.length * 10] : null;
        for (int k = 0; k < kp.length; k++) {
            Rect2d r = rects.get(kp[k]);
            ob[k * 4] = (float) r.x; ob[k * 4 + 1] = (float) r.y;
            ob[k * 4 + 2] = (float) r.width; ob[k * 4 + 3] = (float) r.height;
            os[k] = scs.get(kp[k]);
            if (ol != null) {
                int src = idxs.get(kp[k]) * 10;
                for (int q = 0; q < 5; q++) {
                    ol[k * 10 + q * 2]     = (p.lands[src + q * 2] * toX - ox) * sx;
                    ol[k * 10 + q * 2 + 1] = (p.lands[src + q * 2 + 1] * toY - oy) * sy;
                }
            }
        }
        return new DetectionResult(ob, os, ol);
    }

    private Mat letterbox(Mat bgr) {
        final int W = bgr.cols(), H = bgr.rows();
        float r = Math.min((float) inW / W, (float) inH / H);
        int nw = Math.max(1, Math.round(W * r)), nh = Math.max(1, Math.round(H * r));
        lbScale = r;
        lbPadX = (inW - nw) / 2f;
        lbPadY = (inH - nh) / 2f;

        Imgproc.resize(bgr, resized, new Size(nw, nh));
        if (lbCanvas == null || lbCanvas.cols() != inW || lbCanvas.rows() != inH) {
            if (lbCanvas != null) lbCanvas.release();
            lbCanvas = new Mat(inH, inW, CvType.CV_8UC3);
        }
        lbCanvas.setTo(new Scalar(114, 114, 114));
        int top = (int) lbPadY, left = (int) lbPadX;
        Mat roi = lbCanvas.submat(top, top + nh, left, left + nw);
        resized.copyTo(roi);
        roi.release();
        return lbCanvas;
    }

    private Parsed parse() {
        for (int t = 0; t < outShapes.length; t++) {
            int[] s = outShapes[t];
            if (s.length != 3 || s[0] != 1) continue;
            int d1 = s[1], d2 = s[2];
            float[] buf = outData[t];

            if (d1 <= 1000 && d2 >= 6 && d2 <= 60) {   // e2e: xyxy,score,cls[,5kpt×(2|3)]
                Parsed p = new Parsed();
                p.n = d1;
                p.center = false;
                p.boxes = new float[d1 * 4];
                p.scores = new float[d1];
                int extra = d2 - 6;
                int kd = (extra >= 10 && extra % 5 == 0) ? extra / 5 : 0;
                if (kd >= 2) p.lands = new float[d1 * 10];
                for (int i = 0; i < d1; i++) {
                    int base = i * d2;
                    System.arraycopy(buf, base, p.boxes, i * 4, 4);
                    p.scores[i] = buf[base + 4];
                    if (kd >= 2) {
                        for (int q = 0; q < 5; q++) {
                            p.lands[i * 10 + q * 2]     = buf[base + 6 + q * kd];
                            p.lands[i * 10 + q * 2 + 1] = buf[base + 6 + q * kd + 1];
                        }
                    }
                }
                return p;
            }

            if (d2 == 6 && d1 > 1000) {                 // v5-style: cxcywh,obj,cls
                Parsed p = new Parsed();
                p.n = d1;
                p.center = true;
                p.boxes = new float[d1 * 4];
                p.scores = new float[d1];
                for (int i = 0; i < d1; i++) {
                    System.arraycopy(buf, i * 6, p.boxes, i * 4, 4);
                    p.scores[i] = buf[i * 6 + 4] * buf[i * 6 + 5];
                }
                return p;
            }

            if (d1 >= 5 && d1 <= 200 && d2 > d1 * 8) {   // raw [1,C,N] channels-first
                Parsed p = new Parsed();
                p.n = d2;
                p.center = "center".equals(cfg.box);
                p.boxes = new float[d2 * 4];
                p.scores = new float[d2];
                int kd = (d1 > 5 && (d1 - 5) % 5 == 0) ? (d1 - 5) / 5 : 0;
                if (kd >= 2) p.lands = new float[d2 * 10];
                for (int j = 0; j < d2; j++) {
                    p.boxes[j * 4]     = buf[j];
                    p.boxes[j * 4 + 1] = buf[d2 + j];
                    p.boxes[j * 4 + 2] = buf[2 * d2 + j];
                    p.boxes[j * 4 + 3] = buf[3 * d2 + j];
                    if (kd >= 2) {
                        p.scores[j] = buf[4 * d2 + j];
                        for (int q = 0; q < 5; q++) {
                            p.lands[j * 10 + q * 2]     = buf[(5 + q * kd) * d2 + j];
                            p.lands[j * 10 + q * 2 + 1] = buf[(5 + q * kd + 1) * d2 + j];
                        }
                    } else {
                        float best = 0f;
                        for (int k = 4; k < d1; k++) {
                            float v = buf[k * d2 + j];
                            if (v > best) best = v;
                        }
                        p.scores[j] = best;
                    }
                }
                return p;
            }
        }

        // Tách kênh RFB: score(1|2) + box(4) + landmark(10), row-major
        final int scoreCh = "sigmoid1".equals(cfg.score) ? 1 : 2;
        Parsed p = new Parsed();
        p.center = "center".equals(cfg.box);
        float[] rawScores = null;
        for (int t = 0; t < outShapes.length; t++) {
            int[] s = outShapes[t];
            int c = s[s.length - 1];
            int rows = outData[t].length / c;
            if (c == scoreCh && rawScores == null) {
                rawScores = outData[t];
                p.n = rows;
            } else if (c == 4 && p.boxes == null) {
                p.boxes = outData[t];
            } else if (c == 10 && p.lands == null) {
                p.lands = outData[t];
            }
        }
        if (rawScores == null || p.boxes == null || p.n == 0) return null;
        p.scores = new float[p.n];
        for (int i = 0; i < p.n; i++) {
            p.scores[i] = (scoreCh == 1) ? rawScores[i] : rawScores[i * 2 + 1];
        }
        return p;
    }

    private void logStats(Parsed p) {
        float mx = 0f, sum = 0f;
        int idx = 0;
        for (int i = 0; i < p.n; i++) {
            sum += p.scores[i];
            if (p.scores[i] > mx) { mx = p.scores[i]; idx = i; }
        }
        Log.i(TAG, String.format(Locale.US,
                "[%s] STAT n=%d scoreMax=%.4f mean=%.4f thr=%.2f | best row[%d]: box=[%.2f %.2f %.2f %.2f]",
                label, p.n, mx, sum / p.n, cfg.scoreThreshold, idx,
                p.boxes[idx * 4], p.boxes[idx * 4 + 1], p.boxes[idx * 4 + 2], p.boxes[idx * 4 + 3]));
    }

    private String shapes() {
        StringBuilder sb = new StringBuilder();
        for (int[] s : outShapes) sb.append(Arrays.toString(s)).append(' ');
        return sb.toString();
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
        if (lbCanvas != null) {
            lbCanvas.release();
            lbCanvas = null;
        }
    }
}

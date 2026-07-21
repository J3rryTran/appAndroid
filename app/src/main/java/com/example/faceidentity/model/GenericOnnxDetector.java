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
 * ONNX detector cấu hình bằng ModelConfig. Tự nhận diện dạng output:
 *  - YOLO e2e (YOLO26/v10):    [1, N<=1000, 6] = x1,y1,x2,y2,score,cls (pixel)
 *  - YOLO v5-style:            [1, N>1000, 6]  = cx,cy,w,h,obj,cls (pixel)
 *  - YOLO raw (v8/26 no-e2e):  [1, C, N] channels-first, C=4+nc (pixel)
 *  - Tách kênh (RFB...):       score(1|2) + box(4) + landmark(10) theo cfg
 */
public class GenericOnnxDetector implements FaceDetector {

    private static final String TAG = "GenericOnnx";

    private final String modelPath;
    private final ModelConfig cfg;
    private Net net;
    private List<String> outNames;
    private final List<Mat> outputs = new ArrayList<>();
    private boolean shapesLogged = false;

    private static class Parsed {
        float[] scores;
        float[] boxes;
        float[] lands;
        int n;
        boolean center;
        boolean pixel;
    }

    public GenericOnnxDetector(String modelPath, ModelConfig cfg) {
        this.modelPath = modelPath;
        this.cfg = cfg;
    }

    @Override
    public String name() {
        return "ONNX";
    }

    @Override
    public void init() {
        net = Dnn.readNetFromONNX(modelPath);
        outNames = net.getUnconnectedOutLayersNames();
        Log.i(TAG, "init OK input=" + cfg.inputW + "x" + cfg.inputH + " outputs=" + outNames);
    }

    @Override
    public DetectionResult detect(Mat bgr) {
        if (net == null) return DetectionResult.EMPTY;
        final int W = bgr.cols(), H = bgr.rows();

        Mat blob = Dnn.blobFromImage(bgr, cfg.scale, new Size(cfg.inputW, cfg.inputH),
                new Scalar(cfg.mean[0], cfg.mean[1], cfg.mean[2]), cfg.swapRB, false);
        net.setInput(blob);
        outputs.clear();
        net.forward(outputs, outNames);
        blob.release();

        if (!shapesLogged) {
            shapesLogged = true;
            Log.i(TAG, "outputs=" + shapes());
        }

        Parsed p = parse();
        if (p == null || p.n == 0) {
            throw new IllegalStateException("Output không khớp định dạng nào. Shapes=" + shapes());
        }

        final float sx = p.pixel ? (float) W / cfg.inputW : W;
        final float sy = p.pixel ? (float) H / cfg.inputH : H;

        List<Rect2d> rects = new ArrayList<>();
        List<Float> scs = new ArrayList<>();
        List<Integer> idxs = new ArrayList<>();
        for (int i = 0; i < p.n; i++) {
            float s = p.scores[i];
            if (s < cfg.scoreThreshold) continue;
            float a = p.boxes[i * 4], b = p.boxes[i * 4 + 1];
            float c = p.boxes[i * 4 + 2], d = p.boxes[i * 4 + 3];
            float x1, y1, x2, y2;
            if (p.center) {
                x1 = (a - c / 2) * sx; y1 = (b - d / 2) * sy;
                x2 = (a + c / 2) * sx; y2 = (b + d / 2) * sy;
            } else {
                x1 = a * sx; y1 = b * sy; x2 = c * sx; y2 = d * sy;
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
                    ol[k * 10 + q * 2]     = p.lands[src + q * 2]     * sx;
                    ol[k * 10 + q * 2 + 1] = p.lands[src + q * 2 + 1] * sy;
                }
            }
        }
        return new DetectionResult(ob, os, ol);
    }

    private Parsed parse() {
        for (Mat o : outputs) {
            if (o.dims() != 3 || o.size(0) != 1) continue;
            int d1 = o.size(1), d2 = o.size(2);

            if (d2 == 6) {
                float[] buf = new float[d1 * 6];
                o.get(new int[]{0, 0, 0}, buf);
                Parsed p = new Parsed();
                p.n = d1;
                p.pixel = true;
                p.boxes = new float[d1 * 4];
                p.scores = new float[d1];
                if (d1 <= 1000) {              // YOLO26/v10 e2e: xyxy,score,cls
                    p.center = false;
                    for (int i = 0; i < d1; i++) {
                        System.arraycopy(buf, i * 6, p.boxes, i * 4, 4);
                        p.scores[i] = buf[i * 6 + 4];
                    }
                } else {                        // v5-style: cxcywh,obj,cls
                    p.center = true;
                    for (int i = 0; i < d1; i++) {
                        System.arraycopy(buf, i * 6, p.boxes, i * 4, 4);
                        p.scores[i] = buf[i * 6 + 4] * buf[i * 6 + 5];
                    }
                }
                return p;
            }

            if (d1 >= 5 && d1 <= 200 && d2 > d1 * 8) {   // raw [1,C,N] channels-first
                float[] buf = new float[d1 * d2];
                o.get(new int[]{0, 0, 0}, buf);
                Parsed p = new Parsed();
                p.n = d2;
                p.pixel = true;
                p.center = true;
                p.boxes = new float[d2 * 4];
                p.scores = new float[d2];
                for (int j = 0; j < d2; j++) {
                    p.boxes[j * 4]     = buf[j];
                    p.boxes[j * 4 + 1] = buf[d2 + j];
                    p.boxes[j * 4 + 2] = buf[2 * d2 + j];
                    p.boxes[j * 4 + 3] = buf[3 * d2 + j];
                    float best = 0f;
                    for (int k = 4; k < d1; k++) {
                        float v = buf[k * d2 + j];
                        if (v > best) best = v;
                    }
                    p.scores[j] = best;
                }
                return p;
            }
        }

        // Tách kênh: score(1|2) + box(4) + landmark(10)
        final int scoreCh = "sigmoid1".equals(cfg.score) ? 1 : 2;
        Parsed p = new Parsed();
        p.center = "center".equals(cfg.box);
        p.pixel = !cfg.normalized;
        float[] rawScores = null;
        for (Mat o : outputs) {
            int dims = o.dims();
            int c = o.size(dims - 1);
            int rows = (int) (o.total() / c);
            if (c == scoreCh && rawScores == null) {
                rawScores = new float[rows * c];
                o.get(new int[dims], rawScores);
                p.n = rows;
            } else if (c == 4 && p.boxes == null) {
                p.boxes = new float[rows * 4];
                o.get(new int[dims], p.boxes);
            } else if (c == 10 && p.lands == null) {
                p.lands = new float[rows * 10];
                o.get(new int[dims], p.lands);
            }
        }
        if (rawScores == null || p.boxes == null || p.n == 0) return null;
        p.scores = new float[p.n];
        for (int i = 0; i < p.n; i++) {
            p.scores[i] = (scoreCh == 1) ? rawScores[i] : rawScores[i * 2 + 1];
        }
        return p;
    }

    private String shapes() {
        StringBuilder sb = new StringBuilder();
        for (Mat m : outputs) {
            sb.append('[');
            for (int i = 0; i < m.dims(); i++) {
                if (i > 0) sb.append('x');
                sb.append(m.size(i));
            }
            sb.append("] ");
        }
        return sb.toString();
    }

    private static float clamp(float v, float lo, float hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    @Override
    public void release() {
        net = null;
    }
}

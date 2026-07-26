package com.example.faceidentity.model;

import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect2d;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;

import com.example.faceidentity.utils.CrashLogger;
import com.example.faceidentity.utils.OnnxTopKPatcher;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * ONNX detector cấu hình bằng ModelConfig. Tự nhận diện dạng output:
 *  - YOLO e2e (YOLO26/v10):    [1, N<=1000, 6+k] = x1,y1,x2,y2,score,cls[,5kpt×(2|3)] (pixel)
 *  - YOLO v5-style:            [1, N>1000, 6]    = cx,cy,w,h,obj,cls (pixel)
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
    private int statCalls = 0;

    private Mat lbResized;
    private Mat lbCanvas;
    private float lbScale = 1f, lbPadX = 0f, lbPadY = 0f;

    private static class Parsed {
        float[] scores;
        float[] boxes;
        float[] lands;
        float[] raw;
        int stride;
        int n;
        boolean center;
        boolean pixel;
    }

    private final String label;

    public GenericOnnxDetector(String modelPath, ModelConfig cfg) {
        this.modelPath = modelPath;
        this.cfg = cfg;
        int cut = Math.max(modelPath.lastIndexOf('/'), modelPath.lastIndexOf('\\'));
        this.label = modelPath.substring(cut + 1).replaceAll("\\.[^.]+$", "");
    }

    @Override
    public String name() {
        return "ONNX";
    }

    @Override
    public void init() {
        try {
            net = Dnn.readNetFromONNX(modelPath);
        } catch (Exception e) {
            // YOLO e2e: OpenCV đòi TopK K < N, ONNX cho K <= N -> tự vá bản copy rồi load lại
            if (!OnnxTopKPatcher.isTopKError(e) || !OnnxTopKPatcher.patch(modelPath)) throw e;
            net = Dnn.readNetFromONNX(modelPath);
            Log.i(TAG, "Đã tự vá TopK và load lại thành công");
        }
        outNames = (cfg.outputs != null && cfg.outputs.length > 0)
                ? resolveLayers(cfg.outputs)
                : net.getUnconnectedOutLayersNames();
        Log.i(TAG, "init OK input=" + cfg.inputW + "x" + cfg.inputH
                + " letterbox=" + cfg.letterbox + " outputs=" + outNames);
    }

    @Override
    public DetectionResult detect(Mat bgr) {
        if (net == null) return DetectionResult.EMPTY;
        final int W = bgr.cols(), H = bgr.rows();

        Mat src = cfg.letterbox ? letterbox(bgr) : bgr;
        Mat blob = Dnn.blobFromImage(src, cfg.scale, new Size(cfg.inputW, cfg.inputH),
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
        if (cfg.scoreScale != 1f) {
            for (int i = 0; i < p.n; i++) p.scores[i] *= cfg.scoreScale;
        }
        if (statCalls < 3 || statCalls % 60 == 0) logStats(p);
        statCalls++;

        // x_img = (x_model - ox) * sx
        final float sx, sy, ox, oy;
        if (!p.pixel) {
            sx = W; sy = H; ox = 0f; oy = 0f;
        } else if (cfg.letterbox) {
            sx = 1f / lbScale; sy = 1f / lbScale; ox = lbPadX; oy = lbPadY;
        } else {
            sx = (float) W / cfg.inputW; sy = (float) H / cfg.inputH; ox = 0f; oy = 0f;
        }

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
                int srcI = idxs.get(kp[k]) * 10;
                for (int q = 0; q < 5; q++) {
                    ol[k * 10 + q * 2]     = (p.lands[srcI + q * 2]     - ox) * sx;
                    ol[k * 10 + q * 2 + 1] = (p.lands[srcI + q * 2 + 1] - oy) * sy;
                }
            }
        }
        return new DetectionResult(ob, os, ol);
    }

    /** Tên trong cfg.outputs là tên tensor ONNX; OpenCV 4.12 đặt tên layer "onnx_node!<node>". */
    private List<String> resolveLayers(String[] wanted) {
        List<String> all = net.getLayerNames();
        List<String> resolved = new ArrayList<>();
        for (String n : wanted) {
            String stripped = n.endsWith("_output_0") ? n.substring(0, n.length() - 9) : n;
            String hit = null;
            for (String c : new String[]{n, stripped, "onnx_node!" + n, "onnx_node!" + stripped}) {
                if (all.contains(c)) { hit = c; break; }
            }
            if (hit == null) {
                StringBuilder sb = new StringBuilder();
                for (String l : all) sb.append(l).append('\n');
                CrashLogger.logError(TAG, "Không thấy layer '" + n + "'. Toàn bộ layer:\n" + sb, null);
                throw new IllegalStateException("Config outputs sai tên layer: " + n);
            }
            resolved.add(hit);
        }
        Log.i(TAG, "outputs resolve -> " + resolved);
        return resolved;
    }

    /** Resize giữ tỉ lệ + pad 114 (chuẩn Ultralytics). */
    private Mat letterbox(Mat bgr) {
        final int W = bgr.cols(), H = bgr.rows();
        float r = Math.min((float) cfg.inputW / W, (float) cfg.inputH / H);
        int nw = Math.max(1, Math.round(W * r)), nh = Math.max(1, Math.round(H * r));
        lbScale = r;
        lbPadX = (cfg.inputW - nw) / 2f;
        lbPadY = (cfg.inputH - nh) / 2f;

        if (lbResized == null) lbResized = new Mat();
        Imgproc.resize(bgr, lbResized, new Size(nw, nh));
        if (lbCanvas == null || lbCanvas.cols() != cfg.inputW || lbCanvas.rows() != cfg.inputH) {
            if (lbCanvas != null) lbCanvas.release();
            lbCanvas = new Mat(cfg.inputH, cfg.inputW, CvType.CV_8UC3);
        }
        lbCanvas.setTo(new Scalar(114, 114, 114));
        int top = (int) lbPadY, left = (int) lbPadX;
        Mat roi = lbCanvas.submat(top, top + nh, left, left + nw);
        lbResized.copyTo(roi);
        roi.release();
        return lbCanvas;
    }

    private void logStats(Parsed p) {
        float mx = 0f, sum = 0f;
        int idx = 0;
        for (int i = 0; i < p.n; i++) {
            sum += p.scores[i];
            if (p.scores[i] > mx) { mx = p.scores[i]; idx = i; }
        }
        StringBuilder row = new StringBuilder();
        if (p.raw != null) {
            int base = idx * p.stride;
            for (int c = 0; c < p.stride && c < 24; c++) {
                row.append(String.format(Locale.US, "%.3f ", p.raw[base + c]));
            }
        } else {
            row.append(String.format(Locale.US, "box=[%.1f %.1f %.1f %.1f]",
                    p.boxes[idx * 4], p.boxes[idx * 4 + 1],
                    p.boxes[idx * 4 + 2], p.boxes[idx * 4 + 3]));
        }
        Log.i(TAG, String.format(Locale.US,
                "[%s] STAT n=%d scoreMax=%.4f mean=%.4f thr=%.2f | best row[%d]: %s",
                label, p.n, mx, sum / p.n, cfg.scoreThreshold, idx, row.toString().trim()));
    }

    private Parsed parse() {
        for (Mat o : outputs) {
            if (o.dims() != 3 || o.size(0) != 1) continue;
            int d1 = o.size(1), d2 = o.size(2);

            if (d1 <= 1000 && d2 >= 6 && d2 <= 60) {   // e2e: xyxy,score,cls[,5kpt×(2|3)]
                float[] buf = new float[d1 * d2];
                o.get(new int[]{0, 0, 0}, buf);
                Parsed p = new Parsed();
                p.n = d1;
                p.pixel = true;
                p.center = false;
                p.raw = buf;
                p.stride = d2;
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
                float[] buf = new float[d1 * 6];
                o.get(new int[]{0, 0, 0}, buf);
                Parsed p = new Parsed();
                p.n = d1;
                p.pixel = true;
                p.center = true;
                p.raw = buf;
                p.stride = 6;
                p.boxes = new float[d1 * 4];
                p.scores = new float[d1];
                for (int i = 0; i < d1; i++) {
                    System.arraycopy(buf, i * 6, p.boxes, i * 4, 4);
                    p.scores[i] = buf[i * 6 + 4] * buf[i * 6 + 5];
                }
                return p;
            }

            if (d1 >= 5 && d1 <= 200 && d2 > d1 * 8) {   // raw [1,C,N] channels-first
                float[] buf = new float[d1 * d2];
                o.get(new int[]{0, 0, 0}, buf);
                Parsed p = new Parsed();
                p.n = d2;
                p.pixel = true;
                p.center = "center".equals(cfg.box);   // raw: format box theo config
                p.boxes = new float[d2 * 4];
                p.scores = new float[d2];
                // C = 4 box + 1 score + 5kpt×kd (pose 1 class), hoặc 4 + nc (multi-class)
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
        if (lbResized != null) { lbResized.release(); lbResized = null; }
        if (lbCanvas != null) { lbCanvas.release(); lbCanvas = null; }
    }
}

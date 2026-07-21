package com.example.faceidentity.model;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.Locale;

public class ModelConfig {

    public String backend = "onnx";        // "onnx" | "yunet"
    public int inputW = 640, inputH = 640;
    public double[] mean = {127, 127, 127}; // thứ tự kênh như truyền cho blobFromImage
    public double scale = 1.0 / 128.0;
    public boolean swapRB = true;           // model RGB -> true (OpenCV đọc BGR)
    public String box = "corner";           // "corner" (x1,y1,x2,y2) | "center" (cx,cy,w,h)
    public boolean normalized = true;       // box/landmark 0..1 -> true; pixel theo input -> false
    public String score = "softmax2";       // "softmax2" (lấy kênh 1) | "sigmoid1"
    public float scoreThreshold = 0.7f;
    public float nmsThreshold = 0.3f;

    public static ModelConfig fromJson(String json) {
        ModelConfig c = new ModelConfig();
        try {
            JSONObject o = new JSONObject(json);
            c.backend = o.optString("backend", c.backend);
            if (o.has("input")) {
                JSONArray a = o.getJSONArray("input");
                c.inputW = a.getInt(0);
                c.inputH = a.getInt(1);
            }
            if (o.has("mean")) {
                JSONArray a = o.getJSONArray("mean");
                c.mean = new double[]{a.getDouble(0), a.getDouble(1), a.getDouble(2)};
            }
            c.scale = o.optDouble("scale", c.scale);
            c.swapRB = o.optBoolean("swapRB", c.swapRB);
            c.box = o.optString("box", c.box);
            c.normalized = o.optBoolean("normalized", c.normalized);
            c.score = o.optString("score", c.score);
            c.scoreThreshold = (float) o.optDouble("scoreThreshold", c.scoreThreshold);
            c.nmsThreshold = (float) o.optDouble("nmsThreshold", c.nmsThreshold);
        } catch (Exception e) {
            throw new IllegalArgumentException("Config JSON lỗi: " + e.getMessage(), e);
        }
        return c;
    }

    /** Không có JSON -> đoán backend theo tên file. */
    public static ModelConfig defaultFor(String fileName) {
        ModelConfig c = new ModelConfig();
        if (fileName.toLowerCase(Locale.US).contains("yunet")) c.backend = "yunet";
        return c;
    }
}

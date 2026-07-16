package com.example.faceidentity.model;

/**
 * [MODEL] Kết quả detect - mảng thuần để bàn giao an toàn giữa các thread.
 */
public class DetectionResult {

    public static final DetectionResult EMPTY =
            new DetectionResult(new float[0], new float[0], null);

    /** Mỗi mặt 4 giá trị: x, y, w, h (pixel, hệ ảnh đầu vào). */
    public final float[] boxes;

    /** Điểm tin cậy từng mặt (0..1). */
    public final float[] scores;

    /** 5 landmark/mặt: x1,y1,...,x5,y5 (pixel). Có thể null nếu model không có. */
    public final float[] landmarks;

    public DetectionResult(float[] boxes, float[] scores, float[] landmarks) {
        this.boxes = boxes;
        this.scores = scores;
        this.landmarks = landmarks;
    }

    public int count() {
        return boxes.length / 4;
    }
}

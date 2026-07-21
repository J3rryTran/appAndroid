package com.example.faceidentity.model;
public class DetectionResult {

    public static final DetectionResult EMPTY =
            new DetectionResult(new float[0], new float[0], null);
    public final float[] boxes;

    public final float[] scores;

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

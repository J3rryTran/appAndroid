package com.example.faceidentity.view;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.view.View;

import java.util.Locale;

public class CameraPreview extends View {
    private static final int[] LM_COLORS = {
            Color.BLUE, Color.RED, Color.GREEN, Color.MAGENTA, Color.YELLOW
    };
    private static final float LM_RADIUS = 7f;

    private final Paint boxPaint;
    private final Paint pointPaint;
    private final Paint textPaint;

    private float[] boxes = new float[0];
    private float[] landmarks = null;   // 10 float/mặt, có thể null
    private float[] scores = null;      // confidence 0..1/mặt, có thể null
    private int frameW = 0;
    private int frameH = 0;
    private boolean mirror = false;

    public CameraPreview(Context context, AttributeSet attrs) {
        super(context, attrs);
        boxPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        boxPaint.setColor(Color.GREEN);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(6f);

        pointPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        pointPaint.setStyle(Paint.Style.FILL);

        textPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        textPaint.setColor(Color.GREEN);
        textPaint.setTextSize(34f);
        textPaint.setFakeBoldText(true);
        textPaint.setShadowLayer(4f, 0f, 0f, Color.BLACK);
    }

    public void setMirror(boolean mirror) {
        if (this.mirror == mirror) return;
        this.mirror = mirror;
        invalidate();
    }

    public void setResults(float[] boxes, float[] landmarks, float[] scores,
                           int frameW, int frameH) {
        this.boxes = boxes;
        this.landmarks = landmarks;
        this.scores = scores;
        this.frameW = frameW;
        this.frameH = frameH;
        invalidate();
    }

    public void clear() {
        this.boxes = new float[0];
        this.landmarks = null;
        this.scores = null;
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if (frameW == 0 || frameH == 0 || boxes.length == 0) return;

        final float viewW = getWidth();
        final float viewH = getHeight();

        final float scale = Math.min(viewW / frameW, viewH / frameH);
        final float dx = (viewW - frameW * scale) / 2f;
        final float dy = (viewH - frameH * scale) / 2f;

        final int n = boxes.length / 4;
        final boolean hasLm = landmarks != null && landmarks.length >= n * 10;
        final boolean hasSc = scores != null && scores.length >= n;

        for (int i = 0; i < n; i++) {
            float x = boxes[i * 4];
            float y = boxes[i * 4 + 1];
            float w = boxes[i * 4 + 2];
            float h = boxes[i * 4 + 3];

            float left   = x * scale + dx;
            float top    = y * scale + dy;
            float right  = (x + w) * scale + dx;
            float bottom = (y + h) * scale + dy;

            if (mirror) {
                float mLeft  = viewW - right;
                float mRight = viewW - left;
                left = mLeft;
                right = mRight;
            }

            canvas.drawRect(left, top, right, bottom, boxPaint);

            if (hasSc) {
                float ty = top - 12f;
                if (ty < 36f) ty = bottom + 40f;
                canvas.drawText(String.format(Locale.US, "%.2f", scores[i]),
                        left + 4f, ty, textPaint);
            }

            if (hasLm) {
                for (int p = 0; p < 5; p++) {
                    float px = landmarks[i * 10 + p * 2]     * scale + dx;
                    float py = landmarks[i * 10 + p * 2 + 1] * scale + dy;
                    if (mirror) {
                        px = viewW - px;
                    }
                    pointPaint.setColor(LM_COLORS[p]);
                    canvas.drawCircle(px, py, LM_RADIUS, pointPaint);
                }
            }
        }
    }
}

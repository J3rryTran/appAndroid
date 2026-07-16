package com.example.faceidentity.view;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.view.View;

public class CameraPreview extends View {

    private final Paint boxPaint;
    private float[] boxes = new float[0];
    private int frameW = 0;
    private int frameH = 0;
    private boolean mirror = false;

    public CameraPreview(Context context, AttributeSet attrs) {
        super(context, attrs);
        boxPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        boxPaint.setColor(Color.GREEN);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(6f);
    }

    public void setMirror(boolean mirror) {
        if (this.mirror == mirror) return;
        this.mirror = mirror;
        invalidate();
    }

    public void setResults(float[] boxes, int frameW, int frameH) {
        this.boxes = boxes;
        this.frameW = frameW;
        this.frameH = frameH;
        invalidate();
    }

    public void clear() {
        this.boxes = new float[0];
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
        }
    }
}

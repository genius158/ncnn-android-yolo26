package com.example.yolo26ncnn;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.view.View;

import java.util.ArrayList;
import java.util.List;

public class OverlayView extends View {

    private final List<DetectionResult> results = new ArrayList<>();
    private final Paint boxPaint = new Paint();
    private final Paint textPaint = new Paint();
    private final Paint bgPaint = new Paint();

    private int imageWidth = 0;
    private int imageHeight = 0;

    private static final int[] COLORS = {
            Color.rgb(255, 89, 94),   // Red
            Color.rgb(255, 202, 58),  // Yellow
            Color.rgb(138, 201, 38),  // Green
            Color.rgb(25, 130, 196),  // Blue
            Color.rgb(106, 76, 147),  // Purple
            Color.rgb(255, 146, 76),  // Orange
            Color.rgb(0, 187, 212),   // Cyan
            Color.rgb(233, 30, 99),   // Pink
    };

    public OverlayView(Context context) {
        super(context);
        init();
    }

    public OverlayView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    public OverlayView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        init();
    }

    private void init() {
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(6);
        boxPaint.setAntiAlias(true);

        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(36);
        textPaint.setAntiAlias(true);
        textPaint.setFakeBoldText(true);

        bgPaint.setStyle(Paint.Style.FILL);
        bgPaint.setAntiAlias(true);
    }

    public void setPreviewSize(int width, int height) {
        this.imageWidth = width;
        this.imageHeight = height;
    }


    public void setResults(Yolo26Ncnn.Obj[] objects) {
        results.clear();
        if (objects != null) {
            for (Yolo26Ncnn.Obj obj : objects) {
                results.add(new DetectionResult(
                        obj.x, obj.y, obj.w, obj.h,
                        obj.label != null ? obj.label : "unknown",
                        obj.prob
                ));
            }
        }
        postInvalidate();
    }

    public void clearResults() {
        results.clear();
        postInvalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        if (imageWidth == 0 || imageHeight == 0) return;

        int viewWidth = getWidth();
        int viewHeight = getHeight();

        // Calculate scale to fit image in view while maintaining aspect ratio (center crop style)
        float imageAspect = (float) imageWidth / imageHeight;
        float viewAspect = (float) viewWidth / viewHeight;

        float scale;
        float offsetX = 0;
        float offsetY = 0;

        if (imageAspect > viewAspect) {
            // Image is wider - fit height, crop width
            scale = (float) viewHeight / imageHeight;
            offsetX = (viewWidth - imageWidth * scale) / 2f;
        } else {
            // Image is taller - fit width, crop height
            scale = (float) viewWidth / imageWidth;
            offsetY = (viewHeight - imageHeight * scale) / 2f;
        }

        int colorIndex = 0;
        for (DetectionResult result : results) {
            int color = COLORS[colorIndex % COLORS.length];
            boxPaint.setColor(color);
            bgPaint.setColor(color);
            colorIndex++;

            // Transform coordinates from image space to view space
            float left = result.x * scale + offsetX;
            float top = result.y * scale + offsetY;
            float right = (result.x + result.w) * scale + offsetX;
            float bottom = (result.y + result.h) * scale + offsetY;

            // Clamp to view bounds
            left = Math.max(0, Math.min(left, viewWidth));
            top = Math.max(0, Math.min(top, viewHeight));
            right = Math.max(0, Math.min(right, viewWidth));
            bottom = Math.max(0, Math.min(bottom, viewHeight));

            // Draw detection box
            canvas.drawRect(left, top, right, bottom, boxPaint);

            // Draw label
            String label = String.format("%s %.0f%%", result.label, result.prob * 100);
            float textWidth = textPaint.measureText(label);
            float textHeight = 40;
            float padding = 8;

            // Label background position (above box if possible)
            float labelTop = top - textHeight - padding;
            if (labelTop < 0) {
                labelTop = top + padding;
            }

            // Draw label background
            bgPaint.setAlpha(200);
            canvas.drawRect(left, labelTop, left + textWidth + padding * 2, labelTop + textHeight + padding, bgPaint);

            // Draw label text
            canvas.drawText(label, left + padding, labelTop + textHeight, textPaint);
        }
    }

    private static class DetectionResult {
        float x, y, w, h;
        String label;
        float prob;

        DetectionResult(float x, float y, float w, float h, String label, float prob) {
            this.x = x;
            this.y = y;
            this.w = w;
            this.h = h;
            this.label = label;
            this.prob = prob;
        }
    }
}

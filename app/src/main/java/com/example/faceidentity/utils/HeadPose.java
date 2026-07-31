package com.example.faceidentity.utils;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point;
import org.opencv.core.Point3;

public final class HeadPose {

    private final MatOfPoint3f model = new MatOfPoint3f(
            new Point3(-35, -35, 30),
            new Point3(35, -35, 30),
            new Point3(0, 0, 0),
            new Point3(-28, 28, 25),
            new Point3(28, 28, 25));

    private final MatOfPoint2f imagePts = new MatOfPoint2f();
    private final Point[] pts = new Point[]{
            new Point(), new Point(), new Point(), new Point(), new Point()};
    private final Mat cam = Mat.zeros(3, 3, CvType.CV_64F);
    private final MatOfDouble dist = new MatOfDouble(0, 0, 0, 0);
    private final Mat rvec = new Mat();
    private final Mat tvec = new Mat();
    private final Mat rot = new Mat();
    private final double[] r = new double[9];
    private int camW = -1, camH = -1;

    public boolean estimate(float[] lm, int faceIdx, int frameW, int frameH,
                            float[] out, int outIdx) {
        int b = faceIdx * 10;
        if (lm == null || lm.length < b + 10) return false;

        float e0x = lm[b], e0y = lm[b + 1];
        float e1x = lm[b + 2], e1y = lm[b + 3];
        float m0x = lm[b + 6], m0y = lm[b + 7];
        float m1x = lm[b + 8], m1y = lm[b + 9];
        if (e0x > e1x) {
            float t;
            t = e0x; e0x = e1x; e1x = t;
            t = e0y; e0y = e1y; e1y = t;
        }
        if (m0x > m1x) {
            float t;
            t = m0x; m0x = m1x; m1x = t;
            t = m0y; m0y = m1y; m1y = t;
        }
        pts[0].x = e0x; pts[0].y = e0y;
        pts[1].x = e1x; pts[1].y = e1y;
        pts[2].x = lm[b + 4]; pts[2].y = lm[b + 5];
        pts[3].x = m0x; pts[3].y = m0y;
        pts[4].x = m1x; pts[4].y = m1y;
        imagePts.fromArray(pts);

        if (camW != frameW || camH != frameH) {
            camW = frameW;
            camH = frameH;
            double f = frameW;
            cam.put(0, 0, f, 0, frameW / 2.0, 0, f, frameH / 2.0, 0, 0, 1);
        }

        try {
            if (!Calib3d.solvePnP(model, imagePts, cam, dist, rvec, tvec,
                    false, Calib3d.SOLVEPNP_ITERATIVE)) {
                return false;
            }
            Calib3d.Rodrigues(rvec, rot);
            rot.get(0, 0, r);
        } catch (Exception e) {
            return false;
        }

        double sy = Math.sqrt(r[0] * r[0] + r[3] * r[3]);
        double pitch, yaw, roll;
        if (sy > 1e-6) {
            pitch = Math.atan2(r[7], r[8]);
            yaw = Math.atan2(-r[6], sy);
            roll = Math.atan2(r[3], r[0]);
        } else {
            pitch = Math.atan2(-r[5], r[4]);
            yaw = Math.atan2(-r[6], sy);
            roll = 0;
        }
        out[outIdx] = wrap((float) Math.toDegrees(yaw));
        out[outIdx + 1] = wrap((float) Math.toDegrees(pitch));
        out[outIdx + 2] = wrap((float) Math.toDegrees(roll));
        return true;
    }

    private static float wrap(float deg) {
        if (deg > 90f) return deg - 180f;
        if (deg < -90f) return deg + 180f;
        return deg;
    }

    public void release() {
        model.release();
        imagePts.release();
        cam.release();
        dist.release();
        rvec.release();
        tvec.release();
        rot.release();
    }
}

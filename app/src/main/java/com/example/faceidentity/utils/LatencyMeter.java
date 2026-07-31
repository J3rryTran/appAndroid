package com.example.faceidentity.utils;

import java.util.Locale;

public final class LatencyMeter {

    private final String[] names;
    private final double[] sums;
    private final double[] peaks;
    private int n;

    public LatencyMeter(String... names) {
        this.names = names;
        this.sums = new double[names.length];
        this.peaks = new double[names.length];
    }

    public synchronized void add(double... values) {
        if (values.length != sums.length) return;
        for (int i = 0; i < values.length; i++) {
            sums[i] += values[i];
            if (values[i] > peaks[i]) peaks[i] = values[i];
        }
        n++;
    }

    public synchronized String snapshotAndReset() {
        if (n == 0) return "";
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < names.length; i++) {
            if (i > 0) sb.append(' ');
            sb.append(String.format(Locale.US, "%s=%.1f", names[i], sums[i] / n));
            sums[i] = 0;
            peaks[i] = 0;
        }
        n = 0;
        return sb.toString();
    }

    public synchronized String snapshotWithPeakAndReset() {
        if (n == 0) return "";
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < names.length; i++) {
            if (i > 0) sb.append(' ');
            sb.append(String.format(Locale.US, "%s=%.1f/%.1f", names[i], sums[i] / n, peaks[i]));
            sums[i] = 0;
            peaks[i] = 0;
        }
        n = 0;
        return sb.toString();
    }
}

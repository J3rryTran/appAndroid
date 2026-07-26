package com.example.faceidentity.utils;

import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Vá lỗi OpenCV DNN "TopK: K is out of range" cho ONNX YOLO e2e (YOLO26/v10).
 * OpenCV đòi K < N, ONNX cho phép K <= N. Với 1 class, TopK cuối luôn có K == N
 * và hai node TopK dùng chung một hằng K -> phải nhân bản hằng số rồi trỏ riêng.
 * Thao tác trên bản copy trong internal storage, assets giữ nguyên.
 */
public final class OnnxTopKPatcher {

    private static final String TAG = "OnnxTopKPatcher";

    private OnnxTopKPatcher() { }

    public static boolean isTopKError(Throwable t) {
        String m = t.getMessage();
        return m != null && m.contains("TopK") && m.contains("out of range");
    }

    /** @return true nếu file đã được vá (gọi lại readNetFromONNX sau đó). */
    public static boolean patch(String path) {
        try {
            File f = new File(path);
            byte[] d = readAll(f);

            int gs = -1, ge = -1;
            Cursor c = new Cursor(d, 0, d.length);
            while (c.next()) {
                if (c.fn == 7 && c.wire == 2) { gs = c.ps; ge = c.pe; }
            }
            if (gs < 0) return false;

            List<int[]> kSpans = new ArrayList<>();
            List<String> kNames = new ArrayList<>();
            Map<String, int[]> entries = new HashMap<>();   // name -> {payloadStart, payloadEnd, valueOff}

            Cursor g = new Cursor(d, gs, ge);
            while (g.next()) {
                if (g.fn == 1 && g.wire == 2) {
                    String op = null;
                    List<int[]> ins = new ArrayList<>();
                    Cursor n = new Cursor(d, g.ps, g.pe);
                    while (n.next()) {
                        if (n.fn == 1 && n.wire == 2) ins.add(new int[]{n.ps, n.pe});
                        else if (n.fn == 4 && n.wire == 2) op = str(d, n.ps, n.pe);
                    }
                    if ("TopK".equals(op) && ins.size() >= 2) {
                        int[] sp = ins.get(1);
                        kSpans.add(sp);
                        kNames.add(str(d, sp[0], sp[1]));
                    }
                } else if (g.fn == 5 && g.wire == 2) {
                    String nm = null; long dt = -1; int vo = -1;
                    Cursor t = new Cursor(d, g.ps, g.pe);
                    while (t.next()) {
                        if (t.fn == 8 && t.wire == 2) nm = str(d, t.ps, t.pe);
                        else if (t.fn == 2 && t.wire == 0) dt = t.v;
                        else if (t.fn == 9 && t.wire == 2) vo = t.ps;
                    }
                    if (nm != null && dt == 7 && vo >= 0) {
                        entries.put(nm, new int[]{g.ps, g.pe, vo});
                    }
                }
            }
            if (kSpans.isEmpty()) return false;

            int last = kSpans.size() - 1;
            String lastName = kNames.get(last);
            int[] entry = entries.get(lastName);
            if (entry == null) return false;
            long k = readLE64(d, entry[2]);
            if (k <= 1) return false;

            boolean shared = kNames.indexOf(lastName) != last;
            if (!shared) {
                writeLE64(d, entry[2], k - 1);
                writeAll(f, d);
                Log.i(TAG, "Đã vá K " + k + " -> " + (k - 1) + " cho TopK cuối");
                return true;
            }

            // K dùng chung -> nhân bản initializer (tên cùng độ dài) rồi trỏ TopK cuối sang
            char lc = lastName.charAt(lastName.length() - 1);
            char nc = (lc == '9') ? '8' : '9';
            String newName = lastName.substring(0, lastName.length() - 1) + nc;
            if (entries.containsKey(newName)) return false;

            byte[] payload = new byte[entry[1] - entry[0]];
            System.arraycopy(d, entry[0], payload, 0, payload.length);
            boolean okName = false, okVal = false;
            Cursor t = new Cursor(payload, 0, payload.length);
            while (t.next()) {
                if (t.fn == 8 && t.wire == 2 && str(payload, t.ps, t.pe).equals(lastName)) {
                    payload[t.pe - 1] = (byte) nc;
                    okName = true;
                } else if (t.fn == 9 && t.wire == 2) {
                    writeLE64(payload, t.ps, k - 1);
                    okVal = true;
                }
            }
            if (!okName || !okVal) return false;

            int[] sp = kSpans.get(last);
            d[sp[1] - 1] = (byte) nc;                       // đổi tên K-input của TopK cuối

            byte[] lenNew = varint(payload.length);
            byte[] appended = new byte[1 + lenNew.length + payload.length];
            appended[0] = 0x2a;                              // field 5 (initializer), wire 2
            System.arraycopy(lenNew, 0, appended, 1, lenNew.length);
            System.arraycopy(payload, 0, appended, 1 + lenNew.length, payload.length);

            int oldLen = ge - gs;
            int hdr = gs - varint(oldLen).length - 1;
            if (hdr < 0 || (d[hdr] & 0xFF) != 0x3a) return false;

            byte[] newLen = varint(oldLen + appended.length);
            java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream(
                    d.length + appended.length + 4);
            out.write(d, 0, hdr);
            out.write(0x3a);
            out.write(newLen, 0, newLen.length);
            out.write(d, gs, oldLen);
            out.write(appended, 0, appended.length);
            out.write(d, ge, d.length - ge);
            writeAll(f, out.toByteArray());

            Log.i(TAG, "Đã tách hằng K: TopK cuối dùng " + newName + " = " + (k - 1));
            return true;
        } catch (Exception e) {
            Log.e(TAG, "Vá TopK thất bại", e);
            return false;
        }
    }

    private static final class Cursor {
        final byte[] b; final int end;
        int i, fn, wire, ps, pe;
        long v;

        Cursor(byte[] b, int start, int end) { this.b = b; this.i = start; this.end = end; }

        boolean next() {
            if (i >= end) return false;
            long tag = varint();
            fn = (int) (tag >> 3);
            wire = (int) (tag & 7);
            if (wire == 0) { ps = i; v = varint(); pe = i; }
            else if (wire == 2) { int len = (int) varint(); ps = i; pe = i + len; i = pe; }
            else if (wire == 1) { ps = i; pe = i + 8; i = pe; }
            else if (wire == 5) { ps = i; pe = i + 4; i = pe; }
            else throw new IllegalStateException("wire " + wire);
            return true;
        }

        private long varint() {
            long r = 0; int s = 0;
            while (true) {
                int x = b[i++] & 0xFF;
                r |= (long) (x & 0x7f) << s;
                if ((x & 0x80) == 0) return r;
                s += 7;
            }
        }
    }

    private static byte[] varint(int v) {
        java.io.ByteArrayOutputStream o = new java.io.ByteArrayOutputStream();
        int x = v;
        while (true) {
            int b = x & 0x7f;
            x >>>= 7;
            if (x != 0) o.write(b | 0x80);
            else { o.write(b); return o.toByteArray(); }
        }
    }

    private static String str(byte[] b, int s, int e) {
        return new String(b, s, e - s, java.nio.charset.StandardCharsets.UTF_8);
    }

    private static long readLE64(byte[] b, int o) {
        long r = 0;
        for (int i = 7; i >= 0; i--) r = (r << 8) | (b[o + i] & 0xFFL);
        return r;
    }

    private static void writeLE64(byte[] b, int o, long v) {
        for (int i = 0; i < 8; i++) b[o + i] = (byte) (v >>> (8 * i));
    }

    private static byte[] readAll(File f) throws IOException {
        try (RandomAccessFile r = new RandomAccessFile(f, "r")) {
            byte[] d = new byte[(int) r.length()];
            r.readFully(d);
            return d;
        }
    }

    private static void writeAll(File f, byte[] d) throws IOException {
        try (FileOutputStream o = new FileOutputStream(f)) {
            o.write(d);
        }
    }
}

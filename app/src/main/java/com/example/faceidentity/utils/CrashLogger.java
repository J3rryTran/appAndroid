package com.example.faceidentity.utils;

import android.content.Context;
import android.os.Build;
import android.util.Log;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
public final class CrashLogger {

    private static final String TAG = "CrashLogger";
    private static volatile File logDir;

    private CrashLogger() { }

    public static void install(Context ctx) {
        File base = ctx.getExternalFilesDir(null);
        if (base == null) base = ctx.getFilesDir();
        logDir = new File(base, "logs");
        if (!logDir.exists() && !logDir.mkdirs()) {
            Log.w(TAG, "Không tạo được thư mục log: " + logDir);
        }

        final Thread.UncaughtExceptionHandler systemHandler =
                Thread.getDefaultUncaughtExceptionHandler();
        Thread.setDefaultUncaughtExceptionHandler((thread, e) -> {
            // Ghi file trước, vì sau dòng dưới app sẽ bị hệ thống kill.
            writeFile("crash", "APP CRASH - uncaught exception trên thread ["
                    + thread.getName() + "]", e);
            if (systemHandler != null) {
                systemHandler.uncaughtException(thread, e);
            }
        });
        Log.i(TAG, "Log dir: " + logDir.getAbsolutePath());
    }
    public static void logError(String tag, String msg, Throwable t) {
        Log.e(tag, msg, t);
        writeFile("error", "[" + tag + "] " + msg, t);
    }
    public static String logDirPath() {
        File d = logDir;
        return d != null ? d.getAbsolutePath() : "(chưa install)";
    }

    private static synchronized void writeFile(String prefix, String header, Throwable t) {
        File dir = logDir;
        if (dir == null) return;
        Date now = new Date();
        String time = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US).format(now);
        String day  = new SimpleDateFormat("yyyyMMdd", Locale.US).format(now);
        File f = new File(dir, prefix + "-" + day + ".log");
        try (PrintWriter pw = new PrintWriter(new FileWriter(f, true))) {
            pw.println("==== " + time + " ====");
            pw.println(header);
            pw.println("Device: " + Build.MANUFACTURER + " " + Build.MODEL
                    + " | Android " + Build.VERSION.RELEASE
                    + " (API " + Build.VERSION.SDK_INT + ")");
            if (t != null) {
                t.printStackTrace(pw);
            }
            pw.println();
        } catch (IOException io) {
            Log.e(TAG, "Cannot read and write file log", io);
        }
    }
}

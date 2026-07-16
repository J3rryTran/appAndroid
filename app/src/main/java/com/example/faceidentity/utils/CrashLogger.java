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

/**
 * [UTILS] Ghi log lỗi ra FILE để xem lại được kể cả khi app đã VĂNG.
 *
 * - install(): cài Thread.UncaughtExceptionHandler toàn cục. Mọi exception không được
 *   catch (trên bất kỳ thread nào) sẽ được ghi ra file crash-<ngày>.log TRƯỚC KHI app
 *   chết, rồi chuyển tiếp cho handler mặc định của hệ thống (vẫn hiện hộp thoại crash
 *   + vẫn có trong logcat).
 * - logError(): dùng cho lỗi runtime đã catch được -> vừa ra Logcat vừa ghi nối vào
 *   file error-<ngày>.log.
 *
 * VỊ TRÍ FILE LOG (không cần quyền gì):
 *   /sdcard/Android/data/com.example.faceidentity/files/logs/
 *   -> mở bằng app quản lý file trên máy, hoặc kéo về PC:
 *      adb pull /sdcard/Android/data/com.example.faceidentity/files/logs
 *   (máy không có bộ nhớ ngoài thì fallback vào internal:
 *      adb shell run-as com.example.faceidentity ls files/logs)
 */
public final class CrashLogger {

    private static final String TAG = "CrashLogger";
    private static volatile File logDir;

    private CrashLogger() { }

    /** Gọi 1 lần trong Application.onCreate(). */
    public static void install(Context ctx) {
        File base = ctx.getExternalFilesDir(null);   // app-specific, không cần permission
        if (base == null) base = ctx.getFilesDir();  // fallback internal
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

    /** Lỗi runtime đã catch: ra Logcat + ghi nối file error-<ngày>.log. */
    public static void logError(String tag, String msg, Throwable t) {
        Log.e(tag, msg, t);
        writeFile("error", "[" + tag + "] " + msg, t);
    }

    /** Đường dẫn thư mục log (để in ra cho người dùng biết chỗ lấy). */
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
            Log.e(TAG, "Không ghi được file log", io);
        }
    }
}

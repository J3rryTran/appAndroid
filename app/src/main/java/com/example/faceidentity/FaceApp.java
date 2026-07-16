package com.example.faceidentity;

import android.app.Application;

import com.example.faceidentity.utils.CrashLogger;

/**
 * Application: cài CrashLogger SỚM NHẤT có thể (trước mọi Activity) để mọi crash
 * trên mọi thread đều được ghi ra file logs/crash-<ngày>.log.
 */
public class FaceApp extends Application {

    @Override
    public void onCreate() {
        super.onCreate();
        CrashLogger.install(this);
    }
}

package com.example.faceidentity;

import android.app.Application;

import com.example.faceidentity.utils.CrashLogger;

public class FaceApp extends Application {

    @Override
    public void onCreate() {
        super.onCreate();
        CrashLogger.install(this);
    }
}

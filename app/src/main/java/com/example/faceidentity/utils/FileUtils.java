package com.example.faceidentity.utils;

import android.content.Context;
import android.content.res.AssetManager;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public final class FileUtils {

    private FileUtils() { }
    public static String copyAssetToInternal(Context ctx, String assetPath) throws IOException {
        String fileName = new File(assetPath).getName();
        File outFile = new File(ctx.getFilesDir(), fileName);
        if (outFile.exists() && outFile.length() > 0) {
            return outFile.getAbsolutePath();
        }

        AssetManager am = ctx.getAssets();
        try (InputStream is = am.open(assetPath);
             OutputStream os = new FileOutputStream(outFile)) {
            byte[] buffer = new byte[8192];
            int len;
            while ((len = is.read(buffer)) != -1) {
                os.write(buffer, 0, len);
            }
            os.flush();
        }
        return outFile.getAbsolutePath();
    }
}

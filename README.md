# Face Identity Detection (Android · Java · OpenCV YuNet)

Ứng dụng Android nhận diện khuôn mặt **realtime** bằng **CameraX** + **OpenCV YuNet**
(`FaceDetectorYN`), kiến trúc **MVC**. Hiển thị **bounding box**, **FPS**, **số khuôn mặt**.

- Ngôn ngữ: Java · Min SDK: 24 · Kiến trúc: MVC
- Camera: CameraX (ImageAnalysis, RGBA_8888) — mặc định **cam trước**, có nút đổi
- Detector: **chỉ** YuNet (không Haar, không DNN cũ)
- Xoay kiểu **app camera hệ thống**: cửa sổ khoá dọc, chỉ icon xoay mượt theo cảm biến
- Ràng buộc: **phải nghiêng máy sang NGANG** thì nút START mới bật

---

## 0. Cấu trúc thư mục

```
appAndroid/
├── settings.gradle · build.gradle · gradle.properties
├── gradle/wrapper/gradle-wrapper.properties
├── OpenCV-android-sdk/                ← SDK 4.12 có sẵn, module ':opencv' đã wire
└── app/
    ├── build.gradle · proguard-rules.pro
    └── src/main/
        ├── AndroidManifest.xml
        ├── assets/models/face_detection_yunet_2023mar.onnx   ← đã có sẵn
        ├── java/com/example/faceidentity/
        │   ├── model/FaceDetectorModel.java
        │   ├── controller/CameraController.java
        │   ├── controller/FaceDetectionController.java
        │   ├── view/MainActivity.java
        │   ├── view/CameraPreview.java
        │   └── utils/{FileUtils,ImageUtils,PermissionUtils}.java
        └── res/{layout,values,drawable}/...
```

---

## 1. Model YuNet (assets) — ĐÃ có sẵn ✅

Model **không cần tải từ internet**: OpenCV Android SDK 4.12 trong project đã kèm sẵn nó
ở phần samples. File đã được chép vào đúng chỗ:

```
Nguồn:  OpenCV-android-sdk/samples/face-detection/res/raw/face_detection_yunet_2023mar.onnx
Đích:   app/src/main/assets/models/face_detection_yunet_2023mar.onnx   (232.589 bytes)
```

Tên file khớp `MODEL_ASSET` trong `MainActivity.java`. Nếu cần chép lại:
```powershell
Copy-Item "OpenCV-android-sdk\samples\face-detection\res\raw\face_detection_yunet_2023mar.onnx" `
          "app\src\main\assets\models\" -Force
```
> Assets đóng gói **lúc build** → sau khi thêm/thay model phải **Rebuild + cài lại** app.

---

## 2. OpenCV (ĐÃ tích hợp sẵn)

> OpenCV Android SDK **4.12.0** đã nằm trong `OpenCV-android-sdk/` và đã được nối vào
> project. App dùng `org.opencv.objdetect.FaceDetectorYN` — 4.12 hỗ trợ đầy đủ.

Những gì ĐÃ cấu hình sẵn (không cần import thủ công):
- `settings.gradle`: khai báo module `:opencv` → `OpenCV-android-sdk/sdk`.
- `app/build.gradle`: `implementation project(':opencv')`.
- Đã sửa `OpenCV-android-sdk/sdk/java/AndroidManifest.xml`: bỏ `package="org.opencv"`
  (tránh xung đột `namespace` của AGP 8).

### 2.1. ⚠️ Bắt buộc: cài NDK + CMake
Module OpenCV 4.12 dùng `externalNativeBuild` để đưa `libc++_shared.so` vào APK,
nên máy build cần **NDK** và **CMake**:

`Tools > SDK Manager > SDK Tools` → tick:
- **NDK (Side by side)**
- **CMake**

(Khi sync/build lần đầu, Android Studio thường tự hiện nút "Install NDK/CMake" — bấm là xong.)
Native build này rất nhẹ (chỉ compile 1 file `dummy.cpp`), **không** build lại OpenCV từ source.

### 2.2. Init OpenCV trong code
Đã làm sẵn trong `MainActivity.onCreate()`:
```java
if (!OpenCVLoader.initDebug()) { ... }   // 4.12 cũng dùng được OpenCVLoader.initLocal()
```

### 2.3. Nếu KHÔNG muốn cài NDK/CMake
Bạn có thể tự thêm `libc++_shared.so` (lấy từ NDK) vào
`app/src/main/jniLibs/<abi>/` rồi xoá khối `externalNativeBuild`/`prefab` trong
`OpenCV-android-sdk/sdk/build.gradle`. Tuy nhiên cách cài NDK/CMake ở 2.1 là đơn giản nhất.

### 2.4. Cách khác: dùng Maven (không cần thư mục SDK, không cần NDK)
```groovy
// settings.gradle: xoá 2 dòng include ':opencv' + projectDir
// app/build.gradle:
implementation 'org.opencv:opencv:4.12.0'   // OpenCV chính thức trên Maven Central (từ 4.10)
```

---

## 3. Build

### 3.1. Bằng Android Studio
1. Mở thư mục `appAndroid` (Open an existing project).
2. Chờ Gradle sync.
3. `Build > Make Project` (Ctrl+F9).
4. Chạy: chọn thiết bị → nút ▶ Run.

### 3.2. Bằng dòng lệnh (Windows PowerShell)
```powershell
cd D:\DOC\Face_identity_detection_system\appAndroid
.\gradlew.bat assembleDebug
```
APK xuất ra: `app/build/outputs/apk/debug/app-debug.apk`

> Nếu chưa có `gradlew.bat`/`gradle-wrapper.jar`: mở project bằng Android Studio một lần
> để IDE sinh wrapper, hoặc chạy `gradle wrapper --gradle-version 8.7`.

---

## 4. Test (Kiểm thử)

### 4.1. Trên Emulator
- Tạo AVD (Device Manager) — nên chọn ảnh **x86_64** (khớp `abiFilters`).
- Trong cấu hình AVD, mục **Camera** → đặt **Front/Back = Webcam0** để dùng webcam máy tính
  (hoặc `Emulated` để có ảnh giả).
- Run app → cấp quyền camera → nghiêng "máy" sang ngang (Ctrl+Left/Right xoay emulator) → START.

### 4.2. Trên điện thoại thật
- Xem mục 5 (Deploy) để bật USB Debugging và cài qua adb.

### 4.3. Checklist kiểm tra
| Kiểm tra | Kỳ vọng |
|---|---|
| Quyền camera | Hộp thoại xin quyền hiện ra; sau khi đồng ý, thấy hình camera |
| Model load | Không có Toast lỗi; Logcat in "Model load OK: /data/..." |
| Xoay máy | Cửa sổ KHÔNG xoay; icon quay mượt; tắt auto-rotate hệ thống vẫn quay |
| Gate ngang | Cầm dọc: START mờ + icon xoay hiện; nằm ngang: START bật |
| Detect | Nằm ngang, nhấn START → khung xanh quanh mặt, bám đúng cả khi đổi cam |
| FPS | Chỉ số FPS > 0 và ổn định (thường 12–30 tùy máy) |
| STOP | Khung biến mất, FPS/Count về 0 |

Xem log:
```powershell
adb logcat -s MainActivity FaceDetectorModel CameraController
```

> **Ảnh chụp màn hình thấy preview ĐEN là bình thường**: PreviewView mặc định dùng
> SurfaceView, nội dung không được ghi vào screenshot. Muốn screenshot thấy hình:
> `previewView.setImplementationMode(PreviewView.ImplementationMode.COMPATIBLE);`

---

## 5. Deploy (Cài đặt lên điện thoại)

### 5.1. Bật Developer Options + USB Debugging
1. **Settings > About phone** → chạm **Build number** 7 lần.
2. **Settings > System > Developer options** → bật **USB debugging**.
3. Cắm cáp USB, trên điện thoại chọn **Allow** khi hỏi "Allow USB debugging?".

### 5.2. adb
```powershell
adb devices                       # thấy serial + "device" là OK
adb install app-debug.apk         # cài mới
adb install -r app-debug.apk      # cài đè (giữ dữ liệu)
adb uninstall com.example.faceidentity   # gỡ
```

### 5.3. Build APK/AAB ký (Signed) để phát hành
1. `Build > Generate Signed Bundle / APK...`
2. Chọn **APK** → **Create new...** để tạo keystore.
3. Điền: Key store path, password, alias, validity (>= 25 năm).
4. Chọn **release** → Finish.

Tạo keystore bằng dòng lệnh:
```powershell
keytool -genkeypair -v -keystore my-release-key.jks -keyalg RSA -keysize 2048 -validity 10000 -alias my-alias
```
Ký thủ công (tùy chọn):
```powershell
.\gradlew.bat assembleRelease
```
> Để `gradlew assembleRelease` tự ký, thêm `signingConfigs` vào `app/build.gradle`.

---

## 6. Tham số hiệu năng YuNet (đặt trong `MainActivity`)

| Tham số | Giá trị mặc định | Ý nghĩa & cách chỉnh |
|---|---|---|
| `scoreThreshold` | 0.9 | Ngưỡng tin cậy. **Tăng** → ít false positive, có thể bỏ sót; **giảm** → bắt nhiều hơn nhưng nhiễu. |
| `nmsThreshold` | 0.3 | Ngưỡng gộp box chồng nhau (NMS). Thấp → gộp mạnh (ít box trùng). |
| `topK` | 5000 | Số box tối đa xét trước NMS. **Giảm** (vd 500) → nhanh hơn, đủ cho vài khuôn mặt. |
| Độ phân giải analysis | 640×480 | `targetAnalysisSize` trong `CameraController`. Nhỏ hơn → FPS cao hơn, độ chính xác giảm. |

**Mẹo tăng FPS / giảm GC (đã áp dụng):**
- Tái sử dụng `Mat`/`byte[]` (không `new` mỗi frame) → giảm Garbage Collector.
- `STRATEGY_KEEP_ONLY_LATEST` → không xử lý frame tồn đọng.
- Xử lý trên executor riêng → không chặn UI.
- Giới hạn độ phân giải ImageAnalysis (640×480).
- Có thể hạ `topK` xuống 500 và đặt `targetAnalysisSize` = 480×360 nếu máy yếu.

---

## 7. Lỗi thường gặp & cách khắc phục

| Lỗi | Nguyên nhân | Khắc phục |
|---|---|---|
| `FaceDetectorYN.create() = null` / crash khi create | Sai đường dẫn model hoặc OpenCV < 4.8 | Kiểm tra model trong assets; dùng OpenCV 4.12 sẵn có |
| Toast "THIẾU MODEL..." | Model chưa có trong assets lúc build | Chép model (mục 1) rồi **Rebuild** |
| Toast "OpenCV init thất bại" | Module `:opencv` chưa build / thiếu `.so` | Kiểm tra đã cài NDK+CMake (mục 2.1) và sync lại |
| `UnsatisfiedLinkError: libopencv_java4.so` | ABI không khớp | Đảm bảo `abiFilters` chứa ABI của thiết bị; emulator dùng x86_64 |
| `UnsatisfiedLinkError: libc++_shared.so` | Chưa cài NDK/CMake nên `.so` không được đóng gói | Cài NDK+CMake (mục 2.1) rồi rebuild; hoặc dùng cách 2.3/2.4 |
| `NDK not configured` / `CMake ... not found` | Thiếu NDK hoặc CMake | SDK Manager → cài NDK (Side by side) + CMake |
| Sync lỗi `package="org.opencv"` trong manifest | AGP 8 không cho `package` khi đã có `namespace` | Đã sửa sẵn; nếu tải lại SDK mới, bỏ `package` trong `OpenCV-android-sdk/sdk/java/AndroidManifest.xml` |
| Màn hình đen, không có preview | Chưa cấp quyền camera | Cấp quyền trong Settings > Apps |
| Có preview nhưng không có khung | Chưa nằm ngang / chưa nhấn START / model chưa load | Nghiêng ngang, nhấn START; xem Logcat |
| Box xoay 90° so với mặt | Sai dấu xoay overlay | Đổi `overlay.setRotation(-deviceDegrees)` thành `+deviceDegrees` trong `applyOverlayRotation()` |
| Box soi gương (trái↔phải) khi dùng cam trước | Vấn đề mirror | Kiểm tra `overlay.setMirror(isFront)` qua `LensChangedListener` |
| `adb devices` trống | Chưa bật USB debugging / thiếu driver | Bật USB debugging; cài driver OEM; chọn Allow trên máy |
| APK to bất thường | Đóng gói nhiều ABI | Giữ `abiFilters` gọn; dùng App Bundle (`.aab`) khi phát hành |

---

## 8. Kiến trúc MVC (tóm tắt)

- **Model** — `FaceDetectorModel`: bọc `FaceDetectorYN`, chỉ lo detect.
- **View** — `MainActivity` (UI, nút, quyền), `CameraPreview` (vẽ box overlay).
- **Controller** — `CameraController` (CameraX), `FaceDetectionController` (điều phối detect, FPS, count).
- **Utils** — `FileUtils` (copy model), `ImageUtils` (ImageProxy→Mat), `PermissionUtils` (quyền).

---

## 9. Xoay kiểu app camera (cửa sổ KHOÁ, chỉ icon xoay mượt)

**Hành vi:** cầm dọc → START **mờ** + hiện **icon xoay**. Nghiêng máy sang ngang → START bật,
các icon **xoay mượt** về đúng chiều trọng lực. Đang detect mà rời khỏi ngang → **tự dừng**.
**Cửa sổ không bao giờ xoay** → không có animation xoay giật.

### 9.1. Vì sao khoá cửa sổ thay vì `fullSensor`

`screenOrientation="fullSensor"` làm **cả cửa sổ xoay** → có animation xoay của hệ thống,
nhìn giật và phải counter-rotate thủ công để nút khỏi nhảy chỗ.

Các app camera hệ thống làm ngược lại — **khoá cửa sổ** (`portrait`) và chỉ xoay icon:

| Thành phần | Cách xử lý |
|---|---|
| Cửa sổ | `screenOrientation="portrait"` — không xoay, không animation → **mượt tuyệt đối**. |
| Nút / bảng FPS | Tự động **đứng yên trên thân máy** (vì cửa sổ không xoay) — không cần counter-rotate. |
| Hướng máy | Đọc từ `OrientationEventListener` → `deviceDegrees ∈ {0,90,180,270}`. Chạy **kể cả khi hệ thống TẮT auto-rotate**. ⚠️ `Configuration.orientation` và `Display.getRotation()` **không dùng được nữa** — cửa sổ khoá nên chúng luôn báo "portrait"/`ROTATION_0`. |
| Icon | `animate().rotationBy(...)` tới `-deviceDegrees`, chọn **đường ngắn nhất** (chuẩn hoá delta về `[-180,180)`) → không quay 270° khi chỉ cần 90°. |
| Preview | **KHÔNG** set `targetRotation` theo cảm biến. Ảnh vẫn đúng chiều thực tế vì camera và màn hình xoay **cùng nhau** (ảnh trong hệ cửa sổ xoay ngược đúng bằng góc máy xoay → triệt tiêu). Set theo cảm biến sẽ bị **xoay 2 lần**. |
| ImageAnalysis | **CÓ** set `targetRotation` theo cảm biến → ảnh phân tích thẳng đứng theo trọng lực → **YuNet detect chuẩn** (YuNet chỉ nhận diện tốt mặt đứng thẳng). |
| Overlay | Phải xoay `-deviceDegrees` + hoán đổi w/h để hệ toạ độ **trùng ảnh phân tích** (xem 9.2). |

> Muốn đổi sang **bắt buộc DỌC**: sửa `isDeviceLandscape()` thành `deviceDegrees == 0 \|\| deviceDegrees == 180`.

### 9.2. Vì sao overlay phải xoay

`ImageAnalysis` xoay theo trọng lực còn cửa sổ thì không → **hai hệ toạ độ lệch nhau
`deviceDegrees`**. Nếu vẽ box thẳng vào cửa sổ thì box sẽ lệch khi cầm ngang.

`MainActivity.applyOverlayRotation()`:
1. Đặt kích thước overlay = kích thước cửa sổ **hoán đổi w/h** (khi máy nằm ngang), `layout_gravity="center"`.
2. `overlay.setRotation(-deviceDegrees)` → xoay quanh tâm nên overlay **phủ khít** cửa sổ trở lại.
3. Hệ toạ độ overlay giờ **trùng ảnh phân tích** → `CameraPreview.onDraw()` giữ nguyên
   công thức `fitCenter`, không phải sửa.

**Vùng fitCenter có trùng preview không?** Có. Preview fit ảnh tỉ lệ `fh:fw` vào khung
`(winW × winH)` → `scale = min(winW/fh, winH/fw)`. Overlay fit ảnh `fw:fh` vào khung
`(winH × winW)` → `scale = min(winH/fw, winW/fh)`. **Bằng nhau**, và cùng căn giữa → trùng khít.

**Bẫy đã xử lý:** hàm **idempotent** (chỉ `setLayoutParams` khi kích thước thật sự đổi)
→ gọi từ `OnLayoutChangeListener` không gây vòng lặp layout vô hạn.

## 10. Camera trước / sau

- **Mặc định: CAM TRƯỚC** (`CameraController` đặt `lensFacing = LENS_FACING_FRONT`).
- Nút thứ 3 là **icon đổi camera**; vì chỉ có icon nên có **Toast** báo camera đang dùng.
- `CameraController.switchCamera()` lật `lensFacing` rồi `bindUseCases()` lại.
- **Lật gương:** CameraX mirror preview của cam trước. Callback `LensChangedListener`
  → `overlay.setMirror(isFront)` để bounding box khớp preview. Toạ độ detect nằm trong
  hệ ảnh **chưa mirror**, nên overlay lật `x` theo `viewW - right`.
- **Fallback:** máy không có cam yêu cầu → `bindUseCases()` tự chuyển sang cam còn lại
  (`hasCamera()`), không crash.

> Đổi mặc định về cam sau: trong `MainActivity.onCreate()` sửa
> `setLensFacing(CameraSelector.LENS_FACING_BACK)`.

# Export

- https://platform.ultralytics.com/348-scorpion/example-project/yolo26n
- python py/yolo26n2ncnn.py

# NCNN Android YOLO26

Real-time object detection Android application using YOLO26n with NCNN framework.

[![Android](https://img.shields.io/badge/Platform-Android-green.svg)](https://developer.android.com)
[![NCNN](https://img.shields.io/badge/Framework-NCNN-blue.svg)](https://github.com/Tencent/ncnn)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Features

- 🚀 Real-time object detection with YOLO26n
- 📱 Supports both CPU and GPU (Vulkan) inference
- 🎯 80 COCO classes detection
- 📷 Front/Back camera switching
- ⚡ Optimized for mobile devices

## Performance

| Device | CPU FPS | GPU FPS |
|--------|---------|---------|
| Huawei P40 (Kirin 990) | ~10 | ~4 |
| Solana Seeker (Dimensity 7300) | ~13 | ~4* |

*Note: Some MediaTek GPUs may have Vulkan driver precision issues. CPU mode is recommended for best accuracy.

## Build Instructions

### 1. Export YOLO26n NCNN Model

```python
from ultralytics import YOLO

# Load model
model = YOLO("yolo26n.pt")

# Export to NCNN format
model.export(format="ncnn")
```

Copy the generated `yolo26n.ncnn.param` and `yolo26n.ncnn.bin` files to `app/src/main/assets/`.

### 2. Download Dependencies

#### NCNN

Download from [ncnn releases](https://github.com/Tencent/ncnn/releases):
- `ncnn-YYYYMMDD-android-vulkan.zip`

Extract to `app/src/main/jni/` and rename to `ncnn-android-vulkan`.

#### OpenCV Mobile

Download from [opencv-mobile releases](https://github.com/nihui/opencv-mobile/releases):
- `opencv-mobile-4.10.0-android.zip`

Extract to `app/src/main/jni/`.

### 3. Directory Structure

Ensure JNI directory structure:

```
app/src/main/jni/
├── CMakeLists.txt
├── yolo.h
├── yolo.cpp
├── yolo26ncnn.cpp
├── ncnn-android-vulkan/
│   ├── arm64-v8a/
│   ├── armeabi-v7a/
│   ├── x86/
│   └── x86_64/
└── opencv-mobile-4.10.0-android/
    └── sdk/
```

### 4. Model Files

Place model files in assets directory:

```
app/src/main/assets/
├── yolo26n.ncnn.param
└── yolo26n.ncnn.bin
```

### 5. Build Project

Open with Android Studio or build via command line:

```bash
./gradlew assembleDebug
```

## GPU Acceleration Notes

- Requires Vulkan-capable device
- GPU mode forces FP32 for better accuracy
- First GPU inference may have shader compilation delay
- Some low/mid-range GPUs (Mali-G5xx, Mali-G6xx on MediaTek) may have precision issues - use CPU mode instead

## Customization

### Custom Classes

If using a custom-trained model, modify the `class_names` array in `yolo.h`.

### Confidence Threshold

Default threshold is 0.5. Adjust in `yolo.h`:

```cpp
int detect(..., float prob_threshold = 0.50f, ...);
```

## References

- [ncnn-android-yolov8](https://github.com/nihui/ncnn-android-yolov8)
- [NCNN](https://github.com/Tencent/ncnn)
- [OpenCV Mobile](https://github.com/nihui/opencv-mobile)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)

## License

This project is licensed under the MIT License.

---

[中文文档](README_CN.md)

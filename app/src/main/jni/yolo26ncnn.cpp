#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <android/native_window.h>
#include <jni.h>
#include <string>
#include <vector>

#include <platform.h>
#include <benchmark.h>

#include "yolo.h"
#include "log.h"
#include "byte_tracker.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#if __ARM_NEON
#include <arm_neon.h>

#endif

static Yolo* g_yolo = 0;
static ByteTracker* g_byteTracker = 0;
static ncnn::Mutex lock;

// 定义全局日志开关变量，默认启用
int g_log_enabled = 1;

static jclass objCls = nullptr;
static jmethodID objInit = nullptr;
static jfieldID xId = nullptr;
static jfieldID yId = nullptr;
static jfieldID wId = nullptr;
static jfieldID hId = nullptr;
static jfieldID labelId = nullptr;
static jfieldID probId = nullptr;
static jfieldID trackIdId = nullptr;

// YOLO26n 配置
static const int YOLO26_TARGET_SIZE = 320;
static const float YOLO26_MEAN_VALS[3] = {0.f, 0.f, 0.f};
static const float YOLO26_NORM_VALS[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};

// 设置日志开关状态
void set_log_enabled(int enabled) {
    g_log_enabled = enabled;
}

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    LOGD( "Yolo26Ncnn JNI_OnLoad");
    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved) {
    LOGD( "Yolo26Ncnn JNI_OnUnload");

    ncnn::MutexLockGuard g(lock);
    if (g_yolo != 0) {
        delete g_yolo;
        g_yolo = 0;
    }

    if (g_byteTracker != 0) {
        delete g_byteTracker;
        g_byteTracker = 0;
    }
}

JNIEXPORT jboolean JNICALL Java_com_example_yolo26ncnn_Yolo26Ncnn_loadModel(JNIEnv* env, jobject thiz, jobject assetManager, jint modelid, jint useGpu) {
    if (modelid < 0 || modelid > 0) {
        return JNI_FALSE;
    }

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    LOGD( "Yolo26Ncnn loadModel %p", mgr);

    const char* modeltype = "yolo26n5";
    bool use_gpu = true;

    {
        ncnn::MutexLockGuard g(lock);

        // Check GPU availability
        if (use_gpu && ncnn::get_gpu_count() == 0) {
            FLOGW("GPU not available, falling back to CPU");
            use_gpu = false;
        }

        if (!g_yolo) {
            g_yolo = new Yolo;
        }
        if (!g_byteTracker) {
            g_byteTracker = new ByteTracker(15, 0.5f, 0.1f, 0.7f);
        }

        const char* device_name = use_gpu ? "GPU (FP32)" : "CPU";
        FLOGI( "Yolo26Ncnn Loading model: %s on %s", modeltype, device_name);
        g_yolo->load(mgr, modeltype, YOLO26_TARGET_SIZE, YOLO26_MEAN_VALS, YOLO26_NORM_VALS, use_gpu);
        FLOGI( "Yolo26Ncnn Model loaded successfully");


        if (objCls == nullptr){
            objCls      =  (jclass) env->NewGlobalRef(env->FindClass("com/example/yolo26ncnn/Yolo26Ncnn$Obj"));
            objInit   = env->GetMethodID(objCls, "<init>", "()V");
            xId       = env->GetFieldID(objCls, "x", "F");
            yId       = env->GetFieldID(objCls, "y", "F");
            wId       = env->GetFieldID(objCls, "w", "F");
            hId       = env->GetFieldID(objCls, "h", "F");
            labelId   = env->GetFieldID(objCls, "label", "Ljava/lang/String;");
            probId    = env->GetFieldID(objCls, "prob", "F");
            trackIdId = env->GetFieldID(objCls, "trackId", "I");  // 获取trackId字段ID
        }
    }

    return JNI_TRUE;
}

JNIEXPORT jobjectArray JNICALL
Java_com_example_yolo26ncnn_Yolo26Ncnn_detect(JNIEnv *env, jobject thiz, jobject bitmap, jint rotationDegrees, jboolean isFrontCamera)
{
    double start_time = ncnn::get_current_time();

    AndroidBitmapInfo bitmapInfo;
    void *pixels = nullptr;

    if (AndroidBitmap_getInfo(env, bitmap, &bitmapInfo) != ANDROID_BITMAP_RESULT_SUCCESS) {
        return nullptr;
    }

    if (bitmapInfo.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        FLOGD( "Yolo26Ncnn Only RGBA_8888 supported");
        return nullptr;
    }

    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS) {
        return nullptr;
    }

    // --- 优化核心：物理旋转与镜像 ---

    // 1. 将 Bitmap 像素封装为 cv::Mat (此时还是 RGBA 格式)
    // 这是一个零拷贝操作，直接指向 Bitmap 内存
    cv::Mat rgbaMat(bitmapInfo.height, bitmapInfo.width, CV_8UC4, pixels);

    // 2. 处理物理旋转
    cv::Mat rotatedMat;
    if (rotationDegrees == 90) {
        cv::rotate(rgbaMat, rotatedMat, cv::ROTATE_90_CLOCKWISE);
    } else if (rotationDegrees == 180) {
        cv::rotate(rgbaMat, rotatedMat, cv::ROTATE_180);
    } else if (rotationDegrees == 270) {
        cv::rotate(rgbaMat, rotatedMat, cv::ROTATE_90_COUNTERCLOCKWISE);
    } else {
        rotatedMat = rgbaMat; // 0度不处理
    }

    // 3. 处理镜像 (针对前置摄像头)
    if (isFrontCamera) {
        // flipCode 1 代表水平翻转
        cv::flip(rotatedMat, rotatedMat, 1);
    }

    // 4. 转换为 NCNN Mat (同时完成 RGBA -> RGB 通道转换)
    // 注意：转换后的宽高是 rotatedMat 的宽高
    ncnn::Mat inputMat = ncnn::Mat::from_pixels(
            rotatedMat.data,
            ncnn::Mat::PIXEL_RGBA2RGB,
            rotatedMat.cols,           // 源图像宽度
            rotatedMat.rows            // 源图像高度
    );

    // 释放 Bitmap 锁 (数据已在 rotatedMat 中处理完毕)
    AndroidBitmap_unlockPixels(env, bitmap);


    // 5. 执行推理
    std::vector<Object> objects;
    std::vector<TrackedObject> trackedObjects;

    {
        ncnn::MutexLockGuard g(lock);
        if (g_yolo) {
            g_yolo->detect(inputMat, objects,YOLO26_NORM_VALS);
        }
        if (g_byteTracker) {
            trackedObjects = g_byteTracker->update(objects);  // 使用ByteTracker进行跟踪
        }
    }

    // --- 后处理逻辑 ---
    // 由于我们物理上旋转了图片，objects 里的坐标已经是对应旋转后图片的物理像素坐标
    // 无需再手动翻转 obj.rect.x，直接返回给 Java 层即可

    jobjectArray jObjArray = env->NewObjectArray(trackedObjects.size(), objCls, NULL);

    // 循环遍历trackedObjects
    for (size_t i = 0; i < trackedObjects.size(); i++) {
        jobject jObj = env->NewObject(objCls, objInit);

        // 将bbox从[x1,y1,x2,y2]格式转换回[x,y,w,h]格式
        float x = trackedObjects[i].bbox.x;
        float y = trackedObjects[i].bbox.y;
        float w = trackedObjects[i].bbox.width;
        float h = trackedObjects[i].bbox.height;

        env->SetFloatField(jObj, xId, x);
        env->SetFloatField(jObj, yId, y);
        env->SetFloatField(jObj, wId, w);
        env->SetFloatField(jObj, hId, h);

        int label = trackedObjects[i].classId;
        const char *label_name = (label >= 0 && label < 80) ? class_names[label] : "unknown";
        env->SetObjectField(jObj, labelId, env->NewStringUTF(label_name));
        env->SetFloatField(jObj, probId, trackedObjects[i].score);
        env->SetIntField(jObj, trackIdId, trackedObjects[i].trackId);  // 设置跟踪ID

        env->SetObjectArrayElement(jObjArray, i, jObj);
    }

    double elasped = ncnn::get_current_time() - start_time;
    LOGD( "Yolo26Ncnn %.2fms total detect (with rotate)", elasped);

    return jObjArray;
}

JNIEXPORT void JNICALL
Java_com_example_yolo26ncnn_Yolo26Ncnn_setLogEnabled(JNIEnv *env, jobject thiz, jboolean enabled) {
    set_log_enabled(enabled ? 1 : 0);
}

}

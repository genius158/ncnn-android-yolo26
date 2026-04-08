#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <android/native_window.h>
#include <android/log.h>
#include <jni.h>
#include <string>
#include <vector>

#include <platform.h>
#include <benchmark.h>

#include "yolo.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#if __ARM_NEON
#include <arm_neon.h>
#endif

static Yolo* g_yolo = 0;
static ncnn::Mutex lock;

// YOLO26n 配置
static const int YOLO26_TARGET_SIZE = 320;
static const float YOLO26_MEAN_VALS[3] = {0.f, 0.f, 0.f};
static const float YOLO26_NORM_VALS[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "Yolo26Ncnn", "JNI_OnLoad");
    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "Yolo26Ncnn", "JNI_OnUnload");

    ncnn::MutexLockGuard g(lock);
    delete g_yolo;
    g_yolo = 0;
}

JNIEXPORT jboolean JNICALL Java_com_example_yolo26ncnn_Yolo26Ncnn_loadModel(JNIEnv* env, jobject thiz, jobject assetManager, jint modelid, jint useGpu) {
    if (modelid < 0 || modelid > 0) {
        return JNI_FALSE;
    }

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    __android_log_print(ANDROID_LOG_DEBUG, "Yolo26Ncnn", "loadModel %p", mgr);

    const char* modeltype = "yolo26n5";
    bool use_gpu = (useGpu == 1);

    {
        ncnn::MutexLockGuard g(lock);

        // Check GPU availability
        if (use_gpu && ncnn::get_gpu_count() == 0) {
            __android_log_print(ANDROID_LOG_WARN, "Yolo26Ncnn", "GPU not available, falling back to CPU");
            use_gpu = false;
        }

        if (!g_yolo) {
            g_yolo = new Yolo;
        }

        const char* device_name = use_gpu ? "GPU (FP32)" : "CPU";
        __android_log_print(ANDROID_LOG_DEBUG, "Yolo26Ncnn", "Loading model: %s on %s", modeltype, device_name);
        g_yolo->load(mgr, modeltype, YOLO26_TARGET_SIZE, YOLO26_MEAN_VALS, YOLO26_NORM_VALS, use_gpu);
        __android_log_print(ANDROID_LOG_DEBUG, "Yolo26Ncnn", "Model loaded successfully");
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
        __android_log_print(ANDROID_LOG_DEBUG, "Yolo26Ncnn", "Only RGBA_8888 supported");
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
    {
        ncnn::MutexLockGuard g(lock);
        if (g_yolo) {
            g_yolo->detect(inputMat, objects);
        }
    }

    // --- 后处理逻辑 ---
    // 由于我们物理上旋转了图片，objects 里的坐标已经是对应旋转后图片的物理像素坐标
    // 无需再手动翻转 obj.rect.x，直接返回给 Java 层即可

    jclass objCls = env->FindClass("com/example/yolo26ncnn/Yolo26Ncnn$Obj");
    jmethodID objInit = env->GetMethodID(objCls, "<init>", "(Lcom/example/yolo26ncnn/Yolo26Ncnn;)V");
    jfieldID xId = env->GetFieldID(objCls, "x", "F");
    jfieldID yId = env->GetFieldID(objCls, "y", "F");
    jfieldID wId = env->GetFieldID(objCls, "w", "F");
    jfieldID hId = env->GetFieldID(objCls, "h", "F");
    jfieldID labelId = env->GetFieldID(objCls, "label", "Ljava/lang/String;");
    jfieldID probId = env->GetFieldID(objCls, "prob", "F");

    jobjectArray jObjArray = env->NewObjectArray(objects.size(), objCls, NULL);

    for (size_t i = 0; i < objects.size(); i++) {
        jobject jObj = env->NewObject(objCls, objInit, thiz);

        env->SetFloatField(jObj, xId, objects[i].rect.x);
        env->SetFloatField(jObj, yId, objects[i].rect.y);
        env->SetFloatField(jObj, wId, objects[i].rect.width);
        env->SetFloatField(jObj, hId, objects[i].rect.height);

        int label = objects[i].label;
        const char *label_name = (label >= 0 && label < 80) ? class_names[label] : "unknown";
        env->SetObjectField(jObj, labelId, env->NewStringUTF(label_name));
        env->SetFloatField(jObj, probId, objects[i].prob);

        env->SetObjectArrayElement(jObjArray, i, jObj);
    }

    double elasped = ncnn::get_current_time() - start_time;
    __android_log_print(ANDROID_LOG_DEBUG, "Yolo26Ncnn", "%.2fms total detect (with rotate)", elasped);

    return jObjArray;
}

}

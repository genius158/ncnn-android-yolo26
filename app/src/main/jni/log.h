#ifndef YOLO26NCNN_LOG_H
#define YOLO26NCNN_LOG_H

#include <android/log.h>

#define TAG "Yolo26Ncnn"

// 全局日志开关，可通过函数动态设置
extern int g_log_enabled;

// 动态日志宏定义
#ifdef __cplusplus
extern "C" {
#endif

void set_log_enabled(int enabled);

#ifdef __cplusplus
}
#endif

#if defined(__cplusplus)
#define LOGD(...) if(g_log_enabled) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
#define LOGE(...) if(g_log_enabled) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)
#define LOGI(...) if(g_log_enabled) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGW(...) if(g_log_enabled) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)
#else
#define LOGD(...) if(g_log_enabled) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
#define LOGE(...) if(g_log_enabled) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)
#define LOGI(...) if(g_log_enabled) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGW(...) if(g_log_enabled) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)
#endif

// force log
#define FLOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)
#define FLOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define FLOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)

#endif //YOLO26NCNN_LOG_H

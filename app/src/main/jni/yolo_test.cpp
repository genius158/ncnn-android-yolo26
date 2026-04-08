// YOLO26 ncnn implementation
// out0: dims=2, w=8400, h=84  => [84 rows, 8400 cols]
// row 0..3 : cx, cy, w, h (decoded in 640x640 coords)
// row 4..83: 80 class probs (sigmoid already in graph)

#include "yolo.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cpu.h>
#include <layer.h>

#include <android/log.h>
#include <cfloat>
#include <vector>
#include <algorithm>
#include <cmath>

#define TAG "YOLO26"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)

// 优化 1: 将矩形面积预计算，避免 NMS 中重复计算
struct ObjectOpt : Object {
    float area;
};

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (objects[i].prob > p) i++;
        while (objects[j].prob < p) j--;

        if (i <= j)
        {
            std::swap(objects[i], objects[j]);
            i++;
            j--;
        }
    }

    if (left < j) qsort_descent_inplace(objects, left, j);
    if (i < right) qsort_descent_inplace(objects, i, right);
}

static void qsort_descent_inplace(std::vector<ObjectOpt>& objects, int left, int right) {
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;
    while (i <= j) {
        while (objects[i].prob > p) i++;
        while (objects[j].prob < p) j--;
        if (i <= j) {
            std::swap(objects[i], objects[j]);
            i++;
            j--;
        }
    }
    if (left < j) qsort_descent_inplace(objects, left, j);
    if (i < right) qsort_descent_inplace(objects, i, right);
}

static void nms_sorted_bboxes(const std::vector<ObjectOpt>& objects, std::vector<int>& picked, float nms_threshold, bool agnostic = false) {
    picked.clear();
    const int n = (int)objects.size();
    for (int i = 0; i < n; i++) {
        const ObjectOpt& a = objects[i];
        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++) {
            const ObjectOpt& b = objects[picked[j]];
            if (!agnostic && a.label != b.label) continue;
            float inter_area = intersection_area(a, b);
            float iou = inter_area / (a.area + b.area - inter_area);
            if (iou > nms_threshold) {
                keep = 0;
                break;
            }
        }
        if (keep) picked.push_back(i);
    }
}

// 优化 2: 极大优化内存访问模式
static void generate_proposals_yolo26(const ncnn::Mat& pred, float prob_threshold, std::vector<ObjectOpt>& objects) {
    objects.clear();
    const int num_proposals = pred.w;        // 8400
    const int num_feat      = pred.h;        // 84
    const int num_class     = num_feat - 4;  // 80

    // 预取所有行的指针，避免在循环内调用 pred.row()
    const float* row_ptrs[84];
    for (int i = 0; i < 84; i++) {
        row_ptrs[i] = pred.row(i);
    }

    // 预分配空间，减少 vector 扩容开销
    objects.reserve(256);

    for (int i = 0; i < num_proposals; i++) {
        int label = -1;
        float score = -1.f;

        // 寻找最大置信度类别
        for (int k = 0; k < num_class; k++) {
            float s = row_ptrs[4 + k][i];
            if (s > score) {
                score = s;
                label = k;
            }
        }

        if (score < prob_threshold) continue;

        ObjectOpt obj;
        // 直接读取坐标
        float cx = row_ptrs[0][i];
        float cy = row_ptrs[1][i];
        float bw = row_ptrs[2][i];
        float bh = row_ptrs[3][i];

        obj.rect.x = cx - bw * 0.5f;
        obj.rect.y = cy - bh * 0.5f;
        obj.rect.width  = bw;
        obj.rect.height = bh;
        obj.label = label;
        obj.prob  = score;
        obj.area  = bw * bh; // 预计算面积
        objects.push_back(obj);
    }
}

Yolo::Yolo()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}

Yolo::~Yolo()
{
    yolo.clear();
}

int Yolo::load(const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu)
{
    yolo.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    yolo.opt = ncnn::Option();

#if NCNN_VULKAN
    yolo.opt.use_vulkan_compute = use_gpu;
    if (use_gpu)
    {
        // Force FP32 for better accuracy on GPU
        yolo.opt.use_fp16_packed = false;
        yolo.opt.use_fp16_storage = false;
        yolo.opt.use_fp16_arithmetic = false;
    }
#endif

    yolo.opt.num_threads = ncnn::get_big_cpu_count();
    yolo.opt.blob_allocator = &blob_pool_allocator;
    yolo.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.ncnn.param", modeltype);
    sprintf(modelpath, "%s.ncnn.bin", modeltype);

    yolo.load_param(parampath);
    yolo.load_model(modelpath);

    target_size = _target_size;

    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    return 0;
}

int Yolo::load(AAssetManager* mgr, const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu)
{
    yolo.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

//    ncnn::set_cpu_powersave(2); // 绑定大核
//    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    yolo.opt = ncnn::Option();
    yolo.opt.num_threads = ncnn::get_big_cpu_count();
    yolo.opt.blob_allocator = &blob_pool_allocator;
    yolo.opt.workspace_allocator = &workspace_pool_allocator;

    // 优化 3: 开启 FP16 推理（CPU/GPU 均大幅提速）
    yolo.opt.use_fp16_packed = true;
    yolo.opt.use_fp16_storage = true;
    yolo.opt.use_fp16_arithmetic = true;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.ncnn.param", modeltype);
    sprintf(modelpath, "%s.ncnn.bin", modeltype);

    yolo.load_param(mgr, parampath);
    yolo.load_model(mgr, modelpath);

    target_size = _target_size;

    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    return 0;
}

// 1. 定义 Sigmoid 辅助函数
inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

int Yolo::detect(const ncnn::Mat& input, std::vector<Object>& objects, float prob_threshold, float nms_threshold)
{
    objects.clear();

    const int img_w = input.w;
    const int img_h = input.h;


    __android_log_print(ANDROID_LOG_DEBUG, "Yolo26Ncnn","Resizing and img_w: %dx  img_h: %dx", img_w, img_h);

    // Your model is fixed 640x640 (8400 points), so target_size MUST be 640
    const int dst_size = target_size; // set to 640 in Java/C++ init
    int new_w = dst_size;
    int new_h = dst_size;
    int wpad = 0;
    int hpad = 0;
    ncnn::Mat in_pad;
    if (img_w != dst_size || img_h != dst_size) {
        float scale = std::min((float)target_size / img_w, (float)target_size / img_h);
         new_w = (int)(img_w * scale);
         new_h = (int)(img_h * scale);

        __android_log_print(ANDROID_LOG_DEBUG, "Yolo26Ncnn","Resizing and padding to %dx%d", new_w, new_h);

        // 使用 ncnn 内置的高效缩放并直接归一化
        ncnn::Mat in_resized;
        ncnn::resize_bilinear(input, in_resized, new_w, new_h);

        wpad = dst_size - new_w;
        hpad = dst_size - new_h;
        // 填充边距
        ncnn::copy_make_border(in_resized, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);
    } else {
        in_pad = input.clone();
    }

    // 归一化
    const float norm_vals_ultra[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals_ultra);
    // 执行推理
    ncnn::Extractor ex = yolo.create_extractor();
    ex.set_light_mode(true);
    ex.input("in0", in_pad);
    ncnn::Mat out;
    ex.extract("out0", out);

    LOGD("YOLO26 output: dims=%d, w=%d (proposals), h=%d (features), c=%d", out.dims, out.w, out.h, out.c);

    // 根据实际输出维度来处理
    if (out.dims != 2) {
        LOGD("Unexpected output dims: %d", out.dims);
        return -1;
    }

    // 检查维度顺序 - 如果w=84,h=8400则需要交换处理方式
    const int num_proposals = out.w;  // 实际提案数
    const int num_features = out.h;   // 实际特征数

    LOGD("Processing output - proposals: %d, features: %d", num_proposals, num_features);


    // 检查输出张量的维度顺序 - 修复维度顺序问题
    std::vector<ObjectOpt> proposals;
    proposals.clear();
    proposals.reserve(256);

    // 如果维度顺序是正确的(w=8400, h=84)
    if (num_proposals == 8400 && num_features == 84) {
        // 预取所有行的指针
        const float* row_ptrs[84];
        for (int i = 0; i < 84; i++) {
            row_ptrs[i] = out.row(i);
        }

        for (int i = 0; i < num_proposals; i++) {
            int label = -1;
            float score = -1.f;

            // 寻找最大置信度类别 (从第4个特征开始是80个类别)
            for (int k = 0; k < 80; k++) {  // 80 classes
                float s = row_ptrs[4 + k][i];  // 第4行之后是类别概率
                if (s > score) {
                    score = s;
                    label = k;
                }
            }

            if (score < prob_threshold) continue;

            ObjectOpt obj;
            // 直接读取前4个坐标值
            float cx = row_ptrs[0][i];  // center x
            float cy = row_ptrs[1][i];  // center y
            float bw = row_ptrs[2][i];  // box width
            float bh = row_ptrs[3][i];  // box height

            obj.rect.x = cx - bw * 0.5f;
            obj.rect.y = cy - bh * 0.5f;
            obj.rect.width  = bw;
            obj.rect.height = bh;
            obj.label = label;
            obj.prob  = score;
            obj.area  = bw * bh; // 预计算面积
            proposals.push_back(obj);

            LOGD("Processing1 transposed output format  cx: %.2f, cy: %.2f, bw: %.2f, bh: %.2f, score: %.2f", cx, cy, bw, bh, score);

        }
    }
    else if (num_proposals == 84 && num_features == 8400) {
        // 维度格式: [84 x 8400]，其中 84 是特征(rows)，8400 是点(cols)
        LOGD("Processing format: 84 features x 8400 proposals");

        // 2. 修正后的解析循环
        for (int i = 0; i < num_features; i++) {
            float score = -1.f;
            int label = -1;

            // 只对类别行进行 Sigmoid
            for (int k = 0; k < 80; k++) {
                float s = sigmoid(out.row(k + 4)[i]); // 第 4 行开始是类别
                if (s > score) {
                    score = s;
                    label = k;
                }
            }

            // 必须有置信度过滤，建议设为 0.25f
            if (score < prob_threshold) continue;

            // 坐标处理
            float cx = out.row(0)[i] * dst_size; // 假设输出是 0~1，需还原到 640
            float cy = out.row(1)[i] * dst_size;
            float bw = out.row(2)[i] * dst_size;
            float bh = out.row(3)[i] * dst_size;

            ObjectOpt obj;
            obj.rect.x = cx - bw * 0.5f;
            obj.rect.y = cy - bh * 0.5f;
            obj.rect.width = bw;
            obj.rect.height = bh;
            obj.label = label;
            obj.prob  = score;
            obj.area  = bw * bh;

            proposals.push_back(obj);
        }
    } else {
        LOGD("Unexpected output dimensions: w=%d, h=%d", num_proposals, num_features);
        return -1;
    }

    if (proposals.empty())
        return 0;

    // 排序与 NMS
    if (proposals.size() > 1) {
        qsort_descent_inplace(proposals, 0, proposals.size() - 1);
    }

    // NMS (set nms_threshold<=0 to disable)
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    // 7. 坐标映射还原 (一次循环完成所有转换)
    objects.clear();
    float ratio_w = (float)img_w / new_w;
    float ratio_h = (float)img_h / new_h;
    float off_x = (wpad / 2) * ratio_w;
    float off_y = (hpad / 2) * ratio_h;

    for (int idx : picked) {
        const ObjectOpt& p = proposals[idx];
        Object obj;
        obj.prob = p.prob;
        obj.label = p.label;

        // 映射回原图并裁剪边界
        float x0 = (p.rect.x * ratio_w) - off_x;
        float y0 = (p.rect.y * ratio_h) - off_y;
        float x1 = ((p.rect.x + p.rect.width) * ratio_w) - off_x;
        float y1 = ((p.rect.y + p.rect.height) * ratio_h) - off_y;

        obj.rect.x = std::max(0.f, x0);
        obj.rect.y = std::max(0.f, y0);
        obj.rect.width = std::min((float)img_w - obj.rect.x, x1 - x0);
        obj.rect.height = std::min((float)img_h - obj.rect.y, y1 - y0);

        objects.push_back(obj);
    }

    return 0;
}


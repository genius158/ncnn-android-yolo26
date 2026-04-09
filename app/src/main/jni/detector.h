//
// Created by xianwei.yan on 2026/4/9.
//


#ifndef NCNN_ANDROID_YOLO26_DETECTOR_H
#define NCNN_ANDROID_YOLO26_DETECTOR_H

#include <opencv2/core/types.hpp>

// Bounding box structure
struct BBox {
    float x;       // Top-left x
    float y;       // Top-left y
    float width;
    float height;

    BBox() : x(0), y(0), width(0), height(0) {}
    BBox(float x_, float y_, float w_, float h_) : x(x_), y(y_), width(w_), height(h_) {}

    // 计算面积
    float area() const {
        return width * height;
    }

    // 重载交集运算符
    BBox operator&(const BBox& other) const {
        float inter_x1 = std::max(x, other.x);
        float inter_y1 = std::max(y, other.y);
        float inter_x2 = std::min(x + width, other.x + other.width);
        float inter_y2 = std::min(y + height, other.y + other.height);

        float inter_width = std::max(0.0f, inter_x2 - inter_x1);
        float inter_height = std::max(0.0f, inter_y2 - inter_y1);

        return BBox(inter_x1, inter_y1, inter_width, inter_height);
    }
};

struct Object {
    BBox rect;
    int label;
    float prob;
};


// Tracked object (vehicle with optional plate)
struct TrackedObject {
    int trackId;
    float score;
    BBox bbox;
    int classId;
    int framesSinceUpdate;


    TrackedObject()
            : trackId(-1), score(0), classId(0),framesSinceUpdate(0) {}
};

#endif //NCNN_ANDROID_YOLO26_DETECTOR_H

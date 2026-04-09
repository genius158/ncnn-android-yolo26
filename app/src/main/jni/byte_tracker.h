/**
 * ByteTrack Implementation for Vehicle Tracking
 *
 * Based on: https://github.com/ifzhang/ByteTrack
 * Simplified for single-class vehicle tracking.
 */

#ifndef BYTE_TRACKER_H
#define BYTE_TRACKER_H

#include <vector>
#include <memory>
#include "detector.h"
#include "kalman_filter.h"

// Track state
enum class TrackState {
    New,
    Tracked,
    Lost,
    Removed
};

// Single track
class STrack {
public:
    STrack(const BBox& bbox, float score, int classId);

    void predict();
    void update(const Object& det);
    void markLost();
    void markRemoved();
    void activate(int frameId, int trackId);
    void reActivate(const Object& det, int frameId);

    BBox getBBox() const;
    int getTrackId() const { return m_trackId; }
    TrackState getState() const { return m_state; }
    float getScore() const { return m_score; }
    int getClassId() const { return m_classId; }
    int getFrameId() const { return m_frameId; }
    int getTimeSinceUpdate() const { return m_timeSinceUpdate; }
    void incrementTimeSinceUpdate() { m_timeSinceUpdate++; }
    bool isActivated() const { return m_isActivated; }

    // For cost matrix calculation
    std::vector<float> getTlwh() const;

private:
    KalmanFilter m_kalman;
    std::vector<float> m_mean;  // [x, y, a, h, vx, vy, va, vh]
    std::vector<float> m_covariance;

    BBox m_bbox;
    float m_score;
    int m_classId;
    int m_trackId;
    int m_frameId;
    int m_startFrame;
    int m_timeSinceUpdate;
    bool m_isActivated;
    TrackState m_state;
};

// ByteTrack tracker
class ByteTracker {
public:
    /**
     * Create ByteTracker
     *
     * @param maxAge Maximum frames to keep lost tracks
     * @param highThresh High confidence threshold for first association
     * @param lowThresh Low confidence threshold for second association
     * @param matchThresh IoU threshold for matching
     */
    ByteTracker(int maxAge = 30, float highThresh = 0.5f,
                float lowThresh = 0.1f, float matchThresh = 0.8f);

    /**
     * Update tracker with new detections
     *
     * @param detections Vector of detections from current frame
     * @return Vector of tracked objects
     */
    std::vector<TrackedObject> update(const std::vector<Object>& detections);

    /**
     * Reset tracker state
     */
    void reset();

private:
    int m_maxAge;
    float m_highThresh;
    float m_lowThresh;
    float m_matchThresh;

    int m_frameId;
    int m_trackIdCount;

    std::vector<std::shared_ptr<STrack>> m_trackedTracks;
    std::vector<std::shared_ptr<STrack>> m_lostTracks;
    std::vector<std::shared_ptr<STrack>> m_removedTracks;

    // Hungarian algorithm for assignment
    std::vector<std::pair<int, int>> linearAssignment(
        const std::vector<std::vector<float>>& costMatrix,
        float thresh
    );

    // Calculate IoU cost matrix
    std::vector<std::vector<float>> calcIoUCostMatrix(
        const std::vector<std::shared_ptr<STrack>>& tracks,
        const std::vector<Object>& detections
    );

    // IoU calculation
    float calcIoU(const std::vector<float>& tlwh1, const BBox& bbox2);
};

#endif // BYTE_TRACKER_H

/**
 * ByteTrack Implementation
 */

#include "byte_tracker.h"
#include <android/log.h>
#include <algorithm>
#include <cmath>
#include <limits>

// ============== STrack Implementation ==============

STrack::STrack(const BBox& bbox, float score, int classId)
    : m_bbox(bbox), m_score(score), m_classId(classId), m_trackId(-1), m_frameId(0),
      m_startFrame(0), m_timeSinceUpdate(0), m_isActivated(false),
      m_state(TrackState::New) {

    // Initialize Kalman filter state
    // State: [x, y, a, h, vx, vy, va, vh]
    // where (x, y) is center, a is aspect ratio, h is height
    float cx = bbox.x + bbox.width / 2;
    float cy = bbox.y + bbox.height / 2;
    float a = bbox.width / std::max(bbox.height, 1.0f);
    float h = bbox.height;

    m_mean = {cx, cy, a, h, 0, 0, 0, 0};
    m_covariance.resize(64, 0);  // 8x8 covariance matrix
    m_kalman.initiate(m_mean, m_covariance);
}

void STrack::predict() {
    m_kalman.predict(m_mean, m_covariance);
}

void STrack::update(const Object& det) {
    float cx = det.rect.x + det.rect.width / 2;
    float cy = det.rect.y + det.rect.height / 2;
    float a = det.rect.width / std::max(det.rect.height, 1.0f);
    float h = det.rect.height;

    std::vector<float> measurement = {cx, cy, a, h};
    m_kalman.update(m_mean, m_covariance, measurement);

    m_bbox = det.rect;
    m_score = det.prob;
    m_classId = det.label;
    m_timeSinceUpdate = 0;
    m_state = TrackState::Tracked;
}

void STrack::markLost() {
    m_state = TrackState::Lost;
}

void STrack::markRemoved() {
    m_state = TrackState::Removed;
}

void STrack::activate(int frameId, int trackId) {
    m_trackId = trackId;
    m_frameId = frameId;
    m_startFrame = frameId;
    m_isActivated = true;
    m_state = TrackState::Tracked;
}

void STrack::reActivate(const Object& det, int frameId) {
    update(det);
    m_frameId = frameId;
    m_isActivated = true;
    m_state = TrackState::Tracked;
}

BBox STrack::getBBox() const {
    // Convert from Kalman state to bbox
    float cx = m_mean[0];
    float cy = m_mean[1];
    float a = m_mean[2];
    float h = m_mean[3];
    float w = a * h;

    BBox bbox;
    bbox.x = cx - w / 2;
    bbox.y = cy - h / 2;
    bbox.width = w;
    bbox.height = h;
    return bbox;
}

std::vector<float> STrack::getTlwh() const {
    BBox bbox = getBBox();
    return {bbox.x, bbox.y, bbox.width, bbox.height};
}

// ============== ByteTracker Implementation ==============

ByteTracker::ByteTracker(int maxAge, float highThresh, float lowThresh, float matchThresh)
    : m_maxAge(maxAge), m_highThresh(highThresh), m_lowThresh(lowThresh),
      m_matchThresh(matchThresh), m_frameId(0), m_trackIdCount(0) {
}

std::vector<TrackedObject> ByteTracker::update(const std::vector<Object>& detections) {
    m_frameId++;

    // Separate high and low confidence detections
    std::vector<Object> highDets, lowDets;
    for (const auto& det : detections) {
        if (det.prob >= m_highThresh) {
            highDets.push_back(det);
        } else if (det.prob >= m_lowThresh) {
            lowDets.push_back(det);
        }
    }

    // Predict all tracks
    for (auto& track : m_trackedTracks) {
        track->predict();
    }
    for (auto& track : m_lostTracks) {
        track->predict();
    }

    // First association: high confidence detections with tracked tracks
    std::vector<std::shared_ptr<STrack>> unconfirmedTracks;
    std::vector<std::shared_ptr<STrack>> confirmedTracks;

    for (auto& track : m_trackedTracks) {
        if (track->isActivated()) {
            confirmedTracks.push_back(track);
        } else {
            unconfirmedTracks.push_back(track);
        }
    }

    // Combine confirmed and lost tracks for matching
    std::vector<std::shared_ptr<STrack>> trackPool = confirmedTracks;
    trackPool.insert(trackPool.end(), m_lostTracks.begin(), m_lostTracks.end());

    // Calculate IoU cost matrix
    auto costMatrix = calcIoUCostMatrix(trackPool, highDets);
    auto matches = linearAssignment(costMatrix, m_matchThresh);

    std::vector<int> unmatchedTracks, unmatchedDets;
    std::vector<bool> trackMatched(trackPool.size(), false);
    std::vector<bool> detMatched(highDets.size(), false);

    for (const auto& match : matches) {
        trackMatched[match.first] = true;
        detMatched[match.second] = true;
        trackPool[match.first]->update(highDets[match.second]);
    }

    for (size_t i = 0; i < trackPool.size(); i++) {
        if (!trackMatched[i]) unmatchedTracks.push_back(i);
    }
    for (size_t i = 0; i < highDets.size(); i++) {
        if (!detMatched[i]) unmatchedDets.push_back(i);
    }

    // Second association: low confidence detections with remaining tracks
    std::vector<std::shared_ptr<STrack>> remainTracks;
    for (int idx : unmatchedTracks) {
        if (trackPool[idx]->getState() == TrackState::Tracked) {
            remainTracks.push_back(trackPool[idx]);
        }
    }

    auto costMatrix2 = calcIoUCostMatrix(remainTracks, lowDets);
    auto matches2 = linearAssignment(costMatrix2, 0.5f);

    for (const auto& match : matches2) {
        remainTracks[match.first]->update(lowDets[match.second]);
    }

    // Mark unmatched tracks as lost
    std::vector<std::shared_ptr<STrack>> newTrackedTracks;
    std::vector<std::shared_ptr<STrack>> newLostTracks;

    for (auto& track : m_trackedTracks) {
        if (track->getState() == TrackState::Tracked) {
            track->incrementTimeSinceUpdate();
            if (track->getTimeSinceUpdate() > m_maxAge) {
                track->markRemoved();
            } else {
                newTrackedTracks.push_back(track);
            }
        }
    }

    // Initialize new tracks from unmatched high confidence detections
    for (int idx : unmatchedDets) {
        auto newTrack = std::make_shared<STrack>(highDets[idx].rect, highDets[idx].prob, highDets[idx].label);
        newTrack->activate(m_frameId, ++m_trackIdCount);
        newTrackedTracks.push_back(newTrack);
    }

    // Update lost tracks
    for (auto& track : m_lostTracks) {
        if (track->getState() != TrackState::Tracked) {
            track->incrementTimeSinceUpdate();
            if (track->getTimeSinceUpdate() > m_maxAge) {
                track->markRemoved();
            } else {
                newLostTracks.push_back(track);
            }
        }
    }

    m_trackedTracks = newTrackedTracks;
    m_lostTracks = newLostTracks;

    // Build output
    std::vector<TrackedObject> output;
    for (auto& track : m_trackedTracks) {
        if (track->isActivated()) {
            TrackedObject obj;
            obj.score = track->getScore();
            obj.trackId = track->getTrackId();
            obj.bbox = track->getBBox();
            obj.classId = track->getClassId();
            obj.framesSinceUpdate = track->getTimeSinceUpdate();
            output.push_back(obj);
        }
    }

    return output;
}

void ByteTracker::reset() {
    m_trackedTracks.clear();
    m_lostTracks.clear();
    m_removedTracks.clear();
    m_frameId = 0;
    m_trackIdCount = 0;
}

std::vector<std::pair<int, int>> ByteTracker::linearAssignment(
    const std::vector<std::vector<float>>& costMatrix, float thresh) {

    std::vector<std::pair<int, int>> matches;

    if (costMatrix.empty() || costMatrix[0].empty()) {
        return matches;
    }

    int numTracks = costMatrix.size();
    int numDets = costMatrix[0].size();

    // Simple greedy assignment (Hungarian algorithm would be better but more complex)
    std::vector<bool> trackUsed(numTracks, false);
    std::vector<bool> detUsed(numDets, false);

    // Find best matches greedily
    while (true) {
        float bestCost = thresh;
        int bestTrack = -1, bestDet = -1;

        for (int i = 0; i < numTracks; i++) {
            if (trackUsed[i]) continue;
            for (int j = 0; j < numDets; j++) {
                if (detUsed[j]) continue;
                if (costMatrix[i][j] < bestCost) {
                    bestCost = costMatrix[i][j];
                    bestTrack = i;
                    bestDet = j;
                }
            }
        }

        if (bestTrack == -1) break;

        matches.push_back({bestTrack, bestDet});
        trackUsed[bestTrack] = true;
        detUsed[bestDet] = true;
    }

    return matches;
}

std::vector<std::vector<float>> ByteTracker::calcIoUCostMatrix(
    const std::vector<std::shared_ptr<STrack>>& tracks,
    const std::vector<Object>& detections) {

    std::vector<std::vector<float>> costMatrix(tracks.size(),
        std::vector<float>(detections.size(), 1.0f));

    for (size_t i = 0; i < tracks.size(); i++) {
        auto tlwh = tracks[i]->getTlwh();
        for (size_t j = 0; j < detections.size(); j++) {
            float iou = calcIoU(tlwh, detections[j].rect);
            costMatrix[i][j] = 1.0f - iou;  // Convert IoU to cost
        }
    }

    return costMatrix;
}

float ByteTracker::calcIoU(const std::vector<float>& tlwh1, const BBox& bbox2) {
    float x1 = std::max(tlwh1[0], bbox2.x);
    float y1 = std::max(tlwh1[1], bbox2.y);
    float x2 = std::min(tlwh1[0] + tlwh1[2], bbox2.x + bbox2.width);
    float y2 = std::min(tlwh1[1] + tlwh1[3], bbox2.y + bbox2.height);

    float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float area1 = tlwh1[2] * tlwh1[3];
    float area2 = bbox2.width * bbox2.height;
    float unionArea = area1 + area2 - intersection;

    return unionArea > 0 ? intersection / unionArea : 0;
}

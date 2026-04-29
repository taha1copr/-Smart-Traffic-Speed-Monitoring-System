# sort.py - improved minimal SORT-like tracker (no pip)
# Features:
# - IoU greedy matching
# - max_age (remove stale trackers)
# - min_hits (only confirm trackers after several matches)
# - returns tracker bbox + id + confirmed flag
import numpy as np
from typing import List, Tuple

def iou(bb_test, bb_gt):
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    w = max(0., xx2 - xx1)
    h = max(0., yy2 - yy1)
    inter = w * h
    area1 = max(0., (bb_test[2]-bb_test[0]) * (bb_test[3]-bb_test[1]))
    area2 = max(0., (bb_gt[2]-bb_gt[0]) * (bb_gt[3]-bb_gt[1]))
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union

class Tracker:
    _count = 0
    def __init__(self, bbox: List[float]):
        # bbox = [x1,y1,x2,y2]
        self.bbox = np.array(bbox, dtype=float)
        self.id = Tracker._count
        Tracker._count += 1
        self.hits = 1              # number of total matches
        self.time_since_update = 0 # frames since last matched
        self.age = 0               # total frames alive

    def update(self, bbox: List[float]):
        self.bbox = np.array(bbox, dtype=float)
        self.hits += 1
        self.time_since_update = 0

    def predict(self):
        # No motion model — just increment counters
        self.age += 1
        self.time_since_update += 1
        return self.bbox

class Sort:
    def __init__(self, max_age=15, min_hits=2, iou_threshold=0.3):
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.iou_threshold = float(iou_threshold)
        self.trackers: List[Tracker] = []

    def update(self, dets: np.ndarray) -> List[Tuple[float,float,float,float,int,bool]]:
        """
        dets: Nx5 array [[x1,y1,x2,y2,conf], ...]
        returns list of tuples: (x1,y1,x2,y2, track_id, is_confirmed_bool)
        """
        # 1. predict all trackers (age/time update)
        for trk in self.trackers:
            trk.predict()

        matched_tracker_idx = set()
        matched_det_idx = set()

        if dets is None or len(dets) == 0:
            dets = np.empty((0,5))

        # 2. build IoU matrix
        N = len(self.trackers)
        M = len(dets)
        iou_matrix = np.zeros((N, M), dtype=float)

        for t, trk in enumerate(self.trackers):
            for d in range(M):
                iou_matrix[t, d] = iou(trk.bbox, dets[d,:4])

        # 3. Greedy matching: find highest IoU pairs until threshold
        if N > 0 and M > 0:
            for _ in range(min(N, M)):
                t, d = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                best_iou = iou_matrix[t, d]
                if best_iou < self.iou_threshold:
                    break
                # match t with d
                self.trackers[t].update(dets[d,:4])
                matched_tracker_idx.add(t)
                matched_det_idx.add(d)
                # invalidate row and column
                iou_matrix[t, :] = -1
                iou_matrix[:, d] = -1

        # 4. create new trackers for unmatched detections
        for d in range(M):
            if d not in matched_det_idx:
                bbox = dets[d,:4].tolist()
                self.trackers.append(Tracker(bbox))

        # 5. remove dead trackers (time_since_update > max_age)
        alive_trackers = []
        for trk in self.trackers:
            if trk.time_since_update <= self.max_age:
                alive_trackers.append(trk)
        self.trackers = alive_trackers

        # 6. prepare output (bbox + id + confirmed)
        outputs = []
        for trk in self.trackers:
            x1, y1, x2, y2 = trk.bbox.tolist()
            confirmed = (trk.hits >= self.min_hits)
            outputs.append((x1, y1, x2, y2, trk.id, confirmed))

        return outputs

    def active_ids(self):
        return [trk.id for trk in self.trackers]

"""
PlateTracker — theo dõi biển số qua nhiều frame để giảm jitter và false positive.
"""

from collections import Counter, deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
import config


class PlateTracker:
    """Track và smooth detections qua nhiều frame."""

    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: Dict[int, Track] = {}
        self.next_id = 0
        self.frame_count = 0

    def update(self,
               detections: List[Dict],
               plate_texts: Optional[List[str]] = None,
               confidences: Optional[List[float]] = None) -> List[Dict]:
        """
        Cập nhật tracker với detections mới.
        Trả về: list các track đã được xác nhận (hits >= min_hits).
        """
        self.frame_count += 1
        matched_pairs = self._match_detections(detections)
        matched_det_indices = {idx for _, idx in matched_pairs}

        # Cập nhật track đã match
        for track_id, det_idx in matched_pairs:
            det = detections[det_idx]
            self.tracks[track_id].update(
                bbox=det['bbox'],
                plate_text=plate_texts[det_idx] if plate_texts else None,
                confidence=confidences[det_idx] if confidences else det.get('conf', 0),
            )

        # Tạo track mới cho detection chưa match
        for det_idx, det in enumerate(detections):
            if det_idx in matched_det_indices:
                continue
            track = Track(
                track_id=self.next_id,
                bbox=det['bbox'],
                plate_text=plate_texts[det_idx] if plate_texts else None,
                confidence=confidences[det_idx] if confidences else det.get('conf', 0),
                min_hits=self.min_hits,
            )
            self.tracks[self.next_id] = track
            self.next_id += 1

        # Tăng age cho track không được update, xóa track quá cũ
        for track in self.tracks.values():
            if not any(tid == track.track_id for tid, _ in matched_pairs):
                track.miss()

        self.tracks = {tid: t for tid, t in self.tracks.items() if t.age <= self.max_age}

        return [t.get_state() for t in self.tracks.values() if t.hits >= self.min_hits]

    def _match_detections(self, detections: List[Dict]) -> List[Tuple[int, int]]:
        """Ghép detections với tracks hiện tại dựa trên IOU."""
        if not detections or not self.tracks:
            return []

        matches = []
        for track_id, track in self.tracks.items():
            best_iou, best_idx = 0, -1
            for det_idx, det in enumerate(detections):
                iou = self._iou(track.bbox, det['bbox'])
                if iou > best_iou:
                    best_iou, best_idx = iou, det_idx
            if best_iou > self.iou_threshold:
                matches.append((track_id, best_idx))

        return matches

    @staticmethod
    def _iou(box1: Tuple, box2: Tuple) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def reset(self):
        self.tracks.clear()
        self.frame_count = 0

    def get_stats(self) -> Dict:
        return {
            "frame_count": self.frame_count,
            "active_tracks": sum(1 for t in self.tracks.values() if t.hits >= self.min_hits),
            "total_tracks": len(self.tracks),
        }


class Track:
    """Một track đơn lẻ với lịch sử detection."""

    def __init__(self, track_id: int, bbox: Tuple, plate_text: Optional[str],
                 confidence: float, min_hits: int = 3, max_history: int = 10):
        self.track_id = track_id
        self.hits = 1
        self.age = 0
        self.start_time = datetime.now()

        self.bbox_history: deque = deque([bbox], maxlen=max_history)
        self.text_history: deque = deque([plate_text] if plate_text else [], maxlen=max_history)
        self.conf_history: deque = deque([confidence], maxlen=max_history)

    @property
    def bbox(self) -> Tuple:
        return self.bbox_history[-1]

    def update(self, bbox: Tuple, plate_text: Optional[str], confidence: float):
        self.bbox_history.append(bbox)
        if plate_text:
            self.text_history.append(plate_text)
        self.conf_history.append(confidence)
        self.hits += 1
        self.age = 0

    def miss(self):
        self.age += 1

    def get_state(self) -> Dict:
        bboxes = list(self.bbox_history)
        avg_bbox = (
            int(np.mean([b[0] for b in bboxes])),
            int(np.mean([b[1] for b in bboxes])),
            int(np.mean([b[2] for b in bboxes])),
            int(np.mean([b[3] for b in bboxes])),
        )

        best_text = Counter(self.text_history).most_common(1)[0][0] if self.text_history else ""
        avg_conf = float(np.mean(list(self.conf_history))) if self.conf_history else 0.0

        return {
            "track_id": self.track_id,
            "bbox": avg_bbox,
            "plate_text": best_text,
            "confidence": avg_conf,
            "hits": self.hits,
            "age": self.age,
            "duration": (datetime.now() - self.start_time).total_seconds(),
        }


def draw_tracked_plates(frame: np.ndarray, tracks: List[Dict]) -> np.ndarray:
    """Vẽ các tracked plates lên frame."""
    annotated = frame.copy()

    for track in tracks:
        x1, y1, x2, y2 = track['bbox']
        conf = track['confidence']

        color = (0, 255, 0) if conf > 0.85 else (0, 255, 255) if conf > 0.70 else (0, 0, 255)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated,
                    f"ID:{track['track_id']} {track['plate_text']} ({conf:.2f})",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return annotated
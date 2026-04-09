"""
🎯 PLATE TRACKER - Track same plates across video frames
Để xử lý video, giảm false positives, smooth detection
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import deque
from datetime import datetime, timedelta
import cv2

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import config


class PlateTracker:
    """
    Track và smooth detections qua nhiều frames
    
    Lợi ích:
    - Giảm nhảy nhàn (jitter)
    - Voting từ nhiều frames
    - Giảm false positives
    - Tăng confidence
    """
    
    def __init__(self, 
                 max_age: int = 30,
                 min_hits: int = 3,
                 iou_threshold: float = 0.3):
        """
        Initialize tracker
        
        Args:
            max_age: Tối đa bao nhiêu frames mà không detect được
            min_hits: Cần bao nhiêu frames để confirm detection
            iou_threshold: Ngưỡng IOU để match với track cũ
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self.tracks: Dict[int, 'Track'] = {}  # ID -> Track object
        self.next_id = 0
        self.frame_count = 0
    
    def update(self, 
               detections: List[Dict],
               plate_texts: Optional[List[str]] = None,
               confidences: Optional[List[float]] = None) -> List[Dict]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of {'bbox': (x1,y1,x2,y2), 'conf': float}
            plate_texts: Optional list of OCR texts
            confidences: Optional list of OCR confidences
        
        Returns:
            List of confirmed tracks with averaged values
        """
        self.frame_count += 1
        
        # Match detections to existing tracks
        matched_pairs = self._match_detections(detections)
        
        # Update existing tracks
        for track_id, det_idx in matched_pairs:
            detection = detections[det_idx]
            plate_text = plate_texts[det_idx] if plate_texts else None
            conf = confidences[det_idx] if confidences else detection.get('conf', 0)
            
            self.tracks[track_id].update(
                detection['bbox'],
                plate_text,
                conf
            )
        
        # Create new tracks for unmatched detections
        unmatched_dets = set(range(len(detections))) - \
                        set(idx for _, idx in matched_pairs)
        
        for det_idx in unmatched_dets:
            detection = detections[det_idx]
            plate_text = plate_texts[det_idx] if plate_texts else None
            conf = confidences[det_idx] if confidences else detection.get('conf', 0)
            
            track_id = self.next_id
            self.next_id += 1
            
            self.tracks[track_id] = Track(
                track_id,
                detection['bbox'],
                plate_text,
                conf,
                min_hits=self.min_hits
            )
        
        # Age out old tracks
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if track.age > self.max_age:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        # Return confirmed tracks
        confirmed = []
        for track in self.tracks.values():
            if track.hits >= self.min_hits:
                confirmed.append(track.get_state())
        
        return confirmed
    
    def _match_detections(self, 
                         detections: List[Dict]) -> List[Tuple[int, int]]:
        """
        Match current detections to existing tracks using IOU
        
        Returns: List of (track_id, detection_idx) pairs
        """
        if not detections or not self.tracks:
            return []
        
        matches = []
        
        for track_id, track in self.tracks.items():
            best_iou = 0
            best_det_idx = -1
            
            for det_idx, detection in enumerate(detections):
                iou = self._iou(track.bbox, detection['bbox'])
                
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = det_idx
            
            # Match if IOU above threshold
            if best_iou > self.iou_threshold and best_det_idx >= 0:
                matches.append((track_id, best_det_idx))
        
        return matches
    
    @staticmethod
    def _iou(box1: Tuple, box2: Tuple) -> float:
        """Calculate IOU between two boxes (x1,y1,x2,y2)"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Intersection
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)
        
        intersection = max(0, xi_max - xi_min) * max(0, yi_max - yi_min)
        
        # Union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def reset(self):
        """Reset tracker"""
        self.tracks.clear()
        self.frame_count = 0
    
    def get_stats(self) -> Dict:
        """Get tracker statistics"""
        return {
            "frame_count": self.frame_count,
            "active_tracks": len([t for t in self.tracks.values() 
                                 if t.hits >= self.min_hits]),
            "total_tracks": len(self.tracks),
            "next_id": self.next_id
        }


class Track:
    """
    Single track object với history của detections
    """
    
    def __init__(self, 
                 track_id: int,
                 bbox: Tuple,
                 plate_text: Optional[str],
                 confidence: float,
                 min_hits: int = 3,
                 max_history: int = 10):
        """
        Initialize track
        
        Args:
            track_id: ID of track
            bbox: Initial bounding box (x1,y1,x2,y2)
            plate_text: Initial OCR text
            confidence: Initial confidence
            min_hits: Hits needed to confirm
            max_history: Max detections to keep
        """
        self.track_id = track_id
        self.hits = 1
        self.age = 0
        self.min_hits = min_hits
        
        # History of detections
        self.bbox_history = deque([bbox], maxlen=max_history)
        self.text_history = deque([plate_text] if plate_text else [], 
                                  maxlen=max_history)
        self.conf_history = deque([confidence], maxlen=max_history)
        
        # Time tracking
        self.start_time = datetime.now()
        self.last_update = datetime.now()
    
    def update(self,
              bbox: Tuple,
              plate_text: Optional[str],
              confidence: float):
        """Update track with new detection"""
        self.bbox_history.append(bbox)
        if plate_text:
            self.text_history.append(plate_text)
        self.conf_history.append(confidence)
        
        self.hits += 1
        self.age = 0  # Reset age
        self.last_update = datetime.now()
    
    def miss(self):
        """Mark as a miss (not detected in frame)"""
        self.age += 1
    
    def get_averaged_bbox(self) -> Tuple:
        """Get averaged bounding box từ history"""
        bboxes = list(self.bbox_history)
        
        if not bboxes:
            return bboxes[0]
        
        # Average each coordinate
        x1_avg = int(np.mean([b[0] for b in bboxes]))
        y1_avg = int(np.mean([b[1] for b in bboxes]))
        x2_avg = int(np.mean([b[2] for b in bboxes]))
        y2_avg = int(np.mean([b[3] for b in bboxes]))
        
        return (x1_avg, y1_avg, x2_avg, y2_avg)
    
    def get_best_text(self) -> str:
        """Get most common plate text từ history"""
        if not self.text_history:
            return ""
        
        # Count occurrences
        from collections import Counter
        counter = Counter(self.text_history)
        best_text = counter.most_common(1)[0][0]
        
        return best_text
    
    def get_avg_confidence(self) -> float:
        """Get average confidence"""
        if not self.conf_history:
            return 0.0
        
        return np.mean(list(self.conf_history))
    
    def get_state(self) -> Dict:
        """Get current state of track"""
        return {
            "track_id": self.track_id,
            "bbox": self.get_averaged_bbox(),  # Smoothed
            "plate_text": self.get_best_text(),  # Most common
            "confidence": self.get_avg_confidence(),  # Averaged
            "hits": self.hits,
            "age": self.age,
            "duration": (datetime.now() - self.start_time).total_seconds()
        }


# Utility function for visualization
def draw_tracked_plates(frame: np.ndarray,
                       tracks: List[Dict],
                       draw_history: bool = False) -> np.ndarray:
    """
    Draw tracked plates on frame
    
    Args:
        frame: Input frame
        tracks: List of track states from tracker.update()
        draw_history: Draw trails of tracks
    
    Returns:
        Annotated frame
    """
    annotated = frame.copy()
    
    for track in tracks:
        track_id = track['track_id']
        bbox = track['bbox']
        plate_text = track['plate_text']
        confidence = track['confidence']
        hits = track['hits']
        
        x1, y1, x2, y2 = bbox
        
        # Color based on confidence
        if confidence > 0.85:
            color = (0, 255, 0)  # Green - high confidence
        elif confidence > 0.70:
            color = (0, 255, 255)  # Yellow - medium
        else:
            color = (0, 0, 255)  # Red - low
        
        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Draw ID and text
        label = f"ID:{track_id} {plate_text} ({confidence:.2f})"
        cv2.putText(annotated, label, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw hits (how many frames detected)
        hits_text = f"hits:{hits}"
        cv2.putText(annotated, hits_text, (x1, y2+20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return annotated


# Test
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 TEST - Plate Tracker")
    print("="*70)
    
    try:
        # Initialize tracker
        tracker = PlateTracker(max_age=30, min_hits=2)
        print("\n✅ Tracker initialized")
        
        # Simulate detections across 5 frames
        for frame_num in range(5):
            # Create fake detected plates
            detections = [
                {'bbox': (100, 100, 200, 150), 'conf': 0.9},
                {'bbox': (300, 200, 400, 250), 'conf': 0.85},
            ]
            plate_texts = ["29A1234", "50B5678"]
            confidences = [0.9, 0.85]
            
            # Add slight noise to first plate
            if frame_num > 0:
                detections[0]['bbox'] = (101, 101, 201, 151)
            
            # Update tracker
            tracks = tracker.update(detections, plate_texts, confidences)
            
            print(f"\nFrame {frame_num}:")
            print(f"  Detections: {len(detections)}")
            print(f"  Confirmed tracks: {len(tracks)}")
            for track in tracks:
                print(f"    ID:{track['track_id']} - {track['plate_text']} "
                      f"({track['confidence']:.2f}) hits:{track['hits']}")
        
        # Get stats
        stats = tracker.get_stats()
        print(f"\n📊 Tracker Stats:")
        print(f"  Frames: {stats['frame_count']}")
        print(f"  Active tracks: {stats['active_tracks']}")
        print(f"  Total created: {stats['total_tracks']}")
        
        print(f"\n✅ Plate Tracker Ready!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

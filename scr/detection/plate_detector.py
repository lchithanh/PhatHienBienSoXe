from ultralytics import YOLO
import cv2
from pathlib import Path
import config

class PlateDetector:
    def __init__(self, model_path=None):
        # Hỗ trợ fallback vào model train mới nhất
        candidate_paths = []
        if model_path:
            candidate_paths.append(Path(model_path))
        candidate_paths.append(config.PLATE_MODEL_PATH)
        candidate_paths.append(Path('runs/detect/plate_detection_improved/final_v2/weights/best.pt'))
        candidate_paths.append(Path('runs/detect/train/weights/best.pt'))

        chosen_path = None
        for p in candidate_paths:
            if p and p.exists():
                chosen_path = p
                break

        if chosen_path is None:
            raise FileNotFoundError(
                f"Model biển số không tìm thấy. Kiểm tra các đường dẫn: {candidate_paths}"
            )

        self.model_path = chosen_path
        self.model = YOLO(str(chosen_path))
        self.conf = config.PLATE_CONF
        self.iou = getattr(config, 'IOU_THRESHOLD', 0.45)

    def detect(self, crop_img):
        """
        ✨ ENHANCED: Detect plates with validation and filtering
        Returns: list of valid plate bounding boxes (x1,y1,x2,y2)
        """
        # ✨ NEW: Check if image is valid
        if crop_img.size == 0:
            return []
        
        # ✨ NEW: Ensure minimum size
        h, w = crop_img.shape[:2]
        if h < 30 or w < 30:
            return []
        
        # ✨ NEW: Add IOU threshold
        results = self.model(
            crop_img, 
            conf=self.conf,
            iou=self.iou,  # NMS IOU threshold
            verbose=False
        )[0]
        
        plates = []
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # ✨ NEW: Ensure box is valid
            if x1 >= x2 or y1 >= y2:
                continue
            
            plate_w = x2 - x1
            plate_h = y2 - y1
            
            # ✨ NEW: Filter too small plates
            if config.FILTER_SMALL_PLATES:
                if plate_w < config.MIN_PLATE_WIDTH:
                    continue
                
                aspect_ratio = plate_w / max(plate_h, 1)
                if aspect_ratio > config.MAX_PLATE_ASPECT_RATIO:
                    continue  # Too wide (false positive)
            
            # ✨ NEW: Add confidence score
            plates.append({
                'bbox': (x1, y1, x2, y2),
                'conf': float(box.conf[0]),
                'area': plate_w * plate_h
            })
        
        # ✨ NEW: Sort by confidence
        plates.sort(key=lambda p: p['conf'], reverse=True)
        
        # Return only top plates (usually just 1-2 per vehicle)
        return [p['bbox'] for p in plates[:3]]
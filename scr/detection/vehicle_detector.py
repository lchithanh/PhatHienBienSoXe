from ultralytics import YOLO
import cv2
import config

class VehicleDetector:
    def __init__(self, model_path=None):
        model_path = model_path or config.VEHICLE_MODEL_PATH
        if not model_path.exists():
            # Tự động tải model pretrained nếu chưa có
            self.model = YOLO("yolov8n.pt")
            self.model.save(str(model_path))
        else:
            self.model = YOLO(str(model_path))
        self.conf = config.VEHICLE_CONF
        self.iou = getattr(config, 'IOU_THRESHOLD', 0.45)
        self.classes = config.VEHICLE_CLASSES

    def detect(self, frame):
        """
        ✨ ENHANCED: Better vehicle detection with IOU + filtering
        Returns: list of {'bbox': (x1,y1,x2,y2), 'label': str, 'conf': float, 'area': int}
        """
        # ✨ NEW: Add IOU threshold for NMS
        results = self.model(
            frame, 
            conf=self.conf, 
            iou=self.iou,  # Post-process NMS IOU
            verbose=False
        )[0]
        
        vehicles = []
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            
            # Filter out unwanted classes
            if cls_id not in self.classes:
                continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # ✨ NEW: Ensure valid box
            if x1 >= x2 or y1 >= y2:
                continue
            
            # ✨ NEW: Filter small vehicles
            if config.FILTER_SMALL_VEHICLES:
                area = (x2 - x1) * (y2 - y1)
                if area < config.MIN_VEHICLE_AREA:
                    continue  # Skip too small
            
            vehicles.append({
                'bbox': (x1, y1, x2, y2),
                'label': self.classes[cls_id],
                'conf': float(box.conf[0]),
                'area': (x2 - x1) * (y2 - y1)  # ✨ NEW: Track area
            })
        
        # ✨ NEW: Sort by confidence (highest first)
        vehicles.sort(key=lambda v: v['conf'], reverse=True)
        
        return vehicles[:50]  # ✨ NEW: Limit to 50 vehicles max
from ultralytics import YOLO
import config


class VehicleDetector:
    def __init__(self, model_path=None):
        model_path = model_path or config.VEHICLE_MODEL_PATH
        if not model_path.exists():
            self.model = YOLO("yolov8n.pt")
            self.model.save(str(model_path))
        else:
            self.model = YOLO(str(model_path))

        self.conf = config.VEHICLE_CONF
        self.iou = getattr(config, 'IOU_THRESHOLD', 0.45)
        self.classes = config.VEHICLE_CLASSES

    def detect(self, frame):
        """
        Phát hiện xe trong frame.
        Trả về: list of {'bbox': (x1,y1,x2,y2), 'label': str, 'conf': float, 'area': int}
        """
        results = self.model(frame, conf=self.conf, iou=self.iou, verbose=False)[0]
        vehicles = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in self.classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if x1 >= x2 or y1 >= y2:
                continue

            area = (x2 - x1) * (y2 - y1)
            if config.FILTER_SMALL_VEHICLES and area < config.MIN_VEHICLE_AREA:
                continue

            vehicles.append({
                'bbox': (x1, y1, x2, y2),
                'label': self.classes[cls_id],
                'conf': float(box.conf[0]),
                'area': area,
            })

        vehicles.sort(key=lambda v: v['conf'], reverse=True)
        return vehicles[:50]
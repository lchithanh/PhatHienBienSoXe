from pathlib import Path
from ultralytics import YOLO
import config


class PlateDetector:
    def __init__(self, model_path=None):
        candidates = [
            Path(model_path) if model_path else None,
            config.PLATE_MODEL_PATH,
            Path('runs/detect/plate_detection_improved/final_v2/weights/best.pt'),
            Path('runs/detect/train/weights/best.pt'),
        ]

        chosen = next((p for p in candidates if p and p.exists()), None)
        if chosen is None:
            raise FileNotFoundError(
                f"Không tìm thấy model biển số. Đã kiểm tra: {candidates}"
            )

        self.model = YOLO(str(chosen))
        self.conf = config.PLATE_CONF
        self.iou = getattr(config, 'IOU_THRESHOLD', 0.45)

    def detect(self, crop_img):
        """
        Phát hiện biển số trong ảnh crop xe.
        Trả về: list of (x1, y1, x2, y2) — tối đa 3 biển, sắp xếp theo confidence.
        """
        if crop_img.size == 0:
            return []

        h, w = crop_img.shape[:2]
        if h < 30 or w < 30:
            return []

        results = self.model(crop_img, conf=self.conf, iou=self.iou, verbose=False)[0]
        plates = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if x1 >= x2 or y1 >= y2:
                continue

            plate_w = x2 - x1
            plate_h = y2 - y1

            if config.FILTER_SMALL_PLATES:
                if plate_w < config.MIN_PLATE_WIDTH:
                    continue
                if plate_w / max(plate_h, 1) > config.MAX_PLATE_ASPECT_RATIO:
                    continue

            plates.append({
                'bbox': (x1, y1, x2, y2),
                'conf': float(box.conf[0]),
            })

        plates.sort(key=lambda p: p['conf'], reverse=True)
        return [p['bbox'] for p in plates[:3]]
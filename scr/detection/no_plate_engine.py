import cv2
import config
from .vehicle_detector import VehicleDetector
from .plate_detector import PlateDetector
from scr.ocr.plate_reader import PlateReader


class NoPlateEngine:
    """Engine chính: phát hiện xe không có biển số."""

    def __init__(self, enable_ocr=True):
        self.vehicle_detector = VehicleDetector()
        self.plate_detector = PlateDetector()
        self.enable_ocr = enable_ocr
        self.plate_reader = PlateReader() if enable_ocr else None

    def process_frame(self, frame, logger=None):
        """
        Xử lý một frame, phát hiện xe và biển số.
        Trả về: (output_frame, alerts)
            - alerts: list xe không có biển số
        """
        h, w = frame.shape[:2]
        output = frame.copy()
        alerts = []

        for veh in self.vehicle_detector.detect(frame):
            x1, y1, x2, y2 = veh['bbox']

            # Mở rộng vùng crop để bắt biển số ở rìa xe
            px = int((x2 - x1) * config.EXPAND_RATIO)
            py = int((y2 - y1) * config.EXPAND_RATIO)
            cx1, cy1 = max(0, x1 - px), max(0, y1 - py)
            cx2, cy2 = min(w, x2 + px), min(h, y2 + py)
            crop = frame[cy1:cy2, cx1:cx2]

            if crop.size == 0:
                continue

            plate_boxes = self.plate_detector.detect(crop)
            has_plate = len(plate_boxes) > 0

            # OCR biển số đầu tiên
            plate_text = ""
            if has_plate and self.plate_reader:
                px1, py1, px2, py2 = plate_boxes[0]
                plate_crop = crop[py1:py2, px1:px2]
                if plate_crop.size > 0:
                    plate_text, _ = self.plate_reader.read(plate_crop)
                    cv2.putText(output, plate_text,
                                (cx1 + px1, cy1 + py1 - 5),
                                config.FONT, 0.6, config.COLOR_OCR, 2)

            # Vẽ khung xe
            color = config.COLOR_OK if has_plate else config.COLOR_ALERT
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            tag = f"{veh['label']} {veh['conf']:.0%}"
            if not has_plate:
                tag = "! KHONG BIEN SO - " + tag
            (tw, th), _ = cv2.getTextSize(tag, config.FONT, 0.55, 1)
            cv2.rectangle(output, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(output, tag, (x1 + 3, y1 - 5), config.FONT, 0.55, (255, 255, 255), 1)

            # Vẽ khung biển số
            for pb in plate_boxes:
                bx1, by1, bx2, by2 = pb
                cv2.rectangle(output,
                              (cx1 + bx1, cy1 + by1),
                              (cx1 + bx2, cy1 + by2),
                              config.COLOR_PLATE, 2)

            # Ghi nhận vi phạm
            if not has_plate:
                alert = {
                    'bbox': (x1, y1, x2, y2),
                    'label': veh['label'],
                    'conf': veh['conf'],
                    'crop': crop.copy(),
                    'frame': output.copy(),
                }
                alerts.append(alert)
                if logger:
                    logger.log(alert)

        # HUD tổng quan
        n = len(alerts)
        hud_color = config.COLOR_ALERT if n else config.COLOR_OK
        hud_text = f"CANH BAO: {n} xe khong bien so" if n else "Binh thuong"
        cv2.putText(output, hud_text, (10, 28), config.FONT, 0.7, hud_color, 2)

        return output, alerts
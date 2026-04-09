import cv2
import numpy as np
import config
from .vehicle_detector import VehicleDetector
from .plate_detector import PlateDetector
from scr.ocr.plate_reader import PlateReader   # optional

class NoPlateEngine:
    def __init__(self, enable_ocr=True):
        self.vehicle_detector = VehicleDetector()
        self.plate_detector = PlateDetector()
        self.enable_ocr = enable_ocr
        if enable_ocr:
            self.plate_reader = PlateReader()
        else:
            self.plate_reader = None

    def process_frame(self, frame, logger=None):
        h, w = frame.shape[:2]
        output = frame.copy()
        alerts = []   # xe không biển
        vehicles = self.vehicle_detector.detect(frame)

        for veh in vehicles:
            x1, y1, x2, y2 = veh['bbox']
            label = veh['label']
            conf_v = veh['conf']

            # Mở rộng vùng crop
            pad_x = int((x2 - x1) * config.EXPAND_RATIO)
            pad_y = int((y2 - y1) * config.EXPAND_RATIO)
            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(w, x2 + pad_x)
            cy2 = min(h, y2 + pad_y)
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            # Phát hiện biển số trong crop
            plate_boxes = self.plate_detector.detect(crop)
            has_plate = len(plate_boxes) > 0

            # Nếu có biển và bật OCR, đọc biển số
            plate_text = ""
            if has_plate and self.enable_ocr and self.plate_reader:
                # Lấy box biển đầu tiên (có thể xử lý nhiều biển)
                px1, py1, px2, py2 = plate_boxes[0]
                plate_crop = crop[py1:py2, px1:px2]
                if plate_crop.size > 0:
                    plate_text, ocr_conf = self.plate_reader.read(plate_crop)
                    # Vẽ text lên output (toạ độ tuyệt đối)
                    text_pos = (cx1 + px1, cy1 + py1 - 5)
                    cv2.putText(output, plate_text, text_pos,
                                config.FONT, 0.6, config.COLOR_OCR, 2)

            # Vẽ khung xe
            color = config.COLOR_OK if has_plate else config.COLOR_ALERT
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # Vẽ nhãn xe
            tag = f"{label} {conf_v:.0%}"
            if not has_plate:
                tag = "! KHONG BIEN SO - " + tag
            (tw, th), _ = cv2.getTextSize(tag, config.FONT, 0.55, 1)
            cv2.rectangle(output, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(output, tag, (x1 + 3, y1 - 5),
                        config.FONT, 0.55, (255, 255, 255), 1)

            # Vẽ khung biển số
            for pb in plate_boxes:
                px1, py1, px2, py2 = pb
                cv2.rectangle(output,
                              (cx1 + px1, cy1 + py1),
                              (cx1 + px2, cy1 + py2),
                              config.COLOR_PLATE, 2)

            # Ghi nhận vi phạm nếu không có biển
            if not has_plate:
                alert = {
                    'bbox': (x1, y1, x2, y2),
                    'label': label,
                    'conf': conf_v,
                    'crop': crop.copy(),
                    'frame': output.copy()
                }
                alerts.append(alert)
                if logger:
                    logger.log(alert)

        # HUD
        n_alert = len(alerts)
        hud_color = config.COLOR_ALERT if n_alert else config.COLOR_OK
        hud_text = f"CANH BAO: {n_alert} xe khong bien so" if n_alert else "Binh thuong"
        cv2.putText(output, hud_text, (10, 28),
                    config.FONT, 0.7, hud_color, 2)

        return output, alerts
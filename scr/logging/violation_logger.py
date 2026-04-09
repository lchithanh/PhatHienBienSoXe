import csv
import os
import time
import cv2
from datetime import datetime

import config


class ViolationLogger:
    """Ghi lại xe vi phạm (ảnh + CSV)."""

    def __init__(self):
        os.makedirs(config.VIOLATION_DIR, exist_ok=True)
        os.makedirs(config.FRAME_DIR, exist_ok=True)
        if config.ENABLE_OCR:
            os.makedirs(config.OCR_DIR, exist_ok=True)

        self._init_csv()
        self._last_log_time = 0.0
        self.total_violations = 0
        print(f"[LOGGER] Lưu log tại: {config.LOG_DIR}")

    def _init_csv(self):
        if not os.path.exists(config.CSV_PATH):
            with open(config.CSV_PATH, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'vehicle_type', 'confidence',
                    'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
                    'crop_path', 'frame_path',
                ])

    def log(self, alert: dict):
        """Ghi một vi phạm. Áp dụng cooldown để tránh log trùng."""
        now = time.time()
        if now - self._last_log_time < config.COOLDOWN_SEC:
            return
        self._last_log_time = now
        self.total_violations += 1

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        ts_display = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        crop_path = os.path.join(config.VIOLATION_DIR, f"vio_{ts}.jpg")
        cv2.imwrite(crop_path, alert['crop'])

        frame_path = ""
        if 'frame' in alert:
            frame_path = os.path.join(config.FRAME_DIR, f"frame_{ts}.jpg")
            cv2.imwrite(frame_path, alert['frame'])

        x1, y1, x2, y2 = alert['bbox']
        with open(config.CSV_PATH, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                ts_display, alert['label'], f"{alert['conf']:.3f}",
                x1, y1, x2, y2, crop_path, frame_path,
            ])

        print(f"[LOG #{self.total_violations}] {ts_display} | {alert['label']} "
              f"| conf={alert['conf']:.0%} | vio_{ts}.jpg")

    def summary(self):
        print(f"\n{'='*50}")
        print(f"  Tổng vi phạm: {self.total_violations}")
        print(f"  CSV: {config.CSV_PATH}")
        print(f"  Ảnh vi phạm: {config.VIOLATION_DIR}")
        print(f"{'='*50}\n")
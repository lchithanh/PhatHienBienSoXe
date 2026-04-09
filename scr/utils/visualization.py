"""Các hàm vẽ bounding box, label và HUD lên frame."""

import cv2
import numpy as np
import config


def draw_box_with_label(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                        label: str, color: tuple, font_scale: float = 0.55) -> np.ndarray:
    """Vẽ bounding box kèm label nền màu."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    (tw, th), _ = cv2.getTextSize(label, config.FONT, font_scale, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 5), config.FONT, font_scale, (255, 255, 255), 1)

    return frame


def draw_plate_box(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                   text: str = "") -> np.ndarray:
    """Vẽ khung biển số và text OCR (nếu có)."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), config.COLOR_PLATE, 2)

    if text:
        cv2.putText(frame, text, (x1, y1 - 5), config.FONT, 0.6, config.COLOR_OCR, 2)

    return frame


def draw_hud(frame: np.ndarray, n_violations: int) -> np.ndarray:
    """Vẽ HUD tổng quan góc trên trái."""
    color = config.COLOR_ALERT if n_violations else config.COLOR_OK
    text = f"CANH BAO: {n_violations} xe khong bien so" if n_violations else "Binh thuong"
    cv2.putText(frame, text, (10, 28), config.FONT, 0.7, color, 2)
    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Hiển thị FPS góc trên phải."""
    h, w = frame.shape[:2]
    text = f"FPS: {fps:.1f}"
    (tw, _), _ = cv2.getTextSize(text, config.FONT, 0.6, 1)
    cv2.putText(frame, text, (w - tw - 10, 28), config.FONT, 0.6, (200, 200, 200), 1)
    return frame
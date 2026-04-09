"""Các hàm tiện ích xử lý ảnh dùng chung trong dự án."""

import cv2
import numpy as np


def crop_with_padding(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                      pad_ratio: float = 0.0) -> tuple:
    """
    Crop một vùng trong frame, có thể mở rộng thêm padding.

    Trả về: (crop, cx1, cy1, cx2, cy2) — tọa độ tuyệt đối sau padding.
    """
    h, w = frame.shape[:2]
    px = int((x2 - x1) * pad_ratio)
    py = int((y2 - y1) * pad_ratio)
    cx1 = max(0, x1 - px)
    cy1 = max(0, y1 - py)
    cx2 = min(w, x2 + px)
    cy2 = min(h, y2 + py)
    return frame[cy1:cy2, cx1:cx2], cx1, cy1, cx2, cy2


def enhance_contrast(img: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Tăng độ tương phản bằng CLAHE."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(gray)


def resize_to_width(img: np.ndarray, target_width: int) -> np.ndarray:
    """Resize ảnh theo chiều rộng, giữ tỉ lệ."""
    h, w = img.shape[:2]
    if w == 0 or target_width <= 0:
        return img
    scale = target_width / w
    return cv2.resize(img, (target_width, int(h * scale)))


def is_valid_image(img: np.ndarray, min_size: int = 10) -> bool:
    """Kiểm tra ảnh có hợp lệ không (không rỗng, đủ kích thước)."""
    if img is None or img.size == 0:
        return False
    h, w = img.shape[:2]
    return h >= min_size and w >= min_size
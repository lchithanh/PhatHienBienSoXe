"""Các hàm tiện ích xử lý video."""

import cv2
from typing import Optional, Tuple


def open_source(source) -> cv2.VideoCapture:
    """
    Mở nguồn video (file, webcam, RTSP).
    source: str (đường dẫn/URL) hoặc int (webcam index).
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError(f"Không mở được nguồn video: {source}")
    return cap


def get_video_info(cap: cv2.VideoCapture) -> dict:
    """Lấy thông tin cơ bản của video."""
    return {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }


def create_writer(output_path: str, cap: cv2.VideoCapture,
                  fps: Optional[float] = None) -> cv2.VideoWriter:
    """Tạo VideoWriter để ghi output."""
    info = get_video_info(cap)
    fps = fps or info['fps'] or 25.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (info['width'], info['height']))


def read_frames(cap: cv2.VideoCapture):
    """Generator: lần lượt yield từng frame từ VideoCapture."""
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
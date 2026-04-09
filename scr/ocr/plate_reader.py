"""
PlateReader — đọc và xác thực biển số xe Việt Nam.

Hỗ trợ:
- PaddleOCR (ưu tiên, tốt hơn cho tiếng Việt)
- EasyOCR (fallback)

Xử lý các loại biển:
- Biển chuẩn:   29A-1234
- Biển cũ:      29-BN-001
- Biển ngoại giao: CC/CD + số
- Biển xin số:  XIN SỐ / XS / PENDING
"""

import re
import cv2
import numpy as np
from collections import Counter
from typing import Dict, Tuple, Optional

import config

# Chọn engine OCR
try:
    from paddleocr import PaddleOCR
    _USE_PADDLE = True
except ImportError:
    import easyocr
    _USE_PADDLE = False

# Mã tỉnh hợp lệ
VALID_PROVINCES = {
    '01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
    '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
    '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
    '31', '32', '33', '34', '35', '36', '37', '38', '39', '40',
    '41', '42', '43', '44', '45', '46', '47', '48', '49', '50',
    '51', '52', '99',
}

# Regex các dạng biển
_PATTERNS = {
    'standard':   re.compile(r'^\d{2}[A-Z]{1,2}\d{4,5}$'),
    'old_format': re.compile(r'^\d{2}-[A-Z]{2}-\d{3,5}$'),
    'diplomatic': re.compile(r'^[A-Z]{2}\d+$'),
    'temporary':  re.compile(r'(XIN\s*S[ÔO]|XS|PENDING)', re.IGNORECASE),
}

# Lỗi OCR thường gặp (chỉ áp dụng cho phần số phía sau chữ cái)
_OCR_CORRECTIONS = {'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '9'}


def _preprocess(plate_img: np.ndarray) -> np.ndarray:
    """Tiền xử lý ảnh biển số để tăng độ chính xác OCR."""
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY) if len(plate_img.shape) == 3 else plate_img

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    if config.OCR_RESIZE_WIDTH > 0:
        h, w = gray.shape
        gray = cv2.resize(gray, (config.OCR_RESIZE_WIDTH, int(h * config.OCR_RESIZE_WIDTH / w)))

    return gray


def _classify(text: str) -> str:
    """Phân loại biển số theo nội dung."""
    if not text:
        return 'EMPTY'
    if _PATTERNS['temporary'].search(text):
        return 'TEMPORARY'
    if _PATTERNS['standard'].match(text):
        return 'STANDARD'
    if _PATTERNS['old_format'].match(text):
        return 'OLD_FORMAT'
    if _PATTERNS['diplomatic'].match(text):
        return 'DIPLOMATIC'
    return 'UNKNOWN'


def _is_valid(text: str) -> bool:
    """Kiểm tra biển số có hợp lệ không."""
    if not text or len(text) < 5:
        return False
    has_digit = any(c.isdigit() for c in text)
    has_alpha = any(c.isalpha() for c in text)
    return has_digit and has_alpha


def _correct_text(text: str) -> str:
    """Sửa lỗi OCR phổ biến."""
    if _is_valid(text):
        return text
    corrected = text.translate(str.maketrans(_OCR_CORRECTIONS))
    return corrected if _is_valid(corrected) else text


class PlateReader:
    """Đọc biển số xe Việt Nam từ ảnh crop."""

    def __init__(self, use_gpu: bool = False):
        self.conf_thresh = getattr(config, 'OCR_CONF_THRESH', 0.45)
        self.use_gpu = use_gpu

        if _USE_PADDLE:
            self._ocr = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=use_gpu)
            self.engine = "PaddleOCR"
        else:
            self._ocr = easyocr.Reader(['en', 'vi'], gpu=use_gpu)
            self.engine = "EasyOCR"

    def read(self, plate_img: np.ndarray) -> Tuple[str, float]:
        """
        Đọc biển số từ ảnh.
        Trả về: (text, confidence)
        """
        if plate_img.size == 0:
            return "", 0.0

        processed = _preprocess(plate_img)

        if _USE_PADDLE:
            text, conf = self._read_paddle(processed)
        else:
            text, conf = self._read_easyocr(processed)

        text = _correct_text(text)
        return text, conf

    def read_with_info(self, plate_img: np.ndarray) -> Dict:
        """
        Đọc biển số và trả về thông tin đầy đủ.
        Trả về: {'text', 'confidence', 'type', 'is_valid', 'engine'}
        """
        text, conf = self.read(plate_img)
        plate_type = _classify(text)

        return {
            'text': text,
            'confidence': conf,
            'type': plate_type,
            'is_valid': _is_valid(text),
            'engine': self.engine,
        }

    def _read_paddle(self, processed: np.ndarray) -> Tuple[str, float]:
        img = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        result = self._ocr.ocr(img, cls=True)

        if not result or not result[0]:
            return "", 0.0

        texts, confs = [], []
        for line in result[0]:
            text, conf = line[1]
            if conf > self.conf_thresh:
                cleaned = re.sub(r'[^A-Z0-9]', '', text.strip().upper())
                if cleaned:
                    texts.append(cleaned)
                    confs.append(conf)

        return "".join(texts), float(np.mean(confs)) if confs else 0.0

    def _read_easyocr(self, processed: np.ndarray) -> Tuple[str, float]:
        rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        result = self._ocr.readtext(rgb, detail=1)

        if not result:
            return "", 0.0

        texts, confs = [], []
        for (_, text, conf) in result:
            if conf > self.conf_thresh:
                cleaned = re.sub(r'[^A-Z0-9]', '', text.strip().upper())
                if cleaned:
                    texts.append(cleaned)
                    confs.append(conf)

        return "".join(texts), float(np.mean(confs)) if confs else 0.0
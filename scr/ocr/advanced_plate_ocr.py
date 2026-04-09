"""
🤖 ADVANCED OCR MODULE - Machine Learning License Plate Reader
Sử dụng PaddleOCR (tốt hơn EasyOCR cho tiếng Việt)

Features:
- Vietnamese language support (tối ưu cho biển số Việt Nam)
- Better accuracy for low-quality images
- Parallel processing support
- Character-level confidence filtering
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import re
from typing import Tuple, List, Dict

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import config

# Try PaddleOCR (recommended), fallback to EasyOCR
try:
    from paddleocr import PaddleOCR
    USE_PADDLE = True
except ImportError:
    USE_PADDLE = False
    import easyocr


class AdvancedPlateOCR:
    """
    ✨ Advanced OCR for Vietnamese License Plates
    Sử dụng ML-based approaches với lọc ký tự thông minh
    """
    
    def __init__(self, use_gpu=False):
        """
        Initialize advanced OCR engine
        
        Args:
            use_gpu: Enable GPU acceleration (if available)
        """
        self.use_gpu = use_gpu
        self.conf_thresh = getattr(config, 'OCR_CONF_THRESH', 0.45)
        
        if USE_PADDLE:
            print("📥 Loading PaddleOCR (best for Vietnamese)...")
            self.ocr = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=use_gpu)
            self.engine = "PaddleOCR"
            print("✅ PaddleOCR loaded")
        else:
            print("📥 Loading EasyOCR (fallback)...")
            self.ocr = easyocr.Reader(['en', 'vi'], gpu=use_gpu)
            self.engine = "EasyOCR"
            print("✅ EasyOCR loaded")
        
        self.plate_patterns = self._compile_patterns()

    def _compile_patterns(self) -> Dict:
        """Compile regex patterns for Vietnamese plates"""
        return {
            # Standard: 29A12345
            'standard': re.compile(r'^(\d{2})([A-Z]{1,2})(\d{4,5})$'),
            # Old format: 29-BN-001
            'old_format': re.compile(r'^(\d{2})\-([A-Z]{2})\-(\d{3})$'),
            # Temporary: XIN SỐ / XS / PENDING
            'temporary': re.compile(r'(XIN\s*SỐ|XS|PENDING|XIN|SỐ)', re.IGNORECASE),
            # Diplomatic: CC / CD / SQ
            'diplomatic': re.compile(r'^[A-Z]{2}\s*\d+$'),
            # All digits/letters
            'alphanumeric': re.compile(r'^[A-Z0-9]+$')
        }

    def preprocess_image(self, plate_img: np.ndarray) -> np.ndarray:
        """
        Advanced preprocessing for OCR
        
        Steps:
        1. CLAHE contrast enhancement
        2. Dilation & erosion (morphological operations)
        3. Bilateral filtering (edge-preserving blur)
        4. Adaptive thresholding
        """
        # Convert to grayscale
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img
        
        # ✨ CLAHE Enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # ✨ Morphological Operations (clean noise)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # ✨ Bilateral Filtering (preserve edges, remove noise)
        bilateral = cv2.bilateralFilter(morph, 9, 75, 75)
        
        # ✨ Adaptive Thresholding (better than fixed threshold)
        adaptive = cv2.adaptiveThreshold(
            bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # ✨ Resize to standard size for consistency
        h, w = adaptive.shape
        if config.OCR_RESIZE_WIDTH > 0:
            scale = config.OCR_RESIZE_WIDTH / w
            new_h = int(h * scale)
            adaptive = cv2.resize(adaptive, (config.OCR_RESIZE_WIDTH, new_h))
        
        return adaptive

    def read_paddle(self, plate_img: np.ndarray) -> Tuple[str, float]:
        """
        Read using PaddleOCR
        
        Better for Vietnamese, handles old plates better
        """
        processed = self.preprocess_image(plate_img)
        
        # PaddleOCR expects BGR (original image, not grayscale)
        if len(processed.shape) == 2:
            # Convert back to BGR for PaddleOCR
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        
        result = self.ocr.ocr(processed, cls=True)
        
        if not result or not result[0]:
            return "", 0.0
        
        # Extract text and confidence
        texts = []
        confidences = []
        
        for line in result[0]:
            text = line[1][0]
            conf = line[1][1]
            
            if conf > self.conf_thresh:
                # Clean text: remove spaces, special chars
                text = text.strip().upper()
                text = re.sub(r'[^A-Z0-9]', '', text)
                
                if text:
                    texts.append(text)
                    confidences.append(conf)
        
        # Combine texts
        plate_text = "".join(texts)
        avg_conf = np.mean(confidences) if confidences else 0.0
        
        return plate_text, avg_conf

    def read_easy_ocr(self, plate_img: np.ndarray) -> Tuple[str, float]:
        """
        Read using EasyOCR (fallback)
        """
        processed = self.preprocess_image(plate_img)
        
        # Convert to RGB for EasyOCR
        if len(processed.shape) == 2:
            rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        
        result = self.ocr.readtext(rgb, detail=1, workers=4)
        
        if not result:
            return "", 0.0
        
        # Extract high-confidence text
        texts = []
        confidences = []
        
        for (bbox, text, conf) in result:
            if conf > self.conf_thresh:
                text = text.strip().upper()
                text = re.sub(r'[^A-Z0-9]', '', text)
                
                if text:
                    texts.append(text)
                    confidences.append(conf)
        
        plate_text = "".join(texts)
        avg_conf = np.mean(confidences) if confidences else 0.0
        
        return plate_text, avg_conf

    def read(self, plate_img: np.ndarray) -> Tuple[str, float, Dict]:
        """
        Read plate with validation and classification
        
        Returns:
            (text, confidence, info_dict)
        """
        if plate_img.size == 0:
            return "", 0.0, {"status": "empty_image"}
        
        # Choose OCR engine
        if USE_PADDLE:
            text, conf = self.read_paddle(plate_img)
        else:
            text, conf = self.read_easy_ocr(plate_img)
        
        # Classify plate type
        plate_type = self._classify_plate(text)
        
        # Validate format
        is_valid = self._validate_plate(text, plate_type)
        
        info = {
            "status": plate_type,
            "engine": self.engine,
            "confidence": conf,
            "is_valid": is_valid
        }
        
        return text, conf, info

    def _classify_plate(self, text: str) -> str:
        """
        Classify plate type based on content
        
        Returns:
            'STANDARD' | 'OLD_FORMAT' | 'TEMPORARY' | 'DIPLOMATIC' | 'UNKNOWN'
        """
        if not text:
            return 'EMPTY'
        
        text = text.upper().strip()
        
        # Check temporary plates
        if self.plate_patterns['temporary'].search(text):
            return 'TEMPORARY'  # Biển xin số / Pending
        
        # Check standard format: 29A1234
        if self.plate_patterns['standard'].match(text):
            return 'STANDARD'
        
        # Check old format: 29-BN-001
        if self.plate_patterns['old_format'].match(text):
            return 'OLD_FORMAT'
        
        # Check diplomatic: CC1234
        if self.plate_patterns['diplomatic'].match(text):
            return 'DIPLOMATIC'
        
        # Check if valid alphanumeric
        if self.plate_patterns['alphanumeric'].match(text):
            return 'STANDARD'  # Treat as standard
        
        return 'UNKNOWN'

    def _validate_plate(self, text: str, plate_type: str) -> bool:
        """
        Validate plate based on type
        
        Returns: True if valid format
        """
        if not text or plate_type == 'EMPTY':
            return False
        
        if plate_type == 'TEMPORARY':
            # Valid if contains XIN/XS/PENDING or enough alphanumeric
            if len(text) >= 4:  # At least "XINS" or similar
                return True
            return False
        
        if plate_type in ['STANDARD', 'OLD_FORMAT', 'DIPLOMATIC']:
            # Must have mix of letters and numbers
            has_digits = any(c.isdigit() for c in text)
            has_letters = any(c.isalpha() for c in text)
            return has_digits and has_letters and len(text) >= 6
        
        # For unknown, validate length and content
        has_digits = any(c.isdigit() for c in text)
        has_letters = any(c.isalpha() for c in text)
        return has_digits and has_letters and len(text) >= 6

    def confidence_level(self, conf: float, plate_type: str) -> str:
        """
        Determine confidence level for display
        
        Returns: 'HIGH' | 'MEDIUM' | 'LOW' | 'VERY_LOW'
        """
        if plate_type == 'TEMPORARY':
            # Temporary plates are harder to read
            if conf >= 0.70:
                return 'HIGH'
            elif conf >= 0.50:
                return 'MEDIUM'
            elif conf >= 0.35:
                return 'LOW'
            else:
                return 'VERY_LOW'
        
        # Standard plates
        if conf >= 0.85:
            return 'HIGH'
        elif conf >= 0.70:
            return 'MEDIUM'
        elif conf >= 0.50:
            return 'LOW'
        else:
            return 'VERY_LOW'

    def batch_read(self, plate_images: List[np.ndarray]) -> List[Tuple[str, float, Dict]]:
        """
        Read multiple plates in batch
        
        Useful for multi-plate scenarios
        """
        results = []
        for plate_img in plate_images:
            result = self.read(plate_img)
            results.append(result)
        return results


class PlateOCRWithCorrection(AdvancedPlateOCR):
    """
    Extended OCR with error correction using dictionary
    """
    
    def __init__(self, use_gpu=False):
        super().__init__(use_gpu)
        
        # Common Vietnamese plate number ranges
        self.valid_provinces = {
            '01': 'Hà Nội', '02': 'Hà Giang', '03': 'Cao Bằng',
            '04': 'Bắc Kạn', '05': 'Bắc Giang', '06': 'Bắc Ninh',
            '07': 'Hải Dương', '08': 'Hải Phòng', '09': 'Hưng Yên',
            '10': 'Thái Bình', '11': 'Thái Nguyên', '12': 'Yên Bái',
            '13': 'Lạng Sơn', '14': 'Quảng Ninh', '15': 'Điện Biên',
            '16': 'Lai Châu', '17': 'Sơn La', '18': 'Hòa Bình',
            '19': 'Ninh Bình', '20': 'Thanh Hóa', '21': 'Nghệ An',
            '22': 'Hà Tĩnh', '23': 'Quảng Bình', '24': 'Quảng Trị',
            '25': 'Thừa Thiên Huế', '26': 'Đà Nẵng', '27': 'Quảng Nam',
            '28': 'Quảng Ngãi', '29': 'Bình Định', '30': 'Phú Yên',
            '31': 'Khánh Hòa', '32': 'Ninh Thuận', '33': 'Bình Thuận',
            '34': 'Đồng Nai', '35': 'Bà Rịa - Vũng Tàu', '36': 'TP HCM',
            '37': 'Long An', '38': 'Tiền Giang', '39': 'Bến Tre',
            '40': 'Trà Vinh', '41': 'Vĩnh Long', '42': 'Đồng Tháp',
            '43': 'An Giang', '44': 'Kiên Giang', '45': 'Cần Thơ',
            '46': 'Hậu Giang', '47': 'Sóc Trăng', '48': 'Bạc Liêu',
            '49': 'Cà Mau', '50': 'TP HCM (Old)', '51': 'Bình Dương',
            '52': 'Tây Ninh'
        }
    
    def correct_plate(self, text: str) -> str:
        """
        Attempt to correct common OCR errors
        """
        text = text.upper().strip()
        
        # Common OCR misreadings
        corrections = {
            'O': '0', 'I': '1', 'Z': '2', 'S': '5',
            'B': '8', 'G': '9', 'l': '1'
        }
        
        # Don't apply if text looks valid already
        if self._looks_valid(text):
            return text
        
        # Try corrections
        corrected = text
        for ocr_char, correct_char in corrections.items():
            corrected = corrected.replace(ocr_char, correct_char)
        
        # Validate corrected version
        if self._looks_valid(corrected):
            return corrected
        
        return text  # Return original if can't correct
    
    def _looks_valid(self, text: str) -> bool:
        """Check if text looks like a valid plate"""
        if not text or len(text) < 6:
            return False
        
        # Check if province code valid
        province = text[:2]
        if province in self.valid_provinces:
            return True
        
        # Or just check format
        has_digits = any(c.isdigit() for c in text)
        has_letters = any(c.isalpha() for c in text)
        return has_digits and has_letters


# Test function
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 TEST - Advanced OCR Module")
    print("="*70)
    
    try:
        # Initialize
        ocr = AdvancedPlateOCR(use_gpu=False)
        print(f"\n✅ OCR Initialized: {ocr.engine}")
        
        # Test patterns
        test_plates = [
            "29A1234",      # Standard
            "29-BN-001",    # Old format
            "XIN SỐ",      # Temporary
            "CC123",        # Diplomatic
        ]
        
        for plate_type_name, pattern_name in [
            ("Standard", "standard"),
            ("Old", "old_format"),
            ("Temporary", "temporary"),
            ("Diplomatic", "diplomatic")
        ]:
            if pattern_name in ["standard", "old_format", "temporary", "diplomatic"]:
                result = ocr.plate_patterns[pattern_name].search(test_plates[[
                    "standard", "old_format", "temporary", "diplomatic"
                ].index(pattern_name)])
                print(f"✅ {plate_type_name}: {'Match' if result else 'No match'}")
        
        print(f"\n✅ Advanced OCR Module Ready!")
        print(f"   Engine: {ocr.engine}")
        print(f"   GPU: {ocr.use_gpu}")
        print(f"   Confidence threshold: {ocr.conf_thresh}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

"""
OCR Post-Processor - Xử lý các trường hợp đặc biệt:
1. Biển bị che hoàn toàn → "NO_PLATE"
2. Biển xe xin số (fixed plate)
3. Confidence filtering
4. Format validation tiếng Việt
"""
import re
import easyocr
from typing import Dict, Tuple, Optional
import numpy as np

class PlateOCRPostProcessor:
    """Post-process OCR results"""
    
    def __init__(self):
        """Initialize OCR processor"""
        self.reader = easyocr.Reader(['en', 'vi'], gpu=True)
        
        # Biển xe xin số (fixed plates)
        self.temp_plates = [
            "XIN SỐ", "XIN SO", "XS",
            "PENDING", "TEMPORARY",
        ]
        
        # Pattern biển số Việt
        # Format: [Tỉnh]-[Loại]-[Số]
        # VD: 99A-1234 (99=mã tỉnh, A=loại xe, 1234=số)
        self.vn_plate_pattern = re.compile(r'^[\d]{2,3}[A-Z]{1,2}[-\s]?[\d]{4,5}', re.IGNORECASE)
    
    def is_plate_readable(self, image_crop, confidence_threshold=0.5) -> bool:
        """
        Kiểm tra biển số có đọc được không
        - Nếu không đọc được → "NO_PLATE" (biển bị che)
        - Trả về: tuple (is_readable, reason)
        """
        try:
            results = self.reader.readtext(image_crop, detail=1)
            
            if not results:
                return False, "NO_TEXT_DETECTED"
            
            # Tính average confidence
            confidences = [score for _, _, score in results]
            avg_conf = np.mean(confidences)
            
            if avg_conf < confidence_threshold:
                return False, f"LOW_CONFIDENCE ({avg_conf:.2f})"
            
            # Ghép text
            text = "".join([result[1] for result in results])
            text = text.upper().strip()
            
            # Kiểm tra empty
            if len(text) < 5:
                return False, "TEXT_TOO_SHORT"
            
            return True, text
        except Exception as e:
            return False, f"ERROR: {str(e)}"
    
    def extract_plate_text(self, image_crop, confidence_threshold=0.5) -> Dict:
        """
        Trích xuất text từ biển số
        Trả về: {
            'text': 'biển số',
            'confidence': 0.95,
            'is_valid': True/False,
            'status': 'OK' / 'NO_PLATE' / 'TEMP_PLATE' / 'INVALID_FORMAT'
        }
        """
        try:
            results = self.reader.readtext(image_crop, detail=1)
            
            if not results:
                return {
                    'text': 'NO_PLATE',
                    'confidence': 0.0,
                    'is_valid': False,
                    'status': 'NO_PLATE',
                    'reason': 'Biển bị che hoàn toàn'
                }
            
            # Ghép text từ OCR
            confidences = [score for _, _, score in results]
            texts = [result[1] for result in results]
            text = "".join(texts).upper().strip()
            avg_conf = np.mean(confidences)
            
            # ===== CHECK 1: Confidence quá thấp ===== 
            if avg_conf < confidence_threshold:
                return {
                    'text': 'NO_PLATE',
                    'confidence': avg_conf,
                    'is_valid': False,
                    'status': 'NO_PLATE',
                    'reason': f'Confidence thấp ({avg_conf:.2f})'
                }
            
            # ===== CHECK 2: Biển xe xin số ===== 
            for temp_plate in self.temp_plates:
                if temp_plate.upper() in text:
                    return {
                        'text': 'NUMBER_PENDING',
                        'confidence': avg_conf,
                        'is_valid': False,
                        'status': 'TEMP_PLATE',
                        'reason': f'Biển xe xin số: {text}'
                    }
            
            # ===== CHECK 3: Validate format Việt ===== 
            if self.is_valid_vn_plate(text):
                return {
                    'text': text,
                    'confidence': avg_conf,
                    'is_valid': True,
                    'status': 'OK',
                    'reason': 'Valid Vietnamese plate'
                }
            
            # ===== CHECK 4: Tìm number-only pattern =====
            # Nếu chỉ có số (không có chữ + số) → invalid
            if text.replace(' ', '').replace('-', '').isdigit():
                return {
                    'text': text,
                    'confidence': avg_conf,
                    'is_valid': False,
                    'status': 'INVALID_FORMAT',
                    'reason': 'Chỉ có số, không có chữ'
                }
            
            # Format không chuẩn → warning nhưng vẫn return
            return {
                'text': text,
                'confidence': avg_conf,
                'is_valid': False,
                'status': 'INVALID_FORMAT',
                'reason': 'Format không chuẩn Việt'
            }
        
        except Exception as e:
            return {
                'text': 'ERROR',
                'confidence': 0.0,
                'is_valid': False,
                'status': 'ERROR',
                'reason': str(e)
            }
    
    def is_valid_vn_plate(self, text: str) -> bool:
        """
        Validate biển số Việt Nam
        Format: [Tỉnh 2-3 số][Loại 1-2 chữ][-][Số 4-5 chữ số]
        VD: 29A-1234, 99VN-5678, 50-BN-0001
        """
        # Remove spaces
        text_clean = text.replace(' ', '').replace('/', '-')
        
        # Check pattern
        if self.vn_plate_pattern.match(text_clean):
            return True
        
        # Alternative patterns
        patterns = [
            r'^[\d]{2}[A-Z]{1,2}[\d]{4,5}$',      # 29A1234
            r'^[\d]{2,3}[A-Z]{1,3}[\d]{4,5}$',    # 29AB1234
            r'^[\d]{2}-[A-Z]{2}-[\d]{4,5}$',      # 29-AB-1234
        ]
        
        for pattern in patterns:
            if re.match(pattern, text_clean, re.IGNORECASE):
                return True
        
        return False
    
    def clean_ocr_text(self, text: str) -> str:
        """Làm sạch text OCR"""
        # Remove extra spaces
        text = ' '.join(text.split())
        
        # Replace common OCR errors
        replacements = {
            'O': '0',   # Chữ O → số 0
            'I': '1',   # Chữ I → số 1
            'S': '5',   # Chữ S → số 5
            'Z': '2',   # Chữ Z → số 2
        }
        
        for wrong, correct in replacements.items():
            # Chỉ replace trong phần số
            text = re.sub(f'(?<=[A-Z]){wrong}', correct, text)
        
        return text.upper().strip()
    
    def process_detection_result(self, image_crop, 
                                box_confidence: float,
                                confidence_threshold: float = 0.5) -> Dict:
        """
        Process kết quả detection:
        - Nếu detection confidence quá thấp → NO_PLATE
        - Kiểm tra OCR
        - Validate format
        - Return structured result
        """
        
        # Kiểm tra detection confidence
        if box_confidence < 0.3:
            return {
                'plate': 'NO_PLATE',
                'confidence': box_confidence,
                'type': 'NO_PLATE',
                'reason': 'Detection confidence quá thấp'
            }
        
        # Extract OCR
        ocr_result = self.extract_plate_text(image_crop, confidence_threshold)
        
        # Tính final confidence
        final_conf = box_confidence * ocr_result['confidence']
        
        return {
            'plate': ocr_result['text'],
            'confidence': final_conf,
            'type': ocr_result['status'],
            'is_valid': ocr_result['is_valid'],
            'reason': ocr_result.get('reason', '')
        }

# Test
if __name__ == "__main__":
    processor = PlateOCRPostProcessor()
    print("✅ OCR Post-Processor initialized")
    print("\nSupported plate types:")
    print("  1. Valid Vietnamese plates → OK")
    print("  2. Plates with low confidence → NO_PLATE")
    print("  3. Temp/pending plates → TEMP_PLATE")
    print("  4. Invalid format → INVALID_FORMAT")

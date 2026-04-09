import easyocr
import cv2
import numpy as np
import re
import config

class PlateReader:
    def __init__(self, lang_list=['en', 'vi'], gpu=False):
        """
        ✨ ENHANCED: OCR reader with GPU support
        
        Args:
            lang_list: Languages to recognize (English + Vietnamese)
            gpu: Enable GPU acceleration if available
        """
        self.reader = easyocr.Reader(lang_list, gpu=gpu)
        self.conf_thresh = config.OCR_CONF_THRESH

    def preprocess_image(self, plate_img):
        """
        ✨ NEW: Preprocess plate image for better OCR
        
        Techniques:
        - Convert to grayscale
        - Enhance contrast with CLAHE
        - Resize for consistency
        """
        # Convert to grayscale if needed
        if config.OCR_GRAYSCALE:
            if len(plate_img.shape) == 3:
                gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = plate_img
        else:
            gray = plate_img
        
        # ✨ NEW: Enhance contrast using CLAHE
        if config.OCR_ENHANCE:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        
        # ✨ NEW: Resize to standard width
        h, w = gray.shape[:2]
        if config.OCR_RESIZE_WIDTH > 0:
            scale = config.OCR_RESIZE_WIDTH / w
            new_h = int(h * scale)
            gray = cv2.resize(gray, (config.OCR_RESIZE_WIDTH, new_h))
        
        return gray

    def read(self, plate_img):
        """
        ✨ ENHANCED: OCR with preprocessing and validation
        
        Returns: (text, confidence)
        """
        # ✨ NEW: Preprocess image
        processed = self.preprocess_image(plate_img)
        
        # ✨ NEW: Convert to RGB for EasyOCR
        if len(processed.shape) == 2:
            rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        
        # ✨ NEW: OCR with detailed results
        results = self.reader.readtext(rgb, detail=1)
        
        if not results:
            return "", 0.0
        
        # ✨ NEW: Enhanced aggregation
        best_text = ""
        best_conf = 0.0
        all_texts = []
        
        for (bbox, text, conf) in results:
            # Only keep high-confidence characters
            if conf > self.conf_thresh:
                all_texts.append(text)
                if conf > best_conf:
                    best_conf = conf
        
        # ✨ NEW: Join all texts
        best_text = "".join(all_texts)
        
        # ✨ NEW: Validate plate format
        best_text = self.validate_plate_format(best_text)
        
        return best_text, best_conf

    def validate_plate_format(self, text):
        """
        ✨ NEW: Validate and clean OCR text
        
        Vietnamese plates format: 
        - Pattern: XX[A-Z]{1,2}[0-9]{4} (e.g., 29A1234)
        """
        # Remove spaces
        text = text.replace(" ", "").upper()
        
        # Vietnamese plate pattern: 2-3 digits, 1-2 letters, 4-5 digits
        pattern = r'^(\d{2,3})([A-Z]{1,2})(\d{4,5})$'
        
        match = re.match(pattern, text)
        if match:
            return text
        
        # If no match, try to salvage by removing invalid chars
        # Keep only digits and letters
        cleaned = "".join(c for c in text if c.isalnum())
        
        # Check if it looks like a plate (has mix of letters and numbers)
        has_digits = any(c.isdigit() for c in cleaned)
        has_letters = any(c.isalpha() for c in cleaned)
        
        if has_digits and has_letters and len(cleaned) >= 6:
            return cleaned
        
        return text  # Return as-is if can't validate
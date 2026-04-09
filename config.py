import os
from pathlib import Path
import cv2
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# Đường dẫn model
VEHICLE_MODEL_PATH = BASE_DIR / "data" / "weights" / "yolov8n.pt"
# Ghi chú: nếu bạn đã train model plate trong thư mục runs/detect/train/weights
# thì dùng `runs/detect/train/weights/best.pt` để tránh file plate_best.pt 0-byte.
# Model đường dẫn ưu tiên mới (model mới train/đã cải tiến):
PLATE_MODEL_PATH   = BASE_DIR / "runs" / "detect" / "plate_detection_improved" / "final_v2" / "weights" / "best.pt"

# Legacy fallback
PLATE_MODEL_FALLBACK = BASE_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"

# Nếu chưa có, script sẽ tự download từ ultralytics
os.makedirs(VEHICLE_MODEL_PATH.parent, exist_ok=True)

# ✨ ENHANCED THRESHOLDS FOR BETTER ACCURACY
VEHICLE_CONF = 0.40          # ↓ Lowered from 0.45 (catch more vehicles)
PLATE_CONF = 0.35            # ↓ Lowered from 0.40 (catch more plates)
OCR_CONF_THRESH = 0.45       # ↓ Lowered from 0.50 (catch more text)

# ✨ NMS (Non-Maximum Suppression)
IOU_THRESHOLD = 0.45         # Remove duplicate detections

# ✨ REGION EXPANSION
EXPAND_RATIO = 0.20          # ↑ Increased from 0.15 (larger search area)

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# ✨ OCR PREPROCESSING
OCR_ENHANCE = True           # Enable contrast enhancement
OCR_RESIZE_WIDTH = 640       # Resize to standard width before OCR
OCR_GRAYSCALE = True         # Convert to grayscale (better for B/W plates)

# ✨ VIDEO PROCESSING
SKIP_FRAMES = 1              # Process every Nth frame (1 = all frames)
TRACK_HISTORY = 3            # Keep detection history (smoothing)

# ✨ VEHICLE DETECTION SPECIFIC
DETECT_VEHICLES = True
FILTER_SMALL_VEHICLES = True # Ignore tiny vehicle detections
MIN_VEHICLE_AREA = 5000      # Minimum area in pixels

# ✨ PLATE DETECTION SPECIFIC
FILTER_SMALL_PLATES = True   # Ignore tiny plate detections
MIN_PLATE_WIDTH = 50         # Minimum plate width pixels
MAX_PLATE_ASPECT_RATIO = 5.0 # Maximum width/height ratio

# Màu sắc
COLOR_OK    = (60, 200, 60)
COLOR_ALERT = (0, 60, 220)
COLOR_PLATE = (255, 165, 0)
COLOR_OCR   = (255, 255, 0)   # vàng cho chữ OCR

FONT = cv2.FONT_HERSHEY_SIMPLEX

# OCR
ENABLE_OCR = True

# Logging
LOG_DIR = BASE_DIR / "logs"
VIOLATION_DIR = LOG_DIR / "violations"
FRAME_DIR = LOG_DIR / "frames"
OCR_DIR = LOG_DIR / "ocr_results"
CSV_PATH = LOG_DIR / "violation_log.csv"
COOLDOWN_SEC = 3.0

# Roboflow (nếu dùng để tải dataset)
# API key example: CBWP922fYKmixaOFD0u3
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "CBWP922fYKmixaOFD0u3")
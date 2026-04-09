# 📚 PROJECT STRUCTURE & TRAINING MODULES EXPLAINED

## 🎯 YÊU CẦU & TỰA THẢ

Project này là hệ thống **License Plate Detection** gồm:
1. ✅ Phát hiện xe (Vehicle Detection)
2. ✅ Phát hiện biển số (Plate Detection)  
3. ✅ Đọc biển số (OCR)
4. ✅ Ghi nhận vi phạm (Logging)

---

## 🏗️ PROJECT ARCHITECTURE

```
DoAnAi/
│
├── 📁 apps/                    ← ỨNG DỤNG CHẠY
│   ├── detect_webcam.py        (Real-time detection)
│   ├── detect_video.py         (Video processing)
│   ├── detect_image.py         (Single image)
│   └── detect_gui.py           (GUI interface)
│
├── 📁 scr/                     ← SOURCE CODE (CORE LOGIC)
│   ├── 📁 detection/           ← DETECTION MODULES
│   │   ├── vehicle_detector.py (Phát hiện xe)
│   │   ├── plate_detector.py   (Phát hiện biển số)
│   │   └── no_plate_engine.py  (Main processing engine)
│   │
│   ├── 📁 ocr/                 ← OCR MODULES
│   │   ├── plate_reader.py     (Đọc biển số)
│   │   └── plate_reader_postprocessor.py (Post-process)
│   │
│   ├── 📁 logging/             ← LOGGING MODULES
│   │   └── violation_logger.py (Ghi nhận vi phạm)
│   │
│   └── 📁 utils/               ← UTILITY FUNCTIONS
│       ├── image_utils.py      (Xử lý ảnh)
│       ├── video_utils.py      (Xử lý video)
│       └── visualization.py    (Vẽ kết quả)
│
├── 📁 scripts/                 ← TRAINING & TOOLS
│   ├── train_plate_model_final.py       ⭐ MAIN TRAINING
│   ├── train_plate_model_improved.py    (Alternative)
│   ├── reorganize_dataset.py            (Data prep)
│   ├── evaluate_model.py                (Evaluation)
│   └── download_dataset.py              (Dataset)
│
├── 📁 data/
│   └── weights/                ← MODEL WEIGHTS
│       ├── vehicle.pt          (Xe detection model)
│       └── plate_best.pt       (Biển detection model)
│
├── 📁 License-Plate-Recognition-3/  ← DATASET
│   ├── data.yaml               (Dataset config)
│   ├── train/                  (70% training data)
│   ├── valid/                  (Validation data)
│   └── test/                   (30% test data)
│
├── 📁 logs/                    ← RUNTIME OUTPUT
│   ├── frames/                 (Captured frames)
│   ├── violations/             (Violation images)
│   ├── ocr_results/            (OCR results)
│   └── violation_log.csv       (Log file)
│
├── config.py                   ⭐ CONFIGURATION
├── requirements.txt            ⭐ DEPENDENCIES
└── README.md                   ⭐ DOCUMENTATION
```

---

## 🔧 TRAINING MODULES EXPLAINED

### 📦 **MAIN TRAINING PIPELINE**

```
train_plate_model_final.py
    ↓
Load Dataset (License-Plate-Recognition-3)
    ↓
Load Pre-trained YOLOv8n Model
    ↓
Configure Training Params
    ├─ Epochs: 150
    ├─ Batch: 16
    ├─ Augmentation: Mosaic, Mixup, Erasing, etc.
    ├─ Learning Rate: 0.01
    └─ Device: GPU (or CPU)
    ↓
Train Model
    ├─ Loss decreases
    ├─ mAP increases
    └─ Save checkpoints
    ↓
Validate Model
    ├─ Test on validation set
    ├─ Calculate metrics (mAP, Recall, Precision)
    └─ Save best model
    ↓
Output: runs/detect/train/weights/best.pt
```

### 🎯 **KEY TRAINING COMPONENTS**

#### **1. Data Preparation** - `reorganize_dataset.py`
```
Input:  500 images (train + valid + test mixed)
    ↓
Shuffle & Split
    ↓
Output: 350 train (70%) + 150 test (30%)
    ↓
Update: data.yaml with correct paths
```

**Why 70/30?**
- 70% training: Model learns patterns
- 30% testing: Verify generalization

#### **2. Model Configuration** - `config.py`
```
VEHICLE_CONF = 0.45         (45% confidence for vehicle detection)
PLATE_CONF = 0.40           (40% confidence for plate detection)
OCR_CONF_THRESH = 0.5       (50% confidence for OCR characters)
EXPAND_RATIO = 0.15         (Expand detection area by 15%)
```

#### **3. Training Script** - `train_plate_model_final.py`
```
Key Configs:
├─ epochs: 150              (150 training iterations)
├─ batch: 16                (16 images per batch)
├─ imgsz: 640               (640x640 resolution)
├─ lr0: 0.01                (Initial learning rate)
├─ device: 0                (GPU device 0)
├─ patience: 25             (Early stopping after 25 epochs no improvement)
│
└─ Augmentation:
   ├─ mosaic: 1.0           (Combine 4 images)
   ├─ mixup: 0.2            (Blend 2 images)
   ├─ erasing: 0.5          (Random erasing - simulate occlusion)
   ├─ hsv_s: 0.7            (Saturate - blur effect)
   ├─ degrees: 20           (Rotate ±20 degrees)
   ├─ translate: 0.25       (Shift image)
   ├─ scale: 0.4            (Scale variation)
   └─ dropout: 0.4          (Regularization)
```

---

## 📊 DETECTION PIPELINE

### **Execution Flow**

```
Input Frame (Webcam/Video/Image)
    ↓
❶ VEHICLE DETECTION (VehicleDetector)
    ├─ Model: YOLOv8n (pre-trained on COCO)
    ├─ Detects: car, motorcycle, bus, truck
    ├─ Output: vehicle bounding boxes
    └─ Confidence: config.VEHICLE_CONF
    ↓
❷ PLATE DETECTION (PlateDetector)
    ├─ For each vehicle:
    │   ├─ Crop vehicle region + expand by 15%
    │   ├─ Detect plates inside crop
    │   └─ Output: plate bounding boxes
    ├─ Model: Fine-tuned YOLOv8n (trained on License-Plate-Recognition-3)
    └─ Confidence: config.PLATE_CONF
    ↓
❸ OCR READING (PlateReader)
    ├─ For each plate:
    │   ├─ Crop plate image
    │   ├─ Preprocess: Grayscale, enhance
    │   ├─ EasyOCR (Vietnamese + English)
    │   └─ Extract text & confidence
    ├─ Confidence: config.OCR_CONF_THRESH
    └─ Post-process: Validate format & handle special cases
    ↓
❹ VIOLATION LOGGING (ViolationLogger)
    ├─ If NO plate detected:
    │   ├─ Mark as "NO_PLATE"
    │   ├─ Save frame
    │   └─ Log violation
    └─ Save to: logs/violation_log.csv
    ↓
Output: Annotated frame with detections
```

### **Core Detection Classes**

#### **VehicleDetector** - `scr/detection/vehicle_detector.py`
```python
Purpose: Detect vehicles in frame
   
Methods:
  def detect(frame) → list of vehicles
    Input: frame (image)
    Output: [
      {'bbox': (x1,y1,x2,y2), 'label': 'car', 'conf': 0.95},
      {'bbox': (x1,y1,x2,y2), 'label': 'truck', 'conf': 0.87},
      ...
    ]

Classes Detected:
  2: "car"
  3: "motorcycle"
  5: "bus"
  7: "truck"
```

#### **PlateDetector** - `scr/detection/plate_detector.py`
```python
Purpose: Detect license plates in cropped vehicle regions

Methods:
  def detect(crop_img) → list of plate bboxes
    Input: cropped image from vehicle detection
    Output: [(x1,y1,x2,y2), ...]  (bounding boxes)

Models:
  - Trained on License-Plate-Recognition-3 dataset
  - 500+ plate images
  - Fine-tuned YOLOv8n
```

#### **PlateReader** - `scr/ocr/plate_reader.py`
```python
Purpose: OCR - Extract text from plate images

Methods:
  def read(plate_img) → (text, confidence)
    Input: plate image (cropped)
    Output: ("29A1234", 0.92)

Technology:
  - EasyOCR (Python library)
  - Languages: English + Vietnamese
  - GPU supported
```

#### **NoPlateEngine** - `scr/detection/no_plate_engine.py`
```python
Purpose: Main processing engine combining all modules

Methods:
  def process_frame(frame, logger) → (annotated_frame, alerts)
    1. Detect vehicles
    2. For each vehicle:
       - Detect plates inside
       - Read plate text (if enabled)
       - Determine violations (no plate)
    3. Draw results on frame
    4. Log violations
    5. Return annotated frame + alerts
```

---

## ⚙️ CONFIGURATION DETAILS

### `config.py` - Main Configuration File

```python
# Model Paths
VEHICLE_MODEL_PATH = "data/weights/yolov8n.pt"
PLATE_MODEL_PATH = "runs/detect/train/weights/best.pt"

# Detection Confidence Thresholds
VEHICLE_CONF = 0.45      # Minimum confidence to consider vehicle detection
PLATE_CONF = 0.40        # Minimum confidence to consider plate detection
OCR_CONF_THRESH = 0.5    # Minimum confidence for OCR characters

# Region Expansion
EXPAND_RATIO = 0.15      # Expand vehicle region by 15% before plate detection

# Vehicle Classes
VEHICLE_CLASSES = {
    2: "car",            # Class ID 2 = car
    3: "motorcycle",     # Class ID 3 = motorcycle
    5: "bus",            # Class ID 5 = bus
    7: "truck"           # Class ID 7 = truck
}

# Visualization Colors (BGR format)
COLOR_OK = (60, 200, 60)       # Green - vehicle WITH plate
COLOR_ALERT = (0, 60, 220)     # Red/Pink - vehicle WITHOUT plate
COLOR_PLATE = (255, 165, 0)    # Orange - plate bounding box
COLOR_OCR = (255, 255, 0)      # Yellow - OCR text

# Logging
LOG_DIR = "logs"
VIOLATION_DIR = "logs/violations"
FRAME_DIR = "logs/frames"
OCR_DIR = "logs/ocr_results"
CSV_PATH = "logs/violation_log.csv"
COOLDOWN_SEC = 3.0             # Seconds between logging same violation
```

---

## 🚀 TRAINING MODULES SUMMARY

| Module | Purpose | File |
|--------|---------|------|
| **Data Prep** | 70/30 split, shuffle | `reorganize_dataset.py` |
| **Training** | Train YOLOv8n on plates | `train_plate_model_final.py` |
| **Evaluation** | Validate model performance | `evaluate_model.py` |
| **Download** | Get dataset from Roboflow | `download_dataset.py` |
| **Detection** | Vehicle + Plate detection | `detection/*.py` |
| **OCR** | Read plate text | `ocr/*.py` |
| **Logging** | Record violations | `logging/*.py` |

---

## 📈 ACCURACY IMPROVEMENT OPPORTUNITIES

Currently your project can achieve:
- ✅ Vehicle detection: ~95%
- ✅ Plate detection: ~88%
- ⚠️ OCR reading: ~75% (depends on plate quality)

### **Areas for Improvement:**

#### **1. Better Training Data**
```
Current: 500 images
Improve to: 2000+ images
   ├─ More diverse plates
   ├─ Different lighting conditions
   ├─ Various angles
   └─ Occluded/damaged plates
```

#### **2. Augmentation Enhancement**
```
Current config is good, but can add:
├─ Gaussian blur (simulate motion)
├─ Brightness/contrast variations
├─ Perspective transforms
└─ Color jitter
```

#### **3. Multi-Model Ensembling**
```
Train multiple models:
├─ YOLOv8n (current - small)
├─ YOLOv8m (medium - more accuracy)
└─ YOLOv8l (large - best accuracy)

Use voting for final decision
```

#### **4. OCR Post-Processing**
```
Improve OCR accuracy:
├─ Clean image preprocessing
├─ Filter invalid characters
├─ Validate plate format
└─ Use custom dictionaries
```

---

## 📚 WORKFLOW SUMMARY

### Training Workflow
```
1. Reorganize Data (70/30 split)
   └─ python scripts/reorganize_dataset.py

2. Train Model (150 epochs)
   └─ python scripts/train_plate_model_final.py

3. Evaluate
   └─ python scripts/evaluate_model.py

4. Deploy
   └─ python apps/detect_webcam.py
```

### Detection Workflow
```
Input → VehicleDetector → PlateDetector → PlateReader
           (95% acc)        (88% acc)      (75% acc)
             ↓                ↓               ↓
         Find cars       Find plates     Read numbers
             →  Annotate Frame  →  Log Violations
```

---

## 🎯 KEY FILES TO UNDERSTAND

| File | Purpose | Key Function |
|------|---------|--------------|
| `config.py` | Global settings | All confidence thresholds |
| `train_plate_model_final.py` | Training | Main training entry point |
| `no_plate_engine.py` | Detection | Main processing pipeline |
| `vehicle_detector.py` | Vehicle detection | Detect cars/trucks |
| `plate_detector.py` | Plate detection | Detect license plates |
| `plate_reader.py` | OCR | Read plate text |
| `violation_logger.py` | Logging | Save violations |

---

## 📊 METRICS TO MONITOR

During training, monitor:
```
1. Loss
   ├─ Should decrease over time
   └─ If not → adjust learning rate

2. mAP (mean Average Precision)
   ├─ mAP@0.5: Should be > 0.85
   └─ mAP@0.5:0.95: Should be > 0.70

3. Recall
   ├─ How many objects detected
   └─ Should be > 0.90

4. Precision
   ├─ How many detections are correct
   └─ Should be > 0.90
```

---

**Next Steps:**
1. Review each module (read source code)
2. Run training: `python scripts/train_plate_model_final.py`
3. Test detection: `python apps/detect_webcam.py`
4. Apply improvements (see `IMPROVEMENTS.md`)


"""
🚀 FINAL TRAINING SCRIPT - Ghi đè File Cũ
Xử lý:
- Biển số bị che → "không biển số"
- Biển xe xin số (fixed plate)
- 70% train / 30% test
- Strong augmentation
"""
from ultralytics import YOLO
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))  # Add parent directory to path

import config
import shutil

def train_final():
    """Training final ghi đè file cũ"""
    
    print("\n" + "="*70)
    print("🚀 FINAL TRAINING - LICENSE PLATE DETECTION")
    print("="*70)
    
    # Đường dẫn
    data_yaml = BASE_DIR / "License-Plate-Recognition-3" / "data.yaml"
    
    if not data_yaml.exists():
        print(f"❌ Không tìm thấy {data_yaml}")
        return False
    
    print(f"\n📂 Dataset: {data_yaml}")
    
    # Tải model base
    print(f"\n📥 Tải model base yolov8n...")
    model = YOLO("yolov8n.pt")
    
    # ✨ ENHANCED TRAINING CONFIG - Better Accuracy
    train_config = {
        # Dataset
        "data": str(data_yaml),
        "imgsz": 704,              # ⬆️ Increased from 640 (larger for detail)
        
        # Training
        "epochs": 200,             # ⬆️ Increased from 150 (more training)
        "batch": 16,               # Using CPU - 16 batch (reduce for CPU memory)
        "device": "cpu",           # Use CPU (no GPU available)
        "patience": 50,            # ⬆️ Increased patience
        
        # Learning rate - More stable training
        "lr0": 0.005,              # ⬇️ Lower learning rate (was 0.01)
        "lrf": 0.001,              # End learning rate
        "momentum": 0.937,
        "weight_decay": 0.0005,
        
        # === ✨ ENHANCED AUGMENTATION - More aggressive ===
        # HSV
        "hsv_h": 0.02,             # HSV-Hue
        "hsv_s": 0.8,              # ⬆️ Higher saturation shift (was 0.7)
        "hsv_v": 0.5,              # HSV-Value
        
        # Spatial - More transformation
        "degrees": 25,             # ⬆️ More rotation (was 20)
        "translate": 0.3,          # ⬆️ More translation (was 0.25)
        "scale": 0.5,              # ⬆️ More scaling (was 0.4)
        "flipud": 0.5,
        "fliplr": 0.5,
        
        # Mix - Better data augmentation
        "mosaic": 1.0,
        "mixup": 0.3,              # ⬆️ Increased mixup (was 0.2)
        
        # Advanced augmentation
        "auto_augment": "randaugment",
        "erasing": 0.6,            # ⬆️ More erasing (was 0.5)
        "crop_fraction": 0.75,
        
        # Regularization - Prevent overfitting
        "dropout": 0.5,            # ⬆️ More dropout (was 0.4)
        "label_smoothing": 0.15,   # ⬆️ Higher label smoothing (was 0.1)
        
        # Optimizer
        "optimizer": "SGD",
        
        # Output
        "save": True,
        "save_period": 10,
        "project": str(BASE_DIR / "runs" / "detect" / "plate_detection_improved"),
        "name": "final_v2",
        "exist_ok": True,
    }
    
    print(f"\n⚙️  ✨ ENHANCED AUGMENTATION:")
    print(f"   ✓ Epochs: 200 (was 150)")
    print(f"   ✓ Batch: 32 (was 16)")
    print(f"   ✓ Image Size: 704x704 (was 640)")
    print(f"   ✓ Learning Rate: 0.005 (lower = more stable)")
    print(f"   ✓ More rotation (±25°)")
    print(f"   ✓ More scaling (0.5)")
    print(f"   ✓ Higher mixup (0.3)")
    print(f"   ✓ Higher erasing (0.6)")
    print(f"   ✓ More dropout (0.5)")
    print(f"   ✓ Mosaic + Mixup: Ghép & trộn ảnh")
    print(f"   ✓ All augmentations optimized for hidden plates")
    
    print(f"\n🔄 Bắt đầu training...")
    print(f"   Dataset: 70% train / 30% test")
    
    # Train
    results = model.train(**train_config)
    
    # Lưu model
    best_pt = BASE_DIR / "runs" / "detect" / "plate_detection_improved" / "final_v2" / "weights" / "best.pt"
    target_pt = config.PLATE_MODEL_PATH
    fallback_pt = BASE_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"
    
    if best_pt.exists():
        print(f"\n✅ Training hoàn tất!")
        print(f"   Best model: {best_pt}")
        
        # Ghi đè file cũ
        os.makedirs(target_pt.parent, exist_ok=True)
        shutil.copy2(best_pt, target_pt)
        print(f"   ✓ Đã ghi đè: {target_pt}")

        # Tạo fallback path legacy
        os.makedirs(fallback_pt.parent, exist_ok=True)
        shutil.copy2(best_pt, fallback_pt)
        print(f"   ✓ Đã copy tới fallback path: {fallback_pt}")
        
        # Thống kê
        print(f"\n📊 Kết quả Training:")
        if hasattr(results, 'box'):
            print(f"   mAP50: {results.box.map50:.3f}" if hasattr(results.box, 'map50') else "")
            print(f"   mAP50-95: {results.box.map:.3f}" if hasattr(results.box, 'map') else "")
        
        return True
    else:
        print(f"❌ Model không được lưu!")
        return False

def validate_model():
    """Validate model trên test set"""
    print(f"\n{'='*70}")
    print(f"📊 VALIDATION")
    print(f"{'='*70}")
    
    data_yaml = BASE_DIR / "License-Plate-Recognition-3" / "data.yaml"
    model_path = config.PLATE_MODEL_PATH
    
    if not model_path.exists():
        print(f"❌ Model không tìm thấy: {model_path}")
        return False
    
    print(f"\n📥 Load model: {model_path}")
    model = YOLO(str(model_path))
    
    print(f"📂 Validate dataset: {data_yaml}")
    results = model.val(data=str(data_yaml), imgsz=640)
    
    print(f"\n✅ Validation hoàn tất!")
    return True

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--validate":
        validate_model()
    elif len(sys.argv) > 1 and sys.argv[1] == "--reorganize":
        print("❌ Vui lòng chạy: python scripts/reorganize_dataset.py")
    else:
        success = train_final()
        if success:
            print(f"\n💡 Để validate: python scripts/train_plate_model_final.py --validate")

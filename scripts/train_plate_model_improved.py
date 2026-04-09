"""
Script training cải tiến cho phát hiện biển số
Hỗ trợ các trường hợp: biển số bị che, mờ, bị che một phần
"""
from ultralytics import YOLO
import os
from pathlib import Path
import config

def train_plate_detection():
    """
    Training model phát hiện biển số với augmentation cho trường hợp:
    - Biển số bị che/mờ
    - Biển số bị xoay
    - Biển số bị che một phần
    """
    
    # Tải model base
    model = YOLO("yolov8n.pt")
    
    # Đường dẫn tới data.yaml
    data_yaml = os.path.join(
        os.path.dirname(__file__), 
        "..", 
        "License-Plate-Recognition-3", 
        "data.yaml"
    )
    
    if not os.path.exists(data_yaml):
        print(f"❌ Không tìm thấy {data_yaml}")
        return False
    
    print(f"📂 Dùng dataset: {data_yaml}")
    
    # Cấu hình training với augmentation mạnh cho trường hợp biển số bị che
    train_config = {
        "data": data_yaml,
        "epochs": 100,           # Tăng từ 50 lên 100 để học tốt hơn
        "imgsz": 640,
        "batch": 16,
        "device": 0,             # GPU device 0 (hoặc 'cpu' nếu không có GPU)
        "patience": 20,          # Early stopping
        
        # ===== AUGMENTATION MẠNH cho xử lý biển số bị che =====
        "hsv_h": 0.015,          # HSV-Hue augmentation
        "hsv_s": 0.7,            # HSV-Saturation (che bớt biển địa phương)
        "hsv_v": 0.4,            # HSV-Value (ngói chiếu sáng không đều)
        
        "degrees": 15,           # Xoay ±15 độ
        "translate": 0.2,        # Dịch chuyển
        "scale": 0.3,            # Scale thay đổi
        "flipud": 0.5,           # Flip dọc
        "fliplr": 0.5,           # Flip ngang
        
        "mosaic": 1.0,           # Mosaic augmentation (ghép 4 ảnh)
        "mixup": 0.1,            # Trộn 2 ảnh
        
        "auto_augment": "randaugment",  # AutoAugment - random augmentation
        "erasing": 0.4,          # Xóa ngẫu nhiên (che biển số)
        
        "dropout": 0.3,          # Dropout để tránh overfitting
        
        "optimizer": "SGD",
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        
        "save": True,
        "save_period": 10,
        "project": "runs/detect/plate_detection",
        "name": "improved_training",
    }
    
    print("🚀 Bắt đầu training...")
    print("⚙️  Augmentation config:")
    print("   - Mosaic + Mixup (ghép ảnh)")
    print("   - RandomAugment (xoay, dịch, giãn)")
    print("   - Erasing (che biển số)")
    print("   - HSV mạnh (biển số mờ, sáng tối)")
    print("   - Dropout (tránh overfitting)")
    
    results = model.train(**train_config)
    
    # Lưu model đã train
    best_model_path = config.PLATE_MODEL_PATH
    os.makedirs(best_model_path.parent, exist_ok=True)
    model.save(str(best_model_path))
    
    print(f"\n✅ Training hoàn tất!")
    print(f"📊 Model đã lưu: {best_model_path}")
    
    return True

def validate_model():
    """Validate model trên test set"""
    model = YOLO(str(config.PLATE_MODEL_PATH))
    
    data_yaml = os.path.join(
        os.path.dirname(__file__),
        "..",
        "License-Plate-Recognition-3",
        "data.yaml"
    )
    
    print("\n📊 Validating model...")
    results = model.val(data=data_yaml)
    
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--validate":
        validate_model()
    else:
        success = train_plate_detection()
        if success:
            print("\n🔄 Bạn có muốn validate model? Chạy: python train_plate_model_improved.py --validate")

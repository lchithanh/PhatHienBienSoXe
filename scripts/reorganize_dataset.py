"""
Script tái cấu trúc dataset theo tỷ lệ 70% train - 30% test
Gộp train + valid + test → chia lại theo 70/30 split
"""
import os
import shutil
from pathlib import Path
import random

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "License-Plate-Recognition-3"

def reorganize_dataset():
    """Tái cấu trúc dataset 70/30 split"""
    
    print("📂 Bắt đầu tái cấu trúc dataset...")
    print(f"📁 Dataset location: {DATASET_DIR}")
    
    # Khởi tạo thư mục
    train_img_dir = DATASET_DIR / "train" / "images"
    train_lbl_dir = DATASET_DIR / "train" / "labels"
    valid_img_dir = DATASET_DIR / "valid" / "images"
    valid_lbl_dir = DATASET_DIR / "valid" / "labels"
    test_img_dir = DATASET_DIR / "test" / "images"
    test_lbl_dir = DATASET_DIR / "test" / "labels"
    
    # Gộp tất cả file
    all_images = []
    all_labels = []
    
    for img_dir, lbl_dir in [
        (train_img_dir, train_lbl_dir),
        (valid_img_dir, valid_lbl_dir),
        (test_img_dir, test_lbl_dir)
    ]:
        if img_dir.exists():
            for img_file in sorted(img_dir.glob("*")):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    lbl_file = lbl_dir / f"{img_file.stem}.txt"
                    all_images.append(img_file)
                    all_labels.append(lbl_file)
    
    print(f"\n📊 Tổng ảnh tìm được: {len(all_images)}")
    print(f"   Train: {len(list(train_img_dir.glob('*')))} ảnh")
    print(f"   Valid: {len(list(valid_img_dir.glob('*')))} ảnh")
    print(f"   Test:  {len(list(test_img_dir.glob('*')))} ảnh")
    
    # Shuffle
    combined = list(zip(all_images, all_labels))
    random.shuffle(combined)
    all_images, all_labels = zip(*combined)
    
    # Tính split indices
    total = len(all_images)
    train_count = int(total * 0.7)
    
    train_images = all_images[:train_count]
    train_labels = all_labels[:train_count]
    
    test_images = all_images[train_count:]
    test_labels = all_labels[train_count:]
    
    print(f"\n✂️  Chia dữ liệu:")
    print(f"   Train: {len(train_images)} ảnh (70%)")
    print(f"   Test:  {len(test_images)} ảnh (30%)")
    
    # Backup cũ
    backup_dir = BASE_DIR / "License-Plate-Recognition-3-backup"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    
    print(f"\n📦 Backup dataset cũ → {backup_dir}")
    shutil.copytree(DATASET_DIR, backup_dir)
    
    # Xóa thư mục cũ (giữ structure)
    for dir_path in [train_img_dir, train_lbl_dir, test_img_dir, test_lbl_dir]:
        if dir_path.exists():
            shutil.rmtree(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Copy train files
    print(f"\n📂 Copy train files...")
    for img_file, lbl_file in zip(train_images, train_labels):
        if img_file.exists() and lbl_file.exists():
            shutil.copy2(img_file, train_img_dir / img_file.name)
            shutil.copy2(lbl_file, train_lbl_dir / lbl_file.name)
    
    # Copy test files
    print(f"📂 Copy test files...")
    for img_file, lbl_file in zip(test_images, test_labels):
        if img_file.exists() and lbl_file.exists():
            shutil.copy2(img_file, test_img_dir / img_file.name)
            shutil.copy2(lbl_file, test_lbl_dir / lbl_file.name)
    
    # Update data.yaml
    print(f"\n✏️  Update data.yaml...")
    data_yaml_content = """names:
- License_Plate
nc: 1
test: ../test/images
train: ../train/images
val: ../test/images
"""
    
    with open(DATASET_DIR / "data.yaml", "w") as f:
        f.write(data_yaml_content)
    
    print(f"✅ Tái cấu trúc hoàn tất!")
    print(f"\n📊 Cấu trúc mới:")
    print(f"   {DATASET_DIR}/")
    print(f"   ├── train/images/  ({len(train_images)} ảnh) 70%")
    print(f"   ├── train/labels/")
    print(f"   ├── test/images/   ({len(test_images)} ảnh) 30%")
    print(f"   ├── test/labels/")
    print(f"   └── data.yaml")
    print(f"\n💾 Backup lưu tại: {backup_dir}")
    print(f"\n✅ Sẵn sàng để train!")

if __name__ == "__main__":
    reorganize_dataset()

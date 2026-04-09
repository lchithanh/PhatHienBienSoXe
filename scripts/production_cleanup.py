"""
Production Cleanup - Xóa tất cả file không cần thiết cho deployment
Giữ lại CHỈ các file cần thiết để chạy
"""
import os
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# === FILES CẦN GIỮ LẠI ===
KEEP_FILES = {
    "config.py",
    "requirements.txt",
    "requirements_minimal.txt",
    ".env.example",
}

# === FOLDERS CẦN GIỮ LẠI ===
KEEP_FOLDERS = {
    "apps",
    "scr",
    "scripts",
    "data",
    "logs",
    "License-Plate-Recognition-3",
    "venv",  # Hoặc .venv
}

# === GUIDES CÓ THỂ XÓA ===
DOCS_TO_REMOVE = [
    "CLEANUP_GUIDE.md",
    "FINAL_TRAINING_GUIDE.md",
    "FINAL_TRAINING_QUICKSTART.md",
    "FINAL_TRAINING_SOLUTION.md",
    "IMPLEMENTATION_SUMMARY.md",
    "README_CHANGES.md",
    "TRAINING_GUIDE.md",
    "README.md",  # Original readme
    "START_HERE.txt",
    "SUMMARY.txt",
    "screenshot_20260403_000824.jpg",
]

def production_cleanup():
    """Production-only cleanup"""
    print("🎯 PRODUCTION CLEANUP - Keep only essentials")
    print("="*60)
    
    # Stats
    removed_size = 0
    removed_count = 0
    
    # List current files
    print("\n📂 Current project structure:")
    all_files = list(BASE_DIR.glob("*"))
    
    for item in sorted(all_files):
        if item.name.startswith("."):
            continue
        if item.is_dir():
            size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
            print(f"  {'📁' if item.is_dir() else '📄'} {item.name}/ ({size/1024/1024:.1f}MB)")
        else:
            size = item.stat().st_size
            print(f"  {'📄'} {item.name} ({size/1024:.1f}KB)")
    
    # === REMOVE DOCS ===
    print(f"\n🗑️  Removing documentation files...")
    for doc_file in DOCS_TO_REMOVE:
        file_path = BASE_DIR / doc_file
        if file_path.exists():
            if file_path.is_file():
                size = file_path.stat().st_size
                file_path.unlink()
                removed_size += size
                removed_count += 1
                print(f"  ✓ {doc_file}")
            elif file_path.is_dir():
                size = sum(f.stat().st_size for f in file_path.rglob("*") if f.is_file())
                shutil.rmtree(file_path)
                removed_size += size
                removed_count += 1
                print(f"  ✓ {doc_file}/")
    
    # === REMOVE __PYCACHE__ EVERYWHERE ===
    print(f"\n🗑️  Removing __pycache__ directories...")
    for pycache_dir in BASE_DIR.rglob("__pycache__"):
        try:
            shutil.rmtree(pycache_dir)
            removed_count += 1
        except:
            pass
    
    print(f"\n✅ Production cleanup completed!")
    print(f"   Removed: {removed_count} files/folders")
    print(f"   Size: {removed_size / 1024 / 1024:.2f} MB")
    
    # === FINAL STRUCTURE ===
    print(f"\n📁 PRODUCTION PROJECT STRUCTURE:")
    print(f"""
DoAnAi-Production/
├── apps/                    (Detection apps)
│   ├── detect_webcam.py
│   ├── detect_video.py
│   ├── detect_image.py
│   └── detect_gui.py
│
├── scr/                     (Source code)
│   ├── detection/
│   ├── ocr/
│   ├── logging/
│   └── utils/
│
├── scripts/                 (Training & tools)
│   ├── train_plate_model_final.py
│   ├── reorganize_dataset.py
│   ├── evaluate_model.py
│   └── download_dataset.py
│
├── data/weights/            (Model weights)
│   ├── vehicle.pt
│   └── plate_best.pt
│
├── License-Plate-Recognition-3/  (Dataset)
│   ├── data.yaml
│   ├── train/
│   ├── valid/
│   └── test/
│
├── logs/                    (Runtime logs)
│   ├── frames/
│   ├── violations/
│   └── ocr_results/
│
├── venv/                    (Virtual environment)
│
├── config.py               (Main config)
├── requirements.txt        (Dependencies)
└── requirements_minimal.txt (Minimal deps)

✅ READY FOR PRODUCTION! 🚀
""")
    
    # === FOLDER SIZES ===
    print(f"\n📊 Final folder sizes:")
    for folder in sorted(KEEP_FOLDERS):
        folder_path = BASE_DIR / folder
        if folder_path.exists():
            size = sum(f.stat().st_size for f in folder_path.rglob("*") if f.is_file())
            print(f"  {folder}: {size/1024/1024:.1f} MB")
    
    # === DEPLOYMENT INSTRUCTIONS ===
    print(f"\n🚀 DEPLOYMENT INSTRUCTIONS:")
    print(f"""
1. Create production folder:
   mkdir DoAnAi-Production

2. Copy to production:
   Copy apps/, scr/, scripts/, data/, logs/, venv/
   Copy config.py, requirements.txt, .env.example, venv/

3. Setup production environment:
   cd DoAnAi-Production
   python -m venv venv_prod
   venv_prod\\Scripts\\activate
   pip install -r requirements.txt

4. Run application:
   python apps/detect_webcam.py

5. (Optional) Mount dataset:
   Copy License-Plate-Recognition-3/ from backup
   Or download fresh from Roboflow
""")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--confirm":
        production_cleanup()
    else:
        print("Preview mode - To actually delete, run:")
        print("  python production_cleanup.py --confirm")

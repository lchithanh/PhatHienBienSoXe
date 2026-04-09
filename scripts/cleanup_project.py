"""
Script rút gọn project - xóa file không cần thiết
Giữ lại cấu trúc sạch cho deployment
"""
import os
import shutil
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Danh sách thư mục/file cần xóa
DIRS_TO_REMOVE = [
    "log",                          # Duplicate của logs/
    ".venv",                        # Nếu đã có venv/
    "__pycache__",                  # Python cache
    ".pytest_cache",
    ".env",                         # Nếu có (dùng .env.example thay thế)
    "runs/detect",                  # Weights sau khi train xong
    "images_test",                  # Test images (nếu không cần)
    "screenshot_*.jpg",             # Temp screenshots
]

FILES_TO_REMOVE = [
    "test_webcam.py",               # Test file không cần deploy
    "yolov8n.pt",                   # Weight file lớn (download khi cần)
    ".gitignore",                   # Nếu có
]

# Danh sách thư mục cần giữ lại
DIRS_TO_KEEP = [
    "apps",                         # Ứng dụng chính
    "scr",                          # Source code
    "scripts",                      # Training scripts
    "data",                         # Weights
    "logs",                         # Logging
    "License-Plate-Recognition-3",  # Dataset
]

def cleanup_project():
    """Rút gọn project"""
    print("🧹 Bắt đầu rút gọn project...\n")
    
    removed_size = 0
    
    # Xóa thư mục
    for dir_name in DIRS_TO_REMOVE:
        pattern = dir_name.split("/")[0]  # Lấy phần đầu (VD: "runs" từ "runs/detect")
        dir_path = BASE_DIR / pattern
        
        if dir_path.exists():
            try:
                if "*" in dir_name:  # Wildcard
                    for match in BASE_DIR.glob(dir_name):
                        size = sum(f.stat().st_size for f in match.rglob("*") if f.is_file())
                        shutil.rmtree(match)
                        removed_size += size
                        print(f"  ✓ Xóa: {match.relative_to(BASE_DIR)} ({size / 1024 / 1024:.2f} MB)")
                else:
                    size = sum(f.stat().st_size for f in dir_path.rglob("*") if f.is_file())
                    shutil.rmtree(dir_path)
                    removed_size += size
                    print(f"  ✓ Xóa: {dir_path.relative_to(BASE_DIR)} ({size / 1024 / 1024:.2f} MB)")
            except Exception as e:
                print(f"  ⚠ Lỗi xóa {dir_path}: {e}")
    
    # Xóa file
    for file_pattern in FILES_TO_REMOVE:
        if "*" in file_pattern:
            for file_path in BASE_DIR.glob(file_pattern):
                try:
                    size = file_path.stat().st_size
                    file_path.unlink()
                    removed_size += size
                    print(f"  ✓ Xóa: {file_path.relative_to(BASE_DIR)}")
                except Exception as e:
                    print(f"  ⚠ Lỗi xóa {file_path}: {e}")
        else:
            file_path = BASE_DIR / file_pattern
            if file_path.exists() and file_path.is_file():
                try:
                    size = file_path.stat().st_size
                    file_path.unlink()
                    removed_size += size
                    print(f"  ✓ Xóa: {file_path.relative_to(BASE_DIR)}")
                except Exception as e:
                    print(f"  ⚠ Lỗi xóa {file_path}: {e}")
    
    # Xóa __pycache__ trong tất cả thư mục
    for pycache_dir in BASE_DIR.rglob("__pycache__"):
        try:
            size = sum(f.stat().st_size for f in pycache_dir.rglob("*") if f.is_file())
            shutil.rmtree(pycache_dir)
            removed_size += size
        except Exception as e:
            print(f"  ⚠ Lỗi xóa {pycache_dir}: {e}")
    
    print(f"\n✅ Hoàn tất! Đã xóa {removed_size / 1024 / 1024:.2f} MB")
    
    # Hiển thị cấu trúc mới
    print("\n📁 Cấu trúc project sau khi rút gọn:")
    print_tree(BASE_DIR, max_depth=2)

def print_tree(path, prefix="", max_depth=3, current_depth=0):
    """In cấu trúc thư mục dạng tree"""
    if current_depth >= max_depth:
        return
    
    try:
        entries = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
    except PermissionError:
        return
    
    for i, entry in enumerate(entries):
        if entry.name.startswith("."):
            continue
            
        is_last = i == len(entries) - 1
        current_prefix = "└── " if is_last else "├── "
        print(f"{prefix}{current_prefix}{entry.name}/")
        
        if entry.is_dir() and current_depth < max_depth - 1:
            next_prefix = prefix + ("    " if is_last else "│   ")
            print_tree(entry, next_prefix, max_depth, current_depth + 1)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--confirm":
        cleanup_project()
    else:
        print("🔍 Preview những file sẽ bị xóa:")
        print(f"\nThư mục và file patterns:")
        for item in DIRS_TO_REMOVE + FILES_TO_REMOVE:
            print(f"  - {item}")
        print(f"\n⚠️  Chạy lại với --confirm để xóa thực sự:")
        print(f"   python cleanup_project.py --confirm")
